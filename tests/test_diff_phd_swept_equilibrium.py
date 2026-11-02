# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent constrained-energy oracle for a homogeneous flat plate press.

This checks the reference PD energy and NH Moreau envelope, not raw NH stress.
The broad rigid plate makes contact equivalent to four scalar upper bounds on
the free surface's height. NumPy SVD, SciPy constrained volume minimization and
log-stretch energy minimization provide an oracle independent of production
projections, inverse factors, acceleration, CCD and contact-QP code.

The outer optimization is local within an orientation-preserving neighborhood;
its artificial lateral/lower bounds must stay inactive. The hold resets incoming
velocity to zero and solves another implicit timestep; it is not a claim of
infinite-time quasistatic convergence.
"""

import time
import unittest

import newton
import numpy as np
import warp as wp
from scipy.optimize import least_squares, minimize

from newton_solver_diff_phd import SolverDiffPhD
from tests.test_diff_phd_solver import DEVICE
from tests.test_diff_phd_swept_contact import _box, _surface


def scene():
    block, block_tets = _box((0, 0, 0), (0.1, 0.1, 0.05))
    plate, plate_tets = _box((-0.05, -0.05, 0.0535), (0.15, 0.15, 0.0635))
    rest = np.vstack((block, plate)).astype(np.float32)
    tets = np.vstack((block_tets, plate_tets + 8))
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    for position in rest:
        builder.add_particle(wp.vec3(*position), wp.vec3(), mass=0.1)
    young, poisson = 20000.0, 0.3
    for tet in tets:
        builder.add_tetrahedron(
            *map(int, tet),
            k_mu=young / (2 * (1 + poisson)),
            k_lambda=young * poisson / ((1 + poisson) * (1 - 2 * poisson)),
            k_damp=0.0,
        )
    free = np.flatnonzero(block[:, 2] > 0)
    for index in np.setdiff1d(np.arange(len(rest)), free):
        builder.particle_mass[index] = 0.0
    return (
        builder.finalize(device=DEVICE),
        rest,
        tets,
        free,
        _surface(block, block_tets),
        _surface(plate, plate_tets) + 8,
    )


def project_volume(sigma):
    def objective(stretches):
        delta = stretches - sigma
        return 0.5 * (delta @ delta), delta

    result = minimize(
        objective,
        sigma / np.prod(sigma) ** (1 / 3),
        jac=True,
        method="SLSQP",
        bounds=[(1e-8, None)] * 3,
        constraints={"type": "eq", "fun": lambda s: np.prod(s) - 1, "jac": lambda s: np.prod(s) / s},
        options={"ftol": 1e-14, "maxiter": 200},
    )
    if not result.success or abs(np.prod(result.x) - 1) > 1e-10:
        raise RuntimeError(f"Independent volume minimization failed: {result.message}")
    return result.x


def nh_envelope(sigma, mu, lame):
    ratio = lame / mu

    def objective(log_s):
        s = np.exp(log_s)
        log_j = log_s.sum()
        delta = s - sigma
        value = 0.5 * (s @ s - 3) - log_j + 0.5 * ratio * log_j**2 + delta @ delta
        gradient = s * s - 1 + ratio * log_j + 2 * delta * s
        return value, gradient

    result = minimize(
        objective,
        np.log(sigma),
        jac=True,
        method="BFGS",
        options={"gtol": 2e-11, "maxiter": 200},
    )
    polished = least_squares(
        lambda y: objective(y)[1], result.x, jac="3-point", ftol=1e-13, xtol=1e-13, gtol=1e-13, max_nfev=100
    )
    value, gradient = objective(polished.x)
    if not np.isfinite(value) or np.linalg.norm(gradient, ord=np.inf) > 1e-10:
        raise RuntimeError(f"Independent NH minimization failed: {polished.message}; {gradient}")
    if value > result.fun + 1e-12:
        raise RuntimeError("Stationarity polishing increased the NH objective")
    return mu * value, np.exp(polished.x)


class Energy:
    def __init__(self, rest, tets, free, materials, material, start, velocity, force, dt, prescribed):
        self.rest, self.tets, self.free = rest.astype(np.float64), tets, free
        self.material = material
        elements = self.rest[tets]
        edges = elements[:, 1:] - elements[:, :1]
        self.volume = np.linalg.det(edges).mean() / 6
        self.mass = 800 * self.volume
        inverse = np.linalg.inv(np.transpose(edges, (0, 2, 1)))
        self.gradients = np.concatenate((-inverse.sum(axis=1)[:, None, :], inverse), axis=1)
        self.mu, self.lame = materials[0, :2].astype(np.float64)
        assert np.all(materials[:, :2] == materials[0, :2])
        self.target = start.astype(np.float64) + dt * velocity.astype(np.float64) + dt**2 * force / self.mass
        self.inertia = self.mass / dt**2
        self.prescribed = prescribed.astype(np.float64)
        self.last_x = self.last_result = None

    def __call__(self, flat):
        if self.last_x is not None and np.array_equal(flat, self.last_x):
            return self.last_result
        positions = self.prescribed.copy()
        positions[self.free] = flat.reshape(-1, 3)
        delta = positions[self.free] - self.target[self.free]
        value = 0.5 * self.inertia * np.sum(delta**2)
        gradient = np.zeros_like(positions)
        gradient[self.free] = self.inertia * delta
        # Entirely prescribed plate energy is constant and can be omitted.
        for tet, basis in zip(self.tets[:6], self.gradients[:6], strict=True):
            f = positions[tet].T @ basis
            if np.linalg.det(f) <= 0:
                raise RuntimeError("Oracle attempted an inverted element")
            u, sigma, vt = np.linalg.svd(f)
            if self.material == "pd":
                rotation = u @ vt
                volume = (u * project_volume(sigma)) @ vt
                value += self.volume * (
                    self.mu * np.sum((f - rotation) ** 2) + 0.5 * self.lame * np.sum((f - volume) ** 2)
                )
                stress = 2 * self.mu * (f - rotation) + self.lame * (f - volume)
            else:
                local_value, stretches = nh_envelope(sigma, self.mu, self.lame)
                value += self.volume * local_value
                stress = 2 * self.mu * (f - (u * stretches) @ vt)
            gradient[tet] += self.volume * basis @ stress.T
        self.last_x, self.last_result = flat.copy(), (float(value), gradient[self.free].ravel())
        return self.last_result


def equilibrium_cases():
    """Return scalar validation measurements; no private assets are loaded."""
    reports = []
    for material in ("pd", "neo_hookean"):
        model, rest, tets, free, left, right = scene()
        solver = SolverDiffPhD(
            model,
            density=800,
            material=material,
            swept_contact=(0.0005, 8, left, right),
            max_iterations=500,
            abs_tol=1e-11,
            rel_tol=1e-9,
        )
        target = rest.copy()
        target[8:, 2] -= 0.0085
        force = np.zeros_like(rest)
        force[1, 0] = 3.0
        force[7, 1] = -1.5
        start = model.state()
        start.particle_f.assign(force)
        targets = wp.array(target, dtype=wp.vec3, device=DEVICE)
        for stage in ("press", "hold"):
            before = time.perf_counter()
            oracle = Energy(
                rest,
                tets,
                free,
                model.tet_materials.numpy(),
                material,
                start.particle_q.numpy(),
                start.particle_qd.numpy(),
                force,
                0.01,
                target,
            )
            output = model.state()
            solver.step(start, output, None, None, 0.01, dirichlet_positions=targets)
            solver.check_status()
            actual = output.particle_q.numpy()[free].astype(np.float64)
            ceiling = float(target[8:, 2].min()) - 0.0005
            initial = start.particle_q.numpy()[free].astype(np.float64)
            initial[:, 2] = np.minimum(initial[:, 2], ceiling)
            bounds = [
                (0.025, ceiling) if i % 3 == 2 else (rest[free].ravel()[i] - 0.02, rest[free].ravel()[i] + 0.02)
                for i in range(initial.size)
            ]
            result = minimize(
                oracle,
                initial.ravel(),
                jac=True,
                bounds=bounds,
                method="L-BFGS-B",
                options={"ftol": 1e-15, "gtol": 1e-8, "maxiter": 500, "maxls": 30},
            )
            # Start independently from a different admissible configuration.
            # Agreement excludes accepting an optimizer's success flag alone.
            alternate = initial.copy()
            alternate[:, :2] += np.asarray(((1, -1), (-1, 1), (1, 1), (-1, -1))) * 0.002
            other = minimize(
                oracle,
                alternate.ravel(),
                jac=True,
                bounds=bounds,
                method="L-BFGS-B",
                options={"ftol": 1e-15, "gtol": 1e-8, "maxiter": 500, "maxls": 30},
            )
            # Verify the independent energy's force computation by a central
            # difference at an off-equilibrium point, where the force is nonzero.
            direction = np.asarray(((1, 2, 0), (-2, 1, 0), (1, -1, 0), (2, -1, 0)), dtype=float).ravel()
            direction /= np.linalg.norm(direction)
            epsilon = 2e-6
            check_position = result.x + 0.001 * direction
            finite_difference = (
                oracle(check_position + epsilon * direction)[0] - oracle(check_position - epsilon * direction)[0]
            ) / (2 * epsilon)
            directional_force = oracle(check_position)[1] @ direction
            oracle_gradient = oracle(result.x)[1].reshape(-1, 3)
            actual_energy, gradient = oracle(actual.ravel())
            g = gradient.reshape(-1, 3)
            residual = g.copy()
            active = actual[:, 2] >= ceiling - 1e-7
            residual[active, 2] = np.maximum(g[active, 2], 0.0)
            all_positions = output.particle_q.numpy().astype(np.float64)
            rest_elements = rest[tets].astype(np.float64)
            rest_edges = rest_elements[:, 1:] - rest_elements[:, :1]
            current = all_positions[tets]
            minimum_jacobian = np.min(
                np.linalg.det(current[:, 1:] - current[:, :1]) / np.linalg.det(rest_edges.astype(np.float64))
            )
            lateral_margin = np.min(0.02 - np.abs(result.x.reshape(-1, 3)[:, :2] - rest[free, :2]))
            lower_margin = np.min(result.x.reshape(-1, 3)[:, 2] - 0.025)
            reports.append(
                {
                    "material": material,
                    "stage": stage,
                    "production_iterations": int(solver.iterations.numpy()[0]),
                    "oracle_success": bool(result.success),
                    "oracle_message": result.message,
                    "alternate_oracle_success": bool(other.success),
                    "oracle_restart_difference_m": float(np.max(np.abs(result.x - other.x))),
                    "oracle_tangential_kkt_n": float(np.max(np.abs(oracle_gradient[:, :2]))),
                    "oracle_directional_force_error_n": float(abs(finite_difference - directional_force)),
                    "oracle_checked_directional_force_n": float(directional_force),
                    "oracle_artificial_bound_margin_m": float(min(lateral_margin, lower_margin)),
                    "max_position_error_m": float(np.max(np.abs(actual.ravel() - result.x))),
                    "energy_gap_j": float(actual_energy - result.fun),
                    "force_kkt_residual_n": float(np.max(np.abs(residual))),
                    "reaction_forces_n": (-g[:, 2]).tolist(),
                    "minimum_clearance_slack_m": float(ceiling - actual[:, 2].max()),
                    "minimum_jacobian": float(minimum_jacobian),
                    "max_deformation_m": float(np.max(np.linalg.norm(all_positions[:8] - rest[:8], axis=1))),
                    "max_prescribed_error_m": float(np.max(np.abs(all_positions[8:] - target[8:]))),
                    "max_pinned_error_m": float(
                        np.max(np.abs(all_positions[np.arange(0, 8, 2)] - rest[np.arange(0, 8, 2)]))
                    ),
                    "wall_seconds": time.perf_counter() - before,
                }
            )
            start = output
            start.particle_qd.zero_()
            start.particle_f.assign(force)
    return reports


class TestSweptEquilibriumOracle(unittest.TestCase):
    def test_loaded_press_and_hold_match_independent_energy_minimization(self):
        for result in equilibrium_cases():
            with self.subTest(material=result["material"], stage=result["stage"]):
                self.assertTrue(result["oracle_success"], result["oracle_message"])
                self.assertTrue(result["alternate_oracle_success"])
                self.assertLess(result["oracle_restart_difference_m"], 2e-8)
                self.assertLess(result["oracle_tangential_kkt_n"], 2e-6)
                self.assertLess(result["oracle_directional_force_error_n"], 2e-6)
                self.assertGreater(abs(result["oracle_checked_directional_force_n"]), 0.1)
                self.assertGreater(result["oracle_artificial_bound_margin_m"], 0.01)
                self.assertLess(result["max_position_error_m"], 2e-8)
                self.assertLess(abs(result["energy_gap_j"]), 2e-7)
                self.assertLess(result["force_kkt_residual_n"], 2e-5)
                self.assertGreater(min(result["reaction_forces_n"]), 0.5)
                self.assertGreaterEqual(result["minimum_clearance_slack_m"], -1e-8)
                self.assertGreater(result["minimum_jacobian"], 0.8)
                self.assertGreater(result["max_deformation_m"], 0.005)
                self.assertEqual(result["max_prescribed_error_m"], 0.0)
                self.assertEqual(result["max_pinned_error_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
