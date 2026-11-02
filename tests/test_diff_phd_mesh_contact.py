# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The reference ``mesh_contact`` state force against a literal NumPy transcription."""

import unittest

import newton
import numpy as np
import warp as wp

from newton_solver_diff_phd import SolverDiffPhD
from tests.reference_oracle import mesh_contact_force
from tests.test_diff_phd_solver import DENSITY, DEVICE, _oracle

TIGHT = {"max_iterations": 3000, "abs_tol": 1.0e-12, "rel_tol": 1.0e-11}
_FACES = np.array(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)))


def _boundary(tets):
    faces = tets[:, _FACES].reshape(-1, 3)
    _, first, counts = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    return faces[first[counts == 1]]


def _two_bodies(gap, requires_grad=False):
    """A soft 3x2x2 grid and a single tetrahedron whose apex reaches ``gap`` [m] above the grid top."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=3,
        dim_y=2,
        dim_z=2,
        cell_x=0.1,
        cell_y=0.1,
        cell_z=0.1,
        density=DENSITY,
        k_mu=3.0e4,
        k_lambda=8.0e4,
        k_damp=0.0,
        fix_bottom=True,
    )
    grid = builder.particle_count
    for point in ((0.13, 0.11, 0.2 + gap), (0.33, 0.07, 0.5), (0.05, 0.18, 0.5), (0.17, 0.02, 0.55)):
        builder.add_particle(wp.vec3(*point), wp.vec3(), mass=1.0, radius=0.0)
    builder.add_tetrahedron(grid, grid + 1, grid + 2, grid + 3, k_mu=3.0e4, k_lambda=8.0e4, k_damp=0.0)
    model = builder.finalize(device=DEVICE, requires_grad=requires_grad)
    tets = model.tet_indices.numpy().reshape(-1, 4)
    faces = _boundary(tets)
    faces_a = faces[np.all(faces < grid, axis=1)]
    faces_b = faces[np.all(faces >= grid, axis=1)]
    return model, faces_a, faces_b


class TestDiffPhDMeshContact(unittest.TestCase):
    def test_force_matches_reference_transcription(self):
        rng = np.random.default_rng(7)
        model, faces_a, faces_b = _two_bodies(-0.004)
        options = (0.02, 5.0e3, 5.0e4, 50.0, 0.5, faces_a, faces_b)
        solver = SolverDiffPhD(model, density=DENSITY, mesh_contact=options)
        q = model.particle_q.numpy().astype(np.float64)
        for trial in range(4):
            # Perturb positions so vertex, edge and face regions of the closest-point test all occur.
            positions = q + 0.012 * rng.standard_normal(q.shape) * (trial > 0)
            velocities = rng.standard_normal(q.shape)
            with self.subTest(trial=trial):
                solver._mesh_contact.evaluate(
                    wp.array(positions, dtype=wp.vec3d, device=DEVICE),
                    wp.array(velocities, dtype=wp.vec3d, device=DEVICE),
                )
                expected = mesh_contact_force(positions, velocities, faces_a, faces_b, *options[:5])
                self.assertGreater(np.abs(expected).max(), 0.0)
                np.testing.assert_allclose(
                    solver._mesh_contact.force.numpy(), expected, rtol=0.0, atol=1.0e-12 * np.abs(expected).max()
                )
                self.assertGreater(int(solver.mesh_contact_pairs.numpy()[0]), 0)

    def test_step_matches_reference_with_state_force_in_the_target(self):
        model, faces_a, faces_b = _two_bodies(-0.003)
        options = (0.01, 5.0e3, 5.0e4, 50.0, 0.5, faces_a, faces_b)
        solver = SolverDiffPhD(model, density=DENSITY, mesh_contact=options, **TIGHT)
        rng = np.random.default_rng(3)
        state_in, state_out = model.state(), model.state()
        free = model.particle_inv_mass.numpy() > 0
        velocity = np.where(free[:, None], 0.2 * rng.standard_normal((model.particle_count, 3)), 0.0)
        state_in.particle_qd.assign(velocity.astype(np.float32))
        dt = 0.004
        solver.step(state_in, state_out, None, None, dt)
        solver.check_status()
        q = state_in.particle_q.numpy().astype(np.float64)
        v = state_in.particle_qd.numpy().astype(np.float64)
        force = mesh_contact_force(q, v, faces_a, faces_b, *options[:5])
        self.assertGreater(np.abs(force).max(), 0.0)
        expected, _iterations, converged = _oracle(model, "pd").step(
            q, v, force, np.zeros(3), dt, max_iterations=3000, abs_tol=1.0e-12, rel_tol=1.0e-11
        )
        self.assertTrue(converged)
        np.testing.assert_allclose(solver._x.numpy(), expected, rtol=0.0, atol=1.0e-10 * np.abs(expected).max())

    def test_gradients_hold_the_state_force_fixed(self):
        # BackwardForce returns zero: the step's gradients equal those of the same step
        # with the evaluated contact force supplied as a constant external force.
        model, faces_a, faces_b = _two_bodies(-0.003, requires_grad=True)
        options = (0.01, 5.0e3, 5.0e4, 50.0, 0.5, faces_a, faces_b)
        weights = wp.array(
            np.random.default_rng(5).standard_normal((model.particle_count, 3)), dtype=wp.vec3d, device=DEVICE
        )
        results = []
        for frozen in (False, True):
            solver = SolverDiffPhD(model, density=DENSITY, mesh_contact=None if frozen else options, **TIGHT)
            state_in, state_out = model.state(requires_grad=True), model.state(requires_grad=True)
            if frozen:
                q = state_in.particle_q.numpy().astype(np.float64)
                force = mesh_contact_force(q, np.zeros_like(q), faces_a, faces_b, *options[:5])
                state_in.particle_f.assign(force.astype(np.float32))
            for name in ("particle_q", "particle_qd", "particle_f"):
                getattr(state_in, name).requires_grad = True
            loss = wp.zeros(1, dtype=wp.float64, device=DEVICE, requires_grad=True)
            tape = wp.Tape()
            with tape:
                solver.step(state_in, state_out, None, None, 0.004)
                wp.launch(
                    _dot, model.particle_count, inputs=[state_out.particle_q, weights], outputs=[loss], device=DEVICE
                )
            tape.backward(loss)
            solver.check_status()
            results.append([getattr(state_in, name).grad.numpy() for name in ("particle_q", "particle_qd")])
            results[-1].append(solver._x.numpy())
        for actual, expected in zip(results[0], results[1], strict=True):
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-6 * max(np.abs(expected).max(), 1.0))

    def test_invalid_mesh_contact_is_rejected(self):
        model, faces_a, faces_b = _two_bodies(0.0)
        for options in (
            (0.01, 5.0e3, 5.0e4, 50.0, 0.5, faces_a),
            (-0.01, 5.0e3, 5.0e4, 50.0, 0.5, faces_a, faces_b),
            (0.01, 5.0e3, 5.0e4, 50.0, 0.5, faces_a, np.zeros((0, 3), dtype=np.int32)),
            (0.01, 5.0e3, 5.0e4, 50.0, 0.5, faces_a, faces_b + model.particle_count),
        ):
            with self.subTest(options=len(options)), self.assertRaises(ValueError):
                SolverDiffPhD(model, density=DENSITY, mesh_contact=options)


@wp.kernel
def _dot(q: wp.array[wp.vec3], weights: wp.array[wp.vec3d], loss: wp.array[wp.float64]):
    i = wp.tid()
    wp.atomic_add(loss, 0, wp.dot(wp.vec3d(q[i]), weights[i]))


if __name__ == "__main__":
    unittest.main()
