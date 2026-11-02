# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Portions translated from DiffPD (https://github.com/mit-gfx/diff_pd_public,
# commit 1594aa0), Copyright (c) 2021 MIT Graphics Group, MIT License.
# See THIRD_PARTY_NOTICES.md.

"""Dense NumPy oracle of the DiffPD ``pd_eigen`` forward step for small test meshes.

This is an independent transcription of the executed C++ reference
(``ForwardProjectiveDynamics`` and ``PdNonlinearSolve`` with ``use_bfgs = 1``
in ``cpp/core/src/fem/deformable_projective_dynamics_forward.cpp``, the
corotated and volume PD element energies, and the gravity, planar and arc
contact state forces of https://github.com/mit-gfx/diff_pd_public at commit
1594aa0). It uses NumPy's SVD, a dense matrix and ``numpy.linalg.solve``
instead of the solver's Jacobi SVD, sparse inverse factors and device
reductions. It is a test oracle only.
"""

from collections import deque

import numpy as np

EPS = np.finfo(np.float64).eps


def corotated_projection(u, vt):
    return u @ vt


def volume_projection(u, sigma, vt):
    # VolumePdElementEnergy::ProjectToManifold.
    if sigma.min() < EPS:
        raise ValueError("Singular F.")
    d = sigma / np.prod(sigma) ** (1.0 / 3.0) - sigma
    for _ in range(50):
        s = sigma + d
        constraint = np.prod(s) - 1.0
        gradient = np.array([s[1] * s[2], s[0] * s[2], s[0] * s[1]])
        following = (gradient @ d - constraint) / (gradient @ gradient) * gradient
        if np.max(np.abs(following - d)) <= EPS:
            break
        d = following
    return u @ np.diag(sigma + d) @ vt


def planar_contact(q, v, parameters):
    """``PlanarContactStateForce::ForwardForce`` with the raw 8 parameters."""
    normal = np.asarray(parameters[:3], dtype=np.float64)
    norm = np.linalg.norm(normal)
    normal, offset = normal / norm, parameters[3] / norm
    p, kn, kf, mu = int(parameters[4]), *parameters[5:8]
    f = np.zeros_like(q)
    for i in range(len(q)):
        d = normal @ q[i] + offset
        fn = kn * (-min(0.0, d)) ** (p - 1)
        us = v[i] - (v[i] @ normal) * normal
        f[i] = fn * normal - min(kf, mu * fn / (np.linalg.norm(us) + EPS)) * us
    return f


def arc_contact(q, v, parameters):
    """``ArcContactStateForce::ForwardForce`` with the raw 15 parameters."""
    center = np.asarray(parameters[:3], dtype=np.float64)
    axis = np.asarray(parameters[3:6], dtype=np.float64)
    start = np.asarray(parameters[6:9], dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    start = start / np.linalg.norm(start)
    aux = np.cross(axis, start)
    radius, angle, p, kn, kf, mu = parameters[9], parameters[10], int(parameters[11]), *parameters[12:15]
    f = np.zeros_like(q)
    for i in range(len(q)):
        local = q[i] - center
        projected = local - (local @ axis) * axis
        theta = np.arctan2(projected @ aux, projected @ start)
        if 0.0 < theta < angle:
            d = radius - np.linalg.norm(projected)
            fn = kn * (-min(0.0, d)) ** (p - 1)
            normal = -projected / (np.linalg.norm(projected) + EPS)
            us = v[i] - (v[i] @ normal) * normal
            f[i] = fn * normal - min(kf, mu * fn / (np.linalg.norm(us) + EPS)) * us
    return f


class ReferenceStep:
    """One ``ForwardProjectiveDynamics`` call per :meth:`step` without frictional boundary.

    Args:
        rest: Rest positions ``(n, 3)``.
        tets: Elements ``(e, 4)``.
        density: Density.
        corotated: Corotated stiffness or None.
        volume: Volume stiffness or None.
        free: Boolean mask; False entries are Dirichlet vertices held at ``q``.
        gravity: Gravity vector of the first state force.
        state_forces: ``(kind, parameters)`` entries after gravity.
    """

    def __init__(
        self,
        rest,
        tets,
        density,
        corotated,
        volume,
        free,
        gravity,
        state_forces=(),
        max_pd_iter=4000,
        max_ls_iter=10,
        abs_tol=1.0e-9,
        rel_tol=1.0e-4,
        bfgs_history_size=10,
    ):
        self.rest = np.asarray(rest, dtype=np.float64)
        self.tets = np.asarray(tets, dtype=np.int64)
        self.free = np.asarray(free, dtype=bool)
        self.gravity = np.asarray(gravity, dtype=np.float64)
        self.state_forces = list(state_forces)
        self.energies = [(name, w) for name, w in (("corotated", corotated), ("volume", volume)) if w is not None]
        self.max_pd_iter = max_pd_iter
        self.max_ls_iter = max_ls_iter
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.history = bfgs_history_size
        # TetDeformable::InitializeFiniteElementSamples.
        self.gradients = []
        volumes = []
        for tet in self.tets:
            x0, x1, x2, x3 = self.rest[tet]
            b = np.linalg.inv(np.column_stack((x1 - x0, x2 - x0, x3 - x0)))
            self.gradients.append(np.vstack((-b[0] - b[1] - b[2], b[0], b[1], b[2])))
            a, b, c = x1 - x0, x2 - x0, x3 - x0
            cross = (b[1] * c[2] - b[2] * c[1], b[2] * c[0] - b[0] * c[2], b[0] * c[1] - b[1] * c[0])
            volumes.append(abs((a[0] * cross[0] + a[1] * cross[1] + a[2] * cross[2]) / 6.0))
        # Mesh::Initialize sums the element volumes sequentially.
        total = 0.0
        for volume in volumes:
            total += float(volume)
        self.volume = total / len(volumes)
        self.mass = density * self.volume
        self.dt = None
        # Line-search halvings and gradient fallbacks of the last step, for test coverage.
        self.halvings = 0
        self.fallbacks = 0

    def _setup(self, dt):
        # SetupProjectiveDynamicsSolver for one coordinate; all three are identical.
        n = len(self.rest)
        inertia = self.mass / (dt * dt)
        a = np.diag(np.where(self.free, inertia, 1.0))
        w = sum(stiffness for _, stiffness in self.energies)
        for tet, g in zip(self.tets, self.gradients, strict=True):
            block = g @ g.T * w * self.volume
            for i in range(4):
                for j in range(4):
                    if self.free[tet[i]] and self.free[tet[j]]:
                        a[tet[i], tet[j]] += block[i, j]
        self.matrix = a
        self.dt = dt
        assert a.shape == (n, n)

    def state_force(self, q, v):
        """``ForwardStateForce``: gravity first, then the contacts in order."""
        f = np.zeros_like(q) + self.mass * self.gravity
        for kind, parameters in self.state_forces:
            f = f + (planar_contact if kind == "planar_contact" else arc_contact)(q, v, parameters)
        return f

    def energy_force(self, x):
        """``ComputePdEnergy`` and ``PdEnergyForce`` at ``x``."""
        energy = 0.0
        force = np.zeros_like(x)
        for tet, g in zip(self.tets, self.gradients, strict=True):
            f = x[tet].T @ g
            u, sigma, vt = np.linalg.svd(f)
            for name, w in self.energies:
                p = corotated_projection(u, vt) if name == "corotated" else volume_projection(u, sigma, vt)
                difference = f - p
                energy += w * 0.5 * np.sum(difference * difference) * self.volume
                force[tet] += -(w * difference @ g.T).T * self.volume
        return energy, force

    def step(self, q, v, f_ext, dt):
        """Return ``q_next, v_next, iterations`` of one reference forward step."""
        q = np.asarray(q, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        if self.dt != dt:
            self._setup(dt)
        h2m = dt * dt / self.mass
        inertia = self.mass / (dt * dt)
        rhs = q + dt * v + h2m * np.asarray(f_ext, dtype=np.float64) + h2m * self.state_force(q, v)
        selected = self.free[:, None].astype(np.float64)

        def solve(r):
            return np.linalg.solve(self.matrix, r)

        def objective(x, energy):
            return 0.5 * np.sum((x - rhs) * (inertia * (x - rhs))) + energy

        x = q.copy()
        energy, force = self.energy_force(x)
        obj = objective(x, energy)
        grad = (inertia * (x - rhs) - force) * selected
        threshold = self.rel_tol * np.linalg.norm(selected * (inertia * rhs)) + self.abs_tol
        xs, gs, ss, ys = deque(), deque(), deque(), deque()
        self.halvings = self.fallbacks = 0
        for iteration in range(self.max_pd_iter):
            if not xs:
                xs.append(x)
                gs.append(grad)
                direction = solve(grad)
            else:
                count = len(xs)
                ss.append(x - xs[-1])
                ys.append(grad - gs[-1])
                xs.append(x)
                gs.append(grad)
                if count == self.history + 1:
                    for history in (xs, gs, ss, ys):
                        history.popleft()
                r = grad.copy()
                rhos, alphas = deque(), deque()
                for s, y in zip(reversed(ss), reversed(ys), strict=True):
                    rho = 1.0 / np.sum(y * s)
                    alpha = rho * np.sum(s * r)
                    rhos.appendleft(rho)
                    alphas.appendleft(alpha)
                    r = r - alpha * y
                z = solve(r)
                for s, y, rho, alpha in zip(ss, ys, rhos, alphas, strict=True):
                    beta = rho * np.sum(y * z)
                    z = z + s * (alpha - beta)
                direction = z
            if np.sum(direction * grad) <= 0:
                direction = grad * selected
                self.fallbacks += 1
            step = 1.0
            trial = x - step * direction
            trial_energy, trial_force = self.energy_force(trial)
            trial_obj = objective(trial, trial_energy)
            for _ in range(self.max_ls_iter):
                bound = obj - 1.0e-4 * step * np.sum(grad * direction)
                if not np.isnan(trial_obj) and trial_obj < bound + EPS:
                    break
                step /= 2
                self.halvings += 1
                trial = x - step * direction
                trial_energy, trial_force = self.energy_force(trial)
                trial_obj = objective(trial, trial_energy)
            x, obj = trial, trial_obj
            grad = (inertia * (x - rhs) - trial_force) * selected
            if np.linalg.norm(grad) <= threshold:
                return x, (x - q) / dt, iteration + 1
        raise RuntimeError("PD method fails to converge.")
