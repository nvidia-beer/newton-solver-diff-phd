# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Dense NumPy oracle of the reference DiffPhD step for small test meshes.

This is an independent transcription of the executed C++ reference
(``ForwardProjectiveDynamicsAlgPhd`` and the PD element energies of
https://github.com/Akinesia112/DiffPhD at commit d319435). It uses NumPy's SVD,
dense matrices and ``numpy.linalg.solve`` instead of the solver's Jacobi SVD,
sparse inverse factors, pivoted LDL^T and device reductions. It is a test
oracle only; it never produces reference-implementation results.
"""

import numpy as np

EPS = np.finfo(np.float64).eps


def unsigned_svd(f):
    u, sigma, vh = np.linalg.svd(f)
    return u, sigma, vh.T


def corotated_projection(f):
    u, _sigma, v = unsigned_svd(f)
    return u @ v.T


def volume_projection(f):
    u, sigma, v = unsigned_svd(f)
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
    return u @ np.diag(sigma + d) @ v.T


def nh_prox_stretches(target, mu, lame, penalty):
    sigma = np.maximum(target, 1.0e-4)
    for _ in range(32):
        log_j = np.log(max(np.prod(sigma), 1.0e-12))
        gradient = mu * sigma - mu / sigma + lame * log_j / sigma + penalty * (sigma - target)
        if np.linalg.norm(gradient) < 1.0e-10:
            break
        hessian = lame / np.outer(sigma, sigma)
        np.fill_diagonal(hessian, mu + (mu + lame - lame * log_j) / sigma**2 + penalty)
        delta = np.linalg.solve(hessian + 1.0e-9 * np.eye(3), gradient)
        alpha = 1.0
        trial = sigma - alpha * delta
        for _ in range(20):
            if trial.min() > 0.5e-4:
                break
            alpha *= 0.5
            trial = sigma - alpha * delta
        step = np.max(np.abs(trial - sigma))
        sigma = np.maximum(trial, 1.0e-4)
        if step < 1.0e-10:
            break
    return sigma


def nh_projection(f, mu, lame, penalty):
    u, sigma, v = unsigned_svd(f)
    return u @ np.diag(nh_prox_stretches(sigma, mu, lame, penalty)) @ v.T


def muscle_projection(f, direction, target_length):
    fibre = f @ direction
    length = np.linalg.norm(fibre)
    return np.zeros(3) if length <= EPS else target_length / length * fibre


class ReferenceStep:
    """Dense reference step for a tetrahedral mesh with Dirichlet particles.

    Args:
        tets: ``(E, 4)`` vertex indices.
        rest_poses: ``(E, 3, 3)`` inverse rest edge matrices ``Dm^{-1}``.
        mu, lame: ``(E,)`` Lame parameters [Pa].
        density: Uniform density [kg/m^3].
        free: ``(N,)`` booleans, false for Dirichlet particles.
        material: ``"pd"`` or ``"neo_hookean"``.
        muscle_directions, muscle_stiffness: Optional fibres.
    """

    def __init__(
        self, tets, rest_poses, mu, lame, density, free, material, muscle_directions=None, muscle_stiffness=None
    ):
        self.tets = np.asarray(tets)
        self.free = np.asarray(free, dtype=bool)
        self.n = len(self.free)
        rest_poses = np.asarray(rest_poses, dtype=np.float64)
        self.gradients = np.concatenate((-rest_poses.sum(axis=1)[:, None], rest_poses), axis=1)
        volumes = np.abs(1.0 / np.linalg.det(rest_poses)) / 6.0
        self.vbar = volumes.mean()
        self.mass = density * self.vbar
        self.mu, self.lame = np.asarray(mu, dtype=np.float64), np.asarray(lame, dtype=np.float64)
        self.material = material
        if material == "pd":
            self.energies = [("corotated", 2.0 * self.mu * self.vbar), ("volume", self.lame * self.vbar)]
        else:
            self.energies = [("neo_hookean", 2.0 * self.mu * self.vbar)]
        self.nh = (self.mu.mean(), self.lame.mean(), 2.0 * self.mu.mean())
        self.muscle_directions = None
        if muscle_directions is not None:
            directions = np.asarray(muscle_directions, dtype=np.float64)
            stiffness = np.broadcast_to(np.asarray(muscle_stiffness, dtype=np.float64), (len(self.tets),))
            lengths = np.linalg.norm(directions, axis=1)
            directions = np.where(
                stiffness[:, None] > 0, directions / np.where(lengths > 0, lengths, 1.0)[:, None], 0.0
            )
            self.muscle_directions = directions
            self.muscle_weight = stiffness * self.vbar

    def matrix(self, dt):
        a = np.zeros((self.n, self.n))
        for i in range(self.n):
            a[i, i] = self.mass / dt**2 if self.free[i] else 1.0
        total = sum(weight for _name, weight in self.energies)
        for e, tet in enumerate(self.tets):
            g = self.gradients[e]
            block = total[e] * (g @ g.T)
            if self.muscle_directions is not None:
                fibre = g @ self.muscle_directions[e]
                block += self.muscle_weight[e] * np.outer(fibre, fibre)
            for a_, i in enumerate(tet):
                for b_, j in enumerate(tet):
                    if self.free[i] and self.free[j]:
                        a[i, j] += block[a_, b_]
        return a

    def projection_rhs(self, x, boundary, targets=None):
        p = np.zeros((self.n, 3))
        for e, tet in enumerate(self.tets):
            g = self.gradients[e]
            f = x[tet].T @ g
            f_boundary = boundary[tet].T @ g
            stress = np.zeros((3, 3))
            for name, weight in self.energies:
                if weight[e] == 0:
                    continue
                if name == "corotated":
                    projected = corotated_projection(f)
                elif name == "volume":
                    projected = volume_projection(f)
                else:
                    projected = nh_projection(f, *self.nh)
                stress += weight[e] * (projected - f_boundary)
            if self.muscle_directions is not None and self.muscle_weight[e] > 0:
                m = self.muscle_directions[e]
                fibre = muscle_projection(f, m, targets[e])
                stress += self.muscle_weight[e] * np.outer(fibre - f_boundary @ m, m)
            p[tet] += (stress @ g.T).T
        return p

    def step(
        self,
        q,
        v,
        f_ext,
        gravity,
        dt,
        *,
        max_iterations,
        abs_tol,
        rel_tol,
        window=5,
        boundary=None,
        targets=None,
        contact=None,
    ):
        """Return the positions, iteration count and convergence flag.

        ``contact`` is a :class:`ReferenceContact`; its multipliers carry over
        between calls like the reference warm start.
        """
        q = np.asarray(q, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        if boundary is None:
            boundary = np.where(self.free[:, None], 0.0, q)
        q_tilde = q + dt * v + dt**2 * (np.asarray(f_ext, dtype=np.float64) / self.mass + gravity)
        matrix = self.matrix(dt)
        inertia = self.mass / dt**2
        if contact is not None:
            contact.begin_step(q, np.linalg.inv(matrix), dt)
        x = q.copy()
        previous_rhs = np.zeros_like(x)
        history_x, history_g = [], []
        for iteration in range(max_iterations):
            p = self.projection_rhs(x, boundary, targets)
            x_previous = x
            if contact is not None and len(contact.nodes):
                x = contact.iterate(x, inertia * q_tilde + p, q)
            else:
                rhs = np.where(self.free[:, None], inertia * q_tilde + p, 0.0)
                x = np.linalg.solve(matrix, rhs)
            x[~self.free] = boundary[~self.free]
            if window > 0:
                g = x - x_previous
                history_x.append(x_previous)
                history_g.append(g)
                m = len(history_g) - 1
                if m >= 1:
                    dg = np.stack([(history_g[i + 1] - history_g[i]).ravel() for i in range(m)], axis=1)
                    dx = np.stack([(history_x[i + 1] - history_x[i]).ravel() for i in range(m)], axis=1)
                    gram = dg.T @ dg
                    gram += max(1.0e-6 * np.sum(dg * dg) / m, 1.0e-12) * np.eye(m)
                    gamma = np.linalg.solve(gram, dg.T @ g.ravel())
                    if np.linalg.norm(gamma) >= 10.0:
                        history_x.clear()
                        history_g.clear()
                    else:
                        x = ((x_previous + g).ravel() - (dx + dg) @ gamma).reshape(-1, 3)
                        x[~self.free] = boundary[~self.free]
                del history_x[:-window]
                del history_g[:-window]
            step_small = np.linalg.norm(x - x_previous) <= rel_tol * np.linalg.norm(x_previous) + abs_tol
            rhs_small = np.linalg.norm(p - previous_rhs) <= rel_tol * np.linalg.norm(previous_rhs) + abs_tol
            previous_rhs = p
            if iteration > 0 and step_small and rhs_small:
                break
        else:
            iteration = max_iterations - 1
        converged = iteration < max_iterations - 1 or (iteration > 0 and step_small and rhs_small)
        if contact is not None:
            x = contact.restitution(x, q, v, dt)
        return x, iteration + 1, converged


def _local_frame(normal):
    best = np.zeros(3)
    for i in range(3):
        candidate = np.cross(normal, np.eye(3)[i])
        if candidate @ candidate > best @ best:
            best = candidate
    t1 = best / np.linalg.norm(best)
    t2 = np.cross(normal, t1)
    return t1, t2 / np.linalg.norm(t2)


class ReferenceContact:
    """Literal transcription of the reference planar/spherical frictional boundary.

    It keeps the reference's per-axis ``N x K`` columns ``A^{-1}(n e_c)``, the
    gathered Delassus matrix, tangential columns and the dense solve.
    """

    def __init__(self, kind, parameters, nodes, friction=0.0, restitution=0.0):
        self.kind = kind
        self.parameters = np.asarray(parameters, dtype=np.float64)
        self.nodes = np.asarray(nodes)
        self.friction = friction
        self.restitution_coefficient = restitution
        self.multiplier = np.zeros(len(self.nodes))

    def gap(self, x):
        if self.kind == "planar":
            norm = np.linalg.norm(self.parameters[:3])
            return x @ (self.parameters[:3] / norm) + self.parameters[3] / norm
        return np.linalg.norm(x - self.parameters[:3]) - self.parameters[3]

    def normal(self, x):
        if self.kind == "planar":
            return self.parameters[:3] / np.linalg.norm(self.parameters[:3])
        offset = x - self.parameters[:3]
        return offset / np.linalg.norm(offset)

    def frames(self, q):
        k = len(self.nodes)
        normals = np.array([self.normal(q[node]) for node in self.nodes]).reshape(k, 3)
        frames = [_local_frame(n) for n in normals]
        return normals, np.array([f[0] for f in frames]).reshape(k, 3), np.array([f[1] for f in frames]).reshape(k, 3)

    def gaps(self, x):
        return np.array([self.gap(x[node]) for node in self.nodes])

    def begin_step(self, q, inverse, dt):
        k = len(self.nodes)
        self.inverse = inverse
        self.normals, self.t1, self.t2 = self.frames(q)
        # Ainv_cols[d][ci] = A_d^{-1} (n_ci[d] e_node_ci), likewise for the tangents.
        self.columns = [[inverse[:, node] * self.normals[c, d] for c, node in enumerate(self.nodes)] for d in range(3)]
        self.columns_t1 = [[inverse[:, node] * self.t1[c, d] for c, node in enumerate(self.nodes)] for d in range(3)]
        self.columns_t2 = [[inverse[:, node] * self.t2[c, d] for c, node in enumerate(self.nodes)] for d in range(3)]
        self.delassus = np.zeros((k, k))
        for ci in range(k):
            for cj in range(ci, k):
                value = sum(self.normals[cj, d] * self.columns[d][ci][self.nodes[cj]] for d in range(3))
                self.delassus[ci, cj] = self.delassus[cj, ci] = value
        self.tangential = []
        for ci, node in enumerate(self.nodes):
            w11 = sum(self.t1[ci, d] * self.columns_t1[d][ci][node] for d in range(3))
            w12 = sum(self.t1[ci, d] * self.columns_t2[d][ci][node] for d in range(3))
            w22 = sum(self.t2[ci, d] * self.columns_t2[d][ci][node] for d in range(3))
            self.tangential.append(np.array([[w11, w12], [w12, w22]]))
        self.r = np.maximum(dt * dt * np.diag(self.delassus), 1.0e-10)
        self.lambda_t = np.zeros((k, 2))

    def _combine(self, base, normal_weights, tangential):
        result = base.copy()
        for d in range(3):
            for c in range(len(self.nodes)):
                result[:, d] += normal_weights[c] * self.columns[d][c]
                if tangential:
                    result[:, d] += (
                        self.lambda_t[c, 0] * self.columns_t1[d][c] + self.lambda_t[c, 1] * self.columns_t2[d][c]
                    )
        return result

    def iterate(self, x, b_full, q):
        nodes = self.nodes
        delta = self.gaps(x)
        lam, r = self.multiplier, self.r
        s = np.maximum(np.sqrt(delta**2 + r**2 * lam**2), 1.0e-14)
        omega = 1.0 - delta / s
        diagonal = (1.0 - r * lam / s) * r
        system = omega[:, None] * self.delassus * omega[None, :] + np.diag(diagonal)
        solved = self.inverse @ b_full
        friction = self.friction > 0
        response = self._combine(solved, omega * lam, friction)
        rhs = -(delta + r * lam - s) + omega * np.einsum("ij,ij->i", self.normals, x[nodes])
        rhs -= omega * np.einsum("ij,ij->i", self.normals, response[nodes])
        lam = np.clip(lam + np.linalg.solve(system, rhs), 0.0, 1.0e8)
        self.multiplier = lam
        weights = omega * lam
        if friction:
            normal_only = self._combine(solved, weights, False)
            for c, node in enumerate(nodes):
                if lam[c] <= 0.0:
                    self.lambda_t[c] = 0.0
                    continue
                step = normal_only[node] - q[node]
                n = self.normals[c]
                tangential = step - (step @ n) * n
                u = np.array([tangential @ self.t1[c], tangential @ self.t2[c]])
                w = self.tangential[c]
                stick = np.linalg.inv(w) @ -u if abs(np.linalg.det(w)) > 1.0e-20 else np.zeros(2)
                radius = self.friction * lam[c]
                if np.linalg.norm(stick) <= radius + 1.0e-14:
                    self.lambda_t[c] = stick
                elif np.linalg.norm(u) > 1.0e-14:
                    self.lambda_t[c] = -radius * u / np.linalg.norm(u)
                else:
                    self.lambda_t[c] = 0.0
        return self._combine(solved, weights, friction)

    def restitution(self, x, q, v, dt):
        if self.restitution_coefficient <= 0:
            return x
        x = x.copy()
        for c, node in enumerate(self.nodes):
            if self.multiplier[c] <= 0:
                continue
            n = self.normal(x[node])
            if np.linalg.norm(n) <= 1.0e-12 or self.gap(x[node]) > 1.0e-8:
                continue
            n = n / np.linalg.norm(n)
            incoming = v[node] @ n
            if incoming < 0:
                change = -self.restitution_coefficient * incoming - ((x[node] - q[node]) / dt) @ n
                if change > 0:
                    x[node] += dt * change * n
        return x


class ReferenceMeshContact(ReferenceContact):
    """Literal transcription of the reference ``MeshFrictionalBoundary`` in the AlgPhd forward.

    Unlike the solver's fixed-size arrays, the candidate list is rebuilt every
    step from the 5-radius pre-filter, and the warm start is looked up by node,
    as in ``deformable_projective_dynamics_forward_gpu.cpp``.
    """

    def __init__(self, radius, split, faces_left, faces_right, nodes, friction=0.0, restitution=0.0):
        super().__init__("mesh", np.zeros(4), nodes, friction, restitution)
        self.radius, self.split = float(radius), int(split)
        self.faces_left = np.asarray(faces_left).reshape(-1, 3)
        self.faces_right = np.asarray(faces_right).reshape(-1, 3)
        self.registered = np.asarray(nodes)
        self.previous = {}

    def closest(self, p, node, surface):
        faces = self.faces_right if node < self.split else self.faces_left
        best, closest, normal = np.inf, np.zeros(3), np.array([0.0, 0.0, 1.0])
        for a, b, c in faces:
            point, _weights = _closest_point_triangle(p, surface[a], surface[b], surface[c])
            dist2 = (p - point) @ (p - point)
            if dist2 < best:
                best, closest = dist2, point
                n = np.cross(surface[b] - surface[a], surface[c] - surface[a])
                length = np.linalg.norm(n)
                normal = n / length if length > np.finfo(np.float64).eps else np.array([0.0, 0.0, 1.0])
        return closest, normal, best

    @staticmethod
    def direction(p, closest, face_normal):
        offset = p - closest
        length = np.linalg.norm(offset)
        return offset / length if length > 1.0e-12 else face_normal

    def frames(self, q):
        normals, t1, t2 = [], [], []
        for node in self.nodes:
            closest, face_normal, _dist2 = self.closest(q[node], node, q)
            n = self.direction(q[node], closest, face_normal)
            # Gram-Schmidt from the axis least parallel to n (first one on ties).
            e = np.eye(3)[int(np.argmin(np.abs(n)))]
            u = e - (n @ e) * n
            u /= np.linalg.norm(u)
            w = np.cross(n, u)
            normals.append(n)
            t1.append(u)
            t2.append(w / np.linalg.norm(w))
        k = len(self.nodes)
        return np.reshape(normals, (k, 3)), np.reshape(t1, (k, 3)), np.reshape(t2, (k, 3))

    def begin_step(self, q, inverse, dt):
        self.nodes = np.array(
            [
                node
                for node in self.registered
                if np.sqrt(self.closest(q[node], node, q)[2]) - self.radius <= 5.0 * self.radius
            ],
            dtype=np.int64,
        )
        super().begin_step(q, inverse, dt)
        self.multiplier = np.array([self.previous.get(int(node), 0.0) for node in self.nodes])

    def gaps(self, x):
        # UpdateVertices(q_sol), then GetSignedDistanceWithReferenceNormal.
        self.surface = x.copy()
        return np.array(
            [
                (x[node] - self.closest(x[node], node, x)[0]) @ self.normals[c] - self.radius
                for c, node in enumerate(self.nodes)
            ]
        )

    def registered_multiplier(self):
        lookup = dict(zip(self.nodes.tolist(), self.multiplier, strict=True))
        return np.array([lookup.get(int(node), 0.0) for node in self.registered])

    def restitution(self, x, q, v, dt):
        self.previous = dict(zip(self.nodes.tolist(), self.multiplier, strict=True))
        if self.restitution_coefficient <= 0:
            return x
        x = x.copy()
        for c, node in enumerate(self.nodes):
            if self.multiplier[c] <= 0:
                continue
            reference = self.normals[c]
            closest, face_normal, _dist2 = self.closest(x[node], node, self.surface)
            distance = (x[node] - closest) @ reference - self.radius
            n = self.direction(x[node], closest, face_normal)
            if n @ reference < 0:
                n = -n
            if np.linalg.norm(n) <= 1.0e-12:
                continue
            n = n / np.linalg.norm(n)
            if distance > 0.1 * self.radius:
                continue
            incoming = v[node] @ n
            if incoming < 0:
                change = -self.restitution_coefficient * incoming - ((x[node] - q[node]) / dt) @ n
                if change > 0:
                    x[node] += dt * change * n
        return x


def _projection_jacobian(projection, f, eps=1.0e-6):
    """Central-difference 9x9 Jacobian with Eigen's column-major flattening."""
    jacobian = np.zeros((9, 9))
    for c in range(9):
        df = np.zeros((3, 3))
        df[c % 3, c // 3] = 1.0
        derivative = (projection(f + eps * df) - projection(f - eps * df)) / (2.0 * eps)
        jacobian[:, c] = derivative.flatten(order="F")
    return jacobian


def _backward_tangents(n):
    axis = int(np.argmin(np.abs(n)))
    unit = np.eye(3)[axis]
    t1 = unit - (n @ unit) * n
    t1 = t1 / np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return t1, t2 / np.linalg.norm(t2)


def reference_backward(step, x_next, dt, dl_dq_next, dl_dv_next, contact=None, targets=None):
    """Dense reference backward pass; returns ``mu``, ``z`` and the gradients.

    Activation gradients are with respect to the contraction ``u = 1 - a``.

    ``contact`` is ``(nodes, multiplier, omega, normal, friction)`` from the forward step.
    The local Jacobians come from finite differences of the oracle projections,
    and the adjoint systems are solved directly on the free block.
    """
    n = step.n
    free = np.repeat(step.free, 3)
    k_matrix = np.zeros((3 * n, 3 * n))
    for e, tet in enumerate(step.tets):
        g = step.gradients[e]
        f = x_next[tet].T @ g
        jacobian = np.zeros((9, 9))
        for name, _weight in step.energies:
            if name == "corotated":
                projection = corotated_projection
            elif name == "volume":
                projection = volume_projection
            else:
                projection = lambda m: nh_projection(m, *step.nh)  # noqa: E731
            # The reference uses the mean weight of this energy in K only.
            jacobian += _weight.mean() * _projection_jacobian(projection, f)
        # vec(F) column-major: F[b, n] = sum_k x_k[b] g_k[n].
        mapping = np.zeros((9, 12))
        for k in range(4):
            for b in range(3):
                for col in range(3):
                    mapping[b + 3 * col, 3 * k + b] = g[k, col]
        local = mapping.T @ jacobian @ mapping
        if step.muscle_directions is not None and step.muscle_weight[e] > 0:
            m = step.muscle_directions[e]
            target = targets[e]
            fibre = _projection_jacobian(
                lambda mat, m=m, target=target: np.outer(muscle_projection(mat, m, target), m), f
            )
            local += step.muscle_weight[e] * mapping.T @ fibre @ mapping
        dofs = np.array([3 * i + d for i in tet for d in range(3)])
        k_matrix[np.ix_(dofs, dofs)] += local
    a_matrix = np.kron(step.matrix(dt), np.eye(3))
    operator = (a_matrix - k_matrix)[np.ix_(free, free)]

    def solve(rhs):
        out = np.zeros(3 * n)
        out[free] = np.linalg.solve(operator, rhs[free])
        return out

    rhs = (dl_dq_next + dl_dv_next / dt).ravel()
    mu = solve(rhs)
    z = mu.copy()
    if contact is not None:
        nodes, multiplier, omega, normal, friction = contact
        rows = []
        for c, node in enumerate(nodes):
            if multiplier[c] > 0 and omega[c] >= 0.5:
                rows.append((node, omega[c] * normal[c]))
                if friction > 0 and multiplier[c] > 1.0e-8:
                    rows.extend((node, t) for t in _backward_tangents(normal[c]))
        values = np.array([vector @ mu[3 * node : 3 * node + 3] for node, vector in rows])
        if len(rows):
            gate = max(1.0e-12, 1.0e-3 * np.abs(values).max())
            keep = np.abs(values) >= gate
            rows = [row for row, selected in zip(rows, keep, strict=True) if selected]
            values = values[keep]
            columns = []
            for node, vector in rows:
                j = np.zeros(3 * n)
                j[3 * node : 3 * node + 3] = vector
                columns.append(solve(j))
            gram = np.array(
                [[vector @ columns[k][3 * node : 3 * node + 3] for k in range(len(rows))] for node, vector in rows]
            )
            gram = np.tril(gram) + np.tril(gram, -1).T
            alpha = np.linalg.solve(gram, values)
            z = mu - sum(a * column for a, column in zip(alpha, columns, strict=True))
    inertia = step.mass / dt**2
    gradient_q = -(dl_dv_next.ravel() / dt) + inertia * mu
    gradient_v = dt * inertia * mu
    gradient_f = mu.copy()
    zv = z.reshape(-1, 3)
    material = np.zeros((len(step.tets), 2))
    activation = np.zeros(len(step.tets))
    for e, tet in enumerate(step.tets):
        g = step.gradients[e]
        f = x_next[tet].T @ g
        for name, _weight in step.energies:
            if name == "corotated":
                stress = f - corotated_projection(f)
            elif name == "volume":
                stress = f - volume_projection(f)
            else:
                mu_bar, lame_bar, _penalty = step.nh
                inverse_t = np.linalg.inv(f).T
                pk1 = mu_bar * (f - inverse_t) + lame_bar * np.log(max(np.linalg.det(f), 1.0e-12)) * inverse_t
                stress = pk1 / (2.0 * mu_bar)
            value = -step.vbar * sum(zv[tet[k]] @ (stress @ g[k]) for k in range(4))
            if name == "volume":
                material[e, 1] += value
            else:
                material[e, 0] += 2.0 * value
        if step.muscle_directions is not None and step.muscle_weight[e] > 0:
            m = step.muscle_directions[e]
            fibre = f @ m
            length = np.linalg.norm(fibre)
            if length > EPS:
                activation[e] = -sum(
                    zv[tet[k]] @ (step.muscle_weight[e] / length * (m @ g[k]) * fibre) for k in range(4)
                )
    return {
        "mu": mu.reshape(-1, 3),
        "z": zv,
        "q": gradient_q.reshape(-1, 3),
        "qd": gradient_v.reshape(-1, 3),
        "f": gradient_f.reshape(-1, 3),
        "materials": material,
        "activations": activation,
    }


def _closest_point_triangle(p, a, b, c):
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return a, np.array([1.0, 0.0, 0.0])
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return b, np.array([0.0, 1.0, 0.0])
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3)
        return a + v * ab, np.array([1 - v, v, 0.0])
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return c, np.array([0.0, 0.0, 1.0])
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        return a + w * ac, np.array([1 - w, 0.0, w])
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b), np.array([0.0, 1 - w, w])
    denom = 1 / (va + vb + vc)
    v, w = vb * denom, vc * denom
    return a + ab * v + ac * w, np.array([1 - v - w, v, w])


def mesh_contact_force(q, v, faces_a, faces_b, radius, kn_left, kn_right, kf, mu):
    """Literal ``MeshContactStateForce::ForwardForce`` in the reference's loop order."""
    q, v = np.asarray(q, dtype=np.float64), np.asarray(v, dtype=np.float64)
    force = np.zeros_like(q)
    box_a, box_b = q[np.unique(faces_a)], q[np.unique(faces_b)]
    gap = np.maximum(np.maximum(box_b.min(0) - box_a.max(0), box_a.min(0) - box_b.max(0)), 0.0)
    if gap @ gap > radius * radius:
        return force

    def accumulate(vertices, faces, kn):
        for vid in vertices:
            p = q[vid]
            for face in faces:
                a, b, c = q[face]
                closest, bary = _closest_point_triangle(p, a, b, c)
                dist2 = (p - closest) @ (p - closest)
                if dist2 >= radius * radius:
                    continue
                dist = np.sqrt(dist2)
                if dist > EPS:
                    n = (p - closest) / dist
                else:
                    n = np.cross(b - a, c - a)
                    if np.linalg.norm(n) <= EPS:
                        continue
                    n = n / np.linalg.norm(n)
                if n @ (p - closest) < 0:
                    n = -n
                fn_mag = kn * (radius - dist)
                vrel = v[vid] - bary @ v[face]
                vrel_t = vrel - n * (vrel @ n)
                ff = np.zeros(3)
                if np.linalg.norm(vrel_t) > EPS:
                    ff = -min(kf, mu * fn_mag / (np.linalg.norm(vrel_t) + EPS)) * vrel_t
                total = fn_mag * n + ff
                force[vid] += total
                force[face] -= np.outer(bary, total)

    accumulate(np.unique(faces_a), faces_b, kn_left)
    accumulate(np.unique(faces_b), faces_a, kn_right)
    return force
