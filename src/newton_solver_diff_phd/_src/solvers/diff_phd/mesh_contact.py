# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Portions translated from DiffPhD (https://github.com/Akinesia112/DiffPhD,
# commit d319435), Copyright (c) 2021 MIT Graphics Group, MIT License.
# See THIRD_PARTY_NOTICES.md.

"""The reference ``mesh_contact`` state force between two deformable bodies.

This follows ``MeshContactStateForce`` (``cpp/core/src/state_force/
mesh_contact_state_force.cpp``) as registered by ``Deformable::AddStateForce``.
Every vertex of the faces of body A is tested against every face of body B
with stiffness ``kn_left``, and every vertex of B against every face of A with
``kn_right``. Within ``contact_radius`` of the closest point on a face, the
vertex receives ``kn (radius - dist) n`` along the closest-point direction plus
the friction ``-min(kf, mu fn / (|v_t| + eps)) v_t`` of the tangential velocity
relative to the face; the face's three vertices receive the opposite force
weighted by the closest point's barycentric coordinates.

As in ``ForwardProjectiveDynamicsAlgPhd`` the force is evaluated once per step
on the start-of-step positions and velocities and enters the inertial target
``q~ = q + h v + h^2/m (f_ext + f(q, v))``. Its ``BackwardForce`` returns zero,
so it contributes nothing to gradients. The reference skips the evaluation when
the bodies' bounding boxes are farther apart than ``contact_radius``; no pair can
be in contact then, so evaluating every pair gives the same force. The force on
each particle is summed in a fixed order instead of the reference's loop order.
"""

import numpy as np
import warp as wp

from .projections import DOUBLE_EPSILON


@wp.func
def closest_point_triangle(p: wp.vec3d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d):
    """``ClosestPointTriangle``: closest point, barycentric coordinates and squared distance."""
    zero = wp.float64(0.0)
    one = wp.float64(1.0)
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= zero and d2 <= zero:
        return a, wp.vec3d(one, zero, zero), wp.length_sq(p - a)
    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= zero and d4 <= d3:
        return b, wp.vec3d(zero, one, zero), wp.length_sq(p - b)
    vc = d1 * d4 - d3 * d2
    if vc <= zero and d1 >= zero and d3 <= zero:
        v = d1 / (d1 - d3)
        point = a + v * ab
        return point, wp.vec3d(one - v, v, zero), wp.length_sq(p - point)
    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= zero and d5 <= d6:
        return c, wp.vec3d(zero, zero, one), wp.length_sq(p - c)
    vb = d5 * d2 - d1 * d6
    if vb <= zero and d2 >= zero and d6 <= zero:
        w = d2 / (d2 - d6)
        point = a + w * ac
        return point, wp.vec3d(one - w, zero, w), wp.length_sq(p - point)
    va = d3 * d6 - d5 * d4
    if va <= zero and (d4 - d3) >= zero and (d5 - d6) >= zero:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        point = b + w * (c - b)
        return point, wp.vec3d(zero, one - w, w), wp.length_sq(p - point)
    denom = one / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    point = a + ab * v + ac * w
    return point, wp.vec3d(one - v - w, v, w), wp.length_sq(p - point)


@wp.kernel(enable_backward=False)
def _pair_forces(
    position: wp.array[wp.vec3d],
    velocity: wp.array[wp.vec3d],
    vertices: wp.array[wp.int32],
    faces: wp.array[wp.vec3i],
    radius: wp.float64,
    kn: wp.float64,
    kf: wp.float64,
    mu: wp.float64,
    force: wp.array2d[wp.vec3d],
    bary: wp.array2d[wp.vec3d],
    pairs: wp.array[wp.int32],
):
    s, f = wp.tid()
    zero = wp.vec3d(wp.float64(0.0))
    force[s, f] = zero
    bary[s, f] = zero
    vid = vertices[s]
    face = faces[f]
    p = position[vid]
    a = position[face[0]]
    b = position[face[1]]
    c = position[face[2]]
    closest, weights, dist2 = closest_point_triangle(p, a, b, c)
    if dist2 >= radius * radius:
        return
    dist = wp.sqrt(dist2)
    n = zero
    if dist > DOUBLE_EPSILON:
        n = (p - closest) / dist
    else:
        n = wp.cross(b - a, c - a)
        n_norm = wp.length(n)
        if n_norm <= DOUBLE_EPSILON:
            return
        n = n / n_norm
    if wp.dot(n, p - closest) < wp.float64(0.0):
        n = -n
    fn_mag = kn * (radius - dist)
    fn = fn_mag * n
    vtri = weights[0] * velocity[face[0]] + weights[1] * velocity[face[1]] + weights[2] * velocity[face[2]]
    vrel = velocity[vid] - vtri
    vrel_t = vrel - n * wp.dot(vrel, n)
    vrel_t_mag = wp.length(vrel_t)
    ff = zero
    if vrel_t_mag > DOUBLE_EPSILON:
        ff = -wp.min(kf, mu * fn_mag / (vrel_t_mag + DOUBLE_EPSILON)) * vrel_t
    force[s, f] = fn + ff
    bary[s, f] = weights
    wp.atomic_add(pairs, 0, 1)


@wp.kernel(enable_backward=False)
def _gather_forces(
    slot_a: wp.array[wp.int32],
    slot_b: wp.array[wp.int32],
    force_ab: wp.array2d[wp.vec3d],
    bary_ab: wp.array2d[wp.vec3d],
    force_ba: wp.array2d[wp.vec3d],
    bary_ba: wp.array2d[wp.vec3d],
    incidence_a_offsets: wp.array[wp.int32],
    incidence_a: wp.array[wp.vec2i],
    incidence_b_offsets: wp.array[wp.int32],
    incidence_b: wp.array[wp.vec2i],
    out: wp.array[wp.vec3d],
):
    i = wp.tid()
    total = wp.vec3d(wp.float64(0.0))
    # As a vertex of A against the faces of B, and of B against the faces of A.
    if slot_a[i] >= 0:
        for f in range(force_ab.shape[1]):
            total += force_ab[slot_a[i], f]
    if slot_b[i] >= 0:
        for f in range(force_ba.shape[1]):
            total += force_ba[slot_b[i], f]
    # As a corner of a face of B pushed by the vertices of A, and vice versa.
    for k in range(incidence_b_offsets[i], incidence_b_offsets[i + 1]):
        entry = incidence_b[k]
        for s in range(force_ab.shape[0]):
            total -= force_ab[s, entry[0]] * bary_ab[s, entry[0]][entry[1]]
    for k in range(incidence_a_offsets[i], incidence_a_offsets[i + 1]):
        entry = incidence_a[k]
        for s in range(force_ba.shape[0]):
            total -= force_ba[s, entry[0]] * bary_ba[s, entry[0]][entry[1]]
    out[i] = total


@wp.kernel(enable_backward=False)
def add_state_force(force: wp.array[wp.vec3d], dt: wp.float64, mass: wp.float64, target: wp.array[wp.vec3d]):
    """``q~ += h^2/m f`` for every particle, as in the reference's ``h2m * ForwardStateForce``."""
    i = wp.tid()
    target[i] = target[i] + dt * dt / mass * force[i]


def _incidence(faces: np.ndarray, particle_count: int):
    """Per particle, the (face, corner) entries of ``faces`` in face order (CSR)."""
    entries = [[] for _ in range(particle_count)]
    for f, face in enumerate(faces):
        for corner, node in enumerate(face):
            entries[node].append((f, corner))
    offsets = np.zeros(particle_count + 1, dtype=np.int32)
    offsets[1:] = np.cumsum([len(e) for e in entries])
    flat = np.array([pair for e in entries for pair in e], dtype=np.int32).reshape(-1, 2)
    return offsets, flat


class MeshContact:
    """``MeshContactStateForce`` with the reference parameters.

    Args:
        contact_radius: Contact radius [m].
        kn_left: Normal stiffness [N/m] of A's vertices against B's faces.
        kn_right: Normal stiffness [N/m] of B's vertices against A's faces.
        kf: Upper bound of the friction coefficient per tangential speed [N s/m].
        mu: Coulomb coefficient (dimensionless).
        faces_a, faces_b: ``(F, 3)`` surface triangles of the two bodies in particle indices.
        particle_count: Number of particles.
        device: Warp device.
    """

    def __init__(self, contact_radius, kn_left, kn_right, kf, mu, faces_a, faces_b, particle_count, device):
        for name, value in (("contact_radius", contact_radius), ("kn_left", kn_left), ("kn_right", kn_right),
                            ("kf", kf), ("mu", mu)):  # fmt: skip
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"mesh_contact {name} must be finite and nonnegative")
        faces_a = np.asarray(faces_a, dtype=np.int32).reshape(-1, 3)
        faces_b = np.asarray(faces_b, dtype=np.int32).reshape(-1, 3)
        if not len(faces_a) or not len(faces_b):
            raise ValueError("mesh_contact needs faces on both bodies")
        for faces in (faces_a, faces_b):
            if faces.min() < 0 or faces.max() >= particle_count:
                raise ValueError("mesh_contact faces must reference existing particles")
        self.parameters = tuple(float(value) for value in (contact_radius, kn_left, kn_right, kf, mu))
        self.device = device
        # Initialize: the sorted unique vertices of each body's faces.
        vertices_a, vertices_b = np.unique(faces_a), np.unique(faces_b)
        slot_a = np.full(particle_count, -1, dtype=np.int32)
        slot_b = np.full(particle_count, -1, dtype=np.int32)
        slot_a[vertices_a] = np.arange(len(vertices_a), dtype=np.int32)
        slot_b[vertices_b] = np.arange(len(vertices_b), dtype=np.int32)
        self.vertices_a = wp.array(vertices_a.astype(np.int32), dtype=wp.int32, device=device)
        self.vertices_b = wp.array(vertices_b.astype(np.int32), dtype=wp.int32, device=device)
        self.faces_a = wp.array(faces_a, dtype=wp.vec3i, device=device)
        self.faces_b = wp.array(faces_b, dtype=wp.vec3i, device=device)
        self.slot_a = wp.array(slot_a, dtype=wp.int32, device=device)
        self.slot_b = wp.array(slot_b, dtype=wp.int32, device=device)
        self.incidence_a = tuple(
            wp.array(a, dtype=t, device=device)
            for a, t in zip(_incidence(faces_a, particle_count), (wp.int32, wp.vec2i), strict=True)
        )
        self.incidence_b = tuple(
            wp.array(a, dtype=t, device=device)
            for a, t in zip(_incidence(faces_b, particle_count), (wp.int32, wp.vec2i), strict=True)
        )
        self.force_ab = wp.zeros((len(vertices_a), len(faces_b)), dtype=wp.vec3d, device=device)
        self.bary_ab = wp.zeros_like(self.force_ab)
        self.force_ba = wp.zeros((len(vertices_b), len(faces_a)), dtype=wp.vec3d, device=device)
        self.bary_ba = wp.zeros_like(self.force_ba)
        self.force = wp.zeros(particle_count, dtype=wp.vec3d, device=device)
        # Number of vertex-face pairs within the contact radius in the last step.
        self.pairs = wp.zeros(1, dtype=wp.int32, device=device)

    def evaluate(self, position, velocity) -> None:
        """``ForwardForce(q, v)`` into :attr:`force` [N]."""
        radius, kn_left, kn_right, kf, mu = self.parameters
        device = self.device
        self.pairs.zero_()
        for vertices, faces, kn, force, bary in (
            (self.vertices_a, self.faces_b, kn_left, self.force_ab, self.bary_ab),
            (self.vertices_b, self.faces_a, kn_right, self.force_ba, self.bary_ba),
        ):
            wp.launch(
                _pair_forces,
                force.shape,
                inputs=[position, velocity, vertices, faces, radius, kn, kf, mu],
                outputs=[force, bary, self.pairs],
                device=device,
            )
        wp.launch(
            _gather_forces,
            len(self.force),
            inputs=[
                self.slot_a,
                self.slot_b,
                self.force_ab,
                self.bary_ab,
                self.force_ba,
                self.bary_ba,
                *self.incidence_a,
                *self.incidence_b,
            ],
            outputs=[self.force],
            device=device,
        )
