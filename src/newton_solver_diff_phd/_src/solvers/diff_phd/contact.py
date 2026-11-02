# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Portions translated from DiffPhD (https://github.com/Akinesia112/DiffPhD,
# commit d319435), Copyright (c) 2021 MIT Graphics Group, MIT License.
# See THIRD_PARTY_NOTICES.md.

"""Frictional boundary contact of the reference DiffPhD step.

This follows the contact branch of ``ForwardProjectiveDynamicsAlgPhd``
(``cpp/core/src/fem/deformable_projective_dynamics_forward_gpu.cpp``) with the
planar and spherical boundaries of ``cpp/core/src/friction``. Every registered
vertex is a candidate on every step. The normal ``n``, tangents and Delassus
matrix ``W`` are frozen at the start of the step; each local/global iteration
performs one semismooth Newton step on the Fischer-Burmeister function
``phi = delta + r lambda - sqrt(delta^2 + r^2 lambda^2)`` with
``r = max(h^2 W_cc, 1e-10)``, solving ``(Omega W Omega + E) d lambda = -phi +
Omega n.x - Omega n.(A^{-1} g)`` and clamping ``lambda`` to ``[0, 1e8]``.
Coulomb friction is an isotropic stick/slip projection per contact using the
2x2 tangential Delassus block. The applied force is ``omega lambda n``.

``A`` is the same scalar matrix for every coordinate and the frames are
orthonormal, so the reference Delassus entries reduce to
``W_ij = (n_i . n_j) B_ij`` and the tangential blocks to ``B_ii``, with
``B = A^{-1}`` restricted to the candidate vertices. ``B`` and the ``N x K``
columns ``A^{-1} e_j`` of the registered vertices are formed once per
factorization, as the reference's per-step ``Ainv_cols``; each iteration then
forms ``A^{-1} (b + J^T lambda)`` as ``A^{-1} b`` plus the columns of the coupled
candidates instead of a second sparse solve. Both are reorderings of the
reference arithmetic.

Separated candidates (``lambda = 0``, ``delta > 0``) have ``omega = 0`` exactly
(``sqrt(fl(delta^2)) = delta`` in IEEE arithmetic), so their rows and columns of
``Omega W Omega`` vanish and ``Omega W Omega + E`` is block diagonal. Only the
coupled block is factorized; the other rows reduce to ``d lambda = rhs / E``.
The coupled block ``Omega_a W_aa Omega_a + E_a`` is symmetric positive definite,
so blocks of at most 32 rows use a tile Cholesky factorization and larger ones
Eigen's LDLT as in the reference. This is the same linear algebra.

The ``"mesh"`` kind is ``MeshFrictionalBoundary``
(``cpp/core/src/friction/mesh_frictional_boundary.cpp``) between two bodies of
the model. A registered vertex with index below ``split_index`` is tested
against ``faces_right``, every other one against ``faces_left``. At the step
start the forward pre-filters the candidates: only registered vertices whose
unsigned distance to the opposite surface, minus ``contact_radius``, is at most
``5 contact_radius`` take part in the step. The distance is measured against
the start-of-step surface, including prescribed particles at their starting
positions, exactly as in the reference. The filtered vertices stay in
the fixed-size arrays with ``omega = 0`` and ``lambda = 0``, which removes them
from every product exactly as the reference's shorter candidate list does,
and their multipliers restart from zero as in its warm start. The normal is
anchored at the step start, ``(p - c) / |p - c|`` for the closest point ``c``
(the face normal when ``p`` lies on the face), with Gram-Schmidt tangents. Each
iteration measures the gap ``(p - c).n - contact_radius`` against the opposite
surface at the iterate's positions. Restitution uses a slop of
``0.1 contact_radius`` and the current closest-point normal oriented along the
anchored one, against the surface at the positions of the last iteration's start.


"""

import numpy as np
import warp as wp

from .dense import TILE, ldlt_solve, spd_tile_solve
from .mesh_contact import closest_point_triangle
from .projections import DOUBLE_EPSILON

BOUNDARIES = {"planar": 0, "spherical": 1, "mesh": 2}
# forward_gpu.cpp: mesh candidates lie within 5 contact radii at the step start;
# restitution acts within 0.1 contact radius ("friction_contact_slop").
MESH_CANDIDATE_RADII = 5.0
MESH_RESTITUTION_SLOP = 0.1
# Threads of the single-block candidate compaction.
SCAN_BLOCK = 256


@wp.func
def boundary_gap(x: wp.vec3d, kind: int, parameters: wp.vec4d):
    """``GetDistance`` of the planar (``(n.x + offset) / |n|``) or spherical boundary."""
    result = wp.float64(0.0)
    if kind == 0:
        axis = wp.vec3d(parameters[0], parameters[1], parameters[2])
        norm = wp.length(axis)
        result = wp.dot(x, axis / norm) + parameters[3] / norm
    else:
        center = wp.vec3d(parameters[0], parameters[1], parameters[2])
        result = wp.length(x - center) - parameters[3]
    return result


@wp.func
def boundary_normal(x: wp.vec3d, kind: int, parameters: wp.vec4d):
    """``GetNormal`` of the planar or spherical boundary."""
    if kind == 0:
        return wp.normalize(wp.vec3d(parameters[0], parameters[1], parameters[2]))
    offset = x - wp.vec3d(parameters[0], parameters[1], parameters[2])
    return offset / wp.length(offset)


@wp.func
def mesh_closest(
    p: wp.vec3d,
    node: int,
    x: wp.array[wp.vec3d],
    faces: wp.array[wp.vec3i],
    split: int,
    left_count: int,
):
    """Closest point, unit face normal and squared distance on the surface opposite ``node``.

    The first face with the smallest distance wins, as in the reference loop;
    an empty surface returns an infinite distance.
    """
    first = left_count
    last = faces.shape[0]
    if node >= split:
        first = 0
        last = left_count
    best = wp.float64(wp.inf)
    closest = wp.vec3d(wp.float64(0.0))
    normal = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
    for f in range(first, last):
        face = faces[f]
        a = x[face[0]]
        b = x[face[1]]
        c = x[face[2]]
        point, _weights, dist2 = closest_point_triangle(p, a, b, c)
        if dist2 < best:
            best = dist2
            closest = point
            n = wp.cross(b - a, c - a)
            length = wp.length(n)
            if length > DOUBLE_EPSILON:
                normal = n / length
            else:
                normal = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
    return closest, normal, best


@wp.func
def mesh_direction(p: wp.vec3d, closest: wp.vec3d, face_normal: wp.vec3d):
    """``GetClosestPointAndNormal``: from the opposite surface toward ``p``, or the face normal on it."""
    offset = p - closest
    length = wp.length(offset)
    if length > wp.float64(1.0e-12):
        return offset / length
    return face_normal


@wp.func
def complementarity(delta: wp.float64, lam: wp.float64, r: wp.float64):
    """``omega``, the diagonal ``E`` and ``phi`` of the Fischer-Burmeister function."""
    s = wp.sqrt(delta * delta + r * r * lam * lam)
    if s < wp.float64(1.0e-14):
        s = wp.float64(1.0e-14)
    return wp.float64(1.0) - delta / s, (wp.float64(1.0) - r * lam / s) * r, delta + r * lam - s


@wp.kernel(enable_backward=False)
def _gather_block(columns: wp.array2d[wp.float64], nodes: wp.array[wp.int32], block: wp.array2d[wp.float64]):
    i, j = wp.tid()
    block[i, j] = columns[nodes[i], j]


@wp.kernel(enable_backward=False)
def _unit_columns(nodes: wp.array[wp.int32], columns: wp.array2d[wp.float64]):
    j = wp.tid()
    columns[nodes[j], j] = wp.float64(1.0)


@wp.kernel(enable_backward=False)
def begin_contact_step(
    position: wp.array[wp.vec3d],
    nodes: wp.array[wp.int32],
    kind: int,
    parameters: wp.array[wp.vec4d],
    normal: wp.array[wp.vec3d],
    tangent1: wp.array[wp.vec3d],
    tangent2: wp.array[wp.vec3d],
    friction_lambda: wp.array[wp.vec2d],
):
    c = wp.tid()
    n = boundary_normal(position[nodes[c]], kind, parameters[0])
    # GetLocalFrame: t1 = normalize(n x e_i) for the axis maximizing |n x e_i|.
    best = wp.vec3d(wp.float64(0.0))
    for i in range(3):
        axis = wp.vec3d(wp.float64(0.0))
        axis[i] = wp.float64(1.0)
        candidate = wp.cross(n, axis)
        if wp.dot(candidate, candidate) > wp.dot(best, best):
            best = candidate
    t1 = best / wp.length(best)
    t2 = wp.cross(n, t1)
    normal[c] = n
    tangent1[c] = t1
    tangent2[c] = t2 / wp.length(t2)
    friction_lambda[c] = wp.vec2d(wp.float64(0.0))


@wp.kernel(enable_backward=False)
def begin_mesh_step(
    position: wp.array[wp.vec3d],
    nodes: wp.array[wp.int32],
    faces: wp.array[wp.vec3i],
    split: int,
    left_count: int,
    radius: wp.float64,
    candidate: wp.array[wp.int32],
    normal: wp.array[wp.vec3d],
    tangent1: wp.array[wp.vec3d],
    tangent2: wp.array[wp.vec3d],
    multiplier: wp.array[wp.float64],
    friction_lambda: wp.array[wp.vec2d],
):
    c = wp.tid()
    node = nodes[c]
    p = position[node]
    closest, face_normal, dist2 = mesh_closest(p, node, position, faces, split, left_count)
    # Reference candidate filter uses the surface at the start of the step.
    active = wp.sqrt(dist2) - radius <= wp.float64(MESH_CANDIDATE_RADII) * radius
    n = mesh_direction(p, closest, face_normal)
    # Gram-Schmidt from the axis least parallel to n.
    axis = int(0)
    for d in range(1, 3):
        if wp.abs(n[d]) < wp.abs(n[axis]):
            axis = d
    unit = wp.vec3d(wp.float64(0.0))
    unit[axis] = wp.float64(1.0)
    t1 = unit - wp.dot(n, unit) * n
    t1 = t1 / wp.length(t1)
    t2 = wp.cross(n, t1)
    normal[c] = n
    tangent1[c] = t1
    tangent2[c] = t2 / wp.length(t2)
    candidate[c] = wp.where(active, 1, 0)
    if not active:
        # A vertex outside the candidate list restarts its warm start at zero.
        multiplier[c] = wp.float64(0.0)
    friction_lambda[c] = wp.vec2d(wp.float64(0.0))


@wp.kernel(enable_backward=False)
def contact_delassus(
    block: wp.array2d[wp.float64],
    normal: wp.array[wp.vec3d],
    dt: wp.float64,
    delassus: wp.array2d[wp.float64],
    regularization: wp.array[wp.float64],
):
    i, j = wp.tid()
    low = wp.min(i, j)
    high = wp.max(i, j)
    # W(ci, cj) = sum_d n_cj[d] A^{-1}(n_ci[d] e_ci)[node_cj] for cj >= ci, mirrored.
    delassus[i, j] = wp.dot(normal[low], normal[high]) * block[high, low]
    if i == j:
        regularization[i] = wp.max(dt * dt * delassus[i, i], wp.float64(1.0e-10))


@wp.kernel(enable_backward=False)
def contact_scalars(
    x: wp.array[wp.vec3d],
    nodes: wp.array[wp.int32],
    kind: int,
    parameters: wp.array[wp.vec4d],
    multiplier: wp.array[wp.float64],
    regularization: wp.array[wp.float64],
    gap: wp.array[wp.float64],
    omega: wp.array[wp.float64],
    diagonal: wp.array[wp.float64],
    residual: wp.array[wp.float64],
):
    c = wp.tid()
    delta = boundary_gap(x[nodes[c]], kind, parameters[0])
    gap[c] = delta
    w, e, phi = complementarity(delta, multiplier[c], regularization[c])
    omega[c] = w
    diagonal[c] = e
    residual[c] = phi


@wp.kernel(enable_backward=False)
def mesh_contact_scalars(
    x: wp.array[wp.vec3d],
    nodes: wp.array[wp.int32],
    faces: wp.array[wp.vec3i],
    split: int,
    left_count: int,
    radius: wp.float64,
    candidate: wp.array[wp.int32],
    normal: wp.array[wp.vec3d],
    multiplier: wp.array[wp.float64],
    regularization: wp.array[wp.float64],
    gap: wp.array[wp.float64],
    omega: wp.array[wp.float64],
    diagonal: wp.array[wp.float64],
    residual: wp.array[wp.float64],
):
    c = wp.tid()
    if candidate[c] == 0:
        # Not in this step's candidate list: no row, no force, lambda stays zero.
        gap[c] = wp.float64(0.0)
        omega[c] = wp.float64(0.0)
        diagonal[c] = wp.float64(1.0)
        residual[c] = wp.float64(0.0)
        return
    node = nodes[c]
    p = x[node]
    closest, _normal, _dist2 = mesh_closest(p, node, x, faces, split, left_count)
    # GetSignedDistanceWithReferenceNormal against the surface at the iterate.
    delta = wp.dot(p - closest, normal[c]) - radius
    gap[c] = delta
    w, e, phi = complementarity(delta, multiplier[c], regularization[c])
    omega[c] = w
    diagonal[c] = e
    residual[c] = phi


@wp.kernel(enable_backward=False)
def mesh_contact_scalars_tiled(
    x: wp.array[wp.vec3d],
    nodes: wp.array[wp.int32],
    faces: wp.array[wp.vec3i],
    split: int,
    left_count: int,
    radius: wp.float64,
    candidate: wp.array[wp.int32],
    normal: wp.array[wp.vec3d],
    multiplier: wp.array[wp.float64],
    regularization: wp.array[wp.float64],
    gap: wp.array[wp.float64],
    omega: wp.array[wp.float64],
    diagonal: wp.array[wp.float64],
    residual: wp.array[wp.float64],
):
    # mesh_contact_scalars with one warp per candidate; lanes split the faces.
    c, lane = wp.tid()
    if candidate[c] == 0:
        if lane == 0:
            gap[c] = wp.float64(0.0)
            omega[c] = wp.float64(0.0)
            diagonal[c] = wp.float64(1.0)
            residual[c] = wp.float64(0.0)
        return
    node = nodes[c]
    p = x[node]
    first = left_count
    last = faces.shape[0]
    if node >= split:
        first = 0
        last = left_count
    best = wp.float64(wp.inf)
    best_face = last
    closest = wp.vec3d(wp.float64(0.0))
    for f in range(first + lane, last, 32):
        face = faces[f]
        point, _weights, dist2 = closest_point_triangle(p, x[face[0]], x[face[1]], x[face[2]])
        if dist2 < best:
            best = dist2
            best_face = f
            closest = point
    # The reference keeps the first face with the smallest distance.
    smallest = wp.tile_min(wp.tile(best))[0]
    owner = last
    if best == smallest:
        owner = best_face
    winner = wp.tile_min(wp.tile(owner))[0]
    if best_face == winner and best == smallest:
        delta = wp.dot(p - closest, normal[c]) - radius
        gap[c] = delta
        w, e, phi = complementarity(delta, multiplier[c], regularization[c])
        omega[c] = w
        diagonal[c] = e
        residual[c] = phi


@wp.kernel(enable_backward=False)
def contact_forces(
    normal: wp.array[wp.vec3d],
    tangent1: wp.array[wp.vec3d],
    tangent2: wp.array[wp.vec3d],
    omega: wp.array[wp.float64],
    multiplier: wp.array[wp.float64],
    friction_lambda: wp.array[wp.vec2d],
    tangential: int,
    force: wp.array[wp.vec3d],
):
    c = wp.tid()
    value = omega[c] * multiplier[c] * normal[c]
    if tangential != 0:
        t = friction_lambda[c]
        value += t[0] * tangent1[c] + t[1] * tangent2[c]
    force[c] = value


@wp.kernel(enable_backward=False)
def contact_response(
    block: wp.array2d[wp.float64],
    force: wp.array[wp.vec3d],
    order: wp.array[wp.int32],
    active: wp.array[wp.int32],
    response: wp.array[wp.vec3d],
):
    # (A^{-1} J^T lambda) at the contact points; only coupled candidates carry force.
    c = wp.tid()
    value = wp.vec3d(wp.float64(0.0))
    for k in range(active[0]):
        j = order[k]
        value += block[c, j] * force[j]
    response[c] = value


@wp.kernel(enable_backward=False)
def contact_solution(
    solved: wp.array[wp.vec3d],
    panel: wp.array2d[wp.float64],
    force: wp.array[wp.vec3d],
    order: wp.array[wp.int32],
    active: wp.array[wp.int32],
    x: wp.array[wp.vec3d],
):
    # x = A^{-1} (b + J^T lambda) = A^{-1} b + sum_j A^{-1} e_j f_j over the coupled candidates.
    i = wp.tid()
    value = solved[i]
    for k in range(active[0]):
        j = order[k]
        value += panel[i, j] * force[j]
    x[i] = value


@wp.kernel(enable_backward=False)
def contact_newton_rhs(
    solved: wp.array[wp.vec3d],
    x: wp.array[wp.vec3d],
    nodes: wp.array[wp.int32],
    normal: wp.array[wp.vec3d],
    omega: wp.array[wp.float64],
    residual: wp.array[wp.float64],
    response: wp.array[wp.vec3d],
    rhs: wp.array[wp.float64],
):
    c = wp.tid()
    n = normal[c]
    node = nodes[c]
    point = x[node]
    point_solved = solved[node]
    # -phi + omega J x - omega J (A^{-1} g) with g = b + J^T lambda.
    rhs[c] = -residual[c] + omega[c] * wp.dot(n, point) - omega[c] * wp.dot(n, point_solved + response[c])


@wp.kernel(enable_backward=False)
def contact_active(
    omega: wp.array[wp.float64],
    count: int,
    order: wp.array[wp.int32],
    slot: wp.array[wp.int32],
    active: wp.array[wp.int32],
):
    # Deterministic compaction of the coupled candidates (omega != 0).
    c = wp.tid()
    k = int(0)
    for j in range(c):
        if omega[j] != wp.float64(0.0):
            k += 1
    slot[c] = -1
    if omega[c] != wp.float64(0.0):
        order[k] = c
        slot[c] = k
        k += 1
    if c == count - 1:
        active[0] = k


@wp.kernel(enable_backward=False)
def contact_active_scan(
    omega: wp.array[wp.float64],
    count: int,
    order: wp.array[wp.int32],
    slot: wp.array[wp.int32],
    active: wp.array[wp.int32],
):
    # contact_active with one block: an exclusive scan per chunk keeps the candidate order.
    _block, lane = wp.tid()
    base = int(0)
    for start in range(0, count, SCAN_BLOCK):
        c = start + lane
        flag = int(0)
        if c < count:
            if omega[c] != wp.float64(0.0):
                flag = 1
        flags = wp.tile(flag)
        k = base + wp.untile(wp.tile_scan_exclusive(flags))
        if c < count:
            slot[c] = -1
            if flag == 1:
                order[k] = c
                slot[c] = k
        base += wp.tile_sum(flags)[0]
    if lane == 0:
        active[0] = base


@wp.kernel(enable_backward=False)
def contact_system(
    delassus: wp.array2d[wp.float64],
    omega: wp.array[wp.float64],
    diagonal: wp.array[wp.float64],
    rhs: wp.array[wp.float64],
    order: wp.array[wp.int32],
    active: wp.array[wp.int32],
    matrix: wp.array2d[wp.float64],
    compact: wp.array[wp.float64],
):
    i, j = wp.tid()
    n = active[0]
    if i < n and j < n:
        a = order[i]
        b = order[j]
        value = omega[a] * delassus[a, b] * omega[b]
        if i == j:
            value += diagonal[a]
        matrix[i, j] = value
        if j == 0:
            compact[i] = rhs[a]
    elif i < TILE and j < TILE:
        # Identity padding of the tile solve.
        matrix[i, j] = wp.where(i == j, wp.float64(1.0), wp.float64(0.0))
        if j == 0:
            compact[i] = wp.float64(0.0)


@wp.kernel(enable_backward=False)
def contact_update(
    rhs: wp.array[wp.float64],
    diagonal: wp.array[wp.float64],
    slot: wp.array[wp.int32],
    compact: wp.array[wp.float64],
    multiplier: wp.array[wp.float64],
):
    c = wp.tid()
    k = slot[c]
    step = wp.float64(0.0)
    if k >= 0:
        step = compact[k]
    else:
        step = rhs[c] / diagonal[c]
    multiplier[c] = wp.min(wp.max(multiplier[c] + step, wp.float64(0.0)), wp.float64(1.0e8))


@wp.kernel(enable_backward=False)
def contact_friction(
    solved: wp.array[wp.vec3d],
    position: wp.array[wp.vec3d],
    nodes: wp.array[wp.int32],
    normal: wp.array[wp.vec3d],
    tangent1: wp.array[wp.vec3d],
    tangent2: wp.array[wp.vec3d],
    block: wp.array2d[wp.float64],
    multiplier: wp.array[wp.float64],
    normal_response: wp.array[wp.vec3d],
    friction: wp.float64,
    friction_lambda: wp.array[wp.vec2d],
):
    c = wp.tid()
    if multiplier[c] <= wp.float64(0.0):
        friction_lambda[c] = wp.vec2d(wp.float64(0.0))
        return
    # Normal-only update at this node with the new normal multipliers.
    node = nodes[c]
    step = solved[node] + normal_response[c] - position[node]
    n = normal[c]
    t1 = tangent1[c]
    t2 = tangent2[c]
    tangential = step - wp.dot(step, n) * n
    u1 = wp.dot(tangential, t1)
    u2 = wp.dot(tangential, t2)
    diagonal = block[c, c]
    w11 = wp.dot(t1, t1) * diagonal
    w12 = wp.dot(t1, t2) * diagonal
    w22 = wp.dot(t2, t2) * diagonal
    determinant = w11 * w22 - w12 * w12
    stick = wp.vec2d(wp.float64(0.0))
    if wp.abs(determinant) > wp.float64(1.0e-20):
        stick = wp.vec2d(-(w22 * u1 - w12 * u2), -(-w12 * u1 + w11 * u2)) / determinant
    radius = friction * multiplier[c]
    if wp.length(stick) <= radius + wp.float64(1.0e-14):
        friction_lambda[c] = stick
    else:
        magnitude = wp.sqrt(u1 * u1 + u2 * u2)
        if magnitude > wp.float64(1.0e-14):
            friction_lambda[c] = wp.vec2d(-radius * u1 / magnitude, -radius * u2 / magnitude)
        else:
            friction_lambda[c] = wp.vec2d(wp.float64(0.0))


@wp.kernel(enable_backward=False)
def apply_restitution(
    position: wp.array[wp.vec3d],
    velocity: wp.array[wp.vec3d],
    nodes: wp.array[wp.int32],
    kind: int,
    parameters: wp.array[wp.vec4d],
    multiplier: wp.array[wp.float64],
    restitution: wp.float64,
    dt: wp.float64,
    x: wp.array[wp.vec3d],
):
    c = wp.tid()
    if multiplier[c] <= wp.float64(0.0):
        return
    node = nodes[c]
    point = x[node]
    distance = boundary_gap(point, kind, parameters[0])
    n = boundary_normal(point, kind, parameters[0])
    norm = wp.length(n)
    if norm <= wp.float64(1.0e-12):
        return
    n = n / norm
    if distance > wp.float64(1.0e-8):
        return
    incoming = wp.dot(velocity[node], n)
    if incoming < wp.float64(0.0):
        outgoing = wp.dot((point - position[node]) / dt, n)
        change = -restitution * incoming - outgoing
        if change > wp.float64(0.0):
            x[node] = point + dt * change * n


@wp.kernel(enable_backward=False)
def apply_mesh_restitution(
    position: wp.array[wp.vec3d],
    velocity: wp.array[wp.vec3d],
    nodes: wp.array[wp.int32],
    faces: wp.array[wp.vec3i],
    split: int,
    left_count: int,
    radius: wp.float64,
    surface: wp.array[wp.vec3d],
    normal: wp.array[wp.vec3d],
    multiplier: wp.array[wp.float64],
    restitution: wp.float64,
    dt: wp.float64,
    x: wp.array[wp.vec3d],
):
    c = wp.tid()
    if multiplier[c] <= wp.float64(0.0):
        return
    node = nodes[c]
    point = x[node]
    # The boundary still holds the vertices of the last iteration's start.
    closest, face_normal, _dist2 = mesh_closest(point, node, surface, faces, split, left_count)
    reference = normal[c]
    distance = wp.dot(point - closest, reference) - radius
    n = mesh_direction(point, closest, face_normal)
    if wp.dot(n, reference) < wp.float64(0.0):
        n = -n
    norm = wp.length(n)
    if norm <= wp.float64(1.0e-12):
        return
    n = n / norm
    if distance > wp.float64(MESH_RESTITUTION_SLOP) * radius:
        return
    incoming = wp.dot(velocity[node], n)
    if incoming < wp.float64(0.0):
        outgoing = wp.dot((point - position[node]) / dt, n)
        change = -restitution * incoming - outgoing
        if change > wp.float64(0.0):
            x[node] = point + dt * change * n


def mesh_vertices(parameters) -> np.ndarray:
    """Vertices of both surfaces of ``("mesh", (contact_radius, split_index, faces_left, faces_right))``."""
    faces = [np.asarray(f, dtype=np.int64).reshape(-1) for f in parameters[2:4]]
    return np.unique(np.concatenate(faces)).astype(np.int32)


def surface_vertices(tets: np.ndarray) -> np.ndarray:
    """Vertices of faces that belong to exactly one tetrahedron."""
    faces = np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]])
    faces = np.sort(faces, axis=1)
    unique, counts = np.unique(faces, axis=0, return_counts=True)
    return np.unique(unique[counts == 1]).astype(np.int32)


class FrictionalContact:
    """Device state of one reference frictional boundary and its registered vertices.

    Args:
        kind: ``"planar"`` with parameters ``(nx, ny, nz, offset)`` and gap
            ``(n.x + offset) / |n|``, ``"spherical"`` with ``(cx, cy, cz, radius)``,
            or ``"mesh"`` with ``(contact_radius, split_index, faces_left,
            faces_right)``: radius [m], particle index and two ``(F, 3)``
            triangle arrays of the model's particles.
        parameters: Boundary parameters (see ``kind``).
        nodes: Registered particle indices; each is a contact candidate on every
            step (mesh: subject to the step-start pre-filter).
        particle_count: Number of particles.
        friction: Coulomb coefficient (dimensionless); zero disables friction.
        restitution: Normal restitution coefficient applied after convergence.
        device: Warp device.
    """

    def __init__(
        self,
        kind,
        parameters,
        nodes,
        particle_count,
        friction,
        restitution,
        device,
    ):
        if kind not in BOUNDARIES:
            raise ValueError("frictional boundary must be 'planar', 'spherical' or 'mesh'")
        self.faces = None
        if kind == "mesh":
            if len(parameters) != 4:
                raise ValueError("mesh boundary parameters are (contact_radius, split_index, faces_left, faces_right)")
            radius, split = float(parameters[0]), int(parameters[1])
            if not np.isfinite(radius) or radius < 0:
                raise ValueError("contact_radius must be finite and nonnegative")
            left, right = (np.asarray(f, dtype=np.int32).reshape(-1, 3) for f in parameters[2:4])
            faces = np.concatenate([left, right])
            if len(faces) and (faces.min() < 0 or faces.max() >= particle_count):
                raise ValueError("mesh boundary faces must reference existing particles")
            self.radius = radius
            self.split = split
            self.left_count = len(left)
            self.faces = wp.array(faces, dtype=wp.vec3i, device=device)
            parameters = np.array([radius, 0.0, 0.0, 0.0])
        parameters = np.asarray(parameters, dtype=np.float64)
        if parameters.shape != (4,) or not np.all(np.isfinite(parameters)):
            raise ValueError("frictional boundary parameters must be four finite values")
        if kind == "planar" and np.linalg.norm(parameters[:3]) <= np.finfo(np.float64).eps:
            raise ValueError("planar boundary needs a nonzero normal")
        nodes = np.asarray(nodes, dtype=np.int32)
        if nodes.ndim != 1 or len(np.unique(nodes)) != len(nodes):
            raise ValueError("frictional vertices must be unique particle indices")
        if len(nodes) and (nodes.min() < 0 or nodes.max() >= particle_count):
            raise ValueError("frictional vertices must reference existing particles")
        for name, value in (("friction", friction), ("restitution", restitution)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self.kind = BOUNDARIES[kind]
        self.friction = float(friction)
        self.restitution = float(restitution)
        self.count = len(nodes)
        self.device = device
        self.parameters = wp.array([parameters], dtype=wp.vec4d, device=device)
        self.nodes = wp.array(nodes, dtype=wp.int32, device=device)
        self.particle_count = particle_count
        k = max(self.count, 1)
        self.size = max(k, TILE)
        self.block = wp.zeros((max(self.count, 1), max(self.count, 1)), dtype=wp.float64, device=device)
        self.delassus = wp.zeros((k, k), dtype=wp.float64, device=device)
        self.matrix = wp.zeros((self.size, self.size), dtype=wp.float64, device=device)
        for name in ("normal", "tangent1", "tangent2", "force", "response"):
            setattr(self, name, wp.zeros(k, dtype=wp.vec3d, device=device))
        for name in ("regularization", "multiplier", "gap", "omega", "diagonal", "residual", "rhs"):
            setattr(self, name, wp.zeros(k, dtype=wp.float64, device=device))
        self.compact = wp.zeros(self.size, dtype=wp.float64, device=device)
        self.work = wp.zeros(self.size, dtype=wp.float64, device=device)
        for name in ("order", "slot", "permutation"):
            setattr(self, name, wp.zeros(k, dtype=wp.int32, device=device))
        self.active = wp.zeros(1, dtype=wp.int32, device=device)
        self.friction_lambda = wp.zeros(k, dtype=wp.vec2d, device=device)
        # Mesh candidates are filtered per step; planar and spherical ones always take part.
        self.candidate = wp.full(k, 0 if self.faces is not None else 1, dtype=wp.int32, device=device)
        if self.faces is not None:
            # Vertices of the iterate the boundary last saw (UpdateVertices), for restitution.
            self.surface = wp.zeros(particle_count, dtype=wp.vec3d, device=device)

    def _mesh(self):
        return (self.faces, self.split, self.left_count, self.radius)

    def build(self, inverse) -> None:
        """Form the columns ``A^{-1} e_j`` of the registered vertices and ``B`` from a new factor (setup only).

        The ``N x K`` panel turns each iteration's second solve ``A^{-1} (b + J^T lambda)``
        into ``A^{-1} b`` plus the few columns that carry force.
        """
        if not self.count:
            return
        columns = wp.zeros((inverse.size, self.count), dtype=wp.float64, device=self.device)
        wp.launch(_unit_columns, self.count, inputs=[self.nodes], outputs=[columns], device=self.device)
        self.panel = wp.zeros_like(columns)
        inverse.apply_batch(columns, self.panel)
        wp.launch(
            _gather_block,
            (self.count, self.count),
            inputs=[self.panel, self.nodes],
            outputs=[self.block],
            device=self.device,
        )
        # A factor change starts the multipliers afresh.
        self.multiplier.zero_()

    def begin_step(self, position, dt) -> None:
        """Freeze frames, candidates, Delassus matrix and complementarity scaling at the step start."""
        if not self.count:
            return
        if self.faces is not None:
            wp.launch(
                begin_mesh_step,
                self.count,
                inputs=[position, self.nodes, *self._mesh()],
                outputs=[
                    self.candidate,
                    self.normal,
                    self.tangent1,
                    self.tangent2,
                    self.multiplier,
                    self.friction_lambda,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                begin_contact_step,
                self.count,
                inputs=[position, self.nodes, self.kind, self.parameters],
                outputs=[self.normal, self.tangent1, self.tangent2, self.friction_lambda],
                device=self.device,
            )
        wp.launch(
            contact_delassus,
            (self.count, self.count),
            inputs=[self.block, self.normal, dt],
            outputs=[self.delassus, self.regularization],
            device=self.device,
        )

    def _response(self, force, response):
        wp.launch(
            contact_response,
            self.count,
            inputs=[self.block, force, self.order, self.active],
            outputs=[response],
            device=self.device,
        )

    def _newton(self, x, solved) -> None:
        """The reference's single Fischer-Burmeister Newton step on the multipliers."""
        count, device = self.count, self.device
        if device.is_cuda:
            wp.launch_tiled(
                contact_active_scan,
                dim=[1],
                inputs=[self.omega, count],
                outputs=[self.order, self.slot, self.active],
                block_dim=SCAN_BLOCK,
                device=device,
            )
        else:
            wp.launch(
                contact_active,
                count,
                inputs=[self.omega, count],
                outputs=[self.order, self.slot, self.active],
                device=device,
            )
        wp.launch(
            contact_forces,
            count,
            inputs=[self.normal, self.tangent1, self.tangent2, self.omega, self.multiplier, self.friction_lambda, 1],
            outputs=[self.force],
            device=device,
        )
        self._response(self.force, self.response)
        wp.launch(
            contact_newton_rhs,
            count,
            inputs=[
                solved,
                x,
                self.nodes,
                self.normal,
                self.omega,
                self.residual,
                self.response,
            ],
            outputs=[self.rhs],
            device=device,
        )
        wp.launch(
            contact_system,
            (self.size, self.size),
            inputs=[self.delassus, self.omega, self.diagonal, self.rhs, self.order, self.active],
            outputs=[self.matrix, self.compact],
            device=device,
        )
        wp.launch_tiled(
            spd_tile_solve, dim=[1], inputs=[self.matrix, self.compact, self.active], block_dim=32, device=device
        )
        wp.launch(
            ldlt_solve,
            1,
            inputs=[self.matrix, self.compact, self.active, TILE],
            outputs=[self.permutation, self.work],
            device=device,
        )
        wp.launch(
            contact_update,
            count,
            inputs=[self.rhs, self.diagonal, self.slot, self.compact],
            outputs=[self.multiplier],
            device=device,
        )

    def iterate(self, x, solved, position, out) -> None:
        """One reference Fischer-Burmeister Newton step and friction update.

        ``solved`` is ``A^{-1} b``; writes ``out = A^{-1} (b + J^T lambda)``. Only
        coupled candidates (``omega != 0``) can carry force: ``lambda`` grows only
        where ``omega != 0`` and friction only where ``lambda > 0``.
        """
        count, device = self.count, self.device
        if self.faces is not None:
            if self.restitution > 0:
                wp.copy(self.surface, x)
            tiled = device.is_cuda
            (wp.launch_tiled if tiled else wp.launch)(
                mesh_contact_scalars_tiled if tiled else mesh_contact_scalars,
                **({"dim": [count], "block_dim": 32} if tiled else {"dim": count}),
                inputs=[
                    x,
                    self.nodes,
                    *self._mesh(),
                    self.candidate,
                    self.normal,
                    self.multiplier,
                    self.regularization,
                ],
                outputs=[self.gap, self.omega, self.diagonal, self.residual],
                device=device,
            )
        else:
            wp.launch(
                contact_scalars,
                count,
                inputs=[x, self.nodes, self.kind, self.parameters, self.multiplier, self.regularization],
                outputs=[self.gap, self.omega, self.diagonal, self.residual],
                device=device,
            )
        frames = (self.normal, self.tangent1, self.tangent2)
        self._newton(x, solved)
        if self.friction > 0:
            wp.launch(
                contact_forces,
                count,
                inputs=[*frames, self.omega, self.multiplier, self.friction_lambda, 0],
                outputs=[self.force],
                device=device,
            )
            self._response(self.force, self.response)
            wp.launch(
                contact_friction,
                count,
                inputs=[
                    solved,
                    position,
                    self.nodes,
                    *frames,
                    self.block,
                    self.multiplier,
                    self.response,
                    self.friction,
                ],
                outputs=[self.friction_lambda],
                device=device,
            )
        wp.launch(
            contact_forces,
            count,
            inputs=[*frames, self.omega, self.multiplier, self.friction_lambda, 1],
            outputs=[self.force],
            device=device,
        )
        wp.launch(
            contact_solution,
            self.particle_count,
            inputs=[solved, self.panel, self.force, self.order, self.active],
            outputs=[out],
            device=device,
        )

    def finish_step(self, position, velocity, dt, x) -> None:
        """Apply the reference post-solve restitution correction when enabled."""
        if not self.count or self.restitution <= 0:
            return
        if self.faces is not None:
            wp.launch(
                apply_mesh_restitution,
                self.count,
                inputs=[
                    position,
                    velocity,
                    self.nodes,
                    *self._mesh(),
                    self.surface,
                    self.normal,
                    self.multiplier,
                    self.restitution,
                    dt,
                ],
                outputs=[x],
                device=self.device,
            )
            return
        wp.launch(
            apply_restitution,
            self.count,
            inputs=[position, velocity, self.nodes, self.kind, self.parameters, self.multiplier, self.restitution, dt],
            outputs=[x],
            device=self.device,
        )
