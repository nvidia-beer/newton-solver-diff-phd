# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Continuous contact candidates for two linearly moving triangle meshes.

Both vertex/face directions and edge/edge pairs are tested in double precision.
Swept AABBs reject separated pairs; conservative advancement bounds the speed
of their closest-point distance by the largest relative endpoint displacement.
An exhausted physical sweep is an explicit failure, never a collision-free
result. An exhausted proximity search conservatively retains its last witness;
the independent physical sweep still has to certify the complete trajectory.

``margin`` activates contact rows. ``thickness`` is the smaller physical sweep
threshold: resting at the contact margin therefore does not itself block a
step. This module detects geometry; the coupled solver must solve the rows,
check its residuals, and sweep its corrected positions again before accepting.
Feature selection and time of impact have no automatic derivative.
"""

import numpy as np
import warp as wp

from .mesh_contact import closest_point_triangle

CAPACITY_EXCEEDED = 1
ADVANCEMENT_EXHAUSTED = 2
INVALID_GEOMETRY = 4


@wp.func
def _segment(p: wp.vec3d, a: wp.vec3d, b: wp.vec3d):
    edge = b - a
    length2 = wp.dot(edge, edge)
    alpha = wp.float64(0.0)
    if length2 > wp.float64(0.0):
        alpha = wp.clamp(wp.dot(p - a, edge) / length2, wp.float64(0.0), wp.float64(1.0))
    point = a + alpha * edge
    return point, alpha, wp.length_sq(p - point)


@wp.func
def _triangle(p: wp.vec3d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d):
    # Collinear/zero-area triangles reduce to their three edges. The regular
    # closest-point implementation otherwise uses the triangle's Voronoi regions.
    ab, ac = b - a, c - a
    scale = wp.max(wp.length_sq(ab), wp.max(wp.length_sq(ac), wp.length_sq(c - b)))
    cross = wp.cross(ab, ac)
    if wp.length_sq(cross) > wp.float64(1.0e-28) * scale * scale:
        return closest_point_triangle(p, a, b, c)
    point, alpha, distance2 = _segment(p, a, b)
    weights = wp.vec3d(wp.float64(1.0) - alpha, alpha, wp.float64(0.0))
    other, beta, other2 = _segment(p, b, c)
    if other2 < distance2:
        point, distance2 = other, other2
        weights = wp.vec3d(wp.float64(0.0), wp.float64(1.0) - beta, beta)
    other, beta, other2 = _segment(p, c, a)
    if other2 < distance2:
        point, distance2 = other, other2
        weights = wp.vec3d(beta, wp.float64(0.0), wp.float64(1.0) - beta)
    return point, weights, distance2


@wp.func
def _edges(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, d: wp.vec3d):
    # The minimum is either the unconstrained interior/interior solution or
    # one of the four endpoint/segment solutions. This also covers parallel
    # and zero-length segments without dividing by a small angle estimate.
    pb, beta, best = _segment(a, c, d)
    pa = a
    alpha = wp.float64(0.0)
    other, t, distance2 = _segment(b, c, d)
    if distance2 < best:
        pa, pb, best = b, other, distance2
        alpha, beta = wp.float64(1.0), t
    other, s, distance2 = _segment(c, a, b)
    if distance2 < best:
        pa, pb, best = other, c, distance2
        alpha, beta = s, wp.float64(0.0)
    other, s, distance2 = _segment(d, a, b)
    if distance2 < best:
        pa, pb, best = other, d, distance2
        alpha, beta = s, wp.float64(1.0)
    u, v, r = b - a, d - c, a - c
    cross = wp.cross(u, v)
    denom = wp.length_sq(cross)
    if denom > wp.float64(0.0):
        s = wp.dot(wp.cross(v, r), cross) / denom
        t = wp.dot(wp.cross(u, r), cross) / denom
        if s >= wp.float64(0.0) and s <= wp.float64(1.0) and t >= wp.float64(0.0) and t <= wp.float64(1.0):
            qa, qb = a + s * u, c + t * v
            distance2 = wp.length_sq(qa - qb)
            if distance2 < best:
                pa, pb, best = qa, qb, distance2
                alpha, beta = s, t
    return pa, pb, alpha, beta, best


@wp.func
def _point(start: wp.array[wp.vec3d], end: wp.array[wp.vec3d], node: int, time: wp.float64):
    return start[node] + time * (end[node] - start[node])


@wp.func
def _geometry(start: wp.array[wp.vec3d], end: wp.array[wp.vec3d], ids: wp.vec4i, kind: int, time: wp.float64):
    a = _point(start, end, ids[0], time)
    b = _point(start, end, ids[1], time)
    c = _point(start, end, ids[2], time)
    d = _point(start, end, ids[3], time)
    if kind == 0:
        closest, bary, distance2 = _triangle(a, b, c, d)
        coeff = wp.vec4d(wp.float64(1.0), -bary[0], -bary[1], -bary[2])
        delta = a - closest
        fallback = wp.cross(c - b, d - b)
        return wp.sqrt(wp.max(distance2, wp.float64(0.0))), coeff, delta, fallback
    pa, pb, alpha, beta, distance2 = _edges(a, b, c, d)
    coeff = wp.vec4d(wp.float64(1.0) - alpha, alpha, beta - wp.float64(1.0), -beta)
    return wp.sqrt(wp.max(distance2, wp.float64(0.0))), coeff, pa - pb, wp.cross(b - a, d - c)


@wp.func
def _speed(start: wp.array[wp.vec3d], end: wp.array[wp.vec3d], ids: wp.vec4i, kind: int):
    result = wp.float64(0.0)
    if kind == 0:
        velocity = end[ids[0]] - start[ids[0]]
        for j in range(1, 4):
            result = wp.max(result, wp.length(velocity - (end[ids[j]] - start[ids[j]])))
    else:
        for i in range(2):
            for j in range(2, 4):
                result = wp.max(result, wp.length((end[ids[i]] - start[ids[i]]) - (end[ids[j]] - start[ids[j]])))
    return result


@wp.func
def _bounds_overlap(start: wp.array[wp.vec3d], end: wp.array[wp.vec3d], ids: wp.vec4i, kind: int, radius: wp.float64):
    a_min = wp.min(start[ids[0]], end[ids[0]])
    a_max = wp.max(start[ids[0]], end[ids[0]])
    b_min = wp.min(start[ids[3]], end[ids[3]])
    b_max = wp.max(start[ids[3]], end[ids[3]])
    if kind == 0:
        for j in range(1, 3):
            b_min = wp.min(b_min, wp.min(start[ids[j]], end[ids[j]]))
            b_max = wp.max(b_max, wp.max(start[ids[j]], end[ids[j]]))
    else:
        a_min = wp.min(a_min, wp.min(start[ids[1]], end[ids[1]]))
        a_max = wp.max(a_max, wp.max(start[ids[1]], end[ids[1]]))
        b_min = wp.min(b_min, wp.min(start[ids[2]], end[ids[2]]))
        b_max = wp.max(b_max, wp.max(start[ids[2]], end[ids[2]]))
    for axis in range(3):
        if a_min[axis] > b_max[axis] + radius or b_min[axis] > a_max[axis] + radius:
            return False
    return True


@wp.func
def _separation_bound(
    start: wp.array[wp.vec3d],
    end: wp.array[wp.vec3d],
    ids: wp.vec4i,
    kind: int,
    time: wp.float64,
    normal: wp.vec3d,
):
    # Every primitive point is a convex combination of its vertices. The
    # projection of every relative corner pair is linear in time. Its minimum
    # over both interval endpoints therefore bounds ALL distances from below.
    # This certifies parallel sliding without tiny conservative-advance steps.
    lower = wp.float64(wp.inf)
    if kind == 0:
        point = _point(start, end, ids[0], time)
        for j in range(1, 4):
            lower = wp.min(lower, wp.dot(normal, point - _point(start, end, ids[j], time)))
            lower = wp.min(lower, wp.dot(normal, end[ids[0]] - end[ids[j]]))
    else:
        for i in range(2):
            for j in range(2, 4):
                lower = wp.min(
                    lower, wp.dot(normal, _point(start, end, ids[i], time) - _point(start, end, ids[j], time))
                )
                lower = wp.min(lower, wp.dot(normal, end[ids[i]] - end[ids[j]]))
    return lower


@wp.func
def _advance(
    start: wp.array[wp.vec3d],
    end: wp.array[wp.vec3d],
    ids: wp.vec4i,
    kind: int,
    radius: wp.float64,
    tolerance: wp.float64,
    maximum: int,
):
    time = wp.float64(0.0)
    speed = _speed(start, end, ids, kind)
    coeff = wp.vec4d(wp.float64(0.0))
    normal = wp.vec3d(wp.float64(0.0))
    sampled_time = time
    for _iteration in range(maximum):
        sampled_time = time
        distance, coeff, delta, fallback = _geometry(start, end, ids, kind, time)
        if not wp.isfinite(distance) or not wp.isfinite(speed):
            return time, coeff, normal, int(3)
        if distance > wp.float64(0.0):
            normal = delta / distance
        else:
            length = wp.length(fallback)
            if length > wp.float64(0.0):
                normal = fallback / length
            else:
                return time, coeff, normal, int(3)
        gap = distance - radius
        if gap <= tolerance:
            return time, coeff, normal, int(1)
        if _separation_bound(start, end, ids, kind, time, normal) > radius + tolerance:
            return wp.float64(1.0), coeff, normal, int(0)
        remaining = wp.float64(1.0) - time
        if speed == wp.float64(0.0) or gap > speed * remaining + tolerance:
            return wp.float64(1.0), coeff, normal, int(0)
        step = wp.float64(0.9) * gap / speed
        next_time = wp.min(wp.float64(1.0), time + step)
        if next_time <= time:
            return time, coeff, normal, int(2)
        time = next_time
    return sampled_time, coeff, normal, int(2)


@wp.func
def _same_ids(a: wp.vec4i, b: wp.vec4i):
    return a[0] == b[0] and a[1] == b[1] and a[2] == b[2] and a[3] == b[3]


@wp.func
def _pair(
    start: wp.array[wp.vec3d],
    end: wp.array[wp.vec3d],
    ids: wp.vec4i,
    kind: int,
    margin: wp.float64,
    thickness: wp.float64,
    tolerance: wp.float64,
    maximum: int,
    previous_count: wp.array[int],
    indices: wp.array[wp.vec4i],
    coeffs: wp.array[wp.vec4d],
    normals: wp.array[wp.vec3d],
    tois: wp.array[wp.float64],
    count: wp.array[int],
    flags: wp.array[int],
    hit_count: wp.array[int],
    minimum_toi: wp.array[wp.float64],
):
    # Float32 state storage can leave a resting point a few nanometres beyond
    # the margin. Conservatively INCLUDE that activation band. This does not
    # relax the smaller physical sweep threshold or the response's gap test.
    if not _bounds_overlap(start, end, ids, kind, margin + thickness):
        return
    toi, coeff, normal, result = _advance(start, end, ids, kind, margin, thickness, maximum)
    if result == 0:
        return
    # A proximity search can approach the margin tangentially without finding
    # its first activation time within the budget. Keep its last finite witness
    # conservatively. The endpoint must satisfy that row, and the independent
    # smaller-threshold query below must still certify the entire sweep. Thus
    # this is candidate enrichment, not an exhausted physical query declared
    # collision-free.
    if result == 3:
        wp.atomic_or(flags, 0, INVALID_GEOMETRY)
        return
    slot = int(-1)
    for i in range(wp.min(previous_count[0], indices.shape[0])):
        if _same_ids(indices[i], ids):
            slot = i
            break
    if slot < 0:
        slot = wp.atomic_add(count, 0, 1)
        if slot < indices.shape[0]:
            indices[slot] = ids
            coeffs[slot] = coeff
            normals[slot] = normal
            tois[slot] = toi
        else:
            wp.atomic_or(flags, 0, CAPACITY_EXCEEDED)
    # A contact at the activation margin may already persist harmlessly.
    # A separate smaller threshold certifies the actual proposed trajectory.
    physical_toi, _coeff, _normal, physical = _advance(start, end, ids, kind, thickness, tolerance, maximum)
    if physical != 0:
        wp.atomic_add(hit_count, 0, 1)
        wp.atomic_min(minimum_toi, 0, physical_toi)
        if physical == 2:
            wp.atomic_or(flags, 0, ADVANCEMENT_EXHAUSTED)
        if physical == 3:
            wp.atomic_or(flags, 0, INVALID_GEOMETRY)


@wp.kernel(enable_backward=False)
def _reset(hit_count: wp.array[int], minimum_toi: wp.array[wp.float64]):
    hit_count[0] = 0
    minimum_toi[0] = wp.float64(1.0)


@wp.kernel(enable_backward=False)
def _cap(count: wp.array[int], capacity: int):
    count[0] = wp.min(count[0], capacity)


@wp.kernel(enable_backward=False)
def _validate_positions(start: wp.array[wp.vec3d], end: wp.array[wp.vec3d], flags: wp.array[int]):
    i = wp.tid()
    for axis in range(3):
        if not wp.isfinite(start[i][axis]) or not wp.isfinite(end[i][axis]):
            wp.atomic_or(flags, 0, INVALID_GEOMETRY)


@wp.kernel(enable_backward=False)
def _vertex_faces(
    start: wp.array[wp.vec3d],
    end: wp.array[wp.vec3d],
    vertices: wp.array[int],
    faces: wp.array[wp.vec3i],
    margin: wp.float64,
    thickness: wp.float64,
    tolerance: wp.float64,
    maximum: int,
    previous_count: wp.array[int],
    indices: wp.array[wp.vec4i],
    coeffs: wp.array[wp.vec4d],
    normals: wp.array[wp.vec3d],
    tois: wp.array[wp.float64],
    count: wp.array[int],
    flags: wp.array[int],
    hit_count: wp.array[int],
    minimum_toi: wp.array[wp.float64],
):
    i, j = wp.tid()
    face = faces[j]
    ids = wp.vec4i(vertices[i], face[0], face[1], face[2])
    _pair(
        start,
        end,
        ids,
        0,
        margin,
        thickness,
        tolerance,
        maximum,
        previous_count,
        indices,
        coeffs,
        normals,
        tois,
        count,
        flags,
        hit_count,
        minimum_toi,
    )


@wp.kernel(enable_backward=False)
def _edge_edges(
    start: wp.array[wp.vec3d],
    end: wp.array[wp.vec3d],
    left: wp.array[wp.vec2i],
    right: wp.array[wp.vec2i],
    margin: wp.float64,
    thickness: wp.float64,
    tolerance: wp.float64,
    maximum: int,
    previous_count: wp.array[int],
    indices: wp.array[wp.vec4i],
    coeffs: wp.array[wp.vec4d],
    normals: wp.array[wp.vec3d],
    tois: wp.array[wp.float64],
    count: wp.array[int],
    flags: wp.array[int],
    hit_count: wp.array[int],
    minimum_toi: wp.array[wp.float64],
):
    i, j = wp.tid()
    ids = wp.vec4i(left[i][0], left[i][1], right[j][0], right[j][1])
    _pair(
        start,
        end,
        ids,
        1,
        margin,
        thickness,
        tolerance,
        maximum,
        previous_count,
        indices,
        coeffs,
        normals,
        tois,
        count,
        flags,
        hit_count,
        minimum_toi,
    )


class SweptMeshCCD:
    """Append swept contacts between two disjoint triangle meshes.

    ``start`` and ``end`` are full-particle ``wp.vec3d`` arrays on ``device``.
    Output ``indices``, ``coeffs``, ``normals`` and ``tois`` have one common
    fixed capacity and respective dtypes ``vec4i``, ``vec4d``, ``vec3d`` and
    ``float64``. Each row has gap ``normal · sum(coeff[k]*q[index[k]]) - margin``.
    Callers reset ``count``/``flags`` at each physical step. Repeated calls
    retain previously appended features. Identical ordered feature IDs are
    deduplicated. Their existing witness is retained; a coupled response may
    refresh its closest-point geometry after certifying the corrected sweep.
    ``tois`` records the sampled witness time. A proximity search that exhausts
    its budget contributes a conservative witness instead of claiming an exact
    first activation time; its independent physical sweep must still certify.

    ``hit_count`` and ``minimum_toi`` refer only to the current append's sweep
    at ``thickness`` and are reset internally. Any nonzero flag prevents a
    collision-free claim, including capacity exhaustion. Arrays and kernel
    counts stay fixed, so the operation supports CUDA graph capture.
    """

    def __init__(
        self, faces_left, faces_right, particle_count, *, margin, thickness=1.0e-8, max_iterations=128, device=None
    ):
        self.device = wp.get_device(device)
        self.particle_count = int(particle_count)
        self.margin, self.thickness = float(margin), float(thickness)
        if not np.isfinite((margin, thickness)).all() or not 0.0 < thickness < margin:
            raise ValueError("CCD requires finite 0 < thickness < margin")
        if isinstance(max_iterations, bool) or int(max_iterations) != max_iterations or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        self.max_iterations = int(max_iterations)
        self.tolerance = min(self.thickness * 0.125, 1.0e-10)
        left, right = (np.asarray(f, dtype=np.int32) for f in (faces_left, faces_right))
        for faces in (left, right):
            if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
                raise ValueError("CCD requires nonempty (F,3) faces for both bodies")
            if faces.min() < 0 or faces.max() >= particle_count:
                raise ValueError("CCD face indices must reference existing particles")
            if np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0):
                raise ValueError("CCD faces must have three distinct vertex indices")
            if len(np.unique(np.sort(faces, axis=1), axis=0)) != len(faces):
                raise ValueError("CCD faces must be unique")
        if np.intersect1d(left, right).size:
            raise ValueError("CCD body surfaces must use disjoint particle indices")
        self.faces_left = wp.array(left, dtype=wp.vec3i, device=self.device)
        self.faces_right = wp.array(right, dtype=wp.vec3i, device=self.device)
        self.vertices_left = wp.array(np.unique(left), dtype=int, device=self.device)
        self.vertices_right = wp.array(np.unique(right), dtype=int, device=self.device)
        for name, faces in (("edges_left", left), ("edges_right", right)):
            edges = faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2)
            edges = np.unique(np.sort(edges, axis=1), axis=0)
            setattr(self, name, wp.array(edges, dtype=wp.vec2i, device=self.device))
        self._previous_count = wp.zeros(1, dtype=int, device=self.device)
        self.hit_count = wp.zeros(1, dtype=int, device=self.device)
        self.minimum_toi = wp.ones(1, dtype=wp.float64, device=self.device)

    def append(self, start, end, indices, coeffs, normals, tois, count, flags):
        capacity = len(indices)
        for array, dtype, shape in (
            (start, wp.vec3d, (self.particle_count,)),
            (end, wp.vec3d, (self.particle_count,)),
            (indices, wp.vec4i, (capacity,)),
            (coeffs, wp.vec4d, (capacity,)),
            (normals, wp.vec3d, (capacity,)),
            (tois, wp.float64, (capacity,)),
            (count, wp.int32, (1,)),
            (flags, wp.int32, (1,)),
        ):
            if array.device != self.device or array.dtype != dtype or array.shape != shape:
                raise ValueError("CCD input/output array has an incompatible device, dtype or shape")
        if capacity < 1:
            raise ValueError("CCD output capacity must be positive")
        wp.copy(self._previous_count, count)
        wp.launch(_reset, 1, inputs=[self.hit_count, self.minimum_toi], device=self.device)
        wp.launch(_validate_positions, self.particle_count, inputs=[start, end, flags], device=self.device)
        shared = [
            self.margin,
            self.thickness,
            self.tolerance,
            self.max_iterations,
            self._previous_count,
            indices,
            coeffs,
            normals,
            tois,
            count,
            flags,
            self.hit_count,
            self.minimum_toi,
        ]
        for vertices, faces in ((self.vertices_left, self.faces_right), (self.vertices_right, self.faces_left)):
            wp.launch(
                _vertex_faces,
                (len(vertices), len(faces)),
                inputs=[start, end, vertices, faces, *shared],
                device=self.device,
            )
        wp.launch(
            _edge_edges,
            (len(self.edges_left), len(self.edges_right)),
            inputs=[start, end, self.edges_left, self.edges_right, *shared],
            device=self.device,
        )
        wp.launch(_cap, 1, inputs=[count, capacity], device=self.device)
