# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frozen vertex-face and edge-edge contact rows for tetrahedral solids.

Broad phase uses swept BVHs; narrow phase conservatively advances the Euclidean
feature distance along linear particle trajectories. This detects crossings
between the supplied endpoints, including edge crossings without a nearby
vertex. It does not certify the curved trajectory of a nonlinear solver: callers
must regenerate contacts for every proposed update. Contact normals and feature
coordinates are frozen for the subsequent contact solve and its adjoint.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp
from warp.utils import array_scan, radix_sort_pairs

from .exceptions import ConvergenceError


@wp.func
def _triangle_barycentric(p: wp.vec3d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d):
    """Closest point coordinates, including vertex and edge Voronoi regions."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = wp.dot(ab, ap), wp.dot(ac, ap)
    result = wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0))
    if d1 <= wp.float64(0.0) and d2 <= wp.float64(0.0):
        return result
    bp = p - b
    d3, d4 = wp.dot(ab, bp), wp.dot(ac, bp)
    if d3 >= wp.float64(0.0) and d4 <= d3:
        return wp.vec3d(wp.float64(0.0), wp.float64(1.0), wp.float64(0.0))
    vc = d1 * d4 - d3 * d2
    if vc <= wp.float64(0.0) and d1 >= wp.float64(0.0) and d3 <= wp.float64(0.0):
        v = d1 / (d1 - d3)
        return wp.vec3d(wp.float64(1.0) - v, v, wp.float64(0.0))
    cp = p - c
    d5, d6 = wp.dot(ab, cp), wp.dot(ac, cp)
    if d6 >= wp.float64(0.0) and d5 <= d6:
        return wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
    vb = d5 * d2 - d1 * d6
    if vb <= wp.float64(0.0) and d2 >= wp.float64(0.0) and d6 <= wp.float64(0.0):
        w = d2 / (d2 - d6)
        return wp.vec3d(wp.float64(1.0) - w, wp.float64(0.0), w)
    va = d3 * d6 - d5 * d4
    if va <= wp.float64(0.0) and (d4 - d3) >= wp.float64(0.0) and (d5 - d6) >= wp.float64(0.0):
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return wp.vec3d(wp.float64(0.0), wp.float64(1.0) - w, w)
    denom = va + vb + vc
    if wp.abs(denom) > wp.float64(1.0e-60):
        v, w = vb / denom, vc / denom
        result = wp.vec3d(wp.float64(1.0) - v - w, v, w)
    return result


_vec7d = wp.types.vector(length=7, dtype=wp.float64)


@wp.func
def _norm1(v: wp.vec3d):
    return wp.abs(v[0]) + wp.abs(v[1]) + wp.abs(v[2])


@wp.func
def _separation_polynomial_clear(
    r0: wp.vec3d,
    e0: wp.vec3d,
    f0: wp.vec3d,
    r_end: wp.vec3d,
    e_end: wp.vec3d,
    f_end: wp.vec3d,
    clearance: wp.float64,
):
    """Certify |r.(e x f)| / |e x f| > clearance along affine relative vectors.

    N(t) = e(t) x f(t) is quadratic and V(t) = r(t).N(t) is cubic.
    If every degree-six Bernstein coefficient of V^2-clearance^2*|N|^2
    exceeds its roundoff margin, separation holds throughout [0, 1].
    Inconclusive arithmetic returns False, leaving conservative advancement intact.
    """
    r1, e1, f1 = r_end - r0, e_end - e0, f_end - f0
    n0 = wp.cross(e0, f0)
    n1 = wp.cross(e1, f0) + wp.cross(e0, f1)
    n2 = wp.cross(e1, f1)
    volume = wp.vec4d(wp.dot(r0, n0), wp.dot(r0, n1) + wp.dot(r1, n0), wp.dot(r0, n2) + wp.dot(r1, n1), wp.dot(r1, n2))
    power = _vec7d(wp.float64(0.0))
    for i in range(4):
        for j in range(4):
            power[i + j] += volume[i] * volume[j]
    clearance2 = clearance * clearance
    power[0] -= clearance2 * wp.dot(n0, n0)
    power[1] -= clearance2 * wp.float64(2.0) * wp.dot(n0, n1)
    power[2] -= clearance2 * (wp.dot(n1, n1) + wp.float64(2.0) * wp.dot(n0, n2))
    power[3] -= clearance2 * wp.float64(2.0) * wp.dot(n1, n2)
    power[4] -= clearance2 * wp.dot(n2, n2)
    # Coefficient one-norms satisfy |N| <= E*F and |V| <= R*E*F.
    # Bernstein conversion weights are <= 1. A conservative float64 margin
    # covers endpoint subtraction and this fixed-degree arithmetic circuit;
    # underflow/overflow makes the certificate inconclusive.
    r_bound, e_bound, f_bound = _norm1(r0) + _norm1(r1), _norm1(e0) + _norm1(e1), _norm1(f0) + _norm1(f1)
    scale = (r_bound * r_bound + clearance2) * e_bound * e_bound * f_bound * f_bound
    error = wp.float64(4096.0 * 2.220446049250313e-16) * scale
    if not wp.isfinite(scale) or scale <= wp.float64(0.0) or error < wp.float64(4096.0 * 2.2250738585072014e-308):
        return False
    for k in range(7):
        coefficient = wp.float64(0.0)
        ratio = wp.float64(1.0)
        for j in range(7):
            if j <= k:
                if j > 0:
                    ratio *= wp.float64(k - j + 1) / wp.float64(7 - j)
                coefficient += power[j] * ratio
        if not wp.isfinite(coefficient) or coefficient <= error:
            return False
    return True


@wp.func
def _vf_plane_polynomial_clear(
    p0: wp.vec3d,
    a0: wp.vec3d,
    b0: wp.vec3d,
    c0: wp.vec3d,
    p1: wp.vec3d,
    a1: wp.vec3d,
    b1: wp.vec3d,
    c1: wp.vec3d,
    clearance: wp.float64,
):
    """Triangle distance is bounded below by distance to its moving plane."""
    return _separation_polynomial_clear(p0 - a0, b0 - a0, c0 - a0, p1 - a1, b1 - a1, c1 - a1, clearance)


@wp.func
def _ee_line_polynomial_clear(
    a0: wp.vec3d,
    b0: wp.vec3d,
    c0: wp.vec3d,
    d0: wp.vec3d,
    a1: wp.vec3d,
    b1: wp.vec3d,
    c1: wp.vec3d,
    d1: wp.vec3d,
    clearance: wp.float64,
):
    """Bound segment distance by distance between their supporting lines.

    For any segment coordinates s,t, (c+t*(d-c)-a-s*(b-a)).N = (c-a).N.
    Parallel or degenerate directions make the certificate inconclusive.
    """
    return _separation_polynomial_clear(c0 - a0, b0 - a0, d0 - c0, c1 - a1, b1 - a1, d1 - c1, clearance)


@wp.func
def _edge_coordinates(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, d: wp.vec3d):
    u, v, r = b - a, d - c, a - c
    uu, vv, uv = wp.dot(u, u), wp.dot(v, v), wp.dot(u, v)
    ur, vr = wp.dot(u, r), wp.dot(v, r)
    s, t = wp.float64(0.0), wp.float64(0.0)
    eps = wp.float64(1.0e-30)
    if uu <= eps:
        if vv > eps:
            t = wp.clamp(vr / vv, wp.float64(0.0), wp.float64(1.0))
    elif vv <= eps:
        s = wp.clamp(-ur / uu, wp.float64(0.0), wp.float64(1.0))
    else:
        denominator = uu * vv - uv * uv
        if denominator > wp.float64(1.0e-14) * uu * vv:
            s = wp.clamp((uv * vr - ur * vv) / denominator, wp.float64(0.0), wp.float64(1.0))
        t = (uv * s + vr) / vv
        if t < wp.float64(0.0):
            t = wp.float64(0.0)
            s = wp.clamp(-ur / uu, wp.float64(0.0), wp.float64(1.0))
        elif t > wp.float64(1.0):
            t = wp.float64(1.0)
            s = wp.clamp((uv - ur) / uu, wp.float64(0.0), wp.float64(1.0))
    return wp.vec2d(s, t)


@wp.func
def _neighbor(a: int, b: int, starts: wp.array[int], neighbors: wp.array[int]):
    if a == b:
        return True
    for j in range(starts[a], starts[a + 1]):
        other = neighbors[j]
        if other == b:
            return True
        # Each rest one-ring is sorted when the surface topology is built.
        if other > b:
            break
    return False


@wp.func
def _worlds_match(a: int, b: int):
    return a == b or a < 0 or b < 0


@wp.func
def _retained_feature(
    kind: int,
    ids: wp.vec4i,
    count: int,
    kinds: wp.array[int],
    indices: wp.array[wp.vec4i],
):
    """Find an exact feature in the canonical retained-row ordering."""
    lower = int(0)
    upper = int(count)
    while lower < upper:
        middle = (lower + upper) // 2
        other_kind, other = kinds[middle], indices[middle]
        less = other_kind < kind
        if other_kind == kind:
            for j in range(4):
                if other[j] != ids[j]:
                    less = other[j] < ids[j]
                    break
        if less:
            lower = middle + 1
        else:
            upper = middle
    found = bool(False)
    if lower < count:
        matched = indices[lower]
        found = kinds[lower] == kind
        for j in range(4):
            found = found and matched[j] == ids[j]
    return found


@wp.func
def _write_row(
    ids: wp.vec4i,
    w: wp.vec4d,
    n: wp.vec3d,
    offset: wp.float64,
    kind: int,
    toi: wp.float64,
    capacity: int,
    count: wp.array[int],
    indices: wp.array[wp.vec4i],
    weights: wp.array[wp.vec4d],
    normals: wp.array[wp.vec3d],
    offsets: wp.array[wp.float64],
    kinds: wp.array[int],
    times: wp.array[wp.float64],
):
    row = wp.atomic_add(count, 0, 1)
    if row < capacity:
        indices[row] = ids
        weights[row] = w
        normals[row] = n
        offsets[row] = offset
        kinds[row] = kind
        times[row] = toi


@wp.kernel(enable_backward=False)
def _face_bounds(
    q: wp.array[wp.vec3d],
    previous: wp.array[wp.vec3d],
    faces: wp.array[wp.vec3i],
    padding: wp.float64,
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
):
    i = wp.tid()
    f = faces[i]
    lo = wp.vec3d(q[f[0]])
    hi = wp.vec3d(q[f[0]])
    for j in range(3):
        lo = wp.min(lo, wp.min(q[f[j]], previous[f[j]]))
        hi = wp.max(hi, wp.max(q[f[j]], previous[f[j]]))
    # BVHs store float32 bounds. Expand before casting to cover roundoff.
    scale = wp.max(wp.length(lo), wp.length(hi))
    pad = padding + wp.float64(2.0e-7) * scale + wp.float64(1.0e-12)
    lower[i] = wp.vec3(lo - wp.vec3d(pad))
    upper[i] = wp.vec3(hi + wp.vec3d(pad))


@wp.kernel(enable_backward=False)
def _edge_bounds(
    q: wp.array[wp.vec3d],
    previous: wp.array[wp.vec3d],
    edges: wp.array[wp.vec2i],
    padding: wp.float64,
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
):
    i = wp.tid()
    e = edges[i]
    lo = wp.min(wp.min(q[e[0]], q[e[1]]), wp.min(previous[e[0]], previous[e[1]]))
    hi = wp.max(wp.max(q[e[0]], q[e[1]]), wp.max(previous[e[0]], previous[e[1]]))
    scale = wp.max(wp.length(lo), wp.length(hi))
    pad = padding + wp.float64(2.0e-7) * scale + wp.float64(1.0e-12)
    lower[i] = wp.vec3(lo - wp.vec3d(pad))
    upper[i] = wp.vec3(hi + wp.vec3d(pad))


@wp.kernel(enable_backward=False)
def _vertex_face_contacts(
    q: wp.array[wp.vec3d],
    previous: wp.array[wp.vec3d],
    vertices: wp.array[int],
    faces: wp.array[wp.vec3i],
    worlds: wp.array[int],
    components: wp.array[int],
    starts: wp.array[int],
    neighbors: wp.array[int],
    bvh: wp.uint64,
    retained_count: int,
    retained_indices: wp.array[wp.vec4i],
    retained_kinds: wp.array[int],
    self_contact: bool,
    soft_contact: bool,
    thickness: wp.float64,
    tolerance: wp.float64,
    max_iterations: int,
    capacity: int,
    count: wp.array[int],
    failed: wp.array[int],
    indices: wp.array[wp.vec4i],
    weights: wp.array[wp.vec4d],
    normals: wp.array[wp.vec3d],
    offsets: wp.array[wp.float64],
    kinds: wp.array[int],
    times: wp.array[wp.float64],
):
    vertex = vertices[wp.tid()]
    lo, hi = wp.min(q[vertex], previous[vertex]), wp.max(q[vertex], previous[vertex])
    pad = wp.float64(0.5) * (thickness + tolerance) + wp.float64(2.0e-7) * wp.max(wp.length(lo), wp.length(hi))
    query = wp.bvh_query_aabb(bvh, wp.vec3(lo - wp.vec3d(pad)), wp.vec3(hi + wp.vec3d(pad)))
    face_index = int(0)
    while wp.bvh_query_next(query, face_index):
        f = faces[face_index]
        same_component = components[vertex] == components[f[0]]
        excluded = not _worlds_match(worlds[vertex], worlds[f[0]])
        excluded = excluded or (same_component and not self_contact) or (not same_component and not soft_contact)
        if same_component:
            for j in range(3):
                excluded = excluded or _neighbor(vertex, f[j], starts, neighbors)
        if excluded:
            continue
        if _retained_feature(1, wp.vec4i(vertex, f[0], f[1], f[2]), retained_count, retained_kinds, retained_indices):
            continue
        p0, a0, b0, c0 = previous[vertex], previous[f[0]], previous[f[1]], previous[f[2]]
        dp, da, db, dc = q[vertex] - p0, q[f[0]] - a0, q[f[1]] - b0, q[f[2]] - c0
        # Relative velocities remove rigid translation from the Lipschitz bound.
        speed = wp.sqrt(wp.max(wp.length_sq(dp - da), wp.max(wp.length_sq(dp - db), wp.length_sq(dp - dc))))
        time = wp.float64(0.0)
        done, hit = bool(False), bool(False)
        bary, normal = wp.vec3d(wp.float64(0.0)), wp.vec3d(wp.float64(0.0))
        for _iteration in range(max_iterations):
            p, a, b, c = p0 + time * dp, a0 + time * da, b0 + time * db, c0 + time * dc
            cross = wp.cross(b - a, c - a)
            area = wp.length(cross)
            if area < wp.float64(1.0e-24):
                wp.atomic_add(failed, 0, 1)
                done = True
                break
            bary = _triangle_barycentric(p, a, b, c)
            separation = p - bary[0] * a - bary[1] * b - bary[2] * c
            distance = wp.length(separation)
            if distance <= thickness + tolerance:
                normal = cross / area
                # A point outside a solid can be behind an incident side-face
                # plane. The closest-feature direction preserves the swept
                # entry side and keeps duplicate vertex/edge rows compatible.
                if distance > wp.float64(1.0e-30):
                    normal = separation / distance
                hit, done = True, True
                break
            if speed <= wp.float64(1.0e-30) or time >= wp.float64(1.0):
                done = True
                break
            # Only slow pairs pay for this bounded rotating-plane certificate.
            # Original endpoints avoid extra world-coordinate interpolation.
            if _iteration == 8 and _vf_plane_polynomial_clear(
                p0, a0, b0, c0, q[vertex], q[f[0]], q[f[1]], q[f[2]], thickness + tolerance
            ):
                done = True
                break
            increment = wp.float64(0.9) * (distance - thickness) / speed
            # The closest-point plane separates every triangle vertex from
            # the point. Its projected gaps are affine along this sweep, so
            # their first possible closing time is another safe lower bound.
            # This avoids tiny distance/speed advances for tangential motion
            # near an already supported contact.
            separating_normal = separation / distance
            plane_increment = wp.float64(1.0) - time
            for j in range(3):
                position = a
                displacement = da
                if j == 1:
                    position, displacement = b, db
                elif j == 2:
                    position, displacement = c, dc
                closing = -wp.dot(separating_normal, dp - displacement)
                if closing > wp.float64(1.0e-30):
                    projected_gap = wp.dot(separating_normal, p - position) - thickness
                    plane_increment = wp.min(plane_increment, wp.max(wp.float64(0.0), projected_gap / closing))
            increment = wp.max(increment, wp.float64(0.9) * plane_increment)
            if time + increment > wp.float64(1.0):
                time = wp.float64(1.0)
            else:
                time += increment
        if not done:
            wp.atomic_add(failed, 0, 1)
        if hit:
            ids = wp.vec4i(vertex, f[0], f[1], f[2])
            w = wp.vec4d(wp.float64(1.0), -bary[0], -bary[1], -bary[2])
            _write_row(
                ids,
                w,
                normal,
                wp.float64(0.0),
                1,
                time,
                capacity,
                count,
                indices,
                weights,
                normals,
                offsets,
                kinds,
                times,
            )


@wp.kernel(enable_backward=False)
def _edge_edge_contacts(
    q: wp.array[wp.vec3d],
    previous: wp.array[wp.vec3d],
    edges: wp.array[wp.vec2i],
    worlds: wp.array[int],
    components: wp.array[int],
    starts: wp.array[int],
    neighbors: wp.array[int],
    bvh: wp.uint64,
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    retained_count: int,
    retained_indices: wp.array[wp.vec4i],
    retained_kinds: wp.array[int],
    self_contact: bool,
    soft_contact: bool,
    thickness: wp.float64,
    tolerance: wp.float64,
    max_iterations: int,
    capacity: int,
    count: wp.array[int],
    failed: wp.array[int],
    indices: wp.array[wp.vec4i],
    weights: wp.array[wp.vec4d],
    normals: wp.array[wp.vec3d],
    offsets: wp.array[wp.float64],
    kinds: wp.array[int],
    times: wp.array[wp.float64],
):
    ei = wp.tid()
    e = edges[ei]
    query = wp.bvh_query_aabb(bvh, lower[ei], upper[ei])
    other = int(0)
    while wp.bvh_query_next(query, other):
        if other <= ei:
            continue
        f = edges[other]
        same_component = components[e[0]] == components[f[0]]
        excluded = not _worlds_match(worlds[e[0]], worlds[f[0]])
        excluded = excluded or (same_component and not self_contact) or (not same_component and not soft_contact)
        if same_component:
            for j in range(2):
                for k in range(2):
                    excluded = excluded or _neighbor(e[j], f[k], starts, neighbors)
        if excluded:
            continue
        if _retained_feature(2, wp.vec4i(e[0], e[1], f[0], f[1]), retained_count, retained_kinds, retained_indices):
            continue
        a0, b0, c0, d0 = previous[e[0]], previous[e[1]], previous[f[0]], previous[f[1]]
        da, db, dc, dd = q[e[0]] - a0, q[e[1]] - b0, q[f[0]] - c0, q[f[1]] - d0
        speed = wp.sqrt(
            wp.max(
                wp.max(wp.length_sq(da - dc), wp.length_sq(da - dd)),
                wp.max(wp.length_sq(db - dc), wp.length_sq(db - dd)),
            )
        )
        time = wp.float64(0.0)
        done, hit = bool(False), bool(False)
        coords = wp.vec2d(wp.float64(0.0))
        normal = wp.vec3d(wp.float64(0.0))
        for _iteration in range(max_iterations):
            a, b, c, d = a0 + time * da, b0 + time * db, c0 + time * dc, d0 + time * dd
            coords = _edge_coordinates(a, b, c, d)
            s, t = coords[0], coords[1]
            separation = (wp.float64(1.0) - s) * a + s * b - (wp.float64(1.0) - t) * c - t * d
            distance = wp.length(separation)
            if distance <= thickness + tolerance:
                if distance > tolerance * wp.float64(0.01):
                    normal = separation / distance
                else:
                    normal = wp.cross(b - a, d - c)
                    normal_length = wp.length(normal)
                    if normal_length > wp.float64(1.0e-24):
                        normal /= normal_length
                        old_separation = (wp.float64(1.0) - s) * a0 + s * b0 - (wp.float64(1.0) - t) * c0 - t * d0
                        if wp.dot(normal, old_separation) < wp.float64(0.0):
                            normal = -normal
                    else:
                        # Coincident parallel segments have no unique normal.
                        wp.atomic_add(failed, 0, 1)
                        done = True
                        break
                hit, done = True, True
                break
            if speed <= wp.float64(1.0e-30) or time >= wp.float64(1.0):
                done = True
                break
            # Reuse the rotating-plane proof for the two supporting lines.
            if _iteration == 8 and _ee_line_polynomial_clear(
                a0, b0, c0, d0, q[e[0]], q[e[1]], q[f[0]], q[f[1]], thickness + tolerance
            ):
                done = True
                break
            increment = wp.float64(0.9) * (distance - thickness) / speed
            # A closest-feature separating plane supports both segments.
            # Check all four endpoint differences so this time bound remains
            # valid even if either closest coordinate changes during motion.
            separating_normal = separation / distance
            plane_increment = wp.float64(1.0) - time
            for j in range(2):
                pa, va = a, da
                if j == 1:
                    pa, va = b, db
                for k in range(2):
                    pb, vb = c, dc
                    if k == 1:
                        pb, vb = d, dd
                    closing = -wp.dot(separating_normal, va - vb)
                    if closing > wp.float64(1.0e-30):
                        projected_gap = wp.dot(separating_normal, pa - pb) - thickness
                        plane_increment = wp.min(plane_increment, wp.max(wp.float64(0.0), projected_gap / closing))
            increment = wp.max(increment, wp.float64(0.9) * plane_increment)
            if time + increment > wp.float64(1.0):
                time = wp.float64(1.0)
            else:
                time += increment
        if not done:
            wp.atomic_add(failed, 0, 1)
        if hit:
            s, t = coords[0], coords[1]
            ids = wp.vec4i(e[0], e[1], f[0], f[1])
            w = wp.vec4d(wp.float64(1.0) - s, s, t - wp.float64(1.0), -t)
            _write_row(
                ids,
                w,
                normal,
                wp.float64(0.0),
                2,
                time,
                capacity,
                count,
                indices,
                weights,
                normals,
                offsets,
                kinds,
                times,
            )


@wp.kernel(enable_backward=False)
def _ground_contacts(
    q: wp.array[wp.vec3d],
    previous: wp.array[wp.vec3d],
    vertices: wp.array[int],
    retained_count: int,
    retained_indices: wp.array[wp.vec4i],
    retained_kinds: wp.array[int],
    normal: wp.vec3d,
    offset: wp.float64,
    thickness: wp.float64,
    capacity: int,
    count: wp.array[int],
    indices: wp.array[wp.vec4i],
    weights: wp.array[wp.vec4d],
    normals: wp.array[wp.vec3d],
    offsets: wp.array[wp.float64],
    kinds: wp.array[int],
    times: wp.array[wp.float64],
):
    vertex = vertices[wp.tid()]
    if _retained_feature(0, wp.vec4i(vertex, -1, -1, -1), retained_count, retained_kinds, retained_indices):
        return
    old_gap = wp.dot(normal, previous[vertex]) - offset - thickness
    new_gap = wp.dot(normal, q[vertex]) - offset - thickness
    if wp.min(old_gap, new_gap) <= wp.float64(0.0):
        toi = wp.float64(0.0)
        if old_gap > wp.float64(0.0):
            toi = old_gap / (old_gap - new_gap)
        _write_row(
            wp.vec4i(vertex, -1, -1, -1),
            wp.vec4d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)),
            normal,
            offset,
            0,
            toi,
            capacity,
            count,
            indices,
            weights,
            normals,
            offsets,
            kinds,
            times,
        )


@dataclass(frozen=True)
class ContactRows:
    """Contact rows valid until the next :meth:`ContactGeometry.generate` call.

    Only the first ``count`` entries are valid. Each signed gap [m] is
    ``sum(weights[j] * dot(normals, q[indices[j]])) - offsets - thickness``.
    Indices equal to -1 have zero weight and must be skipped. ``kinds`` uses
    0=ground, 1=vertex-face, 2=edge-edge. ``times`` stores the first detected
    proximity time as a fraction of the supplied trajectory. Consumers retaining
    rows for reverse-mode differentiation must copy the buffers.
    """

    count: int
    indices: wp.array[wp.vec4i]
    weights: wp.array[wp.vec4d]
    normals: wp.array[wp.vec3d]
    offsets: wp.array[wp.float64]
    kinds: wp.array[int]
    times: wp.array[wp.float64]
    thickness: float


@wp.kernel(enable_backward=False)
def _initialize_feature_order(
    previous_count: int,
    previous_indices: wp.array[wp.vec4i],
    previous_kinds: wp.array[int],
    generated_indices: wp.array[wp.vec4i],
    generated_kinds: wp.array[int],
    indices: wp.array[wp.vec4i],
    kinds: wp.array[int],
    keys: wp.array[int],
    order: wp.array[int],
):
    i = wp.tid()
    row = wp.vec4i(0)
    kind = int(0)
    if i < previous_count:
        row = previous_indices[i]
        kind = previous_kinds[i]
    else:
        row = generated_indices[i - previous_count]
        kind = generated_kinds[i - previous_count]
    indices[i] = row
    kinds[i] = kind
    keys[i] = row[3]
    order[i] = i


@wp.kernel(enable_backward=False)
def _feature_sort_keys(
    indices: wp.array[wp.vec4i],
    kinds: wp.array[int],
    order: wp.array[int],
    component: int,
    keys: wp.array[int],
):
    i = wp.tid()
    source = order[i]
    if component < 4:
        keys[i] = indices[source][component]
    else:
        keys[i] = kinds[source]


@wp.kernel(enable_backward=False)
def _unique_features(indices: wp.array[wp.vec4i], kinds: wp.array[int], order: wp.array[int], unique: wp.array[int]):
    i = wp.tid()
    different = i == 0
    if i > 0:
        a, b = order[i - 1], order[i]
        different = kinds[a] != kinds[b]
        for j in range(4):
            if indices[a][j] != indices[b][j]:
                different = True
    unique[i] = int(different)


@wp.kernel(enable_backward=False)
def _gather_unique_rows(
    order: wp.array[int],
    unique: wp.array[int],
    offsets_in: wp.array[int],
    previous_count: int,
    previous_indices: wp.array[wp.vec4i],
    previous_weights: wp.array[wp.vec4d],
    previous_normals: wp.array[wp.vec3d],
    previous_offsets: wp.array[wp.float64],
    previous_kinds: wp.array[int],
    previous_times: wp.array[wp.float64],
    generated_indices: wp.array[wp.vec4i],
    generated_weights: wp.array[wp.vec4d],
    generated_normals: wp.array[wp.vec3d],
    generated_offsets: wp.array[wp.float64],
    generated_kinds: wp.array[int],
    generated_times: wp.array[wp.float64],
    indices: wp.array[wp.vec4i],
    weights: wp.array[wp.vec4d],
    normals: wp.array[wp.vec3d],
    offsets: wp.array[wp.float64],
    kinds: wp.array[int],
    times: wp.array[wp.float64],
):
    i = wp.tid()
    if unique[i] == 0:
        return
    source, target = order[i], offsets_in[i] - 1
    if source < previous_count:
        indices[target] = previous_indices[source]
        weights[target] = previous_weights[source]
        normals[target] = previous_normals[source]
        offsets[target] = previous_offsets[source]
        kinds[target] = previous_kinds[source]
        times[target] = previous_times[source]
    else:
        generated_source = source - previous_count
        indices[target] = generated_indices[generated_source]
        weights[target] = generated_weights[generated_source]
        normals[target] = generated_normals[generated_source]
        offsets[target] = generated_offsets[generated_source]
        kinds[target] = generated_kinds[generated_source]
        times[target] = generated_times[generated_source]


def _surface_topology(tets: np.ndarray, positions: np.ndarray, worlds: np.ndarray):
    n = len(positions)
    parents = np.arange(n)
    adjacency = [set() for _ in range(n)]

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = int(parents[i])
        return i

    occurrences = {}
    for tet in tets:
        if len(set(tet)) != 4:
            raise ValueError("A tetrahedron must contain four distinct particle indices.")
        if np.any(worlds[tet] != worlds[tet[0]]):
            raise ValueError("A tetrahedron cannot span different particle worlds.")
        for a in tet:
            parents[root(int(a))] = root(int(tet[0]))
            adjacency[a].update(int(b) for b in tet if a != b)
        for opposite in range(4):
            face = tuple(int(tet[j]) for j in range(4) if j != opposite)
            a, b, c = positions[list(face)]
            volume6 = float(np.dot(np.cross(b - a, c - a), positions[tet[opposite]] - a))
            if abs(volume6) <= 1.0e-24:
                raise ValueError("Degenerate rest tetrahedron in contact surface.")
            if volume6 > 0:
                face = (face[0], face[2], face[1])
            key = tuple(sorted(face))
            occurrences.setdefault(key, []).append(face)
    if any(len(v) > 2 for v in occurrences.values()):
        raise ValueError("Nonmanifold tetrahedral face belongs to more than two tetrahedra.")
    faces = np.asarray([v[0] for v in occurrences.values() if len(v) == 1], dtype=np.int32).reshape(-1, 3)
    edges = np.asarray(
        sorted({tuple(sorted((int(f[j]), int(f[(j + 1) % 3])))) for f in faces for j in range(3)}), dtype=np.int32
    ).reshape(-1, 2)
    vertices = np.unique(faces).astype(np.int32)
    components = np.asarray([root(i) for i in range(n)], dtype=np.int32)
    starts = np.zeros(n + 1, dtype=np.int32)
    starts[1:] = np.cumsum([len(a) for a in adjacency])
    neighbors = np.asarray([v for a in adjacency for v in sorted(a)], dtype=np.int32)
    return faces, edges, vertices, components, starts, neighbors


class ContactGeometry:
    """Extract a solid boundary once and generate native Warp contact geometry.

    The one-ring exclusion removes rest neighbors connected by tetrahedral
    edges, including opposite edges of the same tetrahedron. Disconnected
    components use ``soft_contact``; nonadjacent parts of a connected solid use
    ``self_contact``. World -1 interacts with all worlds; other worlds interact
    only with themselves. Capacity overflow and unresolved conservative
    advancement raise errors instead of silently dropping contacts.
    Ordinary pairs exit early; the larger bounded iteration budget permits
    slowly closing grazing contacts, which can be substantially more expensive.

    Args:
        tet_indices: Tetrahedral connectivity, shape (tet_count, 4).
        rest_positions: Particle rest positions [m], shape (particle_count, 3).
        particle_world: World index per particle; -1 denotes the global world.
        device: Warp execution device.
        thickness: Separation distance between contacting surfaces [m].
        max_contacts: Maximum number of simultaneous feature pairs.
        self_contact: Enable nonadjacent contact within connected components.
        soft_contact: Enable contact between disconnected components.
        ccd_tolerance: Absolute distance tolerance for conservative advancement [m].
        ccd_iterations: Maximum conservative-advancement iterations per pair.
    """

    def __init__(
        self,
        tet_indices,
        rest_positions,
        particle_world,
        device,
        thickness=1.0e-4,
        max_contacts=65536,
        self_contact=True,
        soft_contact=True,
        ccd_tolerance=1.0e-8,
        ccd_iterations=32768,
    ):
        positions = np.asarray(rest_positions, dtype=np.float64)
        tets = np.asarray(tet_indices, dtype=np.int32)
        worlds = np.asarray(particle_world, dtype=np.int32)
        if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
            raise ValueError("rest_positions must be a finite (particle_count, 3) array.")
        if tets.ndim != 2 or tets.shape[1] != 4 or (tets.size and (tets.min() < 0 or tets.max() >= len(positions))):
            raise ValueError("tet_indices must be a valid (tet_count, 4) particle index array.")
        if worlds.shape != (len(positions),):
            raise ValueError("particle_world must have one world index per particle.")
        if not np.isfinite(thickness) or thickness < 0 or not np.isfinite(ccd_tolerance) or ccd_tolerance <= 0:
            raise ValueError("thickness must be finite and nonnegative; ccd_tolerance must be positive.")
        if max_contacts < 1 or ccd_iterations < 1:
            raise ValueError("max_contacts and ccd_iterations must be positive.")
        self.device = wp.get_device(device)
        self.thickness, self.max_contacts = float(thickness), int(max_contacts)
        self.self_contact, self.soft_contact = bool(self_contact), bool(soft_contact)
        self.ccd_tolerance, self.ccd_iterations = float(ccd_tolerance), int(ccd_iterations)
        self.particle_count = len(positions)
        faces, edges, vertices, components, starts, neighbors = _surface_topology(tets, positions, worlds)
        self.faces = wp.array(faces, dtype=wp.vec3i, device=self.device)
        self.edges = wp.array(edges, dtype=wp.vec2i, device=self.device)
        self.vertices = wp.array(vertices, dtype=int, device=self.device)
        self.worlds = wp.array(worlds, dtype=int, device=self.device)
        self.components = wp.array(components, dtype=int, device=self.device)
        self.neighbor_starts = wp.array(starts, dtype=int, device=self.device)
        self.neighbors = wp.array(neighbors, dtype=int, device=self.device)
        self.indices = wp.empty(max_contacts, dtype=wp.vec4i, device=self.device)
        self.weights = wp.empty(max_contacts, dtype=wp.vec4d, device=self.device)
        self.normals = wp.empty(max_contacts, dtype=wp.vec3d, device=self.device)
        self.offsets = wp.empty(max_contacts, dtype=wp.float64, device=self.device)
        self.kinds = wp.empty(max_contacts, dtype=int, device=self.device)
        self.times = wp.empty(max_contacts, dtype=wp.float64, device=self.device)
        self._generation_status = wp.zeros(2, dtype=int, device=self.device)
        self.count_buffer = self._generation_status[:1]
        self.failure_buffer = self._generation_status[1:]
        self._sort_capacity = 0
        self.face_lower = wp.zeros(len(faces), dtype=wp.vec3, device=self.device)
        self.face_upper = wp.zeros(len(faces), dtype=wp.vec3, device=self.device)
        self.edge_lower = wp.zeros(len(edges), dtype=wp.vec3, device=self.device)
        self.edge_upper = wp.zeros(len(edges), dtype=wp.vec3, device=self.device)
        # Refit preserves the hierarchy, so build it from actual geometry
        # rather than coincident zero boxes with no spatial locality.
        if len(faces):
            rest = wp.array(positions, dtype=wp.vec3d, device=self.device)
            padding = 0.5 * (self.thickness + self.ccd_tolerance)
            wp.launch(
                _face_bounds,
                len(faces),
                [rest, rest, self.faces, padding, self.face_lower, self.face_upper],
                device=self.device,
            )
            wp.launch(
                _edge_bounds,
                len(edges),
                [rest, rest, self.edges, padding, self.edge_lower, self.edge_upper],
                device=self.device,
            )
        self.face_bvh = wp.Bvh(self.face_lower, self.face_upper) if len(faces) else None
        self.edge_bvh = wp.Bvh(self.edge_lower, self.edge_upper) if len(edges) else None

    def retain(self, previous: ContactRows | None, generated: ContactRows) -> ContactRows:
        """Retain all detected features until the current timestep is accepted.

        A contact correction can move a feature outside the detection band.
        Dropping its constraint and solving again then reintroduces penetration,
        causing an alternating candidate set. This union preserves the first
        swept geometry for each feature, including inactive constraints whose
        multipliers can become zero. Start with ``previous=None`` each timestep.
        Returned arrays own their data and survive subsequent generation calls.
        Only the unique contact count is read on the host. Stable device sorting
        keeps the first occurrence of each feature, with previous rows first.
        Canonical feature order removes dependence on CUDA atomic reservations
        and BVH traversal order in subsequent contact iterations.
        Returns ``previous`` unchanged when no new feature was detected.
        """
        names = ("indices", "weights", "normals", "offsets", "kinds", "times")
        if previous is not None and generated.count == 0:
            return previous
        previous_count = previous.count if previous is not None else 0
        if previous_count == 0 and generated.count <= 1:
            return ContactRows(
                generated.count,
                *(wp.clone(getattr(generated, name)[: generated.count]) for name in names),
                generated.thickness,
            )
        total = previous_count + generated.count
        self._ensure_sort_workspace(total)
        source_previous = previous if previous is not None else generated
        wp.launch(
            _initialize_feature_order,
            total,
            [
                previous_count,
                source_previous.indices,
                source_previous.kinds,
                generated.indices,
                generated.kinds,
                self._sort_indices,
                self._sort_kinds,
                self._sort_keys,
                self._sort_order,
            ],
            device=self.device,
        )
        # Stable least-significant-first passes produce the same signed integer
        # lexicographic order as (kind, index0, index1, index2, index3).
        with wp.ScopedDevice(self.device):
            radix_sort_pairs(self._sort_keys, self._sort_order, total)
            for component in (2, 1, 0, 4):
                wp.launch(
                    _feature_sort_keys,
                    total,
                    [self._sort_indices, self._sort_kinds, self._sort_order, component, self._sort_keys],
                    device=self.device,
                )
                radix_sort_pairs(self._sort_keys, self._sort_order, total)
            wp.launch(
                _unique_features,
                total,
                [self._sort_indices, self._sort_kinds, self._sort_order, self._sort_unique],
                device=self.device,
            )
            array_scan(self._sort_unique[:total], self._sort_offsets[:total])
        count = int(self._sort_offsets[total - 1 : total].numpy()[0])
        if previous is not None and count == previous_count:
            return previous
        if count > self.max_contacts:
            raise RuntimeError(
                f"DiffPhD retained contact capacity exceeded: {count} rows require max_contacts >= {count}."
            )
        buffers = [wp.empty(count, dtype=getattr(generated, name).dtype, device=self.device) for name in names]
        wp.launch(
            _gather_unique_rows,
            total,
            [
                self._sort_order,
                self._sort_unique,
                self._sort_offsets,
                previous_count,
                *(getattr(source_previous, name) for name in names),
                *(getattr(generated, name) for name in names),
                *buffers,
            ],
            device=self.device,
        )
        return ContactRows(count, *buffers, generated.thickness)

    def _ensure_sort_workspace(self, count):
        if count <= self._sort_capacity:
            return
        capacity = max(count, 2 * self._sort_capacity)
        self._sort_indices = wp.empty(capacity, dtype=wp.vec4i, device=self.device)
        self._sort_kinds = wp.empty(capacity, dtype=int, device=self.device)
        # Warp radix sort uses the second half of each array as scratch.
        self._sort_keys = wp.empty(2 * capacity, dtype=int, device=self.device)
        self._sort_order = wp.empty(2 * capacity, dtype=int, device=self.device)
        self._sort_unique = wp.empty(capacity, dtype=int, device=self.device)
        self._sort_offsets = wp.empty(capacity, dtype=int, device=self.device)
        self._sort_capacity = capacity

    def generate(
        self,
        q: wp.array[wp.vec3d],
        q_prev: wp.array[wp.vec3d],
        ground=None,
        dt=1.0,
        retained: ContactRows | None = None,
    ) -> ContactRows:
        """Generate frozen rows for linear motion ``q_prev -> q``.

        ``ground`` is ``(normal, offset)`` for the admissible halfspace
        ``dot(normal, x) >= offset``. The normal and offset are normalized
        together. ``dt`` [s] validates the timestep; interpolation uses its
        normalized time, so collision geometry is independent of its magnitude.
        This method reads the contact count on the host and is not graph-capturable.
        ``retained`` may contain canonical, owned rows returned by :meth:`retain`
        earlier in this timestep. Their exact feature pairs skip repeated CCD,
        because the subsequent union keeps their first frozen geometry anyway.
        """
        if q.dtype != wp.vec3d or q_prev.dtype != wp.vec3d:
            raise ValueError("Contact positions must use wp.vec3d.")
        if q.shape != (self.particle_count,) or q_prev.shape != q.shape:
            raise ValueError("Contact position arrays must match the topology particle count.")
        if q.device != self.device or q_prev.device != self.device:
            raise ValueError("Contact positions and geometry must use the same device.")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive.")
        self._generation_status.zero_()
        output = [self.indices, self.weights, self.normals, self.offsets, self.kinds, self.times]
        known = [0, self.indices, self.kinds]
        if retained is not None:
            known = [retained.count, retained.indices, retained.kinds]
        if ground is not None:
            normal = np.asarray(ground[0], dtype=np.float64)
            offset = float(ground[1])
            length = np.linalg.norm(normal)
            if normal.shape != (3,) or not np.isfinite(normal).all() or length <= 0 or not np.isfinite(offset):
                raise ValueError("ground must contain a finite nonzero normal and a finite offset.")
            wp.launch(
                _ground_contacts,
                len(self.vertices),
                [
                    q,
                    q_prev,
                    self.vertices,
                    *known,
                    wp.vec3d(*(normal / length)),
                    offset / length,
                    self.thickness,
                    self.max_contacts,
                    self.count_buffer,
                    *output,
                ],
                device=self.device,
            )
        if (self.self_contact or self.soft_contact) and self.face_bvh is not None:
            # Each participant supplies half of the pair's contact clearance.
            padding = 0.5 * (self.thickness + self.ccd_tolerance)
            wp.launch(
                _face_bounds,
                len(self.faces),
                [q, q_prev, self.faces, padding, self.face_lower, self.face_upper],
                device=self.device,
            )
            wp.launch(
                _edge_bounds,
                len(self.edges),
                [q, q_prev, self.edges, padding, self.edge_lower, self.edge_upper],
                device=self.device,
            )
            self.face_bvh.refit()
            self.edge_bvh.refit()
            common = [self.worlds, self.components, self.neighbor_starts, self.neighbors]
            config = [
                self.self_contact,
                self.soft_contact,
                self.thickness,
                self.ccd_tolerance,
                self.ccd_iterations,
                self.max_contacts,
                self.count_buffer,
                self.failure_buffer,
                *output,
            ]
            wp.launch(
                _vertex_face_contacts,
                len(self.vertices),
                [q, q_prev, self.vertices, self.faces, *common, self.face_bvh.id, *known, *config],
                device=self.device,
                # A single default block serializes divergent CCD paths on
                # one SM. Small surfaces spread those paths across the GPU;
                # larger meshes keep the normal throughput-oriented launch.
                block_dim=1 if self.device.is_cuda and len(self.vertices) <= 256 else 256,
            )
            wp.launch(
                _edge_edge_contacts,
                len(self.edges),
                [q, q_prev, self.edges, *common, self.edge_bvh.id, self.edge_lower, self.edge_upper, *known, *config],
                device=self.device,
                block_dim=1 if self.device.is_cuda and len(self.edges) <= 256 else 256,
            )
        count, failed = map(int, self._generation_status.numpy())
        if count > self.max_contacts:
            raise RuntimeError(f"DiffPhD contact capacity exceeded: {count} rows require max_contacts >= {count}.")
        if failed:
            raise ConvergenceError(
                f"DiffPhD CCD unresolved for {failed} feature pairs; reduce the timestep, increase "
                "ccd_iterations, or remove degenerate/coincident surface features."
            )
        return ContactRows(count, *output, self.thickness)
