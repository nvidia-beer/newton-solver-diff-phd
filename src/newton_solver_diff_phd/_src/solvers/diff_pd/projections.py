# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Portions translated from DiffPD (https://github.com/mit-gfx/diff_pd_public,
# commit 1594aa0), Copyright (c) 2021 MIT Graphics Group, MIT License.
# See THIRD_PARTY_NOTICES.md.

"""Float64 local projections of the DiffPD corotated and volume element energies.

Both projections work on the unsigned singular value decomposition computed by
Eigen's ``JacobiSVD`` in ``DeformationGradientAuxiliaryData::Initialize``:
singular values are nonnegative and sorted in descending order, and ``U`` is a
reflection when ``det F < 0``. Sources in the reference, under ``cpp/core/src``:

* corotated: ``pd_energy/corotated_pd_element_energy.cpp`` (``R = U V^T``);
* volume: ``pd_energy/volume_pd_element_energy.cpp`` ``ProjectToManifold``.
"""

import warp as wp

# std::numeric_limits<double>::epsilon() and ::min().
DOUBLE_EPSILON = wp.constant(wp.float64(2.220446049250313e-16))
DOUBLE_MIN = wp.constant(wp.float64(2.2250738585072014e-308))


@wp.func
def _swap_columns(m: wp.mat33d, i: int, j: int):
    for row in range(3):
        value = m[row, i]
        m[row, i] = m[row, j]
        m[row, j] = value
    return m


# Eigen's two-sided JacobiSVD sweeps until converged; the cap keeps the loop
# bounded inside graphs and is far above the sweeps a 3x3 matrix needs.
JACOBI_MAX_SWEEPS = 64


@wp.func
def _rotate_rows(m: wp.mat33d, p: int, q: int, c: wp.float64, s: wp.float64):
    # Eigen applyOnTheLeft(p, q, J): rows (p, q) <- J (p, q), J = [[c, s], [-s, c]].
    for j in range(3):
        x = m[p, j]
        y = m[q, j]
        m[p, j] = c * x + s * y
        m[q, j] = -s * x + c * y
    return m


@wp.func
def _rotate_columns(m: wp.mat33d, p: int, q: int, c: wp.float64, s: wp.float64):
    # Eigen applyOnTheRight(p, q, J): columns (p, q) <- (p, q) J.
    for i in range(3):
        x = m[i, p]
        y = m[i, q]
        m[i, p] = c * x - s * y
        m[i, q] = s * x + c * y
    return m


@wp.func
def _make_jacobi(x: wp.float64, y: wp.float64, z: wp.float64):
    # JacobiRotation::makeJacobi for real scalars.
    c = wp.float64(1.0)
    s = wp.float64(0.0)
    denominator = wp.float64(2.0) * wp.abs(y)
    if denominator >= DOUBLE_MIN:
        tau = (x - z) / denominator
        w = wp.sqrt(tau * tau + wp.float64(1.0))
        t = wp.float64(0.0)
        if tau > wp.float64(0.0):
            t = wp.float64(1.0) / (tau + w)
        else:
            t = wp.float64(1.0) / (tau - w)
        sign = wp.float64(-1.0)
        if t > wp.float64(0.0):
            sign = wp.float64(1.0)
        n = wp.float64(1.0) / wp.sqrt(t * t + wp.float64(1.0))
        s = -sign * (y / wp.abs(y)) * wp.abs(t) * n
        c = n
    return c, s


@wp.func
def unsigned_svd(f: wp.mat33d):
    """Return ``U, sigma, V`` with ``F = U diag(sigma) V^T`` exactly as Eigen's ``JacobiSVD`` computes them.

    Two-sided Jacobi with ``real_2x2_jacobi_svd`` sweeps, threshold
    ``max(DBL_MIN, 2 eps max|diag|)`` on the scaled matrix, nonnegative singular
    values (negative diagonal entries flip columns of ``U``) and a descending
    sort that stops at the first zero maximum. ``U`` and ``V`` are orthogonal;
    ``U`` is a reflection when ``det F < 0``.
    """
    scale = wp.float64(0.0)
    for i in range(3):
        for j in range(3):
            scale = wp.max(scale, wp.abs(f[i, j]))
    if scale == wp.float64(0.0):
        scale = wp.float64(1.0)
    work = f / scale
    u = wp.identity(n=3, dtype=wp.float64)
    v = wp.identity(n=3, dtype=wp.float64)
    largest = wp.max(wp.max(wp.abs(work[0, 0]), wp.abs(work[1, 1])), wp.abs(work[2, 2]))
    precision = wp.float64(2.0) * DOUBLE_EPSILON
    for _sweep in range(JACOBI_MAX_SWEEPS):
        finished = int(1)
        for pair in range(3):
            # Pairs (p, q) in Eigen's order: (1, 0), (2, 0), (2, 1).
            p = int(1)
            q = int(0)
            if pair == 1:
                p = 2
            elif pair == 2:
                p = 2
                q = 1
            threshold = wp.max(DOUBLE_MIN, precision * largest)
            if wp.abs(work[p, q]) > threshold or wp.abs(work[q, p]) > threshold:
                finished = 0
                # real_2x2_jacobi_svd.
                t = work[p, p] + work[q, q]
                d = work[q, p] - work[p, q]
                c1 = wp.float64(1.0)
                s1 = wp.float64(0.0)
                if wp.abs(d) >= DOUBLE_MIN:
                    ratio = t / d
                    root = wp.sqrt(wp.float64(1.0) + ratio * ratio)
                    s1 = wp.float64(1.0) / root
                    c1 = ratio / root
                m00 = c1 * work[p, p] + s1 * work[q, p]
                m01 = c1 * work[p, q] + s1 * work[q, q]
                m11 = -s1 * work[p, q] + c1 * work[q, q]
                cr, sr = _make_jacobi(m00, m01, m11)
                # j_left = rot1 * j_right^T.
                cl = c1 * cr + s1 * sr
                sl = s1 * cr - c1 * sr
                work = _rotate_rows(work, p, q, cl, sl)
                u = _rotate_columns(u, p, q, cl, -sl)
                work = _rotate_columns(work, p, q, cr, sr)
                v = _rotate_columns(v, p, q, cr, sr)
                largest = wp.max(largest, wp.max(wp.abs(work[p, p]), wp.abs(work[q, q])))
        if finished != 0:
            break
    sigma = wp.vec3d(wp.float64(0.0))
    for i in range(3):
        a = work[i, i]
        sigma[i] = wp.abs(a) * scale
        if a < wp.float64(0.0):
            for row in range(3):
                u[row, i] = -u[row, i]
    for i in range(3):
        position = i
        for j in range(i + 1, 3):
            if sigma[j] > sigma[position]:
                position = j
        if sigma[position] == wp.float64(0.0):
            break
        if position != i:
            value = sigma[i]
            sigma[i] = sigma[position]
            sigma[position] = value
            u = _swap_columns(u, i, position)
            v = _swap_columns(v, i, position)
    return u, sigma, v


@wp.func
def diagonal_matrix(x: wp.vec3d):
    result = wp.mat33d(wp.float64(0.0))
    for i in range(3):
        result[i, i] = x[i]
    return result


@wp.func
def corotated_projection(u: wp.mat33d, v: wp.mat33d):
    """Orthogonal polar factor ``R = U V^T``; a reflection when ``det F < 0``."""
    return u * wp.transpose(v)


@wp.func
def volume_offset(sigma: wp.vec3d):
    """Closest ``D`` with ``prod(sigma + D) = 1``; returns ``D`` and a singular flag.

    The reference raises "Singular F." when ``min(sigma) < eps``; the flag
    carries that failure to the solver's status word.
    """
    singular = int(0)
    if sigma[2] < DOUBLE_EPSILON:
        singular = 1
    product = sigma[0] * sigma[1] * sigma[2]
    d = sigma / wp.pow(product, wp.float64(1.0) / wp.float64(3.0)) - sigma
    for _iteration in range(50):
        s = sigma + d
        constraint = s[0] * s[1] * s[2] - wp.float64(1.0)
        gradient = wp.vec3d(s[1] * s[2], s[0] * s[2], s[0] * s[1])
        following = (wp.dot(gradient, d) - constraint) / wp.dot(gradient, gradient) * gradient
        difference = wp.max(
            wp.max(wp.abs(following[0] - d[0]), wp.abs(following[1] - d[1])), wp.abs(following[2] - d[2])
        )
        if difference <= DOUBLE_EPSILON:
            break
        d = following
    return d, singular
