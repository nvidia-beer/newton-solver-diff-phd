# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Portions translated from DiffPhD (https://github.com/Akinesia112/DiffPhD,
# commit d319435), Copyright (c) 2021 MIT Graphics Group, MIT License.
# See THIRD_PARTY_NOTICES.md.

"""Float64 local projections of the reference DiffPhD element energies.

Every projection works on the unsigned singular value decomposition used by
Eigen's ``JacobiSVD``: singular values are nonnegative and sorted in
descending order, and ``U V^T`` is a reflection when ``det F < 0``. Sources in the
reference, all under ``cpp/core/src``:

* corotated: ``pd_energy/corotated_pd_element_energy.cpp`` with
  ``common/geometry.cpp`` ``PolarDecomposition``;
* volume: ``pd_energy/volume_pd_element_energy.cpp`` ``ProjectToManifold``;
* Neo-Hookean proximal map: ``pd_energy/neohookean_pd_element_energy.cpp``
  ``SolveProximal`` and ``HessianProximal``;
* muscle fibre: ``pd_energy/pd_muscle_energy.cpp`` ``ProjectToManifold``.
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
    ``U V^T`` is a reflection when ``det F < 0``.
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
def ldlt3_solve(a: wp.mat33d, b: wp.vec3d):
    """Solve a symmetric 3x3 system as Eigen's dense ``LDLT`` does.

    Eigen's left-looking factorization picks the largest remaining original
    diagonal entry as pivot and applies the pseudo-inverse of ``D``.
    """
    m = a
    perm = wp.vec3i(0, 1, 2)
    for k in range(3):
        biggest = k
        for i in range(k + 1, 3):
            if wp.abs(m[i, i]) > wp.abs(m[biggest, biggest]):
                biggest = i
        if biggest != k:
            for j in range(3):
                value = m[k, j]
                m[k, j] = m[biggest, j]
                m[biggest, j] = value
            for i in range(3):
                value = m[i, k]
                m[i, k] = m[i, biggest]
                m[i, biggest] = value
            index = perm[k]
            perm[k] = perm[biggest]
            perm[biggest] = index
        for j in range(k):
            # A10 * D * A10^T and A20 * D * A10^T, left-looking.
            m[k, k] = m[k, k] - m[k, j] * m[k, j] * m[j, j]
        for i in range(k + 1, 3):
            for j in range(k):
                m[i, k] = m[i, k] - m[i, j] * m[j, j] * m[k, j]
            if m[k, k] != wp.float64(0.0):
                m[i, k] = m[i, k] / m[k, k]
    y = wp.vec3d(b[perm[0]], b[perm[1]], b[perm[2]])
    for i in range(3):
        for j in range(i):
            y[i] = y[i] - m[i, j] * y[j]
    for i in range(3):
        if wp.abs(m[i, i]) > DOUBLE_MIN:
            y[i] = y[i] / m[i, i]
        else:
            y[i] = wp.float64(0.0)
    for reverse in range(3):
        i = 2 - reverse
        for j in range(i + 1, 3):
            y[i] = y[i] - m[j, i] * y[j]
    x = wp.vec3d(wp.float64(0.0))
    for i in range(3):
        x[perm[i]] = y[i]
    return x


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


@wp.func
def nh_prox_hessian(sigma: wp.vec3d, mu: wp.float64, lame: wp.float64, penalty: wp.float64):
    """Hessian of ``psi(sigma) + penalty / 2 ||sigma - sigma_F||^2`` (``HessianProximal``)."""
    product = sigma[0] * sigma[1] * sigma[2]
    log_j = wp.log(wp.max(product, wp.float64(1.0e-12)))
    h = wp.mat33d(wp.float64(0.0))
    for i in range(3):
        for j in range(3):
            if i == j:
                h[i, j] = mu + (mu + lame - lame * log_j) / (sigma[i] * sigma[i]) + penalty
            else:
                h[i, j] = lame / (sigma[i] * sigma[j])
    return h


@wp.func
def nh_prox_stretches(target: wp.vec3d, mu: wp.float64, lame: wp.float64, penalty: wp.float64):
    """Proximal Neo-Hookean singular values by the reference damped Newton (``SolveProximal``)."""
    floor = wp.float64(1.0e-4)
    tolerance = wp.float64(1.0e-10)
    sigma = wp.vec3d(wp.max(target[0], floor), wp.max(target[1], floor), wp.max(target[2], floor))
    for _iteration in range(32):
        product = sigma[0] * sigma[1] * sigma[2]
        log_j = wp.log(wp.max(product, wp.float64(1.0e-12)))
        gradient = wp.vec3d(wp.float64(0.0))
        for i in range(3):
            gradient[i] = mu * sigma[i] - mu / sigma[i] + lame * log_j / sigma[i] + penalty * (sigma[i] - target[i])
        if wp.length(gradient) < tolerance:
            break
        hessian = nh_prox_hessian(sigma, mu, lame, penalty) + wp.float64(1.0e-9) * wp.identity(n=3, dtype=wp.float64)
        delta = ldlt3_solve(hessian, gradient)
        alpha = wp.float64(1.0)
        trial = sigma - alpha * delta
        for _search in range(20):
            if wp.min(wp.min(trial[0], trial[1]), trial[2]) > floor * wp.float64(0.5):
                break
            alpha *= wp.float64(0.5)
            trial = sigma - alpha * delta
        step = wp.max(wp.max(wp.abs(trial[0] - sigma[0]), wp.abs(trial[1] - sigma[1])), wp.abs(trial[2] - sigma[2]))
        sigma = wp.vec3d(wp.max(trial[0], floor), wp.max(trial[1], floor), wp.max(trial[2], floor))
        if step < tolerance:
            break
    return sigma


@wp.func
def muscle_projection(f: wp.mat33d, direction: wp.vec3d, target_length: wp.float64):
    """Fibre projection ``a F m / |F m|`` of the SoftCon contractile fibre (zero for a collapsed fibre)."""
    fibre = f * direction
    length = wp.length(fibre)
    result = wp.vec3d(wp.float64(0.0))
    if length > DOUBLE_EPSILON:
        result = target_length / length * fibre
    return result


# Directional derivatives used by the reference backward pass
# (``ProjectToManifoldDifferential`` of every energy).


@wp.func
def dsvd(u: wp.mat33d, sigma: wp.vec3d, v: wp.mat33d, df: wp.mat33d):
    """``dSvd`` of ``common/geometry.cpp``: returns ``dU, dsigma, dV`` for sorted ``sigma``.

    Pairs with ``sigma_i - sigma_j <= 10 eps`` contribute nothing, as in the reference.
    """
    dp = wp.transpose(u) * df * v
    dsigma = wp.vec3d(dp[0, 0], dp[1, 1], dp[2, 2])
    s = diagonal_matrix(sigma)
    factor = wp.mat33d(wp.float64(0.0))
    for i in range(3):
        for j in range(i + 1, 3):
            if sigma[i] - sigma[j] > wp.float64(10.0) * DOUBLE_EPSILON:
                factor[i, j] = wp.float64(1.0) / (sigma[j] * sigma[j] - sigma[i] * sigma[i])
                factor[j, i] = -factor[i, j]
    omega_u = wp.cw_mul(factor, dp * s + s * wp.transpose(dp))
    omega_v = wp.cw_mul(factor, s * dp + wp.transpose(dp) * s)
    return u * omega_u, dsigma, v * omega_v


@wp.func
def corotated_jvp(u: wp.mat33d, sigma: wp.vec3d, v: wp.mat33d, df: wp.mat33d):
    """``dRFromdF``: derivative of ``R = U V^T`` with ``S = V diag(sigma) V^T``."""
    r = u * wp.transpose(v)
    s = v * diagonal_matrix(sigma) * wp.transpose(v)
    lhs = wp.transpose(r) * df - wp.transpose(df) * r
    a = wp.mat33d(
        s[0, 0] + s[1, 1],
        s[1, 2],
        -s[0, 2],
        s[1, 2],
        s[0, 0] + s[2, 2],
        s[0, 1],
        -s[0, 2],
        s[0, 1],
        s[1, 1] + s[2, 2],
    )
    xyz = wp.inverse(a) * wp.vec3d(lhs[0, 1], lhs[0, 2], lhs[1, 2])
    w = wp.mat33d(wp.float64(0.0), xyz[0], xyz[1], -xyz[0], wp.float64(0.0), xyz[2], -xyz[1], -xyz[2], wp.float64(0.0))
    return r * w


@wp.func
def volume_jvp(u: wp.mat33d, sigma: wp.vec3d, v: wp.mat33d, offset: wp.vec3d, df: wp.mat33d):
    """Volume-projection derivative from the reference KKT system."""
    s = sigma + offset
    gradient = wp.vec3d(s[1] * s[2], s[0] * s[2], s[0] * s[1])
    # Hess C has zero diagonal and the product of the remaining stretch off it.
    h = wp.mat33d(wp.float64(0.0), s[2], s[1], s[2], wp.float64(0.0), s[0], s[1], s[0], wp.float64(0.0))
    du, dsigma, dv = dsvd(u, sigma, v, df)
    magnitude = wp.length(offset) / wp.length(gradient)
    multiplier = magnitude
    plus = offset - magnitude * gradient
    minus = offset + magnitude * gradient
    if wp.dot(plus, plus) < wp.dot(minus, minus):
        multiplier = -magnitude
    kkt = wp.mat44d(wp.float64(0.0))
    for i in range(3):
        for j in range(3):
            kkt[i, j] = multiplier * h[i, j]
        kkt[i, i] = kkt[i, i] + wp.float64(1.0)
        kkt[i, 3] = gradient[i]
        kkt[3, i] = gradient[i]
    top = -multiplier * (h * dsigma)
    rhs = wp.vec4d(top[0], top[1], top[2], -wp.dot(gradient, dsigma))
    solution = wp.inverse(kkt) * rhs
    doffset = wp.vec3d(solution[0], solution[1], solution[2])
    vt = wp.transpose(v)
    return (
        df
        + du * diagonal_matrix(offset) * vt
        + u * diagonal_matrix(doffset) * vt
        + u * diagonal_matrix(offset) * wp.transpose(dv)
    )


@wp.func
def nh_prox_jvp(
    u: wp.mat33d,
    sigma: wp.vec3d,
    v: wp.mat33d,
    stretches: wp.vec3d,
    df: wp.mat33d,
    mu: wp.float64,
    lame: wp.float64,
    penalty: wp.float64,
):
    """Implicit-function derivative of the proximal Neo-Hookean projection (no eigenvalue filtering)."""
    du, dsigma, dv = dsvd(u, sigma, v, df)
    hessian = nh_prox_hessian(stretches, mu, lame, penalty) + wp.float64(1.0e-9) * wp.identity(n=3, dtype=wp.float64)
    dstretches = ldlt3_solve(hessian, penalty * dsigma)
    vt = wp.transpose(v)
    return (
        du * diagonal_matrix(stretches) * vt
        + u * diagonal_matrix(dstretches) * vt
        + u * diagonal_matrix(stretches) * wp.transpose(dv)
    )


@wp.func
def muscle_jvp(f: wp.mat33d, direction: wp.vec3d, target_length: wp.float64, df: wp.mat33d):
    """Derivative of the fibre projection with respect to ``F`` at fixed target length."""
    fibre = f * direction
    length = wp.length(fibre)
    result = wp.vec3d(wp.float64(0.0))
    if length > DOUBLE_EPSILON:
        dfibre = df * direction
        dlength = wp.dot(fibre / length, dfibre)
        ratio = target_length / length
        result = ratio * dfibre - target_length / (length * length) * dlength * fibre
    return result


@wp.func
def neo_hookean_stress(f: wp.mat33d, mu: wp.float64, lame: wp.float64):
    """True Neo-Hookean first Piola-Kirchhoff stress (``NeoHookeanPdElementEnergy::StressTensor``)."""
    log_j = wp.log(wp.max(wp.determinant(f), wp.float64(1.0e-12)))
    inverse_transpose = wp.transpose(wp.inverse(f))
    return mu * (f - inverse_transpose) + lame * log_j * inverse_transpose
