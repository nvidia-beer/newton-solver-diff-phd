# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Float64 tetrahedral energies and implicit local projection derivatives.

Model 0 is projective dynamics with energies ``mu ||F-R||²`` and
``lambda/2 ||F-V||²``, where R is the closest rotation and V is the
closest positive-volume matrix with determinant one.
Model 1 is the physical compressible logarithmic Neo-Hookean energy.
Both expose the tangent of the *same* stress used by the forward residual.

The Neo-Hookean proximal map is also provided as a local building block.
Its bare local/global fixed point minimizes a Moreau envelope; it must not
replace the physical Neo-Hookean residual without an ADMM dual update.
All implicit derivatives assume converged, nonsingular local stationary
points. No backward-only Hessian filtering changes those derivatives.
"""

import warp as wp

MODEL_COROTATED = 0
MODEL_NEO_HOOKEAN = 1


@wp.func
def matrix_inner(a: wp.mat33d, b: wp.mat33d):
    return wp.dot(a[0], b[0]) + wp.dot(a[1], b[1]) + wp.dot(a[2], b[2])


@wp.func
def diagonal_matrix(x: wp.vec3d):
    result = wp.mat33d(wp.float64(0.0))
    for i in range(3):
        result[i, i] = x[i]
    return result


@wp.func
def signed_svd(f: wp.mat33d):
    """Compute a double-precision signed SVD without differentiating eigenvectors."""
    # One-sided Jacobi avoids squaring the condition number in F.T F. A tiny
    # eigenvalue recovered from F.T F can be entirely cancellation error;
    # dividing F V by its square root then destroys orthogonality of U.
    columns = f
    v = wp.identity(n=3, dtype=wp.float64)
    for _sweep in range(16):
        for pair in range(3):
            p = int(0)
            q = int(1)
            if pair == 1:
                q = 2
            elif pair == 2:
                p = 1
                q = 2
            cp = wp.vec3d(columns[0, p], columns[1, p], columns[2, p])
            cq = wp.vec3d(columns[0, q], columns[1, q], columns[2, q])
            pp, qq = wp.dot(cp, cp), wp.dot(cq, cq)
            off = wp.dot(cp, cq)
            if wp.abs(off) > wp.float64(1.0e-16) * wp.sqrt(pp) * wp.sqrt(qq):
                tau = (qq - pp) / (wp.float64(2.0) * off)
                sign = wp.float64(1.0)
                if tau < wp.float64(0.0):
                    sign = wp.float64(-1.0)
                tangent = sign / (wp.abs(tau) + wp.sqrt(wp.float64(1.0) + tau * tau))
                cosine = wp.float64(1.0) / wp.sqrt(wp.float64(1.0) + tangent * tangent)
                sine = tangent * cosine
                vp = wp.vec3d(v[0, p], v[1, p], v[2, p])
                vq = wp.vec3d(v[0, q], v[1, q], v[2, q])
                new_cp, new_cq = cosine * cp - sine * cq, sine * cp + cosine * cq
                new_vp, new_vq = cosine * vp - sine * vq, sine * vp + cosine * vq
                for row in range(3):
                    columns[row, p] = new_cp[row]
                    columns[row, q] = new_cq[row]
                    v[row, p] = new_vp[row]
                    v[row, q] = new_vq[row]
    sigma = wp.vec3d(wp.float64(0.0))
    u = wp.mat33d(wp.float64(0.0))
    minimum_index = int(0)
    for i in range(3):
        column = wp.vec3d(columns[0, i], columns[1, i], columns[2, i])
        sigma[i] = wp.length(column)
        if sigma[i] > wp.float64(0.0):
            column /= sigma[i]
        for row in range(3):
            u[row, i] = column[row]
        if sigma[i] < sigma[minimum_index]:
            minimum_index = i
    if sigma[minimum_index] == wp.float64(0.0):
        # Complete a rank-two basis without amplifying a manufactured floor.
        p, q = (minimum_index + 1) % 3, (minimum_index + 2) % 3
        up = wp.vec3d(u[0, p], u[1, p], u[2, p])
        uq = wp.vec3d(u[0, q], u[1, q], u[2, q])
        completed = wp.normalize(wp.cross(up, uq))
        for row in range(3):
            u[row, minimum_index] = completed[row]
    if wp.determinant(u) < wp.float64(0.0):
        sigma[minimum_index] = -sigma[minimum_index]
        for row in range(3):
            u[row, minimum_index] = -u[row, minimum_index]
    return u, sigma, v


@wp.func
def rotation_projection(f: wp.mat33d):
    u, _sigma, v = signed_svd(f)
    return u * wp.transpose(v)


@wp.func
def rotation_projection_jvp(f: wp.mat33d, df: wp.mat33d):
    u, sigma, v = signed_svd(f)
    local_df = wp.transpose(u) * df * v
    local_dr = wp.mat33d(wp.float64(0.0))
    for i in range(3):
        for j in range(3):
            if i != j:
                local_dr[i, j] = (local_df[i, j] - local_df[j, i]) / (sigma[i] + sigma[j])
    return u * local_dr * wp.transpose(v)


@wp.func_grad(rotation_projection)
def _rotation_projection_grad(f: wp.mat33d, adj_result: wp.mat33d):
    wp.adjoint[f] += rotation_projection_jvp(f, adj_result)


@wp.func
def volume_kkt(sigma: wp.vec3d, multiplier: wp.float64):
    g = wp.vec3d(sigma[1] * sigma[2], sigma[0] * sigma[2], sigma[0] * sigma[1])
    h = wp.mat44d(wp.float64(0.0))
    for i in range(3):
        h[i, i] = wp.float64(1.0)
        h[i, 3] = g[i]
        h[3, i] = g[i]
    h[0, 1] = multiplier * sigma[2]
    h[1, 0] = h[0, 1]
    h[0, 2] = multiplier * sigma[1]
    h[2, 0] = h[0, 2]
    h[1, 2] = multiplier * sigma[0]
    h[2, 1] = h[1, 2]
    return h


@wp.func
def volume_kkt_residual(target: wp.vec3d, sigma: wp.vec3d, multiplier: wp.float64):
    g = wp.vec3d(sigma[1] * sigma[2], sigma[0] * sigma[2], sigma[0] * sigma[1])
    residual = sigma - target + multiplier * g
    return wp.vec4d(residual[0], residual[1], residual[2], sigma[0] * sigma[1] * sigma[2] - wp.float64(1.0))


@wp.func
def _volume_compression_stretches(target: wp.vec3d, tau: wp.float64):
    result = wp.vec3d(wp.float64(0.0))
    for i in range(3):
        root = wp.sqrt(target[i] * target[i] + wp.float64(4.0) * tau)
        result[i] = wp.float64(0.5) * (target[i] + root)
        if target[i] < wp.float64(0.0):
            result[i] = wp.float64(2.0) * tau / (root - target[i])
    return result


@wp.func
def _volume_expansion_stretches(target: wp.vec3d, x: wp.float64, minus: bool):
    # target is in descending order; only its smallest stretch may use the
    # minus root at a constrained minimum (by rearrangement and second order).
    result = wp.vec3d(wp.float64(0.0))
    for i in range(3):
        ratio = target[2] / target[i]
        root = wp.sqrt(wp.max(wp.float64(0.0), wp.float64(1.0) - ratio * ratio * x))
        result[i] = wp.float64(0.5) * target[i] * (wp.float64(1.0) + root)
    if minus:
        result[2] = target[2] * x / (wp.float64(2.0) * (wp.float64(1.0) + wp.sqrt(wp.float64(1.0) - x)))
    return result


@wp.func
def _volume_product(stretches: wp.vec3d):
    return stretches[0] * stretches[1] * stretches[2]


@wp.func
def _volume_minus_slope(x: wp.float64, r0: wp.float64, r1: wp.float64):
    # The sign of d log(product)/d gamma is H(x). The subtraction below is
    # rationalized to retain accuracy when the two smallest stretches agree.
    value = wp.float64(0.0)
    if x == wp.float64(1.0):
        if r1 < wp.float64(1.0):
            value = wp.float64(wp.inf)
        elif r0 == wp.float64(1.0):
            value = wp.float64(-wp.inf)
        else:
            value = wp.float64(3.0) - wp.float64(1.0) / wp.sqrt(wp.float64(1.0) - r0)
    else:
        a = wp.sqrt(wp.float64(1.0) - x)
        b = wp.sqrt(wp.float64(1.0) - r1 * x)
        difference = x * (wp.float64(1.0) - r1) / (a * b * (a + b))
        value = wp.float64(3.0) - wp.float64(1.0) / wp.sqrt(wp.float64(1.0) - r0 * x) + difference
    return value


@wp.func
def _volume_minus_slope_derivative_sign(x: wp.float64, r0: wp.float64, r1: wp.float64):
    # H' divided by a positive factor is monotone increasing on [0, 1].
    # Hence H has at most one minimum and its zeros can be isolated exactly.
    result = wp.float64(1.0)
    for i in range(2):
        ratio = r0
        if i == 1:
            ratio = r1
        term = wp.float64(1.0)
        if ratio < wp.float64(1.0):
            fraction = (wp.float64(1.0) - x) / (wp.float64(1.0) - ratio * x)
            term = ratio * fraction * wp.sqrt(fraction)
        result -= term
    return result


@wp.func
def _volume_expansion_root(target: wp.vec3d, low: wp.float64, high: wp.float64, minus: bool):
    for _iteration in range(80):
        middle = wp.float64(0.5) * (low + high)
        product = _volume_product(_volume_expansion_stretches(target, middle, minus))
        if (minus and product < wp.float64(1.0)) or (not minus and product > wp.float64(1.0)):
            low = middle
        else:
            high = middle
    x = wp.float64(0.5) * (low + high)
    return _volume_expansion_stretches(target, x, minus), wp.float64(0.25) * x * target[2] * target[2]


@wp.func
def volume_stretches(target: wp.vec3d):
    """Globally closest positive stretches with product one.

    Expansion can have several stationary branches. Enumerate the all-plus
    branch and every second-order-valid one-minus branch, isolate their scalar
    roots using the unimodal slope function, and choose the least distance.
    A NaN multiplier flags a nonunique minimum to the implicit derivative;
    its forward projection remains a valid deterministic minimizer.
    """
    sigma = wp.vec3d(wp.float64(wp.nan))
    multiplier = wp.float64(wp.nan)
    product = _volume_product(target)
    if product <= wp.float64(1.0):
        # For compression gamma=-tau <= 0 and all three positive roots are
        # strictly monotone. This also covers the signed smallest SVD stretch.
        low = wp.float64(0.0)
        high = wp.float64(1.0)
        for _iteration in range(80):
            tau = wp.float64(0.5) * (low + high)
            value = _volume_product(_volume_compression_stretches(target, tau))
            if value < wp.float64(1.0):
                low = tau
            else:
                high = tau
        tau = wp.float64(0.5) * (low + high)
        sigma = _volume_compression_stretches(target, tau)
        multiplier = -tau
        if product == wp.float64(1.0):
            sigma = target
            multiplier = wp.float64(0.0)
    else:
        r0 = (target[2] / target[0]) * (target[2] / target[0])
        r1 = (target[2] / target[1]) * (target[2] / target[1])
        minimum = wp.float64(0.0)
        if _volume_minus_slope_derivative_sign(wp.float64(0.0), r0, r1) < wp.float64(0.0):
            minimum = wp.float64(1.0)
            if _volume_minus_slope_derivative_sign(wp.float64(1.0), r0, r1) > wp.float64(0.0):
                low = wp.float64(0.0)
                high = wp.float64(1.0)
                for _iteration in range(64):
                    middle = wp.float64(0.5) * (low + high)
                    if _volume_minus_slope_derivative_sign(middle, r0, r1) < wp.float64(0.0):
                        low = middle
                    else:
                        high = middle
                minimum = wp.float64(0.5) * (low + high)
        first_high = wp.float64(1.0)
        second_low = wp.float64(1.0)
        has_second = bool(False)
        if _volume_minus_slope(minimum, r0, r1) < wp.float64(0.0):
            low = wp.float64(0.0)
            high = minimum
            for _iteration in range(64):
                middle = wp.float64(0.5) * (low + high)
                if _volume_minus_slope(middle, r0, r1) > wp.float64(0.0):
                    low = middle
                else:
                    high = middle
            first_high = wp.float64(0.5) * (low + high)
            if _volume_minus_slope(wp.float64(1.0), r0, r1) > wp.float64(0.0):
                low = minimum
                high = wp.float64(1.0)
                for _iteration in range(64):
                    middle = wp.float64(0.5) * (low + high)
                    if _volume_minus_slope(middle, r0, r1) < wp.float64(0.0):
                        low = middle
                    else:
                        high = middle
                second_low = wp.float64(0.5) * (low + high)
                has_second = True
        best_cost = wp.float64(wp.inf)
        ambiguous = bool(False)
        for branch in range(3):
            low = wp.float64(0.0)
            high = wp.float64(1.0)
            minus = branch != 0
            valid = bool(False)
            if branch == 0:
                valid = _volume_product(_volume_expansion_stretches(target, high, False)) <= wp.float64(1.0)
            else:
                high = first_high
                if branch == 2:
                    low = second_low
                    high = wp.float64(1.0)
                valid = (branch == 1 or has_second) and (
                    _volume_product(_volume_expansion_stretches(target, low, True)) <= wp.float64(1.0)
                    and _volume_product(_volume_expansion_stretches(target, high, True)) >= wp.float64(1.0)
                )
            if valid:
                candidate, candidate_multiplier = _volume_expansion_root(target, low, high, minus)
                cost = wp.length_sq(candidate - target)
                tie = wp.abs(cost - best_cost) <= wp.float64(1.0e-12) * (wp.float64(1.0) + cost)
                distinct = wp.length_sq(candidate - sigma) > wp.float64(1.0e-16)
                if cost < best_cost:
                    ambiguous = tie and distinct
                    sigma = candidate
                    multiplier = candidate_multiplier
                    best_cost = cost
                elif tie and distinct:
                    ambiguous = True
        # Equal target stretches with unequal minimizing stretches admit a
        # continuum of equally good rotated projections: no unique derivative.
        for i in range(3):
            for j in range(i + 1, 3):
                if wp.abs(target[i] - target[j]) <= wp.float64(1.0e-12) * wp.max(target[i], target[j]):
                    if wp.abs(sigma[i] - sigma[j]) > wp.float64(1.0e-7):
                        ambiguous = True
        if ambiguous:
            multiplier = wp.float64(wp.nan)
    product_error = wp.abs(_volume_product(sigma) - wp.float64(1.0))
    valid = wp.isfinite(product_error) and product_error <= wp.float64(1.0e-10)
    if wp.isfinite(multiplier):
        residual = volume_kkt_residual(target, sigma, multiplier)
        valid = (
            valid
            and wp.isfinite(wp.length_sq(residual))
            and wp.length_sq(residual) <= wp.float64(1.0e-20) * (wp.float64(1.0) + wp.length_sq(target))
        )
    if not valid:
        sigma = wp.vec3d(wp.float64(wp.nan))
        multiplier = wp.float64(wp.nan)
    return sigma, multiplier


@wp.func
def spectral_projection_jvp(
    u: wp.mat33d,
    sigma: wp.vec3d,
    v: wp.mat33d,
    projected: wp.vec3d,
    stretch_jacobian: wp.mat33d,
    df: wp.mat33d,
):
    """Differentiate an isotropic map, including the repeated-stretch limit."""
    local_df = wp.transpose(u) * df * v
    diagonal = stretch_jacobian * wp.vec3d(local_df[0, 0], local_df[1, 1], local_df[2, 2])
    local_dp = diagonal_matrix(diagonal)
    for i in range(3):
        for j in range(3):
            if i != j:
                gap = sigma[i] - sigma[j]
                symmetric_coefficient = stretch_jacobian[i, i] - stretch_jacobian[i, j]
                if wp.abs(gap) > wp.float64(1.0e-7) * wp.max(wp.abs(sigma[i]), wp.abs(sigma[j])):
                    symmetric_coefficient = (projected[i] - projected[j]) / gap
                skew_coefficient = (projected[i] + projected[j]) / (sigma[i] + sigma[j])
                symmetric_df = wp.float64(0.5) * (local_df[i, j] + local_df[j, i])
                skew_df = wp.float64(0.5) * (local_df[i, j] - local_df[j, i])
                local_dp[i, j] = symmetric_coefficient * symmetric_df + skew_coefficient * skew_df
    return u * local_dp * wp.transpose(v)


@wp.func
def volume_projection(f: wp.mat33d):
    u, sigma, v = signed_svd(f)
    projected, _multiplier = volume_stretches(sigma)
    return u * diagonal_matrix(projected) * wp.transpose(v)


@wp.func
def volume_projection_jvp(f: wp.mat33d, df: wp.mat33d):
    u, sigma, v = signed_svd(f)
    projected, multiplier = volume_stretches(sigma)
    if not wp.isfinite(multiplier):
        return wp.mat33d(wp.float64(wp.nan))
    kkt = volume_kkt(projected, multiplier)
    determinant = wp.determinant(kkt)
    if not wp.isfinite(determinant) or determinant == wp.float64(0.0):
        return wp.mat33d(wp.float64(wp.nan))
    inverse_kkt = wp.inverse(kkt)
    stretch_jacobian = wp.mat33d(wp.float64(0.0))
    for i in range(3):
        for j in range(3):
            stretch_jacobian[i, j] = inverse_kkt[i, j]
    return spectral_projection_jvp(u, sigma, v, projected, stretch_jacobian, df)


@wp.func_grad(volume_projection)
def _volume_projection_grad(f: wp.mat33d, adj_result: wp.mat33d):
    # This projection is the gradient of a scalar envelope, so its local
    # Jacobian is symmetric on a smooth, converged branch.
    wp.adjoint[f] += volume_projection_jvp(f, adj_result)


@wp.func
def neohookean_energy(f: wp.mat33d, mu: wp.float64, lame: wp.float64):
    log_j = wp.log(wp.determinant(f))
    return (
        wp.float64(0.5) * mu * (matrix_inner(f, f) - wp.float64(3.0))
        - mu * log_j
        + wp.float64(0.5) * lame * log_j * log_j
    )


@wp.func
def neohookean_stress(f: wp.mat33d, mu: wp.float64, lame: wp.float64):
    inverse_transpose = wp.transpose(wp.inverse(f))
    log_j = wp.log(wp.determinant(f))
    return mu * f + (lame * log_j - mu) * inverse_transpose


@wp.func
def neohookean_stress_jvp(f: wp.mat33d, df: wp.mat33d, mu: wp.float64, lame: wp.float64):
    inverse_transpose = wp.transpose(wp.inverse(f))
    log_j = wp.log(wp.determinant(f))
    dlog_j = matrix_inner(inverse_transpose, df)
    dinverse_transpose = -inverse_transpose * wp.transpose(df) * inverse_transpose
    return mu * df + lame * dlog_j * inverse_transpose + (lame * log_j - mu) * dinverse_transpose


@wp.func
def nh_stretch_energy(sigma: wp.vec3d, mu: wp.float64, lame: wp.float64):
    log_j = wp.log(sigma[0] * sigma[1] * sigma[2])
    return (
        wp.float64(0.5) * mu * (wp.dot(sigma, sigma) - wp.float64(3.0))
        - mu * log_j
        + wp.float64(0.5) * lame * log_j * log_j
    )


@wp.func
def nh_proximal_hessian(sigma: wp.vec3d, mu: wp.float64, lame: wp.float64, penalty: wp.float64):
    log_j = wp.log(sigma[0] * sigma[1] * sigma[2])
    h = wp.mat33d(wp.float64(0.0))
    for i in range(3):
        for j in range(3):
            h[i, j] = lame / (sigma[i] * sigma[j])
        h[i, i] += mu + penalty + (mu - lame * log_j) / (sigma[i] * sigma[i])
    return h


@wp.func
def nh_proximal_stretches(target: wp.vec3d, mu: wp.float64, lame: wp.float64, penalty: wp.float64):
    sigma = wp.vec3d(wp.float64(0.0))
    for i in range(3):
        sigma[i] = wp.max(target[i], wp.float64(1.0e-5))
    for _iteration in range(80):
        log_j = wp.log(sigma[0] * sigma[1] * sigma[2])
        inverse_sigma = wp.vec3d(wp.float64(1.0) / sigma[0], wp.float64(1.0) / sigma[1], wp.float64(1.0) / sigma[2])
        gradient = mu * sigma + (lame * log_j - mu) * inverse_sigma + penalty * (sigma - target)
        if wp.dot(gradient, gradient) < wp.float64(1.0e-26) * (mu + lame + penalty) ** wp.float64(2.0):
            break
        hessian = nh_proximal_hessian(sigma, mu, lame, penalty)
        step = -wp.inverse(hessian) * gradient
        if wp.dot(step, gradient) >= wp.float64(0.0):
            step = -gradient / (mu + lame + penalty)
        objective = nh_stretch_energy(sigma, mu, lame) + wp.float64(0.5) * penalty * wp.length_sq(sigma - target)
        directional = wp.dot(gradient, step)
        alpha = wp.float64(1.0)
        accepted = bool(False)
        for _line_search in range(40):
            trial = sigma + alpha * step
            if wp.min(trial) > wp.float64(1.0e-12):
                trial_objective = nh_stretch_energy(trial, mu, lame) + wp.float64(0.5) * penalty * wp.length_sq(
                    trial - target
                )
                trial_inverse = wp.vec3d(
                    wp.float64(1.0) / trial[0], wp.float64(1.0) / trial[1], wp.float64(1.0) / trial[2]
                )
                trial_gradient = (
                    mu * trial
                    + (lame * wp.log(trial[0] * trial[1] * trial[2]) - mu) * trial_inverse
                    + penalty * (trial - target)
                )
                # Once energy differences are below floating-point resolution,
                # the residual remains a useful acceptance criterion.
                close_to_root = wp.length_sq(gradient) < wp.float64(1.0e-14) * (mu + lame + penalty) ** wp.float64(2.0)
                improves_residual = wp.length_sq(trial_gradient) < wp.length_sq(gradient)
                if trial_objective <= objective + wp.float64(1.0e-4) * alpha * directional or (
                    close_to_root and improves_residual
                ):
                    sigma = trial
                    accepted = True
                    break
            alpha *= wp.float64(0.5)
        if not accepted:
            break
    inverse_sigma = wp.vec3d(wp.float64(1.0) / sigma[0], wp.float64(1.0) / sigma[1], wp.float64(1.0) / sigma[2])
    residual = (
        mu * sigma + (lame * wp.log(sigma[0] * sigma[1] * sigma[2]) - mu) * inverse_sigma + penalty * (sigma - target)
    )
    if not wp.isfinite(wp.length_sq(residual)) or wp.length_sq(residual) > wp.float64(1.0e-20) * (
        mu + lame + penalty
    ) ** wp.float64(2.0):
        sigma = wp.vec3d(wp.float64(wp.nan))
    return sigma


@wp.func
def neohookean_prox(f: wp.mat33d, mu: wp.float64, lame: wp.float64, penalty: wp.float64):
    u, sigma, v = signed_svd(f)
    projected = nh_proximal_stretches(sigma, mu, lame, penalty)
    return u * diagonal_matrix(projected) * wp.transpose(v)


@wp.func
def neohookean_prox_jvp(f: wp.mat33d, df: wp.mat33d, mu: wp.float64, lame: wp.float64, penalty: wp.float64):
    u, sigma, v = signed_svd(f)
    projected = nh_proximal_stretches(sigma, mu, lame, penalty)
    stretch_jacobian = penalty * wp.inverse(nh_proximal_hessian(projected, mu, lame, penalty))
    return spectral_projection_jvp(u, sigma, v, projected, stretch_jacobian, df)


@wp.func_grad(neohookean_prox)
def _neohookean_prox_grad(f: wp.mat33d, mu: wp.float64, lame: wp.float64, penalty: wp.float64, adj_result: wp.mat33d):
    u, sigma, v = signed_svd(f)
    projected = nh_proximal_stretches(sigma, mu, lame, penalty)
    inverse_hessian = wp.inverse(nh_proximal_hessian(projected, mu, lame, penalty))
    local_adj = wp.transpose(u) * adj_result * v
    stretch_adj = inverse_hessian * wp.vec3d(local_adj[0, 0], local_adj[1, 1], local_adj[2, 2])
    inverse_projected = wp.vec3d(
        wp.float64(1.0) / projected[0], wp.float64(1.0) / projected[1], wp.float64(1.0) / projected[2]
    )
    wp.adjoint[f] += spectral_projection_jvp(u, sigma, v, projected, penalty * inverse_hessian, adj_result)
    wp.adjoint[mu] -= wp.dot(stretch_adj, projected - inverse_projected)
    wp.adjoint[lame] -= wp.log(projected[0] * projected[1] * projected[2]) * wp.dot(stretch_adj, inverse_projected)
    wp.adjoint[penalty] -= wp.dot(stretch_adj, projected - sigma)


@wp.func
def elastic_energy(f: wp.mat33d, mu: wp.float64, lame: wp.float64, model: int):
    if model == MODEL_NEO_HOOKEAN:
        return neohookean_energy(f, mu, lame)
    dr = f - rotation_projection(f)
    dv = f - volume_projection(f)
    return mu * matrix_inner(dr, dr) + wp.float64(0.5) * lame * matrix_inner(dv, dv)


@wp.func
def elastic_stress(f: wp.mat33d, mu: wp.float64, lame: wp.float64, model: int):
    if model == MODEL_NEO_HOOKEAN:
        return neohookean_stress(f, mu, lame)
    return wp.float64(2.0) * mu * (f - rotation_projection(f)) + lame * (f - volume_projection(f))


@wp.func
def elastic_stress_jvp(f: wp.mat33d, df: wp.mat33d, mu: wp.float64, lame: wp.float64, model: int):
    if model == MODEL_NEO_HOOKEAN:
        return neohookean_stress_jvp(f, df, mu, lame)
    return wp.float64(2.0) * mu * (df - rotation_projection_jvp(f, df)) + lame * (df - volume_projection_jvp(f, df))


@wp.func
def elastic_stress_parameters(f: wp.mat33d, model: int):
    if model == MODEL_NEO_HOOKEAN:
        inverse_transpose = wp.transpose(wp.inverse(f))
        return f - inverse_transpose, wp.log(wp.determinant(f)) * inverse_transpose
    return wp.float64(2.0) * (f - rotation_projection(f)), f - volume_projection(f)


@wp.func
def tet_deformation(q: wp.array[wp.vec3d], indices: wp.array[wp.vec4i], gradients: wp.array2d[wp.vec3d], element: int):
    tet = indices[element]
    origin = q[tet[0]]
    f = wp.mat33d(wp.float64(0.0))
    for node in range(1, 4):
        f += wp.outer(q[tet[node]] - origin, gradients[element, node])
    return f


@wp.kernel
def material_vjp(
    q: wp.array[wp.vec3d],
    adjoint: wp.array[wp.vec3d],
    indices: wp.array[wp.vec4i],
    gradients: wp.array2d[wp.vec3d],
    volumes: wp.array[wp.float64],
    model: int,
    out_mu: wp.array[wp.float64],
    out_lambda: wp.array[wp.float64],
):
    """Write adjoint contraction with d(positive elastic residual)/d(material)."""
    element = wp.tid()
    f = tet_deformation(q, indices, gradients, element)
    adjoint_f = tet_deformation(adjoint, indices, gradients, element)
    dmu, dlambda = elastic_stress_parameters(f, model)
    out_mu[element] = volumes[element] * matrix_inner(adjoint_f, dmu)
    out_lambda[element] = volumes[element] * matrix_inner(adjoint_f, dlambda)


@wp.kernel(enable_backward=False)
def compute_tet_tangent(
    q: wp.array[wp.vec3d],
    indices: wp.array[wp.vec4i],
    gradients: wp.array2d[wp.vec3d],
    mu: wp.array[wp.float64],
    lame: wp.array[wp.float64],
    model: int,
    tangent: wp.array2d[wp.mat33d],
):
    """Cache exact stress-Jacobian columns at the fixed Newton/adjoint state."""
    element, column = wp.tid()
    f = tet_deformation(q, indices, gradients, element)
    basis = wp.mat33d(wp.float64(0.0))
    basis[column // 3, column % 3] = wp.float64(1.0)
    tangent[element, column] = elastic_stress_jvp(f, basis, mu[element], lame[element], model)


@wp.kernel(enable_backward=False)
def apply_cached_tet_tangent(
    delta: wp.array[wp.vec3d],
    indices: wp.array[wp.vec4i],
    gradients: wp.array2d[wp.vec3d],
    volumes: wp.array[wp.float64],
    tangent: wp.array2d[wp.mat33d],
    out: wp.array[wp.vec3d],
):
    element = wp.tid()
    df = tet_deformation(delta, indices, gradients, element)
    stress = wp.mat33d(wp.float64(0.0))
    for column in range(9):
        stress += df[column // 3, column % 3] * tangent[element, column]
    for node in range(4):
        wp.atomic_add(out, indices[element][node], volumes[element] * stress * gradients[element, node])
