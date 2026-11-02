# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Portions translated from DiffPhD (https://github.com/Akinesia112/DiffPhD,
# commit d319435), Copyright (c) 2021 MIT Graphics Group, MIT License.
# See THIRD_PARTY_NOTICES.md.

"""Reference DiffPhD backward pass for one recorded step.

This follows ``BackwardProjectiveDynamicsAlgPhd``
(``cpp/core/src/fem/deformable_projective_dynamics_backward_gpu.cpp``) and the
local-step differential of ``deformable_projective_dynamics_backward.cpp``:

* ``rhs = dL/dq+ + dL/dv+ / h`` and ``dL/dq -= dL/dv+ / h``;
* the backbone adjoint ``(A - K) mu = rhs`` is solved by the fixed point
  ``mu <- A^{-1}(rhs + K mu)`` with Anderson mixing, Dirichlet entries zeroed
  after every solve, and the stopping rule ``||raw - mu_prev|| < rel ||mu|| + abs``;
  ``K = sum_e G_e^T (mean(w) Vbar dP/dF)(x+) G_e`` uses the reference's
  mean weight per energy, while ``A`` retains per-element weights;
* state gradients come from ``mu`` only, without contact:
  ``dL/dq += m/h^2 mu``, ``dL/dv += h m/h^2 mu``, ``dL/df_ext += mu``;
* active contacts (``lambda > 0`` and ``omega >= 0.5``) enter through the
  Woodbury projection ``z = mu - W_T G^{-1} (J mu)``, with normal rows
  ``omega n`` and, with friction, tangent rows; rows with
  ``|J mu| < max(1e-12, 1e-3 max|J mu|)`` are dropped and every column of
  ``W_T = (A - K)^{-1} J^T`` is a fixed point of at most
  ``min(max_iterations, 120)`` iterations;
* material and activation gradients use ``z``: ``dL/dw_e = -z^T Vbar G_e^T
  (F - P(F))`` per projective energy (the true Neo-Hookean stress divided by
  ``2 mean(mu)`` for the Neo-Hookean energy) and
  ``dL/da_e = z^T Vbar k/l G_e^T (F m m^T)`` for fibres.

Added to the reference: Dirichlet values receive ``dL/dx_D = g_D - (A - K)_DF mu_F``,
chained to the start positions (and ``dt`` times the velocities when integrated).

The reference's nonfinite scrubbing, norm clipping, cascade rescaling and
suppression of unreliable contributions are reproduced. Status bits report
non-convergence and safeguard use. For heterogeneous materials and Neo-Hookean
material parameters these are reference algorithm outputs, not exact forward
derivatives. Homogeneous state derivatives away from repeated singular values
are exact when no contact or safeguard changes the derivative.
"""

import gc

import warp as wp

from .anderson import CHUNK, Anderson
from .dense import ldlt_solve
from .kernels import gather, gather_batch
from .projections import (
    DOUBLE_EPSILON,
    corotated_jvp,
    corotated_projection,
    diagonal_matrix,
    muscle_jvp,
    neo_hookean_stress,
    nh_prox_jvp,
    nh_prox_stretches,
    unsigned_svd,
    volume_jvp,
    volume_offset,
)

STATUS_ADJOINT_NOT_CONVERGED = wp.constant(8)
STATUS_ADJOINT_SAFEGUARD = wp.constant(16)
ADJOINT_CLIP_NORM = 1.0e6
WOODBURY_MAX_ITERATIONS = 120


@wp.kernel(enable_backward=False)
def _seed(
    q_grad: wp.array[wp.vec3],
    qd_grad: wp.array[wp.vec3],
    dt: wp.float64,
    rhs: wp.array[wp.vec3d],
    base: wp.array[wp.vec3d],
):
    i = wp.tid()
    velocity = wp.vec3d(qd_grad[i]) / dt
    rhs[i] = wp.vec3d(q_grad[i]) + velocity
    base[i] = -velocity


@wp.func
def _deformation(x: wp.array[wp.vec3d], tet: wp.vec4i, e: int, gradients: wp.array2d[wp.vec3d]):
    f = wp.mat33d(wp.float64(0.0))
    for k in range(4):
        f += wp.outer(x[tet[k]], gradients[e, k])
    return f


@wp.kernel(enable_backward=False)
def _local_jacobian(
    x: wp.array[wp.vec3d],
    indices: wp.array[wp.vec4i],
    gradients: wp.array2d[wp.vec3d],
    material: int,
    corotated_weight: wp.float64,
    volume_weight: wp.float64,
    neo_hookean_weight: wp.float64,
    nh_mu: wp.float64,
    nh_lame: wp.float64,
    nh_penalty: wp.float64,
    muscle_direction: wp.array[wp.vec3d],
    muscle_weight: wp.array[wp.float64],
    muscle_target: wp.array[wp.float64],
    jacobian: wp.array2d[wp.mat33d],
):
    e = wp.tid()
    f = _deformation(x, indices[e], e, gradients)
    u, sigma, v = unsigned_svd(f)
    offset = wp.vec3d(wp.float64(0.0))
    stretches = wp.vec3d(wp.float64(0.0))
    if material == 0:
        offset, _singular = volume_offset(sigma)
    else:
        stretches = nh_prox_stretches(sigma, nh_mu, nh_lame, nh_penalty)
    wm = muscle_weight[e]
    m = muscle_direction[e]
    wc = corotated_weight
    wv = volume_weight
    wn = neo_hookean_weight
    for c in range(9):
        # Direction e_b e_n^T; c = b + 3 n is Eigen's column-major Flatten index.
        df = wp.mat33d(wp.float64(0.0))
        df[c % 3, c // 3] = wp.float64(1.0)
        dp = wp.mat33d(wp.float64(0.0))
        if material == 0:
            if wc != wp.float64(0.0):
                dp += wc * corotated_jvp(u, sigma, v, df)
            if wv != wp.float64(0.0):
                dp += wv * volume_jvp(u, sigma, v, offset, df)
        else:
            dp += wn * nh_prox_jvp(u, sigma, v, stretches, df, nh_mu, nh_lame, nh_penalty)
        if wm != wp.float64(0.0):
            dp += wm * wp.outer(muscle_jvp(f, m, muscle_target[e], df), m)
        jacobian[e, c] = dp


@wp.kernel(enable_backward=False)
def _apply_local(
    dx: wp.array[wp.vec3d],
    indices: wp.array[wp.vec4i],
    gradients: wp.array2d[wp.vec3d],
    jacobian: wp.array2d[wp.mat33d],
    contributions: wp.array2d[wp.vec3d],
):
    e = wp.tid()
    tet = indices[e]
    df = _deformation(dx, tet, e, gradients)
    ds = wp.mat33d(wp.float64(0.0))
    for c in range(9):
        ds += df[c % 3, c // 3] * jacobian[e, c]
    for k in range(4):
        contributions[e, k] = ds * gradients[e, k]


@wp.kernel(enable_backward=False)
def _set_limit(
    limit: wp.array[wp.int32],
    value: int,
    report: int,
    iteration: wp.array[wp.int32],
    active: wp.array[wp.int32],
):
    limit[0] = value
    limit[1] = report
    iteration[0] = 0
    active[0] = 1


@wp.kernel(enable_backward=False)
def _begin_iteration(
    solution: wp.array[wp.vec3d],
    rhs: wp.array[wp.vec3d],
    previous: wp.array[wp.vec3d],
    work: wp.array[wp.vec3d],
):
    i = wp.tid()
    previous[i] = solution[i]
    work[i] = rhs[i]


@wp.kernel(enable_backward=False)
def _clip_partials(
    values: wp.array2d[wp.vec3d],
    free: wp.array[wp.int32],
    size: int,
    partials: wp.array2d[wp.float64],
    status: wp.array[wp.int32],
):
    col, chunk = wp.tid()
    total = wp.float64(0.0)
    for i in range(chunk * CHUNK, wp.min(size, (chunk + 1) * CHUNK)):
        value = values[col, i]
        if free[i] == 0:
            value = wp.vec3d(wp.float64(0.0))
        for axis in range(3):
            if not wp.isfinite(value[axis]):
                value[axis] = wp.float64(0.0)
                wp.atomic_or(status, 0, STATUS_ADJOINT_SAFEGUARD)
        values[col, i] = value
        total += wp.dot(value, value)
    partials[col, chunk] = total


@wp.kernel(enable_backward=False)
def _clip_apply(
    partials: wp.array2d[wp.float64],
    chunks: int,
    limit: wp.float64,
    values: wp.array2d[wp.vec3d],
    status: wp.array[wp.int32],
):
    col, i = wp.tid()
    total = wp.float64(0.0)
    for chunk in range(chunks):
        total += partials[col, chunk]
    length = wp.sqrt(total)
    if limit > wp.float64(0.0) and wp.isfinite(length) and length > limit:
        values[col, i] = (limit / length) * values[col, i]
        if i == 0:
            wp.atomic_or(status, 0, STATUS_ADJOINT_SAFEGUARD)


@wp.kernel(enable_backward=False)
def _routing_partials(rhs: wp.array[wp.vec3d], mu: wp.array[wp.vec3d], size: int, partials: wp.array[wp.vec2d]):
    chunk = wp.tid()
    total = wp.vec2d(wp.float64(0.0))
    for i in range(chunk * CHUNK, wp.min(size, (chunk + 1) * CHUNK)):
        total += wp.vec2d(wp.dot(rhs[i], rhs[i]), wp.dot(mu[i], mu[i]))
    partials[chunk] = total


@wp.kernel(enable_backward=False)
def _routing_guard(
    partials: wp.array[wp.vec2d],
    chunks: int,
    inertia: wp.float64,
    clip_norm: wp.float64,
    scale: wp.array[wp.float64],
    diverged: wp.array[wp.int32],
    status: wp.array[wp.int32],
):
    total = wp.vec2d(wp.float64(0.0))
    for chunk in range(chunks):
        total += partials[chunk]
    mu_norm = wp.sqrt(total[1])
    failed = clip_norm > wp.float64(0.0) and mu_norm >= wp.float64(0.999) * clip_norm
    diverged[0] = wp.where(failed, 1, 0)
    value = wp.float64(1.0)
    if failed:
        value = wp.float64(0.0)
        wp.atomic_or(status, 0, STATUS_ADJOINT_SAFEGUARD)
    else:
        cap = wp.sqrt(total[0]) + wp.float64(1.0)
        injection_norm = inertia * mu_norm
        if injection_norm > cap:
            value = cap / injection_norm
            wp.atomic_or(status, 0, STATUS_ADJOINT_SAFEGUARD)
    scale[0] = value


@wp.kernel(enable_backward=False)
def _step_partials(
    free: wp.array[wp.int32],
    previous: wp.array[wp.vec3d],
    size: int,
    raw: wp.array[wp.vec3d],
    partials: wp.array[wp.vec2d],
):
    chunk = wp.tid()
    total = wp.float64(0.0)
    for i in range(chunk * CHUNK, wp.min(size, (chunk + 1) * CHUNK)):
        if free[i] == 0:
            raw[i] = wp.vec3d(wp.float64(0.0))
        step = raw[i] - previous[i]
        total += wp.dot(step, step)
    partials[chunk] = wp.vec2d(total, wp.float64(0.0))


@wp.kernel(enable_backward=False)
def _finish_iteration(
    raw: wp.array[wp.vec3d],
    size: int,
    solution: wp.array[wp.vec3d],
    partials: wp.array[wp.vec2d],
):
    chunk = wp.tid()
    total = wp.float64(0.0)
    for i in range(chunk * CHUNK, wp.min(size, (chunk + 1) * CHUNK)):
        solution[i] = raw[i]
        total += wp.dot(raw[i], raw[i])
    partials[chunk] = wp.vec2d(partials[chunk][0], total)


@wp.kernel(enable_backward=False)
def _check(
    partials: wp.array[wp.vec2d],
    chunks: int,
    abs_tol: wp.float64,
    rel_tol: wp.float64,
    limit: wp.array[wp.int32],
    iteration: wp.array[wp.int32],
    active: wp.array[wp.int32],
    status: wp.array[wp.int32],
):
    total = wp.vec2d(wp.float64(0.0))
    for c in range(chunks):
        total += partials[c]
    k = iteration[0]
    iteration[0] = k + 1
    if k > 0 and wp.sqrt(total[0]) < rel_tol * wp.sqrt(total[1]) + abs_tol:
        active[0] = 0
    elif k + 1 >= limit[0]:
        active[0] = 0
        # The Woodbury columns are truncated by design; only the backbone reports.
        if limit[1] != 0:
            status[0] = status[0] | STATUS_ADJOINT_NOT_CONVERGED
    else:
        active[0] = 1


@wp.func
def _backward_tangents(n: wp.vec3d):
    axis = int(0)
    for d in range(1, 3):
        if wp.abs(n[d]) < wp.abs(n[axis]):
            axis = d
    unit = wp.vec3d(wp.float64(0.0))
    unit[axis] = wp.float64(1.0)
    t1 = unit - wp.dot(n, unit) * n
    length = wp.length(t1)
    if length > wp.float64(1.0e-12):
        t1 = t1 / length
    else:
        t1 = wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0))
    t2 = wp.cross(n, t1)
    length = wp.length(t2)
    if length > wp.float64(1.0e-12):
        t2 = t2 / length
    else:
        t2 = wp.vec3d(wp.float64(0.0), wp.float64(1.0), wp.float64(0.0))
    return t1, t2


@wp.kernel(enable_backward=False)
def _basis(
    nodes: wp.array[wp.int32],
    multiplier: wp.array[wp.float64],
    omega: wp.array[wp.float64],
    normal: wp.array[wp.vec3d],
    tangents: int,
    count: int,
    diverged: wp.array[wp.int32],
    row_node: wp.array[wp.int32],
    row_vector: wp.array[wp.vec3d],
    row_valid: wp.array[wp.int32],
):
    c = wp.tid()
    node = nodes[c]
    active = diverged[0] == 0 and multiplier[c] > wp.float64(0.0) and omega[c] >= wp.float64(0.5)
    row_node[c] = node
    row_vector[c] = omega[c] * normal[c]
    row_valid[c] = wp.where(active, 1, 0)
    if tangents != 0:
        t1, t2 = _backward_tangents(normal[c])
        valid = wp.where(active and multiplier[c] > wp.float64(1.0e-8), 1, 0)
        for r in range(2):
            row = count + 2 * c + r
            row_node[row] = node
            row_vector[row] = wp.where(r == 0, t1, t2)
            row_valid[row] = valid


@wp.kernel(enable_backward=False)
def _project_rows(
    z: wp.array[wp.vec3d],
    row_node: wp.array[wp.int32],
    row_vector: wp.array[wp.vec3d],
    row_valid: wp.array[wp.int32],
    value: wp.array[wp.float64],
):
    r = wp.tid()
    value[r] = wp.float64(0.0)
    if row_valid[r] != 0:
        value[r] = wp.dot(row_vector[r], z[row_node[r]])


@wp.kernel(enable_backward=False)
def _select_rows(
    value: wp.array[wp.float64],
    row_valid: wp.array[wp.int32],
    rows: int,
    selected: wp.array[wp.int32],
):
    largest = wp.float64(0.0)
    for r in range(rows):
        if row_valid[r] != 0:
            largest = wp.max(largest, wp.abs(value[r]))
    gate = wp.max(wp.float64(1.0e-12), wp.float64(1.0e-3) * largest)
    for r in range(rows):
        selected[r] = wp.where(row_valid[r] != 0 and wp.abs(value[r]) >= gate, 1, 0)


@wp.kernel(enable_backward=False)
def _compact_rows(selected: wp.array[wp.int32], rows: int, order: wp.array[wp.int32], count: wp.array[wp.int32]):
    k = int(0)
    for r in range(rows):
        if selected[r] != 0:
            order[k] = r
            k += 1
    count[0] = k


@wp.kernel(enable_backward=False)
def _gram(
    columns: wp.array2d[wp.vec3d],
    row_node: wp.array[wp.int32],
    row_vector: wp.array[wp.vec3d],
    order: wp.array[wp.int32],
    count: wp.array[wp.int32],
    value: wp.array[wp.float64],
    matrix: wp.array2d[wp.float64],
    rhs: wp.array[wp.float64],
):
    i, j = wp.tid()
    n = count[0]
    if i >= n or j >= n:
        return
    # G_T(l, k) = J_l W_T[k] over the selected rows in their original order;
    # the LDLT of the reference reads the lower triangle.
    high = order[wp.max(i, j)]
    low = order[wp.min(i, j)]
    matrix[i, j] = wp.dot(row_vector[high], columns[low, row_node[high]])
    if j == 0:
        rhs[i] = value[order[i]]


@wp.kernel(enable_backward=False)
def _zero(values: wp.array[wp.float64]):
    values[wp.tid()] = wp.float64(0.0)


@wp.kernel(enable_backward=False)
def _scatter_alpha(
    order: wp.array[wp.int32],
    count: wp.array[wp.int32],
    coefficients: wp.array[wp.float64],
    alpha: wp.array[wp.float64],
):
    r = wp.tid()
    if r < count[0]:
        alpha[order[r]] = coefficients[r]


@wp.kernel(enable_backward=False)
def _project_adjoint(
    mu: wp.array[wp.vec3d],
    columns: wp.array2d[wp.vec3d],
    alpha: wp.array[wp.float64],
    selected: wp.array[wp.int32],
    rows: int,
    z: wp.array[wp.vec3d],
):
    i = wp.tid()
    value = mu[i]
    for r in range(rows):
        if selected[r] != 0:
            value -= alpha[r] * columns[r, i]
    z[i] = value


@wp.kernel(enable_backward=False)
def _state_gradients(
    base: wp.array[wp.vec3d],
    mu: wp.array[wp.vec3d],
    inertia: wp.float64,
    dt: wp.float64,
    scale: wp.array[wp.float64],
    q_grad: wp.array[wp.vec3],
    qd_grad: wp.array[wp.vec3],
    f_grad: wp.array[wp.vec3],
):
    i = wp.tid()
    injection = scale[0] * inertia * mu[i]
    q_grad[i] = q_grad[i] + wp.vec3(base[i] + injection)
    qd_grad[i] = qd_grad[i] + wp.vec3(dt * injection)
    # dL/df_ext = h^2/m * m/h^2 mu.
    f_grad[i] = f_grad[i] + wp.vec3(scale[0] * mu[i])


@wp.kernel(enable_backward=False)
def _dirichlet_coupling(
    mu: wp.array[wp.vec3d],
    indices: wp.array[wp.vec4i],
    gradients: wp.array2d[wp.vec3d],
    jacobian: wp.array2d[wp.mat33d],
    corotated_weight: wp.array[wp.float64],
    volume_weight: wp.array[wp.float64],
    neo_hookean_weight: wp.array[wp.float64],
    muscle_direction: wp.array[wp.vec3d],
    muscle_weight: wp.array[wp.float64],
    diverged: wp.array[wp.int32],
    contributions: wp.array2d[wp.vec3d],
):
    # Element rows of (A - K) mu without the mass term, which no Dirichlet row couples to.
    e = wp.tid()
    if diverged[0] != 0:
        for k in range(4):
            contributions[e, k] = wp.vec3d(wp.float64(0.0))
        return
    tet = indices[e]
    df = _deformation(mu, tet, e, gradients)
    m = muscle_direction[e]
    ds = (corotated_weight[e] + volume_weight[e] + neo_hookean_weight[e]) * df
    ds += muscle_weight[e] * wp.outer(df * m, m)
    for c in range(9):
        ds -= df[c % 3, c // 3] * jacobian[e, c]
    for k in range(4):
        contributions[e, k] = ds * gradients[e, k]


@wp.kernel(enable_backward=False)
def _dirichlet_gradients(
    free: wp.array[wp.int32],
    rhs: wp.array[wp.vec3d],
    coupling: wp.array[wp.vec3d],
    dt: wp.float64,
    integrate: int,
    q_grad: wp.array[wp.vec3],
    qd_grad: wp.array[wp.vec3],
):
    i = wp.tid()
    if free[i] != 0:
        return
    # The free rows solve R_F(x_F, x_D) = 0, so dL/dx_D = g_D - (A - K)_DF mu_F.
    value = rhs[i] - coupling[i]
    q_grad[i] = q_grad[i] + wp.vec3(value)
    if integrate != 0:
        qd_grad[i] = qd_grad[i] + wp.vec3(dt * value)


@wp.kernel(enable_backward=False)
def _explicit_dirichlet_gradients(
    free: wp.array[wp.int32],
    rhs: wp.array[wp.vec3d],
    coupling: wp.array[wp.vec3d],
    target_grad: wp.array[wp.vec3],
):
    i = wp.tid()
    if free[i] == 0:
        target_grad[i] = target_grad[i] + wp.vec3(rhs[i] - coupling[i])


@wp.kernel(enable_backward=False)
def _material_gradients(
    x: wp.array[wp.vec3d],
    z: wp.array[wp.vec3d],
    indices: wp.array[wp.vec4i],
    gradients: wp.array2d[wp.vec3d],
    material: int,
    volume: wp.float64,
    nh_mu: wp.float64,
    nh_lame: wp.float64,
    grad: wp.array2d[wp.float64],
):
    e = wp.tid()
    tet = indices[e]
    f = _deformation(x, tet, e, gradients)
    first = wp.mat33d(wp.float64(0.0))
    second = wp.mat33d(wp.float64(0.0))
    scale = wp.float64(2.0)
    if material == 0:
        u, sigma, v = unsigned_svd(f)
        offset, _singular = volume_offset(sigma)
        first = f - corotated_projection(u, v)
        second = f - u * diagonal_matrix(sigma + offset) * wp.transpose(v)
    else:
        # StressTensor of the reference Neo-Hookean energy divided by its mean weight 2 mean(mu).
        first = neo_hookean_stress(f, nh_mu, nh_lame) / (wp.float64(2.0) * nh_mu)
    first_value = wp.float64(0.0)
    second_value = wp.float64(0.0)
    for k in range(4):
        first_value -= wp.dot(z[tet[k]], first * gradients[e, k])
        second_value -= wp.dot(z[tet[k]], second * gradients[e, k])
    # Corotated (or Neo-Hookean) weight 2 mu_e, volume weight lambda_e.
    grad[e, 0] = scale * volume * first_value
    grad[e, 1] = wp.float64(0.0)
    if material == 0:
        grad[e, 1] = volume * second_value


@wp.kernel(enable_backward=False)
def _activation_gradients(
    x: wp.array[wp.vec3d],
    z: wp.array[wp.vec3d],
    indices: wp.array[wp.vec4i],
    gradients: wp.array2d[wp.vec3d],
    muscle_direction: wp.array[wp.vec3d],
    muscle_weight: wp.array[wp.float64],
    grad: wp.array[wp.float64],
):
    e = wp.tid()
    grad[e] = wp.float64(0.0)
    wm = muscle_weight[e]
    if wm == wp.float64(0.0):
        return
    tet = indices[e]
    m = muscle_direction[e]
    fibre = _deformation(x, tet, e, gradients) * m
    length = wp.length(fibre)
    if length <= DOUBLE_EPSILON:
        return
    value = wp.float64(0.0)
    for k in range(4):
        value += wp.dot(z[tet[k]], wm / length * wp.dot(m, gradients[e, k]) * fibre)
    # The activation is a contraction u with target length a = 1 - u.
    grad[e] = -value


@wp.kernel(enable_backward=False)
def _parameter_guard(
    materials: wp.array2d[wp.float64],
    activations: wp.array[wp.float64],
    count: int,
    zero_energy_weight: bool,
    diverged: wp.array[wp.int32],
    valid: wp.array[wp.int32],
    status: wp.array[wp.int32],
):
    sums = wp.vec2d(wp.float64(0.0))
    activation_ok = bool(True)
    for e in range(count):
        sums += wp.vec2d(materials[e, 0], materials[e, 1])
        activation_ok = activation_ok and wp.isfinite(activations[e])
    # The reference forms StressTensor / energy->stiffness() literally.
    # A zero PD volume weight gives 0/0, so its aggregate guard suppresses
    # the whole material vector, including the corotated energy entry.
    material_ok = not zero_energy_weight and wp.isfinite(sums[0]) and wp.isfinite(sums[1])
    valid[0] = wp.where(diverged[0] == 0 and material_ok, 1, 0)
    valid[1] = wp.where(diverged[0] == 0 and activation_ok, 1, 0)
    if not material_ok or not activation_ok:
        wp.atomic_or(status, 0, STATUS_ADJOINT_SAFEGUARD)


@wp.kernel(enable_backward=False)
def _add_material_gradient(values: wp.array2d[wp.float64], valid: wp.array[wp.int32], grad: wp.array2d[wp.float32]):
    e = wp.tid()
    if valid[0] != 0:
        grad[e, 0] += wp.float32(values[e, 0])
        grad[e, 1] += wp.float32(values[e, 1])


@wp.kernel(enable_backward=False)
def _add_activation_gradient(values: wp.array[wp.float64], valid: wp.array[wp.int32], grad: wp.array[wp.float32]):
    e = wp.tid()
    if valid[1] != 0:
        grad[e] += wp.float32(values[e])


# Woodbury columns solved together in one batched fixed point.
WOODBURY_BATCH = 64


@wp.kernel(enable_backward=False)
def _batch_setup(
    order: wp.array[wp.int32],
    count: wp.array[wp.int32],
    first: int,
    row_node: wp.array[wp.int32],
    row_vector: wp.array[wp.vec3d],
    rows: wp.array[wp.int32],
    enabled: wp.array[wp.int32],
    rhs: wp.array2d[wp.vec3d],
    solution: wp.array2d[wp.vec3d],
):
    col, i = wp.tid()
    k = first + col
    active = k < count[0]
    value = wp.vec3d(wp.float64(0.0))
    if active and i == row_node[order[k]]:
        value = row_vector[order[k]]
    rhs[col, i] = value
    solution[col, i] = wp.vec3d(wp.float64(0.0))
    if i == 0:
        enabled[col] = wp.where(active, 1, 0)
        rows[col] = wp.where(active, order[k], -1)


@wp.kernel(enable_backward=False)
def _batch_begin(
    solution: wp.array2d[wp.vec3d],
    rhs: wp.array2d[wp.vec3d],
    enabled: wp.array[wp.int32],
    previous: wp.array2d[wp.vec3d],
    work: wp.array2d[wp.vec3d],
):
    col, i = wp.tid()
    if enabled[col] != 0:
        previous[col, i] = solution[col, i]
        work[col, i] = rhs[col, i]


@wp.kernel(enable_backward=False)
def _batch_apply_local(
    dx: wp.array2d[wp.vec3d],
    enabled: wp.array[wp.int32],
    indices: wp.array[wp.vec4i],
    gradients: wp.array2d[wp.vec3d],
    jacobian: wp.array2d[wp.mat33d],
    contributions: wp.array3d[wp.vec3d],
):
    col, e = wp.tid()
    if enabled[col] == 0:
        return
    tet = indices[e]
    df = wp.mat33d(wp.float64(0.0))
    for k in range(4):
        df += wp.outer(dx[col, tet[k]], gradients[e, k])
    ds = wp.mat33d(wp.float64(0.0))
    for c in range(9):
        ds += df[c % 3, c // 3] * jacobian[e, c]
    for k in range(4):
        contributions[col, e, k] = ds * gradients[e, k]


@wp.kernel(enable_backward=False)
def _batch_step_partials(
    free: wp.array[wp.int32],
    previous: wp.array2d[wp.vec3d],
    enabled: wp.array[wp.int32],
    size: int,
    raw: wp.array2d[wp.vec3d],
    partials: wp.array2d[wp.vec2d],
):
    col, chunk = wp.tid()
    if enabled[col] == 0:
        return
    total = wp.float64(0.0)
    for i in range(chunk * CHUNK, wp.min(size, (chunk + 1) * CHUNK)):
        if free[i] == 0:
            raw[col, i] = wp.vec3d(wp.float64(0.0))
        step = raw[col, i] - previous[col, i]
        total += wp.dot(step, step)
    partials[col, chunk] = wp.vec2d(total, wp.float64(0.0))


@wp.kernel(enable_backward=False)
def _batch_finish(
    raw: wp.array2d[wp.vec3d],
    enabled: wp.array[wp.int32],
    size: int,
    solution: wp.array2d[wp.vec3d],
    partials: wp.array2d[wp.vec2d],
):
    col, chunk = wp.tid()
    if enabled[col] == 0:
        return
    total = wp.float64(0.0)
    for i in range(chunk * CHUNK, wp.min(size, (chunk + 1) * CHUNK)):
        solution[col, i] = raw[col, i]
        total += wp.dot(raw[col, i], raw[col, i])
    partials[col, chunk] = wp.vec2d(partials[col, chunk][0], total)


@wp.kernel(enable_backward=False)
def _batch_check(
    partials: wp.array2d[wp.vec2d],
    chunks: int,
    columns: int,
    abs_tol: wp.float64,
    rel_tol: wp.float64,
    limit: int,
    iteration: wp.array[wp.int32],
    enabled: wp.array[wp.int32],
    active: wp.array[wp.int32],
):
    # Every column stops on its own rule, as the reference's separate solves do.
    k = iteration[0]
    remaining = int(0)
    for col in range(columns):
        if enabled[col] != 0:
            total = wp.vec2d(wp.float64(0.0))
            for c in range(chunks):
                total += partials[col, c]
            if k > 0 and wp.sqrt(total[0]) < rel_tol * wp.sqrt(total[1]) + abs_tol:
                enabled[col] = 0
            else:
                remaining += 1
    iteration[0] = k + 1
    active[0] = wp.where(remaining > 0 and k + 1 < limit, 1, 0)


@wp.kernel(enable_backward=False)
def _batch_start(iteration: wp.array[wp.int32], active: wp.array[wp.int32]):
    iteration[0] = 0
    active[0] = 1


@wp.kernel(enable_backward=False)
def _batch_store(solution: wp.array2d[wp.vec3d], rows: wp.array[wp.int32], columns: wp.array2d[wp.vec3d]):
    col, i = wp.tid()
    row = rows[col]
    if row >= 0:
        columns[row, i] = solution[col, i]


class StepRecord:
    """Owned forward data of one recorded step."""

    def __init__(self, solver, dt, dirichlet_positions=None):
        self.dt = dt
        self.inertia = solver.particle_mass / dt**2
        self.inverse = solver.inverse
        self.free = solver._free
        self.x = wp.clone(solver._x)
        self.muscle_target = wp.clone(solver._muscle_target)
        self.activations = solver._activation_source if solver._has_muscles else None
        self.material_code = solver._material_code
        self.dirichlet = solver._has_dirichlet
        self.integrate = solver.integrate_pinned_velocities
        # Keep the actual input alive for gradient routing; no target values are
        # reread during backward, since x and the factorization are recorded.
        self.dirichlet_positions = dirichlet_positions
        self.weights = (solver._corotated_weight, solver._volume_weight, solver._neo_hookean_weight)
        volume = solver.mean_volume
        self.jacobian_weights = (
            2.0 * solver._nh_mu * volume if solver._material_code == 0 else 0.0,
            solver._nh_lame * volume if solver._material_code == 0 else 0.0,
            2.0 * solver._nh_mu * volume if solver._material_code == 1 else 0.0,
        )
        self.nh = (solver._nh_mu, solver._nh_lame, solver._nh_penalty)
        self.contact = None
        contact = solver._contact
        if contact is not None and contact.count:
            self.contact = (
                wp.clone(contact.multiplier),
                wp.clone(contact.omega),
                wp.clone(contact.normal),
            )


class Adjoint:
    """Reusable buffers and loops of the reference backward pass for one solver."""

    def __init__(self, solver):
        self.solver = solver
        device = solver.device
        n = solver._n
        self.chunks = max(1, (n + CHUNK - 1) // CHUNK)
        for name in ("solution", "previous", "raw", "work", "rhs", "base", "mu", "z", "x", "coupling"):
            setattr(self, f"_{name}", wp.zeros(n, dtype=wp.vec3d, device=device))
        self._partials = wp.zeros(self.chunks, dtype=wp.vec2d, device=device)
        self._clip_partials = wp.zeros((1, self.chunks), dtype=wp.float64, device=device)
        self._routing_partials = wp.zeros(self.chunks, dtype=wp.vec2d, device=device)
        self._routing_scale = wp.ones(1, dtype=wp.float64, device=device)
        self._diverged = wp.zeros(1, dtype=wp.int32, device=device)
        self._material_gradient = wp.zeros((len(solver._tets), 2), dtype=wp.float64, device=device)
        self._activation_gradient = wp.zeros(len(solver._tets), dtype=wp.float64, device=device)
        self._parameter_valid = wp.zeros(2, dtype=wp.int32, device=device)
        self._iteration = wp.zeros(1, dtype=wp.int32, device=device)
        self._active = wp.zeros(1, dtype=wp.int32, device=device)
        self._limit = wp.zeros(2, dtype=wp.int32, device=device)
        self._jacobian = wp.zeros((len(solver._tets), 9), dtype=wp.mat33d, device=device)
        self._contributions = wp.zeros((len(solver._tets), 4), dtype=wp.vec3d, device=device)
        self._anderson = Anderson(n, solver._anderson.window, device)
        self._raw_column = self._raw.reshape((1, n))
        self._z_column = self._z.reshape((1, n))
        self._previous_column = self._previous.reshape((1, n))
        self._muscle_target = wp.zeros_like(solver._muscle_target)
        self._graph = None
        self._graph_key = None
        self._stand_ins = {}
        contact = solver._contact
        self._rows = 0
        if contact is not None and contact.count:
            count = contact.count
            self._tangents = int(contact.friction > 0)
            self._rows = count * (3 if self._tangents else 1)
            rows = self._rows
            self._row_node = wp.zeros(rows, dtype=wp.int32, device=device)
            self._row_vector = wp.zeros(rows, dtype=wp.vec3d, device=device)
            self._row_valid = wp.zeros(rows, dtype=wp.int32, device=device)
            self._row_value = wp.zeros(rows, dtype=wp.float64, device=device)
            self._selected = wp.zeros(rows, dtype=wp.int32, device=device)
            self._columns = wp.zeros((rows, n), dtype=wp.vec3d, device=device)
            self._order = wp.zeros(rows, dtype=wp.int32, device=device)
            self._count = wp.zeros(1, dtype=wp.int32, device=device)
            self._gram = wp.zeros((rows, rows), dtype=wp.float64, device=device)
            self._gram_rhs = wp.zeros(rows, dtype=wp.float64, device=device)
            self._permutation = wp.zeros(rows, dtype=wp.int32, device=device)
            self._coefficients = wp.zeros(rows, dtype=wp.float64, device=device)
            self._alpha = wp.zeros(rows, dtype=wp.float64, device=device)
            batch = min(rows, WOODBURY_BATCH)
            self._batch = batch
            for name in ("solution", "previous", "raw", "work", "rhs"):
                setattr(self, f"_batch_{name}", wp.zeros((batch, n), dtype=wp.vec3d, device=device))
            self._batch_partials = wp.zeros((batch, self.chunks), dtype=wp.vec2d, device=device)
            self._batch_clip_partials = wp.zeros((batch, self.chunks), dtype=wp.float64, device=device)
            self._batch_enabled = wp.zeros(batch, dtype=wp.int32, device=device)
            self._batch_contributions = wp.zeros((batch, len(solver._tets), 4), dtype=wp.vec3d, device=device)
            self._batch_rows = wp.zeros(batch, dtype=wp.int32, device=device)
            self._batch_iteration = wp.zeros(1, dtype=wp.int32, device=device)
            self._batch_active = wp.zeros(1, dtype=wp.int32, device=device)
            self._batch_anderson = Anderson(n, solver._anderson.window, device, columns=batch)
            self._batch_graph = None
            self._batch_key = None

    def _safe_clip(self, values, partials):
        columns, size = values.shape
        wp.launch(
            _clip_partials,
            (columns, self.chunks),
            inputs=[values, self._free, size],
            outputs=[partials, self.solver._status],
            device=self.solver.device,
        )
        wp.launch(
            _clip_apply,
            (columns, size),
            inputs=[partials, self.chunks, ADJOINT_CLIP_NORM],
            outputs=[values, self.solver._status],
            device=self.solver.device,
        )

    def _iterate(self):
        solver, device, n = self.solver, self.solver.device, self.solver._n
        wp.launch(
            _begin_iteration, n, inputs=[self._solution, self._rhs], outputs=[self._previous, self._work], device=device
        )
        wp.launch(
            _apply_local,
            len(solver._tets),
            inputs=[self._solution, solver.indices, solver.gradients, self._jacobian],
            outputs=[self._contributions],
            device=device,
        )
        wp.launch(gather, n, inputs=[*solver.incidence, self._contributions, 1], outputs=[self._work], device=device)
        self._inverse.apply(self._work, self._raw)
        self._safe_clip(self._raw_column, self._clip_partials)
        wp.launch(
            _step_partials,
            self.chunks,
            inputs=[self._free, self._previous, n],
            outputs=[self._raw, self._partials],
            device=device,
        )
        self._anderson.mix(self._raw_column, self._previous_column)
        self._safe_clip(self._raw_column, self._clip_partials)
        wp.launch(
            _finish_iteration,
            self.chunks,
            inputs=[self._raw, n],
            outputs=[self._solution, self._partials],
            device=device,
        )
        wp.launch(
            _check,
            1,
            inputs=[self._partials, self.chunks, solver.abs_tol, solver.rel_tol, self._limit],
            outputs=[self._iteration, self._active, solver._status],
            device=device,
        )

    def _batch_iterate(self):
        solver, device, n = self.solver, self.solver.device, self.solver._n
        batch = self._batch
        enabled = self._batch_enabled
        wp.launch(
            _batch_begin,
            (batch, n),
            inputs=[self._batch_solution, self._batch_rhs, enabled],
            outputs=[self._batch_previous, self._batch_work],
            device=device,
        )
        wp.launch(
            _batch_apply_local,
            (batch, len(solver._tets)),
            inputs=[self._batch_solution, enabled, solver.indices, solver.gradients, self._jacobian],
            outputs=[self._batch_contributions],
            device=device,
        )
        wp.launch(
            gather_batch,
            (batch, n),
            inputs=[*solver.incidence, self._batch_contributions, enabled],
            outputs=[self._batch_work],
            device=device,
        )
        self._inverse.apply_vec3_batch(self._batch_work, self._batch_raw)
        self._safe_clip(self._batch_raw, self._batch_clip_partials)
        wp.launch(
            _batch_step_partials,
            (batch, self.chunks),
            inputs=[self._free, self._batch_previous, enabled, n],
            outputs=[self._batch_raw, self._batch_partials],
            device=device,
        )
        self._batch_anderson.mix(self._batch_raw, self._batch_previous, enabled)
        self._safe_clip(self._batch_raw, self._batch_clip_partials)
        wp.launch(
            _batch_finish,
            (batch, self.chunks),
            inputs=[self._batch_raw, enabled, n],
            outputs=[self._batch_solution, self._batch_partials],
            device=device,
        )
        wp.launch(
            _batch_check,
            1,
            inputs=[
                self._batch_partials,
                self.chunks,
                batch,
                solver.abs_tol,
                solver.rel_tol,
                min(solver.max_iterations, WOODBURY_MAX_ITERATIONS),
                self._batch_iteration,
            ],
            outputs=[enabled, self._batch_active],
            device=device,
        )

    def _batch_fixed_point(self):
        """Solve the batch columns of ``W_T = (A - K)^{-1} J^T`` from zero starts."""
        device = self.solver.device
        wp.launch(_batch_start, 1, outputs=[self._batch_iteration, self._batch_active], device=device)
        self._batch_anderson.reset()
        if device.is_capturing:
            wp.capture_while(self._batch_active, self._batch_iterate)
            return
        if device.is_cuda and wp.is_conditional_graph_supported():
            key = (self._inverse, self._free, self.solver.abs_tol, self.solver.rel_tol, self.solver.max_iterations)
            if self._batch_graph is None or self._batch_key != key:
                self._inverse.reserve_vec3_batch(self._batch_work.shape)
                enabled = gc.isenabled()
                gc.disable()
                try:
                    with wp.ScopedCapture(device=device) as capture:
                        wp.capture_while(self._batch_active, self._batch_iterate)
                    self._batch_graph = capture.graph
                    self._batch_key = key
                finally:
                    if enabled:
                        gc.enable()
            wp.capture_launch(self._batch_graph)
            return
        for _ in range(min(self.solver.max_iterations, WOODBURY_MAX_ITERATIONS)):
            self._batch_iterate()
            if not self._batch_active.numpy()[0]:
                break

    def _fixed_point(self, limit, report):
        """Solve ``(A - K) solution = rhs`` from a zero start with at most ``limit`` iterations."""
        device = self.solver.device
        wp.launch(
            _set_limit,
            1,
            inputs=[self._limit, limit, int(report)],
            outputs=[self._iteration, self._active],
            device=device,
        )
        self._anderson.reset()
        if device.is_capturing:
            wp.capture_while(self._active, self._iterate)
            return
        if device.is_cuda and wp.is_conditional_graph_supported():
            key = (self._inverse, self._free, self.solver.abs_tol, self.solver.rel_tol)
            if self._graph is None or self._graph_key != key:
                enabled = gc.isenabled()
                gc.disable()
                try:
                    with wp.ScopedCapture(device=device) as capture:
                        wp.capture_while(self._active, self._iterate)
                    self._graph = capture.graph
                    self._graph_key = key
                finally:
                    if enabled:
                        gc.enable()
            wp.capture_launch(self._graph)
            return
        for _ in range(limit):
            self._iterate()
            if not self._active.numpy()[0]:
                break

    def backward(self, record: StepRecord, state_in, state_out, control):
        solver = self.solver
        device = solver.device
        n = solver._n
        self._inverse = record.inverse
        self._free = record.free
        wp.copy(self._x, record.x)
        wp.copy(self._muscle_target, record.muscle_target)
        wp.launch(
            _seed,
            n,
            inputs=[self._gradient(state_out.particle_q), self._gradient(state_out.particle_qd), record.dt],
            outputs=[self._rhs, self._base],
            device=device,
        )
        self._solution.zero_()
        wp.launch(
            _local_jacobian,
            len(solver._tets),
            inputs=[
                self._x,
                solver.indices,
                solver.gradients,
                record.material_code,
                *record.jacobian_weights,
                record.nh[0],
                record.nh[1],
                record.nh[2],
                solver._muscle_direction,
                solver._muscle_weight,
                self._muscle_target,
            ],
            outputs=[self._jacobian],
            device=device,
        )
        self._fixed_point(solver.max_iterations, True)
        wp.copy(self._mu, self._solution)
        wp.launch(
            _routing_partials,
            self.chunks,
            inputs=[self._rhs, self._mu, n],
            outputs=[self._routing_partials],
            device=device,
        )
        wp.launch(
            _routing_guard,
            1,
            inputs=[self._routing_partials, self.chunks, record.inertia, ADJOINT_CLIP_NORM],
            outputs=[self._routing_scale, self._diverged, solver._status],
            device=device,
        )
        materials = solver.model.tet_materials
        activations = record.activations
        material_gradient = materials.requires_grad and materials.grad is not None
        activation_gradient = activations is not None and activations.requires_grad and activations.grad is not None
        # Only parameter gradients use z. State and prescribed-position
        # gradients use the contact-free backbone mu below.
        if material_gradient or activation_gradient:
            wp.copy(self._z, self._solution)
            if self._rows and record.contact is not None:
                self._woodbury(record)
        wp.launch(
            _state_gradients,
            n,
            inputs=[self._base, self._mu, record.inertia, record.dt, self._routing_scale],
            outputs=[
                self._gradient(state_in.particle_q),
                self._gradient(state_in.particle_qd),
                self._gradient(state_in.particle_f),
            ],
            device=device,
        )
        if record.dirichlet:
            wp.launch(
                _dirichlet_coupling,
                len(solver._tets),
                inputs=[
                    self._mu,
                    solver.indices,
                    solver.gradients,
                    self._jacobian,
                    *record.weights,
                    solver._muscle_direction,
                    solver._muscle_weight,
                    self._diverged,
                ],
                outputs=[self._contributions],
                device=device,
            )
            wp.launch(
                gather, n, inputs=[*solver.incidence, self._contributions, 0], outputs=[self._coupling], device=device
            )
            if record.dirichlet_positions is None:
                wp.launch(
                    _dirichlet_gradients,
                    n,
                    inputs=[self._free, self._rhs, self._coupling, record.dt, int(record.integrate)],
                    outputs=[self._gradient(state_in.particle_q), self._gradient(state_in.particle_qd)],
                    device=device,
                )
            else:
                wp.launch(
                    _explicit_dirichlet_gradients,
                    n,
                    inputs=[self._free, self._rhs, self._coupling],
                    outputs=[self._gradient(record.dirichlet_positions)],
                    device=device,
                )
        self._material_gradient.zero_()
        self._activation_gradient.zero_()
        if material_gradient:
            wp.launch(
                _material_gradients,
                len(solver._tets),
                inputs=[
                    self._x,
                    self._z,
                    solver.indices,
                    solver.gradients,
                    record.material_code,
                    solver.mean_volume,
                    record.nh[0],
                    record.nh[1],
                ],
                outputs=[self._material_gradient],
                device=device,
            )
        if activation_gradient:
            wp.launch(
                _activation_gradients,
                len(solver._tets),
                inputs=[
                    self._x,
                    self._z,
                    solver.indices,
                    solver.gradients,
                    solver._muscle_direction,
                    solver._muscle_weight,
                ],
                outputs=[self._activation_gradient],
                device=device,
            )
        if material_gradient or activation_gradient:
            wp.launch(
                _parameter_guard,
                1,
                inputs=[
                    self._material_gradient,
                    self._activation_gradient,
                    len(solver._tets),
                    material_gradient and record.material_code == 0 and record.nh[1] == 0.0,
                    self._diverged,
                ],
                outputs=[self._parameter_valid, solver._status],
                device=device,
            )
        if material_gradient:
            wp.launch(
                _add_material_gradient,
                len(solver._tets),
                inputs=[self._material_gradient, self._parameter_valid],
                outputs=[materials.grad],
                device=device,
            )
        if activation_gradient:
            wp.launch(
                _add_activation_gradient,
                len(solver._tets),
                inputs=[self._activation_gradient, self._parameter_valid],
                outputs=[activations.grad],
                device=device,
            )

    def _gradient(self, array):
        """The gradient buffer of a state array, or a zero stand-in without one."""
        if array.grad is not None:
            return array.grad
        stand_in = self._stand_ins.get((array.shape, array.dtype))
        if stand_in is None:
            stand_in = wp.zeros(array.shape, dtype=array.dtype, device=array.device)
            self._stand_ins[(array.shape, array.dtype)] = stand_in
        stand_in.zero_()
        return stand_in

    def _woodbury(self, record):
        solver = self.solver
        contact = solver._contact
        device = solver.device
        n = solver._n
        multiplier, omega, normal = record.contact
        wp.launch(
            _basis,
            contact.count,
            inputs=[contact.nodes, multiplier, omega, normal, self._tangents, contact.count, self._diverged],
            outputs=[self._row_node, self._row_vector, self._row_valid],
            device=device,
        )
        wp.launch(
            _project_rows,
            self._rows,
            inputs=[self._mu, self._row_node, self._row_vector, self._row_valid],
            outputs=[self._row_value],
            device=device,
        )
        wp.launch(
            _select_rows,
            1,
            inputs=[self._row_value, self._row_valid, self._rows],
            outputs=[self._selected],
            device=device,
        )
        wp.launch(
            _compact_rows, 1, inputs=[self._selected, self._rows], outputs=[self._order, self._count], device=device
        )
        if device.is_capturing:
            # No host decisions during capture: cover every candidate row.
            count = self._rows
        else:
            # One read per backward step: only the selected rows need columns.
            count = int(self._count.numpy()[0])
        for first in range(0, count, self._batch):
            wp.launch(
                _batch_setup,
                (self._batch, n),
                inputs=[self._order, self._count, first, self._row_node, self._row_vector],
                outputs=[self._batch_rows, self._batch_enabled, self._batch_rhs, self._batch_solution],
                device=device,
            )
            self._batch_fixed_point()
            wp.launch(
                _batch_store,
                (self._batch, n),
                inputs=[self._batch_solution, self._batch_rows],
                outputs=[self._columns],
                device=device,
            )
        wp.launch(
            _gram,
            (self._rows, self._rows),
            inputs=[self._columns, self._row_node, self._row_vector, self._order, self._count, self._row_value],
            outputs=[self._gram, self._gram_rhs],
            device=device,
        )
        wp.launch(
            ldlt_solve,
            1,
            inputs=[self._gram, self._gram_rhs, self._count, 0],
            outputs=[self._permutation, self._coefficients],
            device=device,
        )
        # Zero first in a separate launch: scattered rows can be any index.
        wp.launch(_zero, self._rows, outputs=[self._alpha], device=device)
        wp.launch(
            _scatter_alpha,
            self._rows,
            inputs=[self._order, self._count, self._gram_rhs],
            outputs=[self._alpha],
            device=device,
        )
        wp.launch(
            _project_adjoint,
            n,
            inputs=[self._mu, self._columns, self._alpha, self._selected, self._rows],
            outputs=[self._z],
            device=device,
        )
        self._safe_clip(self._z_column, self._clip_partials)
