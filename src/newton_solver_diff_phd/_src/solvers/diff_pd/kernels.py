# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Portions translated from DiffPD (https://github.com/mit-gfx/diff_pd_public,
# commit 1594aa0), Copyright (c) 2021 MIT Graphics Group, MIT License.
# See THIRD_PARTY_NOTICES.md.

"""Kernels of the DiffPD ``pd_eigen`` forward step with ``use_bfgs = 1``.

The step follows ``ForwardProjectiveDynamics`` and ``PdNonlinearSolve`` of
``cpp/core/src/fem/deformable_projective_dynamics_forward.cpp``: the explicit
state forces give ``rhs = q + h v + h2m f_ext + h2m f_state(q, v)``, then
L-BFGS with the projective-dynamics matrix ``A`` as initial Hessian and a
backtracking Armijo line search minimizes
``Phi(x) = 1/2 m/h^2 |x - rhs|^2 + E_pd(x)`` until
``|g| <= rel_tol |m/h^2 rhs| + abs_tol``.

Scalars of the iteration live on the device so the loop runs without host
synchronization. Every reduction sums fixed chunks and then the chunk partials
in order, so results do not depend on thread scheduling.
"""

import warp as wp

from .projections import DOUBLE_EPSILON, corotated_projection, diagonal_matrix, unsigned_svd, volume_offset

# Status bits accumulated on the device until ``SolverDiffPD.check_status``.
STATUS_NOT_CONVERGED = wp.constant(1)
STATUS_SINGULAR = wp.constant(2)
STATUS_LINE_SEARCH = wp.constant(4)

# State-force kinds and the width of their packed parameter rows.
PLANAR_CONTACT = wp.constant(0)
ARC_CONTACT = wp.constant(1)
STATE_FORCE_WIDTH = 18

# Entries of the device ``scalars`` array.
OBJECTIVE = wp.constant(0)
TRIAL_OBJECTIVE = wp.constant(1)
STEP = wp.constant(2)
SLOPE = wp.constant(3)
THRESHOLD = wp.constant(4)
SCALAR_COUNT = 5

# Entries of the device ``counters`` array.
ITERATION = wp.constant(0)
PAIRS = wp.constant(1)
NEWEST = wp.constant(2)
EVALUATIONS = wp.constant(3)
FALLBACK = wp.constant(4)
SINGULAR = wp.constant(5)
LINE_SEARCH_FAILED = wp.constant(6)
COUNTER_COUNT = 7

# Vertices or elements summed by one thread of a reduction.
CHUNK = 16

# Armijo sufficient-decrease constant of the reference line search.
ARMIJO = wp.constant(wp.float64(1.0e-4))


@wp.kernel(enable_backward=False)
def _count_incidence(indices: wp.array[wp.vec4i], counts: wp.array[wp.int32]):
    e, k = wp.tid()
    wp.atomic_add(counts, indices[e][k], 1)


@wp.kernel(enable_backward=False)
def _fill_incidence(indices: wp.array[wp.vec4i], cursor: wp.array[wp.int32], entries: wp.array[wp.int32]):
    e, k = wp.tid()
    slot = wp.atomic_add(cursor, indices[e][k], 1)
    entries[slot] = 4 * e + k


@wp.kernel(enable_backward=False)
def _sort_incidence(offsets: wp.array[wp.int32], entries: wp.array[wp.int32]):
    # Insertion sort of each vertex's short list fixes the summation order.
    i = wp.tid()
    first = offsets[i]
    for a in range(first + 1, offsets[i + 1]):
        value = entries[a]
        b = a - 1
        while b >= first and entries[b] > value:
            entries[b + 1] = entries[b]
            b -= 1
        entries[b + 1] = value


def build_incidence(indices: wp.array, size: int, device):
    """Vertex-to-(element, corner) lists sorted by element, as CSR ``offsets, entries``."""
    count = len(indices)
    counts = wp.zeros(size + 1, dtype=wp.int32, device=device)
    wp.launch(_count_incidence, (count, 4), inputs=[indices], outputs=[counts], device=device)
    offsets = wp.zeros(size + 1, dtype=wp.int32, device=device)
    wp.utils.array_scan(counts, offsets, inclusive=False)
    cursor = wp.clone(offsets)
    entries = wp.zeros(4 * count, dtype=wp.int32, device=device)
    wp.launch(_fill_incidence, (count, 4), inputs=[indices, cursor], outputs=[entries], device=device)
    wp.launch(_sort_incidence, size, inputs=[offsets], outputs=[entries], device=device)
    return offsets, entries


@wp.kernel(enable_backward=False)
def gather(
    offsets: wp.array[wp.int32],
    entries: wp.array[wp.int32],
    contributions: wp.array2d[wp.vec3d],
    out: wp.array[wp.vec3d],
):
    """Sum per-(element, corner) contributions per vertex in element order, as the reference scatter."""
    i = wp.tid()
    total = wp.vec3d(wp.float64(0.0))
    for a in range(offsets[i], offsets[i + 1]):
        entry = entries[a]
        total += contributions[entry // 4, entry % 4]
    out[i] = total


@wp.func
def _friction(v: wp.vec3d, normal: wp.vec3d, fn: wp.float64, kf: wp.float64, mu: wp.float64):
    # ff = -min(kf, mu fn / (|us| + eps)) us with the tangential velocity us.
    us = v - wp.dot(v, normal) * normal
    return -wp.min(kf, mu * fn / (wp.length(us) + DOUBLE_EPSILON)) * us


@wp.func
def _penalty(kn: wp.float64, d: wp.float64, p: wp.float64):
    # kn pow(-min(0, d), p - 1); negative distances mean penetration.
    return kn * wp.pow(-wp.min(wp.float64(0.0), d), p - wp.float64(1.0))


@wp.func
def _state_force(kind: int, c: wp.array2d[wp.float64], k: int, q: wp.vec3d, v: wp.vec3d):
    force = wp.vec3d(wp.float64(0.0))
    if kind == PLANAR_CONTACT:
        # PlanarContactStateForce::ForwardForce; normal and offset are pre-divided by |n|.
        normal = wp.vec3d(c[k, 0], c[k, 1], c[k, 2])
        fn = _penalty(c[k, 5], wp.dot(normal, q) + c[k, 3], c[k, 4])
        force = fn * normal + _friction(v, normal, fn, c[k, 6], c[k, 7])
    elif kind == ARC_CONTACT:
        # ArcContactStateForce::ForwardForce with unit dir, unit start and aux = dir x start.
        center = wp.vec3d(c[k, 0], c[k, 1], c[k, 2])
        axis = wp.vec3d(c[k, 3], c[k, 4], c[k, 5])
        start = wp.vec3d(c[k, 6], c[k, 7], c[k, 8])
        aux = wp.vec3d(c[k, 9], c[k, 10], c[k, 11])
        local = q - center
        projected = local - wp.dot(local, axis) * axis
        angle = wp.atan2(wp.dot(projected, aux), wp.dot(projected, start))
        if wp.float64(0.0) < angle and angle < c[k, 13]:
            fn = _penalty(c[k, 15], c[k, 12] - wp.length(projected), c[k, 14])
            normal = -projected / (wp.length(projected) + DOUBLE_EPSILON)
            force = fn * normal + _friction(v, normal, fn, c[k, 16], c[k, 17])
    return force


@wp.kernel(enable_backward=False)
def begin_step(
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    f_ext: wp.array[wp.vec3],
    worlds: wp.array[wp.int32],
    model_gravity: wp.array[wp.vec3],
    gravity: wp.vec3d,
    use_model_gravity: int,
    mass: wp.float64,
    kinds: wp.array[wp.int32],
    parameters: wp.array2d[wp.float64],
    force_count: int,
    dt: wp.float64,
    h2m: wp.float64,
    position: wp.array[wp.vec3d],
    velocity: wp.array[wp.vec3d],
    rhs: wp.array[wp.vec3d],
    trial: wp.array[wp.vec3d],
):
    i = wp.tid()
    start = wp.vec3d(q[i])
    speed = wp.vec3d(qd[i])
    position[i] = start
    velocity[i] = speed
    g = gravity
    if use_model_gravity != 0:
        g = wp.vec3d(model_gravity[wp.max(worlds[i], 0)])
    # ForwardStateForce: gravity m g first, then the contacts in insertion order.
    force = mass * g
    for k in range(force_count):
        force += _state_force(kinds[k], parameters, k, start, speed)
    rhs[i] = start + dt * speed + h2m * wp.vec3d(f_ext[i]) + h2m * force
    # The initial guess is q; Dirichlet entries keep their start positions.
    trial[i] = start


@wp.kernel(enable_backward=False)
def threshold_partials(
    rhs: wp.array[wp.vec3d],
    free: wp.array[wp.float64],
    inertia: wp.float64,
    size: int,
    partials: wp.array[wp.float64],
):
    chunk = wp.tid()
    total = wp.float64(0.0)
    for i in range(chunk * CHUNK, wp.min(size, (chunk + 1) * CHUNK)):
        value = free[i] * (inertia * rhs[i])
        total += wp.dot(value, value)
    partials[chunk] = total


@wp.kernel(enable_backward=False)
def begin_solve(
    partials: wp.array[wp.float64],
    chunks: int,
    rel_tol: wp.float64,
    abs_tol: wp.float64,
    history: int,
    scalars: wp.array[wp.float64],
    counters: wp.array[wp.int32],
    active: wp.array[wp.int32],
):
    total = wp.float64(0.0)
    for c in range(chunks):
        total += partials[c]
    # |g| <= rel_tol |selected (m/h^2 rhs)| + abs_tol.
    scalars[THRESHOLD] = rel_tol * wp.sqrt(total) + abs_tol
    counters[ITERATION] = 0
    counters[PAIRS] = 0
    counters[NEWEST] = history - 1
    counters[SINGULAR] = 0
    active[0] = 1


@wp.kernel(enable_backward=False)
def evaluate_elements(
    x: wp.array[wp.vec3d],
    indices: wp.array[wp.vec4i],
    gradients: wp.array2d[wp.vec3d],
    volume: wp.float64,
    corotated: int,
    corotated_weight: wp.float64,
    volumetric: int,
    volume_weight: wp.float64,
    energy: wp.array[wp.float64],
    contributions: wp.array2d[wp.vec3d],
    counters: wp.array[wp.int32],
):
    """``ComputeDeformationGradientAuxiliaryDataAndProjection``, ``ComputePdEnergy`` and ``PdEnergyForce``."""
    e = wp.tid()
    tet = indices[e]
    f = wp.mat33d(wp.float64(0.0))
    for k in range(4):
        f += wp.outer(x[tet[k]], gradients[e, k])
    total = wp.float64(0.0)
    stress_c = wp.mat33d(wp.float64(0.0))
    stress_v = wp.mat33d(wp.float64(0.0))
    if corotated != 0 or volumetric != 0:
        u, sigma, v = unsigned_svd(f)
        if corotated != 0:
            difference = f - corotated_projection(u, v)
            total += corotated_weight * wp.float64(0.5) * wp.ddot(difference, difference) * volume
            stress_c = corotated_weight * difference
        if volumetric != 0:
            offset, singular = volume_offset(sigma)
            if singular != 0:
                counters[SINGULAR] = 1
            difference = f - u * diagonal_matrix(sigma + offset) * wp.transpose(v)
            total += volume_weight * wp.float64(0.5) * wp.ddot(difference, difference) * volume
            stress_v = volume_weight * difference
    energy[e] = total
    # f_k = -Vbar sum_E w_E (F - P_E) g_k, energies in insertion order.
    for k in range(4):
        g = gradients[e, k]
        contributions[e, k] = -(stress_c * g) * volume + -(stress_v * g) * volume


@wp.kernel(enable_backward=False)
def objective_partials(
    x: wp.array[wp.vec3d],
    rhs: wp.array[wp.vec3d],
    inertia: wp.float64,
    vertices: int,
    energy: wp.array[wp.float64],
    elements: int,
    partials: wp.array[wp.vec2d],
):
    chunk = wp.tid()
    residual = wp.float64(0.0)
    for i in range(chunk * CHUNK, wp.min(vertices, (chunk + 1) * CHUNK)):
        difference = x[i] - rhs[i]
        residual += wp.dot(difference, inertia * difference)
    total = wp.float64(0.0)
    for e in range(chunk * CHUNK, wp.min(elements, (chunk + 1) * CHUNK)):
        total += energy[e]
    partials[chunk] = wp.vec2d(residual, total)


@wp.func
def _objective(partials: wp.array[wp.vec2d], chunks: int):
    total = wp.vec2d(wp.float64(0.0))
    for c in range(chunks):
        total += partials[c]
    # 1/2 (x - rhs) . (m/h^2 (x - rhs)) + E_pd.
    return wp.float64(0.5) * total[0] + total[1]


@wp.kernel(enable_backward=False)
def initial_objective(
    partials: wp.array[wp.vec2d],
    chunks: int,
    scalars: wp.array[wp.float64],
    counters: wp.array[wp.int32],
    active: wp.array[wp.int32],
    status: wp.array[wp.int32],
):
    scalars[OBJECTIVE] = _objective(partials, chunks)
    scalars[TRIAL_OBJECTIVE] = scalars[OBJECTIVE]
    if counters[SINGULAR] != 0:
        # The reference throws "Singular F."; the solve stops here.
        status[0] = status[0] | STATUS_SINGULAR
        active[0] = 0


@wp.kernel(enable_backward=False)
def accept(
    trial: wp.array[wp.vec3d],
    force: wp.array[wp.vec3d],
    rhs: wp.array[wp.vec3d],
    free: wp.array[wp.float64],
    inertia: wp.float64,
    size: int,
    x: wp.array[wp.vec3d],
    gradient: wp.array[wp.vec3d],
    partials: wp.array[wp.float64],
):
    chunk = wp.tid()
    total = wp.float64(0.0)
    for i in range(chunk * CHUNK, wp.min(size, (chunk + 1) * CHUNK)):
        x[i] = trial[i]
        # grad = (m/h^2 (x - rhs) - f_pd(x)) .* selected, f_pd from the last evaluation at x.
        value = (inertia * (trial[i] - rhs[i]) - force[i]) * free[i]
        gradient[i] = value
        total += wp.dot(value, value)
    partials[chunk] = total


@wp.kernel(enable_backward=False)
def history_advance(history: int, counters: wp.array[wp.int32]):
    if counters[ITERATION] > 0:
        counters[NEWEST] = (counters[NEWEST] + 1) % history
        counters[PAIRS] = wp.min(counters[PAIRS] + 1, history)


@wp.kernel(enable_backward=False)
def history_store(
    x: wp.array[wp.vec3d],
    gradient: wp.array[wp.vec3d],
    counters: wp.array[wp.int32],
    size: int,
    x_last: wp.array[wp.vec3d],
    gradient_last: wp.array[wp.vec3d],
    s: wp.array2d[wp.vec3d],
    y: wp.array2d[wp.vec3d],
    r: wp.array[wp.vec3d],
    partials: wp.array[wp.float64],
):
    chunk = wp.tid()
    total = wp.float64(0.0)
    store = counters[ITERATION] > 0
    slot = counters[NEWEST]
    for i in range(chunk * CHUNK, wp.min(size, (chunk + 1) * CHUNK)):
        if store:
            step = x[i] - x_last[i]
            change = gradient[i] - gradient_last[i]
            s[slot, i] = step
            y[slot, i] = change
            total += wp.dot(change, step)
        x_last[i] = x[i]
        gradient_last[i] = gradient[i]
        r[i] = gradient[i]
    partials[chunk] = total


@wp.kernel(enable_backward=False)
def history_rho(partials: wp.array[wp.float64], chunks: int, counters: wp.array[wp.int32], rho: wp.array[wp.float64]):
    if counters[ITERATION] > 0:
        total = wp.float64(0.0)
        for c in range(chunks):
            total += partials[c]
        # rho = 1 / (y . s), without a curvature check, as in the reference.
        rho[counters[NEWEST]] = wp.float64(1.0) / total


@wp.func
def _slot(age: int, history: int, counters: wp.array[wp.int32]):
    # Ring-buffer slot of the pair ``age`` updates old, or -1 when absent.
    slot = int(-1)
    if age < counters[PAIRS]:
        slot = (counters[NEWEST] - age + history) % history
    return slot


@wp.kernel(enable_backward=False)
def slot_dot_partials(
    rows: wp.array2d[wp.vec3d],
    vector: wp.array[wp.vec3d],
    age: int,
    history: int,
    counters: wp.array[wp.int32],
    size: int,
    partials: wp.array[wp.float64],
):
    chunk = wp.tid()
    slot = _slot(age, history, counters)
    total = wp.float64(0.0)
    if slot >= 0:
        for i in range(chunk * CHUNK, wp.min(size, (chunk + 1) * CHUNK)):
            total += wp.dot(rows[slot, i], vector[i])
    partials[chunk] = total


@wp.kernel(enable_backward=False)
def set_alpha(
    partials: wp.array[wp.float64],
    chunks: int,
    age: int,
    history: int,
    counters: wp.array[wp.int32],
    rho: wp.array[wp.float64],
    alpha: wp.array[wp.float64],
):
    slot = _slot(age, history, counters)
    if slot >= 0:
        total = wp.float64(0.0)
        for c in range(chunks):
            total += partials[c]
        alpha[slot] = rho[slot] * total


@wp.kernel(enable_backward=False)
def set_coefficient(
    partials: wp.array[wp.float64],
    chunks: int,
    age: int,
    history: int,
    counters: wp.array[wp.int32],
    rho: wp.array[wp.float64],
    alpha: wp.array[wp.float64],
    coefficient: wp.array[wp.float64],
):
    slot = _slot(age, history, counters)
    if slot >= 0:
        total = wp.float64(0.0)
        for c in range(chunks):
            total += partials[c]
        # z += s (alpha - beta) with beta = rho (y . z).
        coefficient[slot] = alpha[slot] - rho[slot] * total


@wp.kernel(enable_backward=False)
def slot_axpy(
    rows: wp.array2d[wp.vec3d],
    coefficient: wp.array[wp.float64],
    sign: wp.float64,
    age: int,
    history: int,
    counters: wp.array[wp.int32],
    vector: wp.array[wp.vec3d],
):
    i = wp.tid()
    slot = _slot(age, history, counters)
    if slot >= 0:
        vector[i] = vector[i] + sign * (coefficient[slot] * rows[slot, i])


@wp.kernel(enable_backward=False)
def direction_partials(
    direction: wp.array[wp.vec3d], gradient: wp.array[wp.vec3d], size: int, partials: wp.array[wp.vec2d]
):
    chunk = wp.tid()
    total = wp.vec2d(wp.float64(0.0))
    for i in range(chunk * CHUNK, wp.min(size, (chunk + 1) * CHUNK)):
        total += wp.vec2d(wp.dot(direction[i], gradient[i]), wp.dot(gradient[i], gradient[i]))
    partials[chunk] = total


@wp.kernel(enable_backward=False)
def choose_direction(
    partials: wp.array[wp.vec2d],
    chunks: int,
    scalars: wp.array[wp.float64],
    counters: wp.array[wp.int32],
    searching: wp.array[wp.int32],
):
    total = wp.vec2d(wp.float64(0.0))
    for c in range(chunks):
        total += partials[c]
    # A direction that does not descend falls back to the masked gradient.
    counters[FALLBACK] = 0
    counters[LINE_SEARCH_FAILED] = 0
    scalars[SLOPE] = total[0]
    if total[0] <= wp.float64(0.0):
        counters[FALLBACK] = 1
        scalars[SLOPE] = total[1]
    scalars[STEP] = wp.float64(1.0)
    counters[EVALUATIONS] = 0
    searching[0] = 1


@wp.kernel(enable_backward=False)
def fallback_direction(
    gradient: wp.array[wp.vec3d],
    free: wp.array[wp.float64],
    counters: wp.array[wp.int32],
    direction: wp.array[wp.vec3d],
):
    i = wp.tid()
    if counters[FALLBACK] != 0:
        direction[i] = gradient[i] * free[i]


@wp.kernel(enable_backward=False)
def trial_position(
    x: wp.array[wp.vec3d], direction: wp.array[wp.vec3d], scalars: wp.array[wp.float64], trial: wp.array[wp.vec3d]
):
    i = wp.tid()
    trial[i] = x[i] - scalars[STEP] * direction[i]


@wp.kernel(enable_backward=False)
def line_search_check(
    partials: wp.array[wp.vec2d],
    chunks: int,
    max_ls_iter: int,
    scalars: wp.array[wp.float64],
    counters: wp.array[wp.int32],
    searching: wp.array[wp.int32],
):
    objective = _objective(partials, chunks)
    scalars[TRIAL_OBJECTIVE] = objective
    evaluations = counters[EVALUATIONS] + 1
    counters[EVALUATIONS] = evaluations
    searching[0] = 0
    if counters[SINGULAR] != 0:
        return
    # After max_ls_iter halvings the last trial is accepted without a test.
    if evaluations > max_ls_iter:
        counters[LINE_SEARCH_FAILED] = 1
        return
    step = scalars[STEP]
    bound = scalars[OBJECTIVE] - ARMIJO * step * scalars[SLOPE]
    if wp.isnan(objective) or not (objective < bound + DOUBLE_EPSILON):
        scalars[STEP] = step / wp.float64(2.0)
        searching[0] = 1


@wp.kernel(enable_backward=False)
def check_convergence(
    partials: wp.array[wp.float64],
    chunks: int,
    max_pd_iter: int,
    scalars: wp.array[wp.float64],
    counters: wp.array[wp.int32],
    active: wp.array[wp.int32],
    status: wp.array[wp.int32],
):
    scalars[OBJECTIVE] = scalars[TRIAL_OBJECTIVE]
    if counters[LINE_SEARCH_FAILED] != 0:
        status[0] = status[0] | STATUS_LINE_SEARCH
    k = counters[ITERATION]
    counters[ITERATION] = k + 1
    if counters[SINGULAR] != 0:
        status[0] = status[0] | STATUS_SINGULAR
        active[0] = 0
        return
    total = wp.float64(0.0)
    for c in range(chunks):
        total += partials[c]
    if wp.sqrt(total) <= scalars[THRESHOLD]:
        active[0] = 0
    elif k + 1 >= max_pd_iter:
        active[0] = 0
        status[0] = status[0] | STATUS_NOT_CONVERGED
    else:
        active[0] = 1


@wp.kernel(enable_backward=False)
def write_state(
    x: wp.array[wp.vec3d],
    position: wp.array[wp.vec3d],
    dt: wp.float64,
    velocity: wp.array[wp.vec3d],
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
):
    i = wp.tid()
    # v_next = (q_next - q) / h.
    value = (x[i] - position[i]) / dt
    velocity[i] = value
    q[i] = wp.vec3(x[i])
    qd[i] = wp.vec3(value)
