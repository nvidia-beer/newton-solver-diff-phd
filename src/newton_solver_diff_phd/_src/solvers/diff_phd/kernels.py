# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Portions translated from DiffPhD (https://github.com/Akinesia112/DiffPhD,
# commit d319435), Copyright (c) 2021 MIT Graphics Group, MIT License.
# See THIRD_PARTY_NOTICES.md.

"""Kernels of the reference DiffPhD projective-dynamics step.

The step follows ``ForwardProjectiveDynamicsAlgPhd`` and
``ProjectiveDynamicsLocalStep`` of the reference: ``q~ = q + h v +
h^2/m (f_ext + m g)``, then repeat ``x <- A^{-1} (m/h^2 q~ + p(x))`` with the
local projections ``p``, Dirichlet elimination, Anderson mixing and the dual
convergence gate on the position update and the projection right-hand side.
"""

import warp as wp

from .anderson import CHUNK
from .projections import (
    corotated_projection,
    diagonal_matrix,
    muscle_projection,
    nh_prox_stretches,
    unsigned_svd,
    volume_offset,
)

# Status bits accumulated on the device until ``SolverDiffPhD.check_status``.
STATUS_NOT_CONVERGED = wp.constant(1)
STATUS_SINGULAR = wp.constant(2)
STATUS_ACTIVATION = wp.constant(4)


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
    accumulate: int,
    out: wp.array[wp.vec3d],
):
    """Sum per-(element, corner) contributions per vertex in a fixed order."""
    i = wp.tid()
    total = wp.vec3d(wp.float64(0.0))
    for a in range(offsets[i], offsets[i + 1]):
        entry = entries[a]
        total += contributions[entry // 4, entry % 4]
    if accumulate != 0:
        total += out[i]
    out[i] = total


@wp.kernel(enable_backward=False)
def gather_batch(
    offsets: wp.array[wp.int32],
    entries: wp.array[wp.int32],
    contributions: wp.array3d[wp.vec3d],
    enabled: wp.array[wp.int32],
    out: wp.array2d[wp.vec3d],
):
    col, i = wp.tid()
    if enabled[col] == 0:
        return
    total = out[col, i]
    for a in range(offsets[i], offsets[i + 1]):
        entry = entries[a]
        total += contributions[col, entry // 4, entry % 4]
    out[col, i] = total


@wp.kernel(enable_backward=False)
def element_weights(
    materials: wp.array2d[wp.float32],
    material: int,
    volume: wp.float64,
    corotated: wp.array[wp.float64],
    volumetric: wp.array[wp.float64],
    neo_hookean: wp.array[wp.float64],
):
    e = wp.tid()
    mu = wp.float64(materials[e, 0])
    lame = wp.float64(materials[e, 1])
    corotated[e] = wp.float64(0.0)
    volumetric[e] = wp.float64(0.0)
    neo_hookean[e] = wp.float64(0.0)
    if material == 0:
        # Corotated weight 2 mu and volume weight lambda, as in the reference examples.
        corotated[e] = wp.float64(2.0) * mu * volume
        volumetric[e] = lame * volume
    else:
        # NeoHookeanPdElementEnergy: A-matrix weight = proximal penalty = 2 mu.
        neo_hookean[e] = wp.float64(2.0) * mu * volume


@wp.kernel(enable_backward=False)
def read_activations(
    source: wp.array[wp.float32],
    weight: wp.array[wp.float64],
    target: wp.array[wp.float64],
    status: wp.array[wp.int32],
):
    e = wp.tid()
    contraction = wp.float64(source[e])
    # Contraction u gives the reference target fibre length a = 1 - u.
    if (
        not wp.isfinite(contraction)
        or contraction > wp.float64(1.0)
        or (weight[e] == wp.float64(0.0) and contraction != wp.float64(0.0))
    ):
        wp.atomic_or(status, 0, STATUS_ACTIVATION)
    target[e] = wp.float64(1.0) - contraction


@wp.kernel(enable_backward=False)
def begin_step(
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    f: wp.array[wp.vec3],
    free: wp.array[wp.int32],
    worlds: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    dt: wp.float64,
    mass: wp.float64,
    integrate_pinned_velocities: bool,
    position: wp.array[wp.vec3d],
    velocity_out: wp.array[wp.vec3d],
    target: wp.array[wp.vec3d],
    boundary: wp.array[wp.vec3d],
    x: wp.array[wp.vec3d],
    projection: wp.array[wp.vec3d],
):
    i = wp.tid()
    start = wp.vec3d(q[i])
    velocity = wp.vec3d(qd[i])
    position[i] = start
    velocity_out[i] = velocity
    # q~ covers every degree of freedom; Dirichlet rows of A are identity and
    # their solution entries are overwritten by the Dirichlet values.
    acceleration = wp.vec3d(f[i]) / mass + wp.vec3d(gravity[wp.max(worlds[i], 0)])
    target[i] = start + dt * velocity + dt * dt * acceleration
    boundary[i] = wp.vec3d(wp.float64(0.0))
    if free[i] == 0:
        value = start
        if integrate_pinned_velocities:
            value += dt * velocity
        boundary[i] = value
    # The reference starts every step from q, including Dirichlet entries.
    x[i] = start
    projection[i] = wp.vec3d(wp.float64(0.0))


@wp.kernel(enable_backward=False)
def begin_solve(iteration: wp.array[wp.int32], active: wp.array[wp.int32]):
    iteration[0] = 0
    active[0] = 1


@wp.kernel(enable_backward=False)
def explicit_boundary(
    positions: wp.array[wp.vec3],
    free: wp.array[wp.int32],
    boundary: wp.array[wp.vec3d],
):
    i = wp.tid()
    if free[i] == 0:
        boundary[i] = wp.vec3d(positions[i])


@wp.kernel(enable_backward=False)
def begin_iteration(
    x: wp.array[wp.vec3d],
    projection: wp.array[wp.vec3d],
    previous: wp.array[wp.vec3d],
    previous_projection: wp.array[wp.vec3d],
):
    i = wp.tid()
    previous[i] = x[i]
    previous_projection[i] = projection[i]


@wp.kernel(enable_backward=False)
def local_step(
    x: wp.array[wp.vec3d],
    boundary: wp.array[wp.vec3d],
    indices: wp.array[wp.vec4i],
    gradients: wp.array2d[wp.vec3d],
    corotated_weight: wp.array[wp.float64],
    volume_weight: wp.array[wp.float64],
    neo_hookean_weight: wp.array[wp.float64],
    nh_mu: wp.float64,
    nh_lame: wp.float64,
    nh_penalty: wp.float64,
    muscle_direction: wp.array[wp.vec3d],
    muscle_weight: wp.array[wp.float64],
    muscle_target: wp.array[wp.float64],
    contributions: wp.array2d[wp.vec3d],
    status: wp.array[wp.int32],
):
    e = wp.tid()
    tet = indices[e]
    f = wp.mat33d(wp.float64(0.0))
    f_boundary = wp.mat33d(wp.float64(0.0))
    for k in range(4):
        g = gradients[e, k]
        f += wp.outer(x[tet[k]], g)
        f_boundary += wp.outer(boundary[tet[k]], g)
    # sum_energies w (P(F) - F_boundary); G_e^T maps it to node k as (.) g_k.
    stress = wp.mat33d(wp.float64(0.0))
    wc = corotated_weight[e]
    wv = volume_weight[e]
    wn = neo_hookean_weight[e]
    if wc != wp.float64(0.0) or wv != wp.float64(0.0) or wn != wp.float64(0.0):
        u, sigma, v = unsigned_svd(f)
        vt = wp.transpose(v)
        if wc != wp.float64(0.0):
            stress += wc * (corotated_projection(u, v) - f_boundary)
        if wv != wp.float64(0.0):
            offset, singular = volume_offset(sigma)
            if singular != 0:
                wp.atomic_or(status, 0, STATUS_SINGULAR)
            stress += wv * (u * diagonal_matrix(sigma + offset) * vt - f_boundary)
        if wn != wp.float64(0.0):
            stretches = nh_prox_stretches(sigma, nh_mu, nh_lame, nh_penalty)
            stress += wn * (u * diagonal_matrix(stretches) * vt - f_boundary)
    wm = muscle_weight[e]
    if wm != wp.float64(0.0):
        m = muscle_direction[e]
        fibre = muscle_projection(f, m, muscle_target[e])
        stress += wm * wp.outer(fibre - f_boundary * m, m)
    for k in range(4):
        contributions[e, k] = stress * gradients[e, k]


@wp.kernel(enable_backward=False)
def global_rhs(
    target: wp.array[wp.vec3d],
    projection: wp.array[wp.vec3d],
    inertia: wp.float64,
    rhs: wp.array[wp.vec3d],
):
    i = wp.tid()
    # b_full = m/h^2 q~ + p; the reference contact branch solves with all rows.
    rhs[i] = inertia * target[i] + projection[i]


@wp.kernel(enable_backward=False)
def impose_boundary(boundary: wp.array[wp.vec3d], free: wp.array[wp.int32], x: wp.array[wp.vec3d]):
    i = wp.tid()
    if free[i] == 0:
        x[i] = boundary[i]


@wp.kernel(enable_backward=False)
def convergence_partials(
    boundary: wp.array[wp.vec3d],
    free: wp.array[wp.int32],
    previous: wp.array[wp.vec3d],
    projection: wp.array[wp.vec3d],
    previous_projection: wp.array[wp.vec3d],
    size: int,
    x: wp.array[wp.vec3d],
    partials: wp.array[wp.vec4d],
):
    chunk = wp.tid()
    total = wp.vec4d(wp.float64(0.0))
    for i in range(chunk * CHUNK, wp.min(size, (chunk + 1) * CHUNK)):
        # Dirichlet entries are re-imposed after Anderson mixing.
        if free[i] == 0:
            x[i] = boundary[i]
        step = x[i] - previous[i]
        change = projection[i] - previous_projection[i]
        total += wp.vec4d(
            wp.dot(step, step),
            wp.dot(previous[i], previous[i]),
            wp.dot(change, change),
            wp.dot(previous_projection[i], previous_projection[i]),
        )
    partials[chunk] = total


@wp.kernel(enable_backward=False)
def check_convergence(
    partials: wp.array[wp.vec4d],
    chunks: int,
    abs_tol: wp.float64,
    rel_tol: wp.float64,
    max_iterations: int,
    iteration: wp.array[wp.int32],
    active: wp.array[wp.int32],
    status: wp.array[wp.int32],
):
    total = wp.vec4d(wp.float64(0.0))
    for c in range(chunks):
        total += partials[c]
    k = iteration[0]
    # ||x - x_prev|| <= rel ||x_prev|| + abs and ||p - p_prev|| <= rel ||p_prev|| + abs.
    step_small = wp.sqrt(total[0]) <= rel_tol * wp.sqrt(total[1]) + abs_tol
    rhs_small = wp.sqrt(total[2]) <= rel_tol * wp.sqrt(total[3]) + abs_tol
    iteration[0] = k + 1
    if k > 0 and step_small and rhs_small:
        active[0] = 0
    elif k + 1 >= max_iterations:
        active[0] = 0
        status[0] = status[0] | STATUS_NOT_CONVERGED
    else:
        active[0] = 1


# Threads of the single-block convergence reduction.
CONVERGENCE_BLOCK = 256


@wp.kernel(enable_backward=False)
def convergence_tiled(
    boundary: wp.array[wp.vec3d],
    free: wp.array[wp.int32],
    previous: wp.array[wp.vec3d],
    projection: wp.array[wp.vec3d],
    previous_projection: wp.array[wp.vec3d],
    size: int,
    abs_tol: wp.float64,
    rel_tol: wp.float64,
    max_iterations: int,
    x: wp.array[wp.vec3d],
    iteration: wp.array[wp.int32],
    active: wp.array[wp.int32],
    status: wp.array[wp.int32],
):
    # convergence_partials and check_convergence in one block.
    _block, lane = wp.tid()
    total = wp.vec4d(wp.float64(0.0))
    for i in range(lane, size, CONVERGENCE_BLOCK):
        if free[i] == 0:
            x[i] = boundary[i]
        step = x[i] - previous[i]
        change = projection[i] - previous_projection[i]
        total += wp.vec4d(
            wp.dot(step, step),
            wp.dot(previous[i], previous[i]),
            wp.dot(change, change),
            wp.dot(previous_projection[i], previous_projection[i]),
        )
    step_norm = wp.sqrt(wp.tile_sum(wp.tile(total[0]))[0])
    previous_norm = wp.sqrt(wp.tile_sum(wp.tile(total[1]))[0])
    change_norm = wp.sqrt(wp.tile_sum(wp.tile(total[2]))[0])
    projection_norm = wp.sqrt(wp.tile_sum(wp.tile(total[3]))[0])
    if lane == 0:
        k = iteration[0]
        iteration[0] = k + 1
        if (
            k > 0
            and step_norm <= rel_tol * previous_norm + abs_tol
            and change_norm <= rel_tol * projection_norm + abs_tol
        ):
            active[0] = 0
        elif k + 1 >= max_iterations:
            active[0] = 0
            status[0] = status[0] | STATUS_NOT_CONVERGED
        else:
            active[0] = 1


@wp.kernel(enable_backward=False)
def write_state(
    x: wp.array[wp.vec3d],
    position: wp.array[wp.vec3d],
    dt: wp.float64,
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
):
    i = wp.tid()
    q[i] = wp.vec3(x[i])
    qd[i] = wp.vec3((x[i] - position[i]) / dt)
