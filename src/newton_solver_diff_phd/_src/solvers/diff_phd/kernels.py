# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Double precision step residuals and native Newton state transfers."""

import warp as wp

from .elasticity import elastic_stress, tet_deformation


@wp.kernel(enable_backward=False)
def check_model(
    indices: wp.array[wp.int32],
    poses: wp.array[wp.mat33],
    worlds: wp.array[wp.int32],
    materials: wp.array2d[wp.float32],
    masses: wp.array[wp.float32],
    inverse_masses: wp.array[wp.float32],
    flags: wp.array[wp.int32],
    reference_indices: wp.array[wp.vec4i],
    reference_poses: wp.array[wp.mat33],
    reference_worlds: wp.array[wp.int32],
    mu: wp.array[wp.float64],
    lame: wp.array[wp.float64],
    mass: wp.array[wp.float64],
    free: wp.array[wp.int32],
    active_flag: int,
    status: wp.array[wp.int32],
):
    i = wp.tid()
    changed = int(0)
    if i < reference_indices.shape[0]:
        for j in range(4):
            if indices[4 * i + j] != reference_indices[i][j]:
                changed |= 1
        for a in range(3):
            for b in range(3):
                if poses[i][a, b] != reference_poses[i][a, b]:
                    changed |= 2
        if wp.float64(materials[i, 0]) != mu[i] or wp.float64(materials[i, 1]) != lame[i] or materials[i, 2] != 0.0:
            changed |= 8
    if i < mass.shape[0]:
        if worlds[i] != reference_worlds[i]:
            changed |= 4
        if (
            wp.float64(masses[i]) != mass[i]
            or int(inverse_masses[i] > 0.0) != free[i]
            or not wp.isfinite(inverse_masses[i])
            or inverse_masses[i] < 0.0
            or (flags[i] & active_flag) == 0
        ):
            changed |= 8
    if changed != 0:
        wp.atomic_or(status, 0, changed)


@wp.kernel(enable_backward=False)
def check_zero(values: wp.array[wp.float32], status: wp.array[wp.int32]):
    if values[wp.tid()] != 0.0:
        wp.atomic_or(status, 0, 8)


@wp.kernel(enable_backward=False)
def warm_start_contacts(
    old_indices: wp.array[wp.vec4i],
    old_kinds: wp.array[wp.int32],
    old_frames: wp.array[wp.mat33d],
    old_lam: wp.array[wp.float64],
    indices: wp.array[wp.vec4i],
    kinds: wp.array[wp.int32],
    frames: wp.array[wp.mat33d],
    friction: wp.float64,
    lam: wp.array[wp.float64],
):
    c = wp.tid()
    value = wp.vec3d(0.0)
    # Contact sets are bounded by max_contacts. Exact feature matching also
    # works when a new geometry pass changes their order or inserts rows.
    for old in range(old_indices.shape[0]):
        equal = bool(kinds[c] == old_kinds[old])
        for j in range(4):
            equal = equal and indices[c][j] == old_indices[old][j]
        if equal:
            force = old_frames[old] * wp.vec3d(old_lam[3 * old], old_lam[3 * old + 1], old_lam[3 * old + 2])
            value = wp.transpose(frames[c]) * force
            value[0] = wp.max(value[0], wp.float64(0.0))
            length = wp.sqrt(value[1] * value[1] + value[2] * value[2])
            if length > friction * value[0]:
                scale = friction * value[0] / length
                value[1] *= scale
                value[2] *= scale
            break
    for j in range(3):
        lam[3 * c + j] = value[j]


@wp.kernel
def minimum_jacobian(
    q: wp.array[wp.vec3d], indices: wp.array[wp.vec4i], gradients: wp.array2d[wp.vec3d], minimum: wp.array[wp.float64]
):
    e = wp.tid()
    f = tet_deformation(q, indices, gradients, e)
    determinant = wp.determinant(f)
    if not wp.isfinite(determinant):
        determinant = wp.float64(-1.0)
    wp.atomic_min(minimum, 0, determinant)


@wp.kernel
def checked_tet_gradient(
    q: wp.array[wp.vec3d],
    indices: wp.array[wp.vec4i],
    gradients: wp.array2d[wp.vec3d],
    volumes: wp.array[wp.float64],
    mu: wp.array[wp.float64],
    lame: wp.array[wp.float64],
    model: int,
    minimum: wp.array[wp.float64],
    out: wp.array[wp.vec3d],
):
    # The preceding device reduction guards logarithms and projections without
    # downloading a determinant before every residual evaluation.
    if minimum[0] > wp.float64(1.0e-10):
        element = wp.tid()
        f = tet_deformation(q, indices, gradients, element)
        stress = elastic_stress(f, mu[element], lame[element], model)
        for node in range(4):
            wp.atomic_add(out, indices[element][node], volumes[element] * stress * gradients[element, node])


@wp.kernel
def restore_invalid_predictor(
    q: wp.array[wp.vec3d],
    previous: wp.array[wp.vec3d],
    free: wp.array[wp.int32],
    integrate_pinned_velocities: bool,
    minimum: wp.array[wp.float64],
):
    if minimum[0] <= wp.float64(1.0e-10):
        i = wp.tid()
        if free[i] != 0 or not integrate_pinned_velocities:
            q[i] = previous[i]


@wp.kernel(enable_backward=False)
def joint_trial(
    q: wp.array[wp.vec3d],
    target: wp.array[wp.vec3d],
    old_lam: wp.array[wp.float64],
    direction: wp.array[wp.float64],
    state: wp.array[wp.int32],
    trial: wp.array[wp.vec3d],
    lam: wp.array[wp.float64],
):
    i = wp.tid()
    alpha = wp.float64(1.0) / wp.float64(1 << state[1])
    if i < q.shape[0]:
        if state[3] == 0:
            trial[i] = (wp.float64(1.0) - alpha) * q[i] + alpha * target[i]
        else:
            trial[i] = q[i] + alpha * wp.vec3d(direction[3 * i], direction[3 * i + 1], direction[3 * i + 2])
    if i < lam.shape[0]:
        offset = int(0)
        if state[3] != 0:
            offset = 3 * q.shape[0]
        lam[i] = old_lam[i] + alpha * direction[offset + i]


@wp.kernel(enable_backward=False)
def accept_joint_trial(
    norm: wp.array[wp.float64],
    bound: wp.array[wp.float64],
    tolerance: wp.float64,
    contact_tolerance: wp.float64,
    state: wp.array[wp.int32],
    result: wp.array[wp.float64],
):
    error = wp.max(norm[0], norm[1] * tolerance / contact_tolerance)
    result[1] = error
    if error < bound[0] or error <= tolerance:
        state[0] = 0
        result[0] = wp.float64(1.0)
    else:
        state[1] += 1
        if state[1] >= state[2]:
            state[0] = 0


@wp.kernel
def read_state(
    q: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    f: wp.array[wp.vec3],
    x: wp.array[wp.vec3d],
    velocity: wp.array[wp.vec3d],
    force: wp.array[wp.vec3d],
):
    i = wp.tid()
    x[i] = wp.vec3d(q[i])
    velocity[i] = wp.vec3d(v[i])
    force[i] = wp.vec3d(f[i])


@wp.kernel
def predictor(
    previous: wp.array[wp.vec3d],
    velocity: wp.array[wp.vec3d],
    force: wp.array[wp.vec3d],
    mass: wp.array[wp.float64],
    free: wp.array[wp.int32],
    worlds: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    dt: wp.float64,
    damping: wp.float64,
    integrate_pinned_velocities: bool,
    q: wp.array[wp.vec3d],
):
    i = wp.tid()
    value = previous[i]
    if free[i] != 0:
        acceleration = wp.vec3d(gravity[wp.max(worlds[i], 0)]) + force[i] / mass[i]
        value += dt * (velocity[i] + dt * acceleration) / (wp.float64(1.0) + damping * dt)
    elif integrate_pinned_velocities:
        value += dt * velocity[i]
    q[i] = value


@wp.kernel
def add_inertia(
    q: wp.array[wp.vec3d],
    previous: wp.array[wp.vec3d],
    velocity: wp.array[wp.vec3d],
    force: wp.array[wp.vec3d],
    mass: wp.array[wp.float64],
    free: wp.array[wp.int32],
    worlds: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    dt: wp.float64,
    damping: wp.float64,
    integrate_pinned_velocities: bool,
    residual: wp.array[wp.vec3d],
):
    i = wp.tid()
    if free[i] == 0:
        residual[i] = q[i] - previous[i]
        if integrate_pinned_velocities:
            residual[i] -= dt * velocity[i]
    else:
        residual[i] += mass[i] / (dt * dt) * (q[i] - previous[i] - dt * velocity[i])
        residual[i] += damping * mass[i] / dt * (q[i] - previous[i])
        residual[i] -= force[i] + mass[i] * wp.vec3d(gravity[wp.max(worlds[i], 0)])


@wp.kernel
def mask_force(force: wp.array[wp.vec3d], free: wp.array[wp.int32]):
    i = wp.tid()
    if free[i] == 0:
        force[i] = wp.vec3d(0.0)


@wp.kernel
def combine(a: wp.array[wp.vec3d], b: wp.array[wp.vec3d], alpha: wp.float64, beta: wp.float64, out: wp.array[wp.vec3d]):
    i = wp.tid()
    out[i] = alpha * a[i] + beta * b[i]


@wp.kernel
def residual_error(
    residual: wp.array[wp.vec3d],
    force: wp.array[wp.vec3d],
    free: wp.array[wp.int32],
    diagonal: wp.array[wp.float64],
    minimum: wp.array[wp.float64],
    norm: wp.array[wp.float64],
):
    i = wp.tid()
    if free[i] == 0:
        force[i] = wp.vec3d(0.0)
    value = wp.length(residual[i] - force[i]) / diagonal[i]
    if minimum[0] <= wp.float64(1.0e-10):
        value = wp.float64(wp.inf)
    elif not wp.isfinite(value):
        value = wp.float64(1.0e300)
    wp.atomic_max(norm, 0, value)


@wp.kernel
def step_error(q: wp.array[wp.vec3d], old: wp.array[wp.vec3d], norm: wp.array[wp.float64]):
    i = wp.tid()
    wp.atomic_max(norm, 0, wp.length(q[i] - old[i]))


@wp.kernel
def write_state(
    q: wp.array[wp.vec3d],
    previous: wp.array[wp.vec3d],
    velocity: wp.array[wp.vec3d],
    free: wp.array[wp.int32],
    dt: wp.float64,
    integrate_pinned_velocities: bool,
    out_q: wp.array[wp.vec3],
    out_v: wp.array[wp.vec3],
):
    i = wp.tid()
    out_q[i] = wp.vec3(q[i])
    value = wp.vec3d(0.0)
    if free[i] != 0:
        value = (q[i] - previous[i]) / dt
    elif integrate_pinned_velocities:
        value = velocity[i]
    out_v[i] = wp.vec3(value)


@wp.kernel
def gram_history(
    differences: wp.array2d[wp.vec3d],
    residual: wp.array[wp.vec3d],
    length: int,
    matrix: wp.array2d[wp.float64],
    rhs: wp.array[wp.float64],
):
    a, b = wp.tid()
    value = wp.float64(0.0)
    for i in range(length):
        value += wp.dot(differences[a, i], differences[b, i])
    matrix[a, b] = value
    if b == 0:
        value_rhs = wp.float64(0.0)
        for i in range(length):
            value_rhs += wp.dot(differences[a, i], residual[i])
        rhs[a] = value_rhs


@wp.kernel
def extrapolate(
    candidate: wp.array[wp.vec3d],
    steps: wp.array2d[wp.vec3d],
    coefficients: wp.array[wp.float64],
    out: wp.array[wp.vec3d],
):
    i = wp.tid()
    value = candidate[i]
    for j in range(coefficients.shape[0]):
        value -= coefficients[j] * steps[j, i]
    out[i] = value


@wp.kernel
def pack_newton(
    residual: wp.array[wp.vec3d],
    contact_force: wp.array[wp.vec3d],
    contact_residual: wp.array[wp.float64],
    rhs: wp.array[wp.float64],
):
    i = wp.tid()
    if i < residual.shape[0] * 3:
        rhs[i] = contact_force[i // 3][i % 3] - residual[i // 3][i % 3]
    else:
        rhs[i] = -contact_residual[i - 3 * residual.shape[0]]


@wp.kernel
def newton_update(
    q: wp.array[wp.vec3d],
    old_lam: wp.array[wp.float64],
    delta: wp.array[wp.float64],
    alpha: wp.float64,
    candidate: wp.array[wp.vec3d],
    lam: wp.array[wp.float64],
):
    i = wp.tid()
    if i < q.shape[0]:
        candidate[i] = q[i] + alpha * wp.vec3d(delta[3 * i], delta[3 * i + 1], delta[3 * i + 2])
    if i < old_lam.shape[0]:
        lam[i] = old_lam[i] + alpha * delta[3 * q.shape[0] + i]
