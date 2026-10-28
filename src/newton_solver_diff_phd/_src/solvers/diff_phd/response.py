# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Position-level Signorini--Coulomb contact with a semismooth Newton solve.

Geometry is frozen during a solve. The normal complementarity and tangential
disk projection define the residual used by both forward and adjoint solves.
"""

import functools
import gc

import numpy as np
import warp as wp
from warp.optim.linear import LinearOperator, gmres

from .contact import ContactRows
from .exceptions import ConvergenceError
from .linear import DenseFactorizationError, DenseLU


@wp.func
def contact_frame(normal: wp.vec3d):
    axis = wp.vec3d(1.0, 0.0, 0.0)
    if wp.abs(normal[0]) > wp.float64(0.8):
        axis = wp.vec3d(0.0, 1.0, 0.0)
    tangent = wp.normalize(wp.cross(normal, axis))
    return wp.matrix_from_cols(normal, tangent, wp.cross(normal, tangent))


@wp.kernel
def make_frames(normals: wp.array[wp.vec3d], frames: wp.array[wp.mat33d]):
    i = wp.tid()
    frames[i] = contact_frame(normals[i])


@wp.kernel
def make_columns(
    indices: wp.array[wp.vec4i], weights: wp.array[wp.vec4d], free: wp.array[wp.int32], columns: wp.array2d[wp.float64]
):
    c = wp.tid()
    for j in range(4):
        i = indices[c][j]
        if i >= 0:
            if free[i] != 0:
                wp.atomic_add(columns, i, c, weights[c][j])


@wp.kernel
def make_compliance(
    indices: wp.array[wp.vec4i],
    weights: wp.array[wp.vec4d],
    inverse_columns: wp.array2d[wp.float64],
    frames: wp.array[wp.mat33d],
    compliance: wp.float64,
    matrix: wp.array2d[wp.float64],
    scales: wp.array[wp.float64],
):
    c, d = wp.tid()
    value = wp.float64(0.0)
    for k in range(4):
        i = indices[c][k]
        if i >= 0:
            value += weights[c][k] * inverse_columns[i, d]
    block = value * wp.transpose(frames[c]) * frames[d]
    for i in range(3):
        for j in range(3):
            v = block[i, j]
            if c == d and i == j:
                v += compliance
            matrix[3 * c + i, 3 * d + j] = v
    if c == d:
        scales[c] = wp.float64(1.0) / wp.max(value + compliance, wp.float64(1.0e-30))


@wp.func
def _relative_displacement(
    q: wp.array[wp.vec3d],
    previous: wp.array[wp.vec3d],
    indices: wp.vec4i,
    weights: wp.vec4d,
    frame: wp.mat33d,
    offset: wp.float64,
    thickness: wp.float64,
):
    x = wp.vec3d(0.0)
    old = wp.vec3d(0.0)
    for j in range(4):
        i = indices[j]
        if i >= 0:
            x += weights[j] * q[i]
            old += weights[j] * previous[i]
    local = wp.transpose(frame) * x
    slip = wp.transpose(frame) * (x - old)
    return wp.vec3d(local[0] - offset - thickness, slip[1], slip[2])


@wp.kernel
def relative_displacement(
    q: wp.array[wp.vec3d],
    previous: wp.array[wp.vec3d],
    indices: wp.array[wp.vec4i],
    weights: wp.array[wp.vec4d],
    frames: wp.array[wp.mat33d],
    offsets: wp.array[wp.float64],
    thickness: wp.float64,
    out: wp.array[wp.float64],
):
    c = wp.tid()
    value = _relative_displacement(q, previous, indices[c], weights[c], frames[c], offsets[c], thickness)
    for a in range(3):
        out[3 * c + a] = value[a]


@wp.func
def _dense_row(matrix: wp.array2d[wp.float64], x: wp.array[wp.float64], i: int):
    value = wp.float64(0.0)
    for j in range(x.shape[0]):
        value += matrix[i, j] * x[j]
    return value


@wp.kernel
def multiply_dense(
    matrix: wp.array2d[wp.float64],
    x: wp.array[wp.float64],
    y: wp.array[wp.float64],
    alpha: wp.float64,
    beta: wp.float64,
    out: wp.array[wp.float64],
):
    i = wp.tid()
    value = _dense_row(matrix, x, i)
    old = wp.float64(0.0)
    if beta != wp.float64(0.0):
        old = beta * y[i]
    out[i] = alpha * value + old


@wp.func
def _natural_residual(lam: wp.vec3d, displacement: wp.vec3d, scale: wp.float64, friction: wp.float64):
    n = lam[0]
    z_n = n - scale * displacement[0]
    residual = wp.vec3d(0.0)
    residual[0] = n - wp.max(wp.float64(0.0), z_n)
    dq = wp.mat33d(0.0)
    dl = wp.mat33d(0.0)
    if z_n > wp.float64(0.0):
        dq[0, 0] = scale
    else:
        dl[0, 0] = wp.float64(1.0)
    z = wp.vec2d(lam[1] - scale * displacement[1], lam[2] - scale * displacement[2])
    length = wp.length(z)
    radius = friction * wp.max(n, wp.float64(0.0))
    projected = wp.vec2d(0.0)
    if radius <= wp.float64(0.0):
        dl[1, 1] = wp.float64(1.0)
        dl[2, 2] = wp.float64(1.0)
    elif length <= radius:
        projected = z
        dq[1, 1] = scale
        dq[2, 2] = scale
    else:
        direction = z / length
        projected = radius * direction
        for i in range(2):
            dl[i + 1, 0] = -friction * direction[i]
            for j in range(2):
                delta = wp.float64(0.0)
                if i == j:
                    delta = wp.float64(1.0)
                b = radius / length * (delta - direction[i] * direction[j])
                dl[i + 1, j + 1] = delta - b
                dq[i + 1, j + 1] = scale * b
    residual[1] = lam[1] - projected[0]
    residual[2] = lam[2] - projected[1]
    return residual, dq, dl


@wp.func
def _contact_jacobian_entry(
    matrix: wp.array2d[wp.float64], dq: wp.array[wp.mat33d], dl: wp.array[wp.mat33d], i: int, j: int
):
    c = i // 3
    a = i % 3
    value = wp.float64(0.0)
    for k in range(3):
        value += dq[c][a, k] * matrix[3 * c + k, j]
    if j // 3 == c:
        value += dl[c][a, j % 3]
    return value


@wp.kernel
def contact_jacobian(
    matrix: wp.array2d[wp.float64], dq: wp.array[wp.mat33d], dl: wp.array[wp.mat33d], jacobian: wp.array2d[wp.float64]
):
    i, j = wp.tid()
    jacobian[i, j] = _contact_jacobian_entry(matrix, dq, dl, i, j)


@wp.kernel(enable_backward=False)
def _contact_jacobian_rhs(
    matrix: wp.array2d[wp.float64],
    dq: wp.array[wp.mat33d],
    dl: wp.array[wp.mat33d],
    residual: wp.array[wp.float64],
    jacobian: wp.array2d[wp.float64],
    rhs: wp.array[wp.float64],
):
    i, j = wp.tid()
    jacobian[i, j] = _contact_jacobian_entry(matrix, dq, dl, i, j)
    if j == 0:
        rhs[i] = -residual[i]


@wp.kernel
def vector_update(
    x: wp.array[wp.float64], direction: wp.array[wp.float64], alpha: wp.float64, out: wp.array[wp.float64]
):
    i = wp.tid()
    out[i] = x[i] + alpha * direction[i]


@wp.kernel(enable_backward=False)
def _linear_defect_metrics(
    matrix: wp.array2d[wp.float64],
    direction: wp.array[wp.float64],
    rhs: wp.array[wp.float64],
    defect: wp.array[wp.float64],
    metrics: wp.array[wp.float64],
):
    i = wp.tid()
    value = -_dense_row(matrix, direction, i) + rhs[i]
    defect[i] = value
    error, magnitude = wp.abs(value), wp.abs(rhs[i])
    if not wp.isfinite(error) or not wp.isfinite(magnitude):
        error = wp.float64(wp.inf)
    wp.atomic_max(metrics, 0, error)
    wp.atomic_max(metrics, 1, magnitude)


@wp.func
def _scaled_residual(residual: wp.float64, displacement: wp.float64, scale: wp.float64):
    value = residual / scale
    # Projection branches and max reductions can otherwise hide a NaN gap.
    if not wp.isfinite(value) or not wp.isfinite(displacement):
        value = wp.float64(wp.inf)
    return wp.abs(value)


@wp.kernel(enable_backward=False)
def natural_residual_norm(
    lam: wp.array[wp.float64],
    displacement: wp.array[wp.float64],
    scales: wp.array[wp.float64],
    friction: wp.float64,
    residual: wp.array[wp.float64],
    d_displacement: wp.array[wp.mat33d],
    d_multiplier: wp.array[wp.mat33d],
    norm: wp.array[wp.float64],
):
    c = wp.tid()
    local = wp.vec3d(displacement[3 * c], displacement[3 * c + 1], displacement[3 * c + 2])
    value, dq, dl = _natural_residual(wp.vec3d(lam[3 * c], lam[3 * c + 1], lam[3 * c + 2]), local, scales[c], friction)
    error = wp.float64(0.0)
    for a in range(3):
        residual[3 * c + a] = value[a]
        error = wp.max(error, _scaled_residual(value[a], local[a], scales[c]))
    wp.atomic_max(norm, 0, error)
    d_displacement[c] = dq
    d_multiplier[c] = dl


@wp.kernel(enable_backward=False)
def physical_contact_residual(
    q: wp.array[wp.vec3d],
    previous: wp.array[wp.vec3d],
    indices: wp.array[wp.vec4i],
    weights: wp.array[wp.vec4d],
    frames: wp.array[wp.mat33d],
    offsets: wp.array[wp.float64],
    thickness: wp.float64,
    lam: wp.array[wp.float64],
    compliance: wp.float64,
    scales: wp.array[wp.float64],
    friction: wp.float64,
    displacement: wp.array[wp.float64],
    residual: wp.array[wp.float64],
    d_displacement: wp.array[wp.mat33d],
    d_multiplier: wp.array[wp.mat33d],
    norm: wp.array[wp.float64],
):
    c = wp.tid()
    local = _relative_displacement(q, previous, indices[c], weights[c], frames[c], offsets[c], thickness)
    multiplier = wp.vec3d(lam[3 * c], lam[3 * c + 1], lam[3 * c + 2])
    for a in range(3):
        local[a] = local[a] + compliance * multiplier[a]
        displacement[3 * c + a] = local[a]
    value, dq, dl = _natural_residual(multiplier, local, scales[c], friction)
    error = wp.float64(0.0)
    for a in range(3):
        residual[3 * c + a] = value[a]
        error = wp.max(error, _scaled_residual(value[a], local[a], scales[c]))
    wp.atomic_max(norm, 0, error)
    d_displacement[c] = dq
    d_multiplier[c] = dl


@wp.kernel
def apply_j(
    q: wp.array[wp.vec3d],
    indices: wp.array[wp.vec4i],
    weights: wp.array[wp.vec4d],
    frames: wp.array[wp.mat33d],
    out: wp.array[wp.float64],
):
    c = wp.tid()
    value = wp.vec3d(0.0)
    for k in range(4):
        i = indices[c][k]
        if i >= 0:
            value += weights[c][k] * q[i]
    value = wp.transpose(frames[c]) * value
    for k in range(3):
        out[3 * c + k] = value[k]


@wp.kernel
def apply_jt(
    lam: wp.array[wp.float64],
    indices: wp.array[wp.vec4i],
    weights: wp.array[wp.vec4d],
    frames: wp.array[wp.mat33d],
    out: wp.array[wp.vec3d],
):
    c = wp.tid()
    force = frames[c] * wp.vec3d(lam[3 * c], lam[3 * c + 1], lam[3 * c + 2])
    for k in range(4):
        i = indices[c][k]
        if i >= 0:
            wp.atomic_add(out, i, weights[c][k] * force)


@wp.kernel(enable_backward=False)
def _projected_sweeps(
    matrix: wp.array2d[wp.float64],
    base: wp.array[wp.float64],
    scales: wp.array[wp.float64],
    friction: wp.float64,
    lam: wp.array[wp.float64],
    sweeps: int,
):
    for _sweep in range(sweeps):
        for c in range(scales.shape[0]):
            value = wp.vec3d(base[3 * c], base[3 * c + 1], base[3 * c + 2])
            for j in range(lam.shape[0]):
                for a in range(3):
                    value[a] += matrix[3 * c + a, j] * lam[j]
            normal = wp.max(wp.float64(0.0), lam[3 * c] - scales[c] * value[0])
            z = wp.vec2d(lam[3 * c + 1] - scales[c] * value[1], lam[3 * c + 2] - scales[c] * value[2])
            radius = friction * normal
            length = wp.length(z)
            if length > radius:
                z *= radius / wp.max(length, wp.float64(1.0e-30))
            lam[3 * c] = normal
            lam[3 * c + 1] = z[0]
            lam[3 * c + 2] = z[1]


@functools.cache
def _make_projected_sweeps_tiled(size):
    @wp.kernel(module="unique", enable_backward=False)
    def projected_sweeps_tiled(
        transposed: wp.array2d[wp.float64],
        initial_displacement: wp.array[wp.float64],
        scales: wp.array[wp.float64],
        friction: wp.float64,
        lam: wp.array[wp.float64],
        sweeps: int,
    ):
        values = wp.tile_load(lam, shape=(size,))
        displacement = wp.tile_load(initial_displacement, shape=(size,))
        for _sweep in range(sweeps):
            for c in range(scales.shape[0]):
                old0 = values[3 * c]
                old1 = values[3 * c + 1]
                old2 = values[3 * c + 2]
                normal = wp.max(wp.float64(0.0), old0 - scales[c] * displacement[3 * c])
                z = wp.vec2d(old1 - scales[c] * displacement[3 * c + 1], old2 - scales[c] * displacement[3 * c + 2])
                radius = friction * normal
                length = wp.length(z)
                if length > radius:
                    z *= radius / wp.max(length, wp.float64(1.0e-30))
                column0 = wp.tile_load(transposed[3 * c], shape=(size,))
                column1 = wp.tile_load(transposed[3 * c + 1], shape=(size,))
                column2 = wp.tile_load(transposed[3 * c + 2], shape=(size,))
                updated = displacement + column0 * (normal - old0) + column1 * (z[0] - old1) + column2 * (z[1] - old2)
                wp.tile_assign(displacement, updated, offset=(0,))
                # Collective assignments synchronize the shared multipliers so
                # the next contact observes these Gauss--Seidel updates.
                wp.tile_assign(values, wp.tile_full(shape=(1,), dtype=wp.float64, value=normal), offset=(3 * c,))
                wp.tile_assign(values, wp.tile_full(shape=(1,), dtype=wp.float64, value=z[0]), offset=(3 * c + 1,))
                wp.tile_assign(values, wp.tile_full(shape=(1,), dtype=wp.float64, value=z[1]), offset=(3 * c + 2,))
        wp.tile_store(lam, values)

    return projected_sweeps_tiled


@wp.kernel(enable_backward=False)
def _transpose_matrix(source: wp.array2d[wp.float64], target: wp.array2d[wp.float64]):
    i, j = wp.tid()
    target[j, i] = source[i, j]


@wp.func
def _initialize_line_search(
    error: wp.float64, tolerance: wp.float64, flags: wp.array[int], values: wp.array[wp.float64]
):
    flags[0] = 1
    flags[1] = 0
    values[0] = error
    values[1] = tolerance
    values[2] = wp.float64(1.0)
    values[3] = wp.float64(wp.inf)
    values[4] = wp.float64(0.0)
    values[5] = wp.float64(0.0)


@wp.kernel(enable_backward=False)
def _begin_line_search(error: wp.float64, tolerance: wp.float64, flags: wp.array[int], values: wp.array[wp.float64]):
    _initialize_line_search(error, tolerance, flags, values)


@wp.kernel(enable_backward=False)
def _line_search_trial(
    lam: wp.array[wp.float64],
    direction: wp.array[wp.float64],
    values: wp.array[wp.float64],
    trial: wp.array[wp.float64],
):
    i = wp.tid()
    trial[i] = lam[i] + values[2] * direction[i]


@wp.kernel(enable_backward=False)
def _check_line_search(norm: wp.array[wp.float64], flags: wp.array[int], values: wp.array[wp.float64]):
    error, tolerance, alpha = values[0], values[1], values[2]
    candidate = norm[0]
    values[3] = candidate
    values[5] += wp.float64(1.0)
    if candidate <= tolerance or (
        candidate < error and candidate <= error * (wp.float64(1.0) - wp.float64(1.0e-4) * alpha)
    ):
        flags[0] = 0
        flags[1] = 1
        values[4] = wp.float64(1.0)
    elif values[5] >= wp.float64(53.0):
        flags[0] = 0
    else:
        values[2] = wp.float64(0.5) * alpha


@wp.func
def _initialize_polish(
    tolerance: wp.float64, active: wp.array[int], values: wp.array[wp.float64], budget: wp.array[int]
):
    active[0] = int(budget[0] != 0)
    values[0] = tolerance
    values[1] = wp.float64(wp.inf)
    values[2] = wp.float64(0.0)
    values[3] = wp.float64(0.0)
    values[4] = wp.float64(int(budget[0] == 0))


@wp.kernel(enable_backward=False)
def _begin_polish(tolerance: wp.float64, active: wp.array[int], values: wp.array[wp.float64], budget: wp.array[int]):
    _initialize_polish(tolerance, active, values, budget)


@wp.kernel(enable_backward=False)
def _check_polish(
    norm: wp.array[wp.float64], active: wp.array[int], values: wp.array[wp.float64], budget: wp.array[int]
):
    values[1] = norm[0]
    values[2] += wp.float64(1.0)
    budget[1] += 1
    if budget[0] > 0:
        budget[0] -= 1
    if norm[0] <= values[0]:
        active[0] = 0
        values[3] = wp.float64(1.0)
    elif budget[0] == 0:
        active[0] = 0
        values[4] = wp.float64(1.0)
    elif values[2] >= wp.float64(1000.0):
        active[0] = 0


# Newton flags: active, factor, refine, direction, polish, finalize, resume,
# completed iterations, slow iterations, iteration limit.
# Newton values: tolerance, current residual, status (0 running, 1 converged,
# 2 needs host Krylov fallback, -1 nonfinite, -2 exhausted, -3 work limit).
@wp.kernel(enable_backward=False)
def _begin_contact_newton(
    tolerance: wp.float64,
    iterations: int,
    flags: wp.array[int],
    values: wp.array[wp.float64],
    work_limit: int,
    budget: wp.array[int],
):
    for i in range(10):
        flags[i] = 0
    flags[0] = int(iterations > 0)
    flags[5] = int(iterations <= 0)
    flags[9] = iterations
    values[0] = tolerance
    values[1] = wp.float64(wp.inf)
    values[2] = wp.float64(0.0)
    budget[0] = work_limit
    budget[1] = 0


@wp.kernel(enable_backward=False)
def _check_contact_newton(norm: wp.array[wp.float64], flags: wp.array[int], values: wp.array[wp.float64]):
    error = norm[0]
    values[1] = error
    for i in range(1, 5):
        flags[i] = 0
    if not wp.isfinite(error):
        flags[0] = 0
        values[2] = wp.float64(-1.0)
    elif error <= values[0]:
        flags[0] = 0
        values[2] = wp.float64(1.0)
    elif flags[6] != 0:
        flags[3] = 1
        flags[6] = 0
    else:
        flags[1] = 1


@wp.kernel(enable_backward=False)
def _check_contact_linear(
    metrics: wp.array[wp.float64], refined: bool, flags: wp.array[int], values: wp.array[wp.float64]
):
    error, magnitude = metrics[0], metrics[1]
    accurate = wp.isfinite(error) and error <= wp.float64(1.0e-9) * wp.max(magnitude, wp.float64(1.0e-30))
    flags[2] = 0
    if accurate:
        flags[3] = 1
    elif not refined and wp.isfinite(error):
        flags[2] = 1
    else:
        flags[0] = 0
        values[2] = wp.float64(2.0)


@wp.kernel(enable_backward=False)
def _begin_contact_line(values: wp.array[wp.float64], line_flags: wp.array[int], line_values: wp.array[wp.float64]):
    _initialize_line_search(values[1], values[0], line_flags, line_values)


@wp.kernel(enable_backward=False)
def _check_contact_line(line_values: wp.array[wp.float64], flags: wp.array[int], values: wp.array[wp.float64]):
    accepted = line_values[4] != wp.float64(0.0)
    if accepted and line_values[2] == wp.float64(1.0) and line_values[3] > wp.float64(0.8) * values[1]:
        flags[8] += 1
    else:
        flags[8] = 0
    flags[4] = int(not accepted or flags[8] >= 2)


@wp.kernel(enable_backward=False)
def _begin_contact_polish(
    values: wp.array[wp.float64], active: wp.array[int], polish_values: wp.array[wp.float64], budget: wp.array[int]
):
    _initialize_polish(values[0], active, polish_values, budget)


@wp.kernel(enable_backward=False)
def _finish_contact_polish(polish_values: wp.array[wp.float64], flags: wp.array[int], values: wp.array[wp.float64]):
    flags[8] = 0
    if polish_values[3] != wp.float64(0.0):
        flags[0] = 0
        values[1] = polish_values[1]
        values[2] = wp.float64(1.0)
    elif polish_values[4] != wp.float64(0.0):
        flags[0] = 0
        values[1] = polish_values[1]
        values[2] = wp.float64(-3.0)


@wp.kernel(enable_backward=False)
def _finish_contact_iteration(flags: wp.array[int], values: wp.array[wp.float64]):
    flags[7] += 1
    if values[2] == wp.float64(0.0) and flags[7] >= flags[9]:
        flags[0] = 0
        flags[5] = 1


@wp.kernel(enable_backward=False)
def _check_contact_final(norm: wp.array[wp.float64], flags: wp.array[int], values: wp.array[wp.float64]):
    values[1] = norm[0]
    flags[4] = 0
    if norm[0] > values[0]:
        flags[4] = 1
    else:
        values[2] = wp.float64(1.0)


@wp.kernel(enable_backward=False)
def _finish_contact_final(
    norm: wp.array[wp.float64], polish_values: wp.array[wp.float64], values: wp.array[wp.float64]
):
    values[1] = norm[0]
    values[2] = wp.float64(-2.0)
    if polish_values[3] != wp.float64(0.0):
        values[2] = wp.float64(1.0)
    elif polish_values[4] != wp.float64(0.0):
        values[2] = wp.float64(-3.0)


@wp.kernel(enable_backward=False)
def _resume_contact_newton(flags: wp.array[int], values: wp.array[wp.float64]):
    flags[0] = 1
    flags[6] = 1
    values[2] = wp.float64(0.0)


class ContactConvergenceError(ConvergenceError):
    """The auxiliary fixed-PD contact NCP exhausted its nonlinear solve."""


class ContactWorkLimit(ContactConvergenceError):
    """An optional auxiliary work limit stopped an unconverged contact trial."""


class ContactResponse:
    """Solve the frozen contact NCP using exact PD compliance and native Warp LU."""

    def __init__(self, rows, inverse, free, *, thickness, friction, compliance=1.0e-12):
        self.count = rows.count
        self.device = inverse.device
        n, c = inverse.size, self.count
        self._particle_count = n
        self.rows = ContactRows(
            c,
            *(
                wp.empty(c, dtype=getattr(rows, name).dtype, device=self.device)
                for name in ("indices", "weights", "normals", "offsets", "kinds", "times")
            ),
            rows.thickness,
        )
        self._columns = wp.empty((n, c), dtype=wp.float64, device=self.device)
        self._inverse_columns = wp.empty_like(self._columns)
        self.frames = wp.empty(c, dtype=wp.mat33d, device=self.device)
        self.scales = wp.empty(c, dtype=wp.float64, device=self.device)
        self.matrix = wp.empty((3 * c, 3 * c), dtype=wp.float64, device=self.device)
        self.dq = wp.empty(c, dtype=wp.mat33d, device=self.device)
        self.dl = wp.empty(c, dtype=wp.mat33d, device=self.device)
        self.lam = wp.zeros(3 * c, dtype=wp.float64, device=self.device)
        self.displacement = wp.empty_like(self.lam)
        self.residual = wp.empty_like(self.lam)
        self.base = wp.empty_like(self.lam)
        self.norm_buffer = wp.zeros(1, dtype=wp.float64, device=self.device)
        self._sweep_matrix = None
        self._use_conditional_graphs = self.device.is_cuda and wp.is_conditional_graph_supported()
        self._line_graph = None
        self._polish_graph = None
        self._newton_graph = None
        self._polish_work_remaining = None
        self._polish_work_used = 0
        self.update(rows, inverse, free, thickness=thickness, friction=friction, compliance=compliance)

    def update(self, rows, inverse, free, *, thickness, friction, compliance=1.0e-12):
        """Replace frozen geometry without changing captured workspace pointers."""
        if rows.count != self.count or inverse.size != self._particle_count or inverse.device != self.device:
            raise ValueError("ContactResponse.update requires unchanged contact count, particle count, and device")
        if free.device != self.device or len(free) != self._particle_count:
            raise ValueError("ContactResponse.update requires a compatible free-particle mask")
        self.thickness = thickness
        self.friction = friction
        self.compliance = compliance
        names = ("indices", "weights", "normals", "offsets", "kinds", "times")
        c = self.count
        if c:
            for name in names:
                wp.copy(getattr(self.rows, name), getattr(rows, name), count=c)
        self.rows = ContactRows(c, *(getattr(self.rows, name) for name in names), rows.thickness)
        self.lam.zero_()
        self.norm_buffer.zero_()
        if c:
            self._columns.zero_()
            wp.launch(make_frames, c, inputs=[self.rows.normals, self.frames], device=self.device)
            wp.launch(
                make_columns, c, inputs=[self.rows.indices, self.rows.weights, free, self._columns], device=self.device
            )
            inverse.apply_batch(self._columns, self._inverse_columns)
            wp.launch(
                make_compliance,
                (c, c),
                inputs=[
                    self.rows.indices,
                    self.rows.weights,
                    self._inverse_columns,
                    self.frames,
                    compliance,
                    self.matrix,
                    self.scales,
                ],
                device=self.device,
            )

    def _evaluate(self, lam):
        wp.launch(
            multiply_dense,
            3 * self.count,
            inputs=[self.matrix, lam, self.base, 1.0, 1.0, self.displacement],
            device=self.device,
        )
        self.norm_buffer.zero_()
        wp.launch(
            natural_residual_norm,
            self.count,
            inputs=[
                lam,
                self.displacement,
                self.scales,
                self.friction,
                self.residual,
                self.dq,
                self.dl,
                self.norm_buffer,
            ],
            device=self.device,
        )

    def evaluate(self, lam):
        self._evaluate(lam)
        return float(self.norm_buffer.numpy()[0])

    def _capture_loop(self, body):
        # Destructors for older CUDA graphs must not run inside this capture.
        enabled = gc.isenabled()
        gc.disable()
        try:
            with wp.ScopedCapture(device=self.device) as capture:
                body()
            return capture.graph
        finally:
            if enabled:
                gc.enable()

    def _line_search(self, direction, trial, error, tolerance):
        if not self._use_conditional_graphs:
            for backtrack in range(53):
                alpha = 0.5**backtrack
                wp.launch(vector_update, 3 * self.count, inputs=[self.lam, direction, alpha, trial], device=self.device)
                candidate = self.evaluate(trial)
                if candidate <= tolerance or (candidate < error and candidate <= error * (1.0 - 1.0e-4 * alpha)):
                    wp.copy(self.lam, trial)
                    return True, alpha, candidate
            return False, alpha, candidate

        if self._line_graph is None or self._line_friction != self.friction:
            self._prepare_line_search()
            self._line_graph = self._capture_loop(self._record_line_search)
            self._line_friction = self.friction
        wp.copy(self._line_direction, direction)
        wp.launch(
            _begin_line_search, 1, inputs=[error, tolerance, self._line_flags, self._line_values], device=self.device
        )
        wp.capture_launch(self._line_graph)
        values = self._line_values.numpy()
        return bool(values[4]), float(values[2]), float(values[3])

    def _prepare_line_search(self):
        if not hasattr(self, "_line_values"):
            self._line_direction = wp.empty_like(self.lam)
            self._line_trial = wp.empty_like(self.lam)
            self._line_flags = wp.zeros(2, dtype=int, device=self.device)
            self._line_active = self._line_flags[:1]
            self._line_accepted = self._line_flags[1:2]
            # error, tolerance, alpha, candidate, accepted, trials
            self._line_values = wp.empty(6, dtype=wp.float64, device=self.device)
        wp.load_module(module=__name__, device=self.device)

    def _record_line_search(self):
        wp.capture_while(self._line_active, self._line_search_iteration)
        wp.capture_if(self._line_accepted, lambda: wp.copy(self.lam, self._line_trial))

    def _line_search_iteration(self):
        wp.launch(
            _line_search_trial,
            3 * self.count,
            inputs=[self.lam, self._line_direction, self._line_values, self._line_trial],
            device=self.device,
        )
        self._evaluate(self._line_trial)
        wp.launch(
            _check_line_search, 1, inputs=[self.norm_buffer, self._line_flags, self._line_values], device=self.device
        )

    def solve(self, q_free, previous, *, tolerance=1.0e-9, iterations=40, max_polish_batches=None):
        """Solve the auxiliary contact problem, optionally bounding GS work.

        ``max_polish_batches`` counts cumulative batches of ten ordered sweeps
        across all polishing attempts in this call. Exhaustion raises
        :class:`ContactWorkLimit` without accepting an unconverged state.
        The default retains the complete original iteration schedule.
        """
        if max_polish_batches is not None:
            if (
                not np.isfinite(max_polish_batches)
                or max_polish_batches < 0
                or max_polish_batches > 2147483647
                or int(max_polish_batches) != max_polish_batches
            ):
                raise ValueError("max_polish_batches must be a nonnegative int32 integer or None")
            max_polish_batches = int(max_polish_batches)
        self._polish_work_remaining = max_polish_batches
        self._polish_work_used = 0
        if not self.count:
            return
        c = self.count
        wp.launch(
            relative_displacement,
            c,
            inputs=[
                q_free,
                previous,
                self.rows.indices,
                self.rows.weights,
                self.frames,
                self.rows.offsets,
                self.thickness,
                self.base,
            ],
            device=self.device,
        )
        if self._use_conditional_graphs:
            self._solve_cuda(tolerance, iterations, max_polish_batches=max_polish_batches)
            return
        jacobian = wp.empty_like(self.matrix)
        direction = wp.empty_like(self.lam)
        rhs = wp.empty_like(self.lam)
        trial = wp.empty_like(self.lam)
        defect = wp.empty_like(self.lam)
        correction = wp.empty_like(self.lam)
        linear_metrics = wp.zeros(2, dtype=wp.float64, device=self.device)

        def matvec(x, y, z, alpha, beta):
            wp.launch(multiply_dense, 3 * c, inputs=[jacobian, x, y, alpha, beta, z], device=self.device)

        operator = LinearOperator((3 * c, 3 * c), wp.float64, self.device, matvec)
        slow_iterations = 0
        for _ in range(iterations):
            error = self.evaluate(self.lam)
            if not np.isfinite(error):
                raise RuntimeError("DiffPhD contact residual is nonfinite")
            if error <= tolerance:
                return
            wp.launch(
                _contact_jacobian_rhs,
                (3 * c, 3 * c),
                inputs=[self.matrix, self.dq, self.dl, self.residual, jacobian, rhs],
                device=self.device,
            )
            accurate = False
            try:
                factor = DenseLU(jacobian)
                factor.solve(rhs, direction)
                # The contact Jacobian is already dense. Pivoted LU avoids
                # repeated Krylov launches, with one same-factor refinement
                # for cancellation in nearly redundant contact modes.
                for refinement in range(2):
                    linear_metrics.zero_()
                    wp.launch(
                        _linear_defect_metrics,
                        3 * c,
                        inputs=[jacobian, direction, rhs, defect, linear_metrics],
                        device=self.device,
                    )
                    linear_error, magnitude = linear_metrics.numpy()
                    accurate = np.isfinite(linear_error) and linear_error <= 1.0e-9 * max(magnitude, 1.0e-30)
                    if accurate:
                        break
                    if refinement == 0 and np.isfinite(linear_error):
                        factor.solve(defect, correction)
                        wp.launch(
                            vector_update, 3 * c, inputs=[direction, correction, 1.0, direction], device=self.device
                        )
            except DenseFactorizationError:
                pass
            if not accurate:
                direction.zero_()
                # Singular selected branches retain the Krylov fallback.
                # Multiplier units scale with mass and stiffness, so an
                # absolute force floor must not suppress a small-body update.
                gmres(
                    operator,
                    rhs,
                    direction,
                    tol=1.0e-9,
                    atol=0.0,
                    maxiter=max(60, 3 * c),
                    restart=min(31, 3 * c),
                    use_cuda_graph=False,
                )
            # A contact branch can be approached through progressively smaller
            # representable steps before a full Newton update becomes valid.
            # Search down to float64 resolution; strict residual decrease also
            # rejects rounded no-op trials.
            accepted, alpha, candidate = self._line_search(direction, trial, error, tolerance)
            # Small accepted alphas can be converging to a nearby branch event;
            # only poorly contracting full Newton steps count as stagnation.
            slow_iterations = slow_iterations + 1 if accepted and alpha == 1.0 and candidate > 0.8 * error else 0
            if not accepted or slow_iterations >= 2:
                # Coulomb friction is nonassociated: a selected semismooth
                # branch can be a poor Newton basin despite a small residual.
                # Cone sweeps can cross that branch with a nonmonotone residual.
                # Resume Newton afterwards to remove its remaining slow modes.
                if self._polish(tolerance):
                    return
                slow_iterations = 0
        error = self.evaluate(self.lam)
        if error > tolerance:
            if self._polish(tolerance):
                return
            error = self.evaluate(self.lam)
            raise ContactConvergenceError(f"DiffPhD contact did not converge: gap residual {error:.3e} m")

    def _polish(self, tolerance):
        # Redundant coplanar surface constraints can make a selected Newton
        # derivative nearly singular. Cone Gauss--Seidel provides a bounded
        # branch transition without changing the equations. The residual may
        # temporarily increase before a stick/slide transition, so it is not a
        # descent line search. Newton resumes if these sweeps do not converge.
        # Update W*lambda incrementally between exact ten-sweep residual checks.
        # The transposed frozen matrix makes each column update coalesced.
        if self._use_conditional_graphs:
            return self._polish_cuda(tolerance)
        tiled = self.device.is_cuda and 3 * self.count <= 256
        kernel = _make_projected_sweeps_tiled(1 << (3 * self.count - 1).bit_length()) if tiled else _projected_sweeps
        inputs = [self.matrix, self.base, self.scales, self.friction, self.lam, 10]
        if tiled:
            if self._sweep_matrix is None:
                self._sweep_matrix = wp.empty_like(self.matrix)
            wp.launch(
                _transpose_matrix, self.matrix.shape, inputs=[self.matrix, self._sweep_matrix], device=self.device
            )
            wp.launch(
                multiply_dense,
                3 * self.count,
                inputs=[self.matrix, self.lam, self.base, 1.0, 1.0, self.displacement],
                device=self.device,
            )
            inputs = [self._sweep_matrix, self.displacement, self.scales, self.friction, self.lam, 10]
        for _ in range(1000):
            if self._polish_work_remaining == 0:
                raise ContactWorkLimit("DiffPhD auxiliary contact work limit reached")
            if tiled:
                wp.launch_tiled(kernel, dim=1, inputs=inputs, block_dim=32, device=self.device)
            else:
                wp.launch(kernel, 1, inputs=inputs, device=self.device)
            self._polish_work_used += 1
            if self._polish_work_remaining is not None:
                self._polish_work_remaining -= 1
            if self.evaluate(self.lam) <= tolerance:
                return True
            if self._polish_work_remaining == 0:
                raise ContactWorkLimit("DiffPhD auxiliary contact work limit reached")
        return False

    def _polish_cuda(self, tolerance):
        if self._polish_graph is None or self._polish_friction != self.friction:
            self._prepare_polish()
            self._polish_graph = self._capture_loop(self._record_polish)
            self._polish_friction = self.friction
        wp.launch(
            _begin_polish,
            1,
            inputs=[tolerance, self._polish_active, self._polish_values, self._polish_budget],
            device=self.device,
        )
        wp.capture_launch(self._polish_graph)
        values = self._polish_values.numpy()
        if values[4]:
            raise ContactWorkLimit("DiffPhD auxiliary contact work limit reached")
        return bool(values[3])

    def _prepare_polish(self):
        if not hasattr(self, "_polish_values"):
            self._polish_active = wp.zeros(1, dtype=int, device=self.device)
            # tolerance, candidate, batches of ten sweeps, converged, work limit
            self._polish_values = wp.empty(5, dtype=wp.float64, device=self.device)
            # Remaining batches (-1 means unlimited), cumulative batches used.
            self._polish_budget = wp.array([-1, 0], dtype=int, device=self.device)
            tiled = 3 * self.count <= 256
            self._polish_kernel = (
                _make_projected_sweeps_tiled(1 << (3 * self.count - 1).bit_length()) if tiled else _projected_sweeps
            )
            if tiled and self._sweep_matrix is None:
                self._sweep_matrix = wp.empty_like(self.matrix)
        wp.load_module(module=__name__, device=self.device)
        wp.load_module(module=self._polish_kernel.module, device=self.device)

    def _record_polish(self):
        if 3 * self.count <= 256:
            wp.launch(
                _transpose_matrix, self.matrix.shape, inputs=[self.matrix, self._sweep_matrix], device=self.device
            )
            wp.launch(
                multiply_dense,
                3 * self.count,
                inputs=[self.matrix, self.lam, self.base, 1.0, 1.0, self.displacement],
                device=self.device,
            )
        wp.capture_while(self._polish_active, self._polish_iteration)

    def _polish_iteration(self):
        if 3 * self.count <= 256:
            wp.launch_tiled(
                self._polish_kernel,
                dim=1,
                inputs=[self._sweep_matrix, self.displacement, self.scales, self.friction, self.lam, 10],
                block_dim=32,
                device=self.device,
            )
        else:
            wp.launch(
                self._polish_kernel,
                1,
                inputs=[self.matrix, self.base, self.scales, self.friction, self.lam, 10],
                device=self.device,
            )
        self._evaluate(self.lam)
        wp.launch(
            _check_polish,
            1,
            inputs=[self.norm_buffer, self._polish_active, self._polish_values, self._polish_budget],
            device=self.device,
        )

    def _solve_cuda(self, tolerance, iterations, *, max_polish_batches=None):
        if self._newton_graph is None or self._newton_friction != self.friction:
            self._prepare_line_search()
            self._prepare_polish()
            self._newton_jacobian = wp.empty_like(self.matrix)
            self._newton_rhs = wp.empty_like(self.lam)
            self._newton_defect = wp.empty_like(self.lam)
            self._newton_correction = wp.empty_like(self.lam)
            self._newton_metrics = wp.zeros(2, dtype=wp.float64, device=self.device)
            self._newton_factor = DenseLU(self.matrix, check=False)
            self._newton_flags = wp.zeros(10, dtype=int, device=self.device)
            self._newton_active = self._newton_flags[:1]
            self._newton_factor_active = self._newton_flags[1:2]
            self._newton_refine_active = self._newton_flags[2:3]
            self._newton_direction_active = self._newton_flags[3:4]
            self._newton_polish_active = self._newton_flags[4:5]
            self._newton_finalize_active = self._newton_flags[5:6]
            self._newton_values = wp.empty(3, dtype=wp.float64, device=self.device)
            self._newton_graph = self._capture_loop(self._record_newton)
            self._newton_friction = self.friction
        wp.launch(
            _begin_contact_newton,
            1,
            inputs=[
                tolerance,
                iterations,
                self._newton_flags,
                self._newton_values,
                -1 if max_polish_batches is None else max_polish_batches,
                self._polish_budget,
            ],
            device=self.device,
        )
        while True:
            wp.capture_launch(self._newton_graph)
            _, error, status = self._newton_values.numpy()
            if status != 2:
                break
            # Krylov's allocation/checkpoint path cannot be nested inside a
            # conditional CUDA graph. Resume the same Newton iteration after
            # this exceptional solve, retaining its selected Jacobian.
            self._line_direction.zero_()

            def matvec(x, y, z, alpha, beta):
                wp.launch(
                    multiply_dense,
                    3 * self.count,
                    inputs=[self._newton_jacobian, x, y, alpha, beta, z],
                    device=self.device,
                )

            gmres(
                LinearOperator(self.matrix.shape, wp.float64, self.device, matvec),
                self._newton_rhs,
                self._line_direction,
                tol=1.0e-9,
                atol=0.0,
                maxiter=max(60, 3 * self.count),
                restart=min(31, 3 * self.count),
                use_cuda_graph=False,
            )
            wp.launch(_resume_contact_newton, 1, inputs=[self._newton_flags, self._newton_values], device=self.device)
        if status == -1:
            raise RuntimeError("DiffPhD contact residual is nonfinite")
        if status == -3:
            raise ContactWorkLimit("DiffPhD auxiliary contact work limit reached")
        if status != 1:
            raise ContactConvergenceError(f"DiffPhD contact did not converge: gap residual {error:.3e} m")

    def _record_newton(self):
        # Accepted line trials and polish batches already leave the residual
        # and derivatives at the current multipliers for the next iteration.
        # A graph entry also covers the initial solve and a Krylov resume.
        self._evaluate(self.lam)
        wp.capture_while(self._newton_active, self._newton_iteration)
        wp.capture_if(self._newton_finalize_active, self._newton_finalize)

    def _newton_iteration(self):
        wp.launch(
            _check_contact_newton,
            1,
            inputs=[self.norm_buffer, self._newton_flags, self._newton_values],
            device=self.device,
        )
        wp.capture_if(self._newton_factor_active, self._newton_factorize)
        wp.capture_if(self._newton_direction_active, self._newton_apply_direction)

    def _newton_factorize(self):
        wp.launch(
            _contact_jacobian_rhs,
            self.matrix.shape,
            inputs=[self.matrix, self.dq, self.dl, self.residual, self._newton_jacobian, self._newton_rhs],
            device=self.device,
        )
        self._newton_factor.refactor(self._newton_jacobian, check=False)
        self._newton_factor.solve(self._newton_rhs, self._line_direction)
        self._newton_check_linear(False)
        wp.capture_if(self._newton_refine_active, self._newton_refine)

    def _newton_check_linear(self, refined):
        self._newton_metrics.zero_()
        wp.launch(
            _linear_defect_metrics,
            3 * self.count,
            inputs=[
                self._newton_jacobian,
                self._line_direction,
                self._newton_rhs,
                self._newton_defect,
                self._newton_metrics,
            ],
            device=self.device,
        )
        wp.launch(
            _check_contact_linear,
            1,
            inputs=[self._newton_metrics, refined, self._newton_flags, self._newton_values],
            device=self.device,
        )

    def _newton_refine(self):
        self._newton_factor.solve(self._newton_defect, self._newton_correction)
        wp.launch(
            vector_update,
            3 * self.count,
            inputs=[self._line_direction, self._newton_correction, 1.0, self._line_direction],
            device=self.device,
        )
        self._newton_check_linear(True)

    def _newton_apply_direction(self):
        wp.launch(
            _begin_contact_line,
            1,
            inputs=[self._newton_values, self._line_flags, self._line_values],
            device=self.device,
        )
        self._record_line_search()
        wp.launch(
            _check_contact_line,
            1,
            inputs=[self._line_values, self._newton_flags, self._newton_values],
            device=self.device,
        )
        wp.capture_if(self._newton_polish_active, self._newton_polish)
        wp.launch(_finish_contact_iteration, 1, inputs=[self._newton_flags, self._newton_values], device=self.device)

    def _newton_polish(self):
        wp.launch(
            _begin_contact_polish,
            1,
            inputs=[self._newton_values, self._polish_active, self._polish_values, self._polish_budget],
            device=self.device,
        )
        self._record_polish()
        wp.launch(
            _finish_contact_polish,
            1,
            inputs=[self._polish_values, self._newton_flags, self._newton_values],
            device=self.device,
        )

    def _newton_finalize(self):
        self._evaluate(self.lam)
        wp.launch(
            _check_contact_final,
            1,
            inputs=[self.norm_buffer, self._newton_flags, self._newton_values],
            device=self.device,
        )
        wp.capture_if(self._newton_polish_active, self._newton_final_polish)

    def _newton_final_polish(self):
        wp.launch(
            _begin_contact_polish,
            1,
            inputs=[self._newton_values, self._polish_active, self._polish_values, self._polish_budget],
            device=self.device,
        )
        self._record_polish()
        self._evaluate(self.lam)
        wp.launch(
            _finish_contact_final,
            1,
            inputs=[self.norm_buffer, self._polish_values, self._newton_values],
            device=self.device,
        )

    def force(self, out):
        out.zero_()
        if self.count:
            wp.launch(
                apply_jt,
                self.count,
                inputs=[self.lam, self.rows.indices, self.rows.weights, self.frames, out],
                device=self.device,
            )
