# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Certified normal-contact QP for the optional swept contact model.

Minimize .5 lambda.T W lambda + q.T lambda with lambda >= 0. The active set
adds one violating row and uses a primal-feasible step to the first bound
before removing rows. No fixed pivot budget is presented as convergence.
Dependent entering rows first exchange along a feasible null direction. If
that exchange is unavailable, projected coordinate descent runs without diagonal
regularization. Every path must meet the same physical KKT tolerance; exhausting
a budget is a failure. Only the frozen normal QP is certified.
"""

import numpy as np
import warp as wp

TILE = 64
PRICE_BLOCK = 128
# The reduced matrix has unit diagonal. Retain a small binary64 roundoff
# guard (about 45 eps), so a feasible bound step can exchange nearly parallel
# rows. The independently recomputed physical KKT tolerance still decides
# success; this factorization guard is not a contact tolerance.
PIVOT_ROUNDOFF = wp.constant(1.0e-14)
ITERATION_LIMIT = 1
DEPENDENT_SYSTEM = 2
IMMOBILE_CONTACT = 4
NONFINITE = 8


@wp.kernel(enable_backward=False)
def _begin(
    active: wp.array[int],
    status: wp.array[int],
    iterations: wp.array[int],
    count: wp.array[int],
    needs: wp.array[int],
    residual: wp.array[wp.float64],
):
    active[0] = 1
    status[0] = 0
    iterations[0] = 0
    count[0] = 0
    needs[0] = 0
    residual[0] = wp.float64(wp.inf)


@wp.kernel(enable_backward=False)
def _initialize(
    W: wp.array2d[wp.float64],
    q: wp.array[wp.float64],
    candidate: wp.array[int],
    tolerance: wp.float64,
    lam: wp.array[wp.float64],
    scale: wp.array[wp.float64],
    status: wp.array[int],
):
    i = wp.tid()
    lam[i] = wp.float64(0.0)
    scale[i] = wp.float64(0.0)
    if candidate[i] != 0:
        d = W[i, i]
        if not wp.isfinite(d) or not wp.isfinite(q[i]):
            wp.atomic_or(status, 0, NONFINITE)
        elif d < wp.float64(0.0):
            wp.atomic_or(status, 0, DEPENDENT_SYSTEM)
        elif d == wp.float64(0.0):
            if q[i] < -tolerance:
                wp.atomic_or(status, 0, IMMOBILE_CONTACT)
        else:
            scale[i] = wp.sqrt(d)


@wp.kernel(enable_backward=False)
def _gradient(
    W: wp.array2d[wp.float64],
    q: wp.array[wp.float64],
    lam: wp.array[wp.float64],
    order: wp.array[int],
    count: wp.array[int],
    candidate: wp.array[int],
    gap: wp.array[wp.float64],
):
    i = wp.tid()
    value = q[i]
    if candidate[i] != 0:
        for k in range(count[0]):
            j = order[k]
            value += W[i, j] * lam[j]
    gap[i] = value


@wp.kernel(enable_backward=False)
def _price(
    W: wp.array2d[wp.float64],
    gap: wp.array[wp.float64],
    lam: wp.array[wp.float64],
    scale: wp.array[wp.float64],
    candidate: wp.array[int],
    size: int,
    tolerance: wp.float64,
    maximum: int,
    order: wp.array[int],
    count: wp.array[int],
    needs: wp.array[int],
    active: wp.array[int],
    status: wp.array[int],
    iterations: wp.array[int],
    residual: wp.array[wp.float64],
):
    if status[0] != 0:
        active[0] = 0
        return
    largest = wp.float64(0.0)
    entering = int(-1)
    most_negative = wp.float64(0.0)
    for i in range(size):
        if candidate[i] != 0:
            if not wp.isfinite(gap[i]) or not wp.isfinite(lam[i]):
                status[0] = NONFINITE
                active[0] = 0
                residual[0] = wp.float64(wp.inf)
                return
            natural = wp.abs(wp.min(gap[i], W[i, i] * lam[i]))
            largest = wp.max(largest, natural)
            if lam[i] == wp.float64(0.0) and scale[i] > wp.float64(0.0) and gap[i] < -tolerance:
                value = gap[i] / scale[i]
                if value < most_negative:
                    entering = i
                    most_negative = value
    residual[0] = largest
    if needs[0] == 0:
        if largest <= tolerance:
            active[0] = 0
            return
        if entering < 0:
            # The reduced solve was numerically stationary only to an error
            # larger than the requested physical contact tolerance.
            status[0] = NONFINITE
            active[0] = 0
            return
        order[count[0]] = entering
        count[0] += 1
    if iterations[0] >= maximum:
        status[0] = ITERATION_LIMIT
        active[0] = 0
        return
    iterations[0] += 1


@wp.kernel(enable_backward=False)
def _price_tiled(
    W: wp.array2d[wp.float64],
    gap: wp.array[wp.float64],
    lam: wp.array[wp.float64],
    scale: wp.array[wp.float64],
    candidate: wp.array[int],
    size: int,
    tolerance: wp.float64,
    maximum: int,
    order: wp.array[int],
    count: wp.array[int],
    needs: wp.array[int],
    active: wp.array[int],
    status: wp.array[int],
    iterations: wp.array[int],
    residual: wp.array[wp.float64],
):
    # The CPU scan selects the first most-negative scaled gap. Parallel min
    # and max reductions preserve that decision, including exact ties, without
    # serializing all contact candidates on one GPU thread.
    _block, lane = wp.tid()
    if status[0] != 0:
        if lane == 0:
            active[0] = 0
        return
    largest = wp.float64(0.0)
    most_negative = wp.float64(0.0)
    entering = size
    invalid = int(0)
    for i in range(lane, size, PRICE_BLOCK):
        if candidate[i] != 0:
            if not wp.isfinite(gap[i]) or not wp.isfinite(lam[i]):
                invalid = 1
            else:
                natural = wp.abs(wp.min(gap[i], W[i, i] * lam[i]))
                largest = wp.max(largest, natural)
                if lam[i] == wp.float64(0.0) and scale[i] > wp.float64(0.0) and gap[i] < -tolerance:
                    value = gap[i] / scale[i]
                    if value < most_negative:
                        most_negative = value
                        entering = i
    maximum_residual = wp.tile_max(wp.tile(largest))[0]
    minimum_gap = wp.tile_min(wp.tile(most_negative))[0]
    any_invalid = wp.tile_max(wp.tile(invalid))[0]
    owner = size
    if most_negative == minimum_gap:
        owner = entering
    winner = wp.tile_min(wp.tile(owner))[0]
    if lane == 0:
        if any_invalid != 0:
            status[0] = NONFINITE
            active[0] = 0
            residual[0] = wp.float64(wp.inf)
            return
        residual[0] = maximum_residual
        if needs[0] == 0:
            if maximum_residual <= tolerance:
                active[0] = 0
                return
            if winner == size:
                status[0] = NONFINITE
                active[0] = 0
                return
            order[count[0]] = winner
            count[0] += 1
        if iterations[0] >= maximum:
            status[0] = ITERATION_LIMIT
            active[0] = 0
            return
        iterations[0] += 1


@wp.kernel(enable_backward=False)
def _small_system(
    W: wp.array2d[wp.float64],
    q: wp.array[wp.float64],
    scale: wp.array[wp.float64],
    order: wp.array[int],
    count: wp.array[int],
    active: wp.array[int],
    matrix: wp.array2d[wp.float64],
    rhs: wp.array[wp.float64],
):
    i, j = wp.tid()
    n = count[0]
    if active[0] == 0 or n > TILE:
        return
    value = wp.where(i == j, wp.float64(1.0), wp.float64(0.0))
    if i < n and j < n:
        a = order[i]
        b = order[j]
        value = W[a, b] / (scale[a] * scale[b])
    matrix[i, j] = value
    if j == 0:
        rhs[i] = wp.float64(0.0)
        if i < n:
            a = order[i]
            rhs[i] = -q[a] / scale[a]


@wp.kernel(enable_backward=False)
def _small_factor(
    matrix: wp.array2d[wp.float64], count: wp.array[int], active: wp.array[int], factor: wp.array2d[wp.float64]
):
    _block = wp.tid()
    if active[0] == 0 or count[0] > TILE:
        return
    L = wp.tile_cholesky(wp.tile_load(matrix, shape=(TILE, TILE), storage="shared"))
    wp.tile_store(factor, L)


@wp.kernel(enable_backward=False)
def _small_guard(factor: wp.array2d[wp.float64], count: wp.array[int], active: wp.array[int], status: wp.array[int]):
    if active[0] == 0 or count[0] > TILE:
        return
    for i in range(count[0]):
        pivot = factor[i, i]
        if not wp.isfinite(pivot) or pivot * pivot <= wp.float64(PIVOT_ROUNDOFF):
            status[0] = DEPENDENT_SYSTEM
            active[0] = 0
            return


@wp.kernel(enable_backward=False)
def _small_solve(
    factor: wp.array2d[wp.float64], count: wp.array[int], active: wp.array[int], rhs: wp.array[wp.float64]
):
    _block = wp.tid()
    if active[0] == 0 or count[0] > TILE:
        return
    L = wp.tile_load(factor, shape=(TILE, TILE), storage="shared")
    value = wp.tile_cholesky_solve(L, wp.tile_load(rhs, shape=TILE, storage="shared"))
    wp.tile_store(rhs, value)


@wp.kernel(enable_backward=False)
def _large_solve(
    W: wp.array2d[wp.float64],
    q: wp.array[wp.float64],
    scale: wp.array[wp.float64],
    order: wp.array[int],
    count: wp.array[int],
    active: wp.array[int],
    factor: wp.array2d[wp.float64],
    rhs: wp.array[wp.float64],
    status: wp.array[int],
):
    n = count[0]
    if active[0] == 0 or n <= TILE:
        return
    for i in range(n):
        a = order[i]
        rhs[i] = -q[a] / scale[a]
        for j in range(i + 1):
            b = order[j]
            value = W[a, b] / (scale[a] * scale[b])
            for k in range(j):
                value -= factor[i, k] * factor[j, k]
            if i == j:
                if not wp.isfinite(value) or value <= wp.float64(PIVOT_ROUNDOFF):
                    status[0] = DEPENDENT_SYSTEM
                    active[0] = 0
                    return
                factor[i, j] = wp.sqrt(value)
            else:
                factor[i, j] = value / factor[j, j]
    for i in range(n):
        for j in range(i):
            rhs[i] -= factor[i, j] * rhs[j]
        rhs[i] /= factor[i, i]
    for reverse in range(n):
        i = n - 1 - reverse
        for j in range(i + 1, n):
            rhs[i] -= factor[j, i] * rhs[j]
        rhs[i] /= factor[i, i]


@wp.kernel(enable_backward=False)
def _dependent_exchange(
    W: wp.array2d[wp.float64],
    q: wp.array[wp.float64],
    scale: wp.array[wp.float64],
    order: wp.array[int],
    count: wp.array[int],
    lam: wp.array[wp.float64],
    factor: wp.array2d[wp.float64],
    rhs: wp.array[wp.float64],
    active: wp.array[int],
    status: wp.array[int],
):
    if status[0] != DEPENDENT_SYSTEM:
        return
    n = count[0]
    if n < 2:
        return
    # The already accepted block must remain independent. Only the newly
    # appended row may be redundant; otherwise retain the coordinate fallback.
    for i in range(n - 1):
        pivot = factor[i, i]
        if not wp.isfinite(pivot) or pivot * pivot <= wp.float64(PIVOT_ROUNDOFF):
            return
    entering = order[n - 1]
    if lam[entering] != wp.float64(0.0):
        return
    # In normalized multiplier coordinates y_i=s_i*lambda_i, the dependent
    # row gives d=(-A_old^-1*a, 1). Following this null direction preserves
    # equilibrium while exchanging a redundant row at the first nonnegative
    # bound. W itself is never regularized or modified.
    for i in range(n - 1):
        row = order[i]
        value = -W[row, entering] / (scale[row] * scale[entering])
        for j in range(i):
            value -= factor[i, j] * rhs[j]
        rhs[i] = value / factor[i, i]
    for reverse in range(n - 1):
        i = n - 2 - reverse
        value = rhs[i]
        for j in range(i + 1, n - 1):
            value -= factor[j, i] * rhs[j]
        rhs[i] = value / factor[i, i]
    rhs[n - 1] = wp.float64(1.0)
    alpha = wp.float64(wp.inf)
    blocking = int(-1)
    for i in range(n):
        if not wp.isfinite(rhs[i]):
            return
        if rhs[i] < wp.float64(0.0):
            row = order[i]
            step = -(lam[row] * scale[row]) / rhs[i]
            if step < alpha:
                alpha = step
                blocking = i
    if blocking < 0 or not wp.isfinite(alpha):
        return
    # Finite precision makes the null direction approximate. Require an actual
    # descent step for the unchanged quadratic before allowing the exchange.
    slope = wp.float64(0.0)
    curvature = wp.float64(0.0)
    for i in range(n):
        row = order[i]
        gradient = q[row] / scale[row]
        action = wp.float64(0.0)
        for j in range(n):
            column = order[j]
            value = W[row, column] / (scale[row] * scale[column])
            gradient += value * (lam[column] * scale[column])
            action += value * rhs[j]
        slope += rhs[i] * gradient
        curvature += rhs[i] * action
    if (
        not wp.isfinite(slope)
        or not wp.isfinite(curvature)
        or slope >= wp.float64(0.0)
        or slope + wp.float64(0.5) * alpha * curvature >= wp.float64(0.0)
    ):
        return
    for i in range(n):
        row = order[i]
        rhs[i] = lam[row] * scale[row] + alpha * rhs[i]
    # _feasible_update performs the same bound removal used by a nonsingular
    # solve. The next reduced solve and full original-W KKT certificate decide
    # success; accepting this exchange is not a convergence claim.
    rhs[blocking] = wp.float64(0.0)
    active[0] = 1
    status[0] = 0


@wp.kernel(enable_backward=False)
def _feasible_update(
    rhs: wp.array[wp.float64],
    scale: wp.array[wp.float64],
    order: wp.array[int],
    count: wp.array[int],
    active: wp.array[int],
    lam: wp.array[wp.float64],
    needs: wp.array[int],
    status: wp.array[int],
):
    if active[0] == 0:
        return
    n = count[0]
    alpha = wp.float64(1.0)
    blocking = int(-1)
    for k in range(n):
        i = order[k]
        trial = rhs[k] / scale[i]
        if not wp.isfinite(trial):
            status[0] = NONFINITE
            active[0] = 0
            return
        if trial <= wp.float64(0.0):
            step = wp.float64(0.0)
            if lam[i] > wp.float64(0.0):
                step = lam[i] / (lam[i] - trial)
            if blocking < 0 or step < alpha:
                alpha = step
                blocking = k
    for k in range(n):
        i = order[k]
        trial = rhs[k] / scale[i]
        lam[i] = wp.max(wp.float64(0.0), lam[i] + alpha * (trial - lam[i]))
    if blocking >= 0:
        lam[order[blocking]] = wp.float64(0.0)
        for k in range(blocking, n - 1):
            order[k] = order[k + 1]
        count[0] = n - 1
        needs[0] = 1
    else:
        needs[0] = 0


@wp.kernel(enable_backward=False)
def _finish(status: wp.array[int], lam: wp.array[wp.float64], omega: wp.array[wp.float64]):
    i = wp.tid()
    if status[0] != 0:
        # The caller must reject a failed step, not apply an uncertified force.
        lam[i] = wp.float64(0.0)
    omega[i] = wp.where(lam[i] > wp.float64(0.0), wp.float64(1.0), wp.float64(0.0))


@wp.kernel(enable_backward=False)
def _coordinate_begin(status: wp.array[int], active: wp.array[int], iterations: wp.array[int], fallback: wp.array[int]):
    active[0] = 0
    fallback[0] = 0
    if status[0] == DEPENDENT_SYSTEM:
        status[0] = 0
        active[0] = 1
        iterations[0] = 0
        fallback[0] = 1


@wp.kernel(enable_backward=False)
def _coordinate_reset(
    q: wp.array[wp.float64], active: wp.array[int], lam: wp.array[wp.float64], gap: wp.array[wp.float64]
):
    i = wp.tid()
    if active[0] != 0:
        lam[i] = wp.float64(0.0)
        gap[i] = q[i]


@wp.kernel(enable_backward=False)
def _coordinate_select(
    W: wp.array2d[wp.float64],
    gap: wp.array[wp.float64],
    candidate: wp.array[int],
    size: int,
    tolerance: wp.float64,
    maximum: int,
    lam: wp.array[wp.float64],
    active: wp.array[int],
    status: wp.array[int],
    iterations: wp.array[int],
    residual: wp.array[wp.float64],
    selected: wp.array[int],
    delta: wp.array[wp.float64],
):
    selected[0] = -1
    delta[0] = wp.float64(0.0)
    if active[0] == 0:
        return
    largest = wp.float64(0.0)
    best_decrease = wp.float64(0.0)
    entering = int(-1)
    for i in range(size):
        if candidate[i] != 0:
            d = W[i, i]
            g = gap[i]
            if not wp.isfinite(d) or not wp.isfinite(g) or not wp.isfinite(lam[i]):
                status[0] = NONFINITE
                active[0] = 0
                residual[0] = wp.float64(wp.inf)
                return
            if d < wp.float64(0.0):
                status[0] = DEPENDENT_SYSTEM
                active[0] = 0
                return
            natural = wp.abs(wp.min(g, d * lam[i]))
            largest = wp.max(largest, natural)
            if d > wp.float64(0.0):
                change = wp.max(wp.float64(0.0), lam[i] - g / d) - lam[i]
                decrease = -change * (g + wp.float64(0.5) * d * change)
                if decrease > best_decrease:
                    best_decrease = decrease
                    entering = i
            elif g < -tolerance:
                status[0] = IMMOBILE_CONTACT
                active[0] = 0
                residual[0] = largest
                return
    residual[0] = largest
    if largest <= tolerance:
        active[0] = 0
    elif iterations[0] >= maximum:
        status[0] = ITERATION_LIMIT
        active[0] = 0
    elif entering < 0:
        status[0] = NONFINITE
        active[0] = 0
    else:
        selected[0] = entering
        change = wp.max(wp.float64(0.0), lam[entering] - gap[entering] / W[entering, entering]) - lam[entering]
        delta[0] = change
        lam[entering] += change
        iterations[0] += 1


@wp.kernel(enable_backward=False)
def _coordinate_gradient(
    W: wp.array2d[wp.float64], selected: wp.array[int], delta: wp.array[wp.float64], gap: wp.array[wp.float64]
):
    i = wp.tid()
    if selected[0] >= 0:
        gap[i] += W[i, selected[0]] * delta[0]


@wp.kernel(enable_backward=False)
def _certificate_begin(status: wp.array[int], residual: wp.array[wp.float64]):
    if status[0] == 0:
        residual[0] = wp.float64(0.0)


@wp.kernel(enable_backward=False)
def _certificate(
    W: wp.array2d[wp.float64],
    q: wp.array[wp.float64],
    candidate: wp.array[int],
    lam: wp.array[wp.float64],
    size: int,
    tolerance: wp.float64,
    status: wp.array[int],
    residual: wp.array[wp.float64],
):
    i = wp.tid()
    if status[0] != 0 or candidate[i] == 0:
        return
    value = q[i]
    for j in range(size):
        value += W[i, j] * lam[j]
    natural = wp.abs(wp.min(value, W[i, i] * lam[i]))
    if not wp.isfinite(value) or not wp.isfinite(lam[i]) or lam[i] < wp.float64(0.0):
        wp.atomic_or(status, 0, NONFINITE)
        wp.atomic_max(residual, 0, wp.float64(wp.inf))
    else:
        wp.atomic_max(residual, 0, natural)
        if natural > tolerance:
            wp.atomic_or(status, 0, NONFINITE)


class CertifiedNormalQP:
    def __init__(self, size, device, tolerance=1.0e-9, max_iterations=1024, coordinate_iterations=8192):
        if not isinstance(size, (int, np.integer)) or size < 1:
            raise ValueError("normal QP size must be a positive integer")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("normal QP tolerance must be finite and positive")
        if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
            raise ValueError("normal QP max_iterations must be a positive integer")
        if not isinstance(coordinate_iterations, (int, np.integer)) or coordinate_iterations < 1:
            raise ValueError("normal QP coordinate_iterations must be a positive integer")
        self.size = int(size)
        self.device = wp.get_device(device)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.coordinate_iterations = int(coordinate_iterations)
        storage = max(self.size, TILE)
        self.scale = wp.zeros(storage, dtype=wp.float64, device=device)
        self.gap = wp.zeros(storage, dtype=wp.float64, device=device)
        self.rhs = wp.zeros(storage, dtype=wp.float64, device=device)
        self.matrix = wp.zeros((TILE, TILE), dtype=wp.float64, device=device)
        self.factor = wp.zeros((storage, storage), dtype=wp.float64, device=device)
        self.order = wp.zeros(storage, dtype=int, device=device)
        for name in ("count", "needs", "active", "status", "iterations", "fallback", "selected"):
            setattr(self, name, wp.zeros(1, dtype=int, device=device))
        self.delta = wp.zeros(1, dtype=wp.float64, device=device)
        self.residual = wp.zeros(1, dtype=wp.float64, device=device)
        self.graph = None
        self.graph_key = None
        self.coordinate_graph = None
        self.coordinate_graph_key = None

    def _iteration(self):
        W, q, candidate, lam = self.arguments
        wp.launch(
            _gradient,
            self.size,
            inputs=[W, q, lam, self.order, self.count, candidate],
            outputs=[self.gap],
            device=self.device,
        )
        price_inputs = [W, self.gap, lam, self.scale, candidate, self.size, self.tolerance, self.max_iterations]
        price_outputs = [self.order, self.count, self.needs, self.active, self.status, self.iterations, self.residual]
        if self.device.is_cuda:
            wp.launch_tiled(
                _price_tiled,
                dim=[1],
                inputs=price_inputs,
                outputs=price_outputs,
                block_dim=PRICE_BLOCK,
                device=self.device,
            )
        else:
            wp.launch(_price, 1, inputs=price_inputs, outputs=price_outputs, device=self.device)
        wp.launch(
            _small_system,
            (TILE, TILE),
            inputs=[W, q, self.scale, self.order, self.count, self.active],
            outputs=[self.matrix, self.rhs],
            device=self.device,
        )
        wp.launch_tiled(
            _small_factor,
            dim=[1],
            inputs=[self.matrix, self.count, self.active],
            outputs=[self.factor],
            block_dim=64,
            device=self.device,
        )
        wp.launch(
            _small_guard, 1, inputs=[self.factor, self.count], outputs=[self.active, self.status], device=self.device
        )
        wp.launch_tiled(
            _small_solve,
            dim=[1],
            inputs=[self.factor, self.count, self.active],
            outputs=[self.rhs],
            block_dim=64,
            device=self.device,
        )
        wp.launch(
            _large_solve,
            1,
            inputs=[W, q, self.scale, self.order, self.count, self.active],
            outputs=[self.factor, self.rhs, self.status],
            device=self.device,
        )
        wp.launch(
            _dependent_exchange,
            1,
            inputs=[W, q, self.scale, self.order, self.count, lam, self.factor],
            outputs=[self.rhs, self.active, self.status],
            device=self.device,
        )
        wp.launch(
            _feasible_update,
            1,
            inputs=[self.rhs, self.scale],
            outputs=[self.order, self.count, self.active, lam, self.needs, self.status],
            device=self.device,
        )

    def _coordinate_iteration(self):
        W, _q, candidate, lam = self.arguments
        wp.launch(
            _coordinate_select,
            1,
            inputs=[W, self.gap, candidate, self.size, self.tolerance, self.coordinate_iterations],
            outputs=[lam, self.active, self.status, self.iterations, self.residual, self.selected, self.delta],
            device=self.device,
        )
        wp.launch(
            _coordinate_gradient,
            self.size,
            inputs=[W, self.selected, self.delta],
            outputs=[self.gap],
            device=self.device,
        )

    def solve(self, W, q, candidate, lam, omega):
        self.arguments = (W, q, candidate, lam)
        wp.launch(
            _begin,
            1,
            outputs=[self.active, self.status, self.iterations, self.count, self.needs, self.residual],
            device=self.device,
        )
        wp.launch(
            _initialize,
            self.size,
            inputs=[W, q, candidate, self.tolerance],
            outputs=[lam, self.scale, self.status],
            device=self.device,
        )
        if self.device.is_capturing:
            wp.capture_while(self.active, self._iteration)
        elif self.device.is_cuda and wp.is_conditional_graph_supported():
            key = (W.ptr, q.ptr, candidate.ptr, lam.ptr)
            if self.graph is None or key != self.graph_key:
                with wp.ScopedCapture(device=self.device) as capture:
                    wp.capture_while(self.active, self._iteration)
                self.graph = capture.graph
                self.graph_key = key
            wp.capture_launch(self.graph)
        else:
            for _ in range(2 * self.max_iterations + 2):
                self._iteration()
                if not self.active.numpy()[0]:
                    break
        wp.launch(
            _coordinate_begin,
            1,
            outputs=[self.status, self.active, self.iterations, self.fallback],
            device=self.device,
        )
        wp.launch(_coordinate_reset, self.size, inputs=[q, self.active], outputs=[lam, self.gap], device=self.device)
        if self.device.is_capturing:
            wp.capture_while(self.active, self._coordinate_iteration)
        elif self.device.is_cuda and wp.is_conditional_graph_supported():
            key = (W.ptr, q.ptr, candidate.ptr, lam.ptr)
            if self.coordinate_graph is None or key != self.coordinate_graph_key:
                with wp.ScopedCapture(device=self.device) as capture:
                    wp.capture_while(self.active, self._coordinate_iteration)
                self.coordinate_graph = capture.graph
                self.coordinate_graph_key = key
            wp.capture_launch(self.coordinate_graph)
        else:
            for _ in range(self.coordinate_iterations + 1):
                if not self.active.numpy()[0]:
                    break
                self._coordinate_iteration()
        wp.launch(_certificate_begin, 1, inputs=[self.status], outputs=[self.residual], device=self.device)
        wp.launch(
            _certificate,
            self.size,
            inputs=[W, q, candidate, lam, self.size, self.tolerance],
            outputs=[self.status, self.residual],
            device=self.device,
        )
        wp.launch(_finish, self.size, inputs=[self.status], outputs=[lam, omega], device=self.device)
