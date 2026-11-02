# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Portions translated from DiffPhD (https://github.com/Akinesia112/DiffPhD,
# commit d319435), Copyright (c) 2021 MIT Graphics Group, MIT License.
# See THIRD_PARTY_NOTICES.md.

"""Type-II Anderson acceleration with device-resident history.

This follows ``ForwardProjectiveDynamicsAlgPhd`` in the reference
(``cpp/core/src/fem/deformable_projective_dynamics_forward_gpu.cpp``): the
history stores pairs ``(x_prev, g = x - x_prev)``; with ``m`` difference
columns ``dG`` and ``dX`` the mixing coefficients solve
``(dG^T dG + reg I) gamma = dG^T g`` with
``reg = max(1e-6 ||dG||_F^2 / m, 1e-12)``, and the accelerated iterate is
``x_prev + g - (dX + dG) gamma``. When ``||gamma|| >= 10`` the history is
cleared and the raw fixed-point step is kept. At most ``window`` pairs are
retained. Every decision stays on the device, so the update can run inside a
CUDA conditional graph; partial sums use a fixed order and are deterministic.

The update is batched over independent columns, each with its own history,
as needed for the reference's per-column Woodbury fixed points; a column
whose ``enabled`` flag is zero is left untouched.
"""

import warp as wp

from .projections import DOUBLE_MIN

# Nodes per partial-sum thread; reductions finish in one fixed-order pass.
CHUNK = 16
MAX_WINDOW = 8

_matrix = wp.types.matrix(shape=(MAX_WINDOW, MAX_WINDOW), dtype=wp.float64)
_vector = wp.types.vector(length=MAX_WINDOW, dtype=wp.float64)
_indices = wp.types.vector(length=MAX_WINDOW, dtype=wp.int32)


@wp.kernel(enable_backward=False)
def _reset(state: wp.array2d[wp.int32], accept: wp.array[wp.int32]):
    col = wp.tid()
    for j in range(4):
        state[col, j] = 0
    accept[col] = 0


@wp.kernel(enable_backward=False)
def _push(
    x: wp.array2d[wp.vec3d],
    previous: wp.array2d[wp.vec3d],
    state: wp.array2d[wp.int32],
    enabled: wp.array[wp.int32],
    window: int,
    history_x: wp.array3d[wp.vec3d],
    history_g: wp.array3d[wp.vec3d],
):
    col, i = wp.tid()
    if enabled[col] == 0:
        return
    slot = (state[col, 1] + state[col, 0]) % (window + 1)
    history_x[slot, col, i] = previous[col, i]
    history_g[slot, col, i] = x[col, i] - previous[col, i]


@wp.kernel(enable_backward=False)
def _gram_partials(
    history_g: wp.array3d[wp.vec3d],
    state: wp.array2d[wp.int32],
    enabled: wp.array[wp.int32],
    window: int,
    size: int,
    partials: wp.array3d[wp.float64],
):
    # One thread per (column, Gram entry or right-hand side, chunk of nodes).
    col, entry, chunk = wp.tid()
    count = state[col, 0]
    start = state[col, 1]
    capacity = window + 1
    a = entry // window
    b = entry % window
    total = wp.float64(0.0)
    if enabled[col] != 0:
        if entry >= window * window:
            a = entry - window * window
            if a < count:
                slot = (start + count) % capacity
                for i in range(chunk * CHUNK, wp.min(size, (chunk + 1) * CHUNK)):
                    da = history_g[(start + a + 1) % capacity, col, i] - history_g[(start + a) % capacity, col, i]
                    total += wp.dot(da, history_g[slot, col, i])
        elif a <= b and b < count:
            for i in range(chunk * CHUNK, wp.min(size, (chunk + 1) * CHUNK)):
                da = history_g[(start + a + 1) % capacity, col, i] - history_g[(start + a) % capacity, col, i]
                db = history_g[(start + b + 1) % capacity, col, i] - history_g[(start + b) % capacity, col, i]
                total += wp.dot(da, db)
    partials[col, chunk, entry] = total


@wp.func
def _ldlt_solve(a: _matrix, b: _vector, n: int):
    # Eigen's dense LDLT: left-looking, largest original diagonal as pivot,
    # pseudo-inverse of D. Only the leading n x n block is used.
    m = a
    perm = _indices()
    for i in range(n):
        perm[i] = i
    for k in range(n):
        biggest = k
        for i in range(k + 1, n):
            if wp.abs(m[i, i]) > wp.abs(m[biggest, biggest]):
                biggest = i
        if biggest != k:
            for j in range(n):
                value = m[k, j]
                m[k, j] = m[biggest, j]
                m[biggest, j] = value
            for i in range(n):
                value = m[i, k]
                m[i, k] = m[i, biggest]
                m[i, biggest] = value
            index = perm[k]
            perm[k] = perm[biggest]
            perm[biggest] = index
        for j in range(k):
            m[k, k] = m[k, k] - m[k, j] * m[k, j] * m[j, j]
        for i in range(k + 1, n):
            for j in range(k):
                m[i, k] = m[i, k] - m[i, j] * m[j, j] * m[k, j]
            if m[k, k] != wp.float64(0.0):
                m[i, k] = m[i, k] / m[k, k]
    y = _vector()
    for i in range(n):
        y[i] = b[perm[i]]
    for i in range(n):
        for j in range(i):
            y[i] = y[i] - m[i, j] * y[j]
    for i in range(n):
        if wp.abs(m[i, i]) > DOUBLE_MIN:
            y[i] = y[i] / m[i, i]
        else:
            y[i] = wp.float64(0.0)
    for reverse in range(n):
        i = n - 1 - reverse
        for j in range(i + 1, n):
            y[i] = y[i] - m[j, i] * y[j]
    x = _vector()
    for i in range(n):
        x[perm[i]] = y[i]
    return x


@wp.kernel(enable_backward=False)
def _reduce(
    partials: wp.array3d[wp.float64], chunks: int, enabled: wp.array[wp.int32], reduced: wp.array2d[wp.float64]
):
    # One thread per (column, Gram entry); chunks are summed in a fixed order.
    col, entry = wp.tid()
    if enabled[col] == 0:
        return
    total = wp.float64(0.0)
    for c in range(chunks):
        total += partials[col, c, entry]
    reduced[col, entry] = total


@wp.kernel(enable_backward=False)
def _solve(
    reduced: wp.array2d[wp.float64],
    window: int,
    enabled: wp.array[wp.int32],
    state: wp.array2d[wp.int32],
    gamma: wp.array2d[wp.float64],
    accept: wp.array[wp.int32],
):
    col = wp.tid()
    if enabled[col] == 0:
        return
    count = state[col, 0]
    start = state[col, 1]
    capacity = window + 1
    # The apply kernel uses the history as it was before trimming.
    state[col, 2] = count
    state[col, 3] = start
    accept[col] = 0
    if count >= 1:
        gram = _matrix()
        rhs = _vector()
        for a in range(count):
            rhs[a] = reduced[col, window * window + a]
            for b in range(a, count):
                gram[a, b] = reduced[col, a * window + b]
        trace = wp.float64(0.0)
        for a in range(count):
            trace += gram[a, a]
            for b in range(a + 1, count):
                gram[b, a] = gram[a, b]
        regularization = wp.max(wp.float64(1.0e-6) * trace / wp.float64(count), wp.float64(1.0e-12))
        for a in range(count):
            gram[a, a] = gram[a, a] + regularization
        coefficients = _ldlt_solve(gram, rhs, count)
        norm = wp.float64(0.0)
        for a in range(count):
            gamma[col, a] = coefficients[a]
            norm += coefficients[a] * coefficients[a]
        if wp.sqrt(norm) >= wp.float64(10.0):
            count = 0
            start = 0
        else:
            accept[col] = 1
            count += 1
    else:
        count += 1
    while count > window:
        start = (start + 1) % capacity
        count -= 1
    state[col, 0] = count
    state[col, 1] = start


@wp.kernel(enable_backward=False)
def _apply(
    state: wp.array2d[wp.int32],
    gamma: wp.array2d[wp.float64],
    accept: wp.array[wp.int32],
    enabled: wp.array[wp.int32],
    window: int,
    history_x: wp.array3d[wp.vec3d],
    history_g: wp.array3d[wp.vec3d],
    x: wp.array2d[wp.vec3d],
):
    col, i = wp.tid()
    if enabled[col] == 0 or accept[col] == 0:
        return
    count = state[col, 2]
    start = state[col, 3]
    capacity = window + 1
    slot = (start + count) % capacity
    value = history_x[slot, col, i] + history_g[slot, col, i]
    for a in range(count):
        low = (start + a) % capacity
        high = (start + a + 1) % capacity
        value -= (
            history_x[high, col, i] - history_x[low, col, i] + history_g[high, col, i] - history_g[low, col, i]
        ) * gamma[col, a]
    x[col, i] = value


class Anderson:
    """Reusable Type-II Anderson mixing for fixed-point maps on ``vec3d`` arrays.

    Args:
        size: Number of vector entries per column.
        window: Maximum retained pairs (the reference ``aa_window``); zero disables mixing.
        device: Warp device.
        columns: Number of independent columns mixed together.
    """

    def __init__(self, size: int, window: int, device, columns: int = 1):
        if not 0 <= window <= MAX_WINDOW:
            raise ValueError(f"Anderson window must be in [0, {MAX_WINDOW}]")
        self.size, self.window, self.device, self.columns = size, window, device, columns
        self.chunks = max(1, (size + CHUNK - 1) // CHUNK)
        self.state = wp.zeros((columns, 4), dtype=wp.int32, device=device)
        self.accept = wp.zeros(columns, dtype=wp.int32, device=device)
        self.enabled = wp.ones(columns, dtype=wp.int32, device=device)
        if window:
            entries = window * (window + 1)
            self.history_x = wp.zeros((window + 1, columns, size), dtype=wp.vec3d, device=device)
            self.history_g = wp.zeros_like(self.history_x)
            self.partials = wp.zeros((columns, self.chunks, entries), dtype=wp.float64, device=device)
            self.reduced = wp.zeros((columns, entries), dtype=wp.float64, device=device)
            self.gamma = wp.zeros((columns, window), dtype=wp.float64, device=device)

    def reset(self):
        """Clear every column's history before a new fixed-point solve."""
        wp.launch(_reset, self.columns, inputs=[self.state, self.accept], device=self.device)

    def mix(self, x: wp.array2d, previous: wp.array2d, enabled: wp.array | None = None) -> None:
        """Replace the map outputs ``x`` (computed from ``previous``), shape ``(columns, size)``, in place."""
        if not self.window:
            return
        enabled = self.enabled if enabled is None else enabled
        device, window, columns = self.device, self.window, self.columns
        wp.launch(
            _push,
            (columns, self.size),
            inputs=[x, previous, self.state, enabled, window],
            outputs=[self.history_x, self.history_g],
            device=device,
        )
        entries = window * (window + 1)
        wp.launch(
            _gram_partials,
            (columns, entries, self.chunks),
            inputs=[self.history_g, self.state, enabled, window, self.size],
            outputs=[self.partials],
            device=device,
        )
        wp.launch(
            _reduce,
            (columns, entries),
            inputs=[self.partials, self.chunks, enabled],
            outputs=[self.reduced],
            device=device,
        )
        wp.launch(
            _solve,
            columns,
            inputs=[self.reduced, window, enabled, self.state],
            outputs=[self.gamma, self.accept],
            device=device,
        )
        wp.launch(
            _apply,
            (columns, self.size),
            inputs=[self.state, self.gamma, self.accept, enabled, window, self.history_x, self.history_g],
            outputs=[x],
            device=device,
        )
