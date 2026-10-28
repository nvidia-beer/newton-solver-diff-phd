# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Resident pivoted LU for reduced contact systems."""

from functools import cache

import warp as wp


@wp.kernel(enable_backward=False)
def _check_finite(matrix: wp.array2d[wp.float64], invalid: wp.array[wp.int32]):
    i, j = wp.tid()
    if not wp.isfinite(matrix[i, j]):
        wp.atomic_max(invalid, 0, 1)


@wp.func
def _pivot_column(
    matrix: wp.array2d[wp.float64],
    pivots: wp.array[wp.int32],
    column: int,
    invalid: wp.array[wp.int32],
):
    if invalid[0] != 0:
        return
    pivot = column
    largest = wp.float64(0.0)
    for row in range(column, matrix.shape[0]):
        value = wp.abs(matrix[row, column])
        if not wp.isfinite(value):
            invalid[0] = 1
            return
        if value > largest:
            largest = value
            pivot = row
    if largest == wp.float64(0.0):
        invalid[0] = 1
        return
    pivots[column] = pivot
    if pivot != column:
        for j in range(matrix.shape[1]):
            value = matrix[column, j]
            matrix[column, j] = matrix[pivot, j]
            matrix[pivot, j] = value


@wp.kernel(enable_backward=False)
def _pivot(matrix: wp.array2d[wp.float64], pivots: wp.array[wp.int32], column: int, invalid: wp.array[wp.int32]):
    _pivot_column(matrix, pivots, column, invalid)


@wp.kernel(enable_backward=False)
def _factor_small(matrix: wp.array2d[wp.float64], pivots: wp.array[wp.int32], invalid: wp.array[wp.int32]):
    # Tiny reduced systems otherwise require three launches per pivot.
    for column in range(matrix.shape[0]):
        _pivot_column(matrix, pivots, column, invalid)
        if invalid[0] != 0:
            return
        for row in range(column + 1, matrix.shape[0]):
            matrix[row, column] /= matrix[column, column]
            for col in range(column + 1, matrix.shape[1]):
                matrix[row, col] -= matrix[row, column] * matrix[column, col]


@wp.func
def _pivot_reciprocal(value: wp.float64):
    return wp.float64(1.0) / value


@wp.func
def _divide_below(value: wp.float64, row: int, column: int, pivot: wp.float64, inverse: wp.float64):
    result = value
    if row > column:
        if wp.isfinite(inverse):
            result = value * inverse
        else:
            # A valid subnormal pivot can overflow its reciprocal.
            result = value / pivot
    return result


@wp.func
def _trailing_value(value: wp.float64, index: int, column: int):
    result = wp.float64(0.0)
    if index > column:
        result = value
    return result


@wp.func
def _update_factor(value: wp.float64, column: wp.float64, row: wp.float64):
    return value - column * row


@wp.func
def _pivot_magnitude(value: wp.float64, row: int, column: int, n: int):
    result = wp.float64(-1.0)
    if row >= column and row < n:
        result = wp.abs(value)
        if not wp.isfinite(result):
            result = wp.float64(wp.inf)
    return result


@wp.func
def _pivot_index(value: wp.float64, row: int, largest: wp.float64):
    result = int(2147483647)
    if value == largest:
        result = row
    return result


@cache
def _factor_tiled(size):
    @wp.kernel(module="unique", enable_backward=False)
    def factor(matrix: wp.array2d[wp.float64], pivots: wp.array[wp.int32], invalid: wp.array[wp.int32]):
        _, lane = wp.tid()
        values = wp.tile_load(matrix, shape=(size, size), storage="shared")
        swap = wp.tile_zeros(shape=(2, size), dtype=wp.float64, storage="shared")
        rows = wp.tile_reshape(wp.tile_arange(size, dtype=int), shape=(size, 1))
        cols = wp.tile_reshape(wp.tile_arange(size, dtype=int), shape=(1, size))
        n = matrix.shape[0]
        if invalid[0] != 0:
            return
        for column in range(n):
            magnitudes = wp.tile_map(
                _pivot_magnitude, wp.tile_view(values, offset=(0, column), shape=(size, 1)), rows, column, n
            )
            largest = wp.tile_max(magnitudes)[0]
            pivot = wp.tile_min(wp.tile_map(_pivot_index, magnitudes, rows, largest))[0]
            if largest == wp.float64(0.0) or not wp.isfinite(largest):
                if lane == 0:
                    invalid[0] = 1
                return
            if lane == 0:
                pivots[column] = pivot
            if pivot != column:
                wp.tile_assign(swap, wp.tile_view(values, offset=(column, 0), shape=(1, size)), offset=(0, 0))
                wp.tile_assign(swap, wp.tile_view(values, offset=(pivot, 0), shape=(1, size)), offset=(1, 0))
                wp.tile_assign(values, wp.tile_view(swap, offset=(1, 0), shape=(1, size)), offset=(column, 0))
                wp.tile_assign(values, wp.tile_view(swap, offset=(0, 0), shape=(1, size)), offset=(pivot, 0))
            # The singleton tile computes one division and broadcasts it.
            inverse = wp.tile_map(_pivot_reciprocal, wp.tile_view(values, offset=(column, column), shape=(1, 1)))[0, 0]
            divided = wp.tile_map(
                _divide_below,
                wp.tile_view(values, offset=(0, column), shape=(size, 1)),
                rows,
                column,
                values[column, column],
                inverse,
            )
            wp.tile_assign(values, divided, offset=(0, column))
            lower = wp.tile_map(_trailing_value, divided, rows, column)
            upper = wp.tile_map(
                _trailing_value, wp.tile_view(values, offset=(column, 0), shape=(1, size)), cols, column
            )
            updated = wp.tile_map(
                _update_factor,
                values,
                wp.tile_broadcast(lower, shape=(size, size)),
                wp.tile_broadcast(upper, shape=(size, size)),
            )
            wp.tile_assign(values, updated)
        wp.tile_store(matrix, values)

    return factor


@wp.kernel(enable_backward=False)
def _divide_column(matrix: wp.array2d[wp.float64], column: int, invalid: wp.array[wp.int32]):
    if invalid[0] == 0:
        row = column + 1 + wp.tid()
        matrix[row, column] /= matrix[column, column]


@wp.kernel(enable_backward=False)
def _update_trailing(matrix: wp.array2d[wp.float64], column: int, invalid: wp.array[wp.int32]):
    if invalid[0] == 0:
        i, j = wp.tid()
        row = column + 1 + i
        col = column + 1 + j
        matrix[row, col] -= matrix[row, column] * matrix[column, col]


@wp.kernel(enable_backward=False)
def _solve(
    factor: wp.array2d[wp.float64],
    pivots: wp.array[wp.int32],
    invalid: wp.array[wp.int32],
    rhs: wp.array[wp.float64],
    transpose: bool,
    out: wp.array[wp.float64],
):
    n = factor.shape[0]
    if invalid[0] != 0:
        for i in range(n):
            out[i] = wp.float64(wp.nan)
        return
    for i in range(n):
        out[i] = rhs[i]
    if not transpose:
        for i in range(n):
            pivot = pivots[i]
            value = out[i]
            out[i] = out[pivot]
            out[pivot] = value
        for i in range(n):
            value = out[i]
            for j in range(i):
                value -= factor[i, j] * out[j]
            out[i] = value
        for offset in range(n):
            i = n - 1 - offset
            value = out[i]
            for j in range(i + 1, n):
                value -= factor[i, j] * out[j]
            out[i] = value / factor[i, i]
    else:
        for i in range(n):
            value = out[i]
            for j in range(i):
                value -= factor[j, i] * out[j]
            out[i] = value / factor[i, i]
        for offset in range(n):
            i = n - 1 - offset
            value = out[i]
            for j in range(i + 1, n):
                value -= factor[j, i] * out[j]
            out[i] = value
        # P A = L U, hence A.T x = b ends with x = P.T y.
        for offset in range(n):
            i = n - 1 - offset
            pivot = pivots[i]
            value = out[i]
            out[i] = out[pivot]
            out[pivot] = value


@cache
def _solve_tiled(tile_size):
    @wp.kernel(module="unique", enable_backward=False)
    def solve(
        factor: wp.array2d[wp.float64],
        pivots: wp.array[wp.int32],
        invalid: wp.array[wp.int32],
        rhs: wp.array[wp.float64],
        transpose: bool,
        out: wp.array[wp.float64],
    ):
        _, row = wp.tid()
        if invalid[0] != 0:
            if row < out.shape[0]:
                out[row] = wp.float64(wp.nan)
            return
        values = wp.tile_load(rhs, shape=(tile_size,), storage="shared")
        scratch = wp.tile_zeros(shape=(tile_size,), dtype=wp.float64, storage="shared")
        n = rhs.shape[0]
        # Copy before each update: other warps may still need the pivot value
        # when its owning lane writes the new value. Each copy is collective.
        if not transpose:
            for i in range(n):
                wp.tile_assign(scratch, values)
                pivot = pivots[i]
                value = scratch[row]
                if row == i:
                    value = scratch[pivot]
                elif row == pivot:
                    value = scratch[i]
                values[row] = value
            for i in range(n):
                wp.tile_assign(scratch, values)
                solved = scratch[i]
                value = scratch[row]
                if row > i and row < n:
                    value -= factor[row, i] * solved
                values[row] = value
            for offset in range(n):
                wp.tile_assign(scratch, values)
                i = n - 1 - offset
                solved = scratch[i] / factor[i, i]
                value = scratch[row]
                if row == i:
                    value = solved
                elif row < i:
                    value -= factor[row, i] * solved
                values[row] = value
        else:
            for i in range(n):
                wp.tile_assign(scratch, values)
                solved = scratch[i] / factor[i, i]
                value = scratch[row]
                if row == i:
                    value = solved
                elif row > i and row < n:
                    value -= factor[i, row] * solved
                values[row] = value
            for offset in range(n):
                wp.tile_assign(scratch, values)
                i = n - 1 - offset
                solved = scratch[i]
                value = scratch[row]
                if row < i:
                    value -= factor[i, row] * solved
                values[row] = value
            for offset in range(n):
                wp.tile_assign(scratch, values)
                i = n - 1 - offset
                pivot = pivots[i]
                value = scratch[row]
                if row == i:
                    value = scratch[pivot]
                elif row == pivot:
                    value = scratch[i]
                values[row] = value
        wp.tile_store(out, values)

    return solve


class DenseFactorizationError(RuntimeError):
    """A reduced contact matrix could not be factored."""


class DenseLU:
    """Factor a small square matrix with partial pivoting on its Warp device.

    Only the reduced contact system is dense; the mechanical operator remains
    matrix-free. Tiny finite pivots are retained because redundant constraints
    can produce small, valid compliance modes. No diagonal shift changes the
    matrix. The factor owns its arrays and each solve writes only its output.

    ``check=False`` keeps factorization status on the device in ``invalid``:
    zero means success, and one means a singular or nonfinite factor. Solves
    with an invalid factor produce NaNs. Create and warm up the factor and
    solve before CUDA graph capture, then use :meth:`refactor` with
    ``check=False`` to reuse the buffers.
    """

    def __init__(self, matrix, *, check=True):
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.dtype != wp.float64:
            raise ValueError("DenseLU requires a square float64 Warp array")
        self.size = matrix.shape[0]
        self.device = matrix.device
        self.factor = wp.empty_like(matrix, requires_grad=False)
        self.pivots = wp.empty(self.size, dtype=wp.int32, device=self.device)
        self.invalid = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.refactor(matrix, check=check)

    def refactor(self, matrix, *, check=True):
        """Replace the matrix without reallocating factorization buffers.

        Args:
            matrix: Square float64 array matching the existing factor.
            check: Download status and raise on failure. Disable for capture;
                callers must then inspect ``invalid`` or their solve residual.
        """
        if matrix.shape != (self.size, self.size) or matrix.dtype != wp.float64 or matrix.device != self.device:
            raise ValueError("DenseLU matrix must match its size, float64 dtype, and device")
        if check and self.device.is_capturing:
            raise RuntimeError("DenseLU checked factorization cannot be captured; use check=False")
        self.invalid.zero_()
        wp.copy(self.factor, matrix)
        if not self.size:
            return
        invalid = self.invalid
        wp.launch(_check_finite, matrix.shape, inputs=[self.factor, invalid], device=self.device)
        tile_size = 32 * ((self.size + 31) // 32)
        if self.size <= 3 or (self.size <= 96 and not self.device.is_cuda):
            wp.launch(_factor_small, 1, inputs=[self.factor, self.pivots, invalid], device=self.device)
        elif self.size <= 96 and 8 * tile_size * tile_size + 64 * tile_size <= self.device.max_shared_memory_per_block:
            wp.launch_tiled(
                _factor_tiled(tile_size),
                dim=1,
                inputs=[self.factor, self.pivots, invalid],
                block_dim=256,
                device=self.device,
            )
        else:
            for column in range(self.size):
                wp.launch(_pivot, 1, inputs=[self.factor, self.pivots, column, invalid], device=self.device)
                trailing = self.size - column - 1
                if trailing:
                    wp.launch(_divide_column, trailing, inputs=[self.factor, column, invalid], device=self.device)
                    wp.launch(
                        _update_trailing,
                        (trailing, trailing),
                        inputs=[self.factor, column, invalid],
                        device=self.device,
                    )
        wp.launch(_check_finite, matrix.shape, inputs=[self.factor, invalid], device=self.device)
        if check and invalid.numpy()[0]:
            raise DenseFactorizationError("DiffPhD reduced contact matrix is singular or nonfinite")

    def solve(self, rhs, out, *, transpose=False):
        """Apply the factor or its transpose; invalid factors write NaNs.

        ``rhs`` and ``out`` may alias. This method does not read device status
        on the host; the factor's ``invalid`` flag remains available to callers.
        """
        for array in (rhs, out):
            if array.shape != (self.size,) or array.dtype != wp.float64 or array.device != self.device:
                raise ValueError("DenseLU vectors must match its size, float64 dtype, and device")
        if self.device.is_cuda and 8 < self.size <= 1024:
            tile_size = 32 * ((self.size + 31) // 32)
            wp.launch_tiled(
                _solve_tiled(tile_size),
                dim=1,
                inputs=[self.factor, self.pivots, self.invalid, rhs, transpose, out],
                block_dim=tile_size,
                device=self.device,
            )
        elif self.size:
            wp.launch(
                _solve, 1, inputs=[self.factor, self.pivots, self.invalid, rhs, transpose, out], device=self.device
            )
