# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exact sparse inverse factors with CPU setup and two resident Warp SpMVs.

For a symmetric positive-definite scalar matrix, ``P A P.T = L D L.T``
and ``S = D**(-1/2) L**(-1) P`` give ``A**(-1) b = S.T (S b)``.
Setup uses sparse row storage, CSR arrays and one linear-sized work row; no
dense matrix or inverse is formed. Exact inverse factors can still have substantial fill.
"""

from collections import deque

import numpy as np
import warp as wp


def _minimum_degree(vertices: set[int], adjacency: list[set[int]]) -> list[int]:
    graph = {i: adjacency[i] & vertices for i in vertices}
    order = []
    while graph:
        node = min(graph, key=lambda i: (len(graph[i]), i))
        neighbors = graph.pop(node)
        order.append(node)
        for other in neighbors:
            graph[other].discard(node)
            graph[other].update(neighbors - {other})
    return order


def _distances(start: int, vertices: set[int], adjacency: list[set[int]]) -> dict[int, int]:
    distance = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for other in sorted(adjacency[node] & vertices):
            if other not in distance:
                distance[other] = distance[node] + 1
                queue.append(other)
    return distance


def _nested_dissection(adjacency: list[set[int]]) -> np.ndarray:
    def visit(vertices):
        if len(vertices) <= 24:
            return _minimum_degree(vertices, adjacency)
        start = min(vertices, key=lambda i: (len(adjacency[i] & vertices), i))
        distances = _distances(start, vertices, adjacency)
        if len(distances) != len(vertices):
            components = [set(distances)]
            remaining = vertices - components[0]
            while remaining:
                component = set(_distances(min(remaining), remaining, adjacency))
                components.append(component)
                remaining -= component
            return [node for component in components for node in visit(component)]
        farthest = max(distances, key=lambda i: (distances[i], -i))
        distances = _distances(farthest, vertices, adjacency)
        levels = np.bincount(list(distances.values()))
        cut = int(np.searchsorted(np.cumsum(levels), len(vertices) / 2))
        lower = {i for i in vertices if distances[i] < cut}
        upper = {i for i in vertices if distances[i] > cut}
        separator = vertices - lower - upper
        if not lower or not upper:
            return _minimum_degree(vertices, adjacency)
        # BFS edges join equal or adjacent levels, so removing this level
        # separates the two interiors without a geometric embedding.
        return visit(lower) + visit(upper) + _minimum_degree(separator, adjacency)

    return np.asarray(visit(set(range(len(adjacency)))), dtype=np.int32)


def _validate_rows(matrix_rows: list[dict[int, float]]) -> list[dict[int, float]]:
    size = len(matrix_rows)
    if size > np.iinfo(np.int32).max:
        raise ValueError("matrix size exceeds int32 sparse indexing")
    rows = []
    for index, source in enumerate(matrix_rows):
        row = {}
        for column, source_value in source.items():
            if not isinstance(column, (int, np.integer)) or not 0 <= column < size:
                raise ValueError("matrix column indices must be integers in [0, size)")
            value = float(source_value)
            if not np.isfinite(value):
                raise ValueError("matrix coefficients must be finite")
            if value != 0.0:
                row[int(column)] = value
        if row.get(index, 0.0) <= 0.0:
            raise ValueError("matrix must be positive definite with positive diagonal entries")
        rows.append(row)
    for index, row in enumerate(rows):
        for column, value in row.items():
            if rows[column].get(index, 0.0) != value:
                raise ValueError("matrix_rows must contain both halves of an exactly symmetric matrix")
    return rows


def _factor_ldlt(rows: list[dict[int, float]], permutation: np.ndarray):
    size = len(rows)
    inverse = np.empty(size, dtype=np.int32)
    inverse[permutation] = np.arange(size, dtype=np.int32)
    work = [
        {int(inverse[j]): value for j, value in rows[int(original)].items() if inverse[j] <= i}
        for i, original in enumerate(permutation)
    ]
    # Discover the same fill as right-looking elimination before numeric work.
    # Each row stores only its lower-triangular nonzeros, including fill; no
    # dense matrix is allocated. Structural zeros remain until inverse setup.
    patterns = [set(row) for row in work]
    columns = [set() for _ in range(size)]
    for i, row in enumerate(patterns):
        for j in row:
            if j < i:
                columns[j].add(i)
    for k in range(size):
        prefix = set()
        for i in sorted(columns[k]):
            added = prefix - patterns[i]
            patterns[i].update(added)
            for j in added:
                columns[j].add(i)
            prefix.add(i)
        columns[k].clear()

    indices = [np.asarray(sorted(row), dtype=np.int32) for row in patterns]
    values = [np.zeros(len(row), dtype=np.float64) for row in indices]
    column_rows = [[] for _ in range(size)]
    column_positions = [[] for _ in range(size)]
    for i, (row_indices, row_values, row) in enumerate(zip(indices, values, work, strict=True)):
        row_values[np.searchsorted(row_indices, list(row))] = list(row.values())
        for position, k in enumerate(row_indices[:-1]):
            column_rows[k].append(i)
            column_positions[k].append(position)
    del patterns, work, columns

    diagonal = np.empty(size, dtype=np.float64)
    for k in range(size):
        pivot = values[k][-1]
        if not np.isfinite(pivot) or pivot <= 0.0:
            raise ValueError(f"matrix is not positive definite: LDLT pivot {k} is {pivot}")
        diagonal[k] = pivot
        neighbors = np.asarray(column_rows[k], dtype=np.int32)
        positions = column_positions[k]
        entries = np.asarray([values[i][position] for i, position in zip(neighbors, positions, strict=True)])
        for offset, (i, position) in enumerate(zip(neighbors, positions, strict=True)):
            multiplier = entries[offset] / pivot
            values[i][position] = multiplier
            locations = np.searchsorted(indices[i], neighbors[: offset + 1])
            # Entries within this row are independent. The outer pivot order
            # and each coefficient's multiply-then-subtract order are unchanged.
            values[i][locations] -= multiplier * entries[: offset + 1]
    lower = [
        dict(zip(map(int, row_indices[:-1]), row_values[:-1], strict=True))
        for row_indices, row_values in zip(indices, values, strict=True)
    ]
    return lower, diagonal


def _inverse_rows(lower, diagonal, permutation):
    columns, values, result = [], [], []
    # One O(n) work row, never an n-by-n inverse. Completed inverse rows stay
    # sparse, and updating distinct columns in NumPy preserves accumulation
    # order while avoiding a Python dictionary lookup for every scalar term.
    current = np.empty(len(lower), dtype=np.float64)
    for i, row in enumerate(lower):
        current[: i + 1] = 0.0
        current[i] = 1.0
        for k, coefficient in sorted(row.items()):
            current[columns[k]] -= coefficient * values[k]
        # Only exact arithmetic zeros are omitted; there is no drop tolerance.
        present = np.flatnonzero(current[: i + 1])
        entries = current[present].copy()
        columns.append(present)
        values.append(entries)
        scale = 1.0 / np.sqrt(diagonal[i])
        result.append(dict(zip(map(int, permutation[present]), entries * scale, strict=True)))
    return result


def _csr(rows):
    offsets = np.empty(len(rows) + 1, dtype=np.int32)
    offsets[0] = 0
    columns, values = [], []
    for i, row in enumerate(rows):
        for column, value in sorted(row.items()):
            columns.append(column)
            values.append(value)
        if len(columns) > np.iinfo(np.int32).max:
            raise ValueError("inverse factor fill exceeds int32 sparse indexing")
        offsets[i + 1] = len(columns)
    return offsets, np.asarray(columns, dtype=np.int32), np.asarray(values, dtype=np.float64)


@wp.kernel(enable_backward=False)
def _multiply_vec3(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    rhs: wp.array[wp.vec3d],
    out: wp.array[wp.vec3d],
):
    row = wp.tid()
    value = wp.vec3d(0.0)
    for entry in range(offsets[row], offsets[row + 1]):
        value += values[entry] * rhs[columns[entry]]
    out[row] = value


@wp.kernel(enable_backward=False)
def _multiply_vec3_tiled(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    rhs: wp.array[wp.vec3d],
    out: wp.array[wp.vec3d],
):
    # One warp cooperates on a CSR row instead of serializing its fill on
    # one thread. The three reductions retain double-precision accumulation.
    row, lane = wp.tid()
    partial = wp.vec3d(wp.float64(0.0))
    for entry in range(offsets[row] + lane, offsets[row + 1], 32):
        partial += values[entry] * rhs[columns[entry]]
    x = wp.tile_sum(wp.tile(partial[0]))[0]
    y = wp.tile_sum(wp.tile(partial[1]))[0]
    z = wp.tile_sum(wp.tile(partial[2]))[0]
    if lane == 0:
        out[row] = wp.vec3d(x, y, z)


@wp.kernel(enable_backward=False)
def _multiply_scalar(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    rhs: wp.array[wp.float64],
    out: wp.array[wp.float64],
):
    row = wp.tid()
    value = wp.float64(0.0)
    for entry in range(offsets[row], offsets[row + 1]):
        value += values[entry] * rhs[columns[entry]]
    out[row] = value


@wp.kernel(enable_backward=False)
def _multiply_batch(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    rhs: wp.array2d[wp.float64],
    out: wp.array2d[wp.float64],
):
    row, column = wp.tid()
    value = wp.float64(0.0)
    for entry in range(offsets[row], offsets[row + 1]):
        value += values[entry] * rhs[columns[entry], column]
    out[row, column] = value


@wp.kernel(enable_backward=False)
def _multiply_vec3_batch(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    rhs: wp.array2d[wp.vec3d],
    out: wp.array2d[wp.vec3d],
):
    column, row = wp.tid()
    value = wp.vec3d(0.0)
    for entry in range(offsets[row], offsets[row + 1]):
        value += values[entry] * rhs[column, columns[entry]]
    out[column, row] = value


class SparseInverse:
    """Reusable exact inverse factor for a constant SPD scalar sparse matrix.

    Setup runs on the CPU using sparse dictionaries and NumPy. Application
    runs on the selected Warp device with two CSR products and no host copies.
    Factors and scratch belong to this instance; simultaneous applications on
    different streams require separate instances. This operator has no automatic
    backward; an implicit adjoint can apply the same symmetric inverse.

    Args:
        matrix_rows: Full symmetric matrix as a list of column/value dictionaries.
            Both halves must agree exactly. Missing entries are zero. Coefficient
            units are determined by the caller's linear system.
        device: Warp execution device.
    """

    def __init__(self, matrix_rows: list[dict[int, float]], device: str | wp.Device | None = None):
        rows = _validate_rows(matrix_rows)
        self.size = len(rows)
        self.device = wp.get_device(device)
        adjacency = [set(row) - {i} for i, row in enumerate(rows)]
        self.permutation = _nested_dissection(adjacency)
        lower, diagonal = _factor_ldlt(rows, self.permutation)
        self.factorization_nnz = self.size + sum(map(len, lower))
        self.matrix_nnz = sum(map(len, rows))
        inverse_rows = _inverse_rows(lower, diagonal, self.permutation)
        transpose = [{} for _ in rows]
        for row, entries in enumerate(inverse_rows):
            for column, value in entries.items():
                transpose[column][row] = value
        self._s_numpy = _csr(inverse_rows)
        self._st_numpy = _csr(transpose)
        self.nnz = len(self._s_numpy[1])
        if not np.isfinite(self._s_numpy[2]).all():
            raise ValueError("inverse factor overflowed float64")
        for prefix, arrays in (("s", self._s_numpy), ("st", self._st_numpy)):
            for name, values, dtype in zip(
                ("offsets", "columns", "values"), arrays, (wp.int32, wp.int32, wp.float64), strict=True
            ):
                setattr(self, f"{prefix}_{name}", wp.array(values, dtype=dtype, device=self.device))
        self._vector_work = wp.empty(self.size, dtype=wp.vec3d, device=self.device)
        self._scalar_work = wp.empty(self.size, dtype=wp.float64, device=self.device)
        self._batch_work = None
        self._batch_capacity = 0
        self.factor_bytes = sum(array.nbytes for arrays in (self._s_numpy, self._st_numpy) for array in arrays)
        self.memory_bytes = self.factor_bytes + self.size * 4 * np.dtype(np.float64).itemsize

    def _apply(self, rhs, out, dtype, kernel, work):
        for name, value in (("rhs", rhs), ("out", out)):
            if value.dtype != dtype or value.shape != (self.size,) or value.device != self.device:
                raise ValueError(f"{name} must have shape ({self.size},), dtype {dtype}, on {self.device}")
        if not self.size:
            return
        tiled = self.device.is_cuda and kernel is _multiply_vec3
        launch = wp.launch_tiled if tiled else wp.launch
        if tiled:
            kernel = _multiply_vec3_tiled
        options = {"block_dim": 32} if tiled else {}
        launch(
            kernel,
            dim=self.size,
            inputs=[self.s_offsets, self.s_columns, self.s_values, rhs],
            outputs=[work],
            device=self.device,
            **options,
        )
        launch(
            kernel,
            dim=self.size,
            inputs=[self.st_offsets, self.st_columns, self.st_values, work],
            outputs=[out],
            device=self.device,
            **options,
        )

    def apply(self, rhs: wp.array[wp.vec3d], out: wp.array[wp.vec3d]) -> None:
        """Solve three coordinate columns; output units are RHS units / matrix units."""
        self._apply(rhs, out, wp.vec3d, _multiply_vec3, self._vector_work)

    def apply_scalar(self, rhs: wp.array[wp.float64], out: wp.array[wp.float64]) -> None:
        """Solve one column; output units are RHS units / matrix units."""
        self._apply(rhs, out, wp.float64, _multiply_scalar, self._scalar_work)

    def prepare_batch(self, columns: int) -> None:
        """Reserve one workspace for up to ``columns`` right-hand sides.

        Reserve the largest required batch before CUDA graph capture. Growing
        the workspace replaces its allocation; recapture any older batch graphs.
        """
        if not isinstance(columns, (int, np.integer)) or columns < 0:
            raise ValueError("columns must be a nonnegative integer")
        columns = int(columns)
        if columns > self._batch_capacity:
            if self.device.is_capturing:
                raise RuntimeError("Call prepare_batch before capturing a larger batch")
            self._batch_work = wp.empty((self.size, columns), dtype=wp.float64, device=self.device)
            self.memory_bytes += self.size * (columns - self._batch_capacity) * np.dtype(np.float64).itemsize
            self._batch_capacity = columns

    def apply_batch(self, rhs: wp.array2d[wp.float64], out: wp.array2d[wp.float64]) -> None:
        """Solve arbitrary coordinate/contact columns entirely on the device.

        Arrays have shape ``(size, columns)``. Scratch grows only when the column
        count exceeds its capacity; use :meth:`prepare_batch` before graph capture.
        Output units are RHS units divided by matrix coefficient units.
        """
        for name, value in (("rhs", rhs), ("out", out)):
            if (
                value.dtype != wp.float64
                or value.ndim != 2
                or value.shape[0] != self.size
                or value.device != self.device
            ):
                raise ValueError(f"{name} must be a float64 array of shape (size, columns) on {self.device}")
        if rhs.shape != out.shape:
            raise ValueError("rhs and out must have identical shapes")
        columns = rhs.shape[1]
        self.prepare_batch(columns)
        if not self.size or not columns:
            return
        work = self._batch_work
        wp.launch(
            _multiply_batch,
            dim=rhs.shape,
            inputs=[self.s_offsets, self.s_columns, self.s_values, rhs],
            outputs=[work],
            device=self.device,
        )
        wp.launch(
            _multiply_batch,
            dim=rhs.shape,
            inputs=[self.st_offsets, self.st_columns, self.st_values, work],
            outputs=[out],
            device=self.device,
        )

    def reserve_vec3_batch(self, shape) -> None:
        """Allocate the scratch of :meth:`apply_vec3_batch` for ``(columns, size)`` fields ahead of graph capture."""
        work = getattr(self, "_vec3_batch_work", None)
        if work is None or work.shape != tuple(shape):
            self._vec3_batch_work = wp.empty(tuple(shape), dtype=wp.vec3d, device=self.device)

    def apply_vec3_batch(self, rhs: wp.array2d[wp.vec3d], out: wp.array2d[wp.vec3d]) -> None:
        """Solve ``(columns, size)`` coordinate fields at once; scratch is reserved on the first call."""
        if rhs.shape != out.shape or rhs.ndim != 2 or rhs.shape[1] != self.size:
            raise ValueError("rhs and out must have identical shapes (columns, size)")
        work = getattr(self, "_vec3_batch_work", None)
        if work is None or work.shape != rhs.shape:
            if self.device.is_capturing:
                raise RuntimeError("Reserve the batch before graph capture")
            work = wp.empty(rhs.shape, dtype=wp.vec3d, device=self.device)
            self._vec3_batch_work = work
        for (offsets, columns, values), source, target in (
            ((self.s_offsets, self.s_columns, self.s_values), rhs, work),
            ((self.st_offsets, self.st_columns, self.st_values), work, out),
        ):
            wp.launch(
                _multiply_vec3_batch,
                rhs.shape,
                inputs=[offsets, columns, values, source],
                outputs=[target],
                device=self.device,
            )

    def apply_numpy(self, rhs: np.ndarray) -> np.ndarray:
        """Apply the factors on the CPU to RHS shape ``(size,)`` or ``(size, k)``.

        Intended for setup and reference comparisons, including contact columns.
        Output units are RHS units divided by matrix coefficient units.
        """
        rhs = np.asarray(rhs, dtype=np.float64)
        if rhs.ndim not in (1, 2) or rhs.shape[0] != self.size or not np.isfinite(rhs).all():
            raise ValueError("rhs must be finite with shape (size,) or (size, columns)")
        result = rhs
        for offsets, columns, values in (self._s_numpy, self._st_numpy):
            output = np.zeros_like(result)
            for row in range(self.size):
                start, end = offsets[row], offsets[row + 1]
                output[row] = values[start:end] @ result[columns[start:end]]
            result = output
        return result
