# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exact sparse inverse factors with CPU setup and two resident Warp SpMVs.

For a symmetric positive-definite scalar matrix, ``P A P.T = L D L.T``
and ``S = D**(-1/2) L**(-1) P`` give ``A**(-1) b = S.T (S b)``.
Only sparse dictionaries and CSR arrays are constructed; the dense inverse of
``A`` is never formed. Exact inverse factors can still have substantial fill.
This replaces the reference's per-coordinate ``Eigen::SimplicialLDLT`` solve
with the same matrix up to rounding.
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
    columns = [set() for _ in range(size)]
    for i, row in enumerate(work):
        for j in row:
            if j < i:
                columns[j].add(i)
    lower = [{} for _ in range(size)]
    diagonal = np.empty(size, dtype=np.float64)
    for k in range(size):
        pivot = work[k][k]
        if not np.isfinite(pivot) or pivot <= 0.0:
            raise ValueError(f"matrix is not positive definite: LDLT pivot {k} is {pivot}")
        diagonal[k] = pivot
        neighbors = sorted(columns[k])
        values = [work[i].pop(k) for i in neighbors]
        for i, value in zip(neighbors, values, strict=True):
            lower[i][k] = value / pivot
        for offset, i in enumerate(neighbors):
            multiplier = lower[i][k]
            for j, value in zip(neighbors[: offset + 1], values[: offset + 1], strict=True):
                if j not in work[i] and i != j:
                    columns[j].add(i)
                work[i][j] = work[i].get(j, 0.0) - multiplier * value
        columns[k].clear()
    return lower, diagonal


def _inverse_rows(lower, diagonal, permutation):
    inverse = []
    result = []
    for i, row in enumerate(lower):
        current = {i: 1.0}
        for k, coefficient in sorted(row.items()):
            for column, value in inverse[k].items():
                current[column] = current.get(column, 0.0) - coefficient * value
        # Only exact arithmetic zeros are omitted; there is no drop tolerance.
        current = {j: value for j, value in current.items() if value != 0.0}
        inverse.append(current)
        scale = 1.0 / np.sqrt(diagonal[i])
        result.append({int(permutation[j]): value * scale for j, value in current.items()})
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


class SparseInverse:
    """Reusable exact inverse factor for a constant SPD scalar sparse matrix.

    Setup runs on the CPU using sparse dictionaries and NumPy. Application
    runs on the selected Warp device with two CSR products and no host copies.

    Args:
        matrix_rows: Full symmetric matrix as a list of column/value dictionaries.
            Both halves must agree exactly. Missing entries are zero.
        device: Warp execution device.
    """

    def __init__(self, matrix_rows: list[dict[int, float]], device: str | wp.Device | None = None):
        rows = _validate_rows(matrix_rows)
        self.size = len(rows)
        self.device = wp.get_device(device)
        adjacency = [set(row) - {i} for i, row in enumerate(rows)]
        self.permutation = _nested_dissection(adjacency)
        lower, diagonal = _factor_ldlt(rows, self.permutation)
        inverse_rows = _inverse_rows(lower, diagonal, self.permutation)
        transpose = [{} for _ in rows]
        for row, entries in enumerate(inverse_rows):
            for column, value in entries.items():
                transpose[column][row] = value
        factors = (_csr(inverse_rows), _csr(transpose))
        self.nnz = len(factors[0][1])
        if not np.isfinite(factors[0][2]).all():
            raise ValueError("inverse factor overflowed float64")
        for prefix, arrays in zip(("s", "st"), factors, strict=True):
            for name, values, dtype in zip(
                ("offsets", "columns", "values"), arrays, (wp.int32, wp.int32, wp.float64), strict=True
            ):
                setattr(self, f"{prefix}_{name}", wp.array(values, dtype=dtype, device=self.device))
        self._work = wp.empty(self.size, dtype=wp.vec3d, device=self.device)

    def apply(self, rhs: wp.array[wp.vec3d], out: wp.array[wp.vec3d]) -> None:
        """Solve the three coordinate columns; output units are RHS units / matrix units."""
        for name, value in (("rhs", rhs), ("out", out)):
            if value.dtype != wp.vec3d or value.shape != (self.size,) or value.device != self.device:
                raise ValueError(f"{name} must have shape ({self.size},), dtype vec3d, on {self.device}")
        if not self.size:
            return
        tiled = self.device.is_cuda
        launch = wp.launch_tiled if tiled else wp.launch
        kernel = _multiply_vec3_tiled if tiled else _multiply_vec3
        options = {"block_dim": 32} if tiled else {}
        for (offsets, columns, values), source, target in (
            ((self.s_offsets, self.s_columns, self.s_values), rhs, self._work),
            ((self.st_offsets, self.st_columns, self.st_values), self._work, out),
        ):
            launch(
                kernel,
                dim=self.size,
                inputs=[offsets, columns, values, source],
                outputs=[target],
                device=self.device,
                **options,
            )
