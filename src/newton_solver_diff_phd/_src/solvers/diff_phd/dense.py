# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Dense symmetric solves with Eigen's ``LDLT`` algorithm on the device.

The reference solves its contact Newton matrix and its Woodbury Gram matrix
with ``Eigen::MatrixXd::ldlt()``: a left-looking factorization that pivots
on the largest remaining original diagonal entry, reads only the lower
triangle and applies the pseudo-inverse of ``D``. It accepts symmetric
indefinite matrices. The kernel runs on one thread for a device-side size,
so it stays inside CUDA graphs; the solver only passes it the coupled
blocks, whose size is the number of active contacts or selected rows.
Small symmetric positive-definite blocks are instead solved by a tile
Cholesky factorization of one thread block, which is about ten times faster.
"""

import warp as wp

from .projections import DOUBLE_MIN

# Coupled SPD blocks up to this size are solved cooperatively in one tile.
TILE = 32


@wp.kernel(enable_backward=False)
def spd_tile_solve(matrix: wp.array2d[wp.float64], rhs: wp.array[wp.float64], count: wp.array[wp.int32]):
    """Overwrite ``rhs[:TILE]`` with the solution for an SPD block of at most ``TILE`` rows.

    The caller pads ``matrix[:TILE, :TILE]`` with the identity and ``rhs`` with
    zeros beyond ``count[0]``; larger blocks are left to :func:`ldlt_solve`.
    """
    _block = wp.tid()
    if count[0] > TILE:
        return
    factor = wp.tile_cholesky(wp.tile_load(matrix, shape=(TILE, TILE), storage="shared"))
    solution = wp.tile_cholesky_solve(factor, wp.tile_load(rhs, shape=TILE, storage="shared"))
    wp.tile_store(rhs, solution)


@wp.kernel(enable_backward=False)
def ldlt_solve(
    matrix: wp.array2d[wp.float64],
    rhs: wp.array[wp.float64],
    count: wp.array[wp.int32],
    minimum: int,
    permutation: wp.array[wp.int32],
    work: wp.array[wp.float64],
):
    """Overwrite ``rhs[:n]`` with the solution of ``matrix[:n, :n] x = rhs[:n]``, ``n = count[0]``.

    The lower triangle of the leading block of ``matrix`` is overwritten
    with the factor. Blocks with ``n <= minimum`` are skipped.
    """
    n = count[0]
    if n <= minimum:
        return
    for i in range(n):
        permutation[i] = i
    for k in range(n):
        biggest = k
        for i in range(k + 1, n):
            if wp.abs(matrix[i, i]) > wp.abs(matrix[biggest, biggest]):
                biggest = i
        if biggest != k:
            # Eigen's transposition restricted to the lower triangle.
            for j in range(k):
                value = matrix[k, j]
                matrix[k, j] = matrix[biggest, j]
                matrix[biggest, j] = value
            for i in range(biggest + 1, n):
                value = matrix[i, k]
                matrix[i, k] = matrix[i, biggest]
                matrix[i, biggest] = value
            value = matrix[k, k]
            matrix[k, k] = matrix[biggest, biggest]
            matrix[biggest, biggest] = value
            for i in range(k + 1, biggest):
                value = matrix[i, k]
                matrix[i, k] = matrix[biggest, i]
                matrix[biggest, i] = value
            index = permutation[k]
            permutation[k] = permutation[biggest]
            permutation[biggest] = index
        for j in range(k):
            matrix[k, k] = matrix[k, k] - matrix[k, j] * matrix[k, j] * matrix[j, j]
        for i in range(k + 1, n):
            for j in range(k):
                matrix[i, k] = matrix[i, k] - matrix[i, j] * matrix[j, j] * matrix[k, j]
            if matrix[k, k] != wp.float64(0.0):
                matrix[i, k] = matrix[i, k] / matrix[k, k]
    for i in range(n):
        work[i] = rhs[permutation[i]]
    for i in range(n):
        for j in range(i):
            work[i] = work[i] - matrix[i, j] * work[j]
    for i in range(n):
        if wp.abs(matrix[i, i]) > DOUBLE_MIN:
            work[i] = work[i] / matrix[i, i]
        else:
            work[i] = wp.float64(0.0)
    for reverse in range(n):
        i = n - 1 - reverse
        for j in range(i + 1, n):
            work[i] = work[i] - matrix[j, i] * work[j]
    for i in range(n):
        rhs[permutation[i]] = work[i]
