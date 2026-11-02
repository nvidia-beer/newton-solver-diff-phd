# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Independent solve and exact-cancellation checks for the sparse inverse setup."""

import os
import unittest

import numpy as np
import warp as wp

from newton_solver_diff_phd._src.solvers.diff_phd.sparse import SparseInverse

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")


def matrix_rows(matrix):
    return [{int(j): float(row[j]) for j in np.flatnonzero(row)} for row in matrix]


class TestSparseInverse(unittest.TestCase):
    def test_exact_cancellation_removes_only_zero_entries(self):
        # A = L diag(2, 3, 5) L.T with L's strict lower triangle all ones.
        # In L^-1, entry (2, 0) cancels exactly; the other five entries remain.
        matrix = np.asarray(((2.0, 2.0, 2.0), (2.0, 5.0, 5.0), (2.0, 5.0, 10.0)))
        factor = SparseInverse(matrix_rows(matrix), DEVICE)
        np.testing.assert_array_equal(factor.permutation, (0, 1, 2))
        np.testing.assert_array_equal(factor._s_numpy[0], (0, 1, 3, 5))
        np.testing.assert_array_equal(factor._s_numpy[1], (0, 0, 1, 1, 2))
        np.testing.assert_array_equal(
            factor._s_numpy[2], (1 / np.sqrt(2), -1 / np.sqrt(3), 1 / np.sqrt(3), -1 / np.sqrt(5), 1 / np.sqrt(5))
        )
        rhs = np.asarray((0.5, -2.0, 7.0))
        result = wp.empty(3, dtype=wp.float64, device=DEVICE)
        factor.apply_scalar(wp.array(rhs, dtype=wp.float64, device=DEVICE), result)
        np.testing.assert_allclose(result.numpy(), np.linalg.solve(matrix, rhs), rtol=2e-15, atol=2e-15)

    def test_grid_fill_and_disconnected_pins_match_dense_oracle(self):
        # Weighted 8x7 grid plus three disconnected prescribed rows exercises
        # nontrivial ordering, fill and multiple connected components.
        nx, ny = 8, 7
        size = nx * ny + 3
        rng = np.random.default_rng(519)
        matrix = np.eye(size)
        for y in range(ny):
            for x in range(nx):
                i = y * nx + x
                neighbors = ([i + 1] if x + 1 < nx else []) + ([i + nx] if y + 1 < ny else [])
                for j in neighbors:
                    weight = rng.uniform(0.1, 5.0)
                    matrix[i, i] += weight
                    matrix[j, j] += weight
                    matrix[i, j] = matrix[j, i] = -weight
        factor = SparseInverse(matrix_rows(matrix), DEVICE)
        self.assertGreater(factor.factorization_nnz, (factor.matrix_nnz + size) // 2)
        rhs = rng.normal(size=(size, 3))
        result = wp.empty(size, dtype=wp.vec3d, device=DEVICE)
        factor.apply(wp.array(rhs, dtype=wp.vec3d, device=DEVICE), result)
        expected = np.linalg.solve(matrix, rhs)
        np.testing.assert_allclose(result.numpy(), expected, rtol=2e-12, atol=2e-13)
        # Isolated prescribed rows are exactly the identity through both SpMVs.
        np.testing.assert_array_equal(result.numpy()[-3:], rhs[-3:])
        for offsets, columns, values in (factor._s_numpy, factor._st_numpy):
            self.assertEqual(len(columns), len(values))
            self.assertEqual(offsets[-1], len(values))
            self.assertTrue(np.all(values != 0.0))
            for i in range(size):
                self.assertTrue(np.all(np.diff(columns[offsets[i] : offsets[i + 1]]) > 0))


if __name__ == "__main__":
    unittest.main()
