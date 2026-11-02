# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Offline embedding geometry regressions with public synthetic tetrahedra."""

import unittest

import numpy as np

from tools.prepare_poke_reference import embedding


class TestPokeReferenceEmbedding(unittest.TestCase):
    def test_containing_elongated_cell_outside_nearest_128_centroids(self):
        long_cell = np.array(((0, 0, 0), (1000, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float64)
        small_cell = np.array(((0, 0, 0), (0.01, 0, 0), (0, 0.01, 0), (0, 0, 0.01)))
        decoys = [small_cell + (1 + i * 0.002, 2, 2) for i in range(130)]
        vertices = np.concatenate([*decoys, long_cell])
        tets = np.arange(len(vertices), dtype=np.int32).reshape(-1, 4)
        point = np.array(((0.1, 0.1, 0.1),))
        # The only containing cell is farther away than every decoy centroid.
        distance = np.linalg.norm(vertices[tets].mean(axis=1) - point, axis=1)
        self.assertEqual(int(np.argmax(distance)), len(tets) - 1)
        indices, weights, error = embedding(point, vertices, tets)
        np.testing.assert_array_equal(indices[0], tets[-1])
        self.assertGreaterEqual(weights.min(), 0)
        np.testing.assert_allclose(weights[0], (0.7999, 0.0001, 0.1, 0.1), atol=1e-15)
        self.assertLess(error, 1e-14)

    def test_genuinely_outside_vertex_keeps_affine_extrapolation(self):
        vertices = np.array(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float64)
        tets = np.array(((0, 1, 2, 3),), dtype=np.int32)
        points = np.array(((2, 2, 2), (0.1, 0.2, 0.3), (0, 0.25, 0.75)), dtype=np.float64)
        before = vertices.copy()
        indices, weights, error = embedding(points, vertices, tets)
        np.testing.assert_array_equal(indices, np.tile(tets, (len(points), 1)))
        np.testing.assert_array_equal(weights[0], (-5, 2, 2, 2))
        self.assertGreaterEqual(weights[1:].min(), 0)
        np.testing.assert_allclose(np.einsum("ni,nij->nj", weights, vertices[indices]), points, atol=1e-15)
        np.testing.assert_array_equal(vertices, before)
        self.assertLess(error, 1e-14)


if __name__ == "__main__":
    unittest.main()
