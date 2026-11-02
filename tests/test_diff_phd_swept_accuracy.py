# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Near-parallel contact features require a more accurate inner normal solve."""

import unittest

import numpy as np
import warp as wp

from newton_solver_diff_phd._src.solvers.diff_phd.ccd import SweptMeshCCD
from newton_solver_diff_phd._src.solvers.diff_phd.sparse import SparseInverse
from newton_solver_diff_phd._src.solvers.diff_phd.swept_contact import SweptContact
from tests.test_diff_phd_solver import DEVICE


class TestSweptNestedAccuracy(unittest.TestCase):
    def test_near_parallel_planes_recover_unique_metric_projection(self):
        # The two prescribed triangles lie on z=+/-slope*x. Only vertex 0
        # moves, with identity compliance. The upper feasible region is
        # z >= slope*abs(x) + margin*sqrt(1+slope**2), so its nearest point
        # to (0, 0, -.003) is analytic and unique. A 1e-8 inner gap tolerance
        # accepts just one plane and misses that point by 3.5 micrometres.
        slope, margin, tolerance = 0.001, 0.0005, 1.0e-7
        triangle = np.array([[-0.05, -0.05, 0], [0.05, -0.05, 0], [0, 0.05, 0]])
        first, second = triangle.copy(), triangle.copy()
        first[:, 2], second[:, 2] = slope * first[:, 0], -slope * second[:, 0]
        start = np.vstack(([[0, 0, 0.003], [-0.2, 0.1, 0.1], [-0.2, -0.1, 0.1]], first, second))
        candidate = start.copy()
        candidate[0, 2] = -0.003
        left, right = [[0, 1, 2]], [[3, 4, 5], [6, 7, 8]]
        contact = SweptContact(left, right, 3, [True] + [False] * 8, margin, DEVICE, capacity=32, tolerance=tolerance)
        contact.build(SparseInverse([{i: 1.0} for i in range(9)], DEVICE), 0.01)
        initial = wp.array(start, dtype=wp.vec3d, device=DEVICE)
        trial = wp.array(candidate, dtype=wp.vec3d, device=DEVICE)
        output = wp.empty_like(initial)
        contact.project(initial, trial, output)
        result = output.numpy()
        self.assertEqual(int(contact.status.numpy()[0]), 0)
        self.assertLessEqual(float(contact.residual.numpy()[0]), tolerance)
        np.testing.assert_array_equal(result[1:], candidate[1:])
        expected = [0, 0, margin * np.sqrt(1 + slope**2)]
        np.testing.assert_allclose(result[0], expected, rtol=0, atol=tolerance)

        # Both current planes carry force at their actual geometric margin.
        count = int(contact.count.numpy()[0])
        self.assertEqual(count, 2)
        multipliers = contact.multiplier.numpy()[:count]
        self.assertTrue(np.all(multipliers > 0))
        plane_normals = np.array([[slope, 0, 1], [-slope, 0, 1]]) / np.sqrt(1 + slope**2)
        np.testing.assert_allclose(plane_normals @ result[0], margin, rtol=0, atol=tolerance)
        normals = contact.normals.numpy()[:count]
        np.testing.assert_allclose(result[0] - candidate[0], multipliers @ normals, rtol=0, atol=tolerance)

        # The complete accepted motion must also pass a fresh physical query.
        detector = SweptMeshCCD(left, right, 9, margin=margin, device=DEVICE)
        rows = [wp.zeros(32, dtype=dtype, device=DEVICE) for dtype in (wp.vec4i, wp.vec4d, wp.vec3d, wp.float64)]
        row_count, flags = (wp.zeros(1, dtype=int, device=DEVICE) for _ in range(2))
        detector.append(initial, output, *rows, row_count, flags)
        self.assertEqual(int(flags.numpy()[0]), 0)
        self.assertEqual(int(detector.hit_count.numpy()[0]), 0)


if __name__ == "__main__":
    unittest.main()
