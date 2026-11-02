# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Current closest geometry, rather than a stale sweep plane, determines force."""

import unittest

import numpy as np
import warp as wp

from newton_solver_diff_phd._src.solvers.diff_phd.ccd import SweptMeshCCD
from newton_solver_diff_phd._src.solvers.diff_phd.sparse import SparseInverse
from newton_solver_diff_phd._src.solvers.diff_phd.swept_contact import SweptContact
from tests.test_diff_phd_solver import DEVICE


class TestSweptCurrentGeometry(unittest.TestCase):
    margin = 0.0005
    tolerance = 1.0e-9

    def _project(self, initial_point, endpoint):
        # Only node 0 moves. Its closest point on the prescribed triangle is
        # always (0, .2, 0), on the interior of the triangle's vertical edge.
        # The two other left vertices keep all other features well separated.
        start = np.array([initial_point, [-0.1, 0.2, 0.1], [-0.1, 0.3, 0.1], [0, 0, 0], [1, 0, 0], [0, 1, 0]])
        candidate = start.copy()
        candidate[0] = endpoint
        contact = SweptContact(
            [[0, 1, 2]],
            [[3, 4, 5]],
            3,
            [True, False, False, False, False, False],
            self.margin,
            DEVICE,
            capacity=32,
            tolerance=self.tolerance,
        )
        contact.build(SparseInverse([{i: 1.0} for i in range(6)], DEVICE), 0.01)
        start_array = wp.array(start, dtype=wp.vec3d, device=DEVICE)
        candidate_array = wp.array(candidate, dtype=wp.vec3d, device=DEVICE)
        output = wp.empty_like(start_array)
        contact.project(start_array, candidate_array, output)
        self.assertEqual(int(contact.status.numpy()[0]), 0)
        self.assertLessEqual(float(contact.residual.numpy()[0]), self.tolerance)
        np.testing.assert_array_equal(output.numpy()[1:], candidate[1:])

        # A new detector with an empty row list independently checks the full
        # accepted motion, rather than trusting cached response diagnostics.
        detector = SweptMeshCCD([[0, 1, 2]], [[3, 4, 5]], 6, margin=self.margin, device=DEVICE)
        rows = [wp.zeros(32, dtype=dtype, device=DEVICE) for dtype in (wp.vec4i, wp.vec4d, wp.vec3d, wp.float64)]
        count, flags = (wp.zeros(1, dtype=int, device=DEVICE) for _ in range(2))
        detector.append(start_array, output, *rows, count, flags)
        self.assertEqual(int(flags.numpy()[0]), 0)
        self.assertEqual(int(detector.hit_count.numpy()[0]), 0)
        return contact, candidate, output.numpy()

    def test_force_uses_current_edge_normal_and_exact_nearest_feasible_point(self):
        contact, candidate, output = self._project([-0.002, 0.2, 0.001], [-0.0001, 0.2, 0.0001])
        closest = np.array([0.0, 0.2, 0.0])
        normal = (candidate[0] - closest) / np.linalg.norm(candidate[0] - closest)
        expected = closest + self.margin * normal
        np.testing.assert_allclose(output[0], expected, rtol=0.0, atol=self.tolerance)

        # A=I makes the unique minimum-displacement projection analytic.
        # Positive normal reaction must occur at the actual geometric margin,
        # and its direction must balance the displacement from the free trial.
        multipliers = contact.multiplier.numpy()[: int(contact.count.numpy()[0])]
        self.assertGreater(float(multipliers.sum()), 0.0)
        self.assertTrue(np.all(multipliers >= 0.0))
        self.assertAlmostEqual(np.linalg.norm(output[0] - closest), self.margin, delta=self.tolerance)
        np.testing.assert_allclose(output[0] - candidate[0], multipliers.sum() * normal, rtol=0, atol=self.tolerance)

    def test_separated_endpoint_has_zero_force_despite_old_active_sweep_plane(self):
        # This collision-free motion passes around the edge. Its initial
        # supporting plane would incorrectly push the final x from -.1 to
        # -.5 mm, although the actual endpoint is already .608 mm away.
        contact, candidate, output = self._project([-0.0005, 0.2, 0.0], [-0.0001, 0.2, 0.0006])
        self.assertGreater(int(contact.count.numpy()[0]), 0)
        self.assertGreater(np.linalg.norm(candidate[0] - [0.0, 0.2, 0.0]), self.margin)
        np.testing.assert_allclose(output, candidate, rtol=0.0, atol=self.tolerance)
        np.testing.assert_array_equal(contact.multiplier.numpy(), 0.0)


if __name__ == "__main__":
    unittest.main()
