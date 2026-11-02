# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A difficult trial sweep can be corrected, but never accepted uncertified."""

import unittest

import numpy as np
import warp as wp

from newton_solver_diff_phd._src.solvers.diff_phd.sparse import SparseInverse
from newton_solver_diff_phd._src.solvers.diff_phd.swept_contact import CCD_UNRESOLVED, SweptContact
from tests.test_diff_phd_solver import DEVICE


@wp.kernel(enable_backward=False)
def _uncertain(flags: wp.array[int]):
    wp.atomic_or(flags, 0, CCD_UNRESOLVED)


class TestSweptResponseCertification(unittest.TestCase):
    def _fixture(self):
        left = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
        start = np.vstack((left, left + [0.0, 0.0, 0.01]))
        candidate = start.copy()
        candidate[3:, 2] = -0.01
        contact = SweptContact(
            [[0, 1, 2]],
            [[3, 4, 5]],
            3,
            [True, True, True, False, False, False],
            0.0005,
            DEVICE,
            capacity=64,
            max_rounds=8,
        )
        contact.build(SparseInverse([{i: 1.0} for i in range(6)], DEVICE), 0.01)
        return (
            contact,
            wp.array(start, dtype=wp.vec3d, device=DEVICE),
            wp.array(candidate, dtype=wp.vec3d, device=DEVICE),
        )

    def test_uncertain_crossing_trial_is_corrected_then_freshly_certified(self):
        contact, start, candidate = self._fixture()
        contact.detector.max_iterations = 1
        contact.detector.append(
            start,
            candidate,
            contact.indices,
            contact.coefficients,
            contact.normals,
            contact.tois,
            contact.count,
            contact.flags,
        )
        self.assertEqual(int(contact.flags.numpy()[0]), CCD_UNRESOLVED)
        self.assertGreater(int(contact.count.numpy()[0]), 0)
        out = wp.empty_like(start)
        contact.project(start, candidate, out)
        self.assertEqual(int(contact.status.numpy()[0]), 0)
        self.assertEqual(int(contact.flags.numpy()[0]), 0)
        self.assertEqual(int(contact.detector.hit_count.numpy()[0]), 0)
        self.assertGreaterEqual(int(contact.iterations.numpy()[0]), 1)
        np.testing.assert_allclose(out.numpy()[:3, 2], -0.0105, rtol=0, atol=1.0e-9)
        np.testing.assert_array_equal(out.numpy()[3:], candidate.numpy()[3:])

    def test_corrected_endpoint_is_rejected_when_fresh_sweep_remains_uncertain(self):
        contact, start, candidate = self._fixture()
        append = contact.detector.append

        def unresolved(*args):
            append(*args)
            wp.launch(_uncertain, 1, outputs=[contact.flags], device=DEVICE)

        contact.detector.append = unresolved
        out = wp.empty_like(start)
        contact.project(start, candidate, out)
        self.assertEqual(int(contact.status.numpy()[0]), CCD_UNRESOLVED)
        self.assertGreaterEqual(int(contact.iterations.numpy()[0]), 1)
        np.testing.assert_array_equal(out.numpy(), start.numpy())

    def test_uncertain_feasible_candidate_is_not_accepted_or_moved(self):
        contact, start, _candidate = self._fixture()
        append = contact.detector.append

        def unresolved(*args):
            append(*args)
            wp.launch(_uncertain, 1, outputs=[contact.flags], device=DEVICE)

        contact.detector.append = unresolved
        out = wp.empty_like(start)
        contact.project(start, start, out)
        self.assertEqual(int(contact.status.numpy()[0]), CCD_UNRESOLVED)
        self.assertEqual(int(contact.iterations.numpy()[0]), 0)
        np.testing.assert_array_equal(out.numpy(), start.numpy())


if __name__ == "__main__":
    unittest.main()
