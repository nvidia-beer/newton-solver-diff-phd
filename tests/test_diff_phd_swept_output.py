# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The noninversion guard must cover the coordinates actually stored in State."""

import unittest

import numpy as np
import warp as wp

from newton_solver_diff_phd._src.solvers.diff_phd.solver import (
    STATUS_SWEPT_INVALID_ELEMENT,
    _validate_swept_elements,
    _write_swept_state,
)
from tests.test_diff_phd_solver import DEVICE


class TestSweptStoredGeometry(unittest.TestCase):
    def test_float32_degenerate_candidate_is_rejected_and_state_is_restored(self):
        rest = np.array([[1, 1, 1], [1.1, 1, 1], [1, 1.1, 1], [1, 1, 1.1]], dtype=np.float32)
        edges = (rest[1:].astype(np.float64) - rest[0]).T
        inverse = np.linalg.inv(edges)
        gradients = np.vstack((-inverse.sum(axis=0), inverse))
        basis = wp.array(gradients[None], dtype=wp.vec3d, device=DEVICE)
        indices = wp.array([[0, 1, 2, 3]], dtype=wp.vec4i, device=DEVICE)
        initial = wp.array(rest, dtype=wp.vec3, device=DEVICE)
        velocities = np.full((4, 3), 0.25, dtype=np.float32)
        incoming_velocity = wp.array(velocities, dtype=wp.vec3, device=DEVICE)
        output, output_velocity = wp.empty_like(initial), wp.empty_like(initial)
        for height, valid in ((5.0e-8, False), (9.0e-8, True)):
            with self.subTest(height=height):
                candidate = rest.astype(np.float64)
                candidate[3, 2] = 1 + height
                jacobian = np.linalg.det((candidate[1:] - candidate[0]).T) / np.linalg.det(edges)
                self.assertGreater(jacobian, 1.0e-8)
                x = wp.array(candidate, dtype=wp.vec3d, device=DEVICE)
                status = wp.zeros(1, dtype=int, device=DEVICE)
                wp.launch(_validate_swept_elements, 1, inputs=[x, indices, basis], outputs=[status], device=DEVICE)
                wp.launch(
                    _write_swept_state,
                    4,
                    inputs=[x, initial, incoming_velocity, 0.01, status],
                    outputs=[output, output_velocity],
                    device=DEVICE,
                )
                if valid:
                    self.assertEqual(int(status.numpy()[0]), 0)
                    np.testing.assert_array_equal(output.numpy(), candidate.astype(np.float32))
                    stored = output.numpy().astype(np.float64)
                    self.assertGreater(np.linalg.det((stored[1:] - stored[0]).T) / np.linalg.det(edges), 1.0e-8)
                else:
                    self.assertTrue(int(status.numpy()[0]) & STATUS_SWEPT_INVALID_ELEMENT)
                    np.testing.assert_array_equal(output.numpy(), rest)
                    np.testing.assert_array_equal(output_velocity.numpy(), velocities)


if __name__ == "__main__":
    unittest.main()
