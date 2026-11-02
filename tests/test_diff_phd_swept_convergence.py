# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Accelerated swept contact must converge to the unaccelerated equilibrium.

This public, twelve-tetrahedron fixture presses the interior of a free face
while applying an asymmetric tangential load. Accelerating the unconstrained
elastic residual before contact exhausted 300 iterations for both materials;
the unaccelerated constrained iteration converged in about thirty. Agreement
is checked through physical states, without inspecting acceleration history
or assuming a particular implementation order.
"""

import unittest

import numpy as np
import warp as wp

from newton_solver_diff_phd import SolverDiffPhD
from tests.test_diff_phd_solver import DEVICE
from tests.test_diff_phd_swept_contact import DENSITY, RADIUS, _scene


class TestSweptConstrainedConvergence(unittest.TestCase):
    def test_acceleration_matches_unaccelerated_loaded_press_and_hold(self):
        for material in ("pd", "neo_hookean"):
            with self.subTest(material=material):
                self._compare_press_and_hold(material)

    def _compare_press_and_hold(self, material):
        model, rest, tets, split, faces_left, faces_right, pinned_die = _scene()
        options = {
            "density": DENSITY,
            "material": material,
            "swept_contact": (RADIUS, split, faces_left, faces_right),
            "max_iterations": 300,
            "abs_tol": 1.0e-11,
            "rel_tol": 1.0e-9,
        }
        unaccelerated = SolverDiffPhD(model, anderson_window=0, **options)
        accelerated = SolverDiffPhD(model, anderson_window=5, **options)
        state = model.state()
        force = np.zeros_like(rest)
        force[1, 0] = 3.0
        force[7, 1] = -1.5
        state.particle_f.assign(force)
        target = rest.copy()
        target[split:, 2] -= 0.0085
        targets = wp.array(target, dtype=wp.vec3, device=DEVICE)
        rest_elements = rest[tets].astype(np.float64)
        rest_determinants = np.linalg.det(rest_elements[:, 1:] - rest_elements[:, :1])

        for stage in ("press", "hold"):
            with self.subTest(stage=stage):
                expected, actual = model.state(), model.state()
                # Both methods see the identical physical input and prescribed
                # endpoint. The hold starts from the converged baseline state.
                unaccelerated.step(state, expected, None, None, 0.01, dirichlet_positions=targets)
                unaccelerated.check_status()
                accelerated.step(state, actual, None, None, 0.01, dirichlet_positions=targets)
                accelerated.check_status()
                q, q_reference = actual.particle_q.numpy(), expected.particle_q.numpy()
                np.testing.assert_allclose(q[:split], q_reference[:split], rtol=0.0, atol=2.0e-7)
                np.testing.assert_allclose(
                    actual.particle_qd.numpy()[:split], expected.particle_qd.numpy()[:split], rtol=0.0, atol=2.0e-5
                )
                np.testing.assert_array_equal(q[split:], target[split:])
                np.testing.assert_array_equal(q[pinned_die], rest[pinned_die])
                self.assertGreater(np.linalg.norm(q[:split] - rest[:split], axis=1).max(), 0.001)
                elements = q[tets].astype(np.float64)
                jacobians = np.linalg.det(elements[:, 1:] - elements[:, :1]) / rest_determinants
                self.assertGreater(jacobians.min(), 0.0)
                state = expected
                state.particle_qd.assign(0.3 * state.particle_qd.numpy())
                state.particle_f.assign(force)


if __name__ == "__main__":
    unittest.main()
