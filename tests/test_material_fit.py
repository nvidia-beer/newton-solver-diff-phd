"""End-to-end checks of the package's synthetic two-material inverse problem."""

import os
import unittest

import numpy as np

from tests.material_fit_demo import MaterialFit, optimize


class TestMaterialFit(unittest.TestCase):
    def test_log_material_rollout_gradients_at_homogeneous_nominal_material(self):
        problem = MaterialFit(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        # At equal nominal weights the reference mean-weight K agrees with the
        # forward Jacobian. Perturb each group's parameter independently, so
        # this still verifies both material derivatives and their rollout chain.
        parameters = np.log([1.2, 1.2])
        _, _, gradient = problem.rollout(parameters, gradient=True)
        problem.solver.check_status()
        # State stores are float32; use two perturbation sizes above their
        # cancellation floor while retaining the existing derivative tolerance.
        for epsilon in (0.005, 0.01):
            for column in range(2):
                direction = np.eye(2)[column]
                plus = problem.rollout(parameters + epsilon * direction)[1]
                minus = problem.rollout(parameters - epsilon * direction)[1]
                finite_difference = (plus - minus) / (2.0 * epsilon)
                np.testing.assert_allclose(gradient[column], finite_difference, rtol=0.003, atol=0.001)

    def test_two_material_fit_reduces_trajectory_error(self):
        # Normalized descent zig-zags along the stiff direction for about ten
        # updates on this PD landscape before it converges.
        result = optimize(os.environ.get("NEWTON_TEST_DEVICE", "cpu"), iterations=12)
        self.assertLess(result["final_loss"], result["initial_loss"] * 0.001)
        np.testing.assert_allclose(result["stiffness_scales"], result["reference_scales"], rtol=0.03)


if __name__ == "__main__":
    unittest.main()
