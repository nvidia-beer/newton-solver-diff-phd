"""End-to-end checks of the package's synthetic two-material inverse problem."""

import os
import unittest

import numpy as np

from tests.material_fit_demo import MaterialFit, optimize


class TestMaterialFit(unittest.TestCase):
    def test_log_material_rollout_gradients(self):
        problem = MaterialFit(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        parameters = np.log([1.6, 0.8])
        _, _, gradient = problem.rollout(parameters, gradient=True)
        for epsilon in (0.002, 0.005):
            for column in range(2):
                direction = np.eye(2)[column]
                plus = problem.rollout(parameters + epsilon * direction)[1]
                minus = problem.rollout(parameters - epsilon * direction)[1]
                finite_difference = (plus - minus) / (2.0 * epsilon)
                np.testing.assert_allclose(gradient[column], finite_difference, rtol=0.003, atol=0.001)

    def test_two_material_fit_reduces_trajectory_error(self):
        result = optimize(os.environ.get("NEWTON_TEST_DEVICE", "cpu"), iterations=6)
        self.assertLess(result["final_loss"], result["initial_loss"] * 0.001)
        np.testing.assert_allclose(result["stiffness_scales"], result["reference_scales"], rtol=0.03)


if __name__ == "__main__":
    unittest.main()
