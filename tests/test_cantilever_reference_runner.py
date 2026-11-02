# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Independent mathematical checks for the native Hex8 comparison diagnostics."""

import unittest

import numpy as np

from tools.parity.cantilever_reference_runner import applied_force, hex_quadrature, initial_twist


class TestCantileverReferenceRunner(unittest.TestCase):
    def test_gauss_gradients_reproduce_affine_fields_at_every_sample(self):
        corners = np.array(
            [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=float
        )
        vertices = corners * 0.016 + [0.24, -0.08, 0.12]
        gradients, volume = hex_quadrature(vertices, np.arange(8)[None])
        affine = np.array([[1.1, 0.2, -0.1], [0.1, 0.95, 0.1], [0.05, 0.0, 1.02]])
        deformed = vertices @ affine.T + [3.0, -2.0, 1.0]
        deformation = np.einsum("ni,snj->sij", deformed, gradients)
        np.testing.assert_allclose(deformation, np.broadcast_to(affine, (8, 3, 3)), rtol=0, atol=6e-14)
        np.testing.assert_allclose(gradients.sum(axis=1), 0, rtol=0, atol=1e-14)
        self.assertAlmostEqual(volume * 8, 0.016**3, places=18)

    def test_loading_uses_source_zero_based_frame_index_and_per_node_force(self):
        asset = {"vertices": np.zeros((40, 3)), "right": np.arange(4, 40)}
        np.testing.assert_array_equal(applied_force(asset, "axial", 1), 0)
        force = applied_force(asset, "axial", 25)
        np.testing.assert_array_equal(force[:4], 0)
        np.testing.assert_allclose(force[4:, 0], 76.8, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(force[:, 1:], 0)
        np.testing.assert_array_equal(applied_force(asset, "twist", 25), 0)

    def test_twist_preserves_section_radius_and_clamped_face(self):
        vertices = np.array([[x, y, z] for x in [0, 0.16, 0.32] for y in [0, 0.08] for z in [0, 0.08]])
        twisted = initial_twist(vertices)
        np.testing.assert_array_equal(twisted[:4], vertices[:4])
        np.testing.assert_array_equal(twisted[:, 0], vertices[:, 0])
        radius = np.linalg.norm(vertices[:, 1:] - 0.04, axis=1)
        np.testing.assert_allclose(np.linalg.norm(twisted[:, 1:] - 0.04, axis=1), radius, rtol=0, atol=2e-17)


if __name__ == "__main__":
    unittest.main()
