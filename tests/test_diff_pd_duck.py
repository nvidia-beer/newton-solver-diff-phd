# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The bundled DiffPD duck mesh and the first steps of its example scene."""

import os
import unittest
from pathlib import Path

import newton.viewer
import numpy as np
import warp as wp

from newton_solver_diff_phd import load_mesh
from newton_solver_diff_phd.examples import example_softbody_diff_pd_duck as duck

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
ASSET = Path(duck.__file__).parent / "assets" / "duck.usda"


class TestDuckMesh(unittest.TestCase):
    def test_tetrahedra_and_surface(self):
        from pxr import Usd

        mesh = load_mesh(ASSET)[0]
        self.assertEqual(mesh.vertices.shape, (5130, 3))
        self.assertEqual(mesh.indices.shape, (21438, 4))
        corners = mesh.vertices[mesh.indices]
        volumes = np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6.0
        self.assertTrue(np.all(volumes > 0.0))
        self.assertEqual(len(np.unique(mesh.indices)), len(mesh.vertices))
        # Faces used by exactly one tetrahedron form the boundary.
        faces = np.sort(mesh.indices[:, [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]].reshape(-1, 3), axis=1)
        unique, counts = np.unique(faces, axis=0, return_counts=True)
        self.assertLessEqual(counts.max(), 2)
        boundary = unique[counts == 1]
        stage = Usd.Stage.Open(str(ASSET))
        baked = np.asarray(stage.GetDefaultPrim().GetAttribute("surfaceFaceVertexIndices").Get())
        np.testing.assert_array_equal(np.unique(np.sort(baked, axis=1), axis=0), boundary)
        # Closed two-manifold: every boundary edge has exactly two faces.
        edges = np.sort(boundary[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
        _, edge_counts = np.unique(edges, axis=0, return_counts=True)
        self.assertTrue(np.all(edge_counts == 2))
        # Baked outward winding: the divergence theorem gives the tetrahedral volume.
        a, b, c = (mesh.vertices[baked[:, i]] for i in range(3))
        enclosed = np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0
        self.assertAlmostEqual(enclosed / volumes.sum(), 1.0, places=12)


class TestDuckScene(unittest.TestCase):
    def test_reference_parameters(self):
        rest = load_mesh(ASSET)[0].vertices
        scene = duck.duck_scene(rest)
        # B1: corotated weight from the arc friction coefficient mu = 1e4.
        self.assertEqual(scene["corotated_stiffness"], 2.0e4)
        self.assertAlmostEqual(scene["volume_stiffness"], 4.0e7 * 0.45 / (1.45 * 0.1), delta=1.0e-6)
        (arc_kind, arc), (plane_kind, plane) = scene["state_forces"]
        self.assertEqual((arc_kind, plane_kind), ("arc_contact", "planar_contact"))
        # B2: start vector (cos 60 deg, 0, -sin 100 deg).
        np.testing.assert_allclose(arc[6:9], [0.5, 0.0, -np.sin(np.deg2rad(100.0))], rtol=1.0e-15)
        np.testing.assert_allclose(arc[9:], [7.5, np.deg2rad(40.0), 3, 1.0e5, 0.025, 1.0e4], rtol=1.0e-15)
        self.assertEqual(plane, [0.0, 0.0, 1.0, 0.0, 3, 1.0e3, 0.1, 1.0e4])
        np.testing.assert_allclose(np.linalg.norm(scene["v0"], axis=1), 10.0, rtol=1.0e-15)

    def test_first_steps_are_ballistic(self):
        steps = 3
        with wp.ScopedDevice(DEVICE):
            example = duck.Example(newton.viewer.ViewerNull(num_frames=steps), None)
            q0 = example.states[0].particle_q.numpy().astype(np.float64)
            v0 = example.states[0].particle_qd.numpy().astype(np.float64)
            for _ in range(steps):
                example.step()
            example.solver.check_status()
            q = example.solver.positions.numpy()
        self.assertTrue(np.isfinite(q).all())
        # No contact yet: every vertex is inside the arc radius and above the ground.
        for positions in (q0, q):
            radial = positions[:, [0, 2]] - duck.CENTER[[0, 2]]
            self.assertTrue(np.all(np.linalg.norm(radial, axis=1) < duck.RADIUS))
            self.assertTrue(np.all(positions[:, 2] > 0.0))
        # Uniform masses and internal PD forces: the center follows implicit Euler
        # free flight, v_k = v0 + k h g. The converged gradient leaves at most
        # rel_tol |rhs| / sqrt(n) <= rel_tol max |rhs| per step in the center, and
        # the float32 state rounds each step by at most eps32 max |q|.
        h, g = duck.DT, np.array(duck.STATE_FORCE_PARAMETERS[:3])
        expected = q0.mean(axis=0) + h * (steps * v0.mean(axis=0) + h * g * steps * (steps + 1) / 2)
        tolerance = steps * (duck.PD_OPTIONS["rel_tol"] + np.finfo(np.float32).eps) * np.abs(q).max()
        np.testing.assert_allclose(q.mean(axis=0), expected, rtol=0.0, atol=tolerance)
        self.assertGreater(np.abs(q - q0).max(), 0.1)


if __name__ == "__main__":
    unittest.main()
