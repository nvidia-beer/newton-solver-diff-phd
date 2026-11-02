# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Original bunny geometry, launch convention and untaped free-flight prefix."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import warp as wp

from newton_solver_diff_phd.examples import _diff_phd_bunny as bunny

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")


class TestBunnyLaunch(unittest.TestCase):
    def test_original_seed_and_rpy_transform(self):
        np.testing.assert_allclose(
            bunny.initial_parameters(),
            [
                -0.11160852421790401,
                -0.2563125111363851,
                -0.6079770402736474,
                0.008572989031895482,
                -0.010529613734250026,
                0.20990413790754708,
                1.7182939842217384,
                1.3859965163254362,
                -7.576811392968015,
            ],
            rtol=0.0,
            atol=1.0e-15,
        )
        rest = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.4]])
        original = rest.copy()
        parameters = np.array([np.pi / 2, np.pi / 2, np.pi / 2, 0.01, -0.02, 0.2, 1.0, 2.0, -6.0])
        q, v = bunny.initial_state(rest, parameters)
        # These three right-angle rotations in Rz Ry Rx order map (x,y,z)
        # to (z,y,-x), independently of the implementation's matrix formula.
        centered = rest - rest.mean(axis=0)
        expected = centered[:, [2, 1, 0]] * [1, 1, -1] + parameters[3:6]
        np.testing.assert_allclose(q, expected, rtol=0.0, atol=1.0e-16)
        np.testing.assert_allclose(q.mean(axis=0), parameters[3:6], rtol=0.0, atol=1.0e-16)
        np.testing.assert_array_equal(v, np.tile(parameters[6:], (len(rest), 1)))
        np.testing.assert_array_equal(rest, original)
        for invalid in (np.zeros(8), np.zeros((3, 3)), np.full(9, np.nan), np.full(9, np.inf)):
            with self.subTest(parameters=invalid), self.assertRaisesRegex(ValueError, "nine finite"):
                bunny.initial_state(rest, invalid)

    def test_objective_uses_vertex_mean_and_original_size(self):
        points = np.array([[0.01, 0.02, 0.03], [0.07, 0.11, 0.19], [0.2, -0.1, 0.06]])
        difference = points.sum(axis=0) / len(points) - [0.15, 0.15, 0.15]
        expected = 50.0 * float(difference @ difference)
        self.assertAlmostEqual(bunny.landing_loss(points), expected, places=14)
        np.testing.assert_array_equal(bunny.TARGET, [0.15, 0.15, 0.15])
        self.assertEqual((bunny.DT, bunny.FRAMES), (0.001, 100))

    def test_missing_asset_explains_offline_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "prepare_bunny_reference.py"):
                bunny.load_bunny(Path(directory) / "missing.usdc")


@unittest.skipUnless(bunny.ASSET.is_file(), "Original bunny asset must be prepared locally")
class TestBunnyAsset(unittest.TestCase):
    def test_original_tetrahedra_boundary_and_material_regions(self):
        mesh, faces, owners, contact, stiff, soft = bunny.load_bunny()
        self.assertEqual(mesh.vertices.shape, (1672, 3))
        self.assertEqual(mesh.indices.shape, (5950, 4))
        self.assertEqual(faces.shape, (2656, 3))
        self.assertEqual((len(contact), len(stiff), len(soft)), (66, 2940, 3010))
        corners = mesh.vertices[mesh.indices]
        volumes = np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6.0
        self.assertTrue(np.all(volumes > 0.0))
        self.assertAlmostEqual(float(volumes.sum()), 0.00019857071212849953, places=15)
        self.assertEqual(len(np.unique(mesh.indices)), len(mesh.vertices))
        self.assertAlmostEqual(float(np.ptp(mesh.vertices, axis=0).max()), 0.1, places=8)

        all_faces = np.sort(mesh.indices[:, [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]].reshape(-1, 3), axis=1)
        unique, counts = np.unique(all_faces, axis=0, return_counts=True)
        self.assertLessEqual(int(counts.max()), 2)
        boundary = unique[counts == 1]
        np.testing.assert_array_equal(np.unique(np.sort(faces, axis=1), axis=0), boundary)
        self.assertTrue(np.all(np.isin(contact, boundary)))
        self.assertTrue(np.all(np.any(faces[:, :, None] == mesh.indices[owners, None, :], axis=2)))
        edges = np.sort(boundary[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
        _, edge_counts = np.unique(edges, axis=0, return_counts=True)
        self.assertTrue(np.all(edge_counts == 2))
        a, b, c = (mesh.vertices[faces[:, i]] for i in range(3))
        enclosed = np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0
        self.assertAlmostEqual(float(enclosed / volumes.sum()), 1.0, places=12)

        np.testing.assert_array_equal(np.sort(np.concatenate([stiff, soft])), np.arange(len(corners)))
        expected_soft = np.flatnonzero(corners[:, :, 1].mean(axis=1) < mesh.vertices[:, 1].mean())
        np.testing.assert_array_equal(np.sort(soft), expected_soft)
        np.testing.assert_array_equal(np.sort(stiff), np.setdiff1d(np.arange(len(corners)), expected_soft))

    def test_ballistic_prefix_and_reset_without_tape(self):
        steps = 3
        with wp.ScopedDevice(DEVICE), mock.patch("warp.Tape", side_effect=AssertionError("Forward only")):
            with mock.patch.object(bunny, "SolverDiffPhD", wraps=bunny.SolverDiffPhD) as constructor:
                simulation = bunny.BunnySimulation(device=DEVICE, capture=False)
            self.assertFalse(simulation.model.requires_grad)
            self.assertEqual(simulation.frame, 0)
            self.assertTrue(np.all(simulation.model.particle_mass.numpy() > 0.0))
            self.assertEqual(simulation.solver.material, "pd")
            self.assertEqual(simulation.solver.density, 1000.0)
            self.assertEqual(constructor.call_args.kwargs["friction"], 0.1)
            self.assertEqual(constructor.call_args.kwargs["restitution"], 0.0)
            self.assertEqual(constructor.call_args.kwargs["frictional_boundary"], ("planar", (0.0, 0.0, 1.0, 0.0)))
            self.assertEqual(simulation.solver.max_iterations, 10000)
            self.assertEqual(simulation.solver.rel_tol, 1.0e-6)
            mu, lam = simulation.model.tet_materials.numpy()[:, :2].T.astype(np.float64)
            young = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
            np.testing.assert_allclose(young, np.where(simulation.materials == 1, 1.0e7, 1.0e5), rtol=1.0e-7)
            np.testing.assert_allclose(lam / (2.0 * (lam + mu)), 0.49, rtol=1.0e-7)
            np.testing.assert_allclose(simulation.model.gravity.numpy().reshape(-1, 3)[0], [0, 0, -9.81], rtol=1.0e-7)
            q0 = simulation.states[0].particle_q.numpy().astype(np.float64)
            v0 = simulation.states[0].particle_qd.numpy().astype(np.float64)
            self.assertGreater(float(q0[:, 2].min()), 0.0)
            trajectory = []
            inverse = None
            for step in range(steps):
                state = simulation.advance()
                self.assertIs(state, simulation.states[0])
                self.assertEqual(simulation.frame, step + 1)
                trajectory.append((state.particle_q.numpy().copy(), state.particle_qd.numpy().copy()))
                if inverse is None:
                    inverse = simulation.solver.inverse
                self.assertIs(simulation.solver.inverse, inverse)
            q = simulation.states[0].particle_q.numpy().astype(np.float64)
            self.assertTrue(np.isfinite(q).all())
            self.assertGreater(float(q[:, 2].min()), 0.0)
            np.testing.assert_array_equal(simulation.solver.contact_multipliers.numpy(), 0.0)
            h, gravity = bunny.DT, np.array([0.0, 0.0, -9.81])
            expected = q0.mean(axis=0) + h * (steps * v0.mean(axis=0) + h * gravity * steps * (steps + 1) / 2)
            # Uniform solver mass and cancelling internal forces conserve linear
            # momentum. Allow its declared convergence tolerance and state rounding.
            tolerance = steps * (simulation.solver.rel_tol + np.finfo(np.float32).eps) * np.abs(q).max()
            np.testing.assert_allclose(q.mean(axis=0), expected, rtol=0.0, atol=tolerance)
            self.assertGreater(float(np.abs(q - q0).max()), 0.02)

            simulation.solver.contact_multipliers.fill_(1.0)
            simulation.reset()
            self.assertEqual(simulation.frame, 0)
            np.testing.assert_array_equal(simulation.solver.contact_multipliers.numpy(), 0.0)
            np.testing.assert_array_equal(simulation.states[0].particle_q.numpy(), q0.astype(np.float32))
            np.testing.assert_array_equal(simulation.states[0].particle_qd.numpy(), v0.astype(np.float32))
            for positions, velocities in trajectory:
                state = simulation.advance()
                np.testing.assert_array_equal(state.particle_q.numpy(), positions)
                np.testing.assert_allclose(state.particle_qd.numpy(), velocities, rtol=0.0, atol=1.0e-8)
                self.assertIs(simulation.solver.inverse, inverse)


if __name__ == "__main__":
    unittest.main()
