# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Original Hex8 geometry and complete axial/twist cantilever trajectories."""

import os
import unittest
from unittest import mock

import numpy as np
import warp as wp

from newton_solver_diff_phd.examples import _diff_phd_cantilever as cantilever

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")


class TestCantilever(unittest.TestCase):
    def test_baked_grid_is_closed_positive_and_preserves_original_cell_regions(self):
        mesh, faces, owners, root, cells = cantilever.load_cantilever()
        self.assertEqual(mesh.vertices.shape, (756, 3))
        self.assertEqual(mesh.indices.shape, (500, 8))
        self.assertEqual(faces.shape, (900, 3))
        self.assertEqual(len(root), 36)
        np.testing.assert_array_equal(mesh.vertices.min(axis=0), 0.0)
        np.testing.assert_allclose(mesh.vertices.max(axis=0), [0.32, 0.08, 0.08], rtol=0.0, atol=1.0e-16)
        np.testing.assert_array_equal(root, np.flatnonzero(mesh.vertices[:, 0] == 0.0))
        corners = mesh.vertices[mesh.indices]
        volumes = np.linalg.det(corners[:, [4, 2, 1]] - corners[:, :1])
        np.testing.assert_allclose(volumes, 0.016**3, rtol=1.0e-14, atol=0.0)
        self.assertAlmostEqual(float(volumes.sum()), 0.32 * 0.08 * 0.08, places=16)
        self.assertEqual(len(np.unique(mesh.indices)), len(mesh.vertices))

        # Original local order is 000,001,010,011,100,101,110,111.
        offsets = np.array([(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]) * 0.016
        np.testing.assert_allclose(corners - corners[:, :1], np.broadcast_to(offsets, corners.shape), atol=5e-17)
        self.assertTrue(np.all(np.any(faces[:, :, None] == mesh.indices[owners, None, :], axis=2)))
        edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
        _, edge_counts = np.unique(edges, axis=0, return_counts=True)
        np.testing.assert_array_equal(edge_counts, 2)
        a, b, c = (mesh.vertices[faces[:, i]] for i in range(3))
        enclosed = np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0
        self.assertAlmostEqual(float(enclosed / volumes.sum()), 1.0, places=13)

        np.testing.assert_array_equal(np.bincount(cells), np.full(20, 25))
        np.testing.assert_allclose(corners[:, :, 0].min(axis=1), cells * 0.016, rtol=0.0, atol=1.0e-16)
        np.testing.assert_allclose(corners[:, :, 0].max(axis=1), (cells + 1) * 0.016, rtol=0.0, atol=1.0e-16)
        self.assertEqual(np.count_nonzero((cells >= 7) & (cells <= 12)), 150)

    def test_initial_twist_preserves_root_radius_and_prescribed_angle(self):
        mesh, _faces, _owners, root, _cells = cantilever.load_cantilever()
        rest = mesh.vertices
        original = rest.copy()
        q = cantilever.twisted_positions(rest)
        np.testing.assert_array_equal(q[root], rest[root])
        np.testing.assert_array_equal(q[:, 0], rest[:, 0])
        offset = rest[:, 1:] - [0.04, 0.04]
        moved = q[:, 1:] - [0.04, 0.04]
        np.testing.assert_allclose(np.linalg.norm(moved, axis=1), np.linalg.norm(offset, axis=1), atol=2.0e-17)
        angles = np.arctan2(offset[:, 0] * moved[:, 1] - offset[:, 1] * moved[:, 0], np.sum(offset * moved, axis=1))
        np.testing.assert_allclose(angles, rest[:, 0] / 0.32 * np.pi / 6, rtol=0.0, atol=2.0e-15)
        np.testing.assert_allclose(cantilever.twisted_positions(rest, angle=0.0), rest, rtol=0.0, atol=7.0e-18)
        np.testing.assert_array_equal(rest, original)
        for angle in (np.nan, np.inf, -np.inf):
            with self.subTest(angle=angle), self.assertRaisesRegex(ValueError, "finite"):
                cantilever.twisted_positions(rest, angle)

    def test_complete_original_hex_problem_and_twist_keep_clamp_and_positive_jacobians(self):
        for motion in ("axial", "twist"):
            with (
                self.subTest(motion=motion),
                wp.ScopedDevice(DEVICE),
                mock.patch("warp.Tape", side_effect=AssertionError("Forward simulation only")),
            ):
                simulation = cantilever.CantileverSimulation(device=DEVICE, motion=motion)
                self.assertFalse(simulation.model.requires_grad)
                self.assertEqual(simulation.model.tet_count, 0)
                self.assertIsNone(simulation.solver.contact_multipliers)
                self.assertEqual((cantilever.DT, cantilever.FRAMES), (0.01, 25))
                np.testing.assert_array_equal(simulation.model.gravity.numpy(), 0.0)
                np.testing.assert_array_equal(
                    np.flatnonzero(simulation.model.particle_mass.numpy() == 0), simulation.pinned
                )
                mu, lam = simulation.solver.hex_materials.numpy()[:, :2].T
                young = mu * (3 * lam + 2 * mu) / (lam + mu)
                np.testing.assert_allclose(young, np.where(simulation.materials == 1, 1.0e7, 1.0e6), rtol=1.0e-14)
                np.testing.assert_allclose(lam / (2 * (lam + mu)), 0.45, rtol=1.0e-14)
                self.assertEqual(np.count_nonzero(simulation.materials), 150)
                self.assertAlmostEqual(simulation.solver.particle_mass, 0.004096, places=14)
                np.testing.assert_array_equal(simulation.solver.muscle_targets.numpy(), 0.0)
                q0 = simulation.states[0].particle_q.numpy().copy()
                initial_buffers = {state.particle_q.ptr for state in simulation.states}
                inverse, prefix = None, []
                min_jacobian, max_motion = 1.0, 0.0
                minimum_twist = 30.0
                tip_rest = simulation.vertices[simulation.tip, 1:] - [0.04, 0.04]
                for frame in range(simulation.frames):
                    state = simulation.advance()
                    self.assertIs(state, simulation.states[0])
                    self.assertEqual(simulation.frame, frame + 1)
                    q, velocity = state.particle_q.numpy(), state.particle_qd.numpy()
                    self.assertTrue(np.isfinite(q).all() and np.isfinite(velocity).all())
                    np.testing.assert_array_equal(q[simulation.pinned], q0[simulation.pinned])
                    np.testing.assert_array_equal(velocity[simulation.pinned], 0.0)
                    determinants = simulation.solver.deformation_determinants(state).numpy()
                    self.assertEqual(determinants.shape, (500, 8))
                    min_jacobian = min(min_jacobian, float(determinants.min()))
                    max_motion = max(max_motion, float(np.linalg.norm(q - q0, axis=1).max()))
                    tip = q[simulation.tip, 1:].astype(float)
                    tip -= tip.mean(axis=0)
                    angle = np.degrees(
                        np.arctan2(
                            np.sum(tip_rest[:, 0] * tip[:, 1] - tip_rest[:, 1] * tip[:, 0]), np.sum(tip_rest * tip)
                        )
                    )
                    minimum_twist = min(minimum_twist, float(angle))
                    if motion == "axial":
                        np.testing.assert_allclose(simulation.forces[simulation.tip, 0], 80.0 * frame / 25, rtol=1e-7)
                    if frame < 3:
                        prefix.append((q.copy(), velocity.copy()))
                    if inverse is None:
                        inverse = simulation.solver.inverse
                    self.assertIs(simulation.solver.inverse, inverse)
                    self.assertEqual({state.particle_q.ptr for state in simulation.states}, initial_buffers)
                self.assertGreater(min_jacobian, 0.8)
                self.assertGreater(max_motion, 0.001)
                if motion == "axial":
                    self.assertGreater(float(q[simulation.tip, 0].mean() - q0[simulation.tip, 0].mean()), 0.01)
                else:
                    self.assertEqual(simulation.dt, 0.001)
                    self.assertLess(minimum_twist, -20.0)  # A visible rotation, not just muscle-induced translation.
                final = simulation.states[0].particle_q.numpy().copy()
                self.assertIs(simulation.advance(), simulation.states[0])
                np.testing.assert_array_equal(simulation.states[0].particle_q.numpy(), final)
                simulation.reset()
                self.assertEqual(simulation.frame, 0)
                np.testing.assert_array_equal(simulation.states[0].particle_q.numpy(), q0)
                np.testing.assert_array_equal(simulation.states[0].particle_qd.numpy(), 0.0)
                # This compares the first uncaptured step with graph replay after reset.
                for q, velocity in prefix:
                    state = simulation.advance()
                    np.testing.assert_array_equal(state.particle_q.numpy(), q)
                    np.testing.assert_allclose(state.particle_qd.numpy(), velocity, rtol=0.0, atol=1.0e-8)
                    self.assertIs(simulation.solver.inverse, inverse)

    def test_motion_switch_restarts_state_and_replaces_captured_tolerance(self):
        with wp.ScopedDevice(DEVICE):
            simulation = cantilever.CantileverSimulation(device=DEVICE)
            for _ in range(3):
                simulation.advance()
            model = simulation.model
            for motion in ("twist", "axial"):
                with self.subTest(motion=motion):
                    simulation.set_motion(motion)
                    fresh = cantilever.CantileverSimulation(device=DEVICE, motion=motion)
                    self.assertIs(simulation.model, model)  # Viewer callbacks/model remain intact.
                    self.assertEqual(simulation.frame, 0)
                    self.assertEqual(simulation.frames, fresh.frames)
                    self.assertEqual(simulation.dt, fresh.dt)
                    self.assertEqual(simulation.solver.rel_tol, fresh.solver.rel_tol)
                    np.testing.assert_array_equal(simulation.states[0].particle_qd.numpy(), 0.0)
                    np.testing.assert_array_equal(simulation.forces, 0.0)
                    for _ in range(3):
                        simulation.advance()
                        fresh.advance()
                        np.testing.assert_array_equal(
                            simulation.states[0].particle_q.numpy(), fresh.states[0].particle_q.numpy()
                        )
                        np.testing.assert_allclose(
                            simulation.states[0].particle_qd.numpy(), fresh.states[0].particle_qd.numpy(), atol=1.0e-8
                        )


if __name__ == "__main__":
    unittest.main()
