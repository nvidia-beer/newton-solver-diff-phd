# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frictional boundary contact of the reference DiffPhD step against the dense oracle."""

import unittest

import numpy as np
import warp as wp

from newton_solver_diff_phd import SolverDiffPhD
from newton_solver_diff_phd._src.solvers.diff_phd.exceptions import ConvergenceError
from tests.reference_oracle import ReferenceContact
from tests.test_diff_phd_solver import DENSITY, DEVICE, _grid, _oracle

TIGHT = {"max_iterations": 2000, "abs_tol": 1.0e-12, "rel_tol": 1.0e-11}
# Tolerances of the reference bunny example (python/example/bunny.py).
REFERENCE = {"max_iterations": 2000, "abs_tol": 1.0e-9, "rel_tol": 1.0e-6}


def _surface(model):
    tets = model.tet_indices.numpy().reshape(-1, 4)
    faces = np.sort(
        np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]]), axis=1
    )
    unique, counts = np.unique(faces, axis=0, return_counts=True)
    return np.unique(unique[counts == 1])


def _lifted(model, height, velocity):
    state = model.state()
    q = model.particle_q.numpy()
    q[:, 2] += height - q[:, 2].min()
    state.particle_q.assign(q.astype(np.float32))
    state.particle_qd.assign(np.tile(np.asarray(velocity, dtype=np.float32), (model.particle_count, 1)))
    return state


class TestDiffPhDContact(unittest.TestCase):
    def _rollout(
        self,
        boundary,
        *,
        friction=0.0,
        restitution=0.0,
        height=0.001,
        velocity=(0.0, 0.0, -1.0),
        steps=3,
        options=TIGHT,
    ):
        model = _grid(fix_left=False)
        nodes = _surface(model)
        solver = SolverDiffPhD(
            model,
            density=DENSITY,
            frictional_boundary=boundary,
            frictional_vertices=nodes,
            friction=friction,
            restitution=restitution,
            **options,
        )
        oracle = _oracle(model, "pd")
        contact = ReferenceContact(*boundary, nodes, friction=friction, restitution=restitution)
        gravity = model.gravity.numpy()[0].astype(np.float64)
        dt = 0.004
        states = [_lifted(model, height, velocity)] + [model.state() for _ in range(steps)]
        for step in range(steps):
            solver.step(states[step], states[step + 1], None, None, dt)
            solver.check_status()
            expected, _iterations, converged = oracle.step(
                states[step].particle_q.numpy(),
                states[step].particle_qd.numpy(),
                states[step].particle_f.numpy(),
                gravity,
                dt,
                contact=contact,
                **options,
            )
            self.assertTrue(converged)
            actual = solver._x.numpy()
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-9)
            np.testing.assert_allclose(solver.contact_multipliers.numpy(), contact.multiplier, rtol=1.0e-6, atol=1.0e-9)
        return solver, contact, actual

    def test_frictionless_plane_impact_matches_reference(self):
        solver, contact, x = self._rollout(("planar", (0.0, 0.0, 1.0, 0.0)))
        self.assertGreater(contact.multiplier.max(), 0.0)
        # Converged Fischer-Burmeister complementarity: lambda >= 0, gap ~ 0 where active.
        nodes = _surface(solver.model)
        gaps = x[nodes, 2]
        active = contact.multiplier > 0
        self.assertTrue(np.all(contact.multiplier >= 0))
        np.testing.assert_allclose(gaps[active], 0.0, atol=1.0e-9)
        self.assertGreater(gaps.min(), -1.0e-9)

    def test_sticking_and_sliding_friction_match_reference(self):
        for name, friction, tangential in (("stick", 2.0, 0.05), ("slide", 0.1, 2.0)):
            with self.subTest(name):
                self._rollout(("planar", (0.0, 0.0, 1.0, 0.0)), friction=friction, velocity=(tangential, 0.0, -1.0))

    def test_spherical_obstacle_matches_reference(self):
        model = _grid(fix_left=False)
        center = model.particle_q.numpy().mean(axis=0)
        self._rollout(
            ("spherical", (center[0], center[1], -0.25, 0.25)), friction=0.4, height=0.0005, options=REFERENCE
        )

    def test_friction_chatter_without_convergence_matches_reference(self):
        # At 1e-12/1e-11 the reference stick/slip projection keeps switching on
        # this sphere and never meets the gate; the port must report the same.
        model = _grid(fix_left=False)
        center = model.particle_q.numpy().mean(axis=0)
        boundary = ("spherical", (center[0], center[1], -0.25, 0.25))
        nodes = _surface(model)
        solver = SolverDiffPhD(
            model, density=DENSITY, frictional_boundary=boundary, frictional_vertices=nodes, friction=0.4, **TIGHT
        )
        contact = ReferenceContact(*boundary, nodes, friction=0.4)
        oracle = _oracle(model, "pd")
        gravity = model.gravity.numpy()[0].astype(np.float64)
        states = [_lifted(model, 0.0005, (0.0, 0.0, -1.0)), model.state(), model.state()]
        for step in range(2):
            solver.step(states[step], states[step + 1], None, None, 0.004)
            _x, iterations, converged = oracle.step(
                states[step].particle_q.numpy(),
                states[step].particle_qd.numpy(),
                states[step].particle_f.numpy(),
                gravity,
                0.004,
                contact=contact,
                **TIGHT,
            )
            self.assertEqual(int(solver.iterations.numpy()[0]), iterations)
        self.assertFalse(converged)
        with self.assertRaises(ConvergenceError):
            solver.check_status()

    def test_restitution_matches_reference(self):
        self._rollout(("planar", (0.0, 0.0, 1.0, 0.0)), restitution=0.5, steps=2)

    @unittest.skipUnless(DEVICE.startswith("cuda"), "CUDA graphs need a CUDA device")
    def test_moving_sphere_in_captured_graph_matches_direct_steps(self):
        if not wp.is_conditional_graph_supported():
            self.skipTest("conditional graph nodes are unavailable")
        model = _grid(fix_left=False)
        center = model.particle_q.numpy().mean(axis=0)
        boundary = ("spherical", (center[0], center[1], -0.25, 0.25))
        direct = SolverDiffPhD(model, density=DENSITY, frictional_boundary=boundary, friction=0.4)
        captured = SolverDiffPhD(model, density=DENSITY, frictional_boundary=boundary, friction=0.4)
        state_in = _lifted(model, 0.0005, (0.0, 0.0, -1.0))
        direct_out, captured_out = model.state(), model.state()
        captured.step(state_in, captured_out, None, None, 0.004)
        with wp.ScopedCapture(device=DEVICE) as capture:
            captured.step(state_in, captured_out, None, None, 0.004)
        for offset in (0.0, 0.01, 0.02):
            parameters = np.array([[center[0] + offset, center[1], -0.25 + offset, 0.25]])
            direct.boundary_parameters.assign(parameters)
            captured.boundary_parameters.assign(parameters)
            # Same multiplier warm start for both solvers.
            wp.copy(captured.contact_multipliers, direct.contact_multipliers)
            direct.step(state_in, direct_out, None, None, 0.004)
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(captured_out.particle_q.numpy(), direct_out.particle_q.numpy())
        direct.check_status()
        captured.check_status()


if __name__ == "__main__":
    unittest.main()
