# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Forward checks of the DiffPD ``pd_eigen`` step against an independent dense oracle."""

import os
import unittest

import newton
import numpy as np
import warp as wp

from newton_solver_diff_phd import SolverDiffPD
from newton_solver_diff_phd._src.solvers.diff_pd.exceptions import ConvergenceError
from tests.reference_oracle_diff_pd import ReferenceStep

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
DENSITY = 1000.0
# Weights as the reference examples pass them: corotated 2 mu, volume lambda.
COROTATED = 4.0e4
VOLUME = 5.0e4
GRAVITY = (0.0, 0.0, -9.81)
DT = 1.0e-2
# Both sides start every step from the same float32 state and follow the same
# iteration path, so they agree to rounding; iteration counts must be equal.
POSITION_TOL = 1.0e-12
VELOCITY_TOL = POSITION_TOL / DT


def _grid(device=DEVICE, *, fix_left=False, pos=(0.0, 0.0, 1.0)):
    builder = newton.ModelBuilder(gravity=GRAVITY)
    builder.add_soft_grid(
        pos=wp.vec3(*pos),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=3,
        dim_y=2,
        dim_z=2,
        cell_x=0.1,
        cell_y=0.1,
        cell_z=0.1,
        density=DENSITY,
        k_mu=2.0e4,
        k_lambda=5.0e4,
        k_damp=0.0,
        fix_left=fix_left,
    )
    return builder.finalize(device=device)


def _pair(model, state_forces=(), gravity=GRAVITY, **options):
    solver = SolverDiffPD(
        model,
        density=DENSITY,
        corotated_stiffness=COROTATED,
        volume_stiffness=VOLUME,
        state_forces=state_forces,
        gravity=gravity,
        **options,
    )
    oracle = ReferenceStep(
        model.particle_q.numpy(),
        model.tet_indices.numpy().reshape(-1, 4),
        DENSITY,
        COROTATED,
        VOLUME,
        model.particle_inv_mass.numpy() > 0,
        GRAVITY if gravity is None else gravity,
        state_forces,
        **options,
    )
    return solver, oracle


def _perturb(model, velocity=(0.0, 0.0, 0.0), noise=0.003, speed=0.1, stretch=(1.0, 1.0, 1.0), seed=0):
    rng = np.random.default_rng(seed)
    state = model.state()
    q = model.particle_q.numpy().astype(np.float64)
    free = model.particle_inv_mass.numpy() > 0
    center = q.mean(axis=0)
    deformed = center + (q - center) * np.asarray(stretch) + noise * rng.standard_normal(q.shape)
    v = np.asarray(velocity) + speed * rng.standard_normal(q.shape)
    state.particle_q.assign(np.where(free[:, None], deformed, q).astype(np.float32))
    state.particle_qd.assign(np.where(free[:, None], v, 0.0).astype(np.float32))
    return state


class TestDiffPDSolver(unittest.TestCase):
    def _compare(self, model, solver, oracle, state, steps, dt=DT):
        out = model.state()
        halvings = fallbacks = 0
        statuses = 0
        for _ in range(steps):
            q = state.particle_q.numpy().astype(np.float64)
            v = state.particle_qd.numpy().astype(np.float64)
            f = state.particle_f.numpy().astype(np.float64)
            solver.step(state, out, None, None, dt)
            expected_q, expected_v, iterations = oracle.step(q, v, f, dt)
            halvings += oracle.halvings
            fallbacks += oracle.fallbacks
            # Bit 4 (all line-search halvings used) is informational, as in the reference.
            status = solver.read_status()
            self.assertEqual(status & ~4, 0)
            statuses |= status
            self.assertEqual(int(solver.iterations.numpy()[0]), iterations)
            np.testing.assert_allclose(solver.positions.numpy(), expected_q, rtol=0.0, atol=POSITION_TOL)
            np.testing.assert_allclose(solver.velocities.numpy(), expected_v, rtol=0.0, atol=VELOCITY_TOL)
            np.testing.assert_array_equal(out.particle_q.numpy(), expected_q.astype(np.float32))
            state, out = out, state
        self.statuses = statuses
        return state, halvings, fallbacks

    def test_free_fall_of_a_deformed_block(self):
        model = _grid()
        solver, oracle = _pair(model)
        state = _perturb(model, stretch=(1.3, 0.8, 1.1), speed=0.5)
        height = state.particle_q.numpy()[:, 2].mean()
        state, _, _ = self._compare(model, solver, oracle, state, 4)
        self.assertEqual(self.statuses, 0)
        self.assertLess(state.particle_q.numpy()[:, 2].mean(), height)

    def test_pinned_cantilever_keeps_dirichlet_particles(self):
        model = _grid(fix_left=True)
        solver, oracle = _pair(model)
        pinned = model.particle_inv_mass.numpy() == 0
        self.assertTrue(np.any(pinned))
        start = model.particle_q.numpy()
        state, _, _ = self._compare(model, solver, oracle, model.state(), 6)
        q = state.particle_q.numpy()
        np.testing.assert_array_equal(q[pinned], start[pinned])
        np.testing.assert_array_equal(state.particle_qd.numpy()[pinned], 0.0)
        self.assertLess(q[~pinned, 2].min(), start[~pinned, 2].min())

    def test_block_on_planar_contact(self):
        # The bottom layer starts 5 mm inside the plane z = 0 while sliding; both
        # friction branches (kf viscous and mu fn Coulomb) occur.
        forces = [("planar_contact", [0.0, 0.0, 2.0, 0.0, 3, 1.0e6, 1.0, 0.1])]
        model = _grid(pos=(-0.15, -0.1, -0.005))
        solver, oracle = _pair(model, forces)
        state = _perturb(model, velocity=(1.0, 0.0, -0.5))
        state, _, _ = self._compare(model, solver, oracle, state, 5)
        self.assertGreater(state.particle_q.numpy()[:, 2].min(), -0.005)

    def test_block_on_arc_contact(self):
        # Circle of radius 1 about y through (0, 0, 1); the window starts 30 degrees
        # right of straight down and ends 2 degrees left of it, inside the block.
        start = (np.sin(np.radians(30.0)), 0.0, -np.cos(np.radians(30.0)))
        forces = [("arc_contact", [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, *start, 1.0, np.radians(32.0), 3, 1.0e6, 1.0, 0.1])]
        model = _grid(pos=(-0.15, -0.1, -0.005))
        solver, oracle = _pair(model, forces)
        state = _perturb(model, velocity=(1.0, 0.0, -0.5))
        q = state.particle_q.numpy().astype(np.float64)
        theta = np.degrees(np.arctan2(q[:, 0], -(q[:, 2] - 1.0)))
        self.assertTrue(np.any(theta < -2.0) and np.any(theta > -2.0))
        self._compare(model, solver, oracle, state, 5)

    def test_history_wraps_and_line_search_halves(self):
        # A large step from a rough state needs more than bfgs_history_size
        # iterations, so the ring buffer wraps, and the Armijo test halves.
        model = _grid()
        solver, oracle = _pair(model)
        state = _perturb(model, noise=0.03, speed=5.0, seed=3)
        _, halvings, _ = self._compare(model, solver, oracle, state, 1, dt=0.2)
        self.assertEqual(self.statuses, 0)
        self.assertGreater(halvings, 0)
        self.assertGreater(int(solver.iterations.numpy()[0]), 2 * solver.bfgs_history_size)

    def test_model_gravity_is_used_without_override(self):
        model = _grid()
        solver, oracle = _pair(model, gravity=None)
        oracle.gravity = model.gravity.numpy()[0].astype(np.float64)
        self._compare(model, solver, oracle, _perturb(model), 2)

    def test_external_forces_enter_the_right_hand_side(self):
        model = _grid()
        solver, oracle = _pair(model)
        state = _perturb(model)
        rng = np.random.default_rng(5)
        state.particle_f.assign(rng.standard_normal((model.particle_count, 3)).astype(np.float32))
        self._compare(model, solver, oracle, state, 1)

    def test_max_pd_iter_raises_convergence_error(self):
        model = _grid()
        solver, _ = _pair(model, max_pd_iter=1)
        solver.step(_perturb(model, stretch=(1.3, 0.8, 1.1)), model.state(), None, None, DT)
        with self.assertRaises(ConvergenceError):
            solver.check_status()
        solver.check_status()

    def test_singular_deformation_gradient_raises(self):
        model = _grid()
        solver, _ = _pair(model)
        state = model.state()
        q = model.particle_q.numpy()
        q[:, 2] = 1.0
        state.particle_q.assign(q)
        solver.step(state, model.state(), None, None, DT)
        self.assertEqual(int(solver.iterations.numpy()[0]), 0)
        with self.assertRaisesRegex(ValueError, "Singular F"):
            solver.check_status()

    def test_tape_is_not_supported(self):
        model = _grid()
        solver, _ = _pair(model)
        with wp.Tape(), self.assertRaises(NotImplementedError):
            solver.step(model.state(), model.state(), None, None, DT)

    @unittest.skipUnless(wp.get_device(DEVICE).is_cuda and wp.is_conditional_graph_supported(), "needs CUDA graphs")
    def test_step_captured_in_a_caller_graph_matches(self):
        forces = [("planar_contact", [0.0, 0.0, 1.0, 0.0, 3, 1.0e6, 1.0, 0.1])]
        model = _grid(pos=(-0.15, -0.1, -0.005))
        reference, _ = _pair(model, forces)
        captured, _ = _pair(model, forces)
        state = _perturb(model, velocity=(1.0, 0.0, -0.5))
        expected, out = model.state(), model.state()
        reference.step(state, expected, None, None, DT)
        captured.step(state, out, None, None, DT)
        with wp.ScopedCapture(device=DEVICE) as capture:
            captured.step(state, out, None, None, DT)
        out.particle_q.zero_()
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(captured.positions.numpy(), reference.positions.numpy())
        np.testing.assert_array_equal(out.particle_q.numpy(), expected.particle_q.numpy())

    def test_invalid_configurations_are_rejected(self):
        model = _grid()
        invalid = [
            {"density": 0.0},
            {"density": np.nan},
            {"corotated_stiffness": -1.0},
            {"volume_stiffness": np.inf},
            {"max_pd_iter": 0},
            {"max_ls_iter": 0},
            {"max_ls_iter": 2.5},
            {"bfgs_history_size": 1},
            {"abs_tol": -1.0},
            {"rel_tol": np.nan},
            {"gravity": (0.0, 1.0)},
            {"rest_positions": np.zeros((2, 3))},
            {"state_forces": [("hydrodynamics", [0.0] * 8)]},
            {"state_forces": [("planar_contact", [0.0] * 7)]},
            {"state_forces": [("planar_contact", [0.0, 0.0, 1.0e-6, 0.0, 3, 1.0, 1.0, 1.0])]},
            {"state_forces": [("arc_contact", [0.0] * 15)]},
            {"state_forces": [("planar_contact",)]},
        ]
        for options in invalid:
            arguments = {"density": DENSITY, **options}
            with self.subTest(options=options), self.assertRaises(ValueError):
                SolverDiffPD(model, **arguments)
        solver = SolverDiffPD(model, density=DENSITY, corotated_stiffness=COROTATED)
        state = model.state()
        for dt in (0.0, -1.0, np.nan):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                solver.step(state, model.state(), None, None, dt)
        with self.assertRaises(ValueError):
            solver.step(state, state, None, None, DT)
        with self.assertRaises(NotImplementedError):
            solver.step(state, model.state(), None, newton.Contacts(0, 0, device=DEVICE), DT)


if __name__ == "__main__":
    unittest.main()
