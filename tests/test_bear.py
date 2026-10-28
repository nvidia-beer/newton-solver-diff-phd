# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Leg-controller derivatives and native ground-contact gait rollouts."""

import os
import unittest
from unittest import mock

import numpy as np
import warp as wp

from newton_solver_diff_phd._src.solvers.diff_phd.bear import AMPLITUDE, FREQUENCY, BearProblem, muscle_commands


class TestDiffsimDiffPhDBear(unittest.TestCase):
    def test_invalid_weights_are_rejected_before_model_construction(self):
        with mock.patch(
            "newton_solver_diff_phd._src.solvers.diff_phd.bear.bear_mesh",
            side_effect=AssertionError("Invalid weights must not construct a model"),
        ):
            for parameters in (np.zeros(15), np.zeros((8, 2)), np.full(16, np.nan), np.full(16, np.inf)):
                with self.subTest(shape=parameters.shape), self.assertRaisesRegex(ValueError, "sixteen finite"):
                    BearProblem(training=False, parameters=parameters)

    def test_grouped_controller_matches_analytic_values_and_weight_vjp(self):
        parameters = np.linspace(-0.65, 0.8, 16)
        groups = np.array([-1, 7, 0, 3, -1, 2, 1, 5, 4, 6, 2, 7], dtype=np.int32)
        seeds = np.linspace(-1.0, 0.7, len(groups), dtype=np.float32)
        time = 0.137
        phase = 2.0 * np.pi * FREQUENCY * time
        basis = np.array([np.sin(phase), np.cos(phase)])
        signal = parameters.reshape(8, 2) @ basis
        expected = np.zeros(len(groups), dtype=np.float32)
        expected[groups >= 0] = AMPLITUDE * np.tanh(signal[groups[groups >= 0]])
        expected_gradient = np.zeros((8, 2))
        for group, seed in zip(groups, seeds, strict=True):
            if group >= 0:
                expected_gradient[group] += float(seed) * AMPLITUDE * (1.0 - np.tanh(signal[group]) ** 2) * basis

        for device in [wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))]:
            with self.subTest(device=device), wp.ScopedDevice(device):
                weights = wp.array(parameters, dtype=wp.float64, device=device, requires_grad=True)
                labels = wp.array(groups, dtype=wp.int32, device=device)
                commands = wp.empty(len(groups), dtype=wp.float32, device=device, requires_grad=True)
                tape = wp.Tape()
                with tape:
                    wp.launch(muscle_commands, len(groups), inputs=[weights, labels, time], outputs=[commands])
                tape.backward(grads={commands: wp.array(seeds, dtype=wp.float32, device=device)})
                np.testing.assert_allclose(commands.numpy(), expected, rtol=1e-7, atol=1e-8)
                np.testing.assert_array_equal(commands.numpy()[groups < 0], 0.0)
                np.testing.assert_allclose(weights.grad.numpy(), expected_gradient.ravel(), rtol=2e-7, atol=1e-8)
                np.testing.assert_array_equal(weights.numpy(), parameters)

    def test_contact_rollout_gradient_fixed_materials_and_factor_reuse(self):
        for device in [wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))]:
            with self.subTest(device=device), wp.ScopedDevice(device):
                problem = BearProblem(frames=5, device=device)
                parameters = problem.x.copy()
                loss, gradient, positions = problem.value, problem.gradient.copy(), problem.initial_positions.copy()
                inverse = problem.solver.inverse
                self.assertEqual(positions.shape, (6, 276, 3))
                self.assertGreater(problem.contact_frames, 0)
                self.assertTrue(np.isfinite(positions).all())
                self.assertTrue(np.isfinite(gradient).all())
                self.assertGreater(np.linalg.norm(gradient), 1e-6)
                self.assertGreaterEqual(float(positions[:, :, 2].min()), 0.0001 - 2e-7)
                self.assertTrue(np.all(problem.model.particle_mass.numpy() > 0.0))
                self.assertIsNone(problem.model.tet_materials.grad)
                self.assertTrue(all(state.particle_f.grad is None for state in problem.states))
                self.assertEqual(len({state.particle_q.ptr for state in problem.states}), problem.frames + 1)
                self.assertEqual(len({control.tet_activations.ptr for control in problem.controls}), problem.frames)
                np.testing.assert_array_equal(positions[0], problem.vertices.astype(np.float32))
                np.testing.assert_array_equal(problem.model.tri_materials.numpy(), 0.0)
                np.testing.assert_array_equal(
                    problem.solver.muscle_stiffness.numpy(), np.where(problem.group_indices >= 0, 7.5e4, 0.0)
                )
                mu, lame = problem.model.tet_materials.numpy()[:, :2].T.astype(np.float64)
                young = mu * (3.0 * lame + 2.0 * mu) / (lame + mu)
                np.testing.assert_allclose(young, np.where(problem.leg_ids >= 0, 5e4, 1.5e5), rtol=1e-7)
                np.testing.assert_allclose(lame / (2.0 * (lame + mu)), 0.35, rtol=1e-7)

                velocities = np.stack([state.particle_qd.numpy() for state in problem.states[1:]])
                centers = np.einsum("fni,n->fi", velocities.astype(np.float64), problem.mass_weights)
                penalty = np.sqrt(centers[:, 1:] ** 2 + 0.001**2) - 0.001
                expected_loss = problem.dt / problem.length * np.sum(0.1 * penalty.sum(axis=1) - centers[:, 0])
                expected_loss += 1e-5 * np.mean(parameters**2)
                self.assertAlmostEqual(loss, expected_loss, delta=1e-12)
                for frame, control in enumerate(problem.controls):
                    phase = 2.0 * np.pi * FREQUENCY * frame * problem.dt
                    signal = parameters.reshape(8, 2) @ [np.sin(phase), np.cos(phase)]
                    expected = np.zeros(problem.model.tet_count, dtype=np.float32)
                    selected = problem.group_indices >= 0
                    expected[selected] = AMPLITUDE * np.tanh(signal[problem.group_indices[selected]])
                    np.testing.assert_allclose(control.tet_activations.numpy(), expected, rtol=1e-7, atol=1e-8)

                with (
                    self.subTest(mode="inference"),
                    mock.patch("warp.Tape", side_effect=AssertionError("Inference must not construct a tape")),
                ):
                    provided = parameters.copy()
                    with mock.patch.object(
                        type(problem.solver), "step", side_effect=AssertionError("Inference must not run at startup")
                    ):
                        inference = BearProblem(frames=2, device=device, training=False, parameters=provided)
                    provided[:] = np.nan
                    self.assertFalse(inference.training)
                    self.assertFalse(inference.model.requires_grad)
                    self.assertEqual(inference.elapsed_steps, 0)
                    self.assertEqual(len(inference.states), 2)
                    self.assertEqual(len(inference.controls), 1)
                    for name in ("loss", "centers", "gradient", "positions", "history", "momentum", "variance"):
                        self.assertFalse(hasattr(inference, name))
                    arrays = [inference.parameters, inference.controls[0].tet_activations]
                    arrays.extend(
                        array
                        for state in inference.states
                        for array in (state.particle_q, state.particle_qd, state.particle_f)
                    )
                    for array in arrays:
                        self.assertFalse(array.requires_grad)
                        self.assertIsNone(array.grad)
                    state_buffers = {state.particle_q.ptr for state in inference.states}
                    control_buffer = inference.controls[0].tet_activations.ptr
                    np.testing.assert_array_equal(inference.states[0].particle_q.numpy(), positions[0])
                    inference_inverse = None
                    for frame in range(problem.frames):
                        state = inference.advance()
                        self.assertIs(state, inference.states[0])
                        self.assertEqual(inference.elapsed_steps, frame + 1)
                        np.testing.assert_array_equal(state.particle_q.numpy(), positions[frame + 1])
                        # CUDA reductions produce roughly 1e-12 m/s cancellation noise
                        # in near-zero velocities. This 1e-11 floor is below
                        # 1/5000 of a float32 ULP at this trajectory's speed.
                        np.testing.assert_allclose(state.particle_qd.numpy(), velocities[frame], rtol=0.0, atol=1e-11)
                        np.testing.assert_array_equal(
                            inference.controls[0].tet_activations.numpy(),
                            problem.controls[frame].tet_activations.numpy(),
                        )
                        self.assertEqual({state.particle_q.ptr for state in inference.states}, state_buffers)
                        self.assertEqual(inference.controls[0].tet_activations.ptr, control_buffer)
                        if inference_inverse is None:
                            inference_inverse = inference.solver.inverse
                        self.assertIs(inference.solver.inverse, inference_inverse)
                    self.assertGreater(inference.elapsed_steps, inference.frames)
                    self.assertGreater(inference.contact_frames, 0)
                    np.testing.assert_array_equal(inference.parameters.numpy(), parameters)
                    np.testing.assert_array_equal(inference.initial_x, parameters)
                    np.testing.assert_array_equal(inference.x, parameters)
                    with self.assertRaisesRegex(RuntimeError, "training=True"):
                        inference.evaluate(parameters)
                    with self.assertRaisesRegex(RuntimeError, "training=True"):
                        inference.optimize_step()
                with self.assertRaisesRegex(RuntimeError, "training=False"):
                    problem.advance()

                for seed in (104, 105):
                    direction = np.random.default_rng(seed).normal(size=16)
                    direction /= np.linalg.norm(direction)
                    # Keep perturbations inside the verified contact branch.
                    epsilon = 0.0002
                    plus = problem.evaluate(parameters + epsilon * direction, gradients=False, save_trajectory=False)
                    minus = problem.evaluate(parameters - epsilon * direction, gradients=False, save_trajectory=False)
                    self.assertIsNone(plus[2])
                    self.assertIsNone(minus[2])
                    numeric = (plus[0] - minus[0]) / (2.0 * epsilon)
                    np.testing.assert_allclose(gradient @ direction, numeric, rtol=0.02, atol=2e-6)
                    self.assertIs(problem.solver.inverse, inverse)
                np.testing.assert_array_equal(problem.states[0].particle_q.numpy(), positions[0])
                problem.optimize_step()
                self.assertEqual(problem.iteration, 1)
                self.assertLess(problem.value, problem.initial_loss)
                self.assertTrue(np.isfinite(problem.gradient).all())


if __name__ == "__main__":
    unittest.main()
