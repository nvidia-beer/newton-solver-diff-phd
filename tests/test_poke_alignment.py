# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Alignment/material adjoints, moving contact, and live forward equivalence."""

import os
import unittest
from unittest.mock import patch

import newton
import numpy as np
import warp as wp

from newton_solver_diff_phd import PokeAlignmentProblem, SolverDiffPhD, damp_poke_velocities
from newton_solver_diff_phd.examples.example_softbody_diff_phd_ur5_poke import Example


def _problem(device, *, contact=False, embedded=False, **options):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81 if contact else 0.0))
    soft = np.array([[0.0, 0.0, 0.01], [0.15, 0.0, 0.01], [0.0, 0.15, 0.01], [0.0, 0.0, 0.16]])
    tool = np.array([[-0.4, -0.4, 0.0], [-0.4, 0.8, 0.0], [0.8, -0.4, 0.0], [-0.4, -0.4, -0.3]])
    if not contact:
        tool[:, 2] -= 1.0
    for i, point in enumerate(np.vstack((soft, tool))):
        builder.add_particle(wp.vec3(*point), wp.vec3(), mass=0.1 if i < 4 else 0.0, radius=0.0)
    for ids in ((0, 1, 2, 3), (4, 5, 6, 7)):
        builder.add_tetrahedron(*ids, k_mu=1200.0, k_lambda=2400.0, k_damp=0.0)
    model = builder.finalize(device=device)
    solver = SolverDiffPhD(
        model,
        material="neo_hookean",
        ground_plane=None,
        self_contact=False,
        soft_contact=contact,
        tolerance=1e-9,
        friction=0.8,
        contact_thickness=0.002,
        integrate_pinned_velocities=True,
    )
    substeps = options.get("substeps", 1)
    endpoints = np.stack([tool + np.array([0.006, 0.0, 0.012]) * (i / substeps) for i in range(2 * substeps + 1)])
    weights = np.array([[0.6, 0.2, 0.1, 0.1], [0.1, 0.6, 0.2, 0.1], [0.1, 0.2, 0.6, 0.1]])
    target = weights @ soft if embedded else soft
    target = target + np.array([0.003, -0.002, 0.0])
    targets = [target.copy() for _ in range(2)]
    settings = {
        "initial_xy": (0.015, 0.008),
        "reference_xy": (0.0, 0.0),
        "frame_dt": 0.04,
        "bounds": ((-0.05, 0.05), (-0.05, 0.05)),
        "surface_embedding": (np.tile(np.arange(4), (3, 1)), weights) if embedded else None,
    }
    settings.update(options)
    problem = PokeAlignmentProblem(
        model,
        solver,
        4,
        np.arange(4),
        endpoints,
        targets,
        **settings,
    )
    return problem, targets


def _stiffness_problem(device, *, young=60000.0, targets=None, **options):
    """Known gravity excites one pinned tetrahedron, making E identifiable."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    soft = np.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0], [0.0, 0.15, 0.0], [0.0, 0.0, 0.15]])
    tool = soft + np.array([1.0, 0.0, 0.0])
    for i, point in enumerate(np.vstack((soft, tool))):
        builder.add_particle(wp.vec3(*point), wp.vec3(), mass=0.2 if i == 3 else 0.0, radius=0.0)
    for ids in ((0, 1, 2, 3), (4, 5, 6, 7)):
        builder.add_tetrahedron(*ids, k_mu=1200.0, k_lambda=2400.0, k_damp=0.0)
    model = builder.finalize(device=device)
    solver = SolverDiffPhD(
        model,
        material="neo_hookean",
        ground_plane=None,
        self_contact=False,
        soft_contact=False,
        tolerance=1e-10,
        max_iterations=300,
        integrate_pinned_velocities=True,
    )
    initial_xy = (0.012, -0.008)
    targets = [soft + np.array([*initial_xy, 0.0]) for _ in range(6)] if targets is None else targets
    settings = {"fit_mode": "stiffness", "young_bounds": (5000.0, 200000.0)}
    settings.update(options)
    return PokeAlignmentProblem(
        model,
        solver,
        4,
        np.arange(4),
        np.tile(tool, (len(targets) + 1, 1, 1)),
        targets,
        initial_xy=initial_xy,
        reference_xy=(0.0, 0.0),
        bounds=((-0.05, 0.05), (-0.05, 0.05)),
        frame_dt=0.03,
        young=young,
        **settings,
    )


class TestDiffPhDPokeAlignment(unittest.TestCase):
    def test_native_compressibility_gradient_and_synthetic_recovery(self):
        for device in [wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))]:
            with self.subTest(device=device):
                truth = _stiffness_problem(device, young=30000.0, fit_mode="compressibility", poisson=0.35)
                observed = truth.evaluate(truth.x, gradients=False)[2][1:, :4]
                problem = _stiffness_problem(
                    device, young=30000.0, targets=observed, fit_mode="compressibility", poisson=0.15
                )
                original_tool = problem.model.tet_materials.numpy()[1].copy()
                self.assertEqual(problem.parameter_names, ("poisson",))
                self.assertEqual(problem.poisson_reference, 0.15)
                np.testing.assert_array_equal(problem.x, [0.15])
                report = problem.check_gradient()
                self.assertEqual(report["epsilon"], 0.001)
                self.assertGreater(abs(report["analytic"][0]), 1e-07)
                np.testing.assert_allclose(report["analytic"], report["finite_difference"], rtol=0.005, atol=1e-09)
                problem.prepare()
                for _ in range(16):
                    if not problem.optimize_step():
                        break
                self.assertLess(abs(problem.poisson - 0.35), 0.001)
                self.assertLess(problem.value, problem.initial_loss * 1e-05)
                self.assertEqual(problem.young, 30000.0)
                np.testing.assert_array_equal(problem.xy, problem.initial_xy)
                np.testing.assert_array_equal(problem.model.tet_materials.numpy()[1], original_tool)
                self.assertEqual(problem.history[-1]["poisson"], problem.poisson)

    def test_native_stiffness_gradient_and_synthetic_recovery(self):
        for device in [wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))]:
            with self.subTest(device=device):
                truth = _stiffness_problem(device, young=20000.0)
                observed = truth.evaluate(truth.x, gradients=False)[2][1:, :4]
                problem = _stiffness_problem(device, targets=observed)
                original_tool = problem.model.tet_materials.numpy()[1].copy()
                self.assertFalse(problem._fixed_xy.requires_grad)
                self.assertTrue(problem.model.tet_materials.requires_grad)
                self.assertEqual(problem.parameter_names, ("log_young",))
                np.testing.assert_array_equal(problem.x, [0.0])
                report = problem.check_gradient(epsilon=0.001)
                self.assertGreater(abs(report["analytic"][0]), 1e-07)
                np.testing.assert_allclose(report["analytic"], report["finite_difference"], rtol=0.005, atol=1e-09)
                first_factor = problem.solver.inverse
                value, gradient, _ = problem.evaluate(problem.x)
                self.assertIsNot(first_factor, problem.solver.inverse)
                repeated = problem.evaluate(problem.x)
                np.testing.assert_allclose(repeated[1], gradient, rtol=1e-08, atol=1e-12)
                self.assertEqual(repeated[0], value)
                problem.prepare()
                for _ in range(16):
                    if not problem.optimize_step():
                        break
                self.assertLess(abs(problem.young / 20000.0 - 1.0), 0.005)
                self.assertLess(problem.value, problem.initial_loss * 1e-05)
                np.testing.assert_array_equal(problem.xy, problem.initial_xy)
                np.testing.assert_array_equal(problem.model.tet_materials.numpy()[1], original_tool)
                np.testing.assert_array_equal(problem.positions[:, 4:], np.tile(problem.rest.numpy()[4:], (7, 1, 1)))
                self.assertEqual(problem.history[-1]["xy"], problem.initial_xy.tolist())
                self.assertEqual(problem.history[-1]["young_pa"], problem.young)

    def test_live_inference_matches_taped_contact_trajectory_and_holds_final_probe(self):
        for device in [wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))]:
            for damping in (1.0, 0.3):
                with self.subTest(device=device, damping=damping):
                    problem, _ = _problem(
                        device, contact=True, velocity_damping=damping, substeps=2 if damping == 0.3 else 1
                    )
                    _, _, expected = problem.evaluate(problem.x)
                    velocities = [state.particle_qd.numpy() for state in problem.states]
                    scene = {
                        "model": problem.model,
                        "solver": problem.solver,
                        "rest": problem.rest.numpy(),
                        "soft_count": problem.soft_count,
                        "data": {"reference_xy": problem.reference_xy},
                        "tool_positions": problem.tool_positions.numpy(),
                        "frame_dt": problem.frame_dt,
                        "substeps": problem.substeps,
                        "velocity_damping": damping,
                    }
                    forward = Example._Forward(scene, problem.x)
                    pointers = {forward.state.particle_q.ptr, forward.next_state.particle_q.ptr}
                    velocity_pointers = {
                        forward.state.particle_qd.ptr,
                        forward.next_state.particle_qd.ptr,
                        forward.damped_velocity.ptr,
                    }
                    with patch.object(wp, "Tape", side_effect=AssertionError("Inference created a Tape")):
                        for frame in range(1, problem.frames + 1):
                            forward.advance()
                            np.testing.assert_array_equal(forward.state.particle_q.numpy(), expected[frame])
                            np.testing.assert_array_equal(
                                forward.state.particle_qd.numpy(), velocities[frame * problem.substeps]
                            )
                        forward.advance()
                    self.assertEqual(pointers, {forward.state.particle_q.ptr, forward.next_state.particle_q.ptr})
                    self.assertEqual(
                        velocity_pointers,
                        {
                            forward.state.particle_qd.ptr,
                            forward.next_state.particle_qd.ptr,
                            forward.damped_velocity.ptr,
                        },
                    )
                    self.assertFalse(forward.state.particle_q.requires_grad)
                    self.assertFalse(forward.state.particle_qd.requires_grad)
                    self.assertEqual(forward.elapsed_steps, problem.steps + problem.substeps)
                    np.testing.assert_array_equal(
                        forward.state.particle_q.numpy()[problem.soft_count :], scene["tool_positions"][-1]
                    )
                    self.assertLessEqual(forward.max_residual, problem.solver.tolerance)

    def test_chamfer_translation_gradient_and_fixed_target_embedding(self):
        for device in [wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))]:
            for embedded in (False, True):
                with self.subTest(device=device, embedded=embedded):
                    problem, targets = _problem(device, embedded=embedded)
                    self.assertIsNone(problem.value)
                    target_snapshot = [points.copy() for points in problem.target_positions]
                    for points in targets:
                        points[:] = 5.0
                    value, gradient, positions = problem.evaluate(problem.x)
                    difference = problem.x - [0.003, -0.002]
                    self.assertAlmostEqual(value, 2.0 * float(difference @ difference), delta=2e-09)
                    np.testing.assert_allclose(gradient, 4.0 * difference, rtol=2e-05, atol=2e-08)
                    np.testing.assert_allclose(positions[:, 4:], problem.tool_positions.numpy(), atol=1e-07)
                    for actual, expected in zip(problem.target_positions, target_snapshot, strict=True):
                        np.testing.assert_array_equal(actual, expected)
                    report = problem.check_gradient(epsilon=0.0002)
                    np.testing.assert_allclose(report["analytic"], report["finite_difference"], rtol=0.001, atol=1e-06)

    def test_contact_alignment_gradients_and_distinct_command_buffers(self):
        for device in [wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))]:
            with self.subTest(device=device):
                problem, _ = _problem(device, contact=True)
                report = problem.check_gradient(epsilon=0.0002)
                self.assertGreater(problem.contact_steps, 0)
                self.assertLessEqual(problem.max_residual, problem.solver.tolerance)
                np.testing.assert_allclose(report["analytic"], report["finite_difference"], rtol=0.01, atol=2e-05)
                for original, command in zip(problem.states, problem.command_states, strict=False):
                    self.assertEqual(original.particle_q.ptr, command.particle_q.ptr)
                    self.assertNotEqual(original.particle_qd.ptr, command.particle_qd.ptr)
                self.assertEqual(len({state.particle_qd.ptr for state in problem.command_states}), problem.steps)

    def test_optimizer_uses_native_gradient_and_reduces_recorded_loss(self):
        for device in [wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))]:
            with self.subTest(device=device):
                problem, _ = _problem(device)
                problem.prepare()
                initial = problem.value
                with patch.object(problem, "check_gradient", side_effect=AssertionError("FD used by optimizer")):
                    for _ in range(8):
                        if not problem.optimize_step():
                            break
                self.assertLess(problem.value, initial * 0.01)
                self.assertLess(np.linalg.norm(problem.x - [0.003, -0.002]), 0.001)
                self.assertTrue(
                    all((b["loss"] < a["loss"] for a, b in zip(problem.history, problem.history[1:], strict=False)))
                )

    def test_damping_chain_rule_and_contact_rollout_buffers(self):
        for device in [wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))]:
            with self.subTest(device=device):
                inverse_mass = wp.array([1.0, 0.0, 2.0], dtype=float, device=device)
                values = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
                seed = np.array([[0.2, 0.3, -0.1], [0.4, -0.5, 0.6], [-0.8, 0.7, 0.9]], dtype=np.float32)
                for damping in (0.0, 0.3, 1.0):
                    source = wp.array(values, dtype=wp.vec3, device=device, requires_grad=True)
                    out = wp.empty_like(source, requires_grad=True)
                    with wp.Tape() as tape:
                        wp.launch(
                            damp_poke_velocities,
                            3,
                            inputs=[source, inverse_mass, damping],
                            outputs=[out],
                            device=device,
                        )
                    tape.backward(grads={out: wp.array(seed, dtype=wp.vec3, device=device)})
                    scales = np.array([damping, 1.0, damping], dtype=np.float32)[:, None]
                    np.testing.assert_array_equal(out.numpy(), values * scales)
                    np.testing.assert_array_equal(source.grad.numpy(), seed * scales)
                problem, _ = _problem(device, contact=True, velocity_damping=0.3)
                report = problem.check_gradient(epsilon=0.0002)
                np.testing.assert_allclose(report["analytic"], report["finite_difference"], rtol=0.01, atol=2e-05)
                problem.evaluate(problem.x)
                free = problem.model.particle_inv_mass.numpy() > 0.0
                for raw, damped in zip(problem.solver_states, problem.states[1:], strict=True):
                    self.assertEqual(raw.particle_q.ptr, damped.particle_q.ptr)
                    self.assertNotEqual(raw.particle_qd.ptr, damped.particle_qd.ptr)
                    raw_v, damped_v = (raw.particle_qd.numpy(), damped.particle_qd.numpy())
                    np.testing.assert_array_equal(damped_v[free], raw_v[free] * np.float32(0.3))
                    np.testing.assert_array_equal(damped_v[~free], raw_v[~free])
                repeated = problem.evaluate(problem.x)[1]
                np.testing.assert_allclose(repeated, report["analytic"], rtol=1e-06, atol=1e-08)
