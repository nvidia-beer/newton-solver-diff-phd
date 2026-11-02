# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Alignment/material adjoints, mesh contact with a moving tool, and live forward equivalence."""

import os
import unittest
from unittest.mock import patch

import newton
import numpy as np
import warp as wp

from newton_solver_diff_phd import SolverDiffPhD
from newton_solver_diff_phd.examples._diff_phd_poke_alignment import (
    PokeAlignmentProblem,
    command_poke_positions,
    damp_poke_velocities,
)
from newton_solver_diff_phd.examples.example_softbody_diff_phd_ur5_poke import Example

_FACES = np.array(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)))
TIGHT = {"abs_tol": 1.0e-13, "rel_tol": 1.0e-11, "max_iterations": 20000}
# mesh_contact radius [m] and stiffness [N/m]: 10 N per vertex at full radius holds
# the 0.4 kg tetrahedron; four substeps keep the tool's 12 mm frame motion below the radius.
RADIUS, KN = 0.005, 2.0e3


def _model(device, soft, tool, masses, gravity):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
    for point, mass in zip(np.vstack((soft, tool)), masses, strict=True):
        builder.add_particle(wp.vec3(*point), wp.vec3(), mass=mass, radius=0.0)
    for ids in ((0, 1, 2, 3), (4, 5, 6, 7)):
        builder.add_tetrahedron(*ids, k_mu=1200.0, k_lambda=2400.0, k_damp=0.0)
    model = builder.finalize(device=device)
    # The reference mass is density * mean volume of both tetrahedra.
    points = np.vstack((soft, tool))
    tets = np.array(((0, 1, 2, 3), (4, 5, 6, 7)))
    volumes = [abs(np.linalg.det(points[tet[1:]] - points[tet[0]])) / 6.0 for tet in tets]
    faces = tets[:, _FACES]
    return model, float(np.mean(volumes)), faces[0], faces[1]


def _problem(device, *, contact=False, embedded=False, material="neo_hookean", root_only=False, **options):
    soft = np.array([[0.0, 0.0, 0.01], [0.15, 0.0, 0.01], [0.0, 0.15, 0.01], [0.0, 0.0, 0.16]])
    tool = np.array([[-0.4, -0.4, 0.0], [-0.4, 0.8, 0.0], [0.8, -0.4, 0.0], [-0.4, -0.4, -0.3]])
    if not contact:
        tool[:, 2] -= 1.0
    tool_masses = [0.0, 0.0, 0.0, 0.1 if root_only else 0.0]
    model, volume, soft_faces, tool_faces = _model(
        device, soft, tool, [0.1] * 4 + tool_masses, -9.81 if contact else 0.0
    )
    solver = SolverDiffPhD(
        model,
        density=0.1 / volume,
        material=material,
        integrate_pinned_velocities=True,
        mesh_contact=(RADIUS, KN, KN, 50.0, 0.8, soft_faces, tool_faces),
        **TIGHT,
    )
    substeps = options.setdefault("substeps", 4 if contact else 1)
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
    problem = PokeAlignmentProblem(model, solver, 4, np.arange(4), endpoints, targets, **settings)
    return problem, targets


def _stiffness_problem(device, *, young=60000.0, targets=None, **options):
    """Known gravity excites one pinned tetrahedron, making E identifiable.

    At the nominal parameter both bodies have the same material, so the
    reference mean-weight adjoint agrees with the local PD derivative. The
    reference Neo-Hookean weight derivative uses the hyperelastic stress at
    ``F`` instead of the proximal residual of the forward step (see
    ``test_diff_phd_adjoint``), so it does not match finite differences.
    """
    soft = np.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0], [0.0, 0.15, 0.0], [0.0, 0.0, 0.15]])
    tool = soft + np.array([1.0, 0.0, 0.0])
    model, volume, soft_faces, tool_faces = _model(device, soft, tool, [0.0, 0.0, 0.0, 0.2] + [0.0] * 4, -9.81)
    poisson = options.get("poisson", 0.4)
    materials = model.tet_materials.numpy()
    materials[:, 0] = young / (2.0 * (1.0 + poisson))
    materials[:, 1] = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    model.tet_materials.assign(materials)
    # The tool stays 1 m away, so the pinned tetrahedron never touches it.
    solver = SolverDiffPhD(
        model,
        density=0.2 / volume,
        material="pd",
        integrate_pinned_velocities=True,
        mesh_contact=(RADIUS, KN, KN, 50.0, 0.8, soft_faces, tool_faces),
        **TIGHT,
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


def _chamfer(points, target):
    distances = np.sum((points[:, None] - target[None]) ** 2, axis=2)
    return 0.5 * np.mean(np.min(distances, axis=1)) + 0.5 * np.mean(np.min(distances, axis=0))


def _replay(problem, xy, solver, forces=None):
    """Untaped rollout of ``problem`` with its mesh contact, or with fixed external ``forces`` instead.

    Returns the loss and, with mesh contact, the state force of every step.
    """
    model, device = problem.model, problem.device
    offset = np.array([xy[0] - problem.reference_xy[0], xy[1] - problem.reference_xy[1], 0.0])
    rest = problem.rest.numpy().astype(np.float64)
    rest[: problem.soft_count] += offset
    state, next_state = model.state(), model.state()
    endpoints = wp.empty_like(state.particle_q)
    state.particle_q.assign(rest.astype(np.float32))
    state.particle_qd.zero_()
    state.particle_f.zero_()
    next_state.particle_f.zero_()
    recorded, loss = [], 0.0
    for frame in range(problem.frames):
        for substep in range(problem.substeps):
            i = frame * problem.substeps + substep
            wp.launch(
                command_poke_positions,
                model.particle_count,
                inputs=[
                    state.particle_q,
                    problem.tool_positions,
                    i,
                    problem.soft_count,
                ],
                outputs=[endpoints],
                device=device,
            )
            if forces is not None:
                state.particle_f.assign(forces[i].astype(np.float32))
            solver.step(state, next_state, None, None, problem.dt, dirichlet_positions=endpoints)
            if forces is None:
                recorded.append(solver._mesh_contact.force.numpy().copy())
            if problem.velocity_damping != 1.0:
                velocity = next_state.particle_qd.numpy()
                velocity *= np.float32(problem.velocity_damping)
                next_state.particle_qd.assign(velocity)
            state, next_state = next_state, state
        points = state.particle_q.numpy().astype(np.float64)[problem.surface_indices]
        loss += _chamfer(points, problem.target_positions[frame])
    solver.check_status()
    return loss, recorded


class TestDiffPhDPokeAlignment(unittest.TestCase):
    def test_root_prescription_keeps_incoming_physical_velocity(self):
        device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        problem, _ = _problem(device, contact=True, root_only=True, velocity_damping=0.3)
        incoming = []
        step = problem.solver.step

        def observe(state, *args, **kwargs):
            before = state.particle_qd.numpy().copy()
            incoming.append(before)
            result = step(state, *args, **kwargs)
            np.testing.assert_array_equal(state.particle_qd.numpy(), before)
            return result

        with patch.object(problem.solver, "step", side_effect=observe):
            problem.evaluate(problem.x, gradients=False)
        np.testing.assert_array_equal(incoming[0], np.zeros((8, 3)))
        root = np.array([4, 5, 6])
        for i in range(1, problem.steps):
            positions = problem.states[i].particle_q.numpy()
            previous = problem.states[i - 1].particle_q.numpy()
            expected = np.float32(0.3) * ((positions[root] - previous[root]) / np.float32(problem.dt))
            np.testing.assert_allclose(incoming[i][root], expected, rtol=2e-6, atol=1e-7)
        # The fourth probe node is elastic: endpoint targets must constrain only
        # the root, while its incoming velocity participates in mesh friction.
        self.assertGreater(
            np.linalg.norm(problem.states[1].particle_q.numpy()[7] - problem.tool_positions.numpy()[1, 3]), 1e-5
        )

    def test_native_compressibility_gradient_and_loss_decrease(self):
        for device in [wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))]:
            with self.subTest(device=device):
                truth = _stiffness_problem(device, young=30000.0, fit_mode="compressibility", poisson=0.35)
                observed = truth.evaluate(truth.x, gradients=False)[2][1:, :4]
                # This scene's loss has a local maximum near nu = 0.2, so the
                # start lies inside the basin of the true value.
                problem = _stiffness_problem(
                    device, young=30000.0, targets=observed, fit_mode="compressibility", poisson=0.25
                )
                self.assertEqual(problem.parameter_names, ("poisson",))
                self.assertEqual(problem.poisson_reference, 0.25)
                np.testing.assert_array_equal(problem.x, [0.25])
                report = problem.check_gradient()
                self.assertEqual(report["epsilon"], 0.001)
                self.assertGreater(abs(report["analytic"][0]), 1e-07)
                np.testing.assert_allclose(report["analytic"], report["finite_difference"], rtol=0.005, atol=1e-09)
                problem.prepare()
                for _ in range(16):
                    if not problem.optimize_step():
                        break
                self.assertGreater(problem.iteration, 0)
                self.assertLess(problem.value, problem.initial_loss)
                self.assertGreaterEqual(problem.poisson, problem.poisson_bounds[0])
                self.assertLessEqual(problem.poisson, problem.poisson_bounds[1])
                self.assertEqual(problem.young, 30000.0)
                np.testing.assert_array_equal(problem.xy, problem.initial_xy)
                self.assertEqual(problem.history[-1]["poisson"], problem.poisson)

    def test_native_stiffness_gradient_and_loss_decrease(self):
        for device in [wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))]:
            with self.subTest(device=device):
                truth = _stiffness_problem(device, young=20000.0)
                observed = truth.evaluate(truth.x, gradients=False)[2][1:, :4]
                problem = _stiffness_problem(device, targets=observed)
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
                self.assertGreater(problem.iteration, 0)
                self.assertLess(problem.value, problem.initial_loss)
                self.assertGreaterEqual(problem.young, problem.young_bounds[0])
                self.assertLessEqual(problem.young, problem.young_bounds[1])
                np.testing.assert_array_equal(problem.xy, problem.initial_xy)
                self.assertEqual(problem.history[-1]["xy"], problem.initial_xy.tolist())
                self.assertEqual(problem.history[-1]["young_pa"], problem.young)

    def test_live_inference_matches_taped_contact_trajectory_and_holds_final_probe(self):
        for device in [wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))]:
            for damping in (1.0, 0.3):
                with self.subTest(device=device, damping=damping):
                    problem, _ = _problem(device, contact=True, velocity_damping=damping)
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
                        self.assertEqual(problem.solver.read_status(), 0)
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
                    example = Example.__new__(Example)
                    example.scene, example.xy, example.forward = scene, problem.x, forward
                    example.frame = problem.frames + 1
                    example.last_step_seconds = 1.0
                    example._draw_step = True
                    example._first_frame = False
                    inverse = problem.solver.inverse
                    with patch.object(wp, "Tape", side_effect=AssertionError("Restart created a Tape")):
                        example._restart_forward()
                        self.assertIs(problem.solver.inverse, inverse)
                        self.assertEqual(example.frame, 0)
                        self.assertEqual(example.forward.elapsed_steps, 0)
                        self.assertEqual(example.last_step_seconds, 0.0)
                        self.assertFalse(example._draw_step)
                        np.testing.assert_array_equal(example.forward.state.particle_q.numpy(), expected[0])
                        np.testing.assert_array_equal(example.forward.state.particle_qd.numpy(), np.zeros((8, 3)))
                        self.assertFalse(example.forward.state.particle_q.requires_grad)
                        self.assertFalse(example.forward.state.particle_qd.requires_grad)
                        example.forward.advance()
                    self.assertIs(problem.solver.inverse, inverse)
                    np.testing.assert_array_equal(example.forward.state.particle_q.numpy(), expected[1])
                    np.testing.assert_array_equal(
                        example.forward.state.particle_qd.numpy(), velocities[problem.substeps]
                    )
                    self.assertEqual(problem.solver.read_status(), 0)

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
                    for actual, expected in zip(problem.target_positions, target_snapshot, strict=True):
                        np.testing.assert_array_equal(actual, expected)
                    report = problem.check_gradient(epsilon=0.0002)
                    np.testing.assert_allclose(report["analytic"], report["finite_difference"], rtol=0.001, atol=1e-06)

    def test_contact_gradient_is_the_frozen_contact_force_derivative(self):
        """The reference mesh_contact force has a zero backward pass, so gradients
        differentiate the rollout with every step's contact force frozen. Finite
        differences of that frozen replay are an independent check."""
        for device in [wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))]:
            for damping in (1.0, 0.3):
                with self.subTest(device=device, damping=damping):
                    problem, _ = _problem(device, contact=True, velocity_damping=damping)
                    value, gradient, _ = problem.evaluate(problem.x)
                    self.assertGreater(problem.contact_steps, 0)
                    replayed, forces = _replay(problem, problem.x, problem.solver)
                    self.assertAlmostEqual(replayed, value, delta=1e-15)
                    frozen = SolverDiffPhD(
                        problem.model,
                        density=problem.solver.density,
                        material="neo_hookean",
                        integrate_pinned_velocities=True,
                        **TIGHT,
                    )
                    self.assertAlmostEqual(_replay(problem, problem.x, frozen, forces)[0], value, delta=1e-10)
                    step = 1.0e-3
                    numeric = [
                        (
                            _replay(problem, problem.x + step * axis, frozen, forces)[0]
                            - _replay(problem, problem.x - step * axis, frozen, forces)[0]
                        )
                        / (2.0 * step)
                        for axis in np.eye(2)
                    ]
                    np.testing.assert_allclose(gradient, numeric, rtol=1e-5, atol=1e-9)

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

    def test_training_timing_includes_rejected_rollouts(self):
        problem, _ = _problem(wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu")))
        problem.prepare()
        initial_timing = problem.last_training_timing
        self.assertEqual(initial_timing["forward_rollouts"], 1)
        self.assertEqual(initial_timing["backward_rollouts"], 1)
        rollout = problem._rollout
        trials = 0

        def reject_first(*args, **kwargs):
            nonlocal trials
            value, tape, progress = rollout(*args, **kwargs)
            trials += 1
            return (problem.value if trials == 1 else value), tape, progress

        with patch.object(problem, "_rollout", side_effect=reject_first):
            self.assertTrue(problem.optimize_step())
        timing = problem.last_training_timing
        self.assertGreaterEqual(trials, 2)
        self.assertEqual(timing["forward_rollouts"], trials)
        self.assertEqual(timing["backward_rollouts"], 1)
        self.assertEqual(timing["forward_substeps"], trials * problem.steps)
        self.assertEqual(timing["backward_substeps"], problem.steps)
        self.assertGreater(timing["forward_seconds"], 0.0)
        self.assertGreater(timing["backward_seconds"], 0.0)
        self.assertGreaterEqual(timing["seconds"], timing["forward_seconds"] + timing["backward_seconds"])
        self.assertIs(problem.history[-1]["timing"], timing)
        self.assertIs(problem.history[0]["timing"], initial_timing)

    def test_damping_chain_rule_and_contact_rollout_buffers(self):
        for device in [wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))]:
            with self.subTest(device=device):
                values = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
                seed = np.array([[0.2, 0.3, -0.1], [0.4, -0.5, 0.6], [-0.8, 0.7, 0.9]], dtype=np.float32)
                for damping in (0.0, 0.3, 1.0):
                    source = wp.array(values, dtype=wp.vec3, device=device, requires_grad=True)
                    out = wp.empty_like(source, requires_grad=True)
                    with wp.Tape() as tape:
                        wp.launch(
                            damp_poke_velocities,
                            3,
                            inputs=[source, damping],
                            outputs=[out],
                            device=device,
                        )
                    tape.backward(grads={out: wp.array(seed, dtype=wp.vec3, device=device)})
                    scales = np.float32(damping)
                    np.testing.assert_array_equal(out.numpy(), values * scales)
                    np.testing.assert_array_equal(source.grad.numpy(), seed * scales)
                problem, _ = _problem(device, contact=True, velocity_damping=0.3)
                gradient = problem.evaluate(problem.x)[1]
                self.assertEqual(len({target.ptr for target in problem.dirichlet_positions}), problem.steps)
                for original, target in zip(problem.states, problem.dirichlet_positions, strict=False):
                    self.assertNotEqual(original.particle_q.ptr, target.ptr)
                    self.assertNotEqual(original.particle_qd.ptr, target.ptr)
                for raw, damped in zip(problem.solver_states, problem.states[1:], strict=True):
                    self.assertEqual(raw.particle_q.ptr, damped.particle_q.ptr)
                    self.assertNotEqual(raw.particle_qd.ptr, damped.particle_qd.ptr)
                    raw_v, damped_v = (raw.particle_qd.numpy(), damped.particle_qd.numpy())
                    np.testing.assert_array_equal(damped_v, raw_v * np.float32(0.3))
                repeated = problem.evaluate(problem.x)[1]
                np.testing.assert_array_equal(repeated, gradient)
