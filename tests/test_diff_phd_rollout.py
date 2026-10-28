# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Recorded trajectories, retained primal snapshots, and full KKT derivatives."""

import os
import unittest
from contextlib import nullcontext

import numpy as np
import warp as wp

from newton_solver_diff_phd import SolverDiffPhD
from newton_solver_diff_phd._src.solvers.diff_phd.adjoint import KKTSystem
from tests.test_diff_phd_solver import _linear_state_loss, _model, _step


def _fixture(device, contact=False, material="pd"):
    model = _model(device, requires_grad=True, gravity=-9.81 if contact else 0.0, origin=(0.0, 0.0, 0.005))
    solver = SolverDiffPhD(
        model,
        material=material,
        ground_plane=(0.0, 0.0, 1.0, 0.0) if contact else None,
        contact_thickness=0.005,
        friction=0.4,
        tolerance=1e-9,
        max_iterations=500,
    )
    rng = np.random.default_rng(462)
    q = model.particle_q.numpy().astype(np.float64)
    q[3] += [0.014, -0.009, 0.019]
    velocity = rng.normal(scale=0.03, size=q.shape)
    forces = rng.normal(scale=0.4, size=(3, *q.shape))
    if contact:
        velocity[:3] = [0.18, 0.03, -0.025]
        forces[:, :3] = 0.0
    weights = rng.normal(size=q.shape).astype(np.float32)
    velocity_weights = rng.normal(scale=0.02, size=q.shape).astype(np.float32)
    return solver, model, q, velocity, forces, weights, velocity_weights


def _rollout(solver, model, q, velocity, forces, weights, velocity_weights, *, tape=None):
    states = [model.state(requires_grad=tape is not None) for _ in range(len(forces) + 1)]
    states[0].particle_q.assign(np.asarray(q, dtype=np.float32))
    states[0].particle_qd.assign(np.asarray(velocity, dtype=np.float32))
    for state, force in zip(states[:-1], forces, strict=True):
        state.particle_f.assign(np.asarray(force, dtype=np.float32))
    loss = wp.zeros(1, dtype=wp.float64, device=model.device, requires_grad=tape is not None)
    with tape if tape is not None else nullcontext():
        for i in range(len(forces)):
            solver.step(states[i], states[i + 1], model.control(), None, 0.025)
        wp.launch(
            _linear_state_loss,
            len(q),
            inputs=[
                states[-1].particle_q,
                states[-1].particle_qd,
                wp.array(weights, dtype=wp.vec3, device=model.device),
                wp.array(velocity_weights, dtype=wp.vec3, device=model.device),
            ],
            outputs=[loss],
            device=model.device,
        )
    return states, loss


def _gradients(states, model):
    return (
        states[0].particle_q.grad.numpy().copy(),
        states[0].particle_qd.grad.numpy().copy(),
        np.stack([state.particle_f.grad.numpy().copy() for state in states[:-1]]),
        model.tet_materials.grad.numpy().copy(),
    )


class TestDiffPhDRollout(unittest.TestCase):
    def _check_rollout(self, device, contact=False, material="pd"):
        solver, model, q, velocity, forces, weights, velocity_weights = _fixture(device, contact, material)
        materials = model.tet_materials.numpy().copy()
        tape = wp.Tape()
        states, loss = _rollout(solver, model, q, velocity, forces, weights, velocity_weights, tape=tape)
        tape.backward(loss=loss)
        q_grad, v_grad, f_grad, material_grad = _gradients(states, model)
        self.assertEqual(len({state.particle_q.ptr for state in states}), 4)
        rng = np.random.default_rng(811)
        for index, (value, gradient, epsilon) in enumerate(
            ((q, q_grad, 3e-4), (velocity, v_grad, 2e-3), (forces, f_grad, 0.04))
        ):
            with self.subTest(parameter=index):
                direction = rng.normal(size=value.shape)
                if contact and index == 0:
                    direction[:3] = 0.0
                losses = []
                for sign in (-1.0, 1.0):
                    arguments = [q, velocity, forces]
                    arguments[index] = value + sign * epsilon * direction
                    _, perturbed = _rollout(solver, model, *arguments, weights, velocity_weights)
                    losses.append(float(perturbed.numpy()[0]))
                finite_difference = (losses[1] - losses[0]) / (2 * epsilon)
                np.testing.assert_allclose(np.sum(gradient * direction), finite_difference, rtol=0.03, atol=4e-4)
        for column in (0, 1):
            with self.subTest(material_column=column):
                losses = []
                for sign in (-1.0, 1.0):
                    perturbed_material = materials.copy()
                    perturbed_material[0, column] += sign * 2.0
                    model.tet_materials.assign(perturbed_material)
                    _, perturbed = _rollout(solver, model, q, velocity, forces, weights, velocity_weights)
                    losses.append(float(perturbed.numpy()[0]))
                finite_difference = (losses[1] - losses[0]) / 4.0
                np.testing.assert_allclose(material_grad[0, column], finite_difference, rtol=0.04, atol=7e-7)
        model.tet_materials.assign(materials)

    def test_three_step_free_trajectory_gradients(self):
        for device in [wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))]:
            for material in ("pd", "neo_hookean"):
                with self.subTest(device=device, material=material), wp.ScopedDevice(device):
                    self._check_rollout(device, material=material)

    def test_three_step_ground_contact_gradients(self):
        for device in [wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))]:
            with self.subTest(device=device), wp.ScopedDevice(device):
                self._check_rollout(device, contact=True)

    def test_backward_retains_state_material_and_factor_snapshots(self):
        for device in [wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))]:
            with self.subTest(device=device), wp.ScopedDevice(device):
                solver, model, q, velocity, forces, weights, velocity_weights = _fixture(device, contact=True)
                tape = wp.Tape()
                states, loss = _rollout(solver, model, q, velocity, forces, weights, velocity_weights, tape=tape)
                original_factor = solver.inverse
                tape.backward(loss=loss)
                expected = _gradients(states, model)
                tape.zero()
                materials = model.tet_materials.numpy().copy()
                materials[:, :2] *= 4.0
                model.tet_materials.assign(materials)
                _rollout(solver, model, q, velocity, forces, weights, velocity_weights)
                self.assertIsNot(solver.inverse, original_factor)
                for state in states:
                    state.particle_q.fill_(wp.vec3(9.0, -7.0, 5.0))
                    state.particle_qd.fill_(wp.vec3(-3.0, 2.0, 1.0))
                    state.particle_f.fill_(wp.vec3(300.0, -200.0, 100.0))
                # The terminal loss is linear: its VJP does not need live state
                # values, so any difference here comes from a lost primal snapshot.
                tape.backward(loss=loss)
                for actual, reference in zip(_gradients(states, model), expected, strict=True):
                    np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=2e-7)

    def test_recorded_ping_pong_state_reuse_is_rejected(self):
        for device in [wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))]:
            with self.subTest(device=device), wp.ScopedDevice(device):
                solver, model, *_ = _fixture(device)
                first, second = model.state(requires_grad=True), model.state(requires_grad=True)
                with wp.Tape():
                    solver.step(first, second, model.control(), None, 0.025)
                    with self.assertRaisesRegex(ValueError, "distinct|reuse|record"):
                        solver.step(second, first, model.control(), None, 0.025)

    def test_full_kkt_forward_and_transpose_against_finite_differences(self):
        for device in [wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))]:
            with self.subTest(device=device), wp.ScopedDevice(device):
                solver, model, q, velocity, forces, _, _ = _fixture(device, contact=True, material="neo_hookean")
                initial, _ = _step(solver, model, q, velocity, forces[0], dt=0.025)
                response = solver._last_response
                system = KKTSystem(solver, solver._last_q, response, 0.025)
                previous = wp.array(initial.particle_q.numpy(), dtype=wp.vec3d, device=device)
                velocity_wp = wp.array(initial.particle_qd.numpy(), dtype=wp.vec3d, device=device)
                force_wp = wp.array(initial.particle_f.numpy(), dtype=wp.vec3d, device=device)
                point = np.r_[solver._last_q.numpy().reshape(-1), response.lam.numpy()]
                contact_force = wp.empty(model.particle_count, dtype=wp.vec3d, device=device)

                def residual(
                    value,
                    device=device,
                    response=response,
                    solver=solver,
                    previous=previous,
                    velocity_wp=velocity_wp,
                    force_wp=force_wp,
                    contact_force=contact_force,
                ):
                    trial = wp.array(value[:12].reshape(4, 3), dtype=wp.vec3d, device=device)
                    response.lam.assign(value[12:])
                    _, mechanical = solver._error(trial, previous, velocity_wp, force_wp, 0.025, response)
                    response.force(contact_force)
                    return np.r_[
                        (mechanical.numpy() - contact_force.numpy()).reshape(-1), response.residual.numpy().copy()
                    ]

                epsilon = 1e-6
                identity = np.eye(len(point))
                matrix = np.column_stack(
                    [
                        (residual(point + epsilon * direction) - residual(point - epsilon * direction)) / (2 * epsilon)
                        for direction in identity
                    ]
                )
                rhs = np.random.default_rng(713).normal(size=len(point))
                rhs_wp = wp.array(rhs, dtype=wp.float64, device=device)
                for transpose in (False, True):
                    actual = system.solve(rhs_wp, transpose=transpose).numpy()
                    expected = np.linalg.solve(matrix.T if transpose else matrix, rhs)
                    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
