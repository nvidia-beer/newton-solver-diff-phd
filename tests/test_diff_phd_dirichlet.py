# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent prescribed positions and their boundary sensitivity."""

import unittest

import numpy as np
import warp as wp

from newton_solver_diff_phd import SolverDiffPhD
from tests.test_diff_phd_adjoint import TIGHT, _linear_loss
from tests.test_diff_phd_solver import DENSITY, DEVICE, _deformed_state, _grid


@wp.kernel
def _targets_from_state(q: wp.array[wp.vec3], offset: wp.vec3, targets: wp.array[wp.vec3]):
    i = wp.tid()
    targets[i] = q[i] + offset


class TestExplicitDirichlet(unittest.TestCase):
    def scene(self):
        model = _grid(requires_grad=True)
        model.tet_materials.requires_grad = False
        solver = SolverDiffPhD(model, density=DENSITY, integrate_pinned_velocities=True, **TIGHT)
        before = _deformed_state(model, stretch=(1.2, 0.85, 1.05), speed=0.3)
        for array in (before.particle_q, before.particle_qd, before.particle_f):
            array.requires_grad = True
        after = model.state(requires_grad=True)
        values = before.particle_q.numpy() + np.array([0.002, -0.001, 0.003], dtype=np.float32)
        targets = wp.array(values, dtype=wp.vec3, device=DEVICE, requires_grad=True)
        return model, solver, before, after, targets

    def test_prescribed_positions_do_not_replace_physical_input_velocity(self):
        model, solver, before, after, targets = self.scene()
        initial_q, initial_v = before.particle_q.numpy(), before.particle_qd.numpy()
        pinned = model.particle_inv_mass.numpy() == 0
        solver.step(before, after, None, None, 0.004, dirichlet_positions=targets)
        solver.check_status()
        np.testing.assert_array_equal(after.particle_q.numpy()[pinned], targets.numpy()[pinned])
        np.testing.assert_array_equal(before.particle_q.numpy(), initial_q)
        np.testing.assert_array_equal(before.particle_qd.numpy(), initial_v)
        np.testing.assert_array_equal(solver._position.numpy(), initial_q)
        np.testing.assert_array_equal(solver._velocity.numpy(), initial_v)
        expected = after.particle_q.numpy()
        altered = targets.numpy()
        altered[~pinned] += 1000.0
        targets.assign(altered)
        solver.step(before, after, None, None, 0.004, dirichlet_positions=targets)
        np.testing.assert_array_equal(after.particle_q.numpy(), expected)

    def test_target_and_physical_state_gradients_match_finite_differences(self):
        model, solver, before, after, targets = self.scene()
        dt = 0.004
        rng = np.random.default_rng(81)
        weights = [rng.standard_normal((model.particle_count, 3)) * scale for scale in (1.0, 0.01)]
        arrays = [wp.array(w, dtype=wp.vec3d, device=DEVICE) for w in weights]
        loss = wp.zeros(1, dtype=wp.float64, device=DEVICE, requires_grad=True)
        tape = wp.Tape()
        with tape:
            solver.step(before, after, None, None, dt, dirichlet_positions=targets)
            wp.launch(
                _linear_loss,
                model.particle_count,
                inputs=[after.particle_q, after.particle_qd, *arrays],
                outputs=[loss],
                device=DEVICE,
            )
        tape.backward(loss)
        solver.check_status()
        pinned = model.particle_inv_mass.numpy() == 0
        np.testing.assert_array_equal(targets.grad.numpy()[~pinned], 0.0)
        np.testing.assert_array_equal(before.particle_qd.grad.numpy()[pinned], 0.0)
        np.testing.assert_allclose(
            before.particle_q.grad.numpy()[pinned], -weights[1].astype(np.float32)[pinned] / dt, rtol=2e-7, atol=1e-7
        )

        def objective():
            solver.step(before, after, None, None, dt, dirichlet_positions=targets)
            x = solver._x.numpy()
            return np.sum(x * weights[0]) + np.sum((x - before.particle_q.numpy()) / dt * weights[1])

        nodes = [np.flatnonzero(pinned)[0], np.flatnonzero(~pinned)[7]]
        for array, epsilon in ((targets, 1e-4), (before.particle_q, 1e-4), (before.particle_qd, 1e-3)):
            gradients = array.grad.numpy().copy()
            original = array.numpy().copy()
            for node in nodes:
                for axis in (0, 2):
                    results, inputs = [], []
                    for sign in (1.0, -1.0):
                        values = original.copy()
                        values[node, axis] += sign * epsilon
                        array.assign(values)
                        inputs.append(float(values[node, axis]))
                        results.append(objective())
                    array.assign(original)
                    numeric = (results[0] - results[1]) / (inputs[0] - inputs[1])
                    self.assertAlmostEqual(gradients[node, axis], numeric, delta=3e-4 * max(abs(numeric), 1e-3))

    def test_upstream_target_kernel_and_captured_backward(self):
        model, solver, before, after, targets = self.scene()
        weights = wp.array(
            np.random.default_rng(33).standard_normal((model.particle_count, 3)), dtype=wp.vec3d, device=DEVICE
        )
        loss = wp.zeros(1, dtype=wp.float64, device=DEVICE, requires_grad=True)
        tape = wp.Tape()
        with tape:
            wp.launch(
                _targets_from_state,
                model.particle_count,
                inputs=[before.particle_q, wp.vec3(0.002, -0.001, 0.003)],
                outputs=[targets],
                device=DEVICE,
            )
            solver.step(before, after, None, None, 0.004, dirichlet_positions=targets)
            wp.launch(
                _linear_loss,
                model.particle_count,
                inputs=[after.particle_q, after.particle_qd, weights, weights],
                outputs=[loss],
                device=DEVICE,
            )
        tape.backward(loss)
        solver.check_status()
        pinned = model.particle_inv_mass.numpy() == 0
        expected = before.particle_q.grad.numpy().copy()
        # Warp consumes the target adjoint when reversing its producer. Check
        # the complete q -> target -> solve chain against forward differences.
        original = before.particle_q.numpy().copy()
        node = np.flatnonzero(pinned)[0]
        results, inputs = [], []
        for sign in (1.0, -1.0):
            values = original.copy()
            values[node, 0] += sign * 1e-4
            before.particle_q.assign(values)
            wp.launch(
                _targets_from_state,
                model.particle_count,
                inputs=[before.particle_q, wp.vec3(0.002, -0.001, 0.003)],
                outputs=[targets],
                device=DEVICE,
            )
            solver.step(before, after, None, None, 0.004, dirichlet_positions=targets)
            x = solver._x.numpy()
            results.append(np.sum((x + (x - values) / 0.004) * weights.numpy()))
            inputs.append(float(values[node, 0]))
        before.particle_q.assign(original)
        numeric = (results[0] - results[1]) / (inputs[0] - inputs[1])
        self.assertAlmostEqual(expected[node, 0], numeric, delta=3e-4 * max(abs(numeric), 1e-3))
        if wp.get_device(DEVICE).is_cuda:
            tape.zero()
            loss.grad.fill_(1.0)
            with wp.ScopedCapture(device=DEVICE) as capture:
                tape.backward()
            tape.zero()
            loss.grad.fill_(1.0)
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(before.particle_q.grad.numpy(), expected)

    def test_target_metadata_and_output_alias_are_rejected(self):
        model, solver, before, after, targets = self.scene()
        invalid = [
            np.zeros((model.particle_count, 3)),
            wp.zeros(model.particle_count, dtype=wp.vec3d, device=DEVICE),
            wp.zeros(model.particle_count - 1, dtype=wp.vec3, device=DEVICE),
            after.particle_q,
        ]
        if wp.get_device(DEVICE).is_cuda:
            invalid.append(wp.zeros(model.particle_count, dtype=wp.vec3, device="cpu"))
        for target in invalid:
            with self.subTest(target=type(target)), self.assertRaises((TypeError, ValueError)):
                solver.step(before, after, None, None, 0.004, dirichlet_positions=target)


if __name__ == "__main__":
    unittest.main()
