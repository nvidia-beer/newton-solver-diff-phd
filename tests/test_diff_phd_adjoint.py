# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reference DiffPhD backward pass: derivative transcriptions and exact-regime gradients."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton_solver_diff_phd import SolverDiffPhD
from newton_solver_diff_phd._src.solvers.diff_phd import adjoint
from newton_solver_diff_phd._src.solvers.diff_phd.projections import (
    corotated_jvp,
    corotated_projection,
    diagonal_matrix,
    muscle_jvp,
    muscle_projection,
    nh_prox_jvp,
    nh_prox_stretches,
    unsigned_svd,
    volume_jvp,
    volume_offset,
)
from tests.reference_oracle import reference_backward
from tests.test_diff_phd_contact import _lifted, _surface
from tests.test_diff_phd_solver import DENSITY, DEVICE, _deformed_state, _grid, _oracle

TIGHT = {"max_iterations": 3000, "abs_tol": 1.0e-12, "rel_tol": 1.0e-12}
MU, LAME = 3.0e4, 8.0e4


@wp.kernel(enable_backward=False)
def _projections(
    f: wp.array[wp.mat33d],
    direction: wp.array[wp.vec3d],
    rotation: wp.array[wp.mat33d],
    volume: wp.array[wp.mat33d],
    neo_hookean: wp.array[wp.mat33d],
    fibre: wp.array[wp.vec3d],
):
    i = wp.tid()
    u, s, v = unsigned_svd(f[i])
    vt = wp.transpose(v)
    rotation[i] = corotated_projection(u, v)
    offset, _singular = volume_offset(s)
    volume[i] = u * diagonal_matrix(s + offset) * vt
    mu, lame = wp.float64(MU), wp.float64(LAME)
    neo_hookean[i] = u * diagonal_matrix(nh_prox_stretches(s, mu, lame, wp.float64(2.0) * mu)) * vt
    fibre[i] = muscle_projection(f[i], direction[i], wp.float64(0.8))


@wp.kernel(enable_backward=False)
def _jvps(
    f: wp.array[wp.mat33d],
    df: wp.array[wp.mat33d],
    direction: wp.array[wp.vec3d],
    rotation: wp.array[wp.mat33d],
    volume: wp.array[wp.mat33d],
    neo_hookean: wp.array[wp.mat33d],
    fibre: wp.array[wp.vec3d],
):
    i = wp.tid()
    u, s, v = unsigned_svd(f[i])
    rotation[i] = corotated_jvp(u, s, v, df[i])
    offset, _singular = volume_offset(s)
    volume[i] = volume_jvp(u, s, v, offset, df[i])
    mu, lame = wp.float64(MU), wp.float64(LAME)
    penalty = wp.float64(2.0) * mu
    stretches = nh_prox_stretches(s, mu, lame, penalty)
    neo_hookean[i] = nh_prox_jvp(u, s, v, stretches, df[i], mu, lame, penalty)
    fibre[i] = muscle_jvp(f[i], direction[i], wp.float64(0.8), df[i])


@wp.kernel
def _linear_loss(
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    q_weight: wp.array[wp.vec3d],
    qd_weight: wp.array[wp.vec3d],
    loss: wp.array[wp.float64],
):
    i = wp.tid()
    wp.atomic_add(loss, 0, wp.dot(wp.vec3d(q[i]), q_weight[i]) + wp.dot(wp.vec3d(qd[i]), qd_weight[i]))


def _evaluate(arrays, inputs, outputs):
    out = [wp.empty(len(inputs[0]), dtype=dtype, device=DEVICE) for dtype in outputs]
    wp.launch(arrays, len(inputs[0]), inputs=inputs, outputs=out, device=DEVICE)
    return [o.numpy() for o in out]


class TestDiffPhDAdjoint(unittest.TestCase):
    def test_zero_volume_energy_suppresses_whole_reference_material_gradient(self):
        model = _grid(requires_grad=True)
        materials = model.tet_materials.numpy()
        materials[:, 1] = 0.0
        model.tet_materials.assign(materials)
        solver = SolverDiffPhD(model, density=DENSITY, **TIGHT)
        state_in = _deformed_state(model, stretch=(1.2, 0.85, 1.05), speed=0.3)
        state_in.particle_q.requires_grad = True
        state_out = model.state(requires_grad=True)
        seed = wp.array(
            np.random.default_rng(40).standard_normal((model.particle_count, 3)), dtype=wp.vec3, device=DEVICE
        )
        tape = wp.Tape()
        with tape:
            solver.step(state_in, state_out, None, None, 0.004)
        tape.backward(grads={state_out.particle_q: seed})
        self.assertEqual(solver.read_status(), 16)
        np.testing.assert_array_equal(model.tet_materials.grad.numpy(), 0.0)
        self.assertGreater(np.linalg.norm(state_in.particle_q.grad.numpy()), 0.0)

    def test_reference_safe_clip_scrubs_components_before_limiting_norm(self):
        source = np.array([[[np.nan, np.inf, -np.inf], [3.0, 4.0, 0.0]]])
        values = wp.array(source, dtype=wp.vec3d, device=DEVICE)
        partials = wp.zeros((1, 1), dtype=wp.float64, device=DEVICE)
        status = wp.zeros(1, dtype=wp.int32, device=DEVICE)
        free = wp.ones(2, dtype=wp.int32, device=DEVICE)
        wp.launch(adjoint._clip_partials, (1, 1), inputs=[values, free, 2], outputs=[partials, status], device=DEVICE)
        wp.launch(adjoint._clip_apply, (1, 2), inputs=[partials, 1, 2.0], outputs=[values, status], device=DEVICE)
        np.testing.assert_allclose(values.numpy(), [[[0.0, 0.0, 0.0], [1.2, 1.6, 0.0]]], rtol=0, atol=1e-15)
        self.assertEqual(status.numpy()[0], 16)

    def test_reference_cascade_guard_uses_rhs_norm_plus_one(self):
        partials = wp.array([[9.0, 4.0]], dtype=wp.vec2d, device=DEVICE)
        scale = wp.zeros(1, dtype=wp.float64, device=DEVICE)
        diverged = wp.zeros(1, dtype=wp.int32, device=DEVICE)
        status = wp.zeros(1, dtype=wp.int32, device=DEVICE)
        wp.launch(
            adjoint._routing_guard, 1, inputs=[partials, 1, 10.0, 1e6], outputs=[scale, diverged, status], device=DEVICE
        )
        self.assertAlmostEqual(scale.numpy()[0], (3.0 + 1.0) / (10.0 * 2.0))
        self.assertEqual(diverged.numpy()[0], 0)
        self.assertEqual(status.numpy()[0], 16)

    def test_single_energy_heterogeneous_adjoint_uses_reference_mean_weights(self):
        model = _grid(contrast=2.0, requires_grad=True)
        solver = SolverDiffPhD(model, density=DENSITY, **TIGHT)
        state_in = _deformed_state(model, stretch=(1.2, 0.85, 1.05), speed=0.3)
        state_out = model.state(requires_grad=True)
        weights = np.random.default_rng(49).standard_normal((model.particle_count, 3)).astype(np.float32)
        tape = wp.Tape()
        with tape:
            solver.step(state_in, state_out, None, None, 0.004)
        tape.backward(grads={state_out.particle_q: wp.array(weights, dtype=wp.vec3, device=DEVICE)})
        bits = solver.read_status()
        self.assertEqual(bits & 8, 0)
        oracle = reference_backward(
            _oracle(model, "pd"), solver._x.numpy(), 0.004, weights.astype(np.float64), np.zeros_like(weights)
        )
        np.testing.assert_allclose(solver._adjoint._mu.numpy(), oracle["mu"].reshape(-1, 3), rtol=2e-5, atol=2e-10)

    def test_reference_backbone_clip_suppresses_unreliable_contributions(self):
        model = _grid(fix_left=False, requires_grad=True)
        solver = SolverDiffPhD(model, density=DENSITY, **TIGHT)
        state_in, state_out = model.state(requires_grad=True), model.state(requires_grad=True)
        tape = wp.Tape()
        with tape:
            solver.step(state_in, state_out, None, None, 0.004)
        seed = wp.full(model.particle_count, wp.vec3(1e12, 0.0, 0.0), dtype=wp.vec3, device=DEVICE)
        tape.backward(grads={state_out.particle_q: seed})
        self.assertTrue(solver.read_status() & 16)
        self.assertEqual(solver._adjoint._diverged.numpy()[0], 1)
        for array in (state_in.particle_q, state_in.particle_qd, state_in.particle_f, model.tet_materials):
            np.testing.assert_array_equal(array.grad.numpy(), 0.0)

    def test_projection_jvps_match_finite_differences(self):
        rng = np.random.default_rng(13)
        # Generic deformations: distinct singular values, where the reference
        # dSvd guard is inactive and the transcribed formulas are exact.
        f = np.array([np.diag([1.3, 0.9, 0.7]) + 0.2 * rng.standard_normal((3, 3)) for _ in range(10)])
        df = rng.standard_normal((10, 3, 3))
        direction = rng.standard_normal((10, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        types = (wp.mat33d, wp.mat33d, wp.mat33d, wp.vec3d)
        wf = lambda a: wp.array(a, dtype=wp.mat33d, device=DEVICE)  # noqa: E731
        wd = wp.array(direction, dtype=wp.vec3d, device=DEVICE)
        eps = 1.0e-6
        plus = _evaluate(_projections, [wf(f + eps * df), wd], types)
        minus = _evaluate(_projections, [wf(f - eps * df), wd], types)
        analytic = _evaluate(_jvps, [wf(f), wf(df), wd], types)
        for name, a, p, m in zip(("rotation", "volume", "neo_hookean", "fibre"), analytic, plus, minus, strict=True):
            with self.subTest(name):
                numeric = (p - m) / (2.0 * eps)
                np.testing.assert_allclose(a, numeric, rtol=0.0, atol=2.0e-6 * max(1.0, np.abs(numeric).max()))

    def _gradient_check(self, material, *, muscles=False, integrate=False):
        model = _grid(requires_grad=True)
        options = dict(TIGHT, density=DENSITY, material=material, integrate_pinned_velocities=integrate)
        rng = np.random.default_rng(21)
        if muscles:
            options["muscle_directions"] = rng.standard_normal((model.tet_count, 3))
            options["muscle_stiffness"] = 4.0e4
        solver = SolverDiffPhD(model, **options)
        dt = 0.004
        state_in = _deformed_state(model, stretch=(1.2, 0.85, 1.05), speed=0.3)
        for name in ("particle_q", "particle_qd", "particle_f"):
            getattr(state_in, name).requires_grad = True
        state_in.particle_f.assign(rng.uniform(-1.0, 1.0, (model.particle_count, 3)).astype(np.float32))
        state_out = model.state(requires_grad=True)
        control = model.control()
        contraction = rng.uniform(-0.2, 0.3, model.tet_count).astype(np.float32)
        if muscles:
            control.tet_activations = wp.array(contraction, dtype=wp.float32, device=DEVICE, requires_grad=True)
        q_weight = rng.standard_normal((model.particle_count, 3))
        qd_weight = 0.01 * rng.standard_normal((model.particle_count, 3))
        loss = wp.zeros(1, dtype=wp.float64, device=DEVICE, requires_grad=True)
        tape = wp.Tape()
        with tape:
            solver.step(state_in, state_out, control, None, dt)
            wp.launch(
                _linear_loss,
                model.particle_count,
                inputs=[
                    state_out.particle_q,
                    state_out.particle_qd,
                    wp.array(q_weight, dtype=wp.vec3d, device=DEVICE),
                    wp.array(qd_weight, dtype=wp.vec3d, device=DEVICE),
                ],
                outputs=[loss],
                device=DEVICE,
            )
        tape.backward(loss)
        solver.check_status()
        free = model.particle_inv_mass.numpy() > 0

        def objective():
            solver.step(state_in, state_out, control, None, dt)
            solver.check_status()
            x = solver._x.numpy()
            q = state_in.particle_q.numpy().astype(np.float64)
            return float(np.sum(x * q_weight) + np.sum((x - q) / dt * qd_weight))

        def central(array, index, component, step):
            values = array.numpy()
            original = values.copy()
            results, deltas = [], []
            for sign in (1.0, -1.0):
                values[index, component] = original[index, component] + sign * step
                array.assign(values)
                deltas.append(float(array.numpy()[index, component]))
                if array is model.tet_materials:
                    solver.notify_model_changed(0)
                results.append(objective())
            array.assign(original)
            if array is model.tet_materials:
                solver.notify_model_changed(0)
            return (results[0] - results[1]) / (deltas[0] - deltas[1])

        nodes = np.flatnonzero(free)[[0, 7, 15]]
        # Dirichlet values (start positions, plus dt * velocity when integrated) are inputs too.
        pinned = np.flatnonzero(~free)[[0, -1]]
        checks = []
        for name, step in (("particle_q", 1.0e-4), ("particle_qd", 1.0e-3), ("particle_f", 1.0e-2)):
            array = getattr(state_in, name)
            for node in np.concatenate((nodes, pinned)) if name != "particle_f" else nodes:
                for axis in (0, 2):
                    checks.append((name, array.grad.numpy()[node, axis], central(array, node, axis, step)))
        if material == "pd":
            gradient = model.tet_materials.grad.numpy()
            for element in (0, 11, 37):
                for column, step in ((0, 20.0), (1, 50.0)):
                    numeric = central(model.tet_materials, element, column, step)
                    checks.append(("tet_materials", gradient[element, column], numeric))
        if muscles:
            gradient = control.tet_activations.grad.numpy()
            values = control.tet_activations
            for element in (2, 19):
                original = values.numpy().copy()
                results = []
                for sign in (1.0, -1.0):
                    trial = original.copy()
                    trial[element] += sign * 1.0e-3
                    values.assign(trial)
                    results.append((objective(), float(values.numpy()[element])))
                values.assign(original)
                numeric = (results[0][0] - results[1][0]) / (results[0][1] - results[1][1])
                checks.append(("activation", gradient[element], numeric))
        for name, analytic, numeric in checks:
            with self.subTest(name=name, numeric=numeric):
                self.assertAlmostEqual(analytic, numeric, delta=2.0e-4 * max(abs(numeric), 1.0e-3))

    def test_pd_state_and_material_gradients_match_finite_differences(self):
        self._gradient_check("pd")

    def test_neo_hookean_state_gradients_match_finite_differences(self):
        self._gradient_check("neo_hookean")

    def test_activation_gradients_match_finite_differences(self):
        self._gradient_check("pd", muscles=True)

    def test_integrated_dirichlet_gradients_match_finite_differences(self):
        self._gradient_check("neo_hookean", muscles=True, integrate=True)

    def test_neo_hookean_material_gradients_follow_the_reference_formula(self):
        # The reference differentiates the Neo-Hookean weight with the hyperelastic
        # stress at F (``StressTensor`` ignores the projection), not with the
        # proximal residual that the forward step uses, so these are not finite
        # differences of the step; they are checked against the dense formula.
        model = _grid(requires_grad=True)
        solver = SolverDiffPhD(model, density=DENSITY, material="neo_hookean", **TIGHT)
        dt = 0.004
        state_in = _deformed_state(model, stretch=(1.2, 0.85, 1.05), speed=0.3)
        state_out = model.state(requires_grad=True)
        rng = np.random.default_rng(23)
        q_weight = rng.standard_normal((model.particle_count, 3))
        qd_weight = 0.01 * rng.standard_normal((model.particle_count, 3))
        loss = wp.zeros(1, dtype=wp.float64, device=DEVICE, requires_grad=True)
        tape = wp.Tape()
        with tape:
            solver.step(state_in, state_out, None, None, dt)
            wp.launch(
                _linear_loss,
                model.particle_count,
                inputs=[
                    state_out.particle_q,
                    state_out.particle_qd,
                    wp.array(q_weight, dtype=wp.vec3d, device=DEVICE),
                    wp.array(qd_weight, dtype=wp.vec3d, device=DEVICE),
                ],
                outputs=[loss],
                device=DEVICE,
            )
        tape.backward(loss)
        solver.check_status()
        expected = reference_backward(
            _oracle(model, "neo_hookean"),
            solver._x.numpy(),
            dt,
            q_weight.astype(np.float32).astype(np.float64),
            qd_weight.astype(np.float32).astype(np.float64),
        )["materials"]
        actual = model.tet_materials.grad.numpy()[:, :2]
        np.testing.assert_array_equal(actual[:, 1], 0.0)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-5 * np.abs(expected).max())

    def test_contact_woodbury_projection_matches_dense_reference(self):
        for friction in (0.0, 0.5):
            with self.subTest(friction=friction):
                model = _grid(fix_left=False, requires_grad=True)
                nodes = _surface(model)
                rng = np.random.default_rng(31)
                directions = rng.standard_normal((model.tet_count, 3))
                solver = SolverDiffPhD(
                    model,
                    density=DENSITY,
                    frictional_boundary=("planar", (0.0, 0.0, 1.0, 0.0)),
                    frictional_vertices=nodes,
                    friction=friction,
                    muscle_directions=directions,
                    muscle_stiffness=4.0e4,
                    **TIGHT,
                )
                dt = 0.004
                control = model.control()
                contraction = rng.uniform(0.0, 0.3, model.tet_count).astype(np.float32)
                control.tet_activations = wp.array(contraction, dtype=wp.float32, device=DEVICE, requires_grad=True)
                states = [_lifted(model, 0.001, (0.3, 0.0, -1.0))] + [model.state(requires_grad=True) for _ in range(3)]
                for i in range(2):
                    solver.step(states[i], states[i + 1], control, None, dt)
                state_in, state_out = states[2], states[3]
                for name in ("particle_q", "particle_qd", "particle_f"):
                    getattr(state_in, name).requires_grad = True
                q_weight = rng.standard_normal((model.particle_count, 3))
                qd_weight = 0.01 * rng.standard_normal((model.particle_count, 3))
                loss = wp.zeros(1, dtype=wp.float64, device=DEVICE, requires_grad=True)
                tape = wp.Tape()
                with tape:
                    solver.step(state_in, state_out, control, None, dt)
                    wp.launch(
                        _linear_loss,
                        model.particle_count,
                        inputs=[
                            state_out.particle_q,
                            state_out.particle_qd,
                            wp.array(q_weight, dtype=wp.vec3d, device=DEVICE),
                            wp.array(qd_weight, dtype=wp.vec3d, device=DEVICE),
                        ],
                        outputs=[loss],
                        device=DEVICE,
                    )
                tape.backward(loss)
                solver.check_status()
                multiplier = solver.contact_multipliers.numpy()
                self.assertGreater(np.count_nonzero(multiplier), 0)
                oracle = _oracle(model, "pd", muscle_directions=directions, muscle_stiffness=4.0e4)
                expected = reference_backward(
                    oracle,
                    solver._x.numpy(),
                    dt,
                    q_weight.astype(np.float32).astype(np.float64),
                    qd_weight.astype(np.float32).astype(np.float64),
                    contact=(
                        nodes,
                        multiplier,
                        solver._contact.omega.numpy(),
                        solver._contact.normal.numpy(),
                        friction,
                    ),
                    targets=1.0 - contraction.astype(np.float64),
                )
                pairs = (
                    ("q", state_in.particle_q.grad.numpy()),
                    ("qd", state_in.particle_qd.grad.numpy()),
                    ("f", state_in.particle_f.grad.numpy()),
                    ("materials", model.tet_materials.grad.numpy()[:, :2]),
                    # Both are dL/du for the contraction u = 1 - a.
                    ("activations", control.tet_activations.grad.numpy()),
                )
                for name, actual in pairs:
                    with self.subTest(name=name):
                        reference = expected[name]
                        scale = max(np.abs(reference).max(), 1.0e-30)
                        np.testing.assert_allclose(actual, reference, rtol=0.0, atol=1.0e-5 * scale)

    def test_contact_state_gradients_skip_unused_parameter_projection(self):
        self._contact_gradient_modes(late_enable=False)

    def test_contact_parameter_gradients_can_be_enabled_after_recording(self):
        self._contact_gradient_modes(late_enable=True)

    def _contact_gradient_modes(self, *, late_enable):
        model = _grid(fix_left=True, requires_grad=True)
        rng = np.random.default_rng(32)
        solver = SolverDiffPhD(
            model,
            density=DENSITY,
            frictional_boundary=("planar", (0.0, 0.0, 1.0, 0.0)),
            frictional_vertices=_surface(model),
            friction=0.5,
            muscle_directions=rng.standard_normal((model.tet_count, 3)),
            muscle_stiffness=4.0e4,
            **TIGHT,
        )
        control = model.control()
        control.tet_activations = wp.array(
            rng.uniform(0.0, 0.3, model.tet_count), dtype=wp.float32, device=DEVICE, requires_grad=True
        )
        states = [_lifted(model, 0.001, (0.3, 0.0, -1.0))] + [model.state(requires_grad=True) for _ in range(3)]
        for i in range(2):
            solver.step(states[i], states[i + 1], control, None, 0.004)
        state_in, state_out = states[2], states[3]
        inputs = (state_in.particle_q, state_in.particle_qd, state_in.particle_f)
        for array in inputs:
            array.requires_grad = True
        weights = [
            wp.array(scale * rng.standard_normal((model.particle_count, 3)), dtype=wp.vec3d, device=DEVICE)
            for scale in (1.0, 0.01)
        ]
        loss = wp.zeros(1, dtype=wp.float64, device=DEVICE, requires_grad=True)
        expected = None
        expected_parameters = None
        modes = [(True, True, False), (True, False, late_enable), (False, True, late_enable), (False, False, False)]
        if late_enable:
            modes.append((True, True, True))
        for material_gradient, activation_gradient, enable_after_recording in modes:
            with self.subTest(
                material_gradient=material_gradient,
                activation_gradient=activation_gradient,
                enable_after_recording=enable_after_recording,
            ):
                model.tet_materials.requires_grad = material_gradient and not enable_after_recording
                control.tet_activations.requires_grad = activation_gradient and not enable_after_recording
                loss.zero_()
                tape = wp.Tape()
                with tape:
                    solver.step(state_in, state_out, control, None, 0.004)
                    wp.launch(
                        _linear_loss,
                        model.particle_count,
                        inputs=[state_out.particle_q, state_out.particle_qd, *weights],
                        outputs=[loss],
                        device=DEVICE,
                    )
                self.assertGreater(np.count_nonzero(solver.contact_multipliers.numpy()), 0)
                parameters_need_projection = material_gradient or activation_gradient
                if enable_after_recording:
                    model.tet_materials.requires_grad = material_gradient
                    control.tet_activations.requires_grad = activation_gradient
                with patch.object(solver._adjoint, "_woodbury", wraps=solver._adjoint._woodbury) as woodbury:
                    tape.backward(loss)
                self.assertEqual(woodbury.call_count, int(parameters_need_projection))
                solver.check_status()
                actual = [array.grad.numpy().copy() for array in inputs]
                parameter_actual = [
                    array.grad.numpy().copy() if array.requires_grad else None
                    for array in (model.tet_materials, control.tet_activations)
                ]
                tape.zero()
                # Late-enabled arrays were absent from Tape.record_func's
                # zero list at recording time, so clear those gradients too.
                for array in (model.tet_materials, control.tet_activations):
                    if array.grad is not None:
                        array.grad.zero_()
                if expected is None:
                    expected = actual
                    expected_parameters = parameter_actual
                    self.assertGreater(np.abs(actual[0]).max(), 0.0)
                else:
                    for gradient, reference in zip(actual, expected, strict=True):
                        np.testing.assert_array_equal(gradient, reference)
                    for gradient, reference in zip(parameter_actual, expected_parameters, strict=True):
                        if gradient is not None:
                            scale = max(np.abs(reference).max(), 1.0e-30)
                            np.testing.assert_allclose(gradient, reference, rtol=0.0, atol=2.0e-7 * scale)

    def test_contact_backward_captured_on_first_use_matches_direct(self):
        if not wp.get_device(DEVICE).is_cuda or not wp.is_conditional_graph_supported():
            self.skipTest("CUDA graphs need a CUDA device")
        for material_gradient, late_enable in ((True, False), (False, False), (True, True)):
            with self.subTest(material_gradient=material_gradient, late_enable=late_enable):
                self._captured_contact_backward(material_gradient=material_gradient, late_enable=late_enable)

    def _captured_contact_backward(self, *, material_gradient, late_enable):
        model = _grid(fix_left=True, requires_grad=True)
        model.tet_materials.requires_grad = material_gradient and not late_enable
        solver = SolverDiffPhD(
            model,
            density=DENSITY,
            frictional_boundary=("planar", (0.0, 0.0, 1.0, 0.0)),
            frictional_vertices=_surface(model),
            friction=0.5,
            **TIGHT,
        )
        states = [_lifted(model, 0.001, (0.3, 0.0, -1.0))] + [model.state(requires_grad=True) for _ in range(3)]
        for i in range(2):
            solver.step(states[i], states[i + 1], None, None, 0.004)
        state_in, state_out = states[2], states[3]
        for name in ("particle_q", "particle_qd", "particle_f"):
            getattr(state_in, name).requires_grad = True
        weights = wp.array(np.random.default_rng(3).standard_normal((model.particle_count, 3)), dtype=wp.vec3d)
        loss = wp.zeros(1, dtype=wp.float64, device=DEVICE, requires_grad=True)
        tape = wp.Tape()
        with tape:
            solver.step(state_in, state_out, None, None, 0.004)
            wp.launch(
                _linear_loss,
                model.particle_count,
                inputs=[state_out.particle_q, state_out.particle_qd, weights, weights],
                outputs=[loss],
                device=DEVICE,
            )
        self.assertGreater(np.count_nonzero(solver.contact_multipliers.numpy()), 0)
        if late_enable:
            model.tet_materials.requires_grad = True
        arrays = [state_in.particle_q, state_in.particle_qd, state_in.particle_f]
        if material_gradient:
            arrays.append(model.tet_materials)
        # The first backward pass on this factor is captured: nothing may allocate.
        loss.grad.fill_(1.0)
        with wp.ScopedCapture(device=DEVICE) as capture:
            tape.backward()
        tape.zero()
        if late_enable:
            model.tet_materials.grad.zero_()
        loss.grad.fill_(1.0)
        wp.capture_launch(capture.graph)
        captured = [array.grad.numpy().copy() for array in arrays]
        tape.zero()
        if late_enable:
            model.tet_materials.grad.zero_()
        tape.backward(loss)
        solver.check_status()
        for array, expected in zip(arrays, captured, strict=True):
            np.testing.assert_array_equal(array.grad.numpy(), expected)
        self.assertGreater(np.abs(captured[0]).max(), 0.0)

    def test_recorded_ping_pong_outputs_are_rejected(self):
        model = _grid(requires_grad=True)
        solver = SolverDiffPhD(model, density=DENSITY)
        first, second = model.state(requires_grad=True), model.state(requires_grad=True)
        tape = wp.Tape()
        with tape:
            solver.step(first, second, None, None, 0.004)
            with self.assertRaises(ValueError):
                solver.step(second, first, None, None, 0.004)


if __name__ == "__main__":
    unittest.main()
