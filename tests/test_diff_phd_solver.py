# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Forward checks of the reference DiffPhD step against an independent dense oracle."""

import os
import unittest

import newton
import numpy as np
import warp as wp

from newton_solver_diff_phd import SolverDiffPhD
from newton_solver_diff_phd._src.solvers.diff_phd.exceptions import ConvergenceError
from newton_solver_diff_phd._src.solvers.diff_phd.projections import (
    corotated_projection,
    diagonal_matrix,
    ldlt3_solve,
    muscle_projection,
    nh_prox_stretches,
    unsigned_svd,
    volume_offset,
)
from tests.reference_oracle import ReferenceStep
from tests.reference_oracle import corotated_projection as reference_corotated
from tests.reference_oracle import muscle_projection as reference_muscle
from tests.reference_oracle import nh_projection as reference_nh
from tests.reference_oracle import volume_projection as reference_volume

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
DENSITY = 1000.0


@wp.kernel(enable_backward=False)
def _evaluate_projections(
    f: wp.array[wp.mat33d],
    direction: wp.array[wp.vec3d],
    mu: wp.float64,
    lame: wp.float64,
    penalty: wp.float64,
    sigma: wp.array[wp.vec3d],
    reconstruction: wp.array[wp.mat33d],
    rotation: wp.array[wp.mat33d],
    volume: wp.array[wp.mat33d],
    neo_hookean: wp.array[wp.mat33d],
    fibre: wp.array[wp.vec3d],
):
    i = wp.tid()
    u, s, v = unsigned_svd(f[i])
    sigma[i] = s
    reconstruction[i] = u * diagonal_matrix(s) * wp.transpose(v)
    rotation[i] = corotated_projection(u, v)
    offset, _singular = volume_offset(s)
    volume[i] = u * diagonal_matrix(s + offset) * wp.transpose(v)
    neo_hookean[i] = u * diagonal_matrix(nh_prox_stretches(s, mu, lame, penalty)) * wp.transpose(v)
    fibre[i] = muscle_projection(f[i], direction[i], wp.float64(0.8))


@wp.kernel(enable_backward=False)
def _evaluate_ldlt(a: wp.array[wp.mat33d], b: wp.array[wp.vec3d], x: wp.array[wp.vec3d]):
    i = wp.tid()
    x[i] = ldlt3_solve(a[i], b[i])


def _grid(device=DEVICE, *, fix_left=True, gravity=-9.81, contrast=1.0, cell=0.1, requires_grad=False):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, 1.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=3,
        dim_y=2,
        dim_z=2,
        cell_x=cell,
        cell_y=cell,
        cell_z=cell,
        density=DENSITY,
        k_mu=2.0e4,
        k_lambda=5.0e4,
        k_damp=0.0,
        fix_left=fix_left,
    )
    model = builder.finalize(device=device, requires_grad=requires_grad)
    if contrast != 1.0:
        # Stiffen the half of the elements away from the pinned face.
        q = model.particle_q.numpy()
        tets = model.tet_indices.numpy().reshape(-1, 4)
        stiff = q[tets].mean(axis=1)[:, 0] > q[:, 0].mean()
        materials = model.tet_materials.numpy()
        materials[stiff, :2] *= contrast
        model.tet_materials.assign(materials)
    return model


def _oracle(model, material, **muscles):
    materials = model.tet_materials.numpy().astype(np.float64)
    return ReferenceStep(
        model.tet_indices.numpy().reshape(-1, 4),
        model.tet_poses.numpy(),
        materials[:, 0],
        materials[:, 1],
        DENSITY,
        model.particle_inv_mass.numpy() > 0,
        material,
        **muscles,
    )


def _deformed_state(model, seed=3, stretch=(1.3, 0.8, 1.1), speed=0.5):
    rng = np.random.default_rng(seed)
    state = model.state()
    q = model.particle_q.numpy()
    center = q.mean(axis=0)
    free = model.particle_inv_mass.numpy() > 0
    deformed = np.where(free[:, None], center + (q - center) * np.asarray(stretch), q)
    deformed[free] += 0.01 * rng.standard_normal((free.sum(), 3))
    velocity = np.where(free[:, None], speed * rng.standard_normal(q.shape), 0.0)
    state.particle_q.assign(deformed.astype(np.float32))
    state.particle_qd.assign(velocity.astype(np.float32))
    return state


class TestDiffPhDSolver(unittest.TestCase):
    def test_projections_match_independent_numpy_reference(self):
        rng = np.random.default_rng(7)
        f = [np.eye(3), np.diag([1.4, 0.7, 1.1]), np.diag([1.0, 1.0, -0.5]), 1.2 * np.eye(3)]
        f += [np.eye(3) + 0.4 * rng.standard_normal((3, 3)) for _ in range(12)]
        f += [rng.standard_normal((3, 3)) for _ in range(8)]
        f = np.array([m if abs(np.linalg.det(m)) > 1.0e-2 else m + np.eye(3) for m in f])
        directions = rng.standard_normal((len(f), 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        mu, lame = 3.0e4, 8.0e4
        outputs = [
            wp.empty(len(f), dtype=dtype, device=DEVICE) for dtype in (wp.vec3d,) + (wp.mat33d,) * 4 + (wp.vec3d,)
        ]
        wp.launch(
            _evaluate_projections,
            len(f),
            inputs=[
                wp.array(f, dtype=wp.mat33d, device=DEVICE),
                wp.array(directions, dtype=wp.vec3d, device=DEVICE),
                mu,
                lame,
                2.0 * mu,
            ],
            outputs=outputs,
            device=DEVICE,
        )
        sigma, reconstruction, rotation, volume, neo_hookean, fibre = (array.numpy() for array in outputs)
        for i, matrix in enumerate(f):
            reference_sigma = np.linalg.svd(matrix, compute_uv=False)
            np.testing.assert_allclose(sigma[i], reference_sigma, rtol=1.0e-13, atol=1.0e-14)
            np.testing.assert_allclose(reconstruction[i], matrix, rtol=0.0, atol=1.0e-13)
            np.testing.assert_allclose(rotation[i], reference_corotated(matrix), rtol=0.0, atol=1.0e-12)
            self.assertAlmostEqual(np.linalg.det(rotation[i]), np.sign(np.linalg.det(matrix)), places=12)
            if reference_sigma.min() > 1.0e-3:
                np.testing.assert_allclose(volume[i], reference_volume(matrix), rtol=0.0, atol=1.0e-12)
                # The reference fixed point stops on its step size, so det = 1 holds approximately.
                self.assertAlmostEqual(abs(np.linalg.det(volume[i])), 1.0, delta=1.0e-9)
            np.testing.assert_allclose(neo_hookean[i], reference_nh(matrix, mu, lame, 2.0 * mu), rtol=0.0, atol=1.0e-9)
            np.testing.assert_allclose(fibre[i], reference_muscle(matrix, directions[i], 0.8), rtol=0.0, atol=1.0e-13)

    def test_reflected_rest_shape_remains_finite_and_stationary(self):
        # The reference uses an unsigned SVD: a reflection is another rest
        # shape of these projections. A signed replacement sends a negative
        # stretch into the volume projection's fractional power and produces NaN.
        model = _grid(fix_left=False, gravity=0.0, cell=0.125)
        state_in, state_out = model.state(), model.state()
        reflected = state_in.particle_q.numpy()
        reflected[:, 0] *= -1.0
        state_in.particle_q.assign(reflected)
        for material in ("pd", "neo_hookean"):
            with self.subTest(material=material):
                solver = SolverDiffPhD(model, density=DENSITY, material=material, max_iterations=20)
                solver.step(state_in, state_out, None, None, 0.001)
                solver.check_status()
                self.assertTrue(np.all(np.isfinite(state_out.particle_q.numpy())))
                np.testing.assert_allclose(state_out.particle_q.numpy(), reflected, rtol=0.0, atol=1.0e-12)
                np.testing.assert_allclose(state_out.particle_qd.numpy(), 0.0, rtol=0.0, atol=1.0e-10)

    def test_degenerate_deformations_follow_eigen_jacobi_svd(self):
        # A flat tetrahedron (exact zero row) and a collapsed axis (diag(1, 0, 1)).
        f = np.array([[[0.3, 0.5, 1.2], [1.0, 0.5, 0.3], [0.0, 0.0, 0.0]], np.diag([1.0, 0.0, 1.0])])
        out = [wp.empty(len(f), dtype=dtype, device=DEVICE) for dtype in (wp.vec3d,) + (wp.mat33d,) * 4 + (wp.vec3d,)]
        wp.launch(
            _evaluate_projections,
            len(f),
            inputs=[
                wp.array(f, dtype=wp.mat33d, device=DEVICE),
                wp.array(np.eye(3)[:2], dtype=wp.vec3d, device=DEVICE),
                3.0e4,
                8.0e4,
                6.0e4,
            ],
            outputs=out,
            device=DEVICE,
        )
        sigma, reconstruction, rotation = out[0].numpy(), out[1].numpy(), out[2].numpy()
        for i, matrix in enumerate(f):
            np.testing.assert_allclose(reconstruction[i], matrix, atol=1.0e-14)
            np.testing.assert_allclose(rotation[i] @ rotation[i].T, np.eye(3), atol=1.0e-14)
            np.testing.assert_allclose(sigma[i], np.linalg.svd(matrix, compute_uv=False), atol=1.0e-14)
        # Eigen keeps U = V = [e0, e2, e1] for diag(1, 0, 1), so R = I.
        np.testing.assert_allclose(rotation[1], np.eye(3), atol=1.0e-15)

    def test_pivoted_ldlt_solves_definite_and_indefinite_systems(self):
        rng = np.random.default_rng(11)
        matrices = []
        for diagonal in ([4.0, 2.0, 1.0], [5.0, -2.0, 0.5], [1.0e-3, 3.0, -7.0]):
            q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
            matrices.append(q @ np.diag(diagonal) @ q.T)
        rhs = rng.standard_normal((len(matrices), 3))
        out = wp.empty(len(matrices), dtype=wp.vec3d, device=DEVICE)
        wp.launch(
            _evaluate_ldlt,
            len(matrices),
            inputs=[wp.array(matrices, dtype=wp.mat33d, device=DEVICE), wp.array(rhs, dtype=wp.vec3d, device=DEVICE)],
            outputs=[out],
            device=DEVICE,
        )
        for matrix, b, x in zip(matrices, rhs, out.numpy(), strict=True):
            np.testing.assert_allclose(x, np.linalg.solve(matrix, b), rtol=1.0e-10, atol=1.0e-12)

    def test_rigid_free_fall_is_the_inertial_target(self):
        # Binary-fraction geometry makes the float32 rest shape exact, so an
        # unstressed translation has F = I and every projection returns I.
        model = _grid(fix_left=False, cell=0.125)
        solver = SolverDiffPhD(model, density=DENSITY)
        state_in, state_out = model.state(), model.state()
        velocity = np.tile([0.25, -0.125, 0.5], (model.particle_count, 1)).astype(np.float32)
        state_in.particle_qd.assign(velocity)
        dt = 2.0**-7
        solver.step(state_in, state_out, None, None, dt)
        solver.check_status()
        gravity = model.gravity.numpy()[0].astype(np.float64)
        q0 = state_in.particle_q.numpy().astype(np.float64)
        expected = q0 + dt * velocity + dt**2 * gravity
        np.testing.assert_allclose(solver._x.numpy(), expected, rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(state_out.particle_qd.numpy(), velocity + dt * gravity, rtol=0.0, atol=2.0e-6)

    def _compare_with_oracle(self, material, contrast, *, fixed_iterations):
        model = _grid(contrast=contrast)
        if fixed_iterations:
            options = {"max_iterations": 12, "abs_tol": 0.0, "rel_tol": 0.0}
        else:
            options = {"max_iterations": 500, "abs_tol": 1.0e-11, "rel_tol": 1.0e-10}
        solver = SolverDiffPhD(model, density=DENSITY, material=material, **options)
        state_in, state_out = _deformed_state(model), model.state()
        dt = 0.004
        solver.step(state_in, state_out, None, None, dt)
        if fixed_iterations:
            with self.assertRaises(ConvergenceError):
                solver.check_status()
        else:
            solver.check_status()
        expected, iterations, converged = _oracle(model, material).step(
            state_in.particle_q.numpy(),
            state_in.particle_qd.numpy(),
            state_in.particle_f.numpy(),
            model.gravity.numpy()[0].astype(np.float64),
            dt,
            **options,
        )
        self.assertEqual(converged, not fixed_iterations)
        if fixed_iterations:
            self.assertEqual(int(solver.iterations.numpy()[0]), iterations)
        scale = np.abs(expected).max()
        np.testing.assert_allclose(solver._x.numpy(), expected, rtol=0.0, atol=1.0e-10 * scale)

    def test_forward_matches_dense_reference_iterates(self):
        for material in ("pd", "neo_hookean"):
            with self.subTest(material=material):
                self._compare_with_oracle(material, 1.0, fixed_iterations=True)

    def test_heterogeneous_pinned_contrast_matches_dense_reference(self):
        for material in ("pd", "neo_hookean"):
            for contrast in (10.0, 100.0):
                with self.subTest(material=material, contrast=contrast):
                    self._compare_with_oracle(material, contrast, fixed_iterations=False)

    def test_muscle_activation_matches_dense_reference(self):
        model = _grid()
        rng = np.random.default_rng(5)
        directions = rng.standard_normal((model.tet_count, 3))
        stiffness = np.where(np.arange(model.tet_count) % 2 == 0, 4.0e4, 0.0)
        options = {"max_iterations": 500, "abs_tol": 1.0e-11, "rel_tol": 1.0e-10}
        solver = SolverDiffPhD(
            model, density=DENSITY, muscle_directions=directions, muscle_stiffness=stiffness, **options
        )
        control = model.control()
        contraction = np.where(stiffness > 0, 0.25, 0.0).astype(np.float32)
        control.tet_activations = wp.array(contraction, dtype=wp.float32, device=DEVICE)
        state_in, state_out = _deformed_state(model, speed=0.0), model.state()
        solver.step(state_in, state_out, control, None, 0.004)
        solver.check_status()
        oracle = _oracle(model, "pd", muscle_directions=directions, muscle_stiffness=stiffness)
        expected, _iterations, converged = oracle.step(
            state_in.particle_q.numpy(),
            state_in.particle_qd.numpy(),
            state_in.particle_f.numpy(),
            model.gravity.numpy()[0].astype(np.float64),
            0.004,
            targets=1.0 - contraction.astype(np.float64),
            **options,
        )
        self.assertTrue(converged)
        np.testing.assert_allclose(solver._x.numpy(), expected, rtol=0.0, atol=1.0e-10 * np.abs(expected).max())

    def test_dirichlet_particles_are_fixed_or_prescribed(self):
        model = _grid()
        pinned = model.particle_inv_mass.numpy() == 0
        velocity = np.zeros((model.particle_count, 3), dtype=np.float32)
        velocity[pinned] = [0.0, 0.5, 0.0]
        for integrate in (False, True):
            with self.subTest(integrate=integrate):
                solver = SolverDiffPhD(model, density=DENSITY, integrate_pinned_velocities=integrate)
                state_in, state_out = model.state(), model.state()
                state_in.particle_qd.assign(velocity)
                solver.step(state_in, state_out, None, None, 0.01)
                solver.check_status()
                q0 = state_in.particle_q.numpy()[pinned]
                expected = q0 + 0.01 * velocity[pinned] if integrate else q0
                np.testing.assert_allclose(state_out.particle_q.numpy()[pinned], expected, atol=1.0e-7)
                np.testing.assert_allclose(
                    state_out.particle_qd.numpy()[pinned], velocity[pinned] if integrate else 0.0, atol=1.0e-5
                )

    def test_reference_mass_uses_density_and_mean_volume(self):
        model = _grid()
        solver = SolverDiffPhD(model, density=DENSITY)
        rest = model.tet_poses.numpy().astype(np.float64)
        mean_volume = np.mean(np.abs(1.0 / np.linalg.det(rest)) / 6.0)
        self.assertAlmostEqual(solver.mean_volume, mean_volume, delta=1.0e-15)
        self.assertAlmostEqual(solver.particle_mass, DENSITY * mean_volume, delta=1.0e-12)

    @unittest.skipUnless(DEVICE.startswith("cuda"), "CUDA graphs need a CUDA device")
    def test_step_captured_in_an_outer_graph_matches_direct_steps(self):
        if not wp.is_conditional_graph_supported():
            self.skipTest("conditional graph nodes are unavailable")
        model = _grid()
        direct = SolverDiffPhD(model, density=DENSITY)
        captured = SolverDiffPhD(model, density=DENSITY)
        state_in = _deformed_state(model)
        direct_out, captured_out = model.state(), model.state()
        direct.step(state_in, direct_out, None, None, 0.004)
        # The first step builds the factor; capture the next identical step.
        captured.step(state_in, captured_out, None, None, 0.004)
        captured_out.particle_q.zero_()
        with wp.ScopedCapture(device=DEVICE) as capture:
            captured.step(state_in, captured_out, None, None, 0.004)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(captured_out.particle_q.numpy(), direct_out.particle_q.numpy())
        np.testing.assert_array_equal(captured.iterations.numpy(), direct.iterations.numpy())
        captured.check_status()

    def test_cpu_graph_capture_replays_the_whole_loop(self):
        if DEVICE != "cpu":
            self.skipTest("CPU capture")
        model = _grid(device="cpu")
        direct = SolverDiffPhD(model, density=DENSITY)
        captured = SolverDiffPhD(model, density=DENSITY)
        state_in = _deformed_state(model)
        direct_out, captured_out = model.state(), model.state()
        direct.step(state_in, direct_out, None, None, 0.004)
        captured.step(state_in, captured_out, None, None, 0.004)
        captured_out.particle_q.zero_()
        with wp.ScopedCapture(device="cpu") as capture:
            captured.step(state_in, captured_out, None, None, 0.004)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(captured_out.particle_q.numpy(), direct_out.particle_q.numpy())
        np.testing.assert_array_equal(captured.iterations.numpy(), direct.iterations.numpy())

    def test_changed_convergence_settings_apply_to_the_next_step(self):
        model = _grid()
        solver = SolverDiffPhD(model, density=DENSITY)
        state_in, state_out = _deformed_state(model), model.state()
        solver.step(state_in, state_out, None, None, 0.004)
        solver.max_iterations, solver.abs_tol, solver.rel_tol = 3, 0.0, 0.0
        solver.step(state_in, state_out, None, None, 0.004)
        self.assertEqual(int(solver.iterations.numpy()[0]), 3)
        with self.assertRaises(ConvergenceError):
            solver.check_status()

    def test_status_reports_nonconvergence_until_checked(self):
        model = _grid()
        solver = SolverDiffPhD(model, density=DENSITY, max_iterations=2, abs_tol=0.0, rel_tol=0.0)
        state_in, state_out = _deformed_state(model), model.state()
        solver.step(state_in, state_out, None, None, 0.004)
        self.assertEqual(int(solver.iterations.numpy()[0]), 2)
        with self.assertRaises(ConvergenceError):
            solver.check_status()
        solver.check_status()

    def test_invalid_configuration_is_rejected(self):
        model = _grid()
        for kwargs in (
            {"density": 0.0},
            {"density": DENSITY, "material": "linear"},
            {"density": DENSITY, "anderson_window": 9},
            {"density": DENSITY, "max_iterations": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SolverDiffPhD(model, **kwargs)
        # Self/soft contact, damping and the previous contact options were removed.
        for name in (
            "self_contact",
            "soft_contact",
            "mass_damping",
            "ground_plane",
            "contact_thickness",
            "tolerance",
            "signed_svd",
            "contact_restart",
            "converged_contact",
            "frictional_face_vertices",
        ):
            with self.subTest(removed=name), self.assertRaises(TypeError):
                SolverDiffPhD(model, density=DENSITY, **{name: 1.0})
        solver = SolverDiffPhD(model, density=DENSITY)
        state_in, state_out = model.state(), model.state()
        with self.assertRaises(NotImplementedError):
            solver.step(state_in, state_out, None, newton.CollisionPipeline(model).contacts(), 0.01)
        with self.assertRaises(ValueError):
            solver.step(state_in, state_in, None, None, 0.01)
        damped = _grid()
        materials = damped.tet_materials.numpy()
        materials[:, 2] = 1.0
        damped.tet_materials.assign(materials)
        with self.assertRaises(ValueError):
            SolverDiffPhD(damped, density=DENSITY).step(damped.state(), damped.state(), None, None, 0.01)


if __name__ == "__main__":
    unittest.main()
