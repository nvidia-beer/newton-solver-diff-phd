# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Mechanics and converged-adjoint checks for the DiffPhD solver."""

import os
import unittest

import newton
import numpy as np
import warp as wp

from newton_solver_diff_phd import SolverDiffPhD


@wp.kernel
def _linear_state_loss(
    q: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
    q_weight: wp.array[wp.vec3],
    velocity_weight: wp.array[wp.vec3],
    loss: wp.array[wp.float64],
):
    i = wp.tid()
    value = wp.dot(wp.vec3d(q[i]), wp.vec3d(q_weight[i]))
    value += wp.dot(wp.vec3d(velocity[i]), wp.vec3d(velocity_weight[i]))
    wp.atomic_add(loss, 0, value)


def _model(device, *, two_tets=False, pins=(), gravity=0.0, requires_grad=False, origin=(0.0, 0.0, 0.0)):
    points = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.4]])
    if two_tets:
        points = np.vstack([points, [0.4, 0.4, 0.4]])
    points += np.asarray(origin)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
    for i, point in enumerate(points):
        builder.add_particle(wp.vec3(*point), wp.vec3(), mass=0.0 if i in pins else 1.0, radius=0.0)
    builder.add_tetrahedron(0, 1, 2, 3, k_mu=1200.0, k_lambda=2400.0, k_damp=0.0)
    if two_tets:
        builder.add_tetrahedron(1, 2, 3, 4, k_mu=120000.0, k_lambda=240000.0, k_damp=0.0)
    return builder.finalize(device=device, requires_grad=requires_grad)


def _step(solver, model, q=None, velocity=None, force=None, *, dt=0.02, requires_grad=False):
    state_in = model.state(requires_grad=requires_grad)
    state_out = model.state(requires_grad=requires_grad)
    if q is not None:
        state_in.particle_q.assign(np.asarray(q, dtype=np.float32))
    if velocity is not None:
        state_in.particle_qd.assign(np.asarray(velocity, dtype=np.float32))
    if force is not None:
        state_in.particle_f.assign(np.asarray(force, dtype=np.float32))
    solver.step(state_in, state_out, model.control(), None, dt)
    return state_in, state_out


def _nh_gradient(reference, q, indices, materials):
    """Independent NumPy physical NH residual, including rest volumes."""
    gradient = np.zeros_like(q, dtype=np.float64)
    for tet, material in zip(indices, materials, strict=True):
        dm = (reference[tet[1:]] - reference[tet[0]]).T
        inverse_dm = np.linalg.inv(dm)
        f = (q[tet[1:]] - q[tet[0]]).T @ inverse_dm
        mu, lame = material[:2]
        stress = mu * f + (lame * np.log(np.linalg.det(f)) - mu) * np.linalg.inv(f).T
        shape_gradients = np.vstack([-inverse_dm.sum(axis=0), inverse_dm])
        values = abs(np.linalg.det(dm)) / 6.0 * (stress @ shape_gradients.T).T
        np.add.at(gradient, tet, values)
    return gradient


class TestDiffPhDSolver(unittest.TestCase):
    def test_public_api_rejects_unsupported_native_contacts_and_shapes(self):
        device = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
        model = _model(device)
        solver = SolverDiffPhD(model, ground_plane=None)
        first, second = model.state(), model.state()
        with self.assertRaisesRegex(NotImplementedError, "own swept contact"):
            solver.step(first, second, None, newton.CollisionPipeline(model).contacts(), 0.01)
        solver.step(first, second, None, None, 0.01, observables=solver.observables([]))
        builder = newton.ModelBuilder()
        for point in model.particle_q.numpy():
            builder.add_particle(wp.vec3(*point), wp.vec3(), mass=1.0)
        builder.add_tetrahedron(0, 1, 2, 3, k_mu=1200.0, k_lambda=2400.0, k_damp=0.0)
        builder.add_shape_box(body=-1, hx=0.1, hy=0.1, hz=0.1)
        with self.assertRaisesRegex(ValueError, "native rigid collision shapes"):
            SolverDiffPhD(builder.finalize(device=device))

    def test_inverted_input_is_rejected_before_constitutive_evaluation(self):
        for device in [wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))]:
            for material in ("pd", "neo_hookean"):
                with self.subTest(device=device, material=material), wp.ScopedDevice(device):
                    model = _model(device)
                    solver = SolverDiffPhD(model, material=material, ground_plane=None)
                    q = model.particle_q.numpy()
                    q[3, 2] = -0.1
                    with self.assertRaisesRegex(ValueError, "inverted|collapsed"):
                        _step(solver, model, q)

    def test_freefall_mass_damping_and_timestep_change(self):
        gravity = np.array([0.0, 0.0, -9.81])
        damping = 3.7
        for device in [wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))]:
            for material in ("pd", "neo_hookean"):
                with self.subTest(device=device, material=material), wp.ScopedDevice(device):
                    model = _model(device, gravity=-9.81)
                    solver = SolverDiffPhD(
                        model,
                        material=material,
                        ground_plane=None,
                        mass_damping=damping,
                        tolerance=1.0e-9,
                        max_iterations=400,
                    )
                    q = model.particle_q.numpy().astype(np.float64)
                    velocity = np.tile([0.13, -0.08, 0.22], (4, 1))
                    force = np.tile([0.2, 0.1, -0.3], (4, 1))
                    last_dt = None
                    last_inverse = None
                    for dt in (0.01, 0.01, 0.035, 0.035, 0.01):
                        actual_in, actual_out = _step(solver, model, q, velocity, force, dt=dt)
                        if dt == last_dt:
                            self.assertIs(solver.inverse, last_inverse)
                        else:
                            self.assertIsNot(solver.inverse, last_inverse)
                        last_dt, last_inverse = dt, solver.inverse
                        self.assertLessEqual(solver.last_step_info.residual, solver.tolerance)
                        q0 = actual_in.particle_q.numpy().astype(np.float64)
                        v0 = actual_in.particle_qd.numpy().astype(np.float64)
                        expected_velocity = (v0 + dt * (gravity + force)) / (1.0 + damping * dt)
                        expected_q = q0 + dt * expected_velocity
                        np.testing.assert_allclose(actual_out.particle_q.numpy(), expected_q, rtol=0.0, atol=1.0e-7)
                        np.testing.assert_allclose(
                            actual_out.particle_qd.numpy(), expected_velocity, rtol=1.0e-5, atol=2.0e-6
                        )
                        q = actual_out.particle_q.numpy()
                        velocity = actual_out.particle_qd.numpy()

    def test_pinned_heterogeneous_neo_hookean_matches_physical_residual(self):
        dt = 0.025
        damping = 1.2
        for device in [wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))]:
            with self.subTest(device=device), wp.ScopedDevice(device):
                model = _model(device, two_tets=True, pins=(0,))
                solver = SolverDiffPhD(
                    model,
                    material="neo_hookean",
                    ground_plane=None,
                    mass_damping=damping,
                    tolerance=1.0e-9,
                    max_iterations=800,
                )
                reference = model.particle_q.numpy().astype(np.float64)
                q = reference.copy()
                q[3] += [0.01, -0.006, 0.018]
                q[4] += [-0.007, 0.008, -0.003]
                force = np.zeros_like(q)
                force[3] = [1.4, -0.2, -0.7]
                state_in, state_out = _step(solver, model, q, force=force, dt=dt)
                result = state_out.particle_q.numpy().astype(np.float64)
                q0 = state_in.particle_q.numpy().astype(np.float64)
                self.assertLessEqual(solver.last_step_info.residual, solver.tolerance)
                np.testing.assert_array_equal(result[0], q0[0])
                np.testing.assert_array_equal(state_out.particle_qd.numpy()[0], 0.0)
                elastic = _nh_gradient(reference, result, model.tet_indices.numpy(), model.tet_materials.numpy())
                residual = (result - q0) / dt**2 + damping * (result - q0) / dt + elastic - force
                # The exported state is float32, which limits measurable force
                # balance after multiplication by the stiff tangent.
                self.assertLess(np.max(np.abs(residual[1:])), 0.025)
                self.assertGreater(np.linalg.norm(result[1:] - q0[1:]), 1.0e-4)

    def test_ground_support_has_no_accumulated_penetration(self):
        thickness = 0.005
        for device in [wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))]:
            with self.subTest(device=device), wp.ScopedDevice(device):
                model = _model(device, gravity=-9.81, origin=(0.0, 0.0, thickness))
                solver = SolverDiffPhD(
                    model,
                    ground_plane=(0.0, 0.0, 1.0, 0.0),
                    friction=0.4,
                    contact_thickness=thickness,
                    mass_damping=8.0,
                    tolerance=1.0e-8,
                    max_iterations=400,
                )
                state_in, state_out = model.state(), model.state()
                for _ in range(20):
                    state_in.clear_forces()
                    solver.step(state_in, state_out, model.control(), None, 0.02)
                    q = state_out.particle_q.numpy()
                    self.assertTrue(np.isfinite(q).all())
                    self.assertGreaterEqual(q[:, 2].min(), thickness - 3.0e-6)
                    state_in, state_out = state_out, state_in
                self.assertLess(np.max(np.abs(state_in.particle_qd.numpy()[:3, 2])), 2.0e-4)

    def test_two_soft_bodies_do_not_pass_through_each_other(self):
        points = np.array(
            [
                [-0.3, -0.2, 0.0],
                [0.3, -0.2, 0.0],
                [0.0, 0.3, 0.0],
                [0.0, 0.0, -0.35],
                [-0.3, -0.2, 0.08],
                [0.3, -0.2, 0.08],
                [0.0, 0.3, 0.08],
                [0.0, 0.0, 0.43],
            ]
        )
        for device in [wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))]:
            with self.subTest(device=device), wp.ScopedDevice(device):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                for point in points:
                    builder.add_particle(wp.vec3(*point), wp.vec3(), mass=1.0, radius=0.0)
                builder.add_tetrahedron(0, 2, 1, 3, k_mu=2000.0, k_lambda=4000.0, k_damp=0.0)
                builder.add_tetrahedron(4, 5, 6, 7, k_mu=2000.0, k_lambda=4000.0, k_damp=0.0)
                model = builder.finalize(device=device)
                solver = SolverDiffPhD(
                    model,
                    ground_plane=None,
                    friction=0.0,
                    contact_thickness=0.005,
                    tolerance=1.0e-8,
                    max_iterations=400,
                )
                velocity = np.zeros_like(points)
                velocity[:4, 2] = 0.5
                velocity[4:, 2] = -0.5
                _, result = _step(solver, model, velocity=velocity, dt=0.1)
                q = result.particle_q.numpy()
                self.assertGreaterEqual(q[4:, 2].min() - q[:4, 2].max(), 0.005 - 2.0e-5)
                np.testing.assert_allclose(result.particle_qd.numpy().sum(axis=0), 0.0, atol=2.0e-5)


if __name__ == "__main__":
    unittest.main()
