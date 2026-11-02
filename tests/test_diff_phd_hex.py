# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Hex8 checks using independently differentiated shape functions and dense solves."""

import os
import unittest

import newton
import numpy as np
import warp as wp

from newton_solver_diff_phd import Mesh, SolverDiffPhD
from tests.reference_oracle import (
    ReferenceStep,
    corotated_projection,
    muscle_projection,
    nh_projection,
    volume_projection,
)

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")


def _fixture(*, pinned=True, requires_grad=False):
    vertices = np.array([[x, y, z] for x in range(3) for y in range(2) for z in range(2)], dtype=float) * 0.1
    elements = np.array([np.arange(8), np.arange(4, 12)], dtype=np.int32)
    mesh = Mesh("two-cubes", vertices, elements)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_particles(
        pos=vertices.tolist(),
        vel=np.zeros_like(vertices).tolist(),
        mass=[0.0 if pinned and i < 4 else 1.0 for i in range(12)],
        radius=[0.001] * 12,
    )
    return mesh, builder.finalize(device=DEVICE, requires_grad=requires_grad)


class HexOracle(ReferenceStep):
    """Dense integration independent of native shared-width gradient formulas."""

    def __init__(self, mesh, materials, free, material="pd", muscles=None, planes=None):
        self.elements = mesh.indices
        self.free = free
        self.n = len(mesh.vertices)
        self.mu, self.lame = materials.T
        self.material = material
        self.nh = (self.mu.mean(), self.lame.mean(), 2 * self.mu.mean())
        self.mass = 1.0  # density 1000 times a full (0.1 metre)^3 cell.
        self.muscles = muscles
        self.planes = planes
        gauss, _ = np.polynomial.legendre.leggauss(2)
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        self.quadrature = []
        for nodes in self.elements:
            samples = []
            for p in np.array([[x, y, z] for x in gauss for y in gauss for z in gauss]):
                dn = np.empty((8, 3))
                for a in range(3):
                    other = [b for b in range(3) if b != a]
                    dn[:, a] = signs[:, a] * np.prod(1 + signs[:, other] * p[other], axis=1) / 8
                jacobian = mesh.vertices[nodes].T @ dn
                samples.append((dn @ np.linalg.inv(jacobian), np.linalg.det(jacobian)))
            self.quadrature.append(samples)

    def matrix(self, dt):
        result = np.diag(np.where(self.free, self.mass / dt**2, 1.0))
        for e, nodes in enumerate(self.elements):
            stiffness = 2 * self.mu[e] + (self.lame[e] if self.material == "pd" else 0)
            for g, weight in self.quadrature[e]:
                block = stiffness * weight * (g @ g.T)
                if self.muscles is not None:
                    directions, stiffnesses = self.muscles
                    fibre = g @ directions[e]
                    block += stiffnesses[e] * weight * np.outer(fibre, fibre)
                for a, i in enumerate(nodes):
                    for b, j in enumerate(nodes):
                        if self.free[i] and self.free[j]:
                            result[i, j] += block[a, b]
        if self.planes is not None:
            nodes, stiffness, _plane = self.planes
            for i in nodes:
                if self.free[i]:
                    result[i, i] += stiffness
        return result

    def projection_rhs(self, x, boundary, targets=None):
        result = np.zeros_like(x)
        for e, nodes in enumerate(self.elements):
            for g, weight in self.quadrature[e]:
                f = x[nodes].T @ g
                fd = boundary[nodes].T @ g
                if self.material == "pd":
                    stress = 2 * self.mu[e] * (corotated_projection(f) - fd)
                    stress += self.lame[e] * (volume_projection(f) - fd)
                else:
                    stress = 2 * self.mu[e] * (nh_projection(f, *self.nh) - fd)
                if self.muscles is not None:
                    directions, stiffnesses = self.muscles
                    m = directions[e]
                    stress += stiffnesses[e] * np.outer(muscle_projection(f, m, targets[e]) - fd @ m, m)
                result[nodes] += weight * (stress @ g.T).T
        if self.planes is not None:
            nodes, stiffness, plane = self.planes
            plane = np.asarray(plane, dtype=float) / np.linalg.norm(plane[:3])
            gap = x[nodes] @ plane[:3] + plane[3]
            projected = x[nodes] - np.minimum(0, gap[:, None]) * plane[:3]
            result[nodes] += stiffness * (projected - boundary[nodes])
        return result


class TestDiffPhDHex(unittest.TestCase):
    def test_affine_patch_uses_all_eight_samples_and_native_cell_mass(self):
        mesh, model = _fixture(pinned=False)
        solver = SolverDiffPhD(model, density=1000, hex_mesh=mesh, hex_materials=[[2e3, 5e3], [2e4, 5e4]])
        self.assertEqual(model.tet_count, 0)
        self.assertAlmostEqual(solver.mean_volume, 0.001, places=16)
        self.assertAlmostEqual(solver.particle_mass, 1.0, places=13)
        f = np.array([[1.05, 0.02, -0.03], [0.06, 0.96, 0.08], [0.0, -0.04, 1.03]])
        state = model.state()
        state.particle_q.assign(mesh.vertices @ f.T + [0.2, -0.3, 0.4])
        determinants = solver.deformation_determinants(state).numpy()
        self.assertEqual(determinants.shape, (2, 8))
        np.testing.assert_allclose(determinants, np.linalg.det(f), atol=6e-7, rtol=0)
        # A non-affine trilinear displacement must produce distinct samples;
        # one central integration point (or a Tet surrogate) cannot pass this.
        q = mesh.vertices.copy()
        q[:, 0] += 2.0 * q[:, 0] * q[:, 1] * q[:, 2]
        state.particle_q.assign(q)
        determinants = solver.deformation_determinants(state).numpy()
        self.assertGreater(np.ptp(determinants[0]), 0.01)

    def test_heterogeneous_step_matches_dense_quadrature_with_muscles_and_planes(self):
        for material in ("pd", "neo_hookean"):
            with self.subTest(material=material):
                mesh, model = _fixture()
                materials = np.array([[2000.0, 4000.0], [20000.0, 40000.0]])
                directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
                stiffness = np.array([1000.0, 300.0])
                targets = np.array([0.0, 0.8])
                # Includes active, inactive and pinned energies; the plane
                # normalization and isotropic diagonal all affect the result.
                planes = (np.array([0, 8, 9]), 120.0, (0.0, 0.0, 2.0, -0.08))
                solver = SolverDiffPhD(
                    model,
                    density=1000,
                    hex_mesh=mesh,
                    hex_materials=materials,
                    material=material,
                    muscle_directions=directions,
                    muscle_stiffness=stiffness,
                    muscle_targets=targets,
                    planar_energies=planes,
                    anderson_window=0,
                    max_iterations=1000,
                    rel_tol=1e-10,
                    abs_tol=1e-12,
                )
                state, out = model.state(), model.state()
                q = mesh.vertices.copy()
                q[4:, 0] *= 1.04
                q[4:, 2] += 0.01 * q[4:, 0]
                state.particle_q.assign(q)
                velocity = np.random.default_rng(4).normal(0, 0.03, q.shape)
                velocity[:4] = 0
                state.particle_qd.assign(velocity)
                force = np.random.default_rng(5).normal(0, 0.2, q.shape)
                state.particle_f.assign(force)
                oracle = HexOracle(
                    mesh, materials, model.particle_inv_mass.numpy() > 0, material, (directions, stiffness), planes
                )
                dt = 0.003
                expected, iterations, converged = oracle.step(
                    state.particle_q.numpy(),
                    state.particle_qd.numpy(),
                    state.particle_f.numpy(),
                    np.zeros(3),
                    dt,
                    max_iterations=1000,
                    abs_tol=1e-12,
                    rel_tol=1e-10,
                    window=0,
                    targets=targets,
                )
                self.assertTrue(converged)
                solver.step(state, out, None, None, dt)
                solver.check_status()
                np.testing.assert_allclose(out.particle_q.numpy(), expected, rtol=0, atol=2e-8)
                self.assertEqual(int(solver.iterations.numpy()[0]), iterations)
                # Test every column of the inverse, including eliminated pins.
                for axis in range(0, model.particle_count, 3):
                    rhs = np.eye(model.particle_count)[:, axis : axis + 3]
                    device_rhs = wp.array(rhs, dtype=wp.vec3d, device=DEVICE)
                    result = wp.empty_like(device_rhs)
                    solver.inverse.apply(device_rhs, result)
                    np.testing.assert_allclose(
                        result.numpy(), np.linalg.solve(oracle.matrix(dt), rhs), atol=2e-14, rtol=2e-12
                    )

    def test_tape_rejected_before_output_mutation(self):
        mesh, model = _fixture(requires_grad=True)
        solver = SolverDiffPhD(model, density=1000, hex_mesh=mesh, hex_materials=[[1000, 2000]] * 2)
        start, out = model.state(), model.state()
        out.particle_q.fill_(wp.vec3(9.0))
        with wp.Tape(), self.assertRaisesRegex(NotImplementedError, "quadrature adjoint"):
            solver.step(start, out, None, None, 0.001)
        np.testing.assert_array_equal(out.particle_q.numpy(), np.full((12, 3), 9.0))
        self.assertIsNone(solver.inverse)

    def test_geometry_and_explicit_data_validation(self):
        mesh, model = _fixture()
        materials = [[1000, 2000]] * 2
        q = mesh.vertices.copy()
        q[-1, 0] += 0.01
        with self.assertRaisesRegex(ValueError, "equally sized axis-aligned"):
            SolverDiffPhD(model, density=1000, hex_mesh=Mesh("distorted", q, mesh.indices), hex_materials=materials)
        with self.assertRaisesRegex(ValueError, "supplied together"):
            SolverDiffPhD(model, density=1000, hex_mesh=mesh)
        with self.assertRaisesRegex(ValueError, "shape"):
            SolverDiffPhD(model, density=1000, hex_mesh=mesh, hex_materials=[[1, 2]])
        with self.assertRaisesRegex(NotImplementedError, "planar_energies"):
            SolverDiffPhD(
                model,
                density=1000,
                hex_mesh=mesh,
                hex_materials=materials,
                frictional_boundary=("planar", (0, 0, 1, 0)),
            )
        with np.errstate(over="ignore"), self.assertRaisesRegex(ValueError, "finite"):
            SolverDiffPhD(
                model,
                density=1000,
                hex_mesh=mesh,
                hex_materials=materials,
                planar_energies=([8], 100.0, (1e200, 0.0, 0.0, 0.0)),
            )

    def test_hex_material_change_rebuilds_with_float64_values(self):
        mesh, model = _fixture()
        materials = wp.array(np.array([[1000.123456789, 2000.0]] * 2), dtype=wp.float64, device=DEVICE)
        solver = SolverDiffPhD(model, density=1000, hex_mesh=mesh, hex_materials=materials)
        self.assertIs(solver.hex_materials, materials)
        start, out = model.state(), model.state()
        q = mesh.vertices.copy()
        q[4:, 0] *= 1.1
        start.particle_q.assign(q)
        solver.step(start, out, None, None, 0.01)
        solver.check_status()
        first = out.particle_q.numpy().copy()
        old_factor = solver.inverse
        materials.assign(materials.numpy() * 5)
        solver.notify_model_changed(0)
        solver.step(start, out, None, None, 0.01)
        solver.check_status()
        self.assertIsNot(old_factor, solver.inverse)
        self.assertGreater(np.linalg.norm(out.particle_q.numpy() - first), 1e-4)
        fresh = SolverDiffPhD(model, density=1000, hex_mesh=mesh, hex_materials=materials)
        expected = model.state()
        fresh.step(start, expected, None, None, 0.01)
        fresh.check_status()
        np.testing.assert_array_equal(out.particle_q.numpy(), expected.particle_q.numpy())

    def test_graph_replay_observes_changed_hex_muscle_targets(self):
        if not wp.get_device(DEVICE).is_cuda or not wp.is_conditional_graph_supported():
            self.skipTest("CUDA conditional graphs required")
        mesh, model = _fixture()
        solver = SolverDiffPhD(
            model,
            density=1000,
            hex_mesh=mesh,
            hex_materials=[[1000, 2000]] * 2,
            muscle_directions=[[1, 0, 0]] * 2,
            muscle_stiffness=2000,
            muscle_targets=[1.0, 1.0],
        )
        start, out = model.state(), model.state()
        solver.step(start, out, None, None, 0.01)
        solver.check_status()
        reference = out.particle_q.numpy().copy()
        with wp.ScopedCapture(device=model.device) as capture:
            solver.step(start, out, None, None, 0.01)
        solver.muscle_targets.zero_()
        wp.capture_launch(capture.graph)
        solver.check_status()
        contracted = out.particle_q.numpy().copy()
        self.assertLess(contracted[8:, 0].mean(), reference[8:, 0].mean() - 1e-4)
        eager = model.state()
        solver.step(start, eager, None, None, 0.01)
        solver.check_status()
        np.testing.assert_array_equal(eager.particle_q.numpy(), contracted)


if __name__ == "__main__":
    unittest.main()
