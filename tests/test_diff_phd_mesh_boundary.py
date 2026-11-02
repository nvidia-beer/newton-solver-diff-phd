# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reference ``MeshFrictionalBoundary`` contact between two soft bodies against the dense oracle."""

import unittest

import newton
import numpy as np
import warp as wp

from newton_solver_diff_phd import SolverDiffPhD
from tests.reference_oracle import ReferenceMeshContact
from tests.test_diff_phd_contact import REFERENCE, TIGHT
from tests.test_diff_phd_solver import DENSITY, DEVICE, _oracle

CELL = 0.1
RADIUS = 0.01


def _surface_faces(tets):
    faces = np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]])
    _unique, index, counts = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    return faces[np.sort(index[counts == 1])]


def _blocks(gap):
    """A free block above a block with a pinned bottom, offset half a cell in x."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    common = {"rot": wp.quat_identity(), "vel": wp.vec3(0.0), "cell_x": CELL, "cell_y": CELL, "cell_z": CELL}
    material = {"density": DENSITY, "k_mu": 2.0e4, "k_lambda": 5.0e4, "k_damp": 0.0}
    builder.add_soft_grid(pos=wp.vec3(0.05, 0.0, 0.2 + gap), dim_x=2, dim_y=1, dim_z=1, **common, **material)
    split = builder.particle_count
    tets_upper = len(builder.tet_indices)
    builder.add_soft_grid(pos=wp.vec3(0.0, 0.0, 0.0), dim_x=3, dim_y=2, dim_z=2, fix_bottom=True, **common, **material)
    model = builder.finalize(device=DEVICE)
    tets = model.tet_indices.numpy().reshape(-1, 4)
    return model, split, _surface_faces(tets[:tets_upper]), _surface_faces(tets[tets_upper:])


class TestDiffPhDMeshBoundary(unittest.TestCase):
    def _rollout(
        self,
        *,
        gap=RADIUS + 0.002,
        velocity=(0.0, 0.0, -1.0),
        friction=0.0,
        restitution=0.0,
        steps=6,
        two_sided=False,
        options=TIGHT,
    ):
        """Step solver and oracle together; return the most active contacts in one step.

        One-sided registers only the upper block's vertices against the lower
        block's faces. Two-sided also registers the lower block's free vertices
        against the upper faces, as the reference poke demo does for both
        bodies; every touching point then has two redundant constraints, so the
        positions are unique but the multipliers are not and are not compared.
        """
        model, split, faces_left, faces_right = _blocks(gap)
        free = model.particle_inv_mass.numpy() > 0
        registered = np.unique(np.concatenate([faces_left, faces_right]))
        registered = registered[free[registered]]
        if not two_sided:
            registered = registered[registered < split]
        boundary = ("mesh", (RADIUS, split, faces_left, faces_right))
        solver = SolverDiffPhD(
            model,
            density=DENSITY,
            frictional_boundary=boundary,
            frictional_vertices=registered,
            friction=friction,
            restitution=restitution,
            **options,
        )
        oracle = _oracle(model, "pd")
        contact = ReferenceMeshContact(RADIUS, split, faces_left, faces_right, registered, friction, restitution)
        gravity = model.gravity.numpy()[0].astype(np.float64)
        dt = 0.004
        state = model.state()
        qd = np.zeros((model.particle_count, 3), dtype=np.float32)
        qd[:split] = velocity
        state.particle_qd.assign(qd)
        states = [state] + [model.state() for _ in range(steps)]
        active = 0
        for step in range(steps):
            solver.step(states[step], states[step + 1], None, None, dt)
            expected, iterations, converged = oracle.step(
                states[step].particle_q.numpy(),
                states[step].particle_qd.numpy(),
                states[step].particle_f.numpy(),
                gravity,
                dt,
                contact=contact,
                **options,
            )
            self.assertEqual(int(solver.iterations.numpy()[0]), iterations)
            self.assertEqual(solver.read_status() == 0, converged)
            np.testing.assert_allclose(solver._x.numpy(), expected, rtol=0.0, atol=1.0e-9)
            np.testing.assert_array_equal(
                solver._contact.candidate.numpy().astype(bool), np.isin(registered, contact.nodes)
            )
            if not two_sided:
                # Multipliers are resolved to the convergence gate, positions far below it.
                np.testing.assert_allclose(
                    solver.contact_multipliers.numpy(), contact.registered_multiplier(), rtol=1.0e-5, atol=1.0e-6
                )
            active = max(active, int((contact.multiplier > 0).sum()))
        return active

    def test_soft_blocks_collide_as_in_reference(self):
        # The upper block starts 2 mm outside the contact radius, closing at 1 m/s.
        self.assertGreater(self._rollout(), 0)

    def test_two_sided_candidates_match_reference_positions(self):
        # Step 1 hits the iteration cap in the reference too.
        self.assertGreater(self._rollout(two_sided=True, steps=3, options=REFERENCE), 0)

    def test_sticking_friction_matches_reference(self):
        self.assertGreater(self._rollout(velocity=(0.5, 0.0, -1.0), friction=2.0, options=REFERENCE), 0)

    def test_sliding_friction_matches_reference(self):
        # From the next step on, both stick/slip projections chatter to the
        # iteration cap and amplify round-off differently.
        self.assertGreater(self._rollout(velocity=(0.5, 0.0, -1.0), friction=0.3, steps=1, options=REFERENCE), 0)

    def test_restitution_matches_reference(self):
        self.assertGreater(self._rollout(restitution=0.5), 0)

    def test_distant_vertices_are_filtered_out(self):
        # 10 radii apart nothing is a candidate, and the step equals the contact-free step.
        self.assertEqual(self._rollout(gap=10 * RADIUS, velocity=(0.0, 0.0, 0.0), steps=2, two_sided=True), 0)

    def test_prescribed_motion_does_not_change_start_of_step_candidates(self):
        # The upper block moves from outside the candidate window to within it.
        # The reference admits lower vertices on the following step, not early
        # by looking ahead to the prescribed upper block's endpoint.
        model, split, faces_left, faces_right = _blocks(10 * RADIUS)
        inverse_mass = model.particle_inv_mass.numpy()
        inverse_mass[:split] = 0.0
        model.particle_inv_mass.assign(inverse_mass)
        registered = np.unique(faces_right)
        registered = registered[inverse_mass[registered] > 0.0]
        solver = SolverDiffPhD(
            model,
            density=DENSITY,
            integrate_pinned_velocities=True,
            frictional_boundary=("mesh", (RADIUS, split, faces_left, faces_right)),
            frictional_vertices=registered,
            **REFERENCE,
        )
        oracle = _oracle(model, "pd")
        contact = ReferenceMeshContact(RADIUS, split, faces_left, faces_right, registered)
        dt = 0.004
        state = model.state()
        velocity = np.zeros((model.particle_count, 3), dtype=np.float32)
        velocity[:split, 2] = -9 * RADIUS / dt
        state.particle_qd.assign(velocity)
        states = [state, model.state(), model.state()]
        for step in range(2):
            if step == 1:
                velocity = states[step].particle_qd.numpy()
                velocity[:split] = 0.0
                states[step].particle_qd.assign(velocity)
            solver.step(states[step], states[step + 1], None, None, dt)
            q = states[step].particle_q.numpy().astype(np.float64)
            v = states[step].particle_qd.numpy().astype(np.float64)
            boundary = np.where(oracle.free[:, None], 0.0, q + dt * v)
            expected, iterations, converged = oracle.step(
                q,
                v,
                states[step].particle_f.numpy(),
                model.gravity.numpy()[0],
                dt,
                boundary=boundary,
                contact=contact,
                **REFERENCE,
            )
            np.testing.assert_array_equal(
                solver._contact.candidate.numpy().astype(bool), np.isin(registered, contact.nodes)
            )
            if step == 0:
                self.assertEqual(len(contact.nodes), 0)
            else:
                self.assertGreater(len(contact.nodes), 0)
            self.assertEqual(int(solver.iterations.numpy()[0]), iterations)
            self.assertEqual(solver.read_status() == 0, converged)
            np.testing.assert_allclose(solver._x.numpy(), expected, rtol=0.0, atol=1.0e-9)


if __name__ == "__main__":
    unittest.main()
