# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""One heterogeneous Neo-Hookean cube drops onto a frictional plane."""

from pathlib import Path

import newton
import newton.examples
import numpy as np
import warp as wp

from newton_solver_diff_phd import SolverDiffPhD, load_mesh


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0
        self.substeps = args.substeps
        if self.substeps < 1:
            raise ValueError("substeps must be positive")
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        mesh = load_mesh(
            Path(__file__).resolve().parents[1] / "src/newton_solver_diff_phd/examples/assets/drop_cube.usda"
        )[0]
        builder.add_soft_mesh(
            pos=wp.vec3(-0.075, -0.075, 0.04),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.1, 0.0, 0.0),
            scale=1.0,
            vertices=mesh.vertices,
            indices=mesh.indices.flatten(),
            density=500.0,
            k_mu=1000.0,
            k_lambda=2000.0,
            k_damp=0.0,
            add_surface_mesh_edges=False,
            particle_radius=0.002,
        )
        # The two halves have a 3:1 stiffness ratio at the same Poisson ratio.
        centers = np.asarray(builder.particle_q)[np.asarray(builder.tet_indices)].mean(axis=1)
        for index, center in enumerate(centers):
            scale = 3.0 if center[0] > 0.0 else 1.0
            builder.tet_materials[index] = (1000.0 * scale, 2000.0 * scale, 0.0)
        surface_centers = np.asarray(builder.particle_q)[np.asarray(builder.tri_indices)].mean(axis=1)
        builder.tri_color = [(0.8, 0.35, 0.8) if p[0] > 0.0 else (0.1, 0.8, 0.95) for p in surface_centers]
        self.model = builder.finalize()
        self.solver = SolverDiffPhD(
            self.model,
            material="neo_hookean",
            friction=0.4,
            contact_thickness=0.002,
            self_contact=False,
            soft_contact=False,
            tolerance=1e-7,
            mass_damping=2.0,
        )
        self.states = [self.model.state(), self.model.state()]
        self.control = self.model.control()
        self.viewer.set_model(self.model)

    def step(self):
        for _ in range(self.substeps):
            self.states[0].clear_forces()
            self.viewer.apply_forces(self.states[0])
            self.solver.step(self.states[0], self.states[1], self.control, None, self.frame_dt / self.substeps)
            self.states.reverse()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.states[0])
        self.viewer.end_frame()

    def test_final(self):
        q = self.states[0].particle_q.numpy()
        if not np.isfinite(q).all() or np.max(np.abs(q)) > 2.0:
            raise AssertionError("The cube left the finite scene bounds")
        if np.min(q[:, 2]) < -1e-5:
            raise AssertionError("The cube crossed the ground")
        if self.sim_time > 0.1 and np.mean(q[:, 2]) >= 0.115:
            raise AssertionError("The cube did not fall")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--substeps", type=int, default=4)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
