# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Scene parameters translated from DiffPD (https://github.com/mit-gfx/diff_pd_public,
# commit 1594aa0) python/example/duck_3d.py and python/py_diff_pd/env/duck_env_3d.py,
# Copyright (c) 2021 MIT Graphics Group, MIT License. See THIRD_PARTY_NOTICES.md.

"""SolverDiffPD: slide a soft duck down a curved ramp onto the ground toward a target.

Run ``python -m newton_solver_diff_phd.examples softbody_diff_pd_duck``. The scene
is the forward rollout of diff_pd_public ``duck_3d.py`` with its initial contact
parameters: the duck enters the reference ``arc_contact`` ramp at 10 m/s and
leaves it onto the ``planar_contact`` ground. The red point is the optimizer's
target for the duck's center. Forward simulation only; no gradients yet.
"""

from pathlib import Path

import newton
import newton.examples
import numpy as np
import warp as wp

from newton_solver_diff_phd import SolverDiffPD, load_mesh

ASSET = Path(__file__).parent / "assets" / "duck.usda"

# duck_3d.py
CENTER = np.array([-0.6, 0.0, 7.7])
START_DEGREE = 60.0
END_DEGREE = 100.0
INITIAL_DEGREE = 70.0
RADIUS = 7.5
INITIAL_SPEED = 10.0
TARGET = np.array([-7.85, 0.0, 0.315])
STATE_FORCE_PARAMETERS = (0.0, 0.0, -9.81, 1.0e5, 0.025, 1.0e4)
DT = 5.0e-3
FRAMES = 200
# duck_3d.py pd_opt (thread_ct and verbose do not apply).
PD_OPTIONS = {"max_pd_iter": 4000, "max_ls_iter": 10, "abs_tol": 1.0e-9, "rel_tol": 1.0e-4, "bfgs_history_size": 10}

# duck_env_3d.py
YOUNGS = 4.0e7
POISSON = 0.45
DENSITY = 1.0e3
PLANAR_CONTACT = (0.0, 0.0, 1.0, 0.0, 3, 1.0e3, 0.1, 1.0e4)
OBSTACLE_SEGMENTS = 50


def duck_scene(rest: np.ndarray) -> dict:
    """``DuckEnv3d`` energies, state forces and the ``duck_3d.py`` initial state for rest vertices [m]."""
    start, end = np.deg2rad(START_DEGREE), np.deg2rad(END_DEGREE)
    lame = YOUNGS * POISSON / ((1 + POISSON) * (1 - 2 * POISSON))
    kn, kf, mu = STATE_FORCE_PARAMETERS[3:]
    # Reference bug B2: the arc start vector uses sin(end_rad), not sin(start_rad),
    # so the contact window spans 63.08-103.08 degrees instead of 60-100.
    c_start, s_start = np.cos(start), np.sin(end)
    arc = [*CENTER, 0, 1, 0, c_start, 0, -s_start, RADIUS, end - start, 3, kn, kf, mu]
    # Reference bug B1: ``kn, kf, mu = state_force_parameters[3:]`` rebinds the
    # Lame mu, so the corotated weight is 2 * (arc friction coefficient) = 2e4 Pa
    # instead of 2 * E / (2 (1 + nu)) = 2.76e7 Pa.
    corotated = 2 * mu
    # Place the rest mesh on the arc at INITIAL_DEGREE.
    x_offset = np.max([-np.min(rest[:, 0]), np.max(rest[:, 0])])
    radius_adjusted = np.sqrt(RADIUS**2 - x_offset**2)
    c, s = np.cos(np.deg2rad(INITIAL_DEGREE)), np.sin(np.deg2rad(INITIAL_DEGREE))
    q0 = rest @ np.array([[s, 0, -c], [0, 1, 0], [c, 0, s]]).T + CENTER + radius_adjusted * np.array([c, 0, -s])
    v0 = np.zeros_like(rest) + np.array([-s, 0, -c]) * INITIAL_SPEED
    return {
        "corotated_stiffness": corotated,
        "volume_stiffness": lame,
        "gravity": STATE_FORCE_PARAMETERS[:3],
        "state_forces": [("arc_contact", arc), ("planar_contact", list(PLANAR_CONTACT))],
        "q0": q0,
        "v0": v0,
    }


def obstacle_mesh() -> tuple[np.ndarray, np.ndarray]:
    """The visual arc obstacle that ``DuckEnv3d`` writes to ``obs.obj``."""
    start, end = np.deg2rad(START_DEGREE), np.deg2rad(END_DEGREE)
    step = (end - start) / OBSTACLE_SEGMENTS
    vertices = []
    for i in range(OBSTACLE_SEGMENTS + 1):
        c, s = np.cos(start + step * i), np.sin(start + step * i)
        front = np.array([c * RADIUS + CENTER[0], CENTER[1] - 1.25, -s * RADIUS + CENTER[2]])
        back = front + np.array([0.0, 2.5, 0.0])
        vertices += [front, back, [front[0], front[1], -1.0], [back[0], back[1], -1.0]]
    faces = []
    for i in range(OBSTACLE_SEGMENTS):
        a = 4 * i
        faces += [(a, a + 1, a + 4), (a + 4, a + 1, a + 5), (a + 2, a, a + 4)]
        faces += [(a + 2, a + 4, a + 6), (a + 1, a + 3, a + 5), (a + 5, a + 3, a + 7)]
    last = 4 * OBSTACLE_SEGMENTS
    faces += [(0, 2, 1), (2, 3, 1), (last, last + 1, last + 2), (last + 2, last + 1, last + 3)]
    faces += [(3, 2, last + 3), (2, last + 2, last + 3)]
    return np.asarray(vertices), np.asarray(faces)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.frame = 0
        mesh = load_mesh(ASSET)[0]
        self.rest = mesh.vertices
        scene = duck_scene(self.rest)
        builder = newton.ModelBuilder()
        builder.add_soft_mesh(
            pos=wp.vec3(),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(),
            vertices=self.rest.astype(np.float32),
            indices=mesh.indices.ravel(),
            density=DENSITY,
            k_mu=0.0,
            k_lambda=0.0,
            k_damp=0.0,
            color=(1.0, 0.8, 0.0),
        )
        self.model = builder.finalize()
        self.solver = SolverDiffPD(
            self.model,
            density=DENSITY,
            corotated_stiffness=scene["corotated_stiffness"],
            volume_stiffness=scene["volume_stiffness"],
            state_forces=scene["state_forces"],
            gravity=scene["gravity"],
            rest_positions=self.rest,
            **PD_OPTIONS,
        )
        self.states = [self.model.state(), self.model.state()]
        self.states[0].particle_q.assign(scene["q0"].astype(np.float32))
        self.states[0].particle_qd.assign(scene["v0"].astype(np.float32))
        self.q0 = scene["q0"]

        device = self.model.device
        points, faces = obstacle_mesh()
        self.obstacle_points = wp.array(points, dtype=wp.vec3, device=device)
        self.obstacle_indices = wp.array(faces.ravel(), dtype=wp.int32, device=device)
        self.ground_points = wp.array(
            [(-10.0, -2.5, 0.0), (4.0, -2.5, 0.0), (4.0, 2.5, 0.0), (-10.0, 2.5, 0.0)], dtype=wp.vec3, device=device
        )
        self.ground_indices = wp.array([0, 1, 2, 0, 2, 3], dtype=wp.int32, device=device)
        self.target = wp.array([TARGET], dtype=wp.vec3, device=device)
        self.target_color = wp.full(1, wp.vec3(0.9, 0.1, 0.1), dtype=wp.vec3, device=device)
        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(-2.5, -12.0, 3.0), pitch=-10.0, yaw=90.0)

    def step(self):
        # The reference rollout has FRAMES steps; the duck then holds its last state.
        if self.frame < FRAMES:
            self.solver.step(self.states[0], self.states[1], None, None, DT)
            self.states.reverse()
            self.frame += 1
            self.sim_time += DT

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.states[0])
        self.viewer.log_mesh("/obstacle", self.obstacle_points, self.obstacle_indices, color=(0.3, 0.3, 0.3))
        self.viewer.log_mesh("/ground", self.ground_points, self.ground_indices, color=(0.4, 0.4, 0.4))
        self.viewer.log_points("/target", self.target, radii=0.08, colors=self.target_color)
        self.viewer.end_frame()

    def test_final(self):
        self.solver.check_status()
        q = self.states[0].particle_q.numpy().astype(np.float64)
        if not np.isfinite(q).all() or not np.isfinite(self.states[0].particle_qd.numpy()).all():
            raise AssertionError("The duck state is not finite")
        tets = self.model.tet_indices.numpy().reshape(-1, 4)
        rest = np.linalg.det(self.rest[tets[:, 1:]] - self.rest[tets[:, :1]])
        if np.any(np.linalg.det(q[tets[:, 1:]] - q[tets[:, :1]]) / rest <= 0.0):
            raise AssertionError("The duck has inverted tetrahedra")
        # Nothing adds energy: kinetic plus gravitational energy of the uniform
        # particle masses cannot exceed its initial value (contact penalty
        # energy is nonnegative and friction dissipates).
        mass, gravity = self.solver.particle_mass, -STATE_FORCE_PARAMETERS[2]
        v = self.states[0].particle_qd.numpy().astype(np.float64)
        initial = mass * (0.5 * INITIAL_SPEED**2 * len(q) + gravity * self.q0[:, 2].sum())
        if mass * (0.5 * np.sum(v * v) + gravity * q[:, 2].sum()) > initial:
            raise AssertionError("The duck gained mechanical energy")
        start = np.linalg.norm(self.q0.mean(axis=0) - TARGET)
        if np.linalg.norm(q.mean(axis=0) - TARGET) >= start:
            raise AssertionError("The duck did not move toward the target")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=FRAMES)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
