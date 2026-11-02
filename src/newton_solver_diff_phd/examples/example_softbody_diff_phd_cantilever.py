# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverDiffPhD: the original Hex8 cantilever under axial loading, with optional twist release.

Run ``python -m newton_solver_diff_phd.examples softbody_diff_phd_cantilever``.
The default follows the released cantilever_3d.py: 500 hexahedra, a fixed root,
a ten-times-stiffer middle, native muscle and vertex energies, and 25 axial
force-ramp steps of 0.01 seconds. --motion twist selects a 30-degree release
on the same Hex8 body; that initial angle is an explicit choice because the
paper's twist initial state is not in the released script. Forward simulation
only; no training or checkpoint. Display playback is slower than simulated time.
"""

from pathlib import Path
from time import perf_counter

import newton.examples
import numpy as np
import warp as wp

from ._diff_phd_cantilever import ASSET, FRAMES, TWIST_FRAMES, CantileverSimulation


@wp.kernel(enable_backward=False)
def _edge_positions(
    q: wp.array[wp.vec3], edges: wp.array[wp.vec2i], starts: wp.array[wp.vec3], ends: wp.array[wp.vec3]
):
    i = wp.tid()
    starts[i] = q[edges[i][0]]
    ends[i] = q[edges[i][1]]


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.started = perf_counter()
        self.interactive = not getattr(args, "headless", False) and not getattr(args, "test", False)
        print(f"Preparing Hex8 cantilever: 756 nodes, 500 hexahedra; {args.motion} forward simulation.", flush=True)
        self.simulation = CantileverSimulation(asset=args.asset, capture=not args.no_capture, motion=args.motion)
        self.model = self.simulation.model
        self.sim_time = 0.0
        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.46, -0.50, 0.30), pitch=-24.0, yaw=122.0)
        self.indices = [
            wp.array(
                self.simulation.faces[self.simulation.face_materials == group].ravel(),
                dtype=wp.int32,
                device=self.model.device,
            )
            for group in (0, 1)
        ]
        self.minimum_jacobian = np.inf
        self.maximum_motion = 0.0
        edges = self.simulation.hexes[
            :, ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))
        ].reshape(-1, 2)
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        self.edges = wp.array(edges, dtype=wp.vec2i, device=self.model.device)
        self.edge_starts = wp.empty(len(edges), dtype=wp.vec3, device=self.model.device)
        self.edge_ends = wp.empty_like(self.edge_starts)
        self.show_edges = True
        self.tip_angle = 30.0 if args.motion == "twist" else 0.0
        self.tip_rest = self.simulation.vertices[self.simulation.tip, 1:] - [0.04, 0.04]
        self.hold_until = perf_counter() + (1.5 if self.interactive and args.motion == "twist" else 0.0)
        print("Cantilever ready. Blue is soft; orange is ten times stiffer.", flush=True)

    def step(self):
        sim = self.simulation
        if self.interactive and perf_counter() < self.hold_until:
            return
        if sim.frame >= sim.frames:
            return
        sim.advance()
        self.sim_time = sim.frame * sim.dt
        q = sim.states[0].particle_q.numpy().astype(np.float64)
        ratios = sim.solver.deformation_determinants(sim.states[0]).numpy()
        self.minimum_jacobian = min(self.minimum_jacobian, float(ratios.min()))
        self.maximum_motion = max(self.maximum_motion, float(np.linalg.norm(q - sim.q0, axis=1).max()))
        tip = q[sim.tip, 1:]
        tip = tip - tip.mean(axis=0)
        cross = np.sum(self.tip_rest[:, 0] * tip[:, 1] - self.tip_rest[:, 1] * tip[:, 0])
        self.tip_angle = float(np.degrees(np.arctan2(cross, np.sum(self.tip_rest * tip))))
        if sim.frame == 1:
            print(f"First elastic step ready after {perf_counter() - self.started:.2f}s (including setup).", flush=True)
        elif sim.frame % 5 == 0:
            print(
                f"Cantilever {sim.motion}: {sim.frame}/{sim.frames} steps; {self.sim_time:.4f} simulated seconds.",
                flush=True,
            )
        if sim.frame == sim.frames:
            self.test_final()
            print("Simulation complete. Holding the final state; use Replay to restart.", flush=True)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        for group, color in enumerate(((0.2, 0.55, 0.95), (0.95, 0.35, 0.25))):
            self.viewer.log_mesh(
                f"/cantilever/material_{group}",
                self.simulation.states[0].particle_q,
                self.indices[group],
                color=color,
            )
        if self.show_edges:
            wp.launch(
                _edge_positions,
                len(self.edges),
                inputs=[self.simulation.states[0].particle_q, self.edges],
                outputs=[self.edge_starts, self.edge_ends],
                device=self.model.device,
            )
        self.viewer.log_lines(
            "/cantilever/edges", self.edge_starts, self.edge_ends, colors=(0.08, 0.1, 0.15), hidden=not self.show_edges
        )
        self.viewer.end_frame()

    def gui(self, ui):
        ui.text(f"Hex8 cantilever; {self.simulation.motion} motion")
        modes = ("axial", "twist")
        changed, selected = ui.combo("Motion", modes.index(self.simulation.motion), ["Axial stretch", "Twist release"])
        if changed:
            self.simulation.set_motion(modes[selected])
            self.started = perf_counter()
            self.sim_time = 0.0
            self.minimum_jacobian = np.inf
            self.maximum_motion = 0.0
            self.tip_angle = 30.0 if modes[selected] == "twist" else 0.0
            self.hold_until = perf_counter() + (1.5 if modes[selected] == "twist" else 0.0)
        ui.text("Fixed root; blue soft, orange 10 times stiffer")
        ui.text(f"Step {self.simulation.frame}/{self.simulation.frames}; time {self.sim_time:.4f} s")
        ui.text("Forward simulation; slow-motion playback")
        ui.text(f"Tip rotation: {self.tip_angle:.1f} degrees")
        _, self.show_edges = ui.checkbox("Show mesh edges", self.show_edges)
        if ui.button("Replay"):
            self.simulation.reset()
            self.sim_time = 0.0
            self.minimum_jacobian = np.inf
            self.maximum_motion = 0.0
            self.tip_angle = 30.0 if self.simulation.motion == "twist" else 0.0
            self.hold_until = perf_counter() + (1.5 if self.simulation.motion == "twist" else 0.0)

    def test_final(self):
        sim = self.simulation
        sim.solver.check_status()
        q, velocity = sim.states[0].particle_q.numpy(), sim.states[0].particle_qd.numpy()
        if not np.isfinite(q).all() or not np.isfinite(velocity).all() or self.minimum_jacobian <= 0:
            raise AssertionError("Cantilever produced nonfinite or inverted elements")
        np.testing.assert_array_equal(q[sim.pinned], sim.q0.astype(np.float32)[sim.pinned])
        np.testing.assert_array_equal(velocity[sim.pinned], 0.0)
        if sim.frame == sim.frames and self.maximum_motion < 0.001:
            raise AssertionError("Cantilever did not respond to its prescribed loading")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--asset", type=Path, default=ASSET, help="Baked cantilever USD")
    parser.add_argument("--no-capture", action="store_true", help="Disable whole-step capture for comparison")
    parser.add_argument(
        "--motion",
        choices=("axial", "twist"),
        default="axial",
        help="Released axial benchmark or explicit 30-degree twist release",
    )
    # Resolve the frame default before Newton initializes its viewer.
    preliminary, _ = parser.parse_known_args()
    parser.set_defaults(num_frames=FRAMES if preliminary.motion == "axial" else TWIST_FRAMES, render_fps=12)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
