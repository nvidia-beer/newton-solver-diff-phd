# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Optimize periodic leg-muscle commands for a native DiffPhD bear gait.

Run ``python -m newton_solver_diff_phd.examples diffsim_diff_phd_bear``. The brown torso and
blue legs have fixed passive materials. Sixteen sine/cosine weights control
eight muscle groups; gray and green paths compare initial and current motion.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import newton.examples
import numpy as np
import warp as wp

from newton_solver_diff_phd import BearProblem

from ._diff_phd_bear_checkpoint import TRAINED_CONTROLLER, load_controller, save_controller

DEFAULT_DT = BearProblem.DEFAULT_DT
DEFAULT_FRAMES = BearProblem.DEFAULT_FRAMES
DEFAULT_TRAIN_ITERS = BearProblem.DEFAULT_TRAIN_ITERS


class Example:
    def __init__(self, viewer, args):
        if args.train_iters < 0:
            raise ValueError("train-iters must be nonnegative")
        if args.trained and args.load_controller:
            raise ValueError("Choose either --trained or --load-controller")
        checkpoint_path = TRAINED_CONTROLLER if args.trained else args.load_controller
        self.inference = checkpoint_path is not None
        if self.inference and (args.check_gradients or args.output or args.save_controller):
            raise ValueError("Gradient checks, --output, and --save-controller apply to training only")
        checkpoint = load_controller(checkpoint_path) if self.inference else None
        resolution = (
            args.resolution if args.resolution is not None else (checkpoint["resolution"] if checkpoint else 10)
        )
        dt = args.dt if args.dt is not None else (checkpoint["dt"] if checkpoint else DEFAULT_DT)
        if checkpoint and (resolution != checkpoint["resolution"] or dt != checkpoint["dt"]):
            raise ValueError("Inference uses the checkpoint's resolution and dt; omit conflicting overrides")
        self.controller_path = (
            None if self.inference else Path(args.save_controller or "outputs/diff_phd_bear/controller.json")
        )
        self.viewer, self.args = viewer, args
        self.frame = 0
        self._advance_playback = False
        self.display = args.display
        self.training_paused = False
        self.gradient_check = None
        if self.inference:
            print(f"Loading bear controller: {checkpoint_path}; forward simulation only.", flush=True)
        else:
            print(f"Preparing bear; accepted controller updates will be saved to {self.controller_path}.", flush=True)
        started = perf_counter()
        self.problem = BearProblem(
            resolution=resolution,
            frames=args.trajectory_frames,
            dt=dt,
            progress=True,
            training=not self.inference,
            parameters=np.asarray(checkpoint["weights"]).reshape(16) if checkpoint else None,
        )
        if args.check_gradients:
            self.gradient_check = self.problem.check_gradient()
            print(
                f"Two-direction gait gradient relative error: {self.gradient_check['relative_error']:.3e}",
                flush=True,
            )
        if self.inference:
            print(f"Bear ready for inference: preparation {perf_counter() - started:.2f}s", flush=True)
        else:
            print(
                f"Bear ready: loss {self.problem.value:.6g}, preparation {perf_counter() - started:.2f}s",
                flush=True,
            )

        p = self.problem
        device = p.model.device
        self.viewer.set_model(p.model)
        if not self.inference:
            self.viewer.log_scalar("/loss", p.value, clear=True)
        self.mesh_points = wp.empty(p.model.particle_count, dtype=wp.vec3, device=device)
        self.mesh_indices = [
            wp.array(p.faces[p.face_materials == material].ravel(), dtype=wp.int32, device=device)
            for material in (0, 1)
        ]
        lower = p.vertices.min(axis=0)
        upper = p.vertices.max(axis=0)
        span = float(np.max(upper - lower))
        focus = 0.5 * (lower + upper)
        self.viewer.set_camera(wp.vec3(*(focus + span * np.array([1.0, -1.8, 0.9]))), pitch=-24.0, yaw=119.0)
        floor_lower = lower[:2] - 2.0 * span
        floor_upper = upper[:2] + 2.0 * span
        self.floor_points = wp.array(
            [
                (floor_lower[0], floor_lower[1], 0.0),
                (floor_upper[0], floor_lower[1], 0.0),
                (floor_upper[0], floor_upper[1], 0.0),
                (floor_lower[0], floor_upper[1], 0.0),
            ],
            dtype=wp.vec3,
            device=device,
        )
        self.floor_indices = wp.array([0, 1, 2, 0, 2, 3], dtype=wp.int32, device=device)
        if self.inference:
            return
        self.initial_centers = np.einsum("fij,i->fj", p.initial_positions, p.mass_weights)
        self.initial_line_starts = wp.array(self.initial_centers[:-1], dtype=wp.vec3, device=device)
        self.initial_line_ends = wp.array(self.initial_centers[1:], dtype=wp.vec3, device=device)
        self.current_line_starts = wp.empty(p.frames, dtype=wp.vec3, device=device)
        self.current_line_ends = wp.empty(p.frames, dtype=wp.vec3, device=device)
        self._update_trajectory()
        self.save_results()

    def _update_trajectory(self):
        p = self.problem
        self.current_centers = np.einsum("fij,i->fj", p.positions, p.mass_weights)
        self.current_line_starts.assign(self.current_centers[:-1])
        self.current_line_ends.assign(self.current_centers[1:])

    def _status(self):
        if self.inference:
            return "inference"
        p = self.problem
        if p.finished:
            return p.stop_reason
        return "iteration budget" if p.iteration >= self.args.train_iters else "training"

    def step(self):
        self._advance_playback = True
        p = self.problem
        if self.inference:
            p.advance()
            return
        if self.training_paused or p.finished or p.iteration >= self.args.train_iters:
            return
        # Replay the initial controller and each accepted update completely.
        if self.frame == 0 or self.frame % (p.frames + 1) != 0:
            return
        started = perf_counter()
        previous_iteration = p.iteration
        p.optimize_step()
        if p.iteration > previous_iteration:
            self.viewer.log_scalar("/loss", p.value)
            save_controller(
                self.controller_path,
                weights=p.x,
                resolution=p.resolution,
                dt=p.dt,
                iterations=p.iteration,
                loss=p.value,
                initial_loss=p.initial_loss,
            )
        self._update_trajectory()
        travel = self.current_centers[-1] - self.current_centers[0]
        print(
            f"Bear update {p.iteration}/{self.args.train_iters}: loss {p.value:.6g}; "
            f"forward {1000 * travel[0]:+.2f} mm, lateral {1000 * travel[1]:+.2f} mm; "
            f"{perf_counter() - started:.2f}s",
            flush=True,
        )
        self.save_results()

    def render(self):
        p = self.problem
        if self.inference:
            wp.copy(self.mesh_points, p.states[0].particle_q)
        else:
            trajectory = p.positions if self.display == "current" else p.initial_positions
            self.mesh_points.assign(trajectory[self.frame % (p.frames + 1)])
        self.viewer.begin_frame(p.elapsed_steps * p.dt if self.inference else self.frame / 60.0)
        self.viewer.log_mesh("/floor", self.floor_points, self.floor_indices, color=(0.32, 0.35, 0.4), roughness=1.0)
        for material, color in enumerate(((0.5, 0.28, 0.12), (0.15, 0.4, 0.85))):
            if len(self.mesh_indices[material]):
                self.viewer.log_mesh(
                    f"/bear/material_{material}", self.mesh_points, self.mesh_indices[material], color=color
                )
        if not self.inference:
            self.viewer.log_lines("/path/initial", self.initial_line_starts, self.initial_line_ends, (0.65, 0.65, 0.65))
            self.viewer.log_lines("/path/current", self.current_line_starts, self.current_line_ends, (0.15, 0.95, 0.25))
        self.viewer.end_frame()
        if self._advance_playback:
            self.frame += 1
            self._advance_playback = False

    def gui(self, ui):
        p = self.problem
        if self.inference:
            ui.text("DiffPhD: saved bear controller")
            ui.text("Forward simulation; fixed weights and materials")
            ui.text(f"Simulation time: {p.elapsed_steps * p.dt:.3f} s")
            ui.text("Brown: torso; blue: legs")
            return
        ui.text("DiffPhD: optimize a bear gait")
        ui.text("Brown: torso; blue: legs; materials stay fixed")
        ui.text("Gray: initial COM path; green: optimized COM path")
        for label, display in (("Optimized gait", "current"), ("Initial gait", "initial")):
            if ui.button(label):
                self.display = display
        ui.text(f"Showing: {self.display}")
        _, self.training_paused = ui.checkbox("Pause optimization", self.training_paused)
        ui.text(f"Update {p.iteration}/{self.args.train_iters}: {self._status()}")
        ui.text(f"Loss: {p.value:.6g} (initial {p.initial_loss:.6g})")
        for label, centers in (("Current", self.current_centers), ("Initial", self.initial_centers)):
            travel = 1000 * (centers[-1] - centers[0])
            ui.text(f"{label}: forward (+X) {travel[0]:+.2f} mm")
            ui.text(f"  Lateral drift (+Y) {travel[1]:+.2f} mm")
        ui.separator()
        ui.text("Eight muscle groups: sine / cosine weights")
        for group, (sine, cosine) in enumerate(p.x.reshape(8, 2)):
            ui.text(f"Group {group + 1}: {sine:+.3f} / {cosine:+.3f}")

    def save_results(self):
        if self.inference or self.args.output is None:
            return
        p = self.problem
        path = Path(self.args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "resolution": p.resolution,
            "trajectory_frames": p.frames,
            "frame_dt": p.dt,
            "initial_weights": p.initial_x.reshape(8, 2).tolist(),
            "optimized_weights": p.x.reshape(8, 2).tolist(),
            "initial_com_displacement_m": (self.initial_centers[-1] - self.initial_centers[0]).tolist(),
            "com_displacement_m": (self.current_centers[-1] - self.current_centers[0]).tolist(),
            "initial_loss": p.initial_loss,
            "loss": p.value,
            "loss_history": p.history,
            "iterations": p.iteration,
            "gradient_check": self.gradient_check,
            "status": self._status(),
        }
        path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        np.savez_compressed(
            path.with_suffix(".npz"),
            initial=p.initial_positions,
            optimized=p.positions,
            initial_com=self.initial_centers,
            optimized_com=self.current_centers,
            weights=p.x,
            initial_weights=p.initial_x,
        )

    def test_final(self):
        p = self.problem
        if not np.isfinite(p.x).all() or (not self.inference and not np.isfinite(p.gradient).all()):
            raise AssertionError("Nonfinite bear gait weights or gradient")
        rest = p.vertices[p.tets].astype(np.float64)
        rest_determinants = np.linalg.det(rest[:, 1:] - rest[:, :1])
        if np.any(rest_determinants <= 0.0) or not np.isfinite(rest_determinants).all():
            raise AssertionError("Bear reference tetrahedra must have positive finite volumes")
        trajectories = (
            (("Inference", p.states[0].particle_q.numpy()[None]),)
            if self.inference
            else (("Initial", p.initial_positions), ("Current", p.positions))
        )
        for label, trajectory in trajectories:
            if not np.isfinite(trajectory).all():
                raise AssertionError(f"{label} bear trajectory contains nonfinite positions")
            corners = trajectory[:, p.tets].astype(np.float64)
            determinants = np.linalg.det(corners[:, :, 1:] - corners[:, :, :1]) / rest_determinants
            if np.any(determinants <= 0.0) or not np.isfinite(determinants).all():
                raise AssertionError(f"{label} bear trajectory contains inverted or collapsed tetrahedra")
            if trajectory[:, :, 2].min() < p.solver.contact_thickness - 1.0e-6:
                raise AssertionError(f"{label} bear trajectory violates ground clearance")
        if self.inference:
            if not np.isfinite(p.states[0].particle_qd.numpy()).all():
                raise AssertionError("Nonfinite inference velocities")
            return
        if not np.isfinite(p.value):
            raise AssertionError("Nonfinite gait objective")
        if p.iteration > 0 and p.value >= p.initial_loss:
            raise AssertionError("Gait optimization did not reduce the objective")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=(DEFAULT_TRAIN_ITERS + 1) * (DEFAULT_FRAMES + 1), render_fps=60)
        parser.add_argument("--resolution", type=int, help="Bundled geometry resolution (supported value: 10).")
        parser.add_argument("--trajectory-frames", type=int, default=DEFAULT_FRAMES)
        parser.add_argument("--dt", type=float, help="Physics timestep [s] (training default: 0.008).")
        parser.add_argument("--train-iters", type=int, default=DEFAULT_TRAIN_ITERS)
        parser.add_argument("--display", choices=("current", "initial"), default="current")
        parser.add_argument("--check-gradients", action="store_true", help="Check two controller-gradient directions.")
        parser.add_argument("--output", help="Training output prefix for metrics JSON and trajectory NPZ files.")
        parser.add_argument(
            "--save-controller", help="Training checkpoint path (default: outputs/diff_phd_bear/controller.json)."
        )
        parser.add_argument("--load-controller", help="Run forward-only inference with a saved controller JSON.")
        parser.add_argument(
            "--trained",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Run inference with the bundled trained controller.",
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
