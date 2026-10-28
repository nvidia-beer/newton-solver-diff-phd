# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fit a die's placement, stiffness or compressibility to PokeFlex deformations.

Run ``python -m newton_solver_diff_phd.examples softbody_diff_phd_ur5_poke`` with locally
prepared recording data. The original scanned surface is embedded in native
DiffPhD tetrahedra; the magenta probe makes physical frictional contact.
Optimization is offline. ``--load-alignment PATH`` instead runs live forward
simulation with the saved placement and material, without a Tape or a reference rollout.
Use ``--fit-mode stiffness --initial-xy X Y`` after registering the die to fit
Young's modulus at fixed placement and Poisson ratio. This calibrates the
simulation; the supplied scans alone do not establish the real foam's modulus.
``--fit-mode compressibility`` instead varies Poisson's ratio at fixed Young's
modulus and placement, avoiding a joint fit of poorly identifiable parameters.
The prescribed robot motion is precalculated; the deformable dynamics remain
live during inference.
The display places the clamped die on the floor; recording and solver
coordinates remain unchanged so saved alignments stay compatible.
"""

import hashlib
import json
import warnings
from pathlib import Path
from time import perf_counter

import newton.examples
import numpy as np
import warp as wp

from newton_solver_diff_phd import PokeAlignmentProblem, UR5Kinematics, command_poke_velocities, damp_poke_velocities

from ._diff_phd_poke import build_poke_scene


def default_recording_path(*, reference=False):
    name = "recording_reference.npz" if reference else "recording_debug.json"
    return Path(__file__).parent / "assets" / "pokeflex_local" / name


@wp.kernel(enable_backward=False)
def _display_positions(q: wp.array[wp.vec3], offset: wp.vec3, points: wp.array[wp.vec3]):
    i = wp.tid()
    points[i] = q[i] + offset


@wp.kernel(enable_backward=False)
def _display_surface(
    q: wp.array[wp.vec3],
    indices: wp.array[wp.vec4i],
    weights: wp.array[wp.vec4d],
    points: wp.array[wp.vec3],
):
    i = wp.tid()
    value = wp.vec3d(0.0)
    for j in range(4):
        value += weights[i][j] * wp.vec3d(q[indices[i][j]])
    points[i] = wp.vec3(value)


def _load_alignment(path):
    with Path(path).open() as stream:
        data = json.load(stream)
    xy = np.asarray(data.get("xy_m"), dtype=np.float64)
    contact_tolerance = data.get("contact_tolerance_m", 1.0e-8)
    velocity_damping = data.get("velocity_damping", 1.0)
    young = data.get("young_pa", 5.0e5)
    poisson = data.get("poisson", 0.40)
    if (
        data.get("format") != "newton.diff_phd.poke_alignment"
        or data.get("version") != 1
        or xy.shape != (2,)
        or not np.isfinite(xy).all()
        or not isinstance(data.get("substeps"), int)
        or isinstance(data["substeps"], bool)
        or data["substeps"] < 1
        or not np.isfinite(data.get("friction", np.nan))
        or data["friction"] < 0.0
        or not isinstance(contact_tolerance, (int, float))
        or isinstance(contact_tolerance, bool)
        or not np.isfinite(contact_tolerance)
        or contact_tolerance <= 0.0
        or not isinstance(velocity_damping, (int, float))
        or isinstance(velocity_damping, bool)
        or not np.isfinite(velocity_damping)
        or not 0.0 <= velocity_damping <= 1.0
        or not isinstance(young, (int, float))
        or isinstance(young, bool)
        or not np.isfinite(young)
        or young <= 0.0
        or not isinstance(poisson, (int, float))
        or isinstance(poisson, bool)
        or not np.isfinite(poisson)
        or not 0.0 <= poisson < 0.5
    ):
        raise ValueError("Invalid DiffPhD poke alignment checkpoint")
    data["contact_tolerance_m"] = float(contact_tolerance)
    data["velocity_damping"] = float(velocity_damping)
    data["young_pa"], data["poisson"] = float(young), float(poisson)
    return data


class Example:
    class _Forward:
        """Two-state, untaped replay of the recorded physical tool commands."""

        def __init__(self, scene, xy):
            self.scene = scene
            self.model, self.solver = scene["model"], scene["solver"]
            self.state = self.model.state(requires_grad=False)
            self.next_state = self.model.state(requires_grad=False)
            self.control = self.model.control(requires_grad=False)
            initial = scene["rest"].copy()
            initial[: scene["soft_count"], :2] += np.asarray(xy) - scene["data"]["reference_xy"]
            self.state.particle_q.assign(initial)
            self.state.particle_qd.zero_()
            self.state.particle_f.zero_()
            self.next_state.particle_f.zero_()
            self.endpoints = wp.array(scene["tool_positions"], dtype=wp.vec3, device=self.model.device)
            self.dt = scene["frame_dt"] / scene["substeps"]
            self.velocity_damping = scene.get("velocity_damping", 1.0)
            self.damped_velocity = wp.empty_like(self.state.particle_qd)
            self.elapsed_steps = 0
            self.max_residual = 0.0

        def advance(self):
            for _ in range(self.scene["substeps"]):
                # After frame 53, keep the last tool pose and continue dynamics.
                index = min(self.elapsed_steps, len(self.endpoints) - 2)
                wp.launch(
                    command_poke_velocities,
                    self.model.particle_count,
                    inputs=[
                        self.state.particle_q,
                        self.state.particle_qd,
                        self.endpoints,
                        index,
                        self.scene["soft_count"],
                        self.dt,
                    ],
                    outputs=[self.state.particle_qd],
                    device=self.model.device,
                )
                self.solver.step(self.state, self.next_state, self.control, None, self.dt)
                if self.velocity_damping != 1.0:
                    wp.launch(
                        damp_poke_velocities,
                        self.model.particle_count,
                        inputs=[self.next_state.particle_qd, self.model.particle_inv_mass, self.velocity_damping],
                        outputs=[self.damped_velocity],
                        device=self.model.device,
                    )
                    self.next_state.particle_qd, self.damped_velocity = (
                        self.damped_velocity,
                        self.next_state.particle_qd,
                    )
                self.max_residual = max(self.max_residual, self.solver.last_step_info.residual)
                self.state, self.next_state = self.next_state, self.state
                self.elapsed_steps += 1

    def __init__(self, viewer, args):
        if args.train_iters < 0:
            raise ValueError("train-iters must be nonnegative")
        if args.refinement is not None:
            warnings.warn(
                "--refinement is deprecated for the recorded die; mesh resolution is set during preparation",
                DeprecationWarning,
                stacklevel=2,
            )
            if args.refinement != 1:
                raise ValueError("Prepare a different recording mesh instead of using --refinement")
        checkpoint = _load_alignment(args.load_alignment) if args.load_alignment else None
        if checkpoint and (args.check_gradients or args.save_alignment or args.output):
            raise ValueError("Gradient checks and training output options cannot be used with --load-alignment")
        if checkpoint and args.target_rms_mm is not None:
            raise ValueError("--target-rms-mm validates a complete training trajectory; omit --load-alignment")
        self.reference = args.reference or bool(checkpoint and checkpoint.get("reference", False))
        if args.reference and checkpoint and not checkpoint.get("reference", False):
            raise ValueError("--reference cannot change the model of an existing fast checkpoint")
        substeps = (
            args.substeps
            if args.substeps is not None
            else checkpoint["substeps"]
            if checkpoint
            else 30
            if self.reference
            else 1
        )
        velocity_damping = (
            args.velocity_damping
            if args.velocity_damping is not None
            else checkpoint["velocity_damping"]
            if checkpoint
            else 0.30
            if self.reference
            else 1.0
        )
        friction = args.friction if args.friction is not None else checkpoint["friction"] if checkpoint else 0.5
        young = args.young if args.young is not None else checkpoint["young_pa"] if checkpoint else 5.0e5
        poisson = args.poisson if args.poisson is not None else checkpoint["poisson"] if checkpoint else 0.40
        contact_tolerance = (
            args.contact_tolerance
            if args.contact_tolerance is not None
            else checkpoint["contact_tolerance_m"]
            if checkpoint
            else 1.0e-8
            if self.reference
            else 1.0e-7
        )
        if not np.isfinite(friction) or friction < 0.0:
            raise ValueError("friction must be finite and nonnegative")
        if not np.isfinite(contact_tolerance) or contact_tolerance <= 0.0:
            raise ValueError("contact-tolerance must be finite and positive")
        if not np.isfinite(velocity_damping) or not 0.0 <= velocity_damping <= 1.0:
            raise ValueError("velocity-damping must be finite and between zero and one")
        if args.target_rms_mm is not None and (not np.isfinite(args.target_rms_mm) or args.target_rms_mm <= 0):
            raise ValueError("target-rms-mm must be finite and positive")
        if checkpoint and (
            substeps != checkpoint["substeps"]
            or friction != checkpoint["friction"]
            or contact_tolerance != checkpoint["contact_tolerance_m"]
            or velocity_damping != checkpoint["velocity_damping"]
            or young != checkpoint["young_pa"]
            or poisson != checkpoint["poisson"]
        ):
            raise ValueError(
                "Inference must use the checkpoint's material, substeps, friction, contact tolerance and damping"
            )
        self.viewer, self.args = viewer, args
        self.inference = checkpoint is not None
        self.saved_rms_m = checkpoint.get("rms_surface_error_m") if checkpoint else None
        self.recording_path = (
            Path(args.recording) if args.recording else default_recording_path(reference=self.reference)
        )
        self.scene = build_poke_scene(
            self.recording_path, substeps=substeps, reference=self.reference, young=young, poisson=poisson
        )
        self.young, self.poisson = float(young), float(poisson)
        self.recording_hash = hashlib.sha256(self.recording_path.read_bytes()).hexdigest()
        s, data = self.scene, self.scene["data"]
        s["solver"].friction = friction
        s["solver"].contact_tolerance = contact_tolerance
        self.friction = float(friction)
        self.contact_tolerance = float(contact_tolerance)
        self.velocity_damping = float(velocity_damping)
        s["velocity_damping"] = self.velocity_damping
        self.target_rms_m = None if args.target_rms_mm is None else args.target_rms_mm / 1000.0
        self.show_target = bool(args.show_target)
        self.target_alpha = 0.30
        recorded_meshes = []
        if "recorded_vertices" in data:
            for start, end in zip(
                data["recorded_vertex_offsets"][:-1], data["recorded_vertex_offsets"][1:], strict=True
            ):
                recorded_meshes.append(data["recorded_vertices"][start:end])
        if self.reference and not recorded_meshes:
            raise ValueError("--reference requires full recorded scans; prepare the recording again with --reference")
        if checkpoint and (
            checkpoint.get("recording_sha256") != self.recording_hash
            or checkpoint.get("frame_dt") != s["frame_dt"]
            or not np.array_equal(checkpoint.get("reference_xy_m"), data["reference_xy"])
        ):
            raise ValueError("The checkpoint requires the same prepared recording and frame interval")
        xy = np.asarray(checkpoint["xy_m"] if checkpoint else args.initial_xy, dtype=np.float64)
        self.xy = xy.copy()
        s["model"].tet_materials.requires_grad = not self.inference and args.fit_mode != "position"
        self.problem = None
        self.forward = None
        if self.inference:
            self.forward = self._Forward(s, xy)
        else:
            sample = s["loss_sample_ids"]
            self.problem = PokeAlignmentProblem(
                s["model"],
                s["solver"],
                s["soft_count"],
                s["surface_indices"],
                s["tool_positions"],
                recorded_meshes[1:] if self.reference else data["target_points"][1:],
                initial_xy=xy,
                reference_xy=data["reference_xy"],
                frame_dt=s["frame_dt"],
                substeps=substeps,
                progress=True,
                optimizer=args.optimizer,
                velocity_damping=self.velocity_damping,
                fit_mode=args.fit_mode,
                young=young,
                poisson=poisson,
                young_bounds=args.young_bounds,
                poisson_bounds=args.poisson_bounds,
                surface_embedding=None
                if self.reference
                else (
                    data["surface_embedding_indices"][sample],
                    data["surface_embedding_weights"][sample],
                ),
            )
        self.save_path = (
            None
            if self.inference
            else Path(
                args.save_alignment
                or (
                    f"outputs/diff_phd_poke/{args.fit_mode if args.fit_mode != 'position' else 'alignment'}"
                    f"_{'reference' if self.reference else 'fast'}.json"
                )
            )
        )
        # Translate only the display, preserving recorded coordinates and checkpoints.
        offset = np.array((0.0, 0.0, -float(data["surface_vertices"][:, 2].min())))
        self.display_offset = wp.vec3(*offset)
        pose = s["tool_poses"][0]
        self.robot = UR5Kinematics(
            asset_path=Path(__file__).parent / "assets" / "universal_robots_ur5e",
            initial_position=pose[:3] + offset,
            initial_rotation=pose[3:],
            base_height=0.5 + offset[2],
            device=s["model"].device,
        )
        self.robot_body_poses = []
        for pose in s["tool_poses"][::substeps]:
            self.robot.update(pose[:3] + offset, pose[3:])
            self.robot_body_poses.append(wp.clone(self.robot.state.body_q))
        self.viewer.set_model(self.robot.model)
        self.viewer.set_camera(wp.vec3(1.45, -1.2, 1.0 + offset[2]), pitch=-24.0, yaw=124.0)
        device = s["model"].device
        rest = s["rest"].copy()
        rest[: s["soft_count"], :2] += xy - data["reference_xy"]
        self.mesh_q = wp.array(rest, dtype=wp.vec3, device=device)
        self.display_q = wp.empty_like(self.mesh_q)
        self.embedding_indices = wp.array(data["surface_embedding_indices"], dtype=wp.vec4i, device=device)
        self.embedding_weights = wp.array(data["surface_embedding_weights"], dtype=wp.vec4d, device=device)
        self.mesh_points = wp.empty(len(self.embedding_indices), dtype=wp.vec3, device=device)
        self.mesh_faces = wp.array(data["surface_faces"].ravel(), dtype=wp.int32, device=device)
        self.tool_faces = wp.array(s["tool_faces"].ravel(), dtype=wp.int32, device=device)
        self.recorded_points = [
            wp.array(points + offset, dtype=wp.vec3, device=device) for points in data["target_points"]
        ]
        self.recorded_colors = wp.full(len(self.recorded_points[0]), wp.vec3(0.75, 0.9, 1.0), device=device)
        self.target_mesh_points = [
            wp.array(points + offset, dtype=wp.vec3, device=device) for points in recorded_meshes
        ]
        self.target_mesh_faces = [
            wp.array(data["recorded_faces"][start:end].ravel(), dtype=wp.int32, device=device)
            for start, end in zip(
                data.get("recorded_face_offsets", [])[:-1], data.get("recorded_face_offsets", [])[1:], strict=True
            )
        ]
        self._target_visible = False
        self.frame = 0
        self.display = "current" if self.inference else args.display
        self.training_paused = False
        self.gradient_check = None
        self.gradient_check_xy = None
        self.gradient_check_parameters = None
        self._first_frame = True
        self._draw_step = False
        self._die_visible = False
        self.last_step_seconds = 0.0
        print(
            "Recorded die ready to display. "
            + (
                "Loaded alignment: live forward simulation."
                if self.inference
                else "Initial loss/adjoint evaluation starts after the first draw; optimization is offline."
            ),
            flush=True,
        )

    def step(self):
        if self._first_frame:
            self._first_frame = False
            return
        started = perf_counter()
        if self.inference:
            self.forward.advance()
            self.frame += 1
        else:
            p = self.problem
            if p.value is None:
                p.prepare()
                if self.args.check_gradients:
                    self.gradient_check = p.check_gradient()
                    self.gradient_check_xy = p.xy
                    self.gradient_check_parameters = p.x.copy()
                    print(
                        f"{p.parameter_names} gradient relative errors: {self.gradient_check['relative_error']}",
                        flush=True,
                    )
                self._save_results()
            else:
                self.frame += 1
                if (
                    self.frame % (p.frames + 1) == 0
                    and not self.training_paused
                    and not p.finished
                    and p.iteration < self.args.train_iters
                ):
                    p.optimize_step()
                    self.xy, self.young, self.poisson = p.xy, p.young, p.poisson
                    print(
                        f"Fit {p.iteration}/{self.args.train_iters}: xy={p.xy}, E={p.young:.6g} Pa, nu={p.poisson:.6g}, "
                        f"loss={p.value:.7g} m²; {perf_counter() - started:.2f}s",
                        flush=True,
                    )
                    self._save_results()
        self.last_step_seconds = perf_counter() - started
        self._draw_step = True

    def render(self):
        s = self.scene
        if self.inference:
            q = self.forward.state.particle_q
            index = min(self.forward.elapsed_steps, len(s["tool_poses"]) - 1)
            observed = min(index // s["substeps"], len(self.recorded_points) - 1)
            clock = self.forward.elapsed_steps * self.forward.dt
        else:
            p = self.problem
            observed = self.frame % (p.frames + 1)
            if p.positions is not None:
                positions = p.initial_positions if self.display == "initial" else p.positions
                self.mesh_q.assign(positions[observed])
            q, index, clock = self.mesh_q, observed * s["substeps"], self.frame * s["frame_dt"]
        wp.copy(self.robot.state.body_q, self.robot_body_poses[observed])
        wp.launch(
            _display_positions,
            len(self.display_q),
            inputs=[q, self.display_offset],
            outputs=[self.display_q],
            device=s["model"].device,
        )
        die_visible = self.display != "recorded"
        if die_visible:
            wp.launch(
                _display_surface,
                len(self.mesh_points),
                inputs=[self.display_q, self.embedding_indices, self.embedding_weights],
                outputs=[self.mesh_points],
                device=s["model"].device,
            )
        self.viewer.begin_frame(clock)
        self.viewer.log_state(self.robot.state)
        if die_visible or self._die_visible:
            self.viewer.log_mesh(
                "/die",
                self.mesh_points,
                self.mesh_faces,
                color=(0.9, 0.9, 0.88),
                roughness=0.8,
                hidden=not die_visible,
            )
        self._die_visible = die_visible
        self.viewer.log_points(
            "/recorded",
            self.recorded_points[observed],
            radii=0.0018,
            colors=self.recorded_colors,
            hidden=self.display != "recorded" and (not self.show_target or bool(self.target_mesh_points)),
        )
        self.viewer.log_mesh("/probe", self.display_q, self.tool_faces, color=(0.8, 0.15, 0.55), roughness=0.5)
        if self.target_mesh_points and (self.show_target or self._target_visible):
            self.viewer.log_mesh(
                "/recorded_target",
                self.target_mesh_points[observed],
                self.target_mesh_faces[observed],
                color=(0.25, 0.75, 1.0),
                roughness=0.8,
                backface_culling=False,
                hidden=not self.show_target,
            )
            target = getattr(self.viewer, "objects", {}).get("/recorded_target")
            if target is not None:
                target.alpha, target.cast_shadow = self.target_alpha, False
        self._target_visible = self.show_target
        if self._draw_step:
            self.viewer.log_scalar("/alignment/x_m", self.xy[0])
            self.viewer.log_scalar("/alignment/y_m", self.xy[1])
            self.viewer.log_scalar("/material/young_pa", self.young)
            self.viewer.log_scalar("/material/poisson", self.poisson)
            self.viewer.log_scalar("/timing/step_seconds", self.last_step_seconds)
            if self.problem is not None and self.problem.value is not None:
                self.viewer.log_scalar("/alignment/loss_m2", self.problem.value)
            self._draw_step = False
        self.viewer.end_frame()

    def gui(self, ui):
        ui.text("UR5e / recorded PokeFlex die")
        ui.text("Native frictional probe contact")
        ui.text(f"Young's modulus: {self.young / 1000:.3f} kPa; Poisson ratio: {self.poisson:.3f}")
        ui.text("Precalculated recorded robot motion")
        ui.text("White: original scan surface (source texture unavailable)")
        ui.text(f"Placement: x={self.xy[0]:.5f}, y={self.xy[1]:.5f} m")
        observed = (
            min(self.frame, len(self.recorded_points) - 1) if self.inference else self.frame % len(self.recorded_points)
        )
        ui.text(f"Recorded frame: {18 + observed}")
        _, self.show_target = ui.checkbox("Show recorded target", self.show_target)
        if self.show_target:
            if self.target_mesh_points:
                _, self.target_alpha = ui.slider_float("Target opacity", self.target_alpha, 0.05, 0.8)
                ui.text("Blue ghost: recorded surface at the same frame")
            else:
                ui.text("Blue points: recorded target (legacy recording has no full meshes)")
        if self.inference:
            ui.text("LIVE forward simulation; no optimizer or adjoint")
            if self.saved_rms_m is not None:
                ui.text(f"Saved fit Chamfer RMS: {1000 * self.saved_rms_m:.2f} mm")
            ui.text(f"Latest frame computation: {1000 * self.last_step_seconds:.1f} ms")
            return
        for label, display in (
            ("Current alignment", "current"),
            ("Initial alignment", "initial"),
            ("Recorded scan points", "recorded"),
        ):
            if ui.button(label):
                self.display = display
        _, self.training_paused = ui.checkbox("Pause optimization", self.training_paused)
        p = self.problem
        if p.fit_mode == "stiffness":
            ui.text("Fitting stiffness at fixed placement and Poisson ratio")
        elif p.fit_mode == "compressibility":
            ui.text("Fitting Poisson ratio at fixed placement and Young's modulus")
        if p.fit_mode != "position":
            ui.text("Model calibration; real foam stiffness is not yet verified")
        ui.text("Offline optimization; recorded trajectories replay at 30 Hz")
        if p.value is None:
            ui.text("Initial evaluation will start after the first draw")
        else:
            status = (
                p.stop_reason
                if p.finished
                else "iteration budget"
                if p.iteration >= self.args.train_iters
                else "training"
            )
            ui.text(f"Update {p.iteration}/{self.args.train_iters}: {status}")
            if p.fit_mode != "position" and np.any((p.x <= p.bounds[:, 0]) | (p.x >= p.bounds[:, 1])):
                ui.text("Search bound reached; this does not identify a unique material value")
            ui.text(f"Loss: {p.value:.7g} m² (initial {p.initial_loss:.7g})")
            rms = np.sqrt(p.value / p.frames)
            ui.text(f"Chamfer RMS: {1000 * rms:.2f} mm (initial {1000 * np.sqrt(p.initial_loss / p.frames):.2f})")
            if self.gradient_check is not None:
                checked = self.gradient_check_parameters
                placement = "current" if np.array_equal(checked, p.x) else "previous"
                ui.text(
                    f"Gradient check: {'PASS' if self._gradient_check_passes() else 'FAIL'} ({placement} parameters)"
                )
            if self.target_rms_m is None:
                ui.text("Accuracy target not set; a lower loss alone does not certify recovery")
            else:
                ui.text(
                    f"Accuracy target: {1000 * self.target_rms_m:.2f} mm — {'MET' if rms <= self.target_rms_m else 'NOT MET'}"
                )
            ui.text(f"Saved alignment: {self.save_path}")

    def _save_results(self):
        p = self.problem
        report = {
            "format": "newton.diff_phd.poke_alignment",
            "version": 1,
            "xy_m": p.xy.tolist(),
            "fit_mode": p.fit_mode,
            "young_pa": p.young,
            "initial_young_pa": p.young_reference,
            "poisson": p.poisson,
            "initial_poisson": p.poisson_reference,
            "young_bounds_pa": p.young_bounds.tolist(),
            "poisson_bounds": p.poisson_bounds.tolist(),
            "parameter_names": list(p.parameter_names),
            "parameters": p.x.tolist(),
            "physical_material_verified": False,
            "parameter_bounds_reached": bool(np.any((p.x <= p.bounds[:, 0]) | (p.x >= p.bounds[:, 1]))),
            "reference_xy_m": p.reference_xy.tolist(),
            "recording_sha256": self.recording_hash,
            "frame_dt": p.frame_dt,
            "substeps": p.substeps,
            "friction": self.friction,
            "contact_tolerance_m": self.contact_tolerance,
            "velocity_damping": self.velocity_damping,
            "reference": self.reference,
            "optimizer": p.optimizer,
            "objective": "boundary_to_full_scan" if self.reference else "sampled_embedded_surface",
            "rms_surface_error_m": float(np.sqrt(p.value / p.frames)),
            "target_rms_m": self.target_rms_m,
            "accuracy_target_met": None
            if self.target_rms_m is None
            else bool(np.sqrt(p.value / p.frames) <= self.target_rms_m),
            "paper_reproduction_verified": False,
            "stop_reason": "iteration budget"
            if not p.finished and p.iteration >= self.args.train_iters
            else p.stop_reason,
            "iterations": p.iteration,
            "loss_m2": p.value,
            "initial_loss_m2": p.initial_loss,
            "history": p.history,
            "gradient_check": None
            if self.gradient_check is None
            else {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in self.gradient_check.items()
            },
            "gradient_check_xy_m": None if self.gradient_check_xy is None else self.gradient_check_xy.tolist(),
            "gradient_check_parameters": None
            if self.gradient_check_parameters is None
            else self.gradient_check_parameters.tolist(),
            "gradient_check_passed": None if self.gradient_check is None else bool(self._gradient_check_passes()),
        }
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.save_path.with_suffix(self.save_path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(self.save_path)
        if self.args.output:
            prefix = Path(self.args.output)
            prefix.parent.mkdir(parents=True, exist_ok=True)
            prefix.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
            np.savez_compressed(
                prefix.with_suffix(".npz"),
                initial=p.initial_positions,
                optimized=p.positions,
                xy=p.xy,
                young=p.young,
                poisson=p.poisson,
            )

    def _gradient_check_passes(self):
        return self.gradient_check is not None and np.allclose(
            self.gradient_check["analytic"],
            self.gradient_check["finite_difference"],
            rtol=0.01,
            atol=2e-5 if self.problem.fit_mode == "position" else 1e-9,
        )

    def test_final(self):
        if self.inference:
            positions = self.forward.state.particle_q.numpy()
            if self.forward.max_residual > self.scene["solver"].tolerance:
                raise AssertionError("Forward inference exceeded the physical residual tolerance")
        elif self.problem.positions is None:
            if self.target_rms_m is not None:
                raise AssertionError("Accuracy target requires a completed alignment evaluation")
            positions = self.mesh_q.numpy()
        else:
            p = self.problem
            positions = p.positions
            if not np.isfinite(p.gradient).all() or (p.iteration > 0 and p.value >= p.initial_loss):
                raise AssertionError("Invalid alignment gradient or nondecreasing training loss")
            if self.target_rms_m is not None and np.sqrt(p.value / p.frames) > self.target_rms_m:
                raise AssertionError("Recorded nearest-vertex Chamfer RMS exceeds the requested accuracy target")
        if not np.isfinite(positions).all():
            raise AssertionError("Nonfinite poke trajectory")
        if (
            self.args.check_gradients
            and self.gradient_check is not None
            and not np.array_equal(self.gradient_check_parameters, self.problem.x)
        ):
            self.gradient_check = self.problem.check_gradient()
            self.gradient_check_xy = self.problem.xy
            self.gradient_check_parameters = self.problem.x.copy()
            self._save_results()
        if self.args.check_gradients and not self._gradient_check_passes():
            raise AssertionError("Recorded-trajectory gradient check exceeds the 1% tolerance")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=2 + 13 * 36, render_fps=30)
        parser.add_argument("--recording", help="Locally prepared PokeFlex recording NPZ or bundled JSON manifest.")
        parser.add_argument(
            "--reference",
            action="store_true",
            help="Use the reference mesh, full-scan objective, and 30 substeps with 0.30 retained velocity.",
        )
        parser.add_argument("--optimizer", choices=("lbfgs", "gradient_descent"), default="lbfgs")
        parser.add_argument(
            "--fit-mode",
            choices=("position", "stiffness", "compressibility"),
            default="position",
            help="Fit XY, or hold --initial-xy fixed and fit Young's modulus or Poisson ratio separately.",
        )
        parser.add_argument(
            "--young", type=float, help="Initial Young's modulus [Pa] (default: 500000, or checkpoint value)."
        )
        parser.add_argument("--poisson", type=float, help="Fixed Poisson ratio (default: 0.40, or checkpoint value).")
        parser.add_argument(
            "--young-bounds",
            type=float,
            nargs=2,
            default=(2e4, 2e6),
            metavar=("MIN", "MAX"),
            help="Stiffness search bounds [Pa]; these are numerical bounds, not measured material limits.",
        )
        parser.add_argument(
            "--poisson-bounds",
            type=float,
            nargs=2,
            default=(0.05, 0.45),
            metavar=("MIN", "MAX"),
            help="Compressibility search bounds; these are numerical bounds, not measured material limits.",
        )
        parser.add_argument(
            "--velocity-damping", type=float, help="Retained velocity per substep (reference: 0.30; fast: 1.0)."
        )
        parser.add_argument(
            "--show-target", action="store_true", help="Overlay the recorded target at the current frame."
        )
        parser.add_argument(
            "--target-rms-mm",
            type=float,
            help="Required nearest-vertex Chamfer RMS [mm] during training; --test fails if exceeded.",
        )
        parser.add_argument("--initial-xy", type=float, nargs=2, default=(0.73, 0.115), metavar=("X", "Y"))
        parser.add_argument(
            "--substeps",
            type=int,
            default=None,
            help="Physics substeps per recorded frame (fast: 1; reference: 30; inference: checkpoint value).",
        )
        parser.add_argument(
            "--friction", type=float, default=None, help="Coulomb coefficient (default: 0.5, or checkpoint value)."
        )
        parser.add_argument(
            "--contact-tolerance",
            type=float,
            default=None,
            help="Contact residual tolerance [m] (fast: 1e-7; reference: 1e-8; inference: checkpoint value).",
        )
        parser.add_argument(
            "--refinement",
            type=int,
            default=None,
            help="Deprecated: set the recorded mesh resolution during preparation.",
        )
        parser.add_argument("--train-iters", type=int, default=12)
        parser.add_argument("--display", choices=("current", "initial", "recorded"), default="current")
        parser.add_argument(
            "--check-gradients",
            action="store_true",
            help="Check the fitted parameters' derivatives by finite differences.",
        )
        parser.add_argument(
            "--save-alignment", help="Training JSON path (default: outputs/diff_phd_poke/alignment_fast.json)."
        )
        parser.add_argument("--load-alignment", help="Run untaped live inference using a saved alignment JSON.")
        parser.add_argument(
            "--output", help="Optional prefix for the training report and initial/optimized trajectories."
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
