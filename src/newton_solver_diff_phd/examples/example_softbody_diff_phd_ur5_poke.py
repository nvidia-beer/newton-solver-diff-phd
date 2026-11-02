# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverDiffPhD: fit a die's placement, stiffness or compressibility to PokeFlex deformations.

Run ``python -m newton_solver_diff_phd.examples softbody_diff_phd_ur5_poke`` with locally
prepared recording data. The original scanned surface is embedded in native
DiffPhD tetrahedra. The magenta probe uses the original training demo's root
prescription and two-way mesh contact state force. These equations do not
guarantee nonpenetration. The prepared geometry differs from the original demo.
Optimization is offline. ``--forward-only`` runs live simulation at the specified
placement and material; ``--load-alignment PATH`` uses a compatible saved fit.
Both forward modes run without a Tape or an optimizer.
Use ``--fit-mode stiffness --initial-xy X Y`` after registering the die to fit
Young's modulus at fixed placement and Poisson ratio. This calibrates the
simulation; the supplied scans alone do not establish the real foam's modulus.
``--fit-mode compressibility`` instead varies Poisson's ratio at fixed Young's
modulus and placement, avoiding a joint fit of poorly identifiable parameters.
The prescribed robot motion is the reference script's hand-authored trajectory,
not measured probe poses from the recording; deformable dynamics remain live.
The display places the clamped die on the floor; recording and solver
coordinates remain unchanged.
"""

import hashlib
import json
import warnings
from pathlib import Path
from time import perf_counter

import newton.examples
import numpy as np
import warp as wp

from ._diff_phd_poke import CONTACT_RADIUS, FRICTION, POKE_PHYSICS, SUBSTEPS, build_poke_scene
from ._diff_phd_poke_alignment import PokeAlignmentProblem, command_poke_positions, damp_poke_velocities
from ._ur5_kinematics import UR5Kinematics


def default_recording_path(*, reference=False):
    name = "recording_reference.json" if reference else "recording_fine.json"
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
    velocity_damping = data.get("velocity_damping", 1.0)
    young = data.get("young_pa", 5.0e5)
    poisson = data.get("poisson", 0.40)
    if (
        data.get("format") != "newton.diff_phd.poke_alignment"
        or data.get("version") != 5
        or data.get("physics") != POKE_PHYSICS
        or not isinstance(data.get("scene_sha256"), str)
        or len(data["scene_sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in data["scene_sha256"])
        or xy.shape != (2,)
        or not np.isfinite(xy).all()
        or not isinstance(data.get("substeps"), int)
        or isinstance(data["substeps"], bool)
        or data["substeps"] < 1
        or not np.isfinite(data.get("friction", np.nan))
        or data["friction"] < 0.0
        or not isinstance(data.get("contact_radius_m"), (int, float))
        or isinstance(data["contact_radius_m"], bool)
        or not np.isfinite(data["contact_radius_m"])
        or data["contact_radius_m"] <= 0.0
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
        raise ValueError(
            "Invalid DiffPhD poke alignment checkpoint; version 5 requires refitting with the original "
            "training contact, probe material and prescribed-position semantics"
        )
    data["velocity_damping"] = float(velocity_damping)
    data["young_pa"], data["poisson"] = float(young), float(poisson)
    return data


def _scene_fingerprint(scene):
    """Identify the loaded geometry, clamp, observations and commanded trajectory."""
    digest = hashlib.sha256()
    arrays = {f"recording/{key}": value for key, value in scene["data"].items() if isinstance(value, np.ndarray)}
    arrays.update({key: scene[key] for key in ("rest", "tets", "tool_positions")})
    arrays["particle_mass"] = scene["model"].particle_mass.numpy()
    for name, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        digest.update(json.dumps([name, value.dtype.str, value.shape], separators=(",", ":")).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


class Example:
    class _Forward:
        """Two-state, untaped simulation driven by the reference tool commands."""

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
            self.dirichlet_positions = wp.empty_like(self.state.particle_q)
            self.dt = scene["frame_dt"] / scene["substeps"]
            self.velocity_damping = scene.get("velocity_damping", 1.0)
            self.damped_velocity = wp.empty_like(self.state.particle_qd)
            self.elapsed_steps = 0

        def advance(self):
            for _ in range(self.scene["substeps"]):
                # After frame 53, keep the last tool pose and continue dynamics.
                index = min(self.elapsed_steps, len(self.endpoints) - 2)
                wp.launch(
                    command_poke_positions,
                    self.model.particle_count,
                    inputs=[
                        self.state.particle_q,
                        self.endpoints,
                        index,
                        self.scene["soft_count"],
                    ],
                    outputs=[self.dirichlet_positions],
                    device=self.model.device,
                )
                self.solver.step(
                    self.state,
                    self.next_state,
                    self.control,
                    None,
                    self.dt,
                    dirichlet_positions=self.dirichlet_positions,
                )
                if self.scene.get("swept_contact", False):
                    # A rejected swept step must not advance the trajectory clock
                    # or silently render the restored state as a successful poke.
                    self.solver.check_status()
                if self.velocity_damping != 1.0:
                    wp.launch(
                        damp_poke_velocities,
                        self.model.particle_count,
                        inputs=[self.next_state.particle_qd, self.velocity_damping],
                        outputs=[self.damped_velocity],
                        device=self.model.device,
                    )
                    self.next_state.particle_qd, self.damped_velocity = (
                        self.damped_velocity,
                        self.next_state.particle_qd,
                    )
                self.state, self.next_state = self.next_state, self.state
                self.elapsed_steps += 1

    def __init__(self, viewer, args):
        self.swept_contact = bool(getattr(args, "swept_contact", False))
        if self.swept_contact and (not args.forward_only or args.load_alignment):
            raise ValueError("--swept-contact requires --forward-only; training and saved fits need a contact adjoint")
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
        inference = checkpoint is not None or args.forward_only
        if inference and (args.check_gradients or args.save_alignment or args.output):
            raise ValueError("Gradient checks and training output options require training mode")
        if inference and args.target_rms_mm is not None:
            raise ValueError("--target-rms-mm validates a complete training trajectory; use training mode")
        self.reference = args.reference or bool(checkpoint and checkpoint.get("reference", False))
        if args.reference and checkpoint and not checkpoint.get("reference", False):
            raise ValueError("--reference cannot change the model of an existing fast checkpoint")
        substeps = args.substeps if args.substeps is not None else checkpoint["substeps"] if checkpoint else SUBSTEPS
        velocity_damping = (
            args.velocity_damping
            if args.velocity_damping is not None
            else checkpoint["velocity_damping"]
            if checkpoint
            else 0.30
        )
        friction = (
            args.friction
            if args.friction is not None
            else checkpoint["friction"]
            if checkpoint
            else 0.0
            if self.swept_contact
            else FRICTION
        )
        contact_radius = (
            args.contact_radius
            if args.contact_radius is not None
            else checkpoint["contact_radius_m"]
            if checkpoint
            else CONTACT_RADIUS
        )
        if not np.isfinite(contact_radius) or contact_radius <= 0.0:
            raise ValueError("contact-radius must be finite and positive")
        young = args.young if args.young is not None else checkpoint["young_pa"] if checkpoint else 5.0e5
        poisson = args.poisson if args.poisson is not None else checkpoint["poisson"] if checkpoint else 0.40
        if not np.isfinite(friction) or friction < 0.0:
            raise ValueError("friction must be finite and nonnegative")
        if not np.isfinite(velocity_damping) or not 0.0 <= velocity_damping <= 1.0:
            raise ValueError("velocity-damping must be finite and between zero and one")
        if args.target_rms_mm is not None and (not np.isfinite(args.target_rms_mm) or args.target_rms_mm <= 0):
            raise ValueError("target-rms-mm must be finite and positive")
        if checkpoint and (
            substeps != checkpoint["substeps"]
            or friction != checkpoint["friction"]
            or contact_radius != checkpoint["contact_radius_m"]
            or velocity_damping != checkpoint["velocity_damping"]
            or young != checkpoint["young_pa"]
            or poisson != checkpoint["poisson"]
        ):
            raise ValueError(
                "Inference must use the checkpoint's material, substeps, friction, contact radius and damping"
            )
        self.viewer, self.args = viewer, args
        self.inference = inference
        self.saved_rms_m = checkpoint.get("rms_surface_error_m") if checkpoint else None
        self.recording_path = (
            Path(args.recording) if args.recording else default_recording_path(reference=self.reference)
        )
        self.scene = build_poke_scene(
            self.recording_path,
            substeps=substeps,
            reference=self.reference,
            young=young,
            poisson=poisson,
            friction=friction,
            contact_radius=contact_radius,
            swept_contact=self.swept_contact,
        )
        self.young, self.poisson = float(young), float(poisson)
        self.recording_hash = hashlib.sha256(self.recording_path.read_bytes()).hexdigest()
        self.scene_hash = _scene_fingerprint(self.scene)
        s, data = self.scene, self.scene["data"]
        self.friction = float(friction)
        self.contact_radius = float(contact_radius)
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
            or checkpoint.get("scene_sha256") != self.scene_hash
            or checkpoint.get("frame_dt") != s["frame_dt"]
            or not np.array_equal(checkpoint.get("reference_xy_m"), data["reference_xy"])
        ):
            raise ValueError("The checkpoint requires the same loaded geometry, motion, recording and frame interval")
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
                    f"_{'reference' if self.reference else 'fast'}_v5.json"
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
                "Live forward simulation from "
                + ("saved alignment." if checkpoint else "specified placement/material.")
                if self.inference
                else "Initial loss/adjoint evaluation starts after the first draw; optimization is offline."
            ),
            flush=True,
        )
        print(
            "Contact mode: swept CCD, kinematic probe, frictionless."
            if self.swept_contact
            else "Contact mode: original reference penalty; CCD is off. "
            "Use --forward-only --swept-contact for swept collision.",
            flush=True,
        )

    def step(self):
        if self._first_frame:
            self._first_frame = False
            return
        started = perf_counter()
        if self.inference:
            previous_steps = self.forward.elapsed_steps
            motion_steps = len(self.scene["tool_positions"]) - 1
            if previous_steps == 0:
                print(
                    "Starting the first simulation frame. Initial solver setup can take about a minute "
                    "for this prepared mesh; the viewer waits during setup."
                    if self.scene["solver"].inverse is None
                    else "Starting the poke with existing solver setup.",
                    flush=True,
                )
            self.forward.advance()
            self.frame += 1
            elapsed_steps = self.forward.elapsed_steps
            if previous_steps == 0:
                wp.synchronize_device(self.scene["model"].device)
                print(
                    f"First simulation frame ready after {perf_counter() - started:.2f}s (including setup).", flush=True
                )
            if elapsed_steps <= motion_steps and (self.frame == 1 or self.frame % 5 == 0):
                print(
                    f"Poke motion: {elapsed_steps}/{motion_steps} substeps; "
                    f"simulation time {elapsed_steps * self.forward.dt:.3f}s.",
                    flush=True,
                )
            if previous_steps < motion_steps <= elapsed_steps:
                wp.synchronize_device(self.scene["model"].device)
                print(
                    "Prescribed poke motion complete. Holding the final tool pose while physics continues. "
                    "Use 'Restart poke' in the example controls to replay it.",
                    flush=True,
                )
        else:
            p = self.problem
            if p.value is None:
                p.prepare()
                self._print_training_timing("Initial evaluation")
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
                    self._print_training_timing("Optimizer update")
                    self._save_results()
        self.last_step_seconds = perf_counter() - started
        self._draw_step = True

    def _restart_forward(self):
        """Restart the prescribed poke with fresh states and the existing solver factors."""
        self.forward = self._Forward(self.scene, self.xy)
        self.frame = 0
        self.last_step_seconds = 0.0
        self._draw_step = False
        self._first_frame = True
        print("Poke restarted at its initial pose; existing solver setup is reused.", flush=True)

    def _print_training_timing(self, label):
        p, timing = self.problem, self.problem.last_training_timing
        print(
            f"{label}: {timing['seconds']:.2f}s total; "
            f"forward {timing['forward_seconds']:.2f}s ({timing['forward_rollouts']} rollouts), "
            f"backward {timing['backward_seconds']:.2f}s ({timing['backward_rollouts']} rollouts). "
            f"Each rollout: {p.frames} recorded frames × {p.substeps} substeps = {p.steps} solver steps. "
            "Viewer FPS includes these offline training pauses.",
            flush=True,
        )

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
        ui.text(
            "Swept mesh contact; kinematic probe; frictionless; forward only"
            if self.swept_contact
            else "Original training demo's mesh contact force; penetration is possible"
        )
        ui.text(f"Young's modulus: {self.young / 1000:.3f} kPa; Poisson ratio: {self.poisson:.3f}")
        ui.text("Hand-authored reference robot trajectory; no measured probe poses")
        ui.text("White: simulated die; blue: recorded scan")
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
            elapsed_steps = self.forward.elapsed_steps
            motion_steps = len(self.scene["tool_positions"]) - 1
            ui.text(f"Simulation time: {elapsed_steps * self.forward.dt:.3f} s")
            ui.text(f"Poke motion: {min(elapsed_steps, motion_steps)}/{motion_steps} substeps")
            if elapsed_steps >= motion_steps:
                ui.text("Prescribed motion complete; holding the final tool pose")
            if ui.button("Restart poke"):
                self._restart_forward()
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
        ui.text("Viewer FPS measures playback updates and includes training pauses")
        ui.text(f"Each training rollout: {p.frames} frames × {p.substeps} substeps = {p.steps} solver steps")
        if p.last_training_timing is not None:
            timing = p.last_training_timing
            ui.text(f"Latest training evaluation/update: {timing['seconds']:.2f} s")
            ui.text(
                f"Forward: {timing['forward_seconds']:.2f} s / {timing['forward_rollouts']} rollouts; "
                f"backward: {timing['backward_seconds']:.2f} s / {timing['backward_rollouts']} rollouts"
            )
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
            "version": 5,
            "physics": self.scene["physics"],
            "scene_sha256": self.scene_hash,
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
            "contact_radius_m": self.contact_radius,
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
            "last_training_timing": p.last_training_timing,
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
            if self.scene["solver"].read_status() & ~1:
                raise AssertionError("Forward inference failed a projection or activation check")
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
            help="Use the separately prepared reference mesh and full-scan objective; solver settings are unchanged.",
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
            "--velocity-damping", type=float, help="Retained velocity per substep (default: 0.30, as in the reference)."
        )
        parser.add_argument(
            "--show-target", action="store_true", help="Overlay the recorded target at the current frame."
        )
        parser.add_argument(
            "--swept-contact",
            action="store_true",
            help="Enable swept mesh collision response with a kinematic probe. Requires --forward-only; friction=0.",
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
            help="Physics substeps per 30 Hz recorded frame (default: 30; inference: checkpoint value).",
        )
        parser.add_argument(
            "--friction", type=float, default=None, help="Coulomb coefficient (default: 0.5, or checkpoint value)."
        )
        parser.add_argument(
            "--contact-radius",
            type=float,
            default=None,
            help="Mesh contact force radius [m] (default: 0.00055, from reference dice_material_optimize.py; "
            "inference: checkpoint value).",
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
            "--save-alignment", help="Training JSON path (default: outputs/diff_phd_poke/alignment_fast_v5.json)."
        )
        forward = parser.add_mutually_exclusive_group()
        forward.add_argument(
            "--load-alignment", help="Run untaped live inference using a compatible saved alignment JSON."
        )
        forward.add_argument(
            "--forward-only",
            action="store_true",
            help="Run live simulation at --initial-xy and the selected material, without training or a checkpoint.",
        )
        parser.add_argument(
            "--output", help="Optional prefix for the training report and initial/optimized trajectories."
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
