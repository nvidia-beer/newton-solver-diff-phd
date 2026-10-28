# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fit a die's placement or homogeneous stiffness to a recorded poke sequence."""

from contextlib import nullcontext
from time import perf_counter

import newton
import numpy as np
import warp as wp

from .progress import _Progress


@wp.kernel
def _assign_material(
    parameters: wp.array[wp.float64],
    young_reference: wp.float64,
    poisson_reference: wp.float64,
    fit_young: int,
    die_tets: wp.array[int],
    original: wp.array2d[float],
    materials: wp.array2d[float],
):
    i = wp.tid()
    mu = float(original[i, 0])
    lame = float(original[i, 1])
    if die_tets[i] != 0:
        young = wp.float64(young_reference)
        poisson = wp.float64(poisson_reference)
        if fit_young != 0:
            young *= wp.exp(parameters[0])
        else:
            poisson = wp.float64(parameters[0])
        mu = float(young / (wp.float64(2.0) * (wp.float64(1.0) + poisson)))
        lame = float(young * poisson / ((wp.float64(1.0) + poisson) * (wp.float64(1.0) - wp.float64(2.0) * poisson)))
    materials[i, 0] = mu
    materials[i, 1] = lame
    materials[i, 2] = original[i, 2]


@wp.kernel
def _initialize_positions(
    rest: wp.array[wp.vec3],
    parameters: wp.array[wp.float64],
    reference: wp.vec2d,
    soft_count: int,
    positions: wp.array[wp.vec3],
):
    i = wp.tid()
    value = wp.vec3d(rest[i])
    if i < soft_count:
        value += wp.vec3d(parameters[0] - reference[0], parameters[1] - reference[1], wp.float64(0.0))
    positions[i] = wp.vec3(value)


@wp.kernel
def command_poke_velocities(
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    endpoints: wp.array2d[wp.vec3],
    step: int,
    soft_count: int,
    dt: float,
    out: wp.array[wp.vec3],
):
    i = wp.tid()
    if i < soft_count:
        out[i] = velocities[i]
    else:
        out[i] = (endpoints[step + 1, i - soft_count] - positions[i]) / dt


@wp.kernel
def damp_poke_velocities(
    velocities: wp.array[wp.vec3],
    inverse_mass: wp.array[float],
    damping: float,
    out: wp.array[wp.vec3],
):
    i = wp.tid()
    value = velocities[i]
    if inverse_mass[i] > 0.0:
        value *= damping
    out[i] = value


@wp.kernel
def _sample_surface(
    q: wp.array[wp.vec3],
    indices: wp.array[wp.vec4i],
    weights: wp.array[wp.vec4d],
    points: wp.array[wp.vec3d],
):
    i = wp.tid()
    value = wp.vec3d(0.0)
    for j in range(4):
        if indices[i][j] >= 0:
            value += weights[i][j] * wp.vec3d(q[indices[i][j]])
    points[i] = value


@wp.kernel(enable_backward=False)
def _nearest_points(source: wp.array[wp.vec3d], target: wp.array[wp.vec3d], nearest: wp.array[int]):
    i = wp.tid()
    best_distance = wp.float64(wp.inf)
    best = int(0)
    for j in range(target.shape[0]):
        distance = wp.length_sq(source[i] - target[j])
        if distance < best_distance:
            best_distance = distance
            best = j
    nearest[i] = best


@wp.kernel
def _chamfer_forward(
    points: wp.array[wp.vec3d],
    target: wp.array[wp.vec3d],
    nearest: wp.array[int],
    loss: wp.array[wp.float64],
):
    i = wp.tid()
    delta = points[i] - target[nearest[i]]
    wp.atomic_add(loss, 0, wp.float64(0.5) * wp.dot(delta, delta) / wp.float64(points.shape[0]))


@wp.kernel
def _chamfer_reverse(
    points: wp.array[wp.vec3d],
    target: wp.array[wp.vec3d],
    nearest: wp.array[int],
    loss: wp.array[wp.float64],
):
    i = wp.tid()
    delta = points[nearest[i]] - target[i]
    wp.atomic_add(loss, 0, wp.float64(0.5) * wp.dot(delta, delta) / wp.float64(target.shape[0]))


class PokeAlignmentProblem:
    """Optimize die placement or stiffness using native implicit adjoints.

    The die occupies the first ``soft_count`` particles; every remaining
    particle belongs to a prescribed tool. All targets and tool endpoints are
    fixed world coordinates [m]. ``fit_mode="position"`` varies initial x/y
    translation. ``fit_mode="stiffness"`` holds ``initial_xy`` fixed and varies
    one log Young's modulus, with fixed ``poisson`` and unchanged tool material.
    ``fit_mode="compressibility"`` instead varies Poisson's ratio at fixed
    placement and Young's modulus; material parameters are never jointly fit.
    This calibrates stiffness within the supplied geometry, mass and contact
    model; motion alone need not identify the physical material uniquely.
    The loss sums symmetric, mean squared vertex Chamfer distances [m²] over
    observed frames. Optional fixed tetrahedral embeddings let the loss use
    sampled recorded-surface vertices instead of the coarse FEM boundary.

    Contact features and nearest-neighbor associations are held fixed in the
    derivative. No finite-difference gradients or adjoint clipping are used by
    the optimizer. Bounded L-BFGS uses a projected search direction and Armijo
    backtracking; ``optimizer="gradient_descent"`` retains the legacy update.
    ``velocity_damping`` retains this fraction of each free velocity after every
    physical substep, and its chain rule is included in the recorded rollout.
    Construction performs no rollout; call :meth:`prepare` when ready to evaluate
    the initial guess.
    """

    def __init__(
        self,
        model,
        solver,
        soft_count,
        surface_indices,
        tool_positions,
        targets,
        *,
        initial_xy=(0.73, 0.115),
        reference_xy=(0.70, 0.089),
        frame_dt=1.0 / 30.0,
        substeps=1,
        progress=False,
        bounds=((0.62, 0.80), (0.02, 0.18)),
        surface_embedding=None,
        optimizer="lbfgs",
        history_size=5,
        velocity_damping=1.0,
        fit_mode="position",
        young=5e5,
        poisson=0.4,
        young_bounds=(2e4, 2e6),
        poisson_bounds=(0.05, 0.45),
    ):
        if isinstance(substeps, bool) or int(substeps) != substeps or substeps < 1:
            raise ValueError("substeps must be a positive integer")
        if not np.isfinite(frame_dt) or frame_dt <= 0.0:
            raise ValueError("frame_dt must be finite and positive")
        if isinstance(soft_count, bool) or int(soft_count) != soft_count or not 0 < soft_count < model.particle_count:
            raise ValueError("soft_count must leave nonempty die and tool particles")
        if solver.model is not model or not solver.integrate_pinned_velocities:
            raise ValueError("Use this model's SolverDiffPhD with integrate_pinned_velocities=True")
        if optimizer not in ("lbfgs", "gradient_descent", "gd"):
            raise ValueError("optimizer must be 'lbfgs' or 'gradient_descent'")
        if isinstance(history_size, bool) or int(history_size) != history_size or history_size < 1:
            raise ValueError("history_size must be a positive integer")
        if not np.isfinite(velocity_damping) or not 0.0 <= velocity_damping <= 1.0:
            raise ValueError("velocity_damping must be finite and between zero and one")
        if fit_mode not in ("position", "stiffness", "compressibility"):
            raise ValueError("fit_mode must be 'position', 'stiffness' or 'compressibility'")
        if not np.isfinite(young) or young <= 0.0 or not np.isfinite(poisson) or not 0.0 <= poisson < 0.5:
            raise ValueError("young must be finite and positive and poisson must be in [0,0.5)")
        young_limits = np.asarray(young_bounds, dtype=np.float64)
        if (
            young_limits.shape != (2,)
            or not np.isfinite(young_limits).all()
            or not 0.0 < young_limits[0] < young_limits[1]
            or (fit_mode == "stiffness" and not young_limits[0] <= young <= young_limits[1])
        ):
            raise ValueError("young_bounds must be positive ordered bounds containing the stiffness initial guess")
        poisson_limits = np.asarray(poisson_bounds, dtype=np.float64)
        if (
            poisson_limits.shape != (2,)
            or not np.isfinite(poisson_limits).all()
            or not 0.0 <= poisson_limits[0] < poisson_limits[1] < 0.5
            or (fit_mode == "compressibility" and not poisson_limits[0] <= poisson <= poisson_limits[1])
        ):
            raise ValueError("poisson_bounds must be ordered in [0,0.5) and contain the compressibility initial guess")
        self.model, self.solver, self.device = model, solver, model.device
        self.fit_mode, self.poisson_reference = fit_mode, float(poisson)
        self.young_reference, self.young_bounds = float(young), young_limits.copy()
        self.poisson_bounds = poisson_limits.copy()
        self.parameter_names = {"position": ("x", "y"), "stiffness": ("log_young",), "compressibility": ("poisson",)}[
            fit_mode
        ]
        self.optimizer = "gradient_descent" if optimizer == "gd" else optimizer
        self.history_size, self.velocity_damping = int(history_size), float(velocity_damping)
        self._curvature = []
        self.soft_count = int(soft_count)
        self.frame_dt, self.substeps = float(frame_dt), int(substeps)
        self.dt, self.progress = self.frame_dt / self.substeps, bool(progress)
        self.reference_xy = np.array(reference_xy, dtype=np.float64, copy=True)
        self.initial_x = np.array(initial_xy, dtype=np.float64, copy=True)
        self.bounds = np.array(bounds, dtype=np.float64, copy=True)
        if (
            self.reference_xy.shape != (2,)
            or self.initial_x.shape != (2,)
            or self.bounds.shape != (2, 2)
            or not np.isfinite(self.reference_xy).all()
            or not np.isfinite(self.initial_x).all()
            or not np.isfinite(self.bounds).all()
            or np.any(self.bounds[:, 0] >= self.bounds[:, 1])
            or np.any(self.initial_x < self.bounds[:, 0])
            or np.any(self.initial_x > self.bounds[:, 1])
        ):
            raise ValueError("Provide finite reference/initial x/y and ordered bounds containing the initial guess")
        if np.any(model.particle_inv_mass.numpy()[self.soft_count :] != 0.0):
            raise ValueError("All tool particles must have zero inverse mass")
        target_arrays = [np.array(points, dtype=np.float64, copy=True) for points in targets]
        if not target_arrays or any(
            points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all()
            for points in target_arrays
        ):
            raise ValueError("targets must contain nonempty finite (point_count,3) observed frames")
        self.frames = len(target_arrays)
        self.steps = self.frames * self.substeps
        endpoints = np.array(tool_positions, dtype=np.float32, copy=True)
        if (
            endpoints.shape != (self.steps + 1, model.particle_count - self.soft_count, 3)
            or not np.isfinite(endpoints).all()
        ):
            raise ValueError("tool_positions must contain every substep endpoint, including the initial pose")
        rest = model.particle_q.numpy().copy()
        if not np.allclose(rest[self.soft_count :], endpoints[0], rtol=0.0, atol=1e-7):
            raise ValueError("The initial tool endpoints must match the model's initial tool positions")
        lower = rest[: self.soft_count, :2].min(axis=0) + self.bounds[:, 0] - self.reference_xy
        upper = rest[: self.soft_count, :2].max(axis=0) + self.bounds[:, 1] - self.reference_xy
        # Parameters are float64, but their translated native positions are
        # float32. Sub-ULP line-search trials cannot resolve a reliable fit step.
        spatial_scale = np.maximum(np.abs(lower), np.abs(upper)).astype(np.float32)
        self.position_resolution_m = np.spacing(spatial_scale).astype(np.float64)
        self.initial_xy, self.xy_bounds = self.initial_x.copy(), self.bounds.copy()
        self._fixed_xy = wp.array(self.initial_xy, dtype=wp.float64, device=self.device)
        if fit_mode != "position":
            tets = model.tet_indices.numpy().reshape(-1, 4)
            die_tets = np.all(tets < self.soft_count, axis=1)
            if not die_tets.any() or np.any(np.any(tets < self.soft_count, axis=1) != die_tets):
                raise ValueError("Material fitting requires die tetrahedra separate from the prescribed tool")
            self._die_tets = wp.array(die_tets.astype(np.int32), dtype=int, device=self.device)
            self._original_materials = wp.clone(model.tet_materials, requires_grad=False)
            model.tet_materials.requires_grad = True
            self.initial_x = np.array([0.0 if fit_mode == "stiffness" else self.poisson_reference])
            self.bounds = (np.log(young_limits / self.young_reference) if fit_mode == "stiffness" else poisson_limits)[
                None, :
            ]
        self.rest = wp.array(rest, dtype=wp.vec3, device=self.device)
        self.tool_positions = wp.array(endpoints, dtype=wp.vec3, device=self.device)
        self.targets = [wp.array(points, dtype=wp.vec3d, device=self.device) for points in target_arrays]
        self.target_positions = target_arrays
        self.surface_indices = np.array(surface_indices, dtype=np.int32, copy=True)
        if (
            self.surface_indices.ndim != 1
            or not len(self.surface_indices)
            or np.any(self.surface_indices < 0)
            or np.any(self.surface_indices >= self.soft_count)
        ):
            raise ValueError("surface_indices must index a nonempty subset of die particles")
        if surface_embedding is None:
            ids = np.full((len(self.surface_indices), 4), -1, dtype=np.int32)
            ids[:, 0] = self.surface_indices
            weights = np.zeros_like(ids, dtype=np.float64)
            weights[:, 0] = 1.0
        else:
            ids = np.asarray(surface_embedding[0], dtype=np.int32)
            weights = np.asarray(surface_embedding[1], dtype=np.float64)
            if (
                ids.ndim != 2
                or ids.shape[1] != 4
                or not len(ids)
                or weights.shape != ids.shape
                or np.any(ids < 0)
                or np.any(ids >= self.soft_count)
                or not np.isfinite(weights).all()
                or not np.allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-6)
            ):
                raise ValueError("surface_embedding requires valid (sample_count,4) die indices and affine weights")
        self.embedding_indices = wp.array(ids, dtype=wp.vec4i, device=self.device)
        self.embedding_weights = wp.array(weights, dtype=wp.vec4d, device=self.device)
        embedded_rest = np.sum(rest[np.maximum(ids, 0)] * weights[:, :, None], axis=1)
        self.surface_extent_m = float(np.max(np.ptp(embedded_rest, axis=0)))
        self.parameters = wp.array(self.initial_x, dtype=wp.float64, device=self.device, requires_grad=True)
        self.states = [model.state(requires_grad=True) for _ in range(self.steps + 1)]
        self.command_states = []
        for i in range(self.steps):
            command = model.state(requires_grad=True)
            command.particle_q = self.states[i].particle_q
            command.particle_f = self.states[i].particle_f
            self.command_states.append(command)
        for state in self.states:
            state.particle_f.requires_grad = False
            state.particle_f.zero_()
        self.solver_states = self.states[1:]
        if self.velocity_damping != 1.0:
            # Preserve raw solver outputs for its adjoint; damping writes a
            # distinct velocity buffer consumed by the next substep.
            self.solver_states = []
            for state in self.states[1:]:
                raw = model.state(requires_grad=True)
                raw.particle_q = state.particle_q
                raw.particle_f = state.particle_f
                self.solver_states.append(raw)
        self.control = model.control()
        self.surface_points = [
            wp.empty(len(ids), dtype=wp.vec3d, device=self.device, requires_grad=True) for _ in range(self.frames)
        ]
        self.forward_nearest = [wp.empty(len(ids), dtype=int, device=self.device) for _ in range(self.frames)]
        self.reverse_nearest = [wp.empty(len(points), dtype=int, device=self.device) for points in target_arrays]
        self.loss = wp.zeros(1, dtype=wp.float64, device=self.device, requires_grad=True)
        self.x = self.initial_x.copy()
        self.value = self.gradient = self.positions = self.initial_positions = self.initial_loss = None
        self.iteration, self.contact_steps = 0, 0
        self.history = []
        self.finished, self.stop_reason = False, "not prepared"
        self.step_size = {"position": 0.012, "stiffness": 0.25, "compressibility": 0.05}[fit_mode]
        self._tape = None

    def _rollout(self, parameters, *, record, stage):
        values = np.asarray(parameters, dtype=np.float64)
        if values.shape != self.initial_x.shape or not np.isfinite(values).all():
            raise ValueError(f"Expected finite parameters {self.parameter_names}")
        if self.fit_mode != "position":
            self._material_from_parameters(values)
        if self._tape is not None:
            self._tape.zero()
        self.parameters.assign(values)
        self.states[0].particle_qd.zero_()
        self.loss.zero_()
        tape = wp.Tape() if record else None
        self._tape = tape
        self.contact_steps = 0
        self.max_residual = 0.0
        if self.fit_mode != "position":
            # Invalidate the factor outside Tape; the differentiable material
            # assignment below supplies the native implicit material VJP.
            self.solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
        with tape if tape is not None else nullcontext():
            if self.fit_mode != "position":
                wp.launch(
                    _assign_material,
                    self.model.tet_count,
                    inputs=[
                        self.parameters,
                        self.young_reference,
                        self.poisson_reference,
                        int(self.fit_mode == "stiffness"),
                        self._die_tets,
                        self._original_materials,
                    ],
                    outputs=[self.model.tet_materials],
                    device=self.device,
                )
            wp.launch(
                _initialize_positions,
                self.model.particle_count,
                inputs=[
                    self.rest,
                    self.parameters if self.fit_mode == "position" else self._fixed_xy,
                    wp.vec2d(*self.reference_xy),
                    self.soft_count,
                ],
                outputs=[self.states[0].particle_q],
                device=self.device,
            )
        backward = _Progress(f"die alignment: {stage} / backward", self.frames, self.progress)
        with _Progress(f"die alignment: {stage} / forward", self.frames, self.progress) as forward:
            for frame in range(self.frames):
                with tape if tape is not None else nullcontext():
                    if tape is not None and self.progress:
                        tape.record_func(lambda done=self.frames - frame: backward.update(done), [])
                    for substep in range(self.substeps):
                        i = frame * self.substeps + substep
                        wp.launch(
                            command_poke_velocities,
                            self.model.particle_count,
                            inputs=[
                                self.states[i].particle_q,
                                self.states[i].particle_qd,
                                self.tool_positions,
                                i,
                                self.soft_count,
                                self.dt,
                            ],
                            outputs=[self.command_states[i].particle_qd],
                            device=self.device,
                        )
                        self.solver.step(self.command_states[i], self.solver_states[i], self.control, None, self.dt)
                        if self.velocity_damping != 1.0:
                            wp.launch(
                                damp_poke_velocities,
                                self.model.particle_count,
                                inputs=[
                                    self.solver_states[i].particle_qd,
                                    self.model.particle_inv_mass,
                                    self.velocity_damping,
                                ],
                                outputs=[self.states[i + 1].particle_qd],
                                device=self.device,
                            )
                        self.contact_steps += int(self.solver.last_step_info.contacts > 0)
                        self.max_residual = max(self.max_residual, self.solver.last_step_info.residual)
                    wp.launch(
                        _sample_surface,
                        len(self.embedding_indices),
                        inputs=[
                            self.states[(frame + 1) * self.substeps].particle_q,
                            self.embedding_indices,
                            self.embedding_weights,
                        ],
                        outputs=[self.surface_points[frame]],
                        device=self.device,
                    )
                # The selected closest-point branches are fixed for reverse mode.
                for source, target, nearest in (
                    (self.surface_points[frame], self.targets[frame], self.forward_nearest[frame]),
                    (self.targets[frame], self.surface_points[frame], self.reverse_nearest[frame]),
                ):
                    wp.launch(
                        _nearest_points, len(source), inputs=[source, target], outputs=[nearest], device=self.device
                    )
                with tape if tape is not None else nullcontext():
                    for kernel, nearest in (
                        (_chamfer_forward, self.forward_nearest[frame]),
                        (_chamfer_reverse, self.reverse_nearest[frame]),
                    ):
                        wp.launch(
                            kernel,
                            len(nearest),
                            inputs=[self.surface_points[frame], self.targets[frame], nearest],
                            outputs=[self.loss],
                            device=self.device,
                        )
                forward.update(frame + 1)
        value = float(self.loss.numpy()[0])
        if not np.isfinite(value):
            raise RuntimeError("Recorded-data alignment produced a nonfinite loss")
        return value, tape, backward

    def _backward(self, tape, progress):
        with progress:
            tape.backward(self.loss)
            gradient = self.parameters.grad.numpy().copy()
        if not np.isfinite(gradient).all():
            raise RuntimeError("Recorded-data alignment produced a nonfinite implicit gradient")
        return gradient

    def _positions(self):
        return np.stack([self.states[i * self.substeps].particle_q.numpy() for i in range(self.frames + 1)])

    def evaluate(self, parameters, gradients=True, *, save_trajectory=True, stage="trajectory"):
        """Return loss [m²], native parameter gradient, and optional frame positions [m]."""
        value, tape, progress = self._rollout(parameters, record=gradients, stage=stage)
        gradient = self._backward(tape, progress) if gradients else None
        return value, gradient, self._positions() if save_trajectory else None

    def _material_from_parameters(self, parameters):
        with np.errstate(over="ignore", under="ignore"):
            young = (
                float(self.young_reference * np.exp(parameters[0]))
                if self.fit_mode == "stiffness"
                else self.young_reference
            )
        poisson = float(parameters[0]) if self.fit_mode == "compressibility" else self.poisson_reference
        if not np.isfinite(young) or young <= 0.0 or not np.isfinite(poisson) or not 0.0 <= poisson < 0.5:
            raise ValueError("Parameters must decode to finite positive Young's modulus and Poisson ratio in [0,0.5)")
        return young, poisson

    @property
    def xy(self):
        """Accepted initial die placement [m], fixed during material fitting."""
        return (self.x if self.fit_mode == "position" else self.initial_xy).copy()

    @property
    def young(self):
        """Accepted homogeneous Young's modulus [Pa]."""
        return self._material_from_parameters(self.x)[0]

    @property
    def poisson(self):
        """Accepted Poisson's ratio, fixed except during compressibility fitting."""
        return self._material_from_parameters(self.x)[1]

    @property
    def rms_surface_error_m(self):
        """Frame-normalized symmetric Chamfer RMS [m] of the accepted trajectory."""
        return None if self.value is None else float(np.sqrt(self.value / self.frames))

    @property
    def normalized_rms_surface_error(self):
        """RMS divided by the longest initial sampled-surface extent; None for a point."""
        if self.value is None or self.surface_extent_m == 0.0:
            return None
        return self.rms_surface_error_m / self.surface_extent_m

    def prepare(self):
        """Evaluate initial parameters once, after the viewer can show its geometry."""
        if self.value is not None:
            return
        self.value, self.gradient, self.initial_positions = self.evaluate(self.x, stage="initial guess")
        self.positions = self.initial_positions.copy()
        self.initial_loss = self.value
        self.stop_reason = "training"
        self.history.append(
            {
                "iteration": 0,
                "xy": self.xy.tolist(),
                "young_pa": self.young,
                "poisson": self.poisson,
                "loss": self.value,
                "rms_surface_error_m": self.rms_surface_error_m,
            }
        )

    def _projected_gradient(self):
        blocked = ((self.x <= self.bounds[:, 0]) & (self.gradient > 0.0)) | (
            (self.x >= self.bounds[:, 1]) & (self.gradient < 0.0)
        )
        return np.where(blocked, 0.0, self.gradient)

    def _lbfgs_direction(self, projected_gradient):
        q = projected_gradient.copy()
        coefficients = []
        for s, y, rho in reversed(self._curvature):
            alpha = rho * float(s @ q)
            coefficients.append(alpha)
            q -= alpha * y
        if self._curvature:
            s, y, _ = self._curvature[-1]
            scale = float(s @ y) / float(y @ y)
        else:
            scale = self.step_size / float(np.linalg.norm(projected_gradient))
        direction = scale * q
        for (s, y, rho), alpha in zip(self._curvature, reversed(coefficients), strict=True):
            direction += s * (alpha - rho * float(y @ direction))
        direction *= -1.0
        direction[(self.x <= self.bounds[:, 0]) & (direction < 0.0)] = 0.0
        direction[(self.x >= self.bounds[:, 1]) & (direction > 0.0)] = 0.0
        if not np.isfinite(direction).all() or float(self.gradient @ direction) >= 0.0:
            self._curvature.clear()
            direction = -self.step_size * projected_gradient / np.linalg.norm(projected_gradient)
        return direction

    def optimize_step(self):
        """Take one bounded analytic-gradient update, retaining its accepted tape."""
        self.prepare()
        if self.finished:
            return False
        lbfgs = self.optimizer == "lbfgs"
        projected_gradient = self._projected_gradient() if lbfgs else self.gradient
        magnitude = float(np.linalg.norm(projected_gradient))
        if magnitude < 1e-9:
            self.finished = True
            self.stop_reason = "projected gradient converged" if lbfgs else "gradient converged"
            return False
        direction = self._lbfgs_direction(projected_gradient) if lbfgs else -self.gradient / magnitude
        distance = 1.0 if lbfgs else self.step_size
        accepted_metrics = self.contact_steps, self.max_residual
        reason = "line search exhausted"
        started = perf_counter()
        for backtrack in range(20 if lbfgs else 6):
            candidate = np.clip(self.x + distance * direction, self.bounds[:, 0], self.bounds[:, 1])
            step = candidate - self.x
            if self.fit_mode == "position":
                unresolved = np.all(np.abs(step) < self.position_resolution_m)
            else:
                # Identical float32 Lame parameters produce the same simulation,
                # even if their float64 log-E values differ.
                materials = []
                for parameters in (self.x, candidate):
                    young, poisson = self._material_from_parameters(parameters)
                    materials.append(
                        np.array(
                            [young / (2 * (1 + poisson)), young * poisson / ((1 + poisson) * (1 - 2 * poisson))],
                            dtype=np.float32,
                        )
                    )
                unresolved = np.array_equal(*materials)
            if unresolved:
                reason = "state precision stagnation"
                break
            slope = float(self.gradient @ step)
            if lbfgs and slope >= 0.0:
                distance *= 0.5
                continue
            try:
                value, tape, progress = self._rollout(candidate, record=True, stage=f"trial {self.iteration + 1}")
                sufficient = value < self.value and value <= self.value + 1e-4 * slope
                if not (sufficient if lbfgs else value < self.value - 1e-12):
                    distance *= 0.5
                    continue
                gradient = self._backward(tape, progress)
            except self.solver.ConvergenceError:
                distance *= 0.5
                continue
            if lbfgs:
                change = gradient - self.gradient
                curvature = float(step @ change)
                if curvature > 1e-10 * np.linalg.norm(step) * np.linalg.norm(change):
                    self._curvature.append((step, change, 1.0 / curvature))
                    del self._curvature[: -self.history_size]
            self.x, self.value, self.gradient = candidate, value, gradient
            self.positions = self._positions()
            self.iteration += 1
            self.history.append(
                {
                    "iteration": self.iteration,
                    "xy": self.xy.tolist(),
                    "young_pa": self.young,
                    "poisson": self.poisson,
                    "loss": self.value,
                    "rms_surface_error_m": self.rms_surface_error_m,
                    "step": float(np.linalg.norm(step)),
                    "backtracks": backtrack,
                    "seconds": perf_counter() - started,
                }
            )
            if not lbfgs and backtrack == 0:
                limit = {"position": 0.024, "stiffness": 0.5, "compressibility": 0.1}[self.fit_mode]
                self.step_size = min(limit, self.step_size * 1.3)
            return True
        self.contact_steps, self.max_residual = accepted_metrics
        self.finished, self.stop_reason = True, reason
        return False

    def check_gradient(self, epsilon=None):
        """Check derivatives with steps of 0.1 mm, 0.01 log-E or 0.001 Poisson ratio."""
        if epsilon is None:
            epsilon = {"position": 1e-4, "stiffness": 1e-2, "compressibility": 1e-3}[self.fit_mode]
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        _, analytic, _ = self.evaluate(self.x, save_trajectory=False, stage="gradient check")
        numeric = np.empty_like(self.x)
        for axis in range(len(self.x)):
            delta = np.zeros_like(self.x)
            delta[axis] = epsilon
            plus = self.evaluate(self.x + delta, gradients=False, save_trajectory=False)[0]
            minus = self.evaluate(self.x - delta, gradients=False, save_trajectory=False)[0]
            numeric[axis] = (plus - minus) / (2.0 * epsilon)
        relative = np.abs(analytic - numeric) / np.maximum(np.maximum(np.abs(analytic), np.abs(numeric)), 1e-12)
        return {"analytic": analytic, "finite_difference": numeric, "relative_error": relative, "epsilon": epsilon}
