# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Learn a gait with eight leg muscles and fixed leg/torso materials."""

from contextlib import nullcontext

import newton
import numpy as np
import warp as wp

from ...geometry.bear import load_bear_geometry
from .progress import _Progress
from .solver import SolverDiffPhD

DEFAULT_FRAMES = 100
DEFAULT_DT = 0.008
DEFAULT_TRAIN_ITERS = 20
FREQUENCY = 2.0
AMPLITUDE = 0.3


@wp.kernel
def muscle_commands(
    parameters: wp.array[wp.float64],
    groups: wp.array[wp.int32],
    time: wp.float64,
    activations: wp.array[wp.float32],
):
    i = wp.tid()
    group = groups[i]
    activation = wp.float64(0.0)
    if group >= 0:
        phase = wp.float64(2.0 * wp.pi * FREQUENCY) * time
        signal = parameters[2 * group] * wp.sin(phase) + parameters[2 * group + 1] * wp.cos(phase)
        activation = wp.float64(AMPLITUDE) * wp.tanh(signal)
    activations[i] = wp.float32(activation)


@wp.kernel
def center_velocity(
    velocities: wp.array[wp.vec3],
    masses: wp.array[wp.float64],
    frame: int,
    centers: wp.array[wp.vec3d],
):
    i = wp.tid()
    wp.atomic_add(centers, frame, masses[i] * wp.vec3d(velocities[i]))


@wp.kernel
def walking_loss(centers: wp.array[wp.vec3d], scale: wp.float64, loss: wp.array[wp.float64]):
    i = wp.tid()
    velocity = centers[i]
    # Foot lifting needs vertical motion; keep drift penalties below forward reward.
    epsilon = wp.float64(0.001)
    lateral = wp.sqrt(velocity[1] * velocity[1] + epsilon * epsilon) - epsilon
    vertical = wp.sqrt(velocity[2] * velocity[2] + epsilon * epsilon) - epsilon
    wp.atomic_add(loss, 0, scale * (wp.float64(0.1) * (lateral + vertical) - velocity[0]))


@wp.kernel
def control_penalty(parameters: wp.array[wp.float64], loss: wp.array[wp.float64]):
    i = wp.tid()
    wp.atomic_add(loss, 0, wp.float64(1.0e-5 / 16.0) * parameters[i] * parameters[i])


def bear_mesh(resolution=10):
    """Classify the baked geometric cells into four legs and eight muscle groups."""
    vertices, tets, faces, owners, cell_centers, source_volume, origin, extent = load_bear_geometry(resolution)
    # At resolution ten this is exactly the bottom two layers of cells.
    legs = cell_centers[:, 2] < origin[2] + extent[2] / 3.0
    cell_leg_ids = np.where(legs, 2 * (cell_centers[:, 0] > -0.5) + (cell_centers[:, 1] > 0.0), -1)
    leg_ids = np.repeat(cell_leg_ids, 6).astype(np.int32)
    muscle_groups = np.full(len(tets), -1, dtype=np.int32)
    centroids = vertices[tets].mean(axis=1)
    for leg in range(4):
        selected = cell_leg_ids == leg
        if not selected.any():
            raise ValueError("bear resolution loses a leg; choose another resolution")
        # Split through a cell's interior when the leg is only one cell wide.
        split = 0.5 * (cell_centers[selected, 0].min() + cell_centers[selected, 0].max())
        members = leg_ids == leg
        muscle_groups[members] = 2 * leg + (centroids[members, 0] >= split)
    if not np.array_equal(np.unique(muscle_groups[muscle_groups >= 0]), np.arange(8)):
        raise ValueError("bear resolution loses a muscle group; choose another resolution")
    face_materials = (leg_ids[owners] >= 0).astype(np.int32)
    return vertices, tets, faces, leg_ids, muscle_groups, face_materials, source_volume


class BearProblem:
    """Train an eight-muscle gait or advance its live physics.

    Geometry is the bundled resolution-ten Tet4 asset; nominal materials,
    contact, actuation, loss and optimizer are defined by this scene.
    Inference retains two states and one control without a recorded trajectory.
    """

    DEFAULT_FRAMES = DEFAULT_FRAMES
    DEFAULT_DT = DEFAULT_DT
    DEFAULT_TRAIN_ITERS = DEFAULT_TRAIN_ITERS
    FREQUENCY = FREQUENCY
    AMPLITUDE = AMPLITUDE

    def __init__(
        self,
        *,
        resolution=10,
        frames=DEFAULT_FRAMES,
        dt=DEFAULT_DT,
        device=None,
        progress=False,
        training=True,
        parameters=None,
    ):
        if isinstance(frames, bool) or int(frames) != frames or frames < 1 or not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("frames must be a positive integer and dt must be finite and positive")
        if not isinstance(training, (bool, np.bool_)):
            raise ValueError("training must be a boolean")
        initial = (
            np.random.default_rng(42).uniform(-0.5, 0.5, 16)
            if parameters is None
            else np.asarray(parameters, dtype=np.float64)
        )
        if initial.shape != (16,) or not np.isfinite(initial).all():
            raise ValueError("Provide sixteen finite sine/cosine controller weights")
        self.training = bool(training)
        self.elapsed_steps = 0
        self.resolution, self.frames, self.dt, self.progress = resolution, int(frames), dt, progress
        self.initial_x = initial.copy()
        self.x = initial.copy()
        vertices, self.tets, self.faces, self.leg_ids, self.group_indices, self.face_materials, source_volume = (
            bear_mesh(resolution)
        )
        self.vertices = vertices * 0.05
        self.vertices[:, 2] += 0.002 - self.vertices[:, 2].min()
        self.length = float(np.ptp(self.vertices[:, 0]))
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        for point in self.vertices:
            builder.add_particle(wp.vec3(*point), wp.vec3(), mass=0.0, radius=0.001)
        points = self.vertices[self.tets]
        volumes = np.linalg.det(np.stack([points[:, j] - points[:, 0] for j in (1, 2, 3)], axis=-1)) / 6.0
        # Preserve source mass when changing the coarse geometric resolution.
        density = 500.0 * source_volume * 0.05**3 / volumes.sum()
        # PD uses these nominal moduli to set projection-energy weights.
        nominal_poisson = 0.35
        for i, tet in enumerate(self.tets):
            nominal_young = 5.0e4 if self.leg_ids[i] >= 0 else 1.5e5
            volume = builder.add_tetrahedron(
                *map(int, tet),
                k_mu=nominal_young / (2.0 * (1.0 + nominal_poisson)),
                k_lambda=nominal_young * nominal_poisson / ((1.0 + nominal_poisson) * (1.0 - 2.0 * nominal_poisson)),
                k_damp=0.0,
            )
            if volume <= 0.0:
                raise ValueError("Bear tetrahedra must have positive reference volumes")
            for node in tet:
                builder.particle_mass[node] += density * volume / 4.0
        for face in self.faces:
            builder.add_triangle(*map(int, face), tri_ke=0.0, tri_ka=0.0, tri_kd=0.0)
        self.model = builder.finalize(device=device, requires_grad=self.training)
        self.model.tet_materials.requires_grad = False
        device = self.model.device
        self.mass_weights = self.model.particle_mass.numpy().astype(np.float64)
        self.mass_weights /= self.mass_weights.sum()
        self.groups = wp.array(self.group_indices, dtype=wp.int32, device=device)
        self.parameters = wp.array(self.initial_x, dtype=wp.float64, device=device, requires_grad=self.training)
        self.controls = [self.model.control() for _ in range(self.frames if self.training else 1)]
        self.states = [self.model.state() for _ in range(self.frames + 1 if self.training else 2)]
        for state in self.states:
            state.particle_f.requires_grad = False
        self.solver = SolverDiffPhD(
            self.model,
            material="pd",
            ground_plane=(0.0, 0.0, 1.0, 0.0),
            friction=0.7,
            contact_thickness=0.0001,
            self_contact=False,
            soft_contact=False,
            tolerance=1.0e-8,
            max_iterations=200,
            mass_damping=2.0,
            muscle_directions=np.tile([0.0, 0.0, 1.0], (len(self.tets), 1)),
            muscle_stiffness=np.where(self.group_indices >= 0, 7.5e4, 0.0),
        )
        self.contact_frames = 0
        if not self.training:
            return
        self.masses = wp.array(self.mass_weights, dtype=wp.float64, device=device)
        self.centers = wp.zeros(self.frames, dtype=wp.vec3d, device=device, requires_grad=True)
        self.loss = wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True)
        self.value, self.gradient, self.initial_positions = self.evaluate(self.x, stage="initial gait")
        self.positions = self.initial_positions.copy()
        self.initial_loss = self.value
        self.history = [self.value]
        self.momentum = np.zeros(16)
        self.variance = np.zeros(16)
        self.iteration = 0
        self.finished = False
        self.stop_reason = "training"

    def advance(self):
        """Advance inference by one timestep and return the current state.

        The gait phase follows elapsed time beyond the training horizon. Only
        two particle states and one control buffer are retained.
        """
        if self.training:
            raise RuntimeError("advance() requires training=False")
        wp.launch(
            muscle_commands,
            self.model.tet_count,
            inputs=[self.parameters, self.groups, self.elapsed_steps * self.dt],
            outputs=[self.controls[0].tet_activations],
            device=self.model.device,
        )
        self.solver.step(self.states[0], self.states[1], self.controls[0], None, self.dt)
        self.states[0], self.states[1] = self.states[1], self.states[0]
        self.elapsed_steps += 1
        self.contact_frames += self.solver.last_step_info.contacts > 0
        return self.states[0]

    def evaluate(self, parameters, gradients=True, *, stage="gait", save_trajectory=True):
        """Return walking loss, control gradient, and optional observed positions."""
        if not self.training:
            raise RuntimeError("evaluate() requires training=True; use advance() for inference")
        loss, tape, backward = self._rollout(parameters, gradients, stage)
        gradient = self._differentiate(tape, backward) if tape is not None else None
        positions = np.stack([state.particle_q.numpy() for state in self.states]) if save_trajectory else None
        return loss, gradient, positions

    def _rollout(self, parameters, record, stage):
        parameters = np.asarray(parameters, dtype=np.float64)
        if parameters.shape != (16,) or not np.isfinite(parameters).all():
            raise ValueError("Provide sixteen finite sine/cosine controller weights")
        self.parameters.assign(parameters)
        self.centers.zero_()
        self.loss.zero_()
        tape = wp.Tape() if record else None
        backward = _Progress(f"bear: {stage} / backward", self.frames, self.progress)
        self.contact_frames = 0
        with _Progress(f"bear: {stage} / forward", self.frames, self.progress) as forward:
            with tape if tape is not None else nullcontext():
                for frame in range(self.frames):
                    if tape is not None and self.progress:
                        tape.record_func(lambda done=self.frames - frame: backward.update(done), [])
                    wp.launch(
                        muscle_commands,
                        self.model.tet_count,
                        inputs=[self.parameters, self.groups, frame * self.dt],
                        outputs=[self.controls[frame].tet_activations],
                    )
                    self.solver.step(self.states[frame], self.states[frame + 1], self.controls[frame], None, self.dt)
                    self.contact_frames += self.solver.last_step_info.contacts > 0
                    wp.launch(
                        center_velocity,
                        self.model.particle_count,
                        inputs=[self.states[frame + 1].particle_qd, self.masses, frame],
                        outputs=[self.centers],
                    )
                    forward.update(frame + 1)
                wp.launch(walking_loss, self.frames, inputs=[self.centers, self.dt / self.length], outputs=[self.loss])
                wp.launch(control_penalty, 16, inputs=[self.parameters], outputs=[self.loss])
            loss = float(self.loss.numpy()[0])
        if not np.isfinite(loss):
            raise FloatingPointError("Nonfinite bear walking objective")
        return loss, tape, backward

    def _differentiate(self, tape, backward):
        with backward:
            try:
                tape.backward(self.loss)
                gradient = self.parameters.grad.numpy().copy()
            finally:
                tape.zero()
        if not np.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite bear controller gradient")
        return gradient

    def optimize_step(self):
        """Take an Adam direction with backtracking and retain only improved gaits."""
        if not self.training:
            raise RuntimeError("optimize_step() requires training=True")
        if self.finished:
            return
        momentum = 0.9 * self.momentum + 0.1 * self.gradient
        variance = 0.999 * self.variance + 0.001 * self.gradient**2
        mean = momentum / (1.0 - 0.9 ** (self.iteration + 1))
        rms = np.sqrt(variance / (1.0 - 0.999 ** (self.iteration + 1)))
        direction = -0.05 * mean / (rms + 1.0e-8)
        if np.dot(direction, self.gradient) >= 0.0:
            direction = -0.05 * self.gradient / max(np.linalg.norm(self.gradient), 1.0e-12)
            momentum = self.gradient * 0.1
            variance = self.gradient**2 * 0.001
        accepted = False
        for backtrack in range(8):
            candidate = self.x + 0.5**backtrack * direction
            try:
                value, tape, backward = self._rollout(
                    candidate, True, f"update {self.iteration + 1}, trial {backtrack + 1}"
                )
            except SolverDiffPhD.ConvergenceError:
                continue
            if value < self.value and value <= self.value + 1.0e-4 * np.dot(self.gradient, candidate - self.x):
                accepted = True
                break
        if not accepted:
            self.finished = True
            self.stop_reason = "line search stopped"
            return
        gradient = self._differentiate(tape, backward)
        positions = np.stack([state.particle_q.numpy() for state in self.states])
        self.x, self.value, self.gradient, self.positions = candidate, value, gradient, positions
        self.momentum, self.variance = momentum, variance
        self.iteration += 1
        self.history.append(value)
        if np.linalg.norm(gradient, ord=np.inf) < 1.0e-7:
            self.finished = True
            self.stop_reason = "gradient converged"

    def check_gradient(self, epsilon=0.0002):
        """Check two full-rollout control directions at the current contact branch."""
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        checks = []
        for seed in (104, 105):
            direction = np.random.default_rng(seed).normal(size=16)
            direction /= np.linalg.norm(direction)
            plus = self.evaluate(
                self.x + epsilon * direction, gradients=False, stage="gradient +", save_trajectory=False
            )[0]
            minus = self.evaluate(
                self.x - epsilon * direction, gradients=False, stage="gradient -", save_trajectory=False
            )[0]
            numeric = (plus - minus) / (2.0 * epsilon)
            analytic = float(self.gradient @ direction)
            relative = float(abs(analytic - numeric) / max(abs(analytic), abs(numeric), 1.0e-8))
            checks.append({"analytic": analytic, "finite_difference": numeric, "relative_error": relative})
        result = {"directions": checks, "relative_error": max(item["relative_error"] for item in checks)}
        if result["relative_error"] > 0.05:
            raise AssertionError(f"Bear controller gradient check failed; contact branches may have changed: {result}")
        return result
