# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Residual-consistent differentiable projective heterogeneous dynamics."""

import gc
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import warp as wp
from newton import Contacts, Control, GeoType, Model, ParticleFlags, State
from newton.solvers import SolverBase

from . import kernels as k
from .adjoint import KKTSystem, record_adjoint
from .contact import ContactGeometry
from .elasticity import MODEL_COROTATED, MODEL_NEO_HOOKEAN
from .exceptions import ConvergenceError
from .muscle import accumulate_muscle_gradient, read_activations
from .response import (
    ContactConvergenceError,
    ContactResponse,
    ContactWorkLimit,
    physical_contact_residual,
    vector_update,
)
from .sparse import SparseInverse


@contextmanager
def _without_tape():
    tape = wp._src.context.runtime.tape
    wp._src.context.runtime.tape = None
    try:
        yield
    finally:
        wp._src.context.runtime.tape = tape


@dataclass(frozen=True)
class StepInfo:
    """Diagnostics for the last converged step."""

    iterations: int
    residual: float
    contacts: int
    geometry_iterations: int


class _JointLineSearch:
    """Replay joint position/force trials with device-side acceptance."""

    def __init__(self, solver, response, previous, velocity, force, dt, parameters):
        self.context = (response, previous, velocity, force, dt)
        self.parameters = parameters
        self.q = wp.empty(solver._n, dtype=wp.vec3d, device=solver.device)
        self.target = wp.empty_like(self.q)
        self.trial = wp.empty_like(self.q)
        self.old_lam = wp.empty_like(response.lam)
        self.direction = wp.empty(3 * (solver._n + response.count), dtype=wp.float64, device=solver.device)
        self.state = wp.empty(4, dtype=wp.int32, device=solver.device)
        self.bound = wp.empty(1, dtype=wp.float64, device=solver.device)
        self.result = wp.zeros(2, dtype=wp.float64, device=solver.device)
        self.residual = wp.empty_like(self.q)
        self.contact_force = wp.empty_like(self.q)

        def body():
            wp.launch(
                k.joint_trial,
                max(solver._n, 3 * response.count),
                inputs=[self.q, self.target, self.old_lam, self.direction, self.state, self.trial, response.lam],
                device=solver.device,
            )
            solver._evaluate_error(
                self.trial, previous, velocity, force, dt, response, self.residual, self.contact_force
            )
            wp.launch(
                k.accept_joint_trial,
                1,
                inputs=[
                    solver._norm,
                    self.bound,
                    solver.tolerance,
                    solver.contact_tolerance,
                    self.state,
                    self.result,
                ],
                device=solver.device,
            )

        # Keep unrelated CUDA destructors out of conditional graph capture.
        enabled = gc.isenabled()
        gc.disable()
        try:
            with wp.ScopedCapture(device=solver.device) as capture:
                wp.capture_while(self.state, body)
            self.graph = capture.graph
        finally:
            if enabled:
                gc.enable()

    def search(self, q, candidate, old_lam, direction, error, *, start, stop, newton=False):
        response = self.context[0]
        wp.copy(self.q, q)
        if not newton:
            wp.copy(self.target, candidate)
        wp.copy(self.old_lam, old_lam)
        wp.copy(self.direction, direction, count=direction.size)
        self.bound.assign([error])
        self.state.assign([1, start, stop, int(newton)])
        self.result.zero_()
        wp.capture_launch(self.graph)
        accepted, candidate_error = self.result.numpy()
        if accepted:
            wp.copy(candidate, self.trial)
            return float(candidate_error)
        wp.copy(response.lam, old_lam)
        return np.inf


class SolverDiffPhD(SolverBase):
    """Implicit tetrahedral dynamics with exact sparse inverse factors.

    The projective material uses corotated and closest-volume projection
    energies. ``neo_hookean`` uses the physical logarithmic Neo-Hookean
    residual with the same fixed PD preconditioner. Both solve backward
    Euler to a checked residual; the adjoint differentiates that residual.

    Ground, soft--soft and self-contact use swept vertex--face and edge--edge
    candidates, position-level Signorini constraints and Coulomb friction.
    Contact differentiation holds the detected features, barycentric weights,
    normals and stick/slip branch fixed. It is not a derivative through
    collision detection or contact creation. Rest shape, mass and timestep
    are fixed in backward. Public state arrays retain Newton's float32 type;
    assembly, projections and linear solves use float64 internally.

    Sparse factorization uses NumPy/Python preprocessing; resident inverse
    applications and mechanics use Warp on CPU or CUDA. Supported CUDA devices
    run contact iterations and line searches in conditional graphs. Outer
    convergence and geometry checks still require host decisions; an entire
    step cannot yet be captured.

    Args:
        model: Newton model containing position-only tetrahedral soft bodies.
        material: ``"pd"`` or ``"neo_hookean"``.
        ground_plane: Plane ``(nx, ny, nz, offset)`` with ``n dot x = offset``
            [m] after normalizing the normal, or None to disable ground.
        friction: Coulomb friction coefficient (dimensionless).
        contact_thickness: Surface separation [m].
        self_contact: Detect nonadjacent features within each connected body.
        soft_contact: Detect features between disconnected soft bodies.
        max_contacts: Contact row capacity; overflow raises an error.
        contact_compliance: Numerical contact compliance [m/N].
        tolerance: Maximum diagonally scaled force residual [m].
        contact_tolerance: Maximum contact residual [m].
        max_iterations: Maximum local/global iterations per geometry solve.
        max_geometry_iterations: Maximum swept-contact geometry refreshes.
        mass_damping: Mass-proportional damping coefficient [1/s].
        anderson_history: Bounded acceleration history; zero disables it.
        muscle_directions: Fixed rest-space fibre directions, shape
            ``(tet_count, 3)``. Directions with positive muscle stiffness are
            normalized. Omit together with ``muscle_stiffness`` to disable.
        muscle_stiffness: Fixed fibre stiffness [Pa], scalar or shape
            ``(tet_count,)``. Zero disables a tetrahedron's fibre energy.
        integrate_pinned_velocities: Advance zero-inverse-mass particles by
            ``dt * state_in.particle_qd`` [m] and preserve their prescribed
            velocities [m/s]. Otherwise pinned positions remain fixed and
            their output velocities are zero. Prescribed motion participates
            in swept contact and friction, and its velocity receives gradients
            with the same frozen-contact convention as dynamic particles.
        warm_start_contacts: Reuse converged forces for matching contact
            features as the next solve's initial guess. Forces are rotated
            into the new contact frames and projected onto the Coulomb cone;
            unmatched features start at zero. Final convergence checks are
            unchanged. Enable for persistent frictional grips.

    Configured fibres use ``control.tet_activations`` as contraction ``u``:
    target stretch is ``1-u``, zero is neutral, and negative values extend.
    Controls must be finite and at most one; disabled fibres require zero.
    With no control, ``model.tet_activations`` supplies the values. The implicit
    adjoint includes activation gradients; fibre directions and stiffness are
    fixed. Use distinct control arrays for time-varying inputs recorded under
    one Tape, rather than overwriting a taped intermediate array.
    """

    StepInfo = StepInfo
    ConvergenceError = ConvergenceError

    def __init__(
        self,
        model: Model,
        *,
        material: str = "pd",
        ground_plane: tuple[float, float, float, float] | None = (0.0, 0.0, 1.0, 0.0),
        friction: float = 0.4,
        contact_thickness: float = 0.002,
        self_contact: bool = True,
        soft_contact: bool = True,
        max_contacts: int = 2048,
        contact_compliance: float = 1.0e-12,
        tolerance: float = 1.0e-7,
        contact_tolerance: float = 1.0e-9,
        max_iterations: int = 200,
        max_geometry_iterations: int = 8,
        mass_damping: float = 0.0,
        anderson_history: int = 1,
        muscle_directions: np.ndarray | None = None,
        muscle_stiffness: float | np.ndarray | None = None,
        integrate_pinned_velocities: bool = False,
        warm_start_contacts: bool = False,
    ):
        super().__init__(model)
        if material not in ("pd", "neo_hookean"):
            raise ValueError("material must be 'pd' or 'neo_hookean'")
        if model.tet_count == 0 or model.particle_count == 0:
            raise ValueError("SolverDiffPhD requires tetrahedral soft bodies")
        if model.body_count or getattr(model, "hex_count", 0):
            raise ValueError("SolverDiffPhD currently accepts tetrahedral particles without rigid bodies or hexahedra")
        if model.shape_count and np.any(model.shape_type.numpy() != int(GeoType.PLANE)):
            raise ValueError("SolverDiffPhD supports its configured ground plane, not native rigid collision shapes")
        for name, value in (
            ("friction", friction),
            ("contact_thickness", contact_thickness),
            ("mass_damping", mass_damping),
            ("contact_compliance", contact_compliance),
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, value in (("tolerance", tolerance), ("contact_tolerance", contact_tolerance)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (
            ("max_iterations", max_iterations),
            ("max_geometry_iterations", max_geometry_iterations),
            ("max_contacts", max_contacts),
        ):
            if not np.isfinite(value) or value < 1 or int(value) != value:
                raise ValueError(f"{name} must be a finite positive integer")
        if (
            not np.isfinite(anderson_history)
            or not 0 <= anderson_history <= 8
            or int(anderson_history) != anderson_history
        ):
            raise ValueError("anderson_history must be an integer in [0,8]")
        if ground_plane is not None:
            plane = np.asarray(ground_plane, dtype=np.float64)
            if plane.shape != (4,) or not np.all(np.isfinite(plane)):
                raise ValueError("ground_plane must contain four finite values")
            normal_length = np.linalg.norm(plane[:3])
            if not np.isfinite(normal_length) or normal_length == 0:
                raise ValueError("ground_plane must have a finite nonzero normal")
        self._initialize_muscles(muscle_directions, muscle_stiffness)
        self._validate_model_physics()
        self.material = material
        self.material_code = MODEL_COROTATED if material == "pd" else MODEL_NEO_HOOKEAN
        self.ground_plane = None if ground_plane is None else (tuple(ground_plane[:3]), ground_plane[3])
        self.friction = float(friction)
        self.contact_thickness = float(contact_thickness)
        self.contact_compliance = float(contact_compliance)
        self.tolerance = float(tolerance)
        self.contact_tolerance = float(contact_tolerance)
        self.max_iterations = int(max_iterations)
        self.max_geometry_iterations = int(max_geometry_iterations)
        self.mass_damping = float(mass_damping)
        self.anderson_history = int(anderson_history)
        self.integrate_pinned_velocities = bool(integrate_pinned_velocities)
        self.warm_start_contacts = bool(warm_start_contacts)
        self._contact_warm_start = None
        self._n = model.particle_count
        self._tets = model.tet_indices.numpy().reshape(-1, 4).astype(np.int32)
        if not np.array_equal(np.unique(self._tets), np.arange(self._n)):
            raise ValueError("Every particle must belong to a valid tetrahedron; standalone particles are unsupported")
        self._worlds = model.particle_world.numpy().copy()
        if np.any(self._worlds < -1) or np.any(self._worlds >= len(model.gravity)):
            raise ValueError("particle_world must reference an existing world or global world -1")
        self.indices = wp.array(self._tets, dtype=wp.vec4i, device=self.device)
        self._rest = model.particle_q.numpy().astype(np.float64)
        self._rest_poses = model.tet_poses.numpy().astype(np.float64)
        self._reference_poses = wp.clone(model.tet_poses, requires_grad=False)
        self._reference_worlds = wp.clone(model.particle_world, requires_grad=False)
        self._gradients = np.concatenate((-self._rest_poses.sum(axis=1)[:, None], self._rest_poses), axis=1)
        self._volumes = np.abs(1.0 / np.linalg.det(self._rest_poses)) / 6.0
        if not np.all(np.isfinite(self._volumes)) or np.any(self._volumes <= 0):
            raise ValueError("tetrahedra must have nondegenerate rest shapes")
        self.gradients = wp.array(self._gradients, dtype=wp.vec3d, device=self.device)
        self.volumes = wp.array(self._volumes, dtype=wp.float64, device=self.device)
        self.geometry = ContactGeometry(
            self._tets,
            self._rest,
            self._worlds,
            self.device,
            thickness=contact_thickness,
            max_contacts=max_contacts,
            self_contact=self_contact,
            soft_contact=soft_contact,
        )
        self._factor_key = None
        self.inverse = None
        self.last_step_info = None
        self._residual_graphs = []
        # Read residuals and the preceding accepted update together. The update
        # is only needed at the next convergence check, not when it is applied.
        self._norm = wp.zeros(3, dtype=wp.float64, device=self.device)
        self._residual_norm = self._norm[:2]
        self._contact_norm = self._norm[1:2]
        self._update_norm = self._norm[2:]
        self._last_update_norm = np.inf
        self._minimum_jacobian = wp.empty(1, dtype=wp.float64, device=self.device)
        self._model_status = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._model_layout = {
            name: (array.shape, array.dtype, array.device)
            for name in (
                "tet_indices",
                "tet_poses",
                "particle_world",
                "tet_materials",
                "particle_mass",
                "particle_inv_mass",
                "particle_flags",
            )
            for array in (getattr(model, name),)
        }
        self._joint_search = None
        self._responses = []
        self._previous = wp.empty(self._n, dtype=wp.vec3d, device=self.device)
        self._velocity = wp.empty_like(self._previous)
        self._force = wp.empty_like(self._previous)
        self._position = wp.empty_like(self._previous)
        self._error_contact_force = wp.empty_like(self._previous)

    def notify_model_changed(self, flags: int) -> None:
        """Invalidate material/mass factors; topology, rest shape or world edits require reconstruction."""
        self._factor_key = None

    @staticmethod
    def _require_zero(array, description):
        if array is not None and np.any(array.numpy() != 0):
            raise ValueError(f"SolverDiffPhD does not support {description}; all values must be zero")

    def _initialize_muscles(self, directions, stiffness):
        self._has_muscles = False
        self._activation_source = None
        if directions is None and stiffness is None:
            return
        if directions is None or stiffness is None:
            raise ValueError("muscle_directions and muscle_stiffness must be supplied together")
        count = self.model.tet_count
        directions = np.array(directions, dtype=np.float64, copy=True)
        stiffness = np.array(stiffness, dtype=np.float64, copy=True)
        if stiffness.ndim == 0:
            stiffness = np.full(count, float(stiffness))
        if directions.shape != (count, 3) or stiffness.shape != (count,):
            raise ValueError("Muscle directions must have shape (tet_count,3), stiffness scalar or (tet_count,)")
        if not np.isfinite(directions).all() or not np.isfinite(stiffness).all() or np.any(stiffness < 0):
            raise ValueError("Muscle directions and stiffness must be finite, with nonnegative stiffness")
        lengths = np.linalg.norm(directions, axis=1)
        active = stiffness > 0
        if np.any(active & ((lengths == 0) | ~np.isfinite(lengths))):
            raise ValueError("Active muscle directions must have finite nonzero length")
        directions[active] /= lengths[active, None]
        directions[~active] = 0.0
        self._has_muscles = bool(np.any(active))
        self._muscle_directions_host = directions
        self._muscle_stiffness_host = stiffness
        self.muscle_directions = wp.array(directions, dtype=wp.vec3d, device=self.device)
        self.muscle_stiffness = wp.array(stiffness, dtype=wp.float64, device=self.device)
        self.muscle_activation = wp.zeros(count, dtype=wp.float64, device=self.device)

    def _validate_model_physics(self):
        model = self.model
        materials = model.tet_materials.numpy().astype(np.float64)
        if np.any(materials[:, :2] < 0) or not np.all(np.isfinite(materials)) or np.any(materials[:, 0] <= 0):
            raise ValueError("DiffPhD requires positive shear modulus and nonnegative Lame lambda")
        if np.any(materials[:, 2] != 0):
            raise ValueError("SolverDiffPhD does not support tetrahedral viscous damping; use mass_damping")
        if not self._has_muscles:
            self._require_zero(model.tet_activations, "tetrahedral activations")
        self._require_zero(model.tri_materials, "triangle elasticity, damping, drag or lift")
        self._require_zero(model.spring_stiffness, "spring stiffness")
        self._require_zero(model.spring_damping, "spring damping")
        self._require_zero(model.edge_bending_properties, "edge bending stiffness or damping")
        if np.any((model.particle_flags.numpy() & int(ParticleFlags.ACTIVE)) == 0):
            raise ValueError("SolverDiffPhD requires ACTIVE particles; pin particles using zero inverse mass")
        masses = model.particle_mass.numpy().astype(np.float64)
        inverse_masses = model.particle_inv_mass.numpy()
        if not np.all(np.isfinite(inverse_masses)) or np.any(inverse_masses < 0):
            raise ValueError("Particle inverse masses must be finite and nonnegative")
        free = (inverse_masses > 0).astype(np.int32)
        if not np.all(np.isfinite(masses)) or np.any(masses < 0) or np.any((masses <= 0) & (free != 0)):
            raise ValueError("Dynamic particles require positive finite masses")
        return materials, masses, free

    def _prepare(self, dt, control=None):
        model = self.model
        if model.particle_count != self._n or model.tet_count != len(self._tets):
            raise ValueError("Reconstruct SolverDiffPhD after topology changes")
        # Check metadata before kernels index the cached topology. Model edits
        # must not turn a device-side value check into an out-of-bounds read.
        for name, layout in self._model_layout.items():
            array = getattr(model, name)
            if array is None or (array.shape, array.dtype, array.device) != layout:
                raise ValueError(f"Reconstruct SolverDiffPhD after {name} layout changes")
        if self._has_muscles:
            source = model.tet_activations
            if control is not None and control.tet_activations is not None:
                source = control.tet_activations
            if (
                source is None
                or source.shape != (len(self._tets),)
                or source.dtype != wp.float32
                or source.device != self.device
            ):
                raise ValueError("Muscle activations must be a float32 (tet_count,) array on the solver device")
            self._activation_source = source
            self._model_status.zero_()
            wp.launch(
                read_activations,
                len(self._tets),
                inputs=[source, self.muscle_stiffness, self.muscle_activation, self._model_status],
                device=self.device,
            )
        if self.inverse is None:
            if not np.array_equal(model.tet_indices.numpy().reshape(-1, 4), self._tets):
                raise ValueError("Reconstruct SolverDiffPhD after topology changes")
            if not np.array_equal(model.tet_poses.numpy().astype(np.float64), self._rest_poses):
                raise ValueError("Reconstruct SolverDiffPhD after rest-shape changes")
            if not np.array_equal(model.particle_world.numpy(), self._worlds):
                raise ValueError("Reconstruct SolverDiffPhD after particle_world changes")
            changed = 8 | (int(self._model_status.numpy()[0]) if self._has_muscles else 0)
        else:
            if not self._has_muscles:
                self._model_status.zero_()
            wp.launch(
                k.check_model,
                max(self._n, len(self._tets)),
                inputs=[
                    model.tet_indices.flatten(),
                    model.tet_poses,
                    model.particle_world,
                    model.tet_materials,
                    model.particle_mass,
                    model.particle_inv_mass,
                    model.particle_flags,
                    self.indices,
                    self._reference_poses,
                    self._reference_worlds,
                    self.mu,
                    self.lame,
                    self.mass,
                    self.free,
                    int(ParticleFlags.ACTIVE),
                    self._model_status,
                ],
                device=self.device,
            )
            arrays = (
                None if self._has_muscles else model.tet_activations,
                model.tri_materials,
                model.spring_stiffness,
                model.spring_damping,
                model.edge_bending_properties,
                control.tet_activations if control is not None and not self._has_muscles else None,
            )
            for array in arrays:
                if array is not None and array.size:
                    wp.launch(
                        k.check_zero, array.size, inputs=[array.flatten(), self._model_status], device=self.device
                    )
            changed = int(self._model_status.numpy()[0])
            for bit, description in ((1, "topology"), (2, "rest-shape"), (4, "particle_world")):
                if changed & bit:
                    raise ValueError(f"Reconstruct SolverDiffPhD after {description} changes")
        if changed & 16:
            raise ValueError("Muscle activations must be finite and <=1, and zero for disabled fibres")
        if changed & 8:
            if control is not None and not self._has_muscles:
                self._require_zero(control.tet_activations, "tetrahedral control activations")
            self._material_data = self._validate_model_physics()
        materials, masses, free = self._material_data
        key = (dt, self.mass_damping, materials[:, :2].tobytes(), masses.tobytes(), free.tobytes())
        if key == self._factor_key:
            return
        self.mu = wp.array(materials[:, 0], dtype=wp.float64, device=self.device)
        self.lame = wp.array(materials[:, 1], dtype=wp.float64, device=self.device)
        self.mass = wp.array(masses, dtype=wp.float64, device=self.device)
        self.free = wp.array(free, dtype=wp.int32, device=self.device)
        rows = [{i: masses[i] * (1.0 / dt**2 + self.mass_damping / dt) if free[i] else 1.0} for i in range(self._n)]
        for element, (tet, gradients, volume, material) in enumerate(
            zip(self._tets, self._gradients, self._volumes, materials, strict=True)
        ):
            block = volume * (2.0 * material[0] + material[1]) * (gradients @ gradients.T)
            if self._has_muscles:
                fibre_gradient = gradients @ self._muscle_directions_host[element]
                block += volume * self._muscle_stiffness_host[element] * np.outer(fibre_gradient, fibre_gradient)
            for a in range(4):
                i = int(tet[a])
                if not free[i]:
                    continue
                for b in range(a, 4):
                    j = int(tet[b])
                    if not free[j]:
                        continue
                    value = float(block[a, b])
                    rows[i][j] = rows[i].get(j, 0.0) + value
                    if i != j:
                        rows[j][i] = rows[j].get(i, 0.0) + value
        self.inverse = SparseInverse(rows, self.device)
        self.diagonal = wp.array([r[i] for i, r in enumerate(rows)], dtype=wp.float64, device=self.device)
        self._factor_key = key
        self._contact_warm_start = None
        # Captured residual evaluations reference these material/mass arrays.
        self._joint_search = None
        self._residual_graphs.clear()

    def _contact_response(self, rows):
        parameters = {
            "thickness": self.contact_thickness,
            "friction": self.friction,
            "compliance": self.contact_compliance,
        }
        for i, response in enumerate(self._responses):
            if response.count == rows.count:
                self._responses.pop(i)
                response.update(rows, self.inverse, self.free, **parameters)
                break
        else:
            response = ContactResponse(rows, self.inverse, self.free, **parameters)
        # Two recent sizes cover alternating contact counts without retaining
        # a dense workspace for every size encountered during a long rollout.
        self._responses.append(response)
        del self._responses[:-2]
        return response

    def _residual(self, q, previous, velocity, force, dt, result=None):
        if result is None:
            result = wp.empty(self._n, dtype=wp.vec3d, device=self.device)
        result.zero_()
        self._minimum_jacobian.fill_(1.0)
        wp.launch(
            k.minimum_jacobian,
            len(self._tets),
            inputs=[q, self.indices, self.gradients, self._minimum_jacobian],
            device=self.device,
        )
        wp.launch(
            k.checked_tet_gradient,
            len(self._tets),
            inputs=[
                q,
                self.indices,
                self.gradients,
                self.volumes,
                self.mu,
                self.lame,
                self.material_code,
                self._minimum_jacobian,
                result,
            ],
            device=self.device,
        )
        if self._has_muscles:
            wp.launch(
                accumulate_muscle_gradient,
                len(self._tets),
                inputs=[
                    q,
                    self.indices,
                    self.gradients,
                    self.volumes,
                    self.muscle_directions,
                    self.muscle_stiffness,
                    self.muscle_activation,
                    self._minimum_jacobian,
                    result,
                ],
                device=self.device,
            )
        wp.launch(
            k.add_inertia,
            self._n,
            inputs=[
                q,
                previous,
                velocity,
                force,
                self.mass,
                self.free,
                self.model.particle_world,
                self.model.gravity,
                dt,
                self.mass_damping,
                self.integrate_pinned_velocities,
                result,
            ],
            device=self.device,
        )
        return result

    class _ResidualEvaluation:
        """Replay residual kernels using stable addresses for arbitrary iterates."""

        def __init__(self, solver, context, parameters):
            self.context, self.parameters = context, parameters
            response, previous, velocity, force, dt = context
            self.q = wp.empty(solver._n, dtype=wp.vec3d, device=solver.device)
            self.residual = wp.empty_like(self.q)
            self.contact_force = wp.empty_like(self.q)
            enabled = gc.isenabled()
            gc.disable()
            try:
                with wp.ScopedCapture(device=solver.device) as capture:
                    solver._evaluate_error(
                        self.q, previous, velocity, force, dt, response, self.residual, self.contact_force
                    )
                self.graph = capture.graph
            finally:
                if enabled:
                    gc.enable()

    def _residual_parameters(self):
        return (
            self.material_code,
            self.mass_damping,
            self.tolerance,
            self.contact_tolerance,
            self.friction,
            self.contact_compliance,
            self.contact_thickness,
            self.model.gravity.ptr,
            self.model.particle_world.ptr,
            self.integrate_pinned_velocities,
        )

    def _error(self, q, previous, velocity, force, dt, response, *, residual=None):
        if residual is None:
            residual = wp.empty_like(q)
        if self.device.is_cuda:
            context = (response, previous, velocity, force, dt)
            parameters = self._residual_parameters()
            for i, evaluation in enumerate(self._residual_graphs):
                if (
                    all(a is b for a, b in zip(evaluation.context[:4], context[:4], strict=True))
                    and evaluation.context[4] == dt
                    and evaluation.parameters == parameters
                ):
                    self._residual_graphs.append(self._residual_graphs.pop(i))
                    break
            else:
                evaluation = self._ResidualEvaluation(self, context, parameters)
                self._residual_graphs.append(evaluation)
                del self._residual_graphs[:-2]
            wp.copy(evaluation.q, q)
            wp.capture_launch(evaluation.graph)
            # Callers retain distinct current/trial residuals, including when
            # a rejected Newton correction falls back to the current PD step.
            wp.copy(residual, evaluation.residual)
        else:
            self._evaluate_error(q, previous, velocity, force, dt, response, residual, self._error_contact_force)
        norms = self._norm.numpy()
        error = max(float(norms[0]), float(norms[1]) * self.tolerance / self.contact_tolerance)
        self._last_update_norm = float(norms[2])
        return error, residual

    def _evaluate_error(self, q, previous, velocity, force, dt, response, residual, contact_force):
        """Enqueue a physical residual and its checks without reading device data."""
        self._residual(q, previous, velocity, force, dt, result=residual)
        response.force(contact_force)
        self._residual_norm.zero_()
        wp.launch(
            k.residual_error,
            self._n,
            inputs=[residual, contact_force, self.free, self.diagonal, self._minimum_jacobian, self._norm],
            device=self.device,
        )
        if response.count:
            wp.launch(
                physical_contact_residual,
                response.count,
                inputs=[
                    q,
                    previous,
                    response.rows.indices,
                    response.rows.weights,
                    response.frames,
                    response.rows.offsets,
                    self.contact_thickness,
                    response.lam,
                    self.contact_compliance,
                    response.scales,
                    self.friction,
                    response.displacement,
                    response.residual,
                    response.dq,
                    response.dl,
                    self._contact_norm,
                ],
                device=self.device,
            )

    def _admissible(self, q):
        self._minimum_jacobian.fill_(1.0)
        wp.launch(
            k.minimum_jacobian,
            len(self._tets),
            inputs=[q, self.indices, self.gradients, self._minimum_jacobian],
            device=self.device,
        )
        return float(self._minimum_jacobian.numpy()[0]) > 1.0e-10

    def _captured_search(
        self,
        q,
        candidate,
        old_lam,
        direction,
        error,
        previous,
        velocity,
        force,
        dt,
        response,
        *,
        start,
        stop,
        newton=False,
    ):
        search = self._joint_search
        context = (response, previous, velocity, force)
        parameters = self._residual_parameters()
        if (
            search is None
            or any(a is not b for a, b in zip(search.context[:4], context, strict=True))
            or search.context[4] != dt
            or search.parameters != parameters
        ):
            search = _JointLineSearch(self, response, previous, velocity, force, dt, parameters)
            self._joint_search = search
        return search.search(q, candidate, old_lam, direction, error, start=start, stop=stop, newton=newton)

    def _accept_candidate(self, q, candidate):
        self._update_norm.zero_()
        wp.launch(k.step_error, self._n, inputs=[candidate, q, self._update_norm], device=self.device)
        wp.copy(q, candidate)

    def _solve(self, q, previous, velocity, force, dt, response):
        residual = wp.empty_like(q)
        trial_residual = wp.empty_like(q)
        inverse_residual = wp.empty_like(q)
        q_free = wp.empty_like(q)
        contact_force = wp.empty_like(q)
        contact_delta = wp.empty_like(q)
        candidate = wp.empty_like(q)
        trial = wp.empty_like(q)
        old_lam = wp.empty_like(response.lam)
        lam_direction = wp.empty_like(response.lam)
        previous_map = None
        previous_residual = None
        dg_history, df_history = [], []
        use_anderson = self.anderson_history and not response.count
        slow_steps = 0
        near_converged_steps = 0
        coupled_only = False
        for iteration in range(self.max_iterations):
            used_newton = False
            error, residual = self._error(q, previous, velocity, force, dt, response, residual=residual)
            if error <= self.tolerance and (iteration == 0 or self._last_update_norm <= self.tolerance):
                return iteration, error
            wp.copy(old_lam, response.lam)
            # Give PD two steps to finish before paying for a physical tangent.
            # Stiff materials can stagnate below the residual tolerance while
            # their updates remain too large; both convergence tests still apply.
            near_converged_steps = near_converged_steps + 1 if error <= self.tolerance else 0
            if near_converged_steps >= 3 and not coupled_only:
                try:
                    self._newton_candidate(q, previous, velocity, force, dt, response, candidate)
                except ConvergenceError:
                    wp.copy(response.lam, old_lam)
                else:
                    self._accept_candidate(q, candidate)
                    df_history.clear()
                    dg_history.clear()
                    previous_map = None
                    previous_residual = None
                    near_converged_steps = 0
                    continue
            if not coupled_only:
                self.inverse.apply(residual, inverse_residual)
                wp.launch(k.combine, self._n, inputs=[q, inverse_residual, 1.0, -1.0, q_free], device=self.device)
                try:
                    used_newton = self._contact_candidate(
                        q, q_free, previous, velocity, force, dt, response, old_lam, candidate, error
                    )
                except ContactConvergenceError:
                    # The fixed-PD contact problem only proposes an iterate.
                    # If it cannot be solved, retain the current physical state
                    # and solve the true coupled equations for this geometry.
                    wp.copy(response.lam, old_lam)
                    coupled_only = True
            if coupled_only:
                self._newton_candidate(q, previous, velocity, force, dt, response, candidate)
                used_newton = True
            if used_newton:
                self._accept_candidate(q, candidate)
                slow_steps = 0
                near_converged_steps = 0
                continue
            response.force(contact_force)
            wp.launch(k.mask_force, self._n, inputs=[contact_force, self.free], device=self.device)
            self.inverse.apply(contact_force, contact_delta)
            wp.launch(k.combine, self._n, inputs=[q_free, contact_delta, 1.0, 1.0, candidate], device=self.device)
            candidate_error, _ = self._error(
                candidate, previous, velocity, force, dt, response, residual=trial_residual
            )

            # Acceleration is safeguarded against the actual force/contact
            # residual, rather than accepted because its step is small.
            if use_anderson:
                map_residual = wp.empty_like(q)
                wp.launch(k.combine, self._n, inputs=[candidate, q, 1.0, -1.0, map_residual], device=self.device)
            if use_anderson and previous_map is not None:
                df = wp.empty_like(q)
                dg = wp.empty_like(q)
                wp.launch(
                    k.combine, self._n, inputs=[map_residual, previous_residual, 1.0, -1.0, df], device=self.device
                )
                wp.launch(k.combine, self._n, inputs=[candidate, previous_map, 1.0, -1.0, dg], device=self.device)
                df_history.append(df)
                dg_history.append(dg)
                df_history = df_history[-self.anderson_history :]
                dg_history = dg_history[-self.anderson_history :]
                m = len(df_history)
                df_array = wp.empty((m, self._n), dtype=wp.vec3d, device=self.device)
                dg_array = wp.empty_like(df_array)
                for j in range(m):
                    wp.copy(df_array[j], df_history[j])
                    wp.copy(dg_array[j], dg_history[j])
                gram = wp.empty((m, m), dtype=wp.float64, device=self.device)
                rhs = wp.empty(m, dtype=wp.float64, device=self.device)
                wp.launch(
                    k.gram_history, (m, m), inputs=[df_array, map_residual, self._n, gram, rhs], device=self.device
                )
                matrix = gram.numpy()
                regularization = max(1.0e-12, 1.0e-6 * np.trace(matrix) / m)
                coefficients = np.linalg.solve(matrix + regularization * np.eye(m), rhs.numpy())
                if np.linalg.norm(coefficients) <= 10:
                    gamma = wp.array(coefficients, dtype=wp.float64, device=self.device)
                    wp.launch(k.extrapolate, self._n, inputs=[candidate, dg_array, gamma, trial], device=self.device)
                    accelerated_error, _ = self._error(
                        trial, previous, velocity, force, dt, response, residual=trial_residual
                    )
                    if accelerated_error < candidate_error:
                        wp.copy(candidate, trial)
                        candidate_error = accelerated_error
                else:
                    df_history.clear()
                    dg_history.clear()
            if use_anderson:
                previous_map = wp.clone(candidate)
                previous_residual = map_residual
            if candidate_error >= error and candidate_error > self.tolerance:
                if response.count:
                    wp.launch(
                        vector_update,
                        3 * response.count,
                        inputs=[response.lam, old_lam, -1.0, lam_direction],
                        device=self.device,
                    )
                accepted = False
                if self.device.is_cuda and wp.is_conditional_graph_supported():
                    trial_error = self._captured_search(
                        q,
                        candidate,
                        old_lam,
                        lam_direction,
                        error,
                        previous,
                        velocity,
                        force,
                        dt,
                        response,
                        start=1,
                        stop=19,
                    )
                    if np.isfinite(trial_error):
                        candidate_error = trial_error
                        accepted = True
                else:
                    for backtrack in range(1, 19):
                        alpha = 0.5**backtrack
                        wp.launch(
                            k.combine, self._n, inputs=[q, candidate, 1.0 - alpha, alpha, trial], device=self.device
                        )
                        if response.count:
                            wp.launch(
                                vector_update,
                                3 * response.count,
                                inputs=[old_lam, lam_direction, alpha, response.lam],
                                device=self.device,
                            )
                        trial_error, _ = self._error(
                            trial, previous, velocity, force, dt, response, residual=trial_residual
                        )
                        if trial_error < error or trial_error <= self.tolerance:
                            wp.copy(candidate, trial)
                            candidate_error = trial_error
                            accepted = True
                            break
                if not accepted:
                    wp.copy(response.lam, old_lam)
                    candidate_error = self._newton_candidate(q, previous, velocity, force, dt, response, candidate)
                    used_newton = True
                    df_history.clear()
                    dg_history.clear()
            # A nearly noncontracting PD map can decrease the residual at every
            # iteration while missing the requested tolerance by many orders
            # of magnitude. Use the physical KKT tangent before that exhausts
            # the iteration budget, keeping the accepted PD step as a fallback.
            slow_steps = slow_steps + 1 if not used_newton and candidate_error > 0.8 * error else 0
            if slow_steps >= 2 and candidate_error > self.tolerance:
                pd_candidate = wp.clone(candidate)
                pd_lam = wp.clone(response.lam)
                wp.copy(response.lam, old_lam)
                newton_error = np.inf
                try:
                    newton_error = self._newton_candidate(q, previous, velocity, force, dt, response, candidate)
                except ConvergenceError:
                    pass
                if newton_error < candidate_error:
                    candidate_error = newton_error
                    df_history.clear()
                    dg_history.clear()
                    previous_map = None
                    previous_residual = None
                else:
                    wp.copy(candidate, pd_candidate)
                    wp.copy(response.lam, pd_lam)
                slow_steps = 0
            self._accept_candidate(q, candidate)
        error, _ = self._error(q, previous, velocity, force, dt, response, residual=residual)
        raise ConvergenceError(f"DiffPhD did not converge in {self.max_iterations} iterations: residual {error:.3e} m")

    def _contact_candidate(self, q, q_free, previous, velocity, force, dt, response, old_lam, candidate, error):
        # The fixed-PD contact solve only proposes a physical iterate. Bound
        # its expensive branch-crossing sweeps before trying the true tangent.
        # Aim for half the current coupled residual in matching units, down to
        # contact_tolerance. Outer checks still enforce the final tolerances.
        tolerance = self.contact_tolerance
        if np.isfinite(error):
            tolerance = max(tolerance, 0.5 * error * self.contact_tolerance / self.tolerance)
        try:
            response.solve(q_free, previous, tolerance=tolerance, max_polish_batches=64)
        except ContactWorkLimit:
            wp.copy(response.lam, old_lam)
            try:
                self._newton_candidate(q, previous, velocity, force, dt, response, candidate)
            except ConvergenceError:
                # A rejected physical correction must not replace the original
                # recovery path or leave its trial multipliers behind.
                wp.copy(response.lam, old_lam)
                response.solve(q_free, previous, tolerance=self.contact_tolerance)
            else:
                return True
        return False

    def _newton_candidate(self, q, previous, velocity, force, dt, response, candidate):
        error, residual = self._error(q, previous, velocity, force, dt, response)
        system = KKTSystem.forward(self, q, response, dt)
        contact_force = wp.empty_like(q)
        response.force(contact_force)
        wp.launch(k.mask_force, self._n, inputs=[contact_force, self.free], device=self.device)
        rhs = wp.empty(system.size, dtype=wp.float64, device=self.device)
        wp.launch(
            k.pack_newton, system.size, inputs=[residual, contact_force, response.residual, rhs], device=self.device
        )
        # This is an inexact Newton direction, not the final state or adjoint.
        # The line search and outer force/contact checks retain the requested
        # physical tolerances; solving every intermediate tangent to 1e-11
        # wastes Krylov iterations without improving the converged trajectory.
        old_lam = wp.clone(response.lam)
        for relative_tolerance in (1.0e-2, 1.0e-6):
            # Inexact directions need not descend the physical max-norm merit.
            # Retry this same tangent/RHS more accurately only after failure.
            delta = system.solve(rhs, transpose=False, relative_tolerance=relative_tolerance)
            if self.device.is_cuda and wp.is_conditional_graph_supported():
                candidate_error = self._captured_search(
                    q,
                    candidate,
                    old_lam,
                    delta,
                    error,
                    previous,
                    velocity,
                    force,
                    dt,
                    response,
                    start=0,
                    stop=20,
                    newton=True,
                )
                if np.isfinite(candidate_error):
                    return candidate_error
            else:
                for backtrack in range(20):
                    alpha = 0.5**backtrack
                    wp.launch(
                        k.newton_update,
                        max(self._n, 3 * response.count),
                        inputs=[q, old_lam, delta, alpha, candidate, response.lam],
                        device=self.device,
                    )
                    candidate_error, _ = self._error(
                        candidate, previous, velocity, force, dt, response, residual=residual
                    )
                    if candidate_error < error or candidate_error <= self.tolerance:
                        return candidate_error
            wp.copy(response.lam, old_lam)
        raise ConvergenceError(f"DiffPhD nonlinear Newton safeguard failed: residual {error:.3e} m")

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
        *,
        observables: SolverBase.Observables | None = None,
    ) -> None:
        """Advance a converged step of length ``dt`` [s]; record its implicit adjoint.

        Raises:
            ConvergenceError: Collision detection or the coupled solve could not
                converge. Input and output particle states have not been written;
                callers may retry with a smaller timestep.
        """
        self.validate_observables(observables, contacts)
        if contacts is not None:
            raise NotImplementedError("SolverDiffPhD generates its own swept contact geometry; pass contacts=None")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if state_in is state_out:
            raise ValueError("DiffPhD requires distinct input and output states")
        if wp.get_device(self.device).is_capturing:
            raise RuntimeError("SolverDiffPhD convergence checks do not support CUDA graph capture")
        tape = wp._src.context.runtime.tape
        if tape is not None:
            # Keep the guard with a recorded callback, so Tape.reset() clears
            # it together with that rollout instead of retaining stale pointers.
            seen = set()
            for launch in reversed(tape.launches):
                recorded_buffers = getattr(launch, "_diff_phd_state_buffers", None)
                if recorded_buffers is not None:
                    seen = recorded_buffers
                    break
            inputs = {state_in.particle_q.ptr, state_in.particle_qd.ptr}
            outputs = {state_out.particle_q.ptr, state_out.particle_qd.ptr}
            if outputs & (seen | inputs):
                raise ValueError(
                    "Recorded DiffPhD steps require distinct output states; ping-pong state reuse aliases gradients"
                )
        with wp.ScopedDevice(self.device), _without_tape():
            self._prepare(float(dt), control)
            previous, velocity, force = self._previous, self._velocity, self._force
            wp.launch(
                k.read_state,
                self._n,
                inputs=[state_in.particle_q, state_in.particle_qd, state_in.particle_f, previous, velocity, force],
                device=self.device,
            )
            if not self._admissible(previous):
                raise ValueError("DiffPhD input contains inverted or collapsed tetrahedra")
            q = self._position
            wp.launch(
                k.predictor,
                self._n,
                inputs=[
                    previous,
                    velocity,
                    force,
                    self.mass,
                    self.free,
                    self.model.particle_world,
                    self.model.gravity,
                    dt,
                    self.mass_damping,
                    self.integrate_pinned_velocities,
                    q,
                ],
                device=self.device,
            )
            self._minimum_jacobian.fill_(1.0)
            wp.launch(
                k.minimum_jacobian,
                len(self._tets),
                inputs=[q, self.indices, self.gradients, self._minimum_jacobian],
                device=self.device,
            )
            wp.launch(
                k.restore_invalid_predictor,
                self._n,
                inputs=[q, previous, self.free, self.integrate_pinned_velocities, self._minimum_jacobian],
                device=self.device,
            )
            total_iterations = 0
            retained_rows = None
            warm_start = self._contact_warm_start if self.warm_start_contacts else None
            for _geometry_iteration in range(self.max_geometry_iterations):
                generated = self.geometry.generate(q, previous, ground=self.ground_plane, dt=dt, retained=retained_rows)
                rows = self.geometry.retain(retained_rows, generated)
                # retain() returns the same owned rows exactly when no new
                # feature was added; their frozen geometry is unchanged.
                if rows is retained_rows:
                    break
                response = self._contact_response(rows)
                if response.count and warm_start is not None:
                    wp.launch(
                        k.warm_start_contacts,
                        response.count,
                        inputs=[
                            *warm_start,
                            response.rows.indices,
                            response.rows.kinds,
                            response.frames,
                            self.friction,
                            response.lam,
                        ],
                        device=self.device,
                    )
                iterations, error = self._solve(q, previous, velocity, force, dt, response)
                total_iterations += iterations
                retained_rows = rows
                if self.warm_start_contacts:
                    # Own the history: response.update() overwrites its buffers.
                    # Publish only after every geometry pass has succeeded.
                    warm_start = (
                        tuple(
                            wp.clone(value, requires_grad=False)
                            for value in (response.rows.indices, response.rows.kinds, response.frames, response.lam)
                        )
                        if response.count
                        else None
                    )
            else:
                raise ConvergenceError("DiffPhD contact geometry did not stabilize; reduce timestep")
            self.last_step_info = StepInfo(total_iterations, error, response.count, _geometry_iteration + 1)
            self._last_q = q
            self._last_response = response
            self._contact_warm_start = warm_start
            wp.launch(
                k.write_state,
                self._n,
                inputs=[
                    q,
                    previous,
                    velocity,
                    self.free,
                    dt,
                    self.integrate_pinned_velocities,
                    state_out.particle_q,
                    state_out.particle_qd,
                ],
                device=self.device,
            )
        if tape is not None:
            launch_count = len(tape.launches)
            record_adjoint(self, tape, state_in, state_out, q, previous, velocity, force, response, float(dt))
            if len(tape.launches) > launch_count:
                tape.launches[-1]._diff_phd_state_buffers = seen | inputs | outputs
