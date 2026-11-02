# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Portions translated from DiffPhD (https://github.com/Akinesia112/DiffPhD,
# commit d319435), Copyright (c) 2021 MIT Graphics Group, MIT License.
# See THIRD_PARTY_NOTICES.md.

"""Reference DiffPhD projective dynamics for Tet4 and regular Hex8 soft bodies."""

import gc
from contextlib import contextmanager

import numpy as np
import warp as wp
from newton import Contacts, Control, GeoType, Model, ParticleFlags, State
from newton.solvers import SolverBase

from . import hex as h
from . import kernels as k
from .adjoint import STATUS_ADJOINT_NOT_CONVERGED, STATUS_ADJOINT_SAFEGUARD, Adjoint, StepRecord
from .anderson import CHUNK, MAX_WINDOW, Anderson
from .contact import FrictionalContact, mesh_vertices, surface_vertices
from .exceptions import ConvergenceError
from .mesh_contact import MeshContact, add_state_force
from .sparse import SparseInverse

MATERIALS = {"pd": 0, "neo_hookean": 1}


@contextmanager
def _without_tape():
    # Warp 1.17 has no public way to pause a Tape; forward kernels have no
    # adjoints, and the step records its own backward with Tape.record_func.
    tape = wp._src.context.runtime.tape
    wp._src.context.runtime.tape = None
    try:
        yield
    finally:
        wp._src.context.runtime.tape = tape


@wp.kernel(enable_backward=False)
def _free_particles(inverse_mass: wp.array[wp.float32], free: wp.array[wp.int32]):
    i = wp.tid()
    free[i] = wp.where(inverse_mass[i] > wp.float32(0.0), 1, 0)


class SolverDiffPhD(SolverBase):
    """Projective dynamics with the executed conventions of the public DiffPhD reference.

    Each step solves backward Euler by the reference local/global fixed point
    ``x <- A^{-1} (m/h^2 q~ + sum_e w_e Vbar G_e^T (P_e(G_e x) - G_e x_D))`` with
    ``q~ = q + h v + h^2/m (f_ext + m g)``, Anderson mixing of the iterates and
    the dual stopping gate ``||dx|| <= rel ||x_prev|| + abs`` and
    ``||dp|| <= rel ||p_prev|| + abs`` on the projection right-hand side ``p``.
    The constant matrix is ``A = m/h^2 I + sum_e w_e Vbar G_e G_e^T`` per
    coordinate with zero-inverse-mass particles eliminated as Dirichlet values.
    ``A^{-1}`` is applied through an exact sparse inverse factor
    ``A^{-1} = S^T S`` that is built once per timestep and material set.

    Reference conventions are kept on purpose: every particle has the same mass
    ``m = density * Vbar``, where ``Vbar`` is the mean rest element volume,
    so the model's particle masses only distinguish dynamic particles from
    zero-inverse-mass Dirichlet particles; every element uses ``Vbar`` as its
    volume; there is no damping. ``material="pd"`` adds a corotated energy of weight ``2 mu_e``
    and a volume-preserving energy of weight ``lambda_e``.
    ``material="neo_hookean"`` uses the reference proximal Neo-Hookean
    projection with element weight ``2 mu_e``; the projection itself uses the
    mean ``mu``, ``lambda`` and penalty ``k = 2 mean(mu)`` of all elements, so
    the fixed point is the reference's Moreau-envelope energy, not the
    hyperelastic residual.

    The iteration runs on the device. On CUDA devices with conditional graph
    support it is a ``wp.capture_while`` loop: inside an enclosing
    ``wp.ScopedCapture`` it becomes part of the caller's graph, otherwise it is
    replayed from a cached graph. CPU devices run the same kernels from a
    host loop. Failures do not interrupt the device: they set bits in
    :attr:`status`, which :meth:`check_status` reads and raises on demand.

    Contact follows the reference frictional boundary: one planar or spherical
    obstacle, every registered vertex a candidate on every step, one
    Fischer-Burmeister Newton step per iteration and an isotropic Coulomb
    stick/slip projection (see ``contact.py``). The obstacle parameters live in
    :attr:`boundary_parameters` and may be changed between steps, also between
    replays of a captured graph.

    Args:
        model: Newton model with position-only tetrahedral soft bodies, or
            particle-only states with explicit ``hex_mesh`` geometry.
        density: Uniform material density [kg/m^3].
        material: ``"pd"`` or ``"neo_hookean"``.
        max_iterations: Maximum local/global iterations per step.
        abs_tol: Absolute tolerance of both gates [m for positions, N for ``p``].
        rel_tol: Relative convergence tolerance (dimensionless).
        anderson_window: Retained Anderson pairs, 0 to disable (reference default 5).
        muscle_directions: Rest-space fibre directions, shape ``(tet_count, 3)``.
        muscle_stiffness: Fibre stiffness [Pa], scalar or ``(tet_count,)``.
        integrate_pinned_velocities: Move zero-inverse-mass particles by
            ``dt * particle_qd`` as prescribed Dirichlet values.
        frictional_boundary: ``("planar", (nx, ny, nz, offset))`` with gap
            ``(n.x + offset) / |n|`` [m], ``("spherical", (cx, cy, cz, radius))``,
            or the reference ``MeshFrictionalBoundary`` between two bodies,
            ``("mesh", (contact_radius, split_index, faces_left, faces_right))``,
            where vertices below ``split_index`` touch ``faces_right`` and the
            others ``faces_left`` (see ``contact.py``); None disables contact.
        frictional_vertices: Registered particle indices; defaults to the
            vertices of boundary faces (mesh: of both face sets).
        friction: Coulomb friction coefficient (dimensionless).
        restitution: Post-solve normal restitution coefficient (dimensionless).
        mesh_contact: The reference ``mesh_contact`` state force between two
            bodies of the model, ``(contact_radius, kn_left, kn_right, kf, mu,
            faces_a, faces_b)`` with radius [m], stiffnesses [N/m], ``kf``
            [N s/m] and ``(F, 3)`` surface triangles (see ``mesh_contact.py``).
            It is evaluated on the start-of-step state, enters the inertial
            target and, as in the reference, has no gradient.
        hex_mesh: Optional public ``Mesh`` containing float64 rest vertices and
            ``(E,8)`` Hex8 connectivity in binary order
            ``000,001,010,011,100,101,110,111``. The reference requires equally
            sized axis-aligned cubic cells. Each cell uses eight trilinear
            Gauss samples of weight ``Vbar/8``; no tetrahedra are introduced.
            The model must then contain particles but no tetrahedra.
        hex_materials: Hex8 float64 ``(E,2)`` array of ``[mu,lambda]`` or ``(E,3)``
            with a zero damping column. NumPy input is copied to the device;
            a Warp array must already be float64 on the model device. The
            public device array can be edited; call :meth:`notify_model_changed`
            before the next step and recapture graphs afterwards.
        muscle_targets: Hex8 float64 ``(E,)`` native target fibre lengths
            (default 1). Zero actively contracts the fibre. Unlike tetrahedra,
            Hex8 does not use ``tet_activations``. Edit the public device array
            in place to update targets, including between graph replays.
        planar_energies: Hex8 reference vertex PD energies
            ``(vertex_indices, stiffness, (nx,ny,nz,offset))``. Stiffness is
            scalar or one value per registered vertex; gap is
            ``(n.x+offset)/|n|``. The constant matrix includes an isotropic
            stiffness diagonal even while the vertex is above the plane.
            This is the original penalty energy, not frictional contact.

    Fibres use ``control.tet_activations`` (or ``model.tet_activations``) as a
    contraction ``u``; the reference target fibre length is ``a = 1 - u``.

    Tet4 steps run under a ``wp.Tape`` record the reference backward pass (see
    ``adjoint.py``) for state, external force, Lame parameter and activation
    gradients. Those are the derivatives of the reference algorithm: state
    gradients ignore contact, and the local Jacobian uses the mean weight per
    energy while ``A`` uses element weights. Reference backward safeguards may
    clip or suppress gradients (status bit 16). Added boundary sensitivities
    propagate to prescribed positions, or to the start positions (plus
    ``dt * qd`` when integrated) when no explicit target is supplied. Contact
    geometry, density and dt are constants. Hex8 currently supports forward
    simulation only: taped steps fail before changing the state, because its
    quadrature adjoint is not implemented. Hex8 frictional and mesh contact
    are explicitly unsupported; use ``planar_energies`` for the native
    cantilever's planar vertex energies.
    """

    ConvergenceError = ConvergenceError

    def __init__(
        self,
        model: Model,
        *,
        density: float,
        material: str = "pd",
        max_iterations: int = 10000,
        abs_tol: float = 1.0e-9,
        rel_tol: float = 1.0e-6,
        anderson_window: int = 5,
        muscle_directions: np.ndarray | None = None,
        muscle_stiffness: float | np.ndarray | None = None,
        integrate_pinned_velocities: bool = False,
        frictional_boundary: tuple[str, tuple] | None = None,
        frictional_vertices: np.ndarray | None = None,
        friction: float = 0.0,
        restitution: float = 0.0,
        mesh_contact: tuple | None = None,
        hex_mesh=None,
        hex_materials: np.ndarray | wp.array | None = None,
        muscle_targets: np.ndarray | wp.array | None = None,
        planar_energies: tuple | None = None,
    ):
        super().__init__(model)
        if material not in MATERIALS:
            raise ValueError("material must be 'pd' or 'neo_hookean'")
        if model.particle_count == 0 or (model.tet_count == 0 and hex_mesh is None):
            raise ValueError("SolverDiffPhD requires tetrahedral soft bodies or an explicit hex_mesh")
        if model.body_count or getattr(model, "hex_count", 0):
            raise ValueError("SolverDiffPhD requires particles without rigid bodies; supply Hex8 using hex_mesh")
        if (hex_mesh is None) != (hex_materials is None):
            raise ValueError("hex_mesh and hex_materials must be supplied together")
        if hex_mesh is not None and model.tet_count:
            raise ValueError("hex_mesh cannot be combined with model tetrahedra")
        if hex_mesh is None and (muscle_targets is not None or planar_energies is not None):
            raise ValueError("Explicit muscle_targets and planar_energies currently require hex_mesh")
        if hex_mesh is not None and any(x is not None for x in (frictional_boundary, mesh_contact)):
            raise NotImplementedError("Hex8 currently supports planar_energies, not frictional or mesh contact")
        if model.shape_count and np.any(model.shape_type.numpy() != int(GeoType.PLANE)):
            raise ValueError("SolverDiffPhD does not collide with native Newton shapes")
        if not np.isfinite(density) or density <= 0:
            raise ValueError("density must be finite and positive")
        for name, value in (("abs_tol", abs_tol), ("rel_tol", rel_tol)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(max_iterations) or max_iterations < 1 or int(max_iterations) != max_iterations:
            raise ValueError("max_iterations must be a positive integer")
        if int(anderson_window) != anderson_window or not 0 <= anderson_window <= MAX_WINDOW:
            raise ValueError(f"anderson_window must be an integer in [0, {MAX_WINDOW}]")
        self.material = material
        self.max_iterations = int(max_iterations)
        self.abs_tol = float(abs_tol)
        self.rel_tol = float(rel_tol)
        self.integrate_pinned_velocities = bool(integrate_pinned_velocities)
        self._n = model.particle_count
        self._hex = None if hex_mesh is None else h.HexElements(hex_mesh, self._n, self.device)
        if self._hex is None:
            self._tets = model.tet_indices.numpy().reshape(-1, 4).astype(np.int32)
            if not np.array_equal(np.unique(self._tets), np.arange(self._n)):
                raise ValueError("Every particle must belong to a tetrahedron")
            self._elements = self._tets
            rest_poses = model.tet_poses.numpy().astype(np.float64)
            self._gradients = np.concatenate((-rest_poses.sum(axis=1)[:, None], rest_poses), axis=1)
            self._volumes = np.abs(1.0 / np.linalg.det(rest_poses)) / 6.0
            if not np.all(np.isfinite(self._volumes)) or np.any(self._volumes <= 0):
                raise ValueError("tetrahedra must have nondegenerate rest shapes")
            self.mean_volume = float(self._volumes.mean())
        else:
            self._elements = self._hex.elements
            self._volumes = self._hex.volumes
            self.mean_volume = self._hex.mean_volume
            if isinstance(hex_materials, wp.array):
                if hex_materials.device != self.device or hex_materials.dtype != wp.float64:
                    raise ValueError("hex_materials Warp array must be float64 on the model device")
                self.hex_materials = hex_materials
            else:
                self.hex_materials = wp.array(np.asarray(hex_materials, dtype=np.float64), device=self.device)
            if self.hex_materials.shape not in ((len(self._elements), 2), (len(self._elements), 3)):
                raise ValueError("hex_materials must have shape (element_count,2 or 3): mu,lambda[,damping]")
        self._element_count = len(self._elements)
        self._sample_count = 1 if self._hex is None else 8
        self._sample_volume = self.mean_volume / self._sample_count
        worlds = model.particle_world.numpy()
        if np.any(worlds < -1) or np.any(worlds >= len(model.gravity)):
            raise ValueError("particle_world must reference an existing world or global world -1")
        self.density = float(density)
        self.particle_mass = self.density * self.mean_volume
        self._initialize_muscles(muscle_directions, muscle_stiffness)
        if self._hex is not None:
            if muscle_targets is not None and not self._has_muscles:
                raise ValueError("muscle_targets requires enabled muscle_directions and muscle_stiffness")
            targets = np.ones(self._element_count) if muscle_targets is None else muscle_targets
            if isinstance(targets, wp.array):
                if targets.dtype != wp.float64 or targets.device != self.device:
                    raise ValueError("muscle_targets Warp array must be float64 on the model device")
                self.muscle_targets = targets
            else:
                self.muscle_targets = wp.array(np.asarray(targets, dtype=np.float64), device=self.device)
            if self.muscle_targets.shape != (self._element_count,):
                raise ValueError("muscle_targets must have shape (element_count,)")
            values = self.muscle_targets.numpy()
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("muscle_targets must be finite nonnegative native target lengths")
            self._muscle_target = self.muscle_targets

        device = self.device
        count = self._element_count
        self.indices = wp.array(self._tets, dtype=wp.vec4i, device=device) if self._hex is None else self._hex.indices
        self.gradients = (
            wp.array(self._gradients, dtype=wp.vec3d, device=device) if self._hex is None else self._hex.gradients
        )
        # Element contributions are gathered per vertex in a fixed order, so
        # results do not depend on atomic scheduling.
        self.incidence = k.build_incidence(self.indices, self._n, device) if self._hex is None else self._hex.incidence
        self._contributions = wp.zeros(
            (count * self._sample_count, 4 if self._hex is None else 8), dtype=wp.vec3d, device=device
        )
        self._free = wp.empty(self._n, dtype=wp.int32, device=device)
        self._corotated_weight = wp.zeros(count, dtype=wp.float64, device=device)
        self._volume_weight = wp.zeros_like(self._corotated_weight)
        self._neo_hookean_weight = wp.zeros_like(self._corotated_weight)
        vectors = (
            "position",
            "velocity",
            "target",
            "boundary",
            "x",
            "previous",
            "projection",
            "previous_projection",
            "rhs",
            "solved",
        )
        for name in vectors:
            setattr(self, f"_{name}", wp.zeros(self._n, dtype=wp.vec3d, device=device))
        self._chunks = max(1, (self._n + CHUNK - 1) // CHUNK)
        self._partials = wp.zeros(self._chunks, dtype=wp.vec4d, device=device)
        self._iteration = wp.zeros(1, dtype=wp.int32, device=device)
        self._active = wp.zeros(1, dtype=wp.int32, device=device)
        self._status = wp.zeros(1, dtype=wp.int32, device=device)
        self._planar_energies = None if planar_energies is None else h.PlanarEnergies(planar_energies, self._n, device)
        self._anderson = Anderson(self._n, int(anderson_window), device)
        # Anderson works on (columns, size) arrays; the step mixes one column.
        self._x_column = self._x.reshape((1, self._n))
        self._previous_column = self._previous.reshape((1, self._n))
        self._contact = None
        if frictional_boundary is not None:
            kind, parameters = frictional_boundary
            if frictional_vertices is None:
                if kind == "mesh":
                    frictional_vertices = mesh_vertices(parameters)
                else:
                    frictional_vertices = surface_vertices(self._tets)
            self._contact = FrictionalContact(
                kind,
                parameters,
                frictional_vertices,
                self._n,
                friction,
                restitution,
                device,
            )
        elif frictional_vertices is not None or friction or restitution:
            raise ValueError("frictional_vertices, friction and restitution need a frictional_boundary")
        self._mesh_contact = None
        if mesh_contact is not None:
            if len(mesh_contact) != 7:
                raise ValueError("mesh_contact is (contact_radius, kn_left, kn_right, kf, mu, faces_a, faces_b)")
            self._mesh_contact = MeshContact(*mesh_contact, self._n, device)
        self.inverse = None
        self._factor_key = None
        self._model_changed = True
        self._loop_graph = None
        self._loop_key = None
        self._activation_source = None
        self._adjoint = None
        self._recorded = None

    @property
    def status(self) -> wp.array:
        """Device status bits accumulated since the last :meth:`check_status`."""
        return self._status

    @property
    def boundary_parameters(self) -> wp.array:
        """Device ``vec4d`` array with the frictional boundary parameters, or None."""
        return None if self._contact is None else self._contact.parameters

    @property
    def contact_multipliers(self) -> wp.array:
        """Device normal multipliers of the registered vertices after the last step, or None."""
        return None if self._contact is None else self._contact.multiplier

    @property
    def mesh_contact_pairs(self) -> wp.array | None:
        """Device count of ``mesh_contact`` vertex-face pairs within the contact radius in the last step, or None."""
        return None if self._mesh_contact is None else self._mesh_contact.pairs

    @property
    def iterations(self) -> wp.array:
        """Device array holding the local/global iteration count of the last step."""
        return self._iteration

    def deformation_determinants(self, state: State) -> wp.array:
        """Evaluate ``det(F)`` at all eight Hex8 Gauss samples as ``(cells,8)``.

        This optional diagnostic allocates an output array; call it outside
        captured simulation steps. It does not alter the state or the solve.
        """
        if self._hex is None:
            raise NotImplementedError("deformation_determinants currently requires hex_mesh")
        if state.particle_q.device != self.device or state.particle_q.shape != (self._n,):
            raise ValueError("state must have this solver's particle count and device")
        result = wp.empty((self._element_count, 8), dtype=wp.float64, device=self.device)
        wp.launch(
            h.deformation_determinants,
            result.shape,
            inputs=[state.particle_q, self.indices, self.gradients],
            outputs=[result],
            device=self.device,
        )
        return result

    def read_status(self) -> int:
        """Read and clear :attr:`status` without raising; one host synchronization.

        Bits: 1 a step reached ``max_iterations`` (the reference warns and keeps
        the last iterate), 2 singular volume projection, 4 invalid activation,
        8 a recorded backward fixed point did not converge, 16 a reference backward
        safeguard scrubbed, clipped or suppressed a contribution.
        """
        bits = int(self._status.numpy()[0])
        self._status.zero_()
        return bits

    def check_status(self) -> None:
        """Read and clear :attr:`status`, raising for any failure since the last check.

        This is the only host synchronization of the solver; call it rarely,
        for example once per frame or rollout.

        Raises:
            ValueError: An element reached the reference's singular volume
                projection, or a fibre activation was invalid.
            ConvergenceError: A step reached ``max_iterations`` without meeting
                both convergence gates, or a recorded backward pass did not
                converge.
        """
        bits = self.read_status()
        if bits & int(k.STATUS_SINGULAR):
            raise ValueError("DiffPhD volume projection met a singular deformation gradient (reference 'Singular F.')")
        if bits & int(k.STATUS_ACTIVATION):
            if self._hex is not None:
                raise ValueError("Hex8 muscle target lengths must be finite and nonnegative")
            raise ValueError("Muscle activations must be finite and <= 1, and zero for disabled fibres")
        if bits & int(k.STATUS_NOT_CONVERGED):
            raise ConvergenceError(f"DiffPhD did not converge within {self.max_iterations} iterations")
        if bits & int(STATUS_ADJOINT_NOT_CONVERGED):
            raise ConvergenceError("A DiffPhD backward fixed point did not converge")
        if bits & int(STATUS_ADJOINT_SAFEGUARD):
            raise ConvergenceError("A reference DiffPhD backward safeguard modified the gradient")

    def notify_model_changed(self, flags: int) -> None:
        """Rebuild the material weights and inverse factor before the next step.

        The rebuild replaces device buffers. Recapture any graph that contains
        :meth:`step`, as after a change of ``dt``.
        """
        if self.device.is_capturing:
            raise RuntimeError("notify_model_changed cannot run during graph capture")
        self._model_changed = True

    def _initialize_muscles(self, directions, stiffness):
        count = self._element_count
        self._has_muscles = False
        # Zero weights disable the fibre term in the local step.
        self._muscle_direction = wp.zeros(count, dtype=wp.vec3d, device=self.device)
        self._muscle_weight = wp.zeros(count, dtype=wp.float64, device=self.device)
        self._muscle_target = wp.zeros(count, dtype=wp.float64, device=self.device)
        if directions is None and stiffness is None:
            return
        if directions is None or stiffness is None:
            raise ValueError("muscle_directions and muscle_stiffness must be supplied together")
        directions = np.array(directions, dtype=np.float64, copy=True)
        stiffness = np.array(stiffness, dtype=np.float64, copy=True)
        if stiffness.ndim == 0:
            stiffness = np.full(count, float(stiffness))
        if directions.shape != (count, 3) or stiffness.shape != (count,):
            raise ValueError(
                "Muscle directions must have shape (element_count,3), stiffness scalar or (element_count,)"
            )
        if not np.isfinite(directions).all() or not np.isfinite(stiffness).all() or np.any(stiffness < 0):
            raise ValueError("Muscle directions and stiffness must be finite, with nonnegative stiffness")
        lengths = np.linalg.norm(directions, axis=1)
        active = stiffness > 0
        if np.any(active & (lengths <= np.finfo(np.float64).eps)):
            raise ValueError("Active muscle directions must have nonzero length")
        directions[active] /= lengths[active, None]
        directions[~active] = 0.0
        self._has_muscles = bool(np.any(active))
        self._muscle_directions_host = directions
        self._muscle_weight_host = stiffness * self._sample_volume
        self._muscle_direction = wp.array(directions, dtype=wp.vec3d, device=self.device)
        self._muscle_weight = wp.array(self._muscle_weight_host, dtype=wp.float64, device=self.device)
        self.muscle_directions = self._muscle_direction
        self.muscle_stiffness = wp.array(stiffness, dtype=wp.float64, device=self.device)

    def _validate_model(self):
        model = self.model
        materials = (model.tet_materials if self._hex is None else self.hex_materials).numpy().astype(np.float64)
        if not np.all(np.isfinite(materials)) or np.any(materials[:, 0] <= 0) or np.any(materials[:, 1] < 0):
            raise ValueError("DiffPhD requires positive shear modulus and nonnegative Lame lambda")
        if materials.shape[1] == 3 and np.any(materials[:, 2] != 0):
            raise ValueError("The reference DiffPhD solver has no damping; element damping must be zero")
        for name, description in (
            ("tri_materials", "triangle elasticity"),
            ("spring_stiffness", "springs"),
            ("edge_bending_properties", "edge bending"),
        ):
            array = getattr(model, name)
            if array is not None and array.size and np.any(array.numpy() != 0):
                raise ValueError(f"SolverDiffPhD does not support {description}")
        if not self._has_muscles and model.tet_activations is not None and np.any(model.tet_activations.numpy()):
            raise ValueError("Tetrahedral activations require muscle_directions and muscle_stiffness")
        if np.any((model.particle_flags.numpy() & int(ParticleFlags.ACTIVE)) == 0):
            raise ValueError("SolverDiffPhD requires ACTIVE particles; pin particles with zero inverse mass")
        inverse_mass = model.particle_inv_mass.numpy()
        if not np.all(np.isfinite(inverse_mass)) or np.any(inverse_mass < 0):
            raise ValueError("Particle inverse masses must be finite and nonnegative")
        free = inverse_mass > 0
        if not np.any(free):
            raise ValueError("SolverDiffPhD needs at least one dynamic particle")
        return materials, free

    def _build(self, dt: float):
        if self.device.is_capturing:
            raise RuntimeError("SolverDiffPhD must rebuild its factor; step once outside graph capture first")
        model = self.model
        materials, free = self._validate_model()
        # Fresh arrays keep earlier recorded steps valid after a rebuild.
        self._free = wp.empty(self._n, dtype=wp.int32, device=self.device)
        self._corotated_weight = wp.empty(self._element_count, dtype=wp.float64, device=self.device)
        self._volume_weight = wp.empty_like(self._corotated_weight)
        self._neo_hookean_weight = wp.empty_like(self._corotated_weight)
        wp.launch(_free_particles, self._n, inputs=[model.particle_inv_mass], outputs=[self._free], device=self.device)
        wp.launch(
            k.element_weights if self._hex is None else h.element_weights,
            self._element_count,
            inputs=[
                model.tet_materials if self._hex is None else self.hex_materials,
                MATERIALS[self.material],
                self._sample_volume,
            ],
            outputs=[self._corotated_weight, self._volume_weight, self._neo_hookean_weight],
            device=self.device,
        )
        weights = self._corotated_weight.numpy() + self._volume_weight.numpy() + self._neo_hookean_weight.numpy()
        # NeoHookeanPdElementEnergy::InitializeNeoHookeanPerElement keeps mean
        # parameters for the projection and k = stiffness() = mean(2 mu_e).
        self._nh_mu = float(materials[:, 0].mean())
        self._nh_lame = float(materials[:, 1].mean())
        self._nh_penalty = 2.0 * self._nh_mu
        self._material_code = MATERIALS[self.material]
        self._has_dirichlet = not bool(np.all(free))
        inertia = self.particle_mass / dt**2
        rows = [{i: inertia if free[i] else 1.0} for i in range(self._n)]
        for element, nodes in enumerate(self._elements):
            samples = self._gradients[element][None] if self._hex is None else self._hex.sample_gradients
            for gradients in samples:
                block = weights[element] * (gradients @ gradients.T)
                if self._has_muscles and self._muscle_weight_host[element] > 0:
                    fibre = gradients @ self._muscle_directions_host[element]
                    block += self._muscle_weight_host[element] * np.outer(fibre, fibre)
                for a in range(len(nodes)):
                    i = int(nodes[a])
                    if not free[i]:
                        continue
                    for b in range(a, len(nodes)):
                        j = int(nodes[b])
                        if not free[j]:
                            continue
                        value = float(block[a, b])
                        rows[i][j] = rows[i].get(j, 0.0) + value
                        if i != j:
                            rows[j][i] = rows[j].get(i, 0.0) + value
        if self._planar_energies is not None:
            for i, stiffness in zip(self._planar_energies.vertices, self._planar_energies.stiffness, strict=True):
                if free[i]:
                    rows[i][i] += float(stiffness)
        self.inverse = SparseInverse(rows, self.device)
        if self._contact is not None:
            self._contact.build(self.inverse)
        self._inertia = inertia
        self._factor_key = dt
        self._model_changed = False
        self._loop_graph = None

    def _read_activations(self, control):
        if not self._has_muscles:
            return
        if self._hex is not None:
            if control is not None and control.tet_activations is not None and control.tet_activations.size:
                raise ValueError("Hex8 uses solver.muscle_targets; tet_activations cannot control Hex8 fibres")
            wp.launch(
                h.validate_targets,
                self._element_count,
                inputs=[self.muscle_targets],
                outputs=[self._status],
                device=self.device,
            )
            return
        source = self.model.tet_activations
        if control is not None and control.tet_activations is not None:
            source = control.tet_activations
        if source is None or source.shape != (len(self._tets),) or source.dtype != wp.float32:
            raise ValueError("Muscle activations must be a float32 (tet_count,) array")
        self._activation_source = source
        wp.launch(
            k.read_activations,
            len(self._tets),
            inputs=[source, self._muscle_weight],
            outputs=[self._muscle_target, self._status],
            device=self.device,
        )

    def _iterate(self):
        n, device = self._n, self.device
        wp.launch(
            k.begin_iteration,
            n,
            inputs=[self._x, self._projection],
            outputs=[self._previous, self._previous_projection],
            device=device,
        )
        wp.launch(
            k.local_step if self._hex is None else h.local_step,
            self._element_count * self._sample_count,
            inputs=[
                self._x,
                self._boundary,
                self.indices,
                self.gradients,
                self._corotated_weight,
                self._volume_weight,
                self._neo_hookean_weight,
                self._nh_mu,
                self._nh_lame,
                self._nh_penalty,
                self._muscle_direction,
                self._muscle_weight,
                self._muscle_target,
            ],
            outputs=[self._contributions, self._status],
            device=device,
        )
        wp.launch(
            k.gather if self._hex is None else h.gather,
            n,
            inputs=[*self.incidence, self._contributions, 0],
            outputs=[self._projection],
            device=device,
        )
        if self._planar_energies is not None:
            self._planar_energies.assemble(self._x, self._boundary, self._projection, device)
        wp.launch(
            k.global_rhs,
            n,
            inputs=[self._target, self._projection, self._inertia],
            outputs=[self._rhs],
            device=device,
        )
        if self._contact is not None and self._contact.count:
            self.inverse.apply(self._rhs, self._solved)
            self._contact.iterate(self._x, self._solved, self._position, self._x)
        else:
            self.inverse.apply(self._rhs, self._x)
        wp.launch(k.impose_boundary, n, inputs=[self._boundary, self._free], outputs=[self._x], device=device)
        self._anderson.mix(self._x_column, self._previous_column)
        convergence = (self._boundary, self._free, self._previous, self._projection, self._previous_projection, n)
        if device.is_cuda:
            wp.launch_tiled(
                k.convergence_tiled,
                dim=[1],
                inputs=[*convergence, self.abs_tol, self.rel_tol, self.max_iterations],
                outputs=[self._x, self._iteration, self._active, self._status],
                block_dim=k.CONVERGENCE_BLOCK,
                device=device,
            )
        else:
            wp.launch(
                k.convergence_partials,
                self._chunks,
                inputs=list(convergence),
                outputs=[self._x, self._partials],
                device=device,
            )
            wp.launch(
                k.check_convergence,
                1,
                inputs=[self._partials, self._chunks, self.abs_tol, self.rel_tol, self.max_iterations],
                outputs=[self._iteration, self._active, self._status],
                device=device,
            )

    def _solve(self):
        device = self.device
        if device.is_capturing:
            # A conditional while node on CUDA, a recorded while operation on CPU.
            wp.capture_while(self._active, self._iterate)
            return
        if device.is_cuda and wp.is_conditional_graph_supported():
            # The graph bakes the convergence scalars; recapture when they change.
            key = (self.max_iterations, self.abs_tol, self.rel_tol)
            if self._loop_graph is None or self._loop_key != key:
                # Keep unrelated CUDA destructors out of the captured graph.
                enabled = gc.isenabled()
                gc.disable()
                try:
                    with wp.ScopedCapture(device=device) as capture:
                        wp.capture_while(self._active, self._iterate)
                    self._loop_graph = capture.graph
                    self._loop_key = key
                finally:
                    if enabled:
                        gc.enable()
            wp.capture_launch(self._loop_graph)
            return
        for _ in range(self.max_iterations):
            self._iterate()
            if not self._active.numpy()[0]:
                break

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
        *,
        observables: SolverBase.Observables | None = None,
        dirichlet_positions: wp.array | None = None,
    ) -> None:
        """Advance one step of length ``dt`` [s] without host synchronization on CUDA.

        ``dirichlet_positions`` optionally supplies independent prescribed endpoint
        positions as a full-particle ``wp.vec3`` array on the solver device. Only
        pinned entries are used. Incoming positions and velocities remain the
        physical state used by state forces. On a Tape, boundary sensitivities
        accumulate in this array's gradient; retain distinct targets for each
        recorded step, as for state arrays.
        """
        self.validate_observables(observables, contacts)
        if contacts is not None:
            raise NotImplementedError("SolverDiffPhD does not consume Newton contacts; pass contacts=None")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if state_in is state_out or state_in.particle_q.ptr == state_out.particle_q.ptr:
            raise ValueError("DiffPhD requires distinct input and output states")
        if dirichlet_positions is not None:
            if not isinstance(dirichlet_positions, wp.array):
                raise TypeError("dirichlet_positions must be a Warp array")
            if dirichlet_positions.dtype != wp.vec3 or dirichlet_positions.shape != (self._n,):
                raise ValueError("dirichlet_positions must have dtype wp.vec3 and shape (particle_count,)")
            if dirichlet_positions.device != self.device:
                raise ValueError("dirichlet_positions must be on the solver device")
            if dirichlet_positions.ptr in (state_out.particle_q.ptr, state_out.particle_qd.ptr):
                raise ValueError("dirichlet_positions must not alias an output state")
        tape = wp._src.context.runtime.tape
        if tape is not None and self._hex is not None:
            raise NotImplementedError(
                "Hex8 supports forward simulation only; its quadrature adjoint is not implemented"
            )
        if tape is not None:
            if self.device.is_capturing:
                raise RuntimeError("Recording SolverDiffPhD steps on a Tape during graph capture is unsupported")
            self._check_aliasing(tape, state_in, state_out)
        dt = float(dt)
        with wp.ScopedDevice(self.device), _without_tape():
            if self._model_changed or self._factor_key != dt:
                self._build(dt)
            self._read_activations(control)
            wp.launch(
                k.begin_step,
                self._n,
                inputs=[
                    state_in.particle_q,
                    state_in.particle_qd,
                    state_in.particle_f,
                    self._free,
                    self.model.particle_world,
                    self.model.gravity,
                    dt,
                    self.particle_mass,
                    self.integrate_pinned_velocities,
                ],
                outputs=[self._position, self._velocity, self._target, self._boundary, self._x, self._projection],
            )
            if dirichlet_positions is not None:
                wp.launch(
                    k.explicit_boundary,
                    self._n,
                    inputs=[dirichlet_positions, self._free],
                    outputs=[self._boundary],
                )
            if self._mesh_contact is not None:
                # q~ = q + h v + h^2/m (f_ext + ForwardStateForce(q, v)).
                self._mesh_contact.evaluate(self._position, self._velocity)
                wp.launch(
                    add_state_force,
                    self._n,
                    inputs=[self._mesh_contact.force, dt, self.particle_mass],
                    outputs=[self._target],
                )
            wp.launch(k.begin_solve, 1, outputs=[self._iteration, self._active])
            self._anderson.reset()
            if self._contact is not None:
                self._contact.begin_step(self._position, dt)
            self._solve()
            if self._contact is not None:
                self._contact.finish_step(self._position, self._velocity, dt, self._x)
            wp.launch(
                k.write_state,
                self._n,
                inputs=[self._x, self._position, dt],
                outputs=[state_out.particle_q, state_out.particle_qd],
            )
            record = StepRecord(self, dt, dirichlet_positions) if tape is not None else None
        if record is not None:
            self._record(tape, record, state_in, state_out)

    def _check_aliasing(self, tape, state_in, state_out):
        # A recorded output overwritten by a later step would corrupt the
        # gradients of the earlier step; reject ping-pong reuse on one tape.
        seen = getattr(tape, "_diff_phd_state_buffers", None)
        if seen is None or self._recorded is not tape or not tape.launches:
            seen = set()
        outputs = {state_out.particle_q.ptr, state_out.particle_qd.ptr}
        inputs = {state_in.particle_q.ptr, state_in.particle_qd.ptr}
        if outputs & (seen | inputs):
            raise ValueError("Recorded DiffPhD steps require distinct output states; ping-pong reuse aliases gradients")
        tape._diff_phd_state_buffers = seen | inputs | outputs
        self._recorded = tape

    def _record(self, tape, record, state_in, state_out):
        if self._adjoint is None:
            self._adjoint = Adjoint(self)
        adjoint = self._adjoint
        if adjoint._rows:
            # A backward pass replayed inside a graph capture cannot allocate.
            record.inverse.reserve_vec3_batch(adjoint._batch_work.shape)

        def backward():
            with wp.ScopedDevice(self.device), _without_tape():
                adjoint.backward(record, state_in, state_out, None)

        arrays = [
            state_in.particle_q,
            state_in.particle_qd,
            state_in.particle_f,
            state_out.particle_q,
            state_out.particle_qd,
        ]
        materials = self.model.tet_materials
        if materials.requires_grad:
            arrays.append(materials)
        if record.activations is not None and record.activations.requires_grad:
            arrays.append(record.activations)
        if record.dirichlet_positions is not None:
            arrays.append(record.dirichlet_positions)
        tape.record_func(backward, [array for array in arrays if array.grad is not None and array.size])
