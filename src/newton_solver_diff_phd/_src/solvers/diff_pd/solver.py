# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Portions translated from DiffPD (https://github.com/mit-gfx/diff_pd_public,
# commit 1594aa0), Copyright (c) 2021 MIT Graphics Group, MIT License.
# See THIRD_PARTY_NOTICES.md.

"""The DiffPD ``pd_eigen`` forward step for Newton tetrahedral soft bodies."""

import gc

import numpy as np
import warp as wp
from newton import Contacts, Control, GeoType, Model, ParticleFlags, State
from newton.solvers import SolverBase

from . import kernels as k
from .exceptions import ConvergenceError
from .sparse import SparseInverse

# Parameter counts of ``Deformable::AddStateForce`` for vertex_dim = 3.
STATE_FORCES = {"planar_contact": 8, "arc_contact": 15}


def _pack_state_force(kind: str, parameters) -> np.ndarray:
    if kind not in STATE_FORCES:
        raise ValueError("state force kinds are 'planar_contact' and 'arc_contact'")
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (STATE_FORCES[kind],) or not np.isfinite(values).all():
        raise ValueError(f"{kind} takes {STATE_FORCES[kind]} finite parameters")
    row = np.zeros(k.STATE_FORCE_WIDTH, dtype=np.float64)
    if kind == "planar_contact":
        # PlanarContactStateForce::Initialize: [normal(3), offset, p, kn, kf, mu].
        normal, offset = values[:3], values[3]
        norm = np.linalg.norm(normal)
        if not norm > 1.0e-5:
            raise ValueError("planar_contact: Singular normal.")
        row[:3] = normal / norm
        row[3] = offset / norm
        row[4] = int(values[4])
        row[5:8] = values[5:8]
    else:
        # ArcContactStateForce::Initialize: [center(3), dir(3), start(3), radius, angle, p, kn, kf, mu].
        axis, start = values[3:6], values[6:9]
        if np.linalg.norm(axis) == 0.0 or np.linalg.norm(start) == 0.0:
            raise ValueError("arc_contact direction and start vectors must be nonzero")
        axis = axis / np.linalg.norm(axis)
        start = start / np.linalg.norm(start)
        row[:3] = values[:3]
        row[3:6] = axis
        row[6:9] = start
        # aux = dir x start, the local y axis of the arc.
        row[9:12] = np.cross(axis, start)
        row[12:14] = values[9:11]
        row[14] = int(values[11])
        row[15:18] = values[12:15]
    return row


def _positive_integer(name: str, value, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class SolverDiffPD(SolverBase):
    """The DiffPD ``pd_eigen`` forward step (``use_bfgs = 1``) of the public diff_pd reference.

    Each step evaluates the explicit state forces at the start of the step,
    ``rhs = q + h v + h^2/m f_ext + h^2/m f_state(q, v)``, and minimizes
    ``Phi(x) = 1/2 m/h^2 |x - rhs|^2 + E_pd(x)`` from ``x = q`` by L-BFGS with
    the constant projective-dynamics matrix
    ``A = m/h^2 I + (w_c + w_v) Vbar sum_e G_e G_e^T`` as initial Hessian and a
    backtracking Armijo line search, until
    ``|g| <= rel_tol |m/h^2 rhs| + abs_tol`` with
    ``g = m/h^2 (x - rhs) - f_pd(x)``. The new velocity is ``(x - q) / h``.
    ``E_pd`` sums a corotated energy ``w_c/2 |F - R|^2`` and a volume energy
    ``w_v/2 |F - P_v(F)|^2`` over the elements, each weighted by ``Vbar``.

    Reference conventions are kept on purpose: every particle has the mass
    ``m = density * Vbar``, where ``Vbar`` is the mean rest tetrahedron volume,
    and every element uses ``Vbar`` as its volume; the energy weights are global
    scalars given here, not ``model.tet_materials``; gravity and the penalty
    contacts are explicit state forces of the start-of-step state; there is no
    damping. Particles with zero inverse mass are Dirichlet particles held at
    their start-of-step positions.

    The iteration, its line search and the L-BFGS history run on the device.
    On CUDA devices with conditional graph support the loops are
    ``wp.capture_while`` nodes, replayed from a cached graph or captured into
    the caller's graph. Failures set bits in :attr:`status`; the reference's
    exceptions are raised by :meth:`check_status`.

    Args:
        model: Newton model with position-only tetrahedral soft bodies.
        density: Uniform density [kg/m^3].
        corotated_stiffness: Weight ``w_c`` [Pa] of the corotated PD energy,
            or None to omit it (reference ``AddPdEnergy("corotated", [w_c])``).
        volume_stiffness: Weight ``w_v`` [Pa] of the volume PD energy, or None
            to omit it (reference ``AddPdEnergy("volume", [w_v])``).
        state_forces: Contacts after gravity, in order, as
            ``("planar_contact", [nx, ny, nz, offset, p, kn, kf, mu])`` or
            ``("arc_contact", [cx, cy, cz, dx, dy, dz, sx, sy, sz, radius,
            angle, p, kn, kf, mu])`` with the reference parameter lists
            (lengths [m], angle [rad], ``kn`` [N/m^(p-1)], ``kf`` [N s/m]).
        gravity: Float64 gravity [m/s^2] of the reference ``gravity`` state
            force; None uses ``model.gravity`` of each particle's world.
        rest_positions: Float64 rest positions ``(particle_count, 3)`` [m] for
            the element gradients and volumes; None uses ``model.particle_q``.
        max_pd_iter: Maximum L-BFGS iterations per step.
        max_ls_iter: Maximum step halvings per line search.
        abs_tol: Absolute gradient tolerance [N].
        rel_tol: Gradient tolerance relative to ``|m/h^2 rhs|``.
        bfgs_history_size: Retained L-BFGS pairs (> 1).
    """

    ConvergenceError = ConvergenceError

    def __init__(
        self,
        model: Model,
        *,
        density: float,
        corotated_stiffness: float | None = None,
        volume_stiffness: float | None = None,
        state_forces=(),
        gravity=None,
        rest_positions: np.ndarray | None = None,
        max_pd_iter: int = 4000,
        max_ls_iter: int = 10,
        abs_tol: float = 1.0e-9,
        rel_tol: float = 1.0e-4,
        bfgs_history_size: int = 10,
    ):
        super().__init__(model)
        if model.tet_count == 0 or model.particle_count == 0:
            raise ValueError("SolverDiffPD requires tetrahedral soft bodies")
        if model.body_count or getattr(model, "hex_count", 0):
            raise ValueError("SolverDiffPD accepts tetrahedral particles without rigid bodies or hexahedra")
        if model.shape_count and np.any(model.shape_type.numpy() != int(GeoType.PLANE)):
            raise ValueError("SolverDiffPD does not collide with Newton shapes; use state_forces")
        if not np.isfinite(density) or density <= 0:
            raise ValueError("density must be finite and positive")
        for name, value in (("corotated_stiffness", corotated_stiffness), ("volume_stiffness", volume_stiffness)):
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be None or finite and nonnegative")
        for name, value in (("abs_tol", abs_tol), ("rel_tol", rel_tol)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self.max_pd_iter = _positive_integer("max_pd_iter", max_pd_iter, 1)
        self.max_ls_iter = _positive_integer("max_ls_iter", max_ls_iter, 1)
        # ForwardProjectiveDynamics requires bfgs_history_size > 1.
        self.bfgs_history_size = _positive_integer("bfgs_history_size", bfgs_history_size, 2)
        self.abs_tol = float(abs_tol)
        self.rel_tol = float(rel_tol)
        self.corotated_stiffness = None if corotated_stiffness is None else float(corotated_stiffness)
        self.volume_stiffness = None if volume_stiffness is None else float(volume_stiffness)
        rows = []
        for entry in state_forces:
            if len(entry) != 2:
                raise ValueError("state_forces entries are (kind, parameters)")
            rows.append(_pack_state_force(*entry))
        self.gravity = None
        if gravity is not None:
            self.gravity = np.asarray(gravity, dtype=np.float64)
            if self.gravity.shape != (3,) or not np.isfinite(self.gravity).all():
                raise ValueError("gravity must be three finite values")

        self._n = model.particle_count
        self._tets = model.tet_indices.numpy().reshape(-1, 4).astype(np.int32)
        if not np.array_equal(np.unique(self._tets), np.arange(self._n)):
            raise ValueError("Every particle must belong to a tetrahedron")
        worlds = model.particle_world.numpy()
        if np.any(worlds < -1) or np.any(worlds >= len(model.gravity)):
            raise ValueError("particle_world must reference an existing world or global world -1")
        if rest_positions is None:
            rest = model.particle_q.numpy().astype(np.float64)
        else:
            rest = np.array(rest_positions, dtype=np.float64, copy=True)
            if rest.shape != (self._n, 3) or not np.isfinite(rest).all():
                raise ValueError("rest_positions must be finite with shape (particle_count, 3)")
        # TetDeformable::InitializeFiniteElementSamples: B = [X1-X0, X2-X0, X3-X0]^-1,
        # gradient rows g0 = -(B0 + B1 + B2), g1..g3 = B0..B2.
        corners = rest[self._tets]
        edges = np.transpose(corners[:, 1:] - corners[:, :1], (0, 2, 1))
        # Mesh::ComputeElementVolume |a . (b x c) / 6| with a, b, c = X1-X0, X2-X0, X3-X0
        # in the reference's operation order, and the sequentially summed mean volume.
        a, b, c = edges[:, :, 0], edges[:, :, 1], edges[:, :, 2]
        cross = np.stack(
            (
                b[:, 1] * c[:, 2] - b[:, 2] * c[:, 1],
                b[:, 2] * c[:, 0] - b[:, 0] * c[:, 2],
                b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0],
            ),
            axis=1,
        )
        volumes = np.abs((a[:, 0] * cross[:, 0] + a[:, 1] * cross[:, 1] + a[:, 2] * cross[:, 2]) / 6.0)
        if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0):
            raise ValueError("tetrahedra must have nondegenerate rest shapes")
        inverse = np.linalg.inv(edges)
        self._gradients = np.concatenate((-inverse.sum(axis=1)[:, None], inverse), axis=1)
        total = 0.0
        for volume in volumes.tolist():
            total += volume
        self.mean_volume = total / len(volumes)
        self.density = float(density)
        self.particle_mass = self.density * self.mean_volume

        device = self.device
        n, count = self._n, len(self._tets)
        self.indices = wp.array(self._tets, dtype=wp.vec4i, device=device)
        self.gradients = wp.array(self._gradients, dtype=wp.vec3d, device=device)
        self.incidence = k.build_incidence(self.indices, n, device)
        self._force_count = len(rows)
        packed = np.zeros((max(1, len(rows)), k.STATE_FORCE_WIDTH), dtype=np.float64)
        kinds = np.zeros(max(1, len(rows)), dtype=np.int32)
        for index, (entry, row) in enumerate(zip(state_forces, rows, strict=True)):
            packed[index] = row
            kinds[index] = int(k.PLANAR_CONTACT) if entry[0] == "planar_contact" else int(k.ARC_CONTACT)
        self._kinds = wp.array(kinds, dtype=wp.int32, device=device)
        self._state_parameters = wp.array(packed, dtype=wp.float64, device=device)
        for name in ("position", "velocity", "rhs", "x", "trial", "force", "gradient", "direction", "r"):
            setattr(self, f"_{name}", wp.zeros(n, dtype=wp.vec3d, device=device))
        self._x_last = wp.zeros(n, dtype=wp.vec3d, device=device)
        self._gradient_last = wp.zeros(n, dtype=wp.vec3d, device=device)
        self._next_velocity = wp.zeros(n, dtype=wp.vec3d, device=device)
        m = self.bfgs_history_size
        self._s = wp.zeros((m, n), dtype=wp.vec3d, device=device)
        self._y = wp.zeros((m, n), dtype=wp.vec3d, device=device)
        self._rho = wp.zeros(m, dtype=wp.float64, device=device)
        self._alpha = wp.zeros(m, dtype=wp.float64, device=device)
        self._coefficient = wp.zeros(m, dtype=wp.float64, device=device)
        self._energy = wp.zeros(count, dtype=wp.float64, device=device)
        self._contributions = wp.zeros((count, 4), dtype=wp.vec3d, device=device)
        self._vertex_chunks = (n + k.CHUNK - 1) // k.CHUNK
        self._objective_chunks = max(self._vertex_chunks, (count + k.CHUNK - 1) // k.CHUNK)
        self._partials = wp.zeros(self._vertex_chunks, dtype=wp.float64, device=device)
        self._pair_partials = wp.zeros(self._objective_chunks, dtype=wp.vec2d, device=device)
        self._scalars = wp.zeros(k.SCALAR_COUNT, dtype=wp.float64, device=device)
        self._counters = wp.zeros(k.COUNTER_COUNT, dtype=wp.int32, device=device)
        self._active = wp.zeros(1, dtype=wp.int32, device=device)
        self._searching = wp.zeros(1, dtype=wp.int32, device=device)
        self._status = wp.zeros(1, dtype=wp.int32, device=device)
        self._free = wp.zeros(n, dtype=wp.float64, device=device)
        self.inverse = None
        self._factor_key = None
        self._model_changed = True
        self._loop_graph = None
        self._loop_key = None

    @property
    def status(self) -> wp.array:
        """Device status bits accumulated since the last :meth:`check_status`."""
        return self._status

    @property
    def iterations(self) -> wp.array:
        """Device ``int32`` array whose first entry is the L-BFGS iteration count of the last step."""
        return self._counters

    @property
    def positions(self) -> wp.array:
        """Float64 device positions [m] after the last step, before rounding into the float32 state."""
        return self._x

    @property
    def velocities(self) -> wp.array:
        """Float64 device velocities [m/s] after the last step, before rounding into the float32 state."""
        return self._next_velocity

    def read_status(self) -> int:
        """Read and clear :attr:`status` without raising; one host synchronization.

        Bits: 1 a step reached ``max_pd_iter`` (the reference throws "PD method
        fails to converge."), 2 a singular deformation gradient met the volume
        projection (the reference throws "Singular F."), 4 a line search used
        all ``max_ls_iter`` halvings (the reference accepts the last trial
        silently; informational).
        """
        bits = int(self._status.numpy()[0])
        self._status.zero_()
        return bits

    def check_status(self) -> None:
        """Read and clear :attr:`status`, raising the reference's failures since the last check.

        Raises:
            ValueError: A volume projection met a singular deformation gradient.
            ConvergenceError: A step reached ``max_pd_iter`` iterations.
        """
        bits = self.read_status()
        if bits & int(k.STATUS_SINGULAR):
            raise ValueError("DiffPD volume projection met a singular deformation gradient (reference 'Singular F.')")
        if bits & int(k.STATUS_NOT_CONVERGED):
            raise ConvergenceError(f"PD method fails to converge within {self.max_pd_iter} iterations")

    def notify_model_changed(self, flags: int) -> None:
        """Rebuild the Dirichlet mask and the factor of ``A`` before the next step."""
        if self.device.is_capturing:
            raise RuntimeError("notify_model_changed cannot run during graph capture")
        self._model_changed = True

    def _validate_model(self):
        model = self.model
        for name, description in (
            ("tri_materials", "triangle elasticity"),
            ("spring_stiffness", "springs"),
            ("edge_bending_properties", "edge bending"),
        ):
            array = getattr(model, name)
            if array is not None and array.size and np.any(array.numpy() != 0):
                raise ValueError(f"SolverDiffPD does not support {description}")
        if np.any((model.particle_flags.numpy() & int(ParticleFlags.ACTIVE)) == 0):
            raise ValueError("SolverDiffPD requires ACTIVE particles; pin particles with zero inverse mass")
        inverse_mass = model.particle_inv_mass.numpy()
        if not np.all(np.isfinite(inverse_mass)) or np.any(inverse_mass < 0):
            raise ValueError("Particle inverse masses must be finite and nonnegative")
        return inverse_mass > 0

    def _build(self, dt: float):
        if self.device.is_capturing:
            raise RuntimeError("SolverDiffPD must factor A; step once outside graph capture first")
        free = self._validate_model()
        self._free = wp.array(free.astype(np.float64), dtype=wp.float64, device=self.device)
        # SetupProjectiveDynamicsSolver: Dirichlet rows are 1, other diagonals
        # m/h^2, plus w Vbar G_e G_e^T with w = sum of the energy stiffnesses
        # and every entry touching a Dirichlet dof skipped.
        inertia = self.particle_mass / (dt * dt)
        weight = 0.0
        for stiffness in (self.corotated_stiffness, self.volume_stiffness):
            if stiffness is not None:
                weight += stiffness
        rows = [{i: inertia if free[i] else 1.0} for i in range(self._n)]
        for tet, gradients in zip(self._tets, self._gradients, strict=True):
            block = gradients @ gradients.T * weight * self.mean_volume
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
        self._inertia = inertia
        self._h2m = dt * dt / self.particle_mass
        self._factor_key = dt
        self._model_changed = False
        self._loop_graph = None

    def _evaluate(self):
        device = self.device
        wp.launch(
            k.evaluate_elements,
            len(self._tets),
            inputs=[
                self._trial,
                self.indices,
                self.gradients,
                self.mean_volume,
                int(self.corotated_stiffness is not None),
                self.corotated_stiffness or 0.0,
                int(self.volume_stiffness is not None),
                self.volume_stiffness or 0.0,
            ],
            outputs=[self._energy, self._contributions, self._counters],
            device=device,
        )
        wp.launch(
            k.gather, self._n, inputs=[*self.incidence, self._contributions], outputs=[self._force], device=device
        )
        wp.launch(
            k.objective_partials,
            self._objective_chunks,
            inputs=[self._trial, self._rhs, self._inertia, self._n, self._energy, len(self._tets)],
            outputs=[self._pair_partials],
            device=device,
        )

    def _accept(self):
        wp.launch(
            k.accept,
            self._vertex_chunks,
            inputs=[self._trial, self._force, self._rhs, self._free, self._inertia, self._n],
            outputs=[self._x, self._gradient, self._partials],
            device=self.device,
        )

    def _line_search(self):
        device = self.device
        wp.launch(k.trial_position, self._n, inputs=[self._x, self._direction, self._scalars], outputs=[self._trial])
        self._evaluate()
        wp.launch(
            k.line_search_check,
            1,
            inputs=[self._pair_partials, self._objective_chunks, self.max_ls_iter],
            outputs=[self._scalars, self._counters, self._searching],
            device=device,
        )

    def _iterate(self):
        n, m, device = self._n, self.bfgs_history_size, self.device
        chunks, counters = self._vertex_chunks, self._counters
        # Push (s, y) = (x - x_last, g - g_last) and drop the oldest beyond m pairs.
        wp.launch(k.history_advance, 1, inputs=[m], outputs=[counters], device=device)
        wp.launch(
            k.history_store,
            chunks,
            inputs=[self._x, self._gradient, counters, n],
            outputs=[self._x_last, self._gradient_last, self._s, self._y, self._r, self._partials],
            device=device,
        )
        wp.launch(k.history_rho, 1, inputs=[self._partials, chunks, counters], outputs=[self._rho], device=device)
        # Two-loop recursion, newest to oldest, H0 = A^{-1}, oldest to newest.
        for age in range(m):
            wp.launch(
                k.slot_dot_partials,
                chunks,
                inputs=[self._s, self._r, age, m, counters, n],
                outputs=[self._partials],
                device=device,
            )
            wp.launch(
                k.set_alpha,
                1,
                inputs=[self._partials, chunks, age, m, counters, self._rho],
                outputs=[self._alpha],
                device=device,
            )
            wp.launch(
                k.slot_axpy, n, inputs=[self._y, self._alpha, -1.0, age, m, counters], outputs=[self._r], device=device
            )
        self.inverse.apply(self._r, self._direction)
        for age in reversed(range(m)):
            wp.launch(
                k.slot_dot_partials,
                chunks,
                inputs=[self._y, self._direction, age, m, counters, n],
                outputs=[self._partials],
                device=device,
            )
            wp.launch(
                k.set_coefficient,
                1,
                inputs=[self._partials, chunks, age, m, counters, self._rho, self._alpha],
                outputs=[self._coefficient],
                device=device,
            )
            wp.launch(
                k.slot_axpy,
                n,
                inputs=[self._s, self._coefficient, 1.0, age, m, counters],
                outputs=[self._direction],
                device=device,
            )
        wp.launch(
            k.direction_partials,
            chunks,
            inputs=[self._direction, self._gradient, n],
            outputs=[self._pair_partials],
            device=device,
        )
        wp.launch(
            k.choose_direction,
            1,
            inputs=[self._pair_partials, chunks],
            outputs=[self._scalars, counters, self._searching],
            device=device,
        )
        wp.launch(
            k.fallback_direction,
            n,
            inputs=[self._gradient, self._free, counters],
            outputs=[self._direction],
            device=device,
        )
        wp.capture_while(self._searching, self._line_search)
        self._accept()
        wp.launch(
            k.check_convergence,
            1,
            inputs=[self._partials, chunks, self.max_pd_iter],
            outputs=[self._scalars, counters, self._active, self._status],
            device=device,
        )

    def _loop(self):
        device = self.device
        wp.launch(
            k.threshold_partials,
            self._vertex_chunks,
            inputs=[self._rhs, self._free, self._inertia, self._n],
            outputs=[self._partials],
            device=device,
        )
        wp.launch(
            k.begin_solve,
            1,
            inputs=[self._partials, self._vertex_chunks, self.rel_tol, self.abs_tol, self.bfgs_history_size],
            outputs=[self._scalars, self._counters, self._active],
            device=device,
        )
        self._evaluate()
        wp.launch(
            k.initial_objective,
            1,
            inputs=[self._pair_partials, self._objective_chunks],
            outputs=[self._scalars, self._counters, self._active, self._status],
            device=device,
        )
        self._accept()
        wp.capture_while(self._active, self._iterate)

    def _solve(self):
        device = self.device
        if device.is_capturing or not (device.is_cuda and wp.is_conditional_graph_supported()):
            # Captured into the caller's graph, or host-driven loops on CPU.
            self._loop()
            return
        # The graph bakes the options and the factor; recapture when they change.
        key = (self.max_pd_iter, self.max_ls_iter, self.abs_tol, self.rel_tol)
        if self._loop_graph is None or self._loop_key != key:
            # Keep unrelated CUDA destructors out of the captured graph.
            enabled = gc.isenabled()
            gc.disable()
            try:
                with wp.ScopedCapture(device=device) as capture:
                    self._loop()
                self._loop_graph = capture.graph
                self._loop_key = key
            finally:
                if enabled:
                    gc.enable()
        wp.capture_launch(self._loop_graph)

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
        """Advance one step of length ``dt`` [s] without host synchronization on CUDA."""
        self.validate_observables(observables, contacts)
        if contacts is not None:
            raise NotImplementedError("SolverDiffPD does not consume Newton contacts; pass contacts=None")
        if wp._src.context.runtime.tape is not None:
            raise NotImplementedError("SolverDiffPD has no backward pass yet; step outside a wp.Tape")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if state_in is state_out or state_in.particle_q.ptr == state_out.particle_q.ptr:
            raise ValueError("DiffPD requires distinct input and output states")
        dt = float(dt)
        with wp.ScopedDevice(self.device):
            # SetupProjectiveDynamicsSolver runs once per Deformable; dt changes refactor here.
            if self._model_changed or self._factor_key != dt:
                self._build(dt)
            gravity = self.gravity if self.gravity is not None else np.zeros(3)
            wp.launch(
                k.begin_step,
                self._n,
                inputs=[
                    state_in.particle_q,
                    state_in.particle_qd,
                    state_in.particle_f,
                    self.model.particle_world,
                    self.model.gravity,
                    wp.vec3d(*gravity),
                    int(self.gravity is None),
                    self.particle_mass,
                    self._kinds,
                    self._state_parameters,
                    self._force_count,
                    dt,
                    self._h2m,
                ],
                outputs=[self._position, self._velocity, self._rhs, self._trial],
            )
            self._solve()
            wp.launch(
                k.write_state,
                self._n,
                inputs=[self._x, self._position, dt],
                outputs=[self._next_velocity, state_out.particle_q, state_out.particle_qd],
            )
