# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Converged implicit adjoints of the frozen-contact DiffPhD equations.

The transposed coupled mechanical/contact Jacobian is applied without assembling
a dense matrix. Elasticity uses the same unfiltered tangent as the forward
residual. Contact features, normals, barycentric weights, and the selected
semismooth branch are fixed. This derivative covers input positions, velocities,
forces, the two elastic material coefficients, and fibre contractions. Geometry,
mass, timestep, gravity, damping, friction, contact compliance, fibre directions,
and fibre stiffnesses are fixed parameters.
"""

import gc

import numpy as np
import warp as wp
from warp.optim.linear import LinearOperator, gmres

from .elasticity import apply_cached_tet_tangent, compute_tet_tangent, material_vjp
from .exceptions import ConvergenceError
from .linear import DenseFactorizationError, DenseLU
from .muscle import activation_vjp, add_muscle_tangent
from .response import apply_j, apply_jt, contact_jacobian


@wp.kernel(enable_backward=False)
def _seed(
    position_grad: wp.array[wp.vec3],
    velocity_grad: wp.array[wp.vec3],
    free: wp.array[wp.int32],
    dt: wp.float64,
    rhs: wp.array[wp.float64],
):
    i = wp.tid()
    value = wp.vec3d(position_grad[i])
    if free[i] != 0:
        value += wp.vec3d(velocity_grad[i]) / dt
    for axis in range(3):
        rhs[3 * i + axis] = value[axis]


@wp.kernel(enable_backward=False)
def _make_row_scale(
    diagonal: wp.array[wp.float64],
    contact_scales: wp.array[wp.float64],
    contact_ratio: wp.float64,
    out: wp.array[wp.float64],
):
    i = wp.tid()
    mechanical_size = 3 * diagonal.shape[0]
    if i < mechanical_size:
        out[i] = wp.float64(1.0) / diagonal[i // 3]
    else:
        out[i] = contact_ratio / contact_scales[(i - mechanical_size) // 3]


@wp.kernel(enable_backward=False)
def _row_scaled_update(
    x: wp.array[wp.float64],
    scale: wp.array[wp.float64],
    y: wp.array[wp.float64],
    alpha: wp.float64,
    beta: wp.float64,
    divide: bool,
    out: wp.array[wp.float64],
):
    i = wp.tid()
    value = x[i] * scale[i]
    if divide:
        value = x[i] / scale[i]
    value = alpha * value
    if beta != wp.float64(0.0):
        value += beta * y[i]
    out[i] = value


@wp.kernel(enable_backward=False)
def _unpack(
    packed: wp.array[wp.float64],
    free: wp.array[wp.int32],
    all_nodes: wp.array[wp.vec3d],
    free_nodes: wp.array[wp.vec3d],
):
    i = wp.tid()
    value = wp.vec3d(packed[3 * i], packed[3 * i + 1], packed[3 * i + 2])
    all_nodes[i] = value
    if free[i] == 0:
        value = wp.vec3d(wp.float64(0.0))
    free_nodes[i] = value


@wp.kernel(enable_backward=False)
def _inertia_transpose(
    all_nodes: wp.array[wp.vec3d],
    mass: wp.array[wp.float64],
    free: wp.array[wp.int32],
    dt: wp.float64,
    damping: wp.float64,
    out: wp.array[wp.vec3d],
):
    i = wp.tid()
    coefficient = wp.float64(1.0)
    if free[i] != 0:
        coefficient = mass[i] * (wp.float64(1.0) / (dt * dt) + damping / dt)
    # Elastic contributions into pinned columns remain present in H.T.
    out[i] += coefficient * all_nodes[i]


@wp.kernel(enable_backward=False)
def _inertia_forward(
    all_nodes: wp.array[wp.vec3d],
    mass: wp.array[wp.float64],
    free: wp.array[wp.int32],
    dt: wp.float64,
    damping: wp.float64,
    out: wp.array[wp.vec3d],
):
    i = wp.tid()
    if free[i] == 0:
        out[i] = all_nodes[i]
    else:
        out[i] += mass[i] * (wp.float64(1.0) / (dt * dt) + damping / dt) * all_nodes[i]


@wp.kernel(enable_backward=False)
def _contact_forward(
    packed: wp.array[wp.float64],
    offset: int,
    dq: wp.array[wp.mat33d],
    dl: wp.array[wp.mat33d],
    compliance: wp.float64,
    displacement: wp.array[wp.float64],
    force: wp.array[wp.float64],
    residual: wp.array[wp.float64],
):
    c = wp.tid()
    start = offset + 3 * c
    multiplier = wp.vec3d(packed[start], packed[start + 1], packed[start + 2])
    position = wp.vec3d(displacement[3 * c], displacement[3 * c + 1], displacement[3 * c + 2])
    value = dq[c] * (position + compliance * multiplier) + dl[c] * multiplier
    for axis in range(3):
        force[3 * c + axis] = -multiplier[axis]
        residual[3 * c + axis] = value[axis]


@wp.kernel(enable_backward=False)
def _contact_transpose(
    packed: wp.array[wp.float64],
    offset: int,
    dq: wp.array[wp.mat33d],
    dl: wp.array[wp.mat33d],
    compliance: wp.float64,
    displacement_adjoint: wp.array[wp.float64],
    multiplier_adjoint: wp.array[wp.float64],
):
    c = wp.tid()
    start = offset + 3 * c
    value = wp.vec3d(packed[start], packed[start + 1], packed[start + 2])
    position = wp.transpose(dq[c]) * value
    multiplier = wp.transpose(dl[c]) * value + compliance * position
    for axis in range(3):
        displacement_adjoint[3 * c + axis] = position[axis]
        multiplier_adjoint[3 * c + axis] = multiplier[axis]


@wp.kernel(enable_backward=False)
def _pack_mechanical(
    value: wp.array[wp.vec3d],
    old: wp.array[wp.float64],
    alpha: wp.float64,
    beta: wp.float64,
    out: wp.array[wp.float64],
):
    i = wp.tid()
    for axis in range(3):
        index = 3 * i + axis
        previous = wp.float64(0.0)
        if beta != wp.float64(0.0):
            previous = beta * old[index]
        out[index] = alpha * value[i][axis] + previous


@wp.kernel(enable_backward=False)
def _pack_contact(
    multiplier: wp.array[wp.float64],
    mechanical: wp.array[wp.float64],
    old: wp.array[wp.float64],
    offset: int,
    alpha: wp.float64,
    beta: wp.float64,
    out: wp.array[wp.float64],
):
    i = wp.tid()
    previous = wp.float64(0.0)
    if beta != wp.float64(0.0):
        previous = beta * old[offset + i]
    out[offset + i] = alpha * (multiplier[i] - mechanical[i]) + previous


@wp.kernel(enable_backward=False)
def _precondition_contact(
    x: wp.array[wp.float64],
    old: wp.array[wp.float64],
    offset: int,
    alpha: wp.float64,
    beta: wp.float64,
    out: wp.array[wp.float64],
):
    i = offset + wp.tid()
    previous = wp.float64(0.0)
    if beta != wp.float64(0.0):
        previous = beta * old[i]
    out[i] = alpha * x[i] + previous


@wp.kernel(enable_backward=False)
def _mask_nodes(value: wp.array[wp.vec3d], free: wp.array[wp.int32], out: wp.array[wp.vec3d]):
    i = wp.tid()
    result = value[i]
    if free[i] == 0:
        result = wp.vec3d(wp.float64(0.0))
    out[i] = result


@wp.kernel(enable_backward=False)
def _schur_rhs(
    packed: wp.array[wp.float64],
    offset: int,
    displacement: wp.array[wp.float64],
    dq: wp.array[wp.mat33d],
    transpose: int,
    out: wp.array[wp.float64],
):
    c = wp.tid()
    start = 3 * c
    value = wp.vec3d(packed[offset + start], packed[offset + start + 1], packed[offset + start + 2])
    mechanical = wp.vec3d(displacement[start], displacement[start + 1], displacement[start + 2])
    if transpose != 0:
        value += mechanical
    else:
        value -= dq[c] * mechanical
    for axis in range(3):
        out[start + axis] = value[axis]


@wp.kernel(enable_backward=False)
def _schur_reaction(
    multiplier: wp.array[wp.float64],
    dq: wp.array[wp.mat33d],
    transpose: int,
    out: wp.array[wp.float64],
):
    c = wp.tid()
    start = 3 * c
    value = wp.vec3d(multiplier[start], multiplier[start + 1], multiplier[start + 2])
    if transpose != 0:
        value = -(wp.transpose(dq[c]) * value)
    for axis in range(3):
        out[start + axis] = value[axis]


@wp.kernel(enable_backward=False)
def _pack_preconditioned(
    base: wp.array[wp.vec3d],
    correction: wp.array[wp.vec3d],
    old: wp.array[wp.float64],
    alpha: wp.float64,
    beta: wp.float64,
    out: wp.array[wp.float64],
):
    i = wp.tid()
    value = base[i] + correction[i]
    for axis in range(3):
        index = 3 * i + axis
        previous = wp.float64(0.0)
        if beta != wp.float64(0.0):
            previous = beta * old[index]
        out[index] = alpha * value[axis] + previous


@wp.kernel(enable_backward=False)
def _verify_residual(product: wp.array[wp.float64], rhs: wp.array[wp.float64], metrics: wp.array[wp.float64]):
    i = wp.tid()
    error = wp.abs(product[i] - rhs[i])
    magnitude = wp.abs(rhs[i])
    if not wp.isfinite(error) or not wp.isfinite(magnitude):
        error = wp.float64(wp.inf)
    wp.atomic_max(metrics, 0, error)
    wp.atomic_max(metrics, 1, magnitude)


@wp.kernel(enable_backward=False)
def _tangential_only(displacement_adjoint: wp.array[wp.float64]):
    displacement_adjoint[3 * wp.tid()] = wp.float64(0.0)


@wp.kernel(enable_backward=False)
def _position_vjp(
    adjoint: wp.array[wp.vec3d],
    contact_adjoint: wp.array[wp.vec3d],
    output_velocity_grad: wp.array[wp.vec3],
    mass: wp.array[wp.float64],
    free: wp.array[wp.int32],
    dt: wp.float64,
    damping: wp.float64,
    out: wp.array[wp.vec3],
):
    i = wp.tid()
    value = adjoint[i]
    if free[i] != 0:
        value *= mass[i] * (wp.float64(1.0) / (dt * dt) + damping / dt)
        value -= wp.vec3d(output_velocity_grad[i]) / dt
    value += contact_adjoint[i]
    out[i] += wp.vec3(value)


@wp.kernel(enable_backward=False)
def _velocity_vjp(
    adjoint: wp.array[wp.vec3d],
    output_velocity_grad: wp.array[wp.vec3],
    mass: wp.array[wp.float64],
    free: wp.array[wp.int32],
    dt: wp.float64,
    integrate_pinned_velocities: bool,
    out: wp.array[wp.vec3],
):
    i = wp.tid()
    if free[i] != 0:
        out[i] += wp.vec3((mass[i] / dt) * adjoint[i])
    elif integrate_pinned_velocities:
        out[i] += wp.vec3(dt * adjoint[i]) + output_velocity_grad[i]


@wp.kernel(enable_backward=False)
def _force_vjp(adjoint: wp.array[wp.vec3d], free: wp.array[wp.int32], out: wp.array[wp.vec3]):
    i = wp.tid()
    if free[i] != 0:
        out[i] += wp.vec3(adjoint[i])


@wp.kernel(enable_backward=False)
def _material_vjp(mu: wp.array[wp.float64], lame: wp.array[wp.float64], out: wp.array2d[wp.float32]):
    i = wp.tid()
    out[i, 0] -= wp.float32(mu[i])
    out[i, 1] -= wp.float32(lame[i])


class KKTSystem:
    """Coupled tangent with residual-checked forward and transpose solves.

    ``snapshot=True`` retains independent mechanics/contact arrays for backward.
    ``forward()`` reuses a separate owned workspace for forward Newton
    corrections. The operator is the unfiltered physical tangent in both modes;
    solves verify the requested true residual and use the existing PD inverse
    only as a preconditioner.
    """

    def __init__(self, solver, q, response, dt, *, snapshot=True):
        self.device = solver.device
        self.inverse = solver.inverse
        self.n, self.c = solver._n, response.count
        self.size = 3 * (self.n + self.c)
        self.dt, self.damping = float(dt), float(solver.mass_damping)
        self.integrate_pinned_velocities = bool(getattr(solver, "integrate_pinned_velocities", False))
        self.material_code, self.compliance = int(solver.material_code), float(response.compliance)
        self._schur_factor = None
        self._schur_storage = None
        self._schur_matrix = None
        self._schur_attempted = False
        self._tangent = None
        self._tangent_dirty = True
        self._operator_cache = {}
        self._scaled_operators = None
        self._graph_cache = {}
        self._reuse_forward = False
        runtime = wp._src.context.runtime
        active_tape = runtime.tape
        runtime.tape = None
        try:
            retain = (lambda value: wp.clone(value, requires_grad=False)) if snapshot else (lambda value: value)
            self.q = retain(q)
            for name in ("mass", "free", "mu", "lame", "indices", "gradients", "volumes"):
                setattr(self, name, retain(getattr(solver, name)))
            self.has_muscles = getattr(solver, "_has_muscles", False)
            if self.has_muscles:
                for name in ("muscle_directions", "muscle_stiffness", "muscle_activation"):
                    setattr(self, name, retain(getattr(solver, name)))
            if self.c:
                self.contact_indices = retain(response.rows.indices)
                self.weights = retain(response.rows.weights)
                self.frames = retain(response.frames)
                self.dq = retain(response.dq)
                self.dl = retain(response.dl)
                self.contact_matrix = retain(response.matrix)
        finally:
            runtime.tape = active_tape

    @classmethod
    def forward(cls, solver, q, response, dt):
        """Refresh an owned forward workspace; backward snapshots never use it.

        Keep at most four recent contact sizes. All captured pointers belong to
        the workspace, and material-dependent inverse replacement invalidates
        the cache rather than retaining obsolete physics in a graph.
        """
        key = (
            response.count,
            float(dt),
            float(solver.mass_damping),
            int(solver.material_code),
            float(response.compliance),
            bool(getattr(solver, "integrate_pinned_velocities", False)),
            float(solver.tolerance / solver.contact_tolerance),
        )
        cache = getattr(solver, "_forward_kkt_cache", [])
        if cache and cache[-1][1].inverse is not solver.inverse:
            cache = []
        system = None
        for index, (old_key, old_system) in enumerate(cache):
            if old_key == key:
                system = old_system
                cache.pop(index)
                break
        if system is None:
            system = cls(solver, q, response, dt)
            system._reuse_forward = True
            system._row_scale = wp.empty(system.size, dtype=wp.float64, device=system.device)
            system._scaled_rhs = wp.empty_like(system._row_scale)
        else:
            wp.copy(system.q, q)
            if system.has_muscles:
                wp.copy(system.muscle_activation, solver.muscle_activation)
            if system.c:
                for target, source in (
                    (system.contact_indices, response.rows.indices),
                    (system.weights, response.rows.weights),
                    (system.frames, response.frames),
                    (system.dq, response.dq),
                    (system.dl, response.dl),
                    (system.contact_matrix, response.matrix),
                ):
                    wp.copy(target, source)
            system._tangent_dirty = True
            system._schur_attempted = False
        # Match the displacement units used by the physical merit. Contact
        # scales can change with geometry even when its row count is unchanged.
        wp.launch(
            _make_row_scale,
            system.size,
            inputs=[
                solver.diagonal,
                response.scales if system.c else solver.diagonal,
                solver.tolerance / solver.contact_tolerance,
                system._row_scale,
            ],
            device=system.device,
        )
        cache.append((key, system))
        solver._forward_kkt_cache = cache[-4:]
        return system

    def _prepare_linearization(self):
        c, device = self.c, self.device
        if self._tangent_dirty:
            # This state is fixed throughout the linear solve. Cache the exact
            # local derivative instead of repeating SVD/projection in GMRES.
            runtime = wp._src.context.runtime
            active_tape = runtime.tape
            runtime.tape = None
            try:
                if self._tangent is None:
                    self._tangent = wp.empty((len(self.indices), 9), dtype=wp.mat33d, device=device)
                tangent = self._tangent
                wp.launch(
                    compute_tet_tangent,
                    (len(self.indices), 9),
                    inputs=[self.q, self.indices, self.gradients, self.mu, self.lame, self.material_code, tangent],
                    device=device,
                )
                if self.has_muscles:
                    wp.launch(
                        add_muscle_tangent,
                        (len(self.indices), 9),
                        inputs=[
                            self.q,
                            self.indices,
                            self.gradients,
                            self.muscle_directions,
                            self.muscle_stiffness,
                            self.muscle_activation,
                            tangent,
                        ],
                        device=device,
                    )
                self._tangent_dirty = False
            finally:
                runtime.tape = active_tape
        if c and not self._schur_attempted:
            self._schur_attempted = True
            # The reduced compliance already includes the regularization.
            # This is the Schur complement of the fixed-PD approximation to
            # the true KKT operator; it changes only the preconditioner.
            runtime = wp._src.context.runtime
            active_tape = runtime.tape
            runtime.tape = None
            try:
                if self._schur_matrix is None:
                    self._schur_matrix = wp.empty_like(self.contact_matrix)
                schur = self._schur_matrix
                wp.launch(
                    contact_jacobian,
                    (3 * c, 3 * c),
                    inputs=[self.contact_matrix, self.dq, self.dl, schur],
                    device=device,
                )
                try:
                    if self._schur_storage is None:
                        self._schur_storage = DenseLU(schur, check=False)
                    else:
                        self._schur_storage.refactor(schur, check=False)
                    invalid = self._schur_storage.invalid.numpy()[0]
                    self._schur_factor = None if invalid else self._schur_storage
                except DenseFactorizationError:
                    self._schur_factor = None
                # A singular approximate Schur complement does not establish
                # singularity of the actual tangent. Retain block-diagonal
                # preconditioning and the unchanged true-residual check.
            finally:
                runtime.tape = active_tape

    def _operators(self, transpose):
        if transpose in self._operator_cache:
            return self._operator_cache[transpose]
        n, c, device = self.n, self.c, self.device
        all_nodes = wp.empty(n, dtype=wp.vec3d, device=device)
        free_nodes = wp.empty_like(all_nodes)
        mechanical = wp.empty_like(all_nodes)
        preconditioned = wp.empty_like(all_nodes)
        contact_displacement = wp.empty(3 * c, dtype=wp.float64, device=device)
        contact_multiplier = wp.empty_like(contact_displacement)
        contact_mechanical = wp.empty_like(contact_displacement)
        contact_zero = wp.zeros_like(contact_displacement)
        contact_rhs = wp.empty_like(contact_displacement)
        contact_solution = wp.empty_like(contact_displacement)

        def matvec(x, y, z, alpha, beta):
            wp.launch(_unpack, n, inputs=[x, self.free, all_nodes, free_nodes], device=device)
            mechanical.zero_()
            wp.launch(
                apply_cached_tet_tangent,
                len(self.indices),
                inputs=[
                    free_nodes if transpose else all_nodes,
                    self.indices,
                    self.gradients,
                    self.volumes,
                    self._tangent,
                    mechanical,
                ],
                device=device,
            )
            if transpose:
                wp.launch(
                    _inertia_transpose,
                    n,
                    inputs=[all_nodes, self.mass, self.free, self.dt, self.damping, mechanical],
                    device=device,
                )
            if c:
                wp.launch(
                    apply_j,
                    c,
                    inputs=[
                        free_nodes if transpose else all_nodes,
                        self.contact_indices,
                        self.weights,
                        self.frames,
                        contact_mechanical,
                    ],
                    device=device,
                )
                if transpose:
                    wp.launch(
                        _contact_transpose,
                        c,
                        inputs=[x, 3 * n, self.dq, self.dl, self.compliance, contact_displacement, contact_multiplier],
                        device=device,
                    )
                else:
                    wp.launch(
                        _contact_forward,
                        c,
                        inputs=[
                            x,
                            3 * n,
                            self.dq,
                            self.dl,
                            self.compliance,
                            contact_mechanical,
                            contact_displacement,
                            contact_multiplier,
                        ],
                        device=device,
                    )
                wp.launch(
                    apply_jt,
                    c,
                    inputs=[contact_displacement, self.contact_indices, self.weights, self.frames, mechanical],
                    device=device,
                )
                wp.launch(
                    _pack_contact,
                    3 * c,
                    inputs=[
                        contact_multiplier,
                        contact_mechanical if transpose else contact_zero,
                        y,
                        3 * n,
                        alpha,
                        beta,
                        z,
                    ],
                    device=device,
                )
            if not transpose:
                wp.launch(
                    _inertia_forward,
                    n,
                    inputs=[all_nodes, self.mass, self.free, self.dt, self.damping, mechanical],
                    device=device,
                )
            wp.launch(_pack_mechanical, n, inputs=[mechanical, y, alpha, beta, z], device=device)

        def precondition(x, y, z, alpha, beta):
            wp.launch(_unpack, n, inputs=[x, self.free, all_nodes, free_nodes], device=device)
            self.inverse.apply(all_nodes, preconditioned)
            if self._schur_factor is None:
                wp.launch(_pack_mechanical, n, inputs=[preconditioned, y, alpha, beta, z], device=device)
                if c:
                    wp.launch(_precondition_contact, 3 * c, inputs=[x, y, 3 * n, alpha, beta, z], device=device)
                return
            # Approximate blocks are B=-D_free J.T and C=dq J. Their
            # different pinned-node masks must also be retained in transpose.
            if transpose:
                wp.launch(_mask_nodes, n, inputs=[preconditioned, self.free, free_nodes], device=device)
            wp.launch(
                apply_j,
                c,
                inputs=[
                    free_nodes if transpose else preconditioned,
                    self.contact_indices,
                    self.weights,
                    self.frames,
                    contact_mechanical,
                ],
                device=device,
            )
            wp.launch(
                _schur_rhs,
                c,
                inputs=[x, 3 * n, contact_mechanical, self.dq, int(transpose), contact_rhs],
                device=device,
            )
            self._schur_factor.solve(contact_rhs, contact_solution, transpose=transpose)
            wp.launch(
                _schur_reaction,
                c,
                inputs=[contact_solution, self.dq, int(transpose), contact_displacement],
                device=device,
            )
            mechanical.zero_()
            wp.launch(
                apply_jt,
                c,
                inputs=[contact_displacement, self.contact_indices, self.weights, self.frames, mechanical],
                device=device,
            )
            if not transpose:
                wp.launch(_mask_nodes, n, inputs=[mechanical, self.free, mechanical], device=device)
            self.inverse.apply(mechanical, all_nodes)
            wp.launch(_pack_preconditioned, n, inputs=[preconditioned, all_nodes, y, alpha, beta, z], device=device)
            wp.launch(
                _pack_contact,
                3 * c,
                inputs=[contact_solution, contact_zero, y, 3 * n, alpha, beta, z],
                device=device,
            )

        operator = LinearOperator((self.size, self.size), wp.float64, device, matvec)
        preconditioner = LinearOperator((self.size, self.size), wp.float64, device, precondition)
        self._operator_cache[transpose] = operator, preconditioner
        return operator, preconditioner

    def _forward_operators(self):
        if self._scaled_operators is not None:
            return self._scaled_operators
        operator, preconditioner = self._operators(False)
        product = wp.empty(self.size, dtype=wp.float64, device=self.device)
        unscaled = wp.empty_like(product)

        def matvec(x, y, z, alpha, beta):
            operator.matvec(x, product, product, 1.0, 0.0)
            wp.launch(
                _row_scaled_update,
                self.size,
                inputs=[product, self._row_scale, y, alpha, beta, False, z],
                device=self.device,
            )

        def precondition(x, y, z, alpha, beta):
            wp.launch(
                _row_scaled_update,
                self.size,
                inputs=[x, self._row_scale, unscaled, 1.0, 0.0, True, unscaled],
                device=self.device,
            )
            preconditioner.matvec(unscaled, y, z, alpha, beta)

        # Row equilibration changes the norm, not the unknown correction:
        # solve D A dx = D rhs using M D^-1 as the approximate inverse.
        self._scaled_operators = (
            LinearOperator((self.size, self.size), wp.float64, self.device, matvec),
            LinearOperator((self.size, self.size), wp.float64, self.device, precondition),
        )
        return self._scaled_operators

    def _forward_graph(self, rhs, operator, preconditioner, restart, relative_tolerance, max_iterations):
        key = (restart, relative_tolerance, max_iterations, self._schur_factor is not None)
        entry = self._graph_cache.get(key)
        if entry is None:
            resident_rhs = wp.clone(rhs, requires_grad=False)
            solution = wp.zeros_like(rhs)
            product = wp.empty_like(rhs)
            metrics = wp.zeros(2, dtype=wp.float64, device=self.device)
            functor = gmres(
                operator,
                resident_rhs,
                solution,
                M=preconditioner,
                tol=relative_tolerance,
                atol=0.0,
                maxiter=max_iterations,
                restart=restart,
                check_every=0,
                use_cuda_graph=True,
                run=False,
            )
            # Compile the conditional loop before capturing its initialization
            # too. Future solves only copy the RHS and replay resident buffers.
            functor()
            with wp.ScopedCapture(device=self.device) as capture:
                solution.zero_()
                functor()
                operator.matvec(solution, product, product, 1.0, 0.0)
                metrics.zero_()
                wp.launch(_verify_residual, self.size, inputs=[product, resident_rhs, metrics], device=self.device)
            entry = resident_rhs, solution, product, metrics, functor, capture.graph
            self._graph_cache[key] = entry
        resident_rhs, solution, _, metrics, _, graph = entry
        wp.copy(resident_rhs, rhs)
        wp.capture_launch(graph)
        return solution, metrics

    def solve(self, rhs, *, transpose=True, relative_tolerance=1e-11, max_iterations=None):
        """Solve a raw packed RHS and verify the true residual.

        Reused forward workspaces equilibrate rows to the physical merit's
        displacement units. Independent systems retain the original row units.
        """
        device = self.device
        if rhs.shape != (self.size,) or rhs.dtype != wp.float64 or rhs.device != device:
            raise ValueError("KKT rhs must be a packed float64 vector on the solver device")
        self._prepare_linearization()
        scaled = self._reuse_forward and not transpose
        if scaled:
            operator, preconditioner = self._forward_operators()
            wp.launch(
                _row_scaled_update,
                self.size,
                inputs=[rhs, self._row_scale, self._scaled_rhs, 1.0, 0.0, False, self._scaled_rhs],
                device=device,
            )
            rhs = self._scaled_rhs
        else:
            operator, preconditioner = self._operators(transpose)
        captured = scaled and device.is_cuda and wp.is_conditional_graph_supported()
        if not captured:
            solution = wp.zeros_like(rhs)
            product = wp.empty_like(rhs)
            metrics = wp.zeros(2, dtype=wp.float64, device=device)
        restarts = (min(7, self.size),) if scaled else (min(31, self.size),)
        if scaled and self.size > 7:
            restarts += (min(31, self.size),)
        if self.size > 31:
            # Stiff heterogeneous systems can stagnate when restarting discards
            # necessary Krylov directions. Retry once with a wider basis only
            # after the original operator fails its unchanged residual check.
            restarts += (min(127, self.size),)
        # Warp replays restart cycles after the first; convergence checks and
        # the true residual verification below retain their original cadence.
        # Keep unrelated graph destructors out of Warp's internal capture.
        gc_enabled = gc.isenabled()
        if device.is_cuda:
            gc.disable()
        try:
            for restart in restarts:
                if captured:
                    solution, metrics = self._forward_graph(
                        rhs,
                        operator,
                        preconditioner,
                        restart,
                        relative_tolerance,
                        max_iterations or max(200, 4 * self.size),
                    )
                else:
                    solution.zero_()
                    gmres(
                        operator,
                        rhs,
                        solution,
                        M=preconditioner,
                        tol=relative_tolerance,
                        atol=0.0,
                        maxiter=max_iterations or max(200, 4 * self.size),
                        restart=restart,
                        use_cuda_graph=device.is_cuda,
                    )
                    operator.matvec(solution, product, product, 1.0, 0.0)
                    metrics.zero_()
                    wp.launch(_verify_residual, self.size, inputs=[product, rhs, metrics], device=device)
                error, magnitude = metrics.numpy()
                if np.isfinite(error) and error <= max(1e-9, 10 * relative_tolerance) * max(magnitude, 1e-30):
                    return solution
        finally:
            if device.is_cuda and gc_enabled:
                gc.enable()
        raise ConvergenceError(f"DiffPhD KKT solve failed its true residual check: {error:.3e}")


def record_adjoint(solver, tape, state_in, state_out, q64, previous64, velocity64, force64, response, dt):
    """Record the converged VJP with independent primal snapshots.

    Positions [m], velocities [m/s], applied forces [N], and elastic coefficients
    [Pa] and fibre contractions receive gradients when their native arrays
    require gradients. Copied mechanics and contact arrays retain the equations
    of this step, including
    when subsequent steps replace the solver's material-dependent factors.
    """
    del previous64, velocity64, force64
    input_q, input_v, input_f = state_in.particle_q, state_in.particle_qd, state_in.particle_f
    output_q, output_v = state_out.particle_q, state_out.particle_qd
    materials = solver.model.tet_materials
    activations = solver._activation_source
    tracked = [
        a
        for a in (input_q, input_v, input_f, output_q, output_v, materials, activations)
        if a is not None and a.requires_grad
    ]
    if not tracked:
        return
    system = KKTSystem(solver, q64, response, dt)
    n, c, device = system.n, system.c, system.device

    def backward():
        with wp.ScopedDevice(device):
            zero_native = wp.zeros(n, dtype=wp.vec3, device=device)
            q_bar = output_q.grad if output_q.grad is not None else zero_native
            v_bar = output_v.grad if output_v.grad is not None else zero_native
            rhs = wp.zeros(system.size, dtype=wp.float64, device=device)
            wp.launch(_seed, n, inputs=[q_bar, v_bar, system.free, dt, rhs], device=device)
            solution = system.solve(rhs, transpose=True)
            all_nodes = wp.empty(n, dtype=wp.vec3d, device=device)
            free_nodes = wp.empty_like(all_nodes)
            wp.launch(_unpack, n, inputs=[solution, system.free, all_nodes, free_nodes], device=device)
            contact_input = wp.zeros(n, dtype=wp.vec3d, device=device)
            if c:
                contact_displacement = wp.empty(3 * c, dtype=wp.float64, device=device)
                contact_multiplier = wp.empty_like(contact_displacement)
                wp.launch(
                    _contact_transpose,
                    c,
                    inputs=[
                        solution,
                        3 * n,
                        system.dq,
                        system.dl,
                        system.compliance,
                        contact_displacement,
                        contact_multiplier,
                    ],
                    device=device,
                )
                wp.launch(_tangential_only, c, inputs=[contact_displacement], device=device)
                wp.launch(
                    apply_jt,
                    c,
                    inputs=[contact_displacement, system.contact_indices, system.weights, system.frames, contact_input],
                    device=device,
                )
            if input_q.grad is not None:
                wp.launch(
                    _position_vjp,
                    n,
                    inputs=[
                        all_nodes,
                        contact_input,
                        v_bar,
                        system.mass,
                        system.free,
                        dt,
                        system.damping,
                        input_q.grad,
                    ],
                    device=device,
                )
            if input_v.grad is not None:
                wp.launch(
                    _velocity_vjp,
                    n,
                    inputs=[
                        all_nodes,
                        v_bar,
                        system.mass,
                        system.free,
                        dt,
                        system.integrate_pinned_velocities,
                        input_v.grad,
                    ],
                    device=device,
                )
            if input_f.grad is not None:
                wp.launch(_force_vjp, n, inputs=[all_nodes, system.free, input_f.grad], device=device)
            if materials.grad is not None:
                dmu = wp.empty(len(system.indices), dtype=wp.float64, device=device)
                dlame = wp.empty_like(dmu)
                wp.launch(
                    material_vjp,
                    len(system.indices),
                    inputs=[
                        system.q,
                        free_nodes,
                        system.indices,
                        system.gradients,
                        system.volumes,
                        system.material_code,
                        dmu,
                        dlame,
                    ],
                    device=device,
                )
                wp.launch(_material_vjp, len(system.indices), inputs=[dmu, dlame, materials.grad], device=device)
            if activations is not None and activations.grad is not None:
                wp.launch(
                    activation_vjp,
                    len(system.indices),
                    inputs=[
                        system.q,
                        free_nodes,
                        system.indices,
                        system.gradients,
                        system.volumes,
                        system.muscle_directions,
                        system.muscle_stiffness,
                        activations.grad,
                    ],
                    device=device,
                )

    tape.record_func(backward, tracked)
