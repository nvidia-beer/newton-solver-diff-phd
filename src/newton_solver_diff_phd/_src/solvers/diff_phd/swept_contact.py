# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frictionless swept mesh contact with a certified coupled normal response.

This optional contact model is separate from the reference DiffPhD contact
equations.  It supports a deformable left mesh and a fully prescribed right
mesh. Vertex/face and edge/edge features are accumulated across correction
rounds. Once a sweep is clear, their closest points and normals are refreshed
at the corrected endpoint. Each linearized response uses the implicit system
metric. Acceptance requires nonlinear distance complementarity, consistency
between the current normals and mechanical response, and a fresh clear sweep.  Failure is explicit and restores the
input position; there is no automatic derivative of this contact model.
"""

import numpy as np
import warp as wp

from .ccd import SweptMeshCCD, _geometry
from .swept_qp import CertifiedNormalQP

CCD_OVERFLOW = 1
CCD_UNRESOLVED = 2
CCD_INVALID = 4
QP_FAILURE = 8
ITERATION_LIMIT = 16
NONFINITE = 32


@wp.kernel(enable_backward=False)
def _unit_columns(nodes: wp.array[int], columns: wp.array2d[wp.float64]):
    j = wp.tid()
    columns[nodes[j], j] = wp.float64(1.0)


@wp.kernel(enable_backward=False)
def _begin_project(
    status: wp.array[int],
    active: wp.array[int],
    count: wp.array[int],
    flags: wp.array[int],
    iterations: wp.array[int],
    residual: wp.array[wp.float64],
    qp_status: wp.array[int],
    qp_iterations: wp.array[int],
    qp_residual: wp.array[wp.float64],
):
    active[0] = int(status[0] == 0)
    count[0] = 0
    flags[0] = 0
    iterations[0] = 0
    residual[0] = wp.float64(0.0)
    if status[0] == 0:
        qp_status[0] = 0
        qp_iterations[0] = 0
        qp_residual[0] = wp.float64(0.0)


@wp.kernel(enable_backward=False)
def _clear_multiplier(status: wp.array[int], multiplier: wp.array[wp.float64], omega: wp.array[wp.float64]):
    i = wp.tid()
    if status[0] == 0:
        multiplier[i] = wp.float64(0.0)
        omega[i] = wp.float64(0.0)


@wp.kernel(enable_backward=False)
def _clear_residual(residual: wp.array[wp.float64]):
    residual[0] = wp.float64(0.0)


@wp.kernel(enable_backward=False)
def _refresh_geometry(
    position: wp.array[wp.vec3d],
    indices: wp.array[wp.vec4i],
    count: wp.array[int],
    split: int,
    hits: wp.array[int],
    flags: wp.array[int],
    coefficients: wp.array[wp.vec4d],
    normals: wp.array[wp.vec3d],
    status: wp.array[int],
):
    row = wp.tid()
    # Until CCD certifies the current sweep, its witness normals preserve the
    # collision side. Afterwards actual closest-point directions allow safe
    # sliding around features without retaining obsolete separating planes.
    if row >= count[0] or hits[0] != 0 or flags[0] != 0:
        return
    ids = indices[row]
    kind = int(ids[0] < split and ids[1] < split)
    distance, coeff, delta, _fallback = _geometry(position, position, ids, kind, wp.float64(0.0))
    normal = normals[row]
    if not wp.isfinite(distance):
        wp.atomic_or(status, 0, NONFINITE)
        return
    if distance > wp.float64(0.0):
        updated = delta / distance
        normal = updated
    coefficients[row] = coeff
    normals[row] = normal


@wp.kernel(enable_backward=False)
def _nonlinear_certificate(
    position: wp.array[wp.vec3d],
    indices: wp.array[wp.vec4i],
    coefficients: wp.array[wp.vec4d],
    normals: wp.array[wp.vec3d],
    count: wp.array[int],
    margin: wp.float64,
    matrix: wp.array2d[wp.float64],
    multiplier: wp.array[wp.float64],
    residual: wp.array[wp.float64],
    status: wp.array[int],
):
    row = wp.tid()
    if row >= count[0]:
        return
    value = wp.vec3d(0.0)
    for k in range(4):
        value += coefficients[row][k] * position[indices[row][k]]
    gap = wp.dot(normals[row], value) - margin
    natural = wp.abs(wp.min(gap, matrix[row, row] * multiplier[row]))
    if not wp.isfinite(gap) or not wp.isfinite(multiplier[row]) or multiplier[row] < wp.float64(0.0):
        wp.atomic_or(status, 0, NONFINITE)
        wp.atomic_max(residual, 0, wp.float64(wp.inf))
    else:
        wp.atomic_max(residual, 0, natural)


@wp.kernel(enable_backward=False)
def _mechanical_certificate(
    base: wp.array[wp.vec3d],
    position: wp.array[wp.vec3d],
    free: wp.array[int],
    indices: wp.array[wp.vec4i],
    coefficients: wp.array[wp.vec4d],
    normals: wp.array[wp.vec3d],
    count: wp.array[int],
    carrier: wp.array[int],
    panel: wp.array2d[wp.float64],
    multiplier: wp.array[wp.float64],
    residual: wp.array[wp.float64],
    status: wp.array[int],
):
    node = wp.tid()
    expected = base[node]
    if free[node] != 0:
        for row in range(count[0]):
            if multiplier[row] > wp.float64(0.0):
                weight = wp.float64(0.0)
                for k in range(4):
                    column = carrier[indices[row][k]]
                    if column >= 0:
                        weight += coefficients[row][k] * panel[node, column]
                expected += weight * multiplier[row] * normals[row]
    error = wp.length(expected - position[node])
    if not wp.isfinite(error):
        wp.atomic_or(status, 0, NONFINITE)
        wp.atomic_max(residual, 0, wp.float64(wp.inf))
    else:
        wp.atomic_max(residual, 0, error)


@wp.kernel(enable_backward=False)
def _decide(
    flags: wp.array[int],
    hit_count: wp.array[int],
    residual: wp.array[wp.float64],
    tolerance: wp.float64,
    maximum: int,
    status: wp.array[int],
    active: wp.array[int],
    iterations: wp.array[int],
):
    # An unresolved candidate sweep may still be corrected when the contact
    # equations require a response. It cannot be accepted: the corrected
    # path needs a new, independently clear CCD query. Capacity/invalid data
    # cannot supply a trustworthy complete response and fail immediately.
    fatal = flags[0] & (CCD_OVERFLOW | CCD_INVALID)
    if fatal != 0:
        status[0] = status[0] | flags[0]
    elif flags[0] & CCD_UNRESOLVED != 0 and residual[0] <= tolerance:
        status[0] = status[0] | CCD_UNRESOLVED
    if status[0] != 0:
        active[0] = 0
    elif flags[0] == 0 and hit_count[0] == 0 and residual[0] <= tolerance:
        active[0] = 0
    elif iterations[0] >= maximum:
        status[0] = ITERATION_LIMIT | flags[0]
        active[0] = 0
    else:
        iterations[0] += 1


@wp.kernel(enable_backward=False)
def _assemble(
    base: wp.array[wp.vec3d],
    indices: wp.array[wp.vec4i],
    coefficients: wp.array[wp.vec4d],
    normals: wp.array[wp.vec3d],
    count: wp.array[int],
    carrier: wp.array[int],
    panel: wp.array2d[wp.float64],
    active: wp.array[int],
    margin: wp.float64,
    matrix: wp.array2d[wp.float64],
    gap: wp.array[wp.float64],
    candidate: wp.array[int],
):
    i, j = wp.tid()
    enabled = active[0] != 0 and i < count[0]
    matrix[i, j] = wp.float64(0.0)
    if j == 0:
        candidate[i] = int(enabled)
        gap[i] = wp.float64(0.0)
        if enabled:
            point = wp.vec3d(0.0)
            for k in range(4):
                point += coefficients[i][k] * base[indices[i][k]]
            gap[i] = wp.dot(normals[i], point) - margin
    if enabled and j < count[0]:
        value = wp.float64(0.0)
        for a in range(4):
            node_a = indices[i][a]
            if carrier[node_a] >= 0:
                for b in range(4):
                    column_b = carrier[indices[j][b]]
                    if column_b >= 0:
                        value += coefficients[i][a] * coefficients[j][b] * panel[node_a, column_b]
        matrix[i, j] = wp.dot(normals[i], normals[j]) * value


@wp.kernel(enable_backward=False)
def _check_qp(
    qp_status: wp.array[int],
    qp_residual: wp.array[wp.float64],
    status: wp.array[int],
    active: wp.array[int],
    residual: wp.array[wp.float64],
):
    if active[0] != 0:
        residual[0] = qp_residual[0]
        if qp_status[0] != 0:
            status[0] = status[0] | QP_FAILURE
            active[0] = 0


@wp.kernel(enable_backward=False)
def _correct(
    base: wp.array[wp.vec3d],
    free: wp.array[int],
    indices: wp.array[wp.vec4i],
    coefficients: wp.array[wp.vec4d],
    normals: wp.array[wp.vec3d],
    count: wp.array[int],
    carrier: wp.array[int],
    panel: wp.array2d[wp.float64],
    multiplier: wp.array[wp.float64],
    active: wp.array[int],
    out: wp.array[wp.vec3d],
):
    node = wp.tid()
    if active[0] == 0:
        return
    value = base[node]
    if free[node] != 0:
        for row in range(count[0]):
            if multiplier[row] > wp.float64(0.0):
                weight = wp.float64(0.0)
                for k in range(4):
                    column = carrier[indices[row][k]]
                    if column >= 0:
                        weight += coefficients[row][k] * panel[node, column]
                value += weight * multiplier[row] * normals[row]
    out[node] = value


@wp.kernel(enable_backward=False)
def _restore_failed(start: wp.array[wp.vec3d], status: wp.array[int], out: wp.array[wp.vec3d]):
    node = wp.tid()
    if status[0] != 0:
        out[node] = start[node]


class SweptContact:
    """Reusable, forward-only CCD response for a prescribed mesh probe.

    ``build`` is setup and may allocate. ``project`` has no allocations or
    host copies on CUDA with conditional graph support. The CPU fallback
    reads only loop-control scalars. ``begin_step`` clears a sticky failure;
    every project call preserves a previous failure until that reset.

    Status bits are 1 capacity, 2 unresolved CCD, 4 invalid geometry, 8 failed
    normal QP, 16 enrichment budget, and 32 nonfinite arithmetic. A finite
    iteration budget is never reported as successful convergence.
    """

    def __init__(
        self,
        faces_left,
        faces_right,
        split,
        free_mask,
        margin,
        device=None,
        *,
        friction=0.0,
        capacity=1024,
        max_rounds=32,
        tolerance=1.0e-7,
    ):
        self.device = wp.get_device(device)
        self.free_mask = np.asarray(free_mask, dtype=bool).copy()
        if self.free_mask.ndim != 1:
            raise ValueError("free_mask must be a one-dimensional particle mask")
        self.size = len(self.free_mask)
        self.split = int(split)
        if not 0 < self.split < self.size:
            raise ValueError("swept contact split must separate two nonempty particle sets")
        self.validate_free_mask(self.free_mask)
        if not np.isfinite(friction) or friction != 0.0:
            raise ValueError("swept contact currently supports only friction=0")
        self.margin = float(margin)
        self.tolerance = float(tolerance)
        if not np.isfinite(self.margin) or self.margin <= 0.0:
            raise ValueError("swept contact margin must be finite and positive")
        if not np.isfinite(self.tolerance) or not 0.0 < self.tolerance < self.margin:
            raise ValueError("swept contact tolerance must be finite, positive, and smaller than margin")
        if not isinstance(capacity, (int, np.integer)) or capacity < 1:
            raise ValueError("swept contact capacity must be a positive integer")
        if not isinstance(max_rounds, (int, np.integer)) or max_rounds < 1:
            raise ValueError("swept contact max_rounds must be a positive integer")
        self.capacity = int(capacity)
        self.max_rounds = int(max_rounds)
        left = self._validate_faces(faces_left, 0, self.split, "left")
        right = self._validate_faces(faces_right, self.split, self.size, "right")
        self.detector = SweptMeshCCD(
            left, right, self.size, margin=self.margin, thickness=min(1.0e-8, 0.01 * self.margin), device=self.device
        )
        nodes = np.unique(left)
        nodes = nodes[self.free_mask[nodes]]
        self.carrier_count = len(nodes)
        self.nodes = wp.array(nodes, dtype=int, device=self.device)
        carrier = np.full(self.size, -1, dtype=np.int32)
        carrier[nodes] = np.arange(self.carrier_count, dtype=np.int32)
        self.carrier = wp.array(carrier, dtype=int, device=self.device)
        self.free = wp.array(self.free_mask.astype(np.int32), dtype=int, device=self.device)
        self.indices = wp.zeros(self.capacity, dtype=wp.vec4i, device=self.device)
        self.coefficients = wp.zeros(self.capacity, dtype=wp.vec4d, device=self.device)
        self.normals = wp.zeros(self.capacity, dtype=wp.vec3d, device=self.device)
        self.tois = wp.zeros(self.capacity, dtype=wp.float64, device=self.device)
        for name in ("count", "flags", "status", "active", "iterations"):
            setattr(self, name, wp.zeros(1, dtype=int, device=self.device))
        self.residual = wp.zeros(1, dtype=wp.float64, device=self.device)
        self.matrix = wp.zeros((self.capacity, self.capacity), dtype=wp.float64, device=self.device)
        self.gap = wp.zeros(self.capacity, dtype=wp.float64, device=self.device)
        self.candidate = wp.zeros(self.capacity, dtype=int, device=self.device)
        self.multiplier = wp.zeros(self.capacity, dtype=wp.float64, device=self.device)
        self.omega = wp.zeros(self.capacity, dtype=wp.float64, device=self.device)
        self.base = wp.empty(self.size, dtype=wp.vec3d, device=self.device)
        # Nearly dependent features amplify gap errors into response errors.
        # Reserve accuracy for the nonlinear geometry and force certificates.
        self.qp = CertifiedNormalQP(self.capacity, self.device, tolerance=1.0e-5 * self.tolerance)
        self.qp_status = self.qp.status
        self.qp_residual = self.qp.residual
        self.panel = None
        self.graph = None
        self.graph_key = None

    @staticmethod
    def _validate_faces(source, lower, upper, name):
        source = np.asarray(source)
        if source.ndim != 2 or source.shape[1] != 3 or not len(source):
            raise ValueError(f"{name} swept contact faces must have nonempty shape (faces, 3)")
        if not np.issubdtype(source.dtype, np.integer) or np.any(source < lower) or np.any(source >= upper):
            raise ValueError(f"{name} swept contact faces must use global particle indices in [{lower}, {upper})")
        if (
            np.any(source[:, 0] == source[:, 1])
            or np.any(source[:, 0] == source[:, 2])
            or np.any(source[:, 1] == source[:, 2])
        ):
            raise ValueError(f"{name} swept contact faces must have three distinct vertices")
        return source.astype(np.int32)

    def validate_free_mask(self, free_mask):
        mask = np.asarray(free_mask, dtype=bool)
        if mask.shape != (self.size,):
            raise ValueError("swept contact free mask must match the particle count")
        if np.any(mask[self.split :]):
            raise ValueError("swept contact requires every right-side probe particle to be prescribed")
        if not np.array_equal(mask, self.free_mask):
            raise ValueError("changing swept contact free particles requires constructing a new solver")

    def build(self, inverse, dt):
        """Cache inverse columns for every free left-surface carrier (setup)."""
        if inverse.size != self.size or inverse.device != self.device:
            raise ValueError("swept contact inverse must match its particle count and device")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("swept contact timestep must be finite and positive")
        columns = wp.zeros((self.size, max(1, self.carrier_count)), dtype=wp.float64, device=self.device)
        if self.carrier_count:
            wp.launch(_unit_columns, self.carrier_count, inputs=[self.nodes], outputs=[columns], device=self.device)
        self.panel = wp.zeros_like(columns)
        inverse.apply_batch(columns, self.panel)
        self.graph = None
        self.graph_key = None

    def begin_step(self):
        """Clear a previous physical step's failure before starting the next."""
        self.status.zero_()

    def _iteration(self):
        start, out = self.arguments
        # These are query-local flags. Only a fully certified corrected sweep
        # can terminate successfully; self.status keeps actual failures sticky.
        self.flags.zero_()
        self.detector.append(
            start, out, self.indices, self.coefficients, self.normals, self.tois, self.count, self.flags
        )
        wp.launch(
            _refresh_geometry,
            self.capacity,
            inputs=[out, self.indices, self.count, self.split, self.detector.hit_count, self.flags],
            outputs=[self.coefficients, self.normals, self.status],
            device=self.device,
        )
        wp.launch(
            _assemble,
            (self.capacity, self.capacity),
            inputs=[
                self.base,
                self.indices,
                self.coefficients,
                self.normals,
                self.count,
                self.carrier,
                self.panel,
                self.active,
                self.margin,
            ],
            outputs=[self.matrix, self.gap, self.candidate],
            device=self.device,
        )
        wp.launch(_clear_residual, 1, outputs=[self.residual], device=self.device)
        wp.launch(
            _nonlinear_certificate,
            self.capacity,
            inputs=[
                out,
                self.indices,
                self.coefficients,
                self.normals,
                self.count,
                self.margin,
                self.matrix,
                self.multiplier,
            ],
            outputs=[self.residual, self.status],
            device=self.device,
        )
        wp.launch(
            _mechanical_certificate,
            self.size,
            inputs=[
                self.base,
                out,
                self.free,
                self.indices,
                self.coefficients,
                self.normals,
                self.count,
                self.carrier,
                self.panel,
                self.multiplier,
            ],
            outputs=[self.residual, self.status],
            device=self.device,
        )
        wp.launch(
            _decide,
            1,
            inputs=[self.flags, self.detector.hit_count, self.residual, self.tolerance, self.max_rounds],
            outputs=[self.status, self.active, self.iterations],
            device=self.device,
        )
        wp.capture_if(self.active, self._response)

    def _response(self):
        _start, out = self.arguments
        self.qp.solve(self.matrix, self.gap, self.candidate, self.multiplier, self.omega)
        wp.launch(
            _check_qp,
            1,
            inputs=[self.qp.status, self.qp.residual],
            outputs=[self.status, self.active, self.residual],
            device=self.device,
        )
        wp.launch(
            _correct,
            self.size,
            inputs=[
                self.base,
                self.free,
                self.indices,
                self.coefficients,
                self.normals,
                self.count,
                self.carrier,
                self.panel,
                self.multiplier,
                self.active,
            ],
            outputs=[out],
            device=self.device,
        )

    def project(self, start, candidate, out):
        """Project the full commanded endpoint; failure restores ``start``.

        All arrays have shape ``(particle_count,)`` and dtype ``wp.vec3d``.
        ``candidate`` already contains prescribed endpoint positions. Return
        the device status and physical gap residual arrays without a host read.
        """
        if self.panel is None:
            raise RuntimeError("build swept contact before projecting")
        for name, array in (("start", start), ("candidate", candidate), ("out", out)):
            if array.dtype != wp.vec3d or array.shape != (self.size,) or array.device != self.device:
                raise ValueError(f"{name} must be a vec3d particle array on {self.device}")
        wp.copy(self.base, candidate)
        wp.copy(out, self.base)
        wp.launch(
            _begin_project,
            1,
            inputs=[self.status],
            outputs=[
                self.active,
                self.count,
                self.flags,
                self.iterations,
                self.residual,
                self.qp.status,
                self.qp.iterations,
                self.qp.residual,
            ],
            device=self.device,
        )
        wp.launch(
            _clear_multiplier,
            self.capacity,
            inputs=[self.status],
            outputs=[self.multiplier, self.omega],
            device=self.device,
        )
        self.arguments = (start, out)
        if self.device.is_capturing:
            wp.capture_while(self.active, self._iteration)
        elif self.device.is_cuda and wp.is_conditional_graph_supported():
            key = (start.ptr, out.ptr)
            if self.graph is None or self.graph_key != key:
                with wp.ScopedCapture(device=self.device) as capture:
                    wp.capture_while(self.active, self._iteration)
                self.graph = capture.graph
                self.graph_key = key
            wp.capture_launch(self.graph)
        else:
            for _ in range(self.max_rounds + 1):
                if not self.active.numpy()[0]:
                    break
                self._iteration()
        wp.launch(_restore_failed, self.size, inputs=[start, self.status], outputs=[out], device=self.device)
        return self.status, self.residual
