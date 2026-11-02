# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Portions translated from DiffPhD, commit d319435, Copyright (c) 2021
# MIT Graphics Group, MIT License. See THIRD_PARTY_NOTICES.md.

"""Native regular Hex8 quadrature and projective vertex energies.

The reference HexDeformable uses one shared cubic cell width, obtained from
the mean rest cell volume, and eight tensor-product Gauss samples. These are
trilinear finite elements; no tetrahedra are constructed.
"""

import numpy as np
import warp as wp

from .kernels import STATUS_ACTIVATION, STATUS_SINGULAR
from .projections import (
    corotated_projection,
    diagonal_matrix,
    muscle_projection,
    nh_prox_stretches,
    unsigned_svd,
    volume_offset,
)


class HexElements:
    """Validated native binary-order cubes and their shared Gauss gradients."""

    def __init__(self, mesh, particle_count, device):
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        indices = np.asarray(mesh.indices)
        if vertices.shape != (particle_count, 3) or not np.isfinite(vertices).all():
            raise ValueError("hex_mesh vertices must contain a finite rest position for every model particle")
        if (
            indices.ndim != 2
            or indices.shape[1] != 8
            or len(indices) == 0
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
            or np.any(indices >= particle_count)
            or np.any(np.diff(np.sort(indices, axis=1), axis=1) == 0)
        ):
            raise ValueError("hex_mesh indices must have shape (element_count,8) with eight distinct valid vertices")
        if not np.array_equal(np.unique(indices), np.arange(particle_count)):
            raise ValueError("Every particle must belong to a hexahedron")
        self.elements = indices.astype(np.int32, copy=True)
        corners = np.array([[a, b, c] for a in range(2) for b in range(2) for c in range(2)], dtype=np.float64)
        rest = vertices[self.elements]
        edges = rest[:, [1, 2, 4]] - rest[:, :1]
        # Match Mesh<3,8>::ComputeElementVolume, including its global arithmetic
        # mean rather than incident-volume mass lumping.
        self.volumes = np.abs(np.einsum("ij,ij->i", edges[:, 0], np.cross(edges[:, 1], edges[:, 2])))
        self.mean_volume = sum(float(value) for value in self.volumes) / len(self.elements)
        if not np.isfinite(self.mean_volume) or self.mean_volume <= 0 or np.any(self.volumes <= 0):
            raise ValueError("Hex8 cells must have positive, finite rest volume")
        dx = self.mean_volume ** (1.0 / 3.0)
        expected = rest[:, :1] + dx * corners[None]
        if not np.allclose(rest, expected, rtol=0.0, atol=1.0e-9 * dx):
            raise ValueError("Reference Hex8 requires equally sized axis-aligned cubes in binary corner order")
        self.rest = vertices.copy()
        self.width = dx
        inv_dx = 1.0 / dx
        samples = ((corners - 0.5) * (1.0 / np.sqrt(3.0)) + 0.5) * dx
        gradients = np.empty((8, 8, 3), dtype=np.float64)
        for s, point in enumerate(samples):
            nx, ny, nz = point * inv_dx
            cnx, cny, cnz = 1.0 - nx, 1.0 - ny, 1.0 - nz
            # Keep the native multiplication order (HexDeformable samples).
            gradients[s] = [
                [-inv_dx * cny * cnz, cnx * -inv_dx * cnz, cnx * cny * -inv_dx],
                [-inv_dx * cny * nz, cnx * -inv_dx * nz, cnx * cny * inv_dx],
                [-inv_dx * ny * cnz, cnx * inv_dx * cnz, cnx * ny * -inv_dx],
                [-inv_dx * ny * nz, cnx * inv_dx * nz, cnx * ny * inv_dx],
                [inv_dx * cny * cnz, nx * -inv_dx * cnz, nx * cny * -inv_dx],
                [inv_dx * cny * nz, nx * -inv_dx * nz, nx * cny * inv_dx],
                [inv_dx * ny * cnz, nx * inv_dx * cnz, nx * ny * -inv_dx],
                [inv_dx * ny * nz, nx * inv_dx * nz, nx * ny * inv_dx],
            ]
        self.sample_gradients = gradients
        self.indices = wp.array(self.elements, dtype=wp.int32, device=device)
        self.gradients = wp.array(gradients, dtype=wp.vec3d, device=device)
        # Each (element, sample) has eight corner contributions. Stable grouping
        # sums in element/sample order without atomics in the numerical kernel.
        vertices_by_entry = np.repeat(self.elements, 8, axis=0).ravel()
        counts = np.bincount(vertices_by_entry, minlength=particle_count)
        offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
        entries = np.argsort(vertices_by_entry, kind="stable").astype(np.int32)
        self.incidence = (
            wp.array(offsets, dtype=wp.int32, device=device),
            wp.array(entries, dtype=wp.int32, device=device),
        )


@wp.kernel(enable_backward=False)
def element_weights(
    materials: wp.array2d[wp.float64],
    material: int,
    sample_volume: wp.float64,
    corotated: wp.array[wp.float64],
    volumetric: wp.array[wp.float64],
    neo_hookean: wp.array[wp.float64],
):
    e = wp.tid()
    corotated[e] = wp.float64(0.0)
    volumetric[e] = wp.float64(0.0)
    neo_hookean[e] = wp.float64(0.0)
    if material == 0:
        corotated[e] = wp.float64(2.0) * materials[e, 0] * sample_volume
        volumetric[e] = materials[e, 1] * sample_volume
    else:
        neo_hookean[e] = wp.float64(2.0) * materials[e, 0] * sample_volume


@wp.kernel(enable_backward=False)
def validate_targets(targets: wp.array[wp.float64], status: wp.array[wp.int32]):
    e = wp.tid()
    if not wp.isfinite(targets[e]) or targets[e] < wp.float64(0.0):
        wp.atomic_or(status, 0, STATUS_ACTIVATION)


@wp.kernel(enable_backward=False)
def local_step(
    x: wp.array[wp.vec3d],
    boundary: wp.array[wp.vec3d],
    indices: wp.array2d[wp.int32],
    gradients: wp.array2d[wp.vec3d],
    corotated_weight: wp.array[wp.float64],
    volume_weight: wp.array[wp.float64],
    neo_hookean_weight: wp.array[wp.float64],
    nh_mu: wp.float64,
    nh_lame: wp.float64,
    nh_penalty: wp.float64,
    muscle_direction: wp.array[wp.vec3d],
    muscle_weight: wp.array[wp.float64],
    muscle_target: wp.array[wp.float64],
    contributions: wp.array2d[wp.vec3d],
    status: wp.array[wp.int32],
):
    sample = wp.tid()
    e = sample // 8
    s = sample % 8
    f = wp.mat33d(wp.float64(0.0))
    f_boundary = wp.mat33d(wp.float64(0.0))
    for j in range(8):
        g = gradients[s, j]
        f += wp.outer(x[indices[e, j]], g)
        f_boundary += wp.outer(boundary[indices[e, j]], g)
    stress = wp.mat33d(wp.float64(0.0))
    wc = corotated_weight[e]
    wv = volume_weight[e]
    wn = neo_hookean_weight[e]
    u, sigma, v = unsigned_svd(f)
    if wc != wp.float64(0.0):
        stress += wc * (corotated_projection(u, v) - f_boundary)
    if wv != wp.float64(0.0):
        offset, singular = volume_offset(sigma)
        if singular != 0:
            wp.atomic_or(status, 0, STATUS_SINGULAR)
        stress += wv * (u * diagonal_matrix(sigma + offset) * wp.transpose(v) - f_boundary)
    if wn != wp.float64(0.0):
        stretches = nh_prox_stretches(sigma, nh_mu, nh_lame, nh_penalty)
        stress += wn * (u * diagonal_matrix(stretches) * wp.transpose(v) - f_boundary)
    wm = muscle_weight[e]
    if wm != wp.float64(0.0):
        m = muscle_direction[e]
        stress += wm * wp.outer(muscle_projection(f, m, muscle_target[e]) - f_boundary * m, m)
    for j in range(8):
        contributions[sample, j] = stress * gradients[s, j]


@wp.kernel(enable_backward=False)
def gather(
    offsets: wp.array[wp.int32],
    entries: wp.array[wp.int32],
    contributions: wp.array2d[wp.vec3d],
    accumulate: int,
    out: wp.array[wp.vec3d],
):
    i = wp.tid()
    total = wp.vec3d(wp.float64(0.0))
    for a in range(offsets[i], offsets[i + 1]):
        entry = entries[a]
        total += contributions[entry // 8, entry % 8]
    if accumulate != 0:
        total += out[i]
    out[i] = total


@wp.kernel(enable_backward=False)
def deformation_determinants(
    q: wp.array[wp.vec3],
    indices: wp.array2d[wp.int32],
    gradients: wp.array2d[wp.vec3d],
    determinants: wp.array2d[wp.float64],
):
    e, s = wp.tid()
    f = wp.mat33d(wp.float64(0.0))
    for j in range(8):
        f += wp.outer(wp.vec3d(q[indices[e, j]]), gradients[s, j])
    determinants[e, s] = wp.determinant(f)


class PlanarEnergies:
    """Reference planar PD vertex energies, independent of frictional contact."""

    def __init__(self, specification, particle_count, device):
        indices, stiffness, plane = specification
        indices = np.asarray(indices)
        if (
            indices.ndim != 1
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
            or np.any(indices >= particle_count)
            or len(np.unique(indices)) != len(indices)
        ):
            raise ValueError("planar_energies needs distinct valid particle indices")
        stiffness = np.broadcast_to(np.asarray(stiffness, dtype=np.float64), indices.shape)
        plane = np.asarray(plane, dtype=np.float64)
        normal_length = np.linalg.norm(plane[:3]) if plane.shape == (4,) else 0.0
        if (
            plane.shape != (4,)
            or not np.isfinite(plane).all()
            or not np.isfinite(normal_length)
            or normal_length <= 1.0e-5
        ):
            raise ValueError("planar_energies needs a finite (nx,ny,nz,offset) with nonzero normal")
        if not np.isfinite(stiffness).all() or np.any(stiffness < 0):
            raise ValueError("planar_energies stiffness must be finite and nonnegative")
        self.vertices = indices.astype(np.int32, copy=True)
        self.stiffness = stiffness.copy()
        self.indices = wp.array(self.vertices, dtype=wp.int32, device=device)
        self.weights = wp.array(stiffness, dtype=wp.float64, device=device)
        self.plane = wp.vec4d(*(plane / normal_length))

    def assemble(self, x, boundary, projection, device):
        wp.launch(
            planar_projection,
            len(self.vertices),
            inputs=[x, boundary, self.indices, self.weights, self.plane],
            outputs=[projection],
            device=device,
        )


@wp.kernel(enable_backward=False)
def planar_projection(
    x: wp.array[wp.vec3d],
    boundary: wp.array[wp.vec3d],
    indices: wp.array[wp.int32],
    weights: wp.array[wp.float64],
    plane: wp.vec4d,
    projection: wp.array[wp.vec3d],
):
    entry = wp.tid()
    i = indices[entry]
    n = wp.vec3d(plane[0], plane[1], plane[2])
    d = wp.dot(n, x[i]) + plane[3]
    p = x[i] - wp.min(wp.float64(0.0), d) * n
    projection[i] += weights[entry] * (p - boundary[i])
