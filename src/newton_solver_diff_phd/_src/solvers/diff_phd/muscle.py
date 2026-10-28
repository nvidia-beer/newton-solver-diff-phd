# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Projective fibre-length energy and its exact implicit derivatives."""

import warp as wp

from .elasticity import tet_deformation


@wp.func
def muscle_stress(f: wp.mat33d, direction: wp.vec3d, stiffness: wp.float64, activation: wp.float64):
    fibre = f * direction
    target = wp.float64(1.0) - activation
    length = wp.length(fibre)
    stress = stiffness * wp.outer(fibre, direction)
    if target != wp.float64(0.0):
        if length > wp.float64(0.0):
            stress *= wp.float64(1.0) - target / length
        else:
            # Nonzero target length has no unique derivative at a collapsed
            # fibre. Admissible positive-J solver states exclude this point.
            stress = wp.mat33d(wp.float64(0.0))
    return stress


@wp.func
def muscle_stress_jvp(f: wp.mat33d, df: wp.mat33d, direction: wp.vec3d, stiffness: wp.float64, activation: wp.float64):
    fibre = f * direction
    delta = df * direction
    target = wp.float64(1.0) - activation
    length = wp.length(fibre)
    derivative = stiffness * delta
    if target != wp.float64(0.0):
        if length > wp.float64(0.0):
            normal = fibre / length
            ratio = target / length
            derivative = stiffness * ((wp.float64(1.0) - ratio) * delta + ratio * wp.dot(normal, delta) * normal)
        else:
            derivative = wp.vec3d(wp.float64(0.0))
    return wp.outer(derivative, direction)


@wp.kernel(enable_backward=False)
def read_activations(
    source: wp.array[wp.float32],
    stiffness: wp.array[wp.float64],
    activation: wp.array[wp.float64],
    status: wp.array[wp.int32],
):
    i = wp.tid()
    value = wp.float64(source[i])
    # A contraction of one has zero target length; negative controls extend.
    if (
        not wp.isfinite(value)
        or value > wp.float64(1.0)
        or (stiffness[i] == wp.float64(0.0) and value != wp.float64(0.0))
    ):
        wp.atomic_or(status, 0, 16)
    activation[i] = value


@wp.kernel(enable_backward=False)
def accumulate_muscle_gradient(
    q: wp.array[wp.vec3d],
    indices: wp.array[wp.vec4i],
    gradients: wp.array2d[wp.vec3d],
    volumes: wp.array[wp.float64],
    directions: wp.array[wp.vec3d],
    stiffness: wp.array[wp.float64],
    activation: wp.array[wp.float64],
    minimum: wp.array[wp.float64],
    out: wp.array[wp.vec3d],
):
    element = wp.tid()
    if minimum[0] > wp.float64(1.0e-10) and stiffness[element] > wp.float64(0.0):
        f = tet_deformation(q, indices, gradients, element)
        stress = muscle_stress(f, directions[element], stiffness[element], activation[element])
        for node in range(4):
            wp.atomic_add(out, indices[element][node], volumes[element] * stress * gradients[element, node])


@wp.kernel(enable_backward=False)
def activation_vjp(
    q: wp.array[wp.vec3d],
    adjoint: wp.array[wp.vec3d],
    indices: wp.array[wp.vec4i],
    gradients: wp.array2d[wp.vec3d],
    volumes: wp.array[wp.float64],
    directions: wp.array[wp.vec3d],
    stiffness: wp.array[wp.float64],
    out: wp.array[wp.float32],
):
    element = wp.tid()
    if stiffness[element] > wp.float64(0.0):
        fibre = tet_deformation(q, indices, gradients, element) * directions[element]
        length = wp.length(fibre)
        if length > wp.float64(0.0):
            adjoint_fibre = tet_deformation(adjoint, indices, gradients, element) * directions[element]
            out[element] -= wp.float32(volumes[element] * stiffness[element] * wp.dot(adjoint_fibre, fibre / length))


@wp.kernel(enable_backward=False)
def add_muscle_tangent(
    q: wp.array[wp.vec3d],
    indices: wp.array[wp.vec4i],
    gradients: wp.array2d[wp.vec3d],
    directions: wp.array[wp.vec3d],
    stiffness: wp.array[wp.float64],
    activation: wp.array[wp.float64],
    tangent: wp.array2d[wp.mat33d],
):
    element, column = wp.tid()
    if stiffness[element] > wp.float64(0.0):
        f = tet_deformation(q, indices, gradients, element)
        basis = wp.mat33d(wp.float64(0.0))
        basis[column // 3, column % 3] = wp.float64(1.0)
        tangent[element, column] += muscle_stress_jvp(
            f, basis, directions[element], stiffness[element], activation[element]
        )
