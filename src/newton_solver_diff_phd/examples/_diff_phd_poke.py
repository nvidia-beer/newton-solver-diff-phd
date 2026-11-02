# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Original DiffPhD training physics on a prepared PokeFlex tetrahedral scene.

The contact law, material parameters, root constraints and rollout settings
follow ``dice_material_optimize.py``. The default prepared meshes are distinct
from that script's generated slab and 24-section TetGen cylinder.
"""

from pathlib import Path

import newton
import numpy as np
import warp as wp

from newton_solver_diff_phd import SolverDiffPhD, load_mesh

from ._poke_recording import load_poke_recording


def _rotation(axis: int, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    if axis == 0:
        return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))
    if axis == 1:
        return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def reference_tool_frame(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the original UR5 end-effector frame [m], with world Z shifted 0.5 m.

    Joint order is shoulder pan/lift, elbow, wrist 1/2/3 [rad]. The probe
    extends 0.10 m along local +X. Dimensions and fixed joint rotations are
    the UR5 chain used in DiffPhD's PokeFlex alignment reference, not UR5e IK.
    """
    offsets = ((0, 0, 0.089159), (0, 0.13585, 0), (0, -0.1197, 0.425), (0, 0, 0.39225), (0, 0.093, 0), (0, 0, 0.09465))
    axes = (2, 1, 1, 1, 2, 1)
    position, rotation = np.array((0.0, 0.0, 0.51)), np.eye(3)
    for i, (offset, axis, joint) in enumerate(zip(offsets, axes, joints, strict=True)):
        position += rotation @ offset
        fixed = _rotation(1, np.pi / 2) if i in (1, 3) else np.eye(3)
        rotation = rotation @ fixed @ _rotation(axis, float(joint))
    position += rotation @ (0.0, 0.0823, 0.0)
    return position, rotation @ _rotation(2, np.pi / 2)


def reference_joint_trajectory(substeps: int) -> np.ndarray:
    """Return 35 scan-frame intervals of hand-authored reference joint commands [rad].

    The first interval holds the ready pose. Subsequent intervals linearly
    interpolate consecutive source keyframes, matching the source rollout's
    frame-19 through frame-53 comparison schedule at 30 Hz. These commands
    come from the reference script, not measured robot poses in the recording.
    """
    if isinstance(substeps, bool) or not isinstance(substeps, (int, np.integer)) or substeps < 1:
        raise ValueError("substeps must be a positive integer")
    ready = np.array((0.0, -0.45, 1.10, -2.228, -np.pi / 2, 0.0))
    keyframes = np.tile(ready, (35, 1))

    def smooth(value):
        return (1.0 - np.cos(np.pi * np.clip(value, 0.0, 1.0))) / 2.0

    for i in range(35):
        if i < 8:
            offset = 0.0095 * smooth(i / 8)
        elif i < 23:
            offset = 0.0095 + (0.18 - 0.0095) * smooth((i - 8) / 15)
        else:
            offset = 0.18 * (1.0 - smooth((i - 23) / 11))
        keyframes[i, 1] += offset
    result = [ready]
    for i, current in enumerate(keyframes):
        previous = keyframes[max(i - 1, 0)]
        for step in range(substeps):
            alpha = (step + 1) / substeps
            result.append((1.0 - alpha) * previous + alpha * current)
    return np.asarray(result)


def _append_mesh(builder, vertices, tets, young, poisson, density):
    start = len(builder.particle_q)
    for point in vertices:
        builder.add_particle(wp.vec3(*point), wp.vec3(), mass=0.0, radius=0.00055)
    for original in tets:
        tet = original.copy()
        if np.linalg.det(vertices[tet[1:]] - vertices[tet[0]]) < 0.0:
            tet[[1, 2]] = tet[[2, 1]]
        indices = tuple(int(i + start) for i in tet)
        volume = builder.add_tetrahedron(
            *indices,
            k_mu=young / (2.0 * (1.0 + poisson)),
            k_lambda=young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson)),
            k_damp=0.0,
        )
        for i in indices:
            builder.particle_mass[i] += density * volume / 4.0


# Executed defaults from the reference dice_material_optimize.py.
CONTACT_RADIUS = 5.5e-4
SUBSTEPS = 30
FRICTION = 0.5
POKE_PHYSICS = {
    "reference": "dice_material_optimize.py",
    "material": "neo_hookean",
    "contact_mode": "mesh_contact",
    "contact_kn_left": 5.0e3,
    "contact_kn_right": 5.0e4,
    "contact_kf": 50.0,
    "probe_young": 1.0e13,
    "probe_poisson": 0.45,
    "density": 800.0,
    "gravity": [0.0, 0.0, 0.0],
    "probe_prescription": "root_cap_local_z_minus_min_below_1e-4_m",
    "dirichlet_motion": "explicit_endpoint_positions",
    "contact_velocity": "incoming_physical_velocity",
    "velocity_damping_scope": "all_particles_after_each_substep",
    "max_iterations": 5000,
    "abs_tol": 1.0e-9,
    "rel_tol": 1.0e-4,
    "anderson_window": 5,
}


def build_poke_scene(
    recording: str | Path | None = None,
    substeps: int = SUBSTEPS,
    device=None,
    *,
    reference=False,
    young=5.0e5,
    poisson=0.40,
    friction=FRICTION,
    contact_radius=CONTACT_RADIUS,
    swept_contact=False,
) -> dict:
    """Build the native tetrahedral die/probe model and prescribed trajectory.

    The die defaults to E=500 kPa, Poisson ratio 0.40 and density 800 kg/m³,
    with the prepared recording's bottom vertices clamped and gravity disabled.
    The probe has E=1e13 Pa, Poisson ratio 0.45 and the same density; only its
    root cap is prescribed, and the remaining probe particles are free. Both
    bodies interact through the reference bidirectional ``mesh_contact`` state
    force with ``contact_radius`` [m], stiffnesses 5e3/5e4 N/m, tangential
    damping 50 N s/m and coefficient ``friction``. The prescribed trajectory uses
    30 substeps per 30 Hz observation, as in ``dice_material_optimize.py``.
    These are the original training equations and defaults on prepared meshes;
    the default meshes differ from the original generated slab and cylinder.
    Material fitting changes only the die. The probe's fixed parameters remain
    included in the reference projection's mean Lame parameters.
    Tool poses use Warp's XYZ+XYZW convention. Their local +Z axis is the
    original end-effector's +X probe axis, for direct native robot IK use.

    ``swept_contact=True`` opts into a forward-only, frictionless CCD extension.
    The complete probe is kinematic in that mode and the penalty force is
    replaced by swept mesh contact response; material equations are unchanged.
    """
    if not np.isfinite(young) or young <= 0.0 or not np.isfinite(poisson) or not 0.0 <= poisson < 0.5:
        raise ValueError("young must be finite and positive and poisson must be in [0, 0.5)")
    if swept_contact and friction != 0.0:
        raise ValueError("Swept probe contact is currently frictionless; set friction=0")
    joints = reference_joint_trajectory(substeps)
    if recording is None:
        name = "recording_reference.json" if reference else "recording_fine.json"
        recording = Path(__file__).parent / "assets" / "pokeflex_local" / name
    data = load_poke_recording(recording)
    if reference and not bool(data.get("reference_mesh", False)):
        raise ValueError("--reference requires a recording prepared with --reference")
    if reference:
        if "probe_vertices" not in data or "probe_tets" not in data:
            raise ValueError("--reference requires the prepared reference probe mesh; prepare the recording again")
        tool_local, tool_tets = data["probe_vertices"], data["probe_tets"]
    else:
        probe = load_mesh(Path(__file__).parent / "assets" / "poke_probe.usda")[0]
        tool_local, tool_tets = probe.vertices, probe.indices
    tool_positions, tool_poses = [], []
    for q in joints:
        position, rotation = reference_tool_frame(q)
        rotation = rotation @ _rotation(1, np.pi / 2)
        tool_positions.append(tool_local @ rotation.T + position)
        quaternion = np.asarray(wp.quat_from_matrix(wp.mat33(*rotation.ravel())), dtype=np.float64)
        tool_poses.append(np.concatenate((position, quaternion)))
    tool_positions, tool_poses = np.asarray(tool_positions), np.asarray(tool_poses)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    _append_mesh(builder, data["vertices"], data["tets"], young, poisson, POKE_PHYSICS["density"])
    soft_count = len(data["vertices"])
    for i in data["pinned"]:
        builder.particle_mass[i] = 0.0
    _append_mesh(
        builder,
        tool_positions[0],
        tool_tets,
        POKE_PHYSICS["probe_young"],
        POKE_PHYSICS["probe_poisson"],
        POKE_PHYSICS["density"],
    )
    # Asset local +Z is the original cylinder's local +X root-to-tip axis.
    tool_root_indices = soft_count + np.flatnonzero(tool_local[:, 2] - tool_local[:, 2].min() < 1.0e-4)
    for i in tool_root_indices:
        builder.particle_mass[i] = 0.0
    if swept_contact:
        for i in range(soft_count, len(builder.particle_q)):
            builder.particle_mass[i] = 0.0
    model = builder.finalize(device=device, requires_grad=True)
    tets = np.asarray(builder.tet_indices, dtype=np.int32)
    oriented = tets[len(data["tets"]) :]
    faces = oriented[:, ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))].reshape(-1, 3)
    _, first, counts = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    tool_faces = faces[first[counts == 1]]
    contact_options = (
        {"swept_contact": (contact_radius, soft_count, data["faces"], tool_faces)}
        if swept_contact
        else {
            "mesh_contact": (
                contact_radius,
                POKE_PHYSICS["contact_kn_left"],
                POKE_PHYSICS["contact_kn_right"],
                POKE_PHYSICS["contact_kf"],
                friction,
                data["faces"],
                tool_faces,
            )
        }
    )
    physics = dict(POKE_PHYSICS)
    if swept_contact:
        # Resolve the constrained equilibrium more tightly than the original
        # penalty demo: its norm-relative threshold allows millimetre errors
        # over this many particles in world coordinates.
        physics.update(
            contact_mode="swept_mesh_extension", probe_prescription="all_nodes_kinematic", adjoint=False, rel_tol=1.0e-7
        )
    solver = SolverDiffPhD(
        model,
        density=POKE_PHYSICS["density"],
        material="neo_hookean",
        max_iterations=POKE_PHYSICS["max_iterations"],
        abs_tol=POKE_PHYSICS["abs_tol"],
        rel_tol=physics["rel_tol"],
        anderson_window=POKE_PHYSICS["anderson_window"],
        **contact_options,
    )
    return {
        "data": data,
        "model": model,
        "solver": solver,
        "soft_count": soft_count,
        "surface_indices": np.unique(data["faces"]),
        "loss_sample_ids": data["target_vertex_indices"][0].copy(),
        "tool_positions": tool_positions,
        "tool_poses": tool_poses,
        "tool_faces": tool_faces,
        "tool_local": tool_local,
        "tool_root_indices": tool_root_indices,
        "joint_positions": joints,
        "rest": np.asarray(builder.particle_q, dtype=np.float64),
        "tets": tets,
        "frame_dt": 1.0 / 30.0,
        "substeps": int(substeps),
        "young": float(young),
        "poisson": float(poisson),
        "friction": float(friction),
        "contact_radius": float(contact_radius),
        "physics": physics,
        "swept_contact": bool(swept_contact),
    }
