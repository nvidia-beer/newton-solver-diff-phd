# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prescribed UR5e kinematics shared by the DiffPhD interaction examples."""

from pathlib import Path

import newton
import newton.ik as ik
import numpy as np
import warp as wp


def _unit_rotation(rotation):
    value = np.asarray(rotation, dtype=np.float64)
    if value.shape != (4,) or not np.isfinite(value).all() or np.linalg.norm(value) < 1.0e-12:
        raise ValueError("Tool rotation must be a finite nonzero quaternion in (x, y, z, w) order")
    return (value / np.linalg.norm(value)).astype(np.float32)


@wp.kernel
def _attachment_pose(
    body_q: wp.array[wp.transform],
    body: int,
    attachment: wp.transform,
    tool_pose: wp.array[wp.transform],
):
    tool_pose[0] = body_q[body] * attachment


class UR5Kinematics:
    """UR5e arm with a prescribed tool frame.

    ``tool_pose`` contains the actual FK attachment transform on the model's
    device. By default its local +Z axis points down.
    The arm is kinematic and does not receive forces from the deformable body.

    Args:
        initial_position: Initial attachment position [m] in world coordinates.
        initial_rotation: Initial tool quaternion (x, y, z, w), or downward-facing.
        base_height: Robot base and supporting pedestal height [m].
        asset_path: Local directory containing the Menagerie UR5e XML and meshes.
        device: Warp device for the robot model, IK, and attachment transform.
    """

    def __init__(
        self, initial_position=(0.45, 0.0, 0.20), initial_rotation=None, base_height=0.0, device=None, *, asset_path
    ):
        if not np.isfinite(base_height) or base_height < 0.0:
            raise ValueError("UR5e base_height must be finite and nonnegative")
        rotation = _unit_rotation((1.0, 0.0, 0.0, 0.0) if initial_rotation is None else initial_rotation)
        asset = Path(asset_path)
        if not (asset / "ur5e.xml").is_file():
            raise FileNotFoundError(asset / "ur5e.xml")
        builder = newton.ModelBuilder()
        builder.add_mjcf(
            str(asset / "ur5e.xml"),
            xform=wp.transform(wp.vec3(0.0, 0.0, base_height)),
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
        )
        site = next(i for i, label in enumerate(builder.shape_label) if label.rsplit("/", 1)[-1] == "attachment_site")
        self.tool_body = builder.shape_body[site]
        self.tool_local = builder.shape_transform[site]
        # Menagerie's home keyframe starts away from the straight-arm singularity.
        builder.joint_q[:] = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
        if base_height > 0.0:
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.5 * base_height)),
                hx=0.12,
                hy=0.12,
                hz=0.5 * base_height,
                color=(0.25, 0.27, 0.29),
                label="robot_pedestal",
            )
        builder.add_ground_plane()
        self.model = builder.finalize(device=device)
        self.state = self.model.state()
        self.tool_pose = wp.empty(1, dtype=wp.transform, device=self.model.device)
        self.joint_q = self.model.joint_q.reshape((1, self.model.joint_coord_count))

        self._position = ik.IKObjectivePosition(
            link_index=self.tool_body,
            link_offset=wp.transform_get_translation(self.tool_local),
            target_positions=wp.array([initial_position], dtype=wp.vec3, device=self.model.device),
        )
        self._rotation = ik.IKObjectiveRotation(
            link_index=self.tool_body,
            link_offset_rotation=wp.transform_get_rotation(self.tool_local),
            target_rotations=wp.array([rotation], dtype=wp.vec4, device=self.model.device),
        )
        limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=10.0,
        )
        self._ik = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[self._position, self._rotation, limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self._graph = None
        self.update(initial_position)
        if self.model.device.is_cuda:
            with wp.ScopedCapture(device=self.model.device) as capture:
                self._solve()
            self._graph = capture.graph

    def update(self, target_position, target_rotation=None):
        """Update the FK tool pose [m, quaternion], retaining orientation if omitted."""
        target = np.asarray(target_position, dtype=np.float32)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("UR5e target_position must contain three finite coordinates")
        rotation = None if target_rotation is None else _unit_rotation(target_rotation)
        self._position.set_target_position(0, wp.vec3(*target))
        if rotation is not None:
            self._rotation.set_target_rotation(0, wp.vec4(*rotation))
        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            self._solve()

    def _solve(self):
        self._ik.step(self.joint_q, self.joint_q, iterations=24)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        wp.launch(
            _attachment_pose,
            dim=1,
            inputs=[self.state.body_q, self.tool_body, self.tool_local],
            outputs=[self.tool_pose],
            device=self.model.device,
        )
