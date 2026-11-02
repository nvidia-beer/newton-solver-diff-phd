# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Scene translated from DiffPhD (d319435), python/example/bunny.py and
# python/py_diff_pd/env/bunny_env.py. MIT License; see THIRD_PARTY_NOTICES.md.

"""Released DiffPhD heterogeneous bunny landing, using an offline USD mesh."""

from pathlib import Path

import newton
import numpy as np
import warp as wp

from newton_solver_diff_phd import SolverDiffPhD, load_mesh

ASSET = Path(__file__).parent / "assets" / "bunny_reference_local" / "bunny_reference.usdc"
DT = 0.001
FRAMES = 100
DENSITY = 1000.0
YOUNGS = 1.0e6
POISSON = 0.49
TARGET = np.array([0.15, 0.15, 0.15])
LOWER = np.array([-np.pi / 3, -np.pi / 3, -np.pi / 3, -0.03, -0.03, 0.19, -2.0, -2.0, -8.0])
UPPER = np.array([np.pi / 3, np.pi / 3, np.pi / 3, 0.03, 0.03, 0.21, 2.0, 4.0, -4.0])


def initial_parameters():
    """The original seed-211 launch; RPY, initial center, uniform velocity."""
    return np.random.default_rng(211).uniform(LOWER, UPPER)


def initial_state(rest, parameters):
    """Apply the released Rz(yaw) Ry(pitch) Rx(roll) launch transform."""
    parameters = np.asarray(parameters, dtype=np.float64)
    if parameters.shape != (9,) or not np.isfinite(parameters).all():
        raise ValueError("The bunny launch needs nine finite RPY/position/velocity values")
    roll, pitch, yaw = parameters[:3]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    q = (rest - rest.mean(axis=0)) @ (rz @ ry @ rx).T + parameters[3:6]
    return q, np.broadcast_to(parameters[6:], q.shape).copy()


def landing_loss(q):
    """Original unweighted center objective, normalized by the 0.1 m size."""
    difference = np.asarray(q, dtype=np.float64).mean(axis=0) - TARGET
    return 0.5 * float(difference @ difference) / 0.1**2


def load_bunny(asset=ASSET):
    """Read volume, outward surface, owners and original geometric subsets."""
    from pxr import Usd

    asset = Path(asset)
    if not asset.is_file():
        raise FileNotFoundError(
            f"Bunny asset not prepared: {asset}. Run tools/prepare_bunny_reference.py "
            "with --reference-root pointing to your DiffPhD checkout; "
            "see docs/usage.md for the offline dependencies and asset terms."
        )
    meshes = {mesh.name: mesh for mesh in load_mesh(asset)}
    if "/Bunny" not in meshes:
        raise ValueError("Bunny USD must contain /Bunny TetMesh")
    mesh = meshes["/Bunny"]
    stage = Usd.Stage.Open(str(asset))
    prim = stage.GetPrimAtPath("/Bunny")

    def indices(name, limit):
        attr = prim.GetAttribute(name)
        values = np.asarray(attr.Get() if attr else None)
        if not np.issubdtype(values.dtype, np.integer) or np.any(values < 0) or np.any(values >= limit):
            raise ValueError(f"Bunny USD has invalid {name}")
        return values.astype(np.int32)

    faces = indices("surfaceFaceVertexIndices", len(mesh.vertices))
    owners = indices("diffphd:surfaceElementIndices", len(mesh.indices))
    contact = indices("diffphd:contactVertexIndices", len(mesh.vertices))
    stiff = indices("diffphd:stiffElementIndices", len(mesh.indices))
    soft = indices("diffphd:softElementIndices", len(mesh.indices))
    if faces.ndim != 2 or faces.shape[1] != 3 or owners.shape != (len(faces),):
        raise ValueError("Bunny USD surface faces and their owners have inconsistent shapes")
    for group in (contact, stiff, soft):
        if group.ndim != 1 or not len(group) or len(np.unique(group)) != len(group):
            raise ValueError("Bunny USD geometric subsets must be nonempty unique index vectors")
    if not np.array_equal(np.sort(np.concatenate([stiff, soft])), np.arange(len(mesh.indices))):
        raise ValueError("Bunny material regions must partition all tetrahedra")
    return mesh, faces, owners, contact, stiff, soft


class BunnySimulation:
    """Forward reproduction of the paper's seed-211 initial-state task.

    This does not train a network or load a checkpoint. The original grouped
    material adjoint and contact-dependent launch gradients are not implemented
    here, so there is deliberately no optimization interface. Each complete
    episode is the released 100 steps at dt=0.001, including impact.
    """

    def __init__(self, *, asset=ASSET, device=None, parameters=None, capture=True):
        mesh, self.faces, owners, self.contact_vertices, stiff, soft = load_bunny(asset)
        self.vertices, self.tets = mesh.vertices, mesh.indices
        self.parameters = (
            initial_parameters() if parameters is None else np.asarray(parameters, dtype=np.float64).copy()
        )
        self.q0, self.v0 = initial_state(self.vertices, self.parameters)
        self.materials = np.zeros(len(self.tets), dtype=np.int32)
        self.materials[stiff] = 1
        self.face_materials = self.materials[owners]
        youngs = YOUNGS * np.where(self.materials == 1, 10.0, 0.1)
        mu = youngs / (2 * (1 + POISSON))
        lam = youngs * POISSON / ((1 + POISSON) * (1 - 2 * POISSON))
        builder = newton.ModelBuilder()
        builder.add_soft_mesh(
            pos=wp.vec3(),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(),
            vertices=self.vertices.astype(np.float32),
            indices=self.tets.ravel(),
            density=DENSITY,
            k_mu=1.0,
            k_lambda=1.0,
            k_damp=0.0,
        )
        self.model = builder.finalize(device=device, requires_grad=False)
        self.model.tet_materials.assign(np.column_stack([mu, lam, np.zeros(len(mu))]).astype(np.float32))
        self.solver = SolverDiffPhD(
            self.model,
            density=DENSITY,
            material="pd",
            frictional_boundary=("planar", (0.0, 0.0, 1.0, 0.0)),
            frictional_vertices=self.contact_vertices,
            friction=0.1,
            restitution=0.0,
            max_iterations=10000,
            abs_tol=1.0e-9,
            rel_tol=1.0e-6,
            anderson_window=5,
        )
        self.states = [self.model.state(), self.model.state()]
        self.frame = 0
        self.capture = bool(capture and self.model.device.is_cuda and wp.is_conditional_graph_supported())
        self.graphs = {}
        self.prepared = False
        self.reset()

    def reset(self):
        """Restart the original launch, including the contact warm start."""
        self.states[0].particle_q.assign(self.q0)
        self.states[0].particle_qd.assign(self.v0)
        for state in self.states:
            state.particle_f.zero_()
        self.solver.contact_multipliers.zero_()
        self.solver.status.zero_()
        self.frame = 0

    def advance(self):
        """Advance one real physics step; hold the final state after step 100."""
        if self.frame >= FRAMES:
            return self.states[0]
        if self.capture and self.prepared:
            key = self.states[0].particle_q.ptr
            if key not in self.graphs:
                with wp.ScopedCapture(device=self.model.device) as capture:
                    self.solver.step(self.states[0], self.states[1], None, None, DT)
                self.graphs[key] = capture.graph
            wp.capture_launch(self.graphs[key])
        else:
            self.solver.step(self.states[0], self.states[1], None, None, DT)
            self.prepared = True
        self.solver.check_status()
        self.states.reverse()
        self.frame += 1
        return self.states[0]
