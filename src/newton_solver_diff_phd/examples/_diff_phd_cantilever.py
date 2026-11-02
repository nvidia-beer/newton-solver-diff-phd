# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Dimensions and stiffness regions follow DiffPhD (d319435), CantileverEnv3d.
# MIT License; see THIRD_PARTY_NOTICES.md.

"""Original Hex8 cantilever, materials, fibres, vertex energies and axial loading.

The default reproduces the released cantilever_3d.py forward problem. The
optional twist motion uses this same Hex environment with an explicit 30-degree
release; the paper describes this motion but does not release its initial state.
"""

from pathlib import Path

import newton
import numpy as np
import warp as wp

from newton_solver_diff_phd import SolverDiffPhD, load_mesh

ASSET = Path(__file__).parent / "assets" / "cantilever.usda"
DT = 0.01
TWIST_DT = 0.001
FRAMES = 25
TWIST_FRAMES = 100
DENSITY = 1000.0
YOUNG = 1.0e6
POISSON = 0.45
TWIST = np.pi / 6.0


def load_cantilever(asset=ASSET):
    """Read the baked volume, surface and geometric subsets."""
    from pxr import Usd

    asset = Path(asset)
    meshes = {mesh.name: mesh for mesh in load_mesh(asset)}
    if "/Cantilever" not in meshes:
        raise ValueError("Cantilever USD must contain /Cantilever Hex8 volume")
    mesh = meshes["/Cantilever"]
    if mesh.indices.shape != (500, 8):
        raise ValueError("Cantilever must contain the original 500 Hex8 cells")
    stage = Usd.Stage.Open(str(asset))
    prim = stage.GetPrimAtPath("/Cantilever")

    def indices(name, limit):
        attribute = prim.GetAttribute(name)
        values = np.asarray(attribute.Get() if attribute else None)
        if not np.issubdtype(values.dtype, np.integer) or np.any(values < 0) or np.any(values >= limit):
            raise ValueError(f"Cantilever USD has invalid {name}")
        return values.astype(np.int32)

    faces = indices("faceVertexIndices", len(mesh.vertices)).reshape(-1, 3)
    owners = indices("diffphd:surfaceElementIndices", len(mesh.indices))
    root = indices("diffphd:rootVertexIndices", len(mesh.vertices))
    cells = indices("diffphd:xCellIndices", 20)
    if faces.ndim != 2 or faces.shape[1] != 3 or owners.shape != (len(faces),):
        raise ValueError("Cantilever surface and owner arrays have inconsistent shapes")
    if cells.shape != (len(mesh.indices),) or root.ndim != 1 or not len(root):
        raise ValueError("Cantilever geometric subsets have inconsistent shapes")
    return mesh, faces, owners, root, cells


def twisted_positions(rest, angle=TWIST):
    """Rotate cross-sections about their center, from zero at the root to angle at the tip."""
    if not np.isfinite(angle):
        raise ValueError("The release angle must be finite")
    rest = np.asarray(rest, dtype=np.float64)
    low, high = rest.min(axis=0), rest.max(axis=0)
    theta = float(angle) * (rest[:, 0] - low[0]) / (high[0] - low[0])
    center = 0.5 * (low[1:] + high[1:])
    offset = rest[:, 1:] - center
    q = rest.copy()
    q[:, 1] = center[0] + np.cos(theta) * offset[:, 0] - np.sin(theta) * offset[:, 1]
    q[:, 2] = center[1] + np.sin(theta) * offset[:, 0] + np.cos(theta) * offset[:, 1]
    q[rest[:, 0] == low[0]] = rest[rest[:, 0] == low[0]]
    return q


class CantileverSimulation:
    """Two-state simulation of the released Hex8 beam with its original energies."""

    def __init__(self, *, asset=ASSET, device=None, capture=True, motion="axial"):
        if motion not in ("axial", "twist"):
            raise ValueError("Cantilever motion must be 'axial' or 'twist'")
        self.motion = motion
        self.dt = DT if motion == "axial" else TWIST_DT
        self.frames = FRAMES if motion == "axial" else TWIST_FRAMES
        mesh, self.faces, owners, self.pinned, cells = load_cantilever(asset)
        self.vertices, self.hexes = mesh.vertices, mesh.indices
        # Original CantileverEnv3d refinement=5: x cells 7 through 12 are 10x stiffer.
        self.materials = ((cells > 20 // 3) & (cells <= (20 // 3) * 2)).astype(np.int32)
        self.face_materials = self.materials[owners]
        young = YOUNG * np.where(self.materials == 1, 10.0, 1.0)
        mu = young / (2.0 * (1.0 + POISSON))
        lam = young * POISSON / ((1.0 + POISSON) * (1.0 - 2.0 * POISSON))
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        # Newton supplies particle states. Hex8 connectivity and integration
        # remain owned by SolverDiffPhD; no tetrahedral surrogate is created.
        builder.add_particles(
            pos=self.vertices.tolist(),
            vel=np.zeros_like(self.vertices).tolist(),
            mass=np.full(len(self.vertices), DENSITY * 0.016**3).tolist(),
            radius=np.full(len(self.vertices), 0.0005).tolist(),
        )
        for node in self.pinned:
            builder.particle_mass[int(node)] = 0.0
        self.model = builder.finalize(device=device, requires_grad=False)
        muscle_stiffness = np.zeros(len(self.hexes))
        muscle_stiffness[np.arange(20) * 25] = 1.0e5
        directions = np.tile([1.0, 0.0, 0.0], (len(self.hexes), 1))
        self.solver = SolverDiffPhD(
            self.model,
            hex_mesh=mesh,
            hex_materials=np.column_stack([mu, lam]),
            density=DENSITY,
            material="pd",
            muscle_directions=directions,
            muscle_stiffness=muscle_stiffness,
            muscle_targets=np.zeros(len(self.hexes)),
            planar_energies=(np.array([720, 750], dtype=np.int32), 100.0, (0.0, 0.0, 1.0, 0.1)),
            max_iterations=5000,
            abs_tol=1.0e-9,
            rel_tol=1.0e-4 if motion == "axial" else 1.0e-6,
            anderson_window=5,
        )
        self.q0 = self.vertices.copy() if motion == "axial" else twisted_positions(self.vertices)
        self.tip = np.flatnonzero(self.vertices[:, 0] == self.vertices[:, 0].max())
        self.forces = np.zeros_like(self.vertices, dtype=np.float32)
        self.states = [self.model.state(), self.model.state()]
        self.capture = bool(capture and self.model.device.is_cuda and wp.is_conditional_graph_supported())
        self.graphs = {}
        self.prepared = False
        self.frame = 0
        self.reset()

    def set_motion(self, motion):
        """Restart in the selected mode, rebuilding graphs for its tolerance."""
        if motion not in ("axial", "twist"):
            raise ValueError("Cantilever motion must be 'axial' or 'twist'")
        if motion != self.motion:
            self.motion = motion
            self.dt = DT if motion == "axial" else TWIST_DT
            self.frames = FRAMES if motion == "axial" else TWIST_FRAMES
            self.q0 = self.vertices.copy() if motion == "axial" else twisted_positions(self.vertices)
            self.solver.rel_tol = 1.0e-4 if motion == "axial" else 1.0e-6
            # Convergence constants are captured into the device loop. Rebuild
            # through the public invalidation API and recapture the outer step.
            self.solver.notify_model_changed(0)
            self.graphs.clear()
            self.prepared = False
        self.reset()

    def reset(self):
        """Restart the chosen source load or explicit twist with zero velocity."""
        self.states[0].particle_q.assign(self.q0)
        for state in self.states:
            state.particle_qd.zero_()
            state.particle_f.zero_()
        self.forces.fill(0.0)
        self.solver.status.zero_()
        self.frame = 0

    def advance(self):
        """Advance one actual physics step, then hold after the complete episode."""
        if self.frame >= self.frames:
            return self.states[0]
        if self.motion == "axial":
            # Original script: step t has +X force 80*t/25 N at each tip node.
            self.forces[self.tip, 0] = 80.0 * self.frame / FRAMES
            self.states[0].particle_f.assign(self.forces)
        if self.capture and self.prepared:
            key = self.states[0].particle_q.ptr
            if key not in self.graphs:
                with wp.ScopedCapture(device=self.model.device) as capture:
                    self.solver.step(self.states[0], self.states[1], None, None, self.dt)
                self.graphs[key] = capture.graph
            wp.capture_launch(self.graphs[key])
        else:
            self.solver.step(self.states[0], self.states[1], None, None, self.dt)
            self.prepared = True
        self.solver.check_status()
        self.states.reverse()
        self.frame += 1
        return self.states[0]
