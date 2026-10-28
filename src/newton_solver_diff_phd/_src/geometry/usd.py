# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Read static USD tetrahedra without importing material or solver settings."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Mesh:
    """One TetMesh in world-space metres with Z up and positive Tet4 winding."""

    name: str
    vertices: np.ndarray
    indices: np.ndarray


def load_mesh(path: str | Path) -> list[Mesh]:
    """Load static TetMesh geometry, applying USD transforms, units and up axis.

    Only points and tetrahedral connectivity are imported. Materials, density,
    contacts and all other physics attributes remain the caller's responsibility.
    OpenUSD is imported lazily; install this package's ``examples`` extra to use it.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        from pxr import Usd, UsdGeom
    except ModuleNotFoundError as error:
        if error.name != "pxr":
            raise
        raise ImportError("USD loading requires newton-solver-diff-phd[examples]") from error

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Cannot open USD stage {path}")
    metres = float(UsdGeom.GetStageMetersPerUnit(stage))
    up = str(UsdGeom.GetStageUpAxis(stage))
    if not np.isfinite(metres) or metres <= 0 or up not in ("Y", "Z"):
        raise ValueError("The USD stage must have positive units and Y or Z up")
    transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    meshes = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "TetMesh":
            continue
        name = str(prim.GetPath())
        reference_points = prim.GetAttribute("newton:referencePoints")
        vertices = np.asarray(
            reference_points.Get() if reference_points else prim.GetAttribute("points").Get(), dtype=np.float64
        )
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
            raise ValueError(f"{name}: points must contain three coordinates per vertex")
        transform = np.asarray(transforms.GetLocalToWorldTransform(prim), dtype=np.float64)
        vertices = (vertices @ transform[:3, :3] + transform[3, :3]) * metres
        if up == "Y":
            vertices = vertices[:, [0, 2, 1]] * [1, -1, 1]
        if not np.isfinite(vertices).all():
            raise ValueError(f"{name}: points must be finite")
        indices = np.asarray(prim.GetAttribute("tetVertexIndices").Get())
        if (
            indices.ndim != 2
            or indices.shape[1] != 4
            or not len(indices)
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
            or np.any(indices >= len(vertices))
        ):
            raise ValueError(f"{name}: tetVertexIndices must contain four valid indices per element")
        indices = indices.astype(np.int32, copy=True)
        edges = vertices[indices[:, 1:]] - vertices[indices[:, :1]]
        determinant = np.linalg.det(edges)
        scale = np.prod(np.linalg.norm(edges, axis=2), axis=1)
        if np.any(~np.isfinite(determinant)) or np.any(np.abs(determinant) <= 1.0e-12 * scale):
            raise ValueError(f"{name}: a tetrahedron is degenerate")
        inverted = determinant < 0
        indices[inverted] = indices[inverted][:, [0, 2, 1, 3]]
        meshes.append(Mesh(name, vertices, indices))
    if not meshes:
        raise ValueError(f"No TetMesh volume geometry found in {path}")
    return meshes
