# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Read static USD volume geometry without importing material or solver settings."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Mesh:
    """A volume mesh in world-space metres with Z up: Tet4 or binary-order Hex8."""

    name: str
    vertices: np.ndarray
    indices: np.ndarray


def load_mesh(path: str | Path) -> list[Mesh]:
    """Load static Tet4 or Hex8 geometry, applying transforms, units and up axis.

    Hex8 geometry is a USD Mesh with a flattened ``newton:hexVertexIndices``
    integer attribute, ordered 000,001,010,011,100,101,110,111 per cell.
    Only points and volume connectivity are imported. Materials, density,
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
        is_hex = prim.GetTypeName() == "Mesh" and bool(prim.GetAttribute("newton:hexVertexIndices"))
        if prim.GetTypeName() != "TetMesh" and not is_hex:
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
        attribute = "newton:hexVertexIndices" if is_hex else "tetVertexIndices"
        width = 8 if is_hex else 4
        indices = np.asarray(prim.GetAttribute(attribute).Get())
        if is_hex and indices.ndim == 1 and indices.size % width == 0:
            indices = indices.reshape(-1, width)
        if (
            indices.ndim != 2
            or indices.shape[1] != width
            or not len(indices)
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
            or np.any(indices >= len(vertices))
        ):
            raise ValueError(f"{name}: {attribute} must contain {width} valid indices per element")
        indices = indices.astype(np.int32, copy=True)
        if is_hex:
            if np.any(np.diff(np.sort(indices, axis=1), axis=1) == 0):
                raise ValueError(f"{name}: a hexahedron has repeated vertices")
            # Quadrature and positive rest Jacobians are checked by the solver.
            meshes.append(Mesh(name, vertices, indices))
            continue
        edges = vertices[indices[:, 1:]] - vertices[indices[:, :1]]
        determinant = np.linalg.det(edges)
        scale = np.prod(np.linalg.norm(edges, axis=2), axis=1)
        if np.any(~np.isfinite(determinant)) or np.any(np.abs(determinant) <= 1.0e-12 * scale):
            raise ValueError(f"{name}: a tetrahedron is degenerate")
        inverted = determinant < 0
        indices[inverted] = indices[inverted][:, [0, 2, 1, 3]]
        meshes.append(Mesh(name, vertices, indices))
    if not meshes:
        raise ValueError(f"No Tet4 or Hex8 volume geometry found in {path}")
    return meshes
