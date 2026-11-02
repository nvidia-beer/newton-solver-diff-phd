# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bake the original DiffPhD Hex8 cantilever grid into a geometry-only USD.

The original CantileverEnv3d uses a 20 x 5 x 5 Hex8 grid at refinement 5,
with 0.016 m spacing and binary (x, y, z) local corner ordering.
The geometry is generated offline; no material or solver data is baked.
"""

import argparse
from pathlib import Path

import numpy as np
from pxr import Sdf, Usd, UsdGeom, Vt

OUTPUT = (
    Path(__file__).resolve().parents[1] / "src" / "newton_solver_diff_phd" / "examples" / "assets" / "cantilever.usda"
)


def geometry():
    """Return original Hex8 connectivity, outward surface and geometric subsets."""
    nx, ny, nz = 20, 5, 5
    vertices = np.array([(i, j, k) for i in range(nx + 1) for j in range(ny + 1) for k in range(nz + 1)]) * 0.016

    def node(i, j, k):
        return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

    hexes, cells = [], []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                hexes.append([node(i + a, j + b, k + c) for a in (0, 1) for b in (0, 1) for c in (0, 1)])
                cells.append(i)
    hexes = np.asarray(hexes, dtype=np.int32)
    cells = np.asarray(cells, dtype=np.int32)
    # Outward quadrilateral winding in the reference binary corner convention.
    all_faces = hexes[:, ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))].reshape(
        -1, 4
    )
    _, first, counts = np.unique(np.sort(all_faces, axis=1), axis=0, return_index=True, return_counts=True)
    if counts.max() != 2:
        raise ValueError("The cantilever must have manifold hexahedral connectivity")
    boundary = first[counts == 1]
    quads, owners = all_faces[boundary], boundary // 6
    # Triangles are for display only; the solver consumes the 500 intact Hex8 cells.
    faces = quads[:, ((0, 1, 2), (0, 2, 3))].reshape(-1, 3)
    return vertices, hexes, faces, np.repeat(owners, 2), np.flatnonzero(vertices[:, 0] == 0.0), cells


def bake(path):
    """Write only volume/surface geometry and geometric subset indices."""
    vertices, hexes, faces, owners, root, cells = geometry()
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.GetRootLayer().documentation = (
        "Geometry-only original Hex8 grid of DiffPhD CantileverEnv3d refinement=5: "
        "0.32 x 0.08 x 0.08 m, 20 x 5 x 5 cells. Binary x/y/z corner ordering. "
        "Surface triangles are display geometry only. Baked by tools/bake_cantilever.py. "
        "Materials, quadrature and loading are supplied at runtime."
    )
    mesh = UsdGeom.Mesh.Define(stage, "/Cantilever")
    prim = mesh.GetPrim()
    stage.SetDefaultPrim(prim)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
    prim.CreateAttribute("newton:referencePoints", Sdf.ValueTypeNames.Double3Array, custom=True).Set(
        Vt.Vec3dArray.FromNumpy(vertices)
    )
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.ravel()))
    prim.CreateAttribute("newton:hexVertexIndices", Sdf.ValueTypeNames.IntArray, custom=True).Set(
        Vt.IntArray.FromNumpy(hexes.ravel())
    )
    for name, values in (
        ("diffphd:surfaceElementIndices", owners),
        ("diffphd:rootVertexIndices", root),
        ("diffphd:xCellIndices", cells),
    ):
        prim.CreateAttribute(name, Sdf.ValueTypeNames.IntArray, custom=True).Set(
            Vt.IntArray.FromNumpy(values.astype(np.int32))
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stage.GetRootLayer().ExportToString(), encoding="utf-8")
    print(f"{path}: {len(vertices)} nodes, {len(hexes)} hexahedra, {len(faces)} boundary triangles")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    bake(parser.parse_args().output)
