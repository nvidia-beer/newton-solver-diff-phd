# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Load the baked bear volumes and geometric parent-cell reference data."""

from pathlib import Path

import numpy as np

from .usd import load_mesh

_FACES = np.array(((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)))
BEAR_ASSET = Path(__file__).resolve().parents[2] / "examples" / "assets" / "bear.usda"


def _boundary(tets):
    faces = tets[:, _FACES].reshape(-1, 3)
    _, first, counts = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("bear resolution produces a nonmanifold tetrahedral face")
    exterior = first[counts == 1]
    boundary = faces[exterior]
    edges = np.sort(boundary[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1)
    if np.any(np.unique(edges, axis=0, return_counts=True)[1] != 2):
        raise ValueError("bear resolution produces a nonmanifold boundary; choose another resolution")
    # Edge incidence alone cannot detect two surface sheets meeting at one node.
    links = {}
    for a, b, c in boundary:
        for node, edge in ((a, (b, c)), (b, (c, a)), (c, (a, b))):
            links.setdefault(node, []).append(edge)
    for edges in links.values():
        neighbors = {}
        for a, b in edges:
            neighbors.setdefault(a, set()).add(b)
            neighbors.setdefault(b, set()).add(a)
        pending = [next(iter(neighbors))]
        seen = set(pending)
        while pending:
            for node in neighbors[pending.pop()] - seen:
                seen.add(node)
                pending.append(node)
        if len(seen) != len(neighbors):
            raise ValueError("bear resolution produces a nonmanifold vertex; choose another resolution")
    return boundary.astype(np.int32), exterior // 4


def load_bear_geometry(resolution=10):
    """Load the fixed reference volumes without generating tetrahedra at runtime.

    The coarse cell centers preserve the source cell-to-tet ordering for
    anatomical classification. The original volume supplies source geometry
    for mass integration; this function imports no physical parameters.
    """
    if isinstance(resolution, (bool, np.bool_)) or not isinstance(resolution, (int, np.integer)) or resolution != 10:
        raise ValueError("The bundled bear geometry supports integer resolution 10 only")
    meshes = {mesh.name: mesh for mesh in load_mesh(BEAR_ASSET)}
    coarse, source = meshes["/Bear/Coarse"], meshes["/Bear/Source"]
    from pxr import Usd  # noqa: PLC0415 -- optional example dependency

    stage = Usd.Stage.Open(str(BEAR_ASSET))
    centers = np.asarray(stage.GetPrimAtPath("/Bear/Coarse").GetAttribute("newton:cellCenters").Get(), dtype=np.float64)
    if centers.shape != (len(coarse.indices) // 6, 3) or not np.isfinite(centers).all():
        raise ValueError("Bear cell centers must match the six tetrahedra per source cell")
    corners = source.vertices[source.indices]
    source_volume = float(np.abs(np.linalg.det((corners[:, 1:] - corners[:, :1]).transpose(0, 2, 1))).sum() / 6.0)
    faces, owners = _boundary(coarse.indices)
    return (
        coarse.vertices,
        coarse.indices,
        faces,
        owners,
        centers,
        source_volume,
        source.vertices.min(axis=0),
        np.ptp(source.vertices, axis=0),
    )
