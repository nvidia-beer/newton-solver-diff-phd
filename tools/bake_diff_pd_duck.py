"""Bake the DiffPD duck tetrahedra into ``examples/assets/duck.usda`` (development tool).

Reproduces ``DuckEnv3d`` mesh preparation of diff_pd_public (1594aa0):
``tetrahedralize(bob.obj, normalize_input=False, options={"minratio": 1.1})``
from ``python/py_diff_pd/common/tet_mesh.py``: trimesh load with a watertight
assert, STL round trip into pyvista, ``TetGen.make_manifold``, ``tetrahedralize``,
the first four element columns, the reference's sign fix and
``filter_unused_vertices``. The reference pins tetgen 0.5.1 / pyvista 0.26.1;
tetgen keyword defaults of 0.5.1 that differ from the installed version are
passed explicitly, but the bundled TetGen library differs, so the mesh is not
bit-identical to the reference's.

Run with the mesh-tools environment::

    uv run --no-project --with-editable ../../IsaacLab/v300_beta/IsaacLab/third_party/newton-mesh-tools \
        --with pymeshfix python tools/bake_diff_pd_duck.py --source ../../diff_pd_public/asset/mesh/bob.obj
"""

import argparse
import hashlib
import os
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pyvista as pv
import tetgen
import trimesh
from pxr import Sdf, Usd, UsdGeom, Vt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "src" / "newton_solver_diff_phd" / "examples" / "assets" / "duck.usda"

# tetgen 0.5.1 ``TetGen.tetrahedralize`` defaults that differ from later releases.
TETGEN_051_DEFAULTS = {
    "nobisect": True,
    "order": 2,
    "steinerleft": 10000,
    "shellfaceperblock": 4092,
    "opt_max_flip_level": 2,
    "optmaxdihedral": 165.0,
    "quiet": False,
}


def tetrahedralize(path: Path):
    """``tet_mesh.tetrahedralize`` with ``normalize_input=False`` and ``minratio=1.1``."""
    tri_mesh = trimesh.load(path)
    assert tri_mesh.is_watertight
    with tempfile.TemporaryDirectory() as folder:
        stl = os.path.join(folder, ".tmp.stl")
        tri_mesh.export(stl)
        mesh = pv.read(stl)
    tet = tetgen.TetGen(mesh)
    tet.make_manifold()
    result = tet.tetrahedralize(**TETGEN_051_DEFAULTS, minratio=1.1)
    nodes = np.asarray(result[0], dtype=np.float64)
    elements_unsigned = np.asarray(result[1]).astype(int)[:, :4]
    # Reference sign fix: keep e when (v1 - v0) x (v2 - v1) . (v3 - v0) < 0, else swap 1 and 2.
    elements = []
    for e in elements_unsigned:
        v0, v1, v2, v3 = nodes[e]
        if np.cross(v1 - v0, v2 - v1).dot(v3 - v0) < 0:
            elements.append(e)
        else:
            elements.append([e[0], e[2], e[1], e[3]])
    elements = np.asarray(elements, dtype=int)
    # filter_unused_vertices (common/common.py): drop unused nodes, keep order.
    used = np.zeros(len(nodes), dtype=bool)
    used[elements.ravel()] = True
    remap = np.full(len(nodes), -1)
    remap[used] = np.arange(np.count_nonzero(used))
    return nodes[used], remap[elements]


def surface_faces(vertices: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """Outward boundary triangles, as ``tet2obj`` collects them with ``fix_tet_faces``."""
    faces = {}
    for element in elements:
        v0, v1, v2, v3 = vertices[element]
        if np.cross(v1 - v0, v2 - v1).dot(v3 - v0) < 0:
            local = ((0, 1, 2), (2, 1, 3), (1, 0, 3), (0, 2, 3))
        else:
            local = ((1, 0, 2), (1, 2, 3), (0, 1, 3), (2, 0, 3))
        for face in local:
            indices = [int(element[i]) for i in face]
            key = tuple(sorted(indices))
            if key in faces:
                del faces[key]
            else:
                faces[key] = indices
    return np.asarray(list(faces.values()), dtype=int)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="diff_pd_public asset/mesh/bob.obj")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    vertices, elements = tetrahedralize(args.source)
    faces = surface_faces(vertices, elements)
    edges = vertices[elements[:, 1:]] - vertices[elements[:, :1]]
    determinants = np.linalg.det(edges)
    assert np.all(determinants < 0), "the reference sign fix leaves negative determinants"
    edge_count = {}
    for face in faces:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_count[(min(a, b), max(a, b))] = edge_count.get((min(a, b), max(a, b)), 0) + 1
    assert set(edge_count.values()) == {2}, "the boundary surface is not a closed manifold"

    versions = {name: version(name) for name in ("tetgen", "pyvista", "trimesh", "pymeshfix")}
    stage = Usd.Stage.CreateInMemory()
    stage.GetRootLayer().documentation = (
        "Geometry-only tetrahedralization of diff_pd_public asset/mesh/bob.obj (Keenan Crane's Bob, CC0 1.0) "
        "by its DuckEnv3d recipe: tetgen minratio 1.1, unnormalized, reference element winding "
        "(negative determinants). Baked by tools/bake_diff_pd_duck.py with "
        + ", ".join(f"{name} {value}" for name, value in versions.items())
        + "; the reference used tetgen 0.5.1 and pyvista 0.26.1. Material and placement are supplied by the example."
    )
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    mesh = UsdGeom.TetMesh.Define(stage, "/Duck")
    stage.SetDefaultPrim(mesh.GetPrim())
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
    mesh.GetPrim().CreateAttribute("newton:referencePoints", Sdf.ValueTypeNames.Double3Array, custom=True).Set(
        Vt.Vec3dArray.FromNumpy(vertices)
    )
    mesh.CreateTetVertexIndicesAttr(Vt.Vec4iArray.FromNumpy(elements.astype(np.int32)))
    mesh.CreateSurfaceFaceVertexIndicesAttr(Vt.Vec3iArray.FromNumpy(faces.astype(np.int32)))
    mesh.GetPrim().CreateAttribute("newton:sourceSha256", Sdf.ValueTypeNames.String, custom=True).Set(
        hashlib.sha256(args.source.read_bytes()).hexdigest()
    )
    args.output.write_text(stage.GetRootLayer().ExportToString(), encoding="utf-8")
    volumes = -determinants / 6.0
    print(
        f"{args.output}: {len(vertices)} vertices, {len(elements)} tetrahedra, {len(faces)} surface triangles, "
        f"volume {volumes.sum():.6f} m^3, mean {volumes.mean():.6e} m^3; {versions}"
    )


if __name__ == "__main__":
    main()
