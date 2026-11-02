# SPDX-FileCopyrightText: Copyright (c) 2021 MIT Graphics Group
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: MIT AND Apache-2.0
"""Bake the released DiffPhD bunny geometry for local research use.

The geometry operations reproduce ``common.tet_mesh.tetrahedralize`` and
``BunnyEnv(mesh_type="tet")``. Mesher defaults come from the installed TetGen;
their versions are recorded, so this is not a claim of bitwise identity to a
historical TetGen release. No runtime meshing dependencies are added.

Requires numpy, trimesh, pyvista, tetgen, pymeshfix and OpenUSD in an offline
preparation environment. The Stanford bunny has separate research-use terms;
the default generated directory is excluded from redistribution.
"""

import argparse
import hashlib
import json
import subprocess
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "src/newton_solver_diff_phd/examples/assets/bunny_reference_local"
SOURCE_MESH = Path("asset/mesh/bunny_watertight_simplified2.obj")
SOURCE_FILES = (
    SOURCE_MESH,
    Path("python/py_diff_pd/common/tet_mesh.py"),
    Path("python/py_diff_pd/env/bunny_env.py"),
    Path("python/example/bunny.py"),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tetrahedralize(path):
    """Keep the reference normalization, winding, filtering and node order."""
    import pyvista as pv
    import tetgen
    import trimesh

    surface = trimesh.load(path)
    if not surface.is_watertight:
        raise ValueError("The source bunny surface is not watertight")
    surface.vertices -= np.min(surface.vertices, axis=0)
    surface.vertices /= np.max(np.asarray(surface.bounding_box.extents, dtype=np.float64))
    with tempfile.TemporaryDirectory(prefix="diffphd-bunny-") as folder:
        stl = Path(folder) / ".tmp.stl"
        surface.export(stl)
        volume = tetgen.TetGen(pv.read(stl))
    volume.make_manifold()
    result = volume.tetrahedralize()
    vertices = np.asarray(result[0], dtype=np.float64)
    raw = np.asarray(result[1]).astype(int)[:, :4]
    elements = []
    for element in raw:
        v0, v1, v2, v3 = vertices[element]
        if np.cross(v1 - v0, v2 - v1).dot(v3 - v0) < 0:
            elements.append(element)
        else:
            elements.append([element[0], element[2], element[1], element[3]])
    elements = np.asarray(elements, dtype=np.int32)
    used = np.zeros(len(vertices), dtype=bool)
    used[elements.ravel()] = True
    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[used] = np.arange(np.count_nonzero(used))
    return vertices[used], remap[elements]


def contact_vertices(vertices, elements):
    """Reference ``get_contact_vertex`` with the bunny's solid-angle cutoff.

    Keep the original atan(abs(tan_half)) convention rather than substituting
    atan2 or a boundary-face test: the resulting subset defines the scene.
    """
    solid_angles = np.zeros(len(vertices))
    for element in elements:
        points = vertices[element]
        for corners, apex_index in (((1, 2, 3), 0), ((0, 2, 3), 1), ((0, 1, 3), 2), ((0, 1, 2), 3)):
            a_vec, b_vec, c_vec = points[list(corners)] - points[apex_index]
            a, b, c = np.linalg.norm(a_vec), np.linalg.norm(b_vec), np.linalg.norm(c_vec)
            tan_half = a_vec.dot(np.cross(b_vec, c_vec)) / (
                a * b * c + a_vec.dot(b_vec) * c + a_vec.dot(c_vec) * b + b_vec.dot(c_vec) * a
            )
            solid_angles[element[apex_index]] += np.arctan(np.abs(tan_half)) * 2
    return np.flatnonzero(solid_angles < 1.2 * np.pi).astype(np.int32)


def regions(vertices, elements):
    """The original split uses y, despite comments describing top/bottom."""
    midpoint = np.mean(vertices, axis=0)
    soft, stiff = [], []
    for index, element in enumerate(elements):
        v0, v1, v2, v3 = vertices[element]
        center = (v0 + v1 + v2 + v3) / 4
        (soft if center[1] < midpoint[1] else stiff).append(index)
    return np.asarray(soft, dtype=np.int32), np.asarray(stiff, dtype=np.int32)


def boundary_faces(elements, with_elements=False):
    """Outward faces for the original negative-determinant winding."""
    faces = {}
    for element_index, element in enumerate(elements):
        for local in ((0, 1, 2), (2, 1, 3), (1, 0, 3), (0, 2, 3)):
            face = element[list(local)]
            key = tuple(sorted(face))
            if key in faces:
                del faces[key]
            else:
                faces[key] = (face, element_index)
    triangles = np.asarray([value[0] for value in faces.values()], dtype=np.int32)
    if with_elements:
        return triangles, np.asarray([value[1] for value in faces.values()], dtype=np.int32)
    return triangles


def write_geometry(path, vertices, elements, contact, soft, stiff, source_hash):
    from pxr import Sdf, Usd, UsdGeom, Vt

    stage = Usd.Stage.CreateNew(str(path))
    stage.GetRootLayer().documentation = (
        "Local research geometry derived from the Stanford bunny supplied by DiffPhD. "
        "Credit: Stanford University Computer Graphics Laboratory. See adjacent manifest and asset NOTICE. "
        "Original node order, element order and negative determinant winding are retained. "
        "Materials, motion and solver settings belong to the example."
    )
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    prim = stage.DefinePrim("/Bunny", "TetMesh")
    stage.SetDefaultPrim(prim)
    prim.CreateAttribute("points", Sdf.ValueTypeNames.Point3fArray).Set(
        Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32))
    )
    prim.CreateAttribute("newton:referencePoints", Sdf.ValueTypeNames.Point3dArray).Set(
        Vt.Vec3dArray.FromNumpy(vertices)
    )
    prim.CreateAttribute("tetVertexIndices", Sdf.ValueTypeNames.Int4Array).Set(Vt.Vec4iArray.FromNumpy(elements))
    surface, surface_elements = boundary_faces(elements, with_elements=True)
    prim.CreateAttribute("surfaceFaceVertexIndices", Sdf.ValueTypeNames.Int3Array).Set(Vt.Vec3iArray.FromNumpy(surface))
    prim.CreateAttribute("diffphd:surfaceElementIndices", Sdf.ValueTypeNames.IntArray).Set(
        Vt.IntArray.FromNumpy(surface_elements)
    )
    for name, indices in (
        ("contactVertexIndices", contact),
        ("softElementIndices", soft),
        ("stiffElementIndices", stiff),
    ):
        prim.CreateAttribute(f"diffphd:{name}", Sdf.ValueTypeNames.IntArray).Set(Vt.IntArray.FromNumpy(indices))
    prim.CreateAttribute("newton:sourceSha256", Sdf.ValueTypeNames.String).Set(source_hash)
    stage.GetRootLayer().Save()


def prepare(reference_root, output):
    reference_root, output = Path(reference_root).resolve(), Path(output).resolve()
    hashes = {str(name): digest(reference_root / name) for name in SOURCE_FILES}
    unit_vertices, elements = tetrahedralize(reference_root / SOURCE_MESH)
    vertices = unit_vertices * 0.1
    contact = contact_vertices(vertices, elements)
    soft, stiff = regions(vertices, elements)
    determinant = np.linalg.det(vertices[elements[:, 1:]] - vertices[elements[:, :1]])
    if np.any(~np.isfinite(determinant)) or np.any(determinant >= 0):
        raise ValueError("Expected nondegenerate tetrahedra with the reference negative winding")
    output.mkdir(parents=True, exist_ok=True)
    geometry = output / "bunny_reference.usdc"
    write_geometry(geometry, vertices, elements, contact, soft, stiff, hashes[str(SOURCE_MESH)])
    revision = subprocess.run(
        ["git", "-C", str(reference_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    metadata = {
        "format": "diffphd-bunny-reference-v1",
        "geometry": geometry.name,
        "geometry_sha256": digest(geometry),
        "prim": "/Bunny",
        "source_repository": "https://github.com/Akinesia112/DiffPhD",
        "source_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "source_sha256": hashes,
        "preparation_versions": {
            name: version(name) for name in ("numpy", "trimesh", "pyvista", "tetgen", "pymeshfix", "vtk")
        },
        "recipe": {
            "normalization": "subtract source minimum; divide by largest bounding-box extent; STL round trip",
            "tetgen": "make_manifold then tetrahedralize with installed defaults; first four element columns",
            "scale": 0.1,
            "winding": "original helper negative determinants; runtime loader swaps corners 1 and 2",
            "ordering": "original node and element order; remove only unused nodes",
            "contact": "original solid-angle sum < 1.2*pi on scaled mesh",
            "regions": "soft when element centroid y < mean vertex y; stiff otherwise",
        },
        "counts": {
            "vertices": len(vertices),
            "tetrahedra": len(elements),
            "surface_triangles": len(boundary_faces(elements)),
            "contact_vertices": len(contact),
            "soft_elements": len(soft),
            "stiff_elements": len(stiff),
        },
        "volume_m3": float(-determinant.sum() / 6),
        "license": {
            "credit": "Stanford University Computer Graphics Laboratory",
            "terms_url": "https://graphics.stanford.edu/data/3Dscanrep/",
            "scope": "Local research asset; not covered by the solver Apache-2.0 or reference code MIT license",
            "terms_summary": "Research and free redistribution with acknowledgment; commercial use requires permission",
        },
    }
    manifest = output / "bunny_reference.json"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return manifest, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-root", required=True, type=Path, help="Local DiffPhD checkout containing the source mesh"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest, metadata = prepare(args.reference_root, args.output_dir)
    print(
        json.dumps(
            {"manifest": str(manifest), **metadata["counts"], "versions": metadata["preparation_versions"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
