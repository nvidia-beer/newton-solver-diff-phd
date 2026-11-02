# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Prepared recording geometry round-trips without private recorded scans."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from newton_solver_diff_phd.examples._poke_recording import load_poke_recording


def _data():
    vertices = np.array(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float64)
    faces = np.array(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)), dtype=np.int32)
    return {
        "vertices": vertices,
        "tets": np.array(((0, 1, 2, 3),), dtype=np.int32),
        "faces": faces,
        "pinned": np.array([0, 1, 2], dtype=np.int32),
        "surface_vertices": vertices.copy(),
        "surface_faces": faces.copy(),
        "surface_uv": vertices[:, :2].copy(),
        "surface_embedding_indices": np.tile(np.arange(4, dtype=np.int32), (4, 1)),
        "surface_embedding_weights": np.eye(4),
        "frame_ids": np.array([18, 19], dtype=np.int32),
        "target_points": np.stack((vertices, vertices + (0, 0, 0.1))),
        "target_vertex_indices": np.tile(np.arange(4, dtype=np.int32), (2, 1)),
        "reference_xy": np.zeros(2),
        "reference_mesh": np.array(True),
        "probe_vertices": vertices * 0.125 + (0.5, 0.5, 2),
        "probe_tets": np.array(((0, 1, 2, 3),), dtype=np.int32),
    }


def _manifest(directory, data, *, usd_probe=True, archive_probe=False):
    from pxr import Sdf, Usd, UsdGeom, Vt

    geometry = directory / "die.usda"
    stage = Usd.Stage.CreateNew(str(geometry))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    # Probe deliberately precedes Die in stage traversal. A first-mesh loader
    # silently changes both the physics geometry and the embedding's meaning.
    names = [("/Probe", "probe_vertices", "probe_tets")] if usd_probe else []
    names += [("/Die/Volume", "vertices", "tets")]
    for name, points, tets in names:
        prim = stage.DefinePrim(name, "TetMesh")
        prim.CreateAttribute("points", Sdf.ValueTypeNames.Point3fArray).Set(
            Vt.Vec3fArray.FromNumpy(data[points].astype(np.float32))
        )
        prim.CreateAttribute("newton:referencePoints", Sdf.ValueTypeNames.Point3dArray).Set(
            Vt.Vec3dArray.FromNumpy(data[points])
        )
        prim.CreateAttribute("tetVertexIndices", Sdf.ValueTypeNames.Int4Array).Set(Vt.Vec4iArray.FromNumpy(data[tets]))
    volume = stage.GetPrimAtPath("/Die/Volume")
    volume.CreateAttribute("surfaceFaceVertexIndices", Sdf.ValueTypeNames.Int3Array).Set(
        Vt.Vec3iArray.FromNumpy(data["faces"])
    )
    surface = UsdGeom.Mesh.Define(stage, "/Die/Surface")
    surface.CreateFaceVertexIndicesAttr(data["surface_faces"].ravel().tolist())
    surface.CreateFaceVertexCountsAttr([3] * len(data["surface_faces"]))
    prim = surface.GetPrim()
    for name, field, dtype, vector in (
        ("newton:referencePoints", "surface_vertices", Sdf.ValueTypeNames.Point3dArray, Vt.Vec3dArray),
        ("primvars:st", "surface_uv", Sdf.ValueTypeNames.TexCoord2dArray, Vt.Vec2dArray),
        ("newton:embeddingIndices", "surface_embedding_indices", Sdf.ValueTypeNames.Int4Array, Vt.Vec4iArray),
        ("newton:embeddingWeights", "surface_embedding_weights", Sdf.ValueTypeNames.Double4Array, Vt.Vec4dArray),
    ):
        prim.CreateAttribute(name, dtype).Set(vector.FromNumpy(data[field]))
    stage.GetRootLayer().Save()
    archive = directory / "observations.npz"
    fields = ("pinned", "frame_ids", "target_points", "target_vertex_indices", "reference_xy", "reference_mesh")
    if archive_probe:
        fields += ("probe_vertices", "probe_tets")
    np.savez(archive, **{key: data[key] for key in fields})
    manifest = {
        "format": "newton.diff_phd.poke_recording.v1",
        "geometry": geometry.name,
        "recording": archive.name,
        "geometry_sha256": hashlib.sha256(geometry.read_bytes()).hexdigest(),
        "recording_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    path = directory / "recording.json"
    path.write_text(json.dumps(manifest))
    return path


class TestPokeRecording(unittest.TestCase):
    def test_both_volumes_load_by_name_with_probe_first(self):
        data = _data()
        with tempfile.TemporaryDirectory() as directory:
            path = _manifest(Path(directory), data)
            loaded = load_poke_recording(path)
            for key, value in data.items():
                with self.subTest(field=key):
                    np.testing.assert_array_equal(loaded[key], value)
            reconstructed = np.einsum(
                "ni,nij->nj",
                loaded["surface_embedding_weights"],
                loaded["vertices"][loaded["surface_embedding_indices"]],
            )
            np.testing.assert_array_equal(reconstructed, data["surface_vertices"])
            loaded["vertices"][:] = 100
            loaded["probe_vertices"][:] = 100
            reloaded = load_poke_recording(path)
            np.testing.assert_array_equal(reloaded["vertices"], data["vertices"])
            np.testing.assert_array_equal(reloaded["probe_vertices"], data["probe_vertices"])

    def test_checksums_reject_modified_geometry_and_recording(self):
        for name in ("die.usda", "observations.npz"):
            with self.subTest(asset=name), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                path = _manifest(directory, _data())
                asset = directory / name
                asset.write_bytes(asset.read_bytes() + b"modified")
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    load_poke_recording(path)

    def test_die_only_manifest_preserves_legacy_probe_arrays(self):
        data = _data()
        with tempfile.TemporaryDirectory() as directory:
            path = _manifest(Path(directory), data, usd_probe=False, archive_probe=True)
            loaded = load_poke_recording(path)
            for key in ("vertices", "tets", "probe_vertices", "probe_tets"):
                np.testing.assert_array_equal(loaded[key], data[key])

    def test_legacy_self_contained_npz_remains_supported(self):
        data = _data()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.npz"
            np.savez(path, **data)
            loaded = load_poke_recording(path)
            for key, value in data.items():
                np.testing.assert_array_equal(loaded[key], value)


if __name__ == "__main__":
    unittest.main()
