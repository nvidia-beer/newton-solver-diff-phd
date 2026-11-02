# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Bundled drop geometry and static Tet4/Hex8 USD coordinate conventions."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import newton
import numpy as np
import warp as wp
from newton.viewer import ViewerNull

from newton_solver_diff_phd import load_mesh
from tests import demo_drop as example_drop


class TestGeometry(unittest.TestCase):
    def test_bundled_drop_preserves_original_model(self):
        # Frozen output of the former tet_grid((2, 1, 1), (0.15, 0.15, 0.15)).
        vertices = np.array(
            [
                (0, 0, 0),
                (0, 0, 0.15),
                (0, 0.15, 0),
                (0, 0.15, 0.15),
                (0.075, 0, 0),
                (0.075, 0, 0.15),
                (0.075, 0.15, 0),
                (0.075, 0.15, 0.15),
                (0.15, 0, 0),
                (0.15, 0, 0.15),
                (0.15, 0.15, 0),
                (0.15, 0.15, 0.15),
            ]
        )
        indices = np.array(
            [
                (4, 5, 6, 0),
                (1, 0, 3, 5),
                (2, 3, 0, 6),
                (7, 6, 5, 3),
                (6, 5, 3, 0),
                (4, 8, 6, 5),
                (9, 5, 11, 8),
                (10, 6, 8, 11),
                (7, 11, 5, 6),
                (6, 8, 11, 5),
            ],
            dtype=np.int32,
        )
        mesh = load_mesh(
            Path(__file__).resolve().parents[1] / "src/newton_solver_diff_phd/examples/assets/drop_cube.usda"
        )[0]
        np.testing.assert_array_equal(mesh.vertices, vertices.astype(np.float32))
        np.testing.assert_array_equal(mesh.indices, indices)

        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder()
            builder.add_ground_plane()
            builder.add_soft_mesh(
                pos=wp.vec3(-0.075, -0.075, 0.04),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.1, 0.0, 0.0),
                scale=1.0,
                vertices=vertices,
                indices=indices.flatten(),
                density=500.0,
                k_mu=1000.0,
                k_lambda=2000.0,
                k_damp=0.0,
                add_surface_mesh_edges=False,
                particle_radius=0.002,
            )
            centers = np.asarray(builder.particle_q)[np.asarray(builder.tet_indices)].mean(axis=1)
            builder.tet_materials = [(3000.0, 6000.0, 0.0) if p[0] > 0 else (1000.0, 2000.0, 0.0) for p in centers]
            expected = builder.finalize(device="cpu")
            actual = example_drop.Example(ViewerNull(), SimpleNamespace(substeps=4)).model
            for name in ("particle_q", "particle_qd", "particle_mass", "tet_indices", "tet_materials", "tri_indices"):
                with self.subTest(array=name):
                    np.testing.assert_array_equal(getattr(actual, name).numpy(), getattr(expected, name).numpy())
            self.assertAlmostEqual(float(actual.particle_mass.numpy().sum()), 500.0 * 0.15**3, places=6)

    def test_usd_transforms_units_and_up_axis(self):
        from pxr import Gf, Usd, UsdGeom, Vt

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tet.usda"
            stage = Usd.Stage.CreateNew(str(path))
            UsdGeom.SetStageMetersPerUnit(stage, 0.01)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
            parent = UsdGeom.Xform.Define(stage, "/Model")
            parent.AddTranslateOp().Set(Gf.Vec3d(10, 20, 30))
            parent.AddScaleOp().Set(Gf.Vec3f(-1, 1, 1))
            mesh = UsdGeom.TetMesh.Define(stage, "/Model/Tet")
            mesh.CreatePointsAttr(Vt.Vec3fArray([(0, 0, 0), (100, 0, 0), (0, 100, 0), (0, 0, 100)]))
            mesh.CreateTetVertexIndicesAttr(Vt.Vec4iArray([(0, 1, 2, 3)]))
            stage.GetRootLayer().Save()
            loaded = load_mesh(path)[0]
            np.testing.assert_allclose(
                loaded.vertices, [(0.1, -0.3, 0.2), (-0.9, -0.3, 0.2), (0.1, -0.3, 1.2), (0.1, -1.3, 0.2)]
            )
            np.testing.assert_array_equal(loaded.indices, [[0, 2, 1, 3]])
            edges = loaded.vertices[loaded.indices[0, 1:]] - loaded.vertices[loaded.indices[0, 0]]
            self.assertGreater(np.linalg.det(edges), 0.0)

    def test_invalid_connectivity_is_rejected(self):
        from pxr import Usd, UsdGeom, Vt

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.usda"
            stage = Usd.Stage.CreateNew(str(path))
            mesh = UsdGeom.TetMesh.Define(stage, "/Tet")
            mesh.CreatePointsAttr(Vt.Vec3fArray([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]))
            for indices, message in (((0, 1, 2, 4), "valid indices"), ((0, 1, 2, 2), "degenerate")):
                with self.subTest(indices=indices):
                    mesh.CreateTetVertexIndicesAttr(Vt.Vec4iArray([indices]))
                    stage.GetRootLayer().Save()
                    with self.assertRaisesRegex(ValueError, message):
                        load_mesh(path)

    def test_hex_transforms_units_up_axis_and_native_corner_order(self):
        from pxr import Gf, Sdf, Usd, UsdGeom, Vt

        # Reference Hex8 corners use binary x/y/z order, not the cyclic order
        # convention commonly used by surface meshes. Shuffle vertex storage
        # so that preserving connectivity requires more than returning 0..7.
        corners = np.array(
            [
                (0, 0, 0),
                (0, 0, 100),
                (0, 100, 0),
                (0, 100, 100),
                (100, 0, 0),
                (100, 0, 100),
                (100, 100, 0),
                (100, 100, 100),
            ],
            dtype=np.float64,
        )
        storage_order = np.array([4, 1, 7, 2, 0, 6, 3, 5])
        connectivity = np.argsort(storage_order).astype(np.int32)
        expected_corners = np.array(
            [
                (0.1, -0.3, 0.2),
                (0.1, -0.8, 0.2),
                (0.1, -0.3, 2.2),
                (0.1, -0.8, 2.2),
                (-0.9, -0.3, 0.2),
                (-0.9, -0.8, 0.2),
                (-0.9, -0.3, 2.2),
                (-0.9, -0.8, 2.2),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hex.usda"
            stage = Usd.Stage.CreateNew(str(path))
            UsdGeom.SetStageMetersPerUnit(stage, 0.01)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
            parent = UsdGeom.Xform.Define(stage, "/Model")
            parent.AddTranslateOp().Set(Gf.Vec3d(10, 20, 30))
            parent.AddScaleOp().Set(Gf.Vec3f(-1, 2, 0.5))
            mesh = UsdGeom.Mesh.Define(stage, "/Model/Hex")
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(corners[storage_order].astype(np.float32)))
            mesh.GetPrim().CreateAttribute("newton:hexVertexIndices", Sdf.ValueTypeNames.IntArray).Set(
                Vt.IntArray.FromNumpy(connectivity)
            )
            stage.GetRootLayer().Save()

            (loaded,) = load_mesh(path)
            self.assertEqual(loaded.name, "/Model/Hex")
            np.testing.assert_allclose(loaded.vertices, expected_corners[storage_order], rtol=0, atol=1e-15)
            np.testing.assert_array_equal(loaded.indices, connectivity[None, :])
            np.testing.assert_allclose(loaded.vertices[loaded.indices[0]], expected_corners, rtol=0, atol=1e-15)
            self.assertEqual(loaded.indices.dtype, np.dtype(np.int32))
            # Even a reflected transform must not invoke Tet4 orientation
            # repair on Hex8. The solver validates the rest Jacobians.

    def test_hex_invalid_flat_connectivity_is_rejected(self):
        from pxr import Sdf, Usd, UsdGeom, Vt

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid_hex.usda"
            stage = Usd.Stage.CreateNew(str(path))
            mesh = UsdGeom.Mesh.Define(stage, "/Hex")
            mesh.CreatePointsAttr(
                Vt.Vec3fArray([(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)])
            )
            attribute = mesh.GetPrim().CreateAttribute("newton:hexVertexIndices", Sdf.ValueTypeNames.IntArray)
            cases = (
                ([], "8 valid indices"),
                (list(range(7)), "8 valid indices"),
                (list(range(8)) + [0], "8 valid indices"),
                ([0, 1, 2, 3, 4, 5, 6, 8], "8 valid indices"),
                ([0, 1, 2, 3, 4, 5, 6, -1], "8 valid indices"),
                ([0, 1, 2, 3, 4, 5, 6, 6], "repeated vertices"),
            )
            for indices, message in cases:
                with self.subTest(indices=indices):
                    attribute.Set(Vt.IntArray(indices))
                    stage.GetRootLayer().Save()
                    with self.assertRaisesRegex(ValueError, message):
                        load_mesh(path)

    def test_mixed_hex_and_tet_stage_keeps_element_types(self):
        from pxr import Sdf, Usd, UsdGeom, Vt

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.usda"
            stage = Usd.Stage.CreateNew(str(path))
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            mesh = UsdGeom.Mesh.Define(stage, "/Hex")
            mesh.CreatePointsAttr(
                Vt.Vec3fArray([(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)])
            )
            mesh.GetPrim().CreateAttribute("newton:hexVertexIndices", Sdf.ValueTypeNames.IntArray).Set(
                Vt.IntArray(range(8))
            )
            tet = UsdGeom.TetMesh.Define(stage, "/Tet")
            tet.CreatePointsAttr(Vt.Vec3fArray([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]))
            tet.CreateTetVertexIndicesAttr(Vt.Vec4iArray([(0, 2, 1, 3)]))
            UsdGeom.Mesh.Define(stage, "/SurfaceOnly")
            stage.GetRootLayer().Save()

            loaded = {mesh.name: mesh for mesh in load_mesh(path)}
            self.assertEqual(set(loaded), {"/Hex", "/Tet"})
            np.testing.assert_array_equal(loaded["/Hex"].indices, [range(8)])
            np.testing.assert_array_equal(loaded["/Tet"].indices, [[0, 1, 2, 3]])


if __name__ == "__main__":
    unittest.main()
