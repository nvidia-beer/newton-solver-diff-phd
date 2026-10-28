# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Topology and anatomical labels of the coarse DiffPhD bear."""

import hashlib
import importlib.util
import unittest

import numpy as np

from newton_solver_diff_phd._src.geometry.bear import BEAR_ASSET
from newton_solver_diff_phd._src.solvers.diff_phd.bear import bear_mesh


def _component_count(tets):
    neighbors = [set() for _ in tets]
    owners = {}
    for index, tet in enumerate(tets):
        for omitted in range(4):
            key = tuple(sorted(np.delete(tet, omitted)))
            if key in owners:
                other = owners[key]
                neighbors[index].add(other)
                neighbors[other].add(index)
            else:
                owners[key] = index
    unseen = set(range(len(tets)))
    components = 0
    while unseen:
        components += 1
        pending = [unseen.pop()]
        while pending:
            for neighbor in neighbors[pending.pop()] & unseen:
                unseen.remove(neighbor)
                pending.append(neighbor)
    return components


@unittest.skipUnless(importlib.util.find_spec("pxr"), "requires the existing USD example dependency")
class TestDiffPhDBearMesh(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mesh = bear_mesh()

    def test_default_is_smaller_connected_positive_volume_with_closed_outward_boundary(self):
        vertices, tets, faces, _, _, _, source_volume = self.mesh
        self.assertEqual(vertices.shape, (276, 3))
        self.assertEqual(tets.shape, (768, 4))
        self.assertEqual(faces.shape, (464, 3))
        self.assertEqual(len(np.unique(tets)), len(vertices))
        self.assertEqual(_component_count(tets), 1)
        corners = vertices[tets]
        determinants = np.linalg.det((corners[:, 1:] - corners[:, :1]).transpose(0, 2, 1))
        self.assertTrue(np.all(determinants > 0.0))
        self.assertAlmostEqual(determinants.sum() / 6.0, 113.0845859969893, places=9)
        self.assertAlmostEqual(source_volume, 98.4293, places=3)
        # Coarsening changes volume; the returned reference volume must not hide this.
        self.assertGreater(determinants.sum() / 6.0, source_volume * 1.14)
        edges = np.sort(faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1)
        unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
        np.testing.assert_array_equal(edge_counts, 2)
        self.assertEqual(len(np.unique(faces)) - len(unique_edges) + len(faces), 2)
        face_owners = {}
        for tet in tets:
            for omitted in range(4):
                key = tuple(sorted(np.delete(tet, omitted)))
                face_owners.setdefault(key, []).append(tet[omitted])
        for face in faces:
            opposite = face_owners[tuple(sorted(face))]
            self.assertEqual(len(opposite), 1)
            a, b, c = vertices[face]
            self.assertLess(np.dot(np.cross(b - a, c - a), vertices[opposite[0]] - a), 0.0)

    def test_four_feet_and_eight_muscle_parts_are_separate_from_connected_torso(self):
        vertices, tets, faces, legs, groups, face_materials, _ = self.mesh
        np.testing.assert_array_equal(np.unique(legs), [-1, 0, 1, 2, 3])
        np.testing.assert_array_equal(np.unique(groups), np.arange(-1, 8))
        self.assertEqual(int(np.sum(legs >= 0)), 96)
        self.assertEqual(_component_count(tets[legs < 0]), 1)
        np.testing.assert_array_equal(groups < 0, legs < 0)
        for leg in range(4):
            elements = tets[legs == leg]
            self.assertEqual(_component_count(elements), 1)
            np.testing.assert_array_equal(np.unique(groups[legs == leg]), [2 * leg, 2 * leg + 1])
            points = vertices[np.unique(elements)]
            self.assertAlmostEqual(points[:, 2].min(), vertices[:, 2].min(), places=12)
            self.assertLess(points[:, 2].max(), 2.0)
            self.assertEqual(bool(points[:, 0].mean() > -0.5), bool(leg // 2))
            self.assertEqual(bool(points[:, 1].mean() > 0.0), bool(leg % 2))
        self.assertEqual(_component_count(tets[legs >= 0]), 4)
        for face, material in zip(faces, face_materials, strict=True):
            owners = [index for index, tet in enumerate(tets) if set(face).issubset(tet)]
            self.assertEqual(len(owners), 1)
            self.assertEqual(material, int(legs[owners[0]] >= 0))

    def test_baked_reference_geometry_is_exact_and_has_no_material_parameters(self):
        from pxr import Usd

        digest = hashlib.sha256(BEAR_ASSET.read_bytes()).digest()
        repeat = bear_mesh()
        for first, second in zip(self.mesh, repeat, strict=True):
            np.testing.assert_array_equal(first, second)
        self.assertEqual(hashlib.sha256(BEAR_ASSET.read_bytes()).digest(), digest)
        stage = Usd.Stage.Open(str(BEAR_ASSET))
        coarse = stage.GetPrimAtPath("/Bear/Coarse")
        source = stage.GetPrimAtPath("/Bear/Source")
        points = np.asarray(source.GetAttribute("newton:referencePoints").Get(), dtype=np.float64)
        tets = np.asarray(source.GetAttribute("tetVertexIndices").Get(), dtype=np.int32)
        self.assertEqual(len(points), 1986)
        self.assertEqual(len(tets), 5354)
        corners = points[tets]
        volume = np.abs(np.linalg.det((corners[:, 1:] - corners[:, :1]).transpose(0, 2, 1))).sum() / 6.0
        self.assertAlmostEqual(self.mesh[-1], volume, places=12)
        np.testing.assert_array_equal(np.asarray(coarse.GetAttribute("newton:referencePoints").Get()), self.mesh[0])
        for prim in stage.Traverse():
            self.assertFalse(
                any(
                    "material" in attribute.GetName().lower()
                    or "stiffness" in attribute.GetName().lower()
                    or "density" in attribute.GetName().lower()
                    for attribute in prim.GetAttributes()
                )
            )

    def test_unsupported_resolutions_are_rejected_without_runtime_meshing(self):
        for resolution in (True, 9, 0, -1, 10.0, "10", 12, 13):
            with self.subTest(resolution=resolution), self.assertRaisesRegex(ValueError, "integer resolution 10 only"):
                bear_mesh(resolution)


if __name__ == "__main__":
    unittest.main()
