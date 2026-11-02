"""Poke configuration uses the restored solver and rejects incompatible fits."""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import warp as wp

from newton_solver_diff_phd import SolverDiffPhD
from newton_solver_diff_phd.examples._diff_phd_poke import POKE_PHYSICS, build_poke_scene
from newton_solver_diff_phd.examples.example_softbody_diff_phd_ur5_poke import (
    Example,
    _load_alignment,
    _scene_fingerprint,
)

_EXAMPLE = "newton_solver_diff_phd.examples.example_softbody_diff_phd_ur5_poke"


def _scene():
    """Public tetrahedron with a fixed surface embedding, without recording data."""
    vertices = np.array(((0, 0, 0), (0.1, 0, 0), (0, 0.1, 0), (0, 0, 0.1)), dtype=np.float64)
    faces = np.array(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)))
    data = {
        "vertices": vertices,
        "tets": np.array(((0, 1, 2, 3),)),
        "faces": faces,
        "pinned": np.array((0, 1, 2)),
        "target_vertex_indices": np.array([np.arange(4)]),
        "reference_xy": np.zeros(2),
        "surface_vertices": vertices.copy(),
        "surface_faces": faces.copy(),
        "surface_embedding_indices": np.tile(np.arange(4), (4, 1)),
        "surface_embedding_weights": np.eye(4),
        "target_points": np.tile(vertices, (36, 1, 1)),
    }
    with patch("newton_solver_diff_phd.examples._diff_phd_poke.load_poke_recording", return_value=data):
        return build_poke_scene("synthetic.npz", device=os.environ.get("NEWTON_TEST_DEVICE", "cpu"))


def _checkpoint(**updates):
    data = {
        "format": "newton.diff_phd.poke_alignment",
        "version": 5,
        "physics": dict(POKE_PHYSICS),
        "scene_sha256": "0" * 64,
        "xy_m": [0.7, 0.1],
        "substeps": 30,
        "friction": 0.5,
        "contact_radius_m": 0.00055,
        "velocity_damping": 0.3,
    }
    data.update(updates)
    return data


class TestPokeScene(unittest.TestCase):
    def test_prepared_scene_uses_reference_contact(self):
        with patch("newton_solver_diff_phd.examples._diff_phd_poke.SolverDiffPhD", wraps=SolverDiffPhD) as constructor:
            scene = _scene()
        solver = scene["solver"]
        self.assertEqual(scene["contact_radius"], 0.00055)
        self.assertEqual(scene["friction"], 0.5)
        self.assertEqual(scene["substeps"], 30)
        self.assertEqual(solver.material, "neo_hookean")
        self.assertEqual(solver.rel_tol, 1e-4)
        self.assertEqual(solver.max_iterations, 5000)
        self.assertEqual(solver.abs_tol, 1e-9)
        self.assertEqual(solver.density, 800.0)
        self.assertIsNone(solver.contact_multipliers)
        self.assertEqual(constructor.call_args.kwargs["mesh_contact"][:5], (0.00055, 5e3, 5e4, 50.0, 0.5))
        np.testing.assert_array_equal(constructor.call_args.kwargs["mesh_contact"][5], scene["data"]["faces"])
        self.assertEqual(scene["physics"], POKE_PHYSICS)
        soft_count = scene["soft_count"]
        mass = scene["model"].particle_mass.numpy()
        root = soft_count + np.flatnonzero(scene["tool_local"][:, 2] - scene["tool_local"][:, 2].min() < 1e-4)
        np.testing.assert_array_equal(scene["tool_root_indices"], root)
        np.testing.assert_array_equal(np.flatnonzero(mass[soft_count:] == 0.0) + soft_count, root)
        self.assertGreater(np.count_nonzero(mass[soft_count:] > 0.0), 0)
        materials = scene["model"].tet_materials.numpy()
        np.testing.assert_allclose(materials[0, :2], [5e5 / 2.8, 5e5 * 0.4 / (1.4 * 0.2)], rtol=1e-6)
        np.testing.assert_allclose(materials[1:, 0], 1e13 / 2.9, rtol=1e-6)
        np.testing.assert_allclose(materials[1:, 1], 1e13 * 0.45 / (1.45 * 0.1), rtol=1e-6)

    def test_checkpoint_requires_refit_after_contact_change(self):
        data = _checkpoint()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alignment.json"
            path.write_text(json.dumps(data))
            self.assertEqual(_load_alignment(path)["version"], 5)
            for version in (1, 2, 3, 4):
                with self.subTest(version=version):
                    data["version"] = version
                    path.write_text(json.dumps(data))
                    with self.assertRaisesRegex(ValueError, "refitting"):
                        _load_alignment(path)

    def test_checkpoint_rejects_changed_physics_and_malformed_geometry_hash(self):
        bad_physics = dict(POKE_PHYSICS, contact_velocity="endpoint_displacement_over_dt")
        for updates in ({"physics": bad_physics}, {"scene_sha256": "G" * 64}, {"scene_sha256": "abcd"}):
            with self.subTest(updates=updates), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "alignment.json"
                path.write_text(json.dumps(_checkpoint(**updates)))
                with self.assertRaisesRegex(ValueError, "refitting"):
                    _load_alignment(path)

    def test_checkpoint_identifies_loaded_geometry_motion_and_targets(self):
        scene = _scene()
        fingerprint = _scene_fingerprint(scene)
        for array in (scene["rest"], scene["tool_positions"], scene["data"]["target_points"]):
            original = array.flat[0]
            array.flat[0] += 0.001
            self.assertNotEqual(_scene_fingerprint(scene), fingerprint)
            array.flat[0] = original
        self.assertEqual(_scene_fingerprint(scene), fingerprint)
        with tempfile.TemporaryDirectory() as directory:
            recording = Path(directory) / "recording.npz"
            recording.write_bytes(b"synthetic recording identity")
            checkpoint = Path(directory) / "alignment.json"
            checkpoint.write_text(
                json.dumps(
                    _checkpoint(
                        recording_sha256=hashlib.sha256(recording.read_bytes()).hexdigest(),
                        frame_dt=scene["frame_dt"],
                        reference_xy_m=scene["data"]["reference_xy"].tolist(),
                    )
                )
            )
            args = Example.create_parser().parse_args(
                ["--recording", str(recording), "--load-alignment", str(checkpoint)]
            )
            with patch(f"{_EXAMPLE}.build_poke_scene", return_value=scene):
                with self.assertRaisesRegex(ValueError, "same loaded geometry, motion"):
                    Example(MagicMock(), args)

    def test_forward_only_starts_without_checkpoint_or_optimizer(self):
        scene = _scene()
        robot = MagicMock()
        robot.state.body_q = wp.zeros(1, dtype=wp.transform, device=scene["model"].device)
        with tempfile.TemporaryDirectory() as directory:
            recording = Path(directory) / "recording.npz"
            recording.write_bytes(b"synthetic recording identity")
            args = Example.create_parser().parse_args(
                ["--forward-only", "--recording", str(recording), "--initial-xy", "0.01", "0.02"]
            )
            with (
                patch(f"{_EXAMPLE}.build_poke_scene", return_value=scene),
                patch(f"{_EXAMPLE}.UR5Kinematics", return_value=robot),
                patch(f"{_EXAMPLE}.PokeAlignmentProblem", side_effect=AssertionError("Forward mode created optimizer")),
                patch.object(wp, "Tape", side_effect=AssertionError("Forward mode created Tape")),
            ):
                example = Example(MagicMock(), args)
                self.assertTrue(example.inference)
                self.assertIsNone(example.problem)
                self.assertIsNone(example.save_path)
                np.testing.assert_array_equal(example.xy, [0.01, 0.02])
                with patch.object(example.forward, "advance") as advance:
                    example.step()
                    example.step()
                    advance.assert_called_once()

    def test_forward_only_rejects_training_outputs(self):
        for option in ("--save-alignment", "--output", "--target-rms-mm", "--check-gradients"):
            with self.subTest(option=option):
                values = ["--forward-only", option] + ([] if option == "--check-gradients" else ["1"])
                args = Example.create_parser().parse_args(values)
                with self.assertRaisesRegex(ValueError, "training"):
                    Example(MagicMock(), args)


if __name__ == "__main__":
    unittest.main()
