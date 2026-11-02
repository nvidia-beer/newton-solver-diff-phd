"""Surface comparison checks use public planar meshes, without private data."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.compare_poke_recording import compare_recording, main


def _fixture(white_heights=(0.0, 0.0), scan_heights=(0.0, 0.0)):
    square = np.array(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)), dtype=np.float64)
    white_faces = np.array(((0, 1, 2), (0, 2, 3)))
    # The recorded mesh has an extra centre vertex and four triangles, so its
    # vertex-nearest-neighbour distances differ from surface distances.
    scan = np.vstack((square, [0.5, 0.5, 0.0]))
    scan_faces = np.array(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)))
    frames = len(scan_heights)
    targets = np.stack([scan + [0, 0, height] for height in scan_heights])
    data = {
        "vertices": square.copy(),
        "tets": np.array(((0, 1, 2, 3),)),
        "faces": white_faces.copy(),
        "pinned": np.array([0]),
        "surface_vertices": square.copy(),
        "surface_faces": white_faces,
        "surface_uv": square[:, :2].copy(),
        "surface_embedding_indices": np.tile(np.arange(4), (4, 1)),
        "surface_embedding_weights": np.eye(4),
        "reference_xy": np.zeros(2),
        "frame_ids": np.arange(18, 18 + frames),
        "target_vertex_indices": np.tile(np.arange(4), (frames, 1)),
        "target_points": targets,
        "recorded_vertices": targets.reshape(-1, 3),
        "recorded_vertex_offsets": np.arange(frames + 1) * len(scan),
        "recorded_faces": np.tile(scan_faces, (frames, 1)),
        "recorded_face_offsets": np.arange(frames + 1) * len(scan_faces),
    }
    positions = np.stack([square + [0, 0, height] for height in white_heights])
    return data, positions


class TestComparePokeRecording(unittest.TestCase):
    def test_equal_surfaces_with_different_vertices_and_triangulations(self):
        data, positions = _fixture()
        original = copy.deepcopy(data)
        snapshot = positions.copy()
        report, arrays = compare_recording(data, positions)
        for frame in report["frames"]:
            for name in ("trajectory", "frozen_frame0"):
                self.assertLess(frame[name]["symmetric_rms_m"], 1e-7)
                self.assertLess(frame[name]["symmetric_p95_m"], 1e-7)
                self.assertLess(frame[name]["max_distance_m"], 1e-7)
                self.assertGreater(frame[name]["training_chamfer_m2"], 0.0)
        np.testing.assert_array_equal(arrays["white_vertices"], positions)
        np.testing.assert_array_equal(positions, snapshot)
        for key in data:
            np.testing.assert_array_equal(data[key], original[key])

    def test_offset_surfaces_have_known_unsigned_distances(self):
        data, positions = _fixture(scan_heights=(0.2, 0.2))
        report, _ = compare_recording(data, positions)
        for frame in report["frames"]:
            metric = frame["trajectory"]
            for direction in ("white_to_scan", "scan_to_white"):
                for key in ("rms_m", "p95_m", "max_m"):
                    self.assertAlmostEqual(metric[direction][key], 0.2, delta=1e-7)
            self.assertAlmostEqual(metric["symmetric_rms_m"], 0.2, delta=1e-7)

    def test_frozen_baseline_excludes_frame_zero_and_does_not_follow_motion(self):
        data, positions = _fixture(white_heights=(0.0, 0.25, 0.5), scan_heights=(0.0, 0.25, 0.5))
        report, arrays = compare_recording(data, positions)
        aggregate = report["aggregate_excluding_frame0"]
        expected = np.sqrt((0.25**2 + 0.5**2) / 2)
        self.assertEqual(aggregate["trajectory"]["frames"], 2)
        self.assertLess(aggregate["trajectory"]["symmetric_rms_m"], 1e-7)
        self.assertAlmostEqual(aggregate["frozen_frame0"]["symmetric_rms_m"], expected, delta=1e-7)
        self.assertAlmostEqual(aggregate["rms_improvement_over_frozen_m"], expected, delta=1e-7)
        self.assertEqual(arrays["white_to_scan"].shape, (3, 4))
        self.assertEqual(arrays["scan_to_white"].shape, (15,))
        np.testing.assert_allclose(arrays["frozen_white_to_scan"][2], 0.5, atol=1e-7)

    def test_rejects_invalid_trajectory_embedding_and_missing_meshes(self):
        data, positions = _fixture()
        for invalid in (positions[:1], positions[:, :3], positions[:, :, :2], positions * np.nan):
            with self.subTest(shape=invalid.shape), self.assertRaisesRegex(ValueError, "positions"):
                compare_recording(data, invalid)
        invalid = copy.deepcopy(data)
        invalid["surface_embedding_weights"][0] *= 0.5
        with self.assertRaisesRegex(ValueError, "embedding"):
            compare_recording(invalid, positions)
        invalid = copy.deepcopy(data)
        del invalid["recorded_faces"]
        with self.assertRaisesRegex(ValueError, "Full recorded triangle meshes"):
            compare_recording(invalid, positions)
        invalid = copy.deepcopy(data)
        invalid["recorded_faces"][0, 0] = 5
        with self.assertRaisesRegex(ValueError, "triangle indices"):
            compare_recording(invalid, positions)

    def test_cli_records_hashes_distances_and_rejects_frame_id_mismatch(self):
        data, positions = _fixture()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            recording, trajectory = directory / "recording.npz", directory / "trajectory.npz"
            output, distances = directory / "metrics.json", directory / "distances.npz"
            np.savez(recording, **data)
            np.savez(trajectory, saved=positions, frame_ids=data["frame_ids"])
            args = [
                "--recording",
                str(recording),
                "--trajectory",
                str(trajectory),
                "--positions-key",
                "saved",
                "--output",
                str(output),
                "--distances-output",
                str(distances),
            ]
            main(args)
            report = json.loads(output.read_text())
            self.assertEqual(report["positions_key"], "saved")
            self.assertEqual(len(report["inputs"]["trajectory"]["sha256"]), 64)
            with np.load(distances) as archive:
                np.testing.assert_array_equal(archive["white_vertices"], positions)
            np.savez(trajectory, saved=positions, frame_ids=data["frame_ids"][::-1])
            with self.assertRaisesRegex(ValueError, "frame_ids must match"):
                main(args)


if __name__ == "__main__":
    unittest.main()
