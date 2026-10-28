# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Portable bear controller checkpoints and invalid-save protection."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from newton_solver_diff_phd.examples._diff_phd_bear_checkpoint import (
    TRAINED_CONTROLLER,
    load_controller,
    save_controller,
)


class TestDiffPhDBearCheckpoint(unittest.TestCase):
    def test_round_trip_and_invalid_save_preserves_previous_controller(self):
        weights = np.random.default_rng(42).normal(size=16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "controller.json"
            settings = {
                "weights": weights,
                "resolution": 10,
                "dt": 0.008,
                "iterations": 20,
                "loss": -0.08,
                "initial_loss": 0.01,
            }
            save_controller(path, **settings)
            original = path.read_bytes()
            data = load_controller(path)
            np.testing.assert_array_equal(np.asarray(data["weights"]).reshape(16), weights)
            self.assertEqual(data["resolution"], 10)
            self.assertEqual(data["dt"], 0.008)
            self.assertEqual(data["training"]["iterations"], 20)
            for key, value in (("weights", np.full(16, np.nan)), ("dt", 0.0), ("loss", np.inf)):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    save_controller(path, **(settings | {key: value}))
                self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_load_rejects_incompatible_or_malformed_checkpoints(self):
        data = load_controller(TRAINED_CONTROLLER)
        self.assertEqual(data["training"]["iterations"], 20)
        self.assertLess(data["training"]["loss"], data["training"]["initial_loss"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for key, value in (
                ("format", "another-scene"),
                ("resolution", True),
                ("dt", -0.008),
                ("frequency_hz", 3.0),
                ("amplitude", 0.5),
                ("weights", [0.0] * 16),
                ("weights", [[float("nan"), 0.0]] * 8),
            ):
                path.write_text(json.dumps(data | {key: value}))
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    load_controller(path)


if __name__ == "__main__":
    unittest.main()
