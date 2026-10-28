# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Portable controller parameters for the fixed DiffPhD bear scene."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from newton_solver_diff_phd import BearProblem

AMPLITUDE = BearProblem.AMPLITUDE
FREQUENCY = BearProblem.FREQUENCY

TRAINED_CONTROLLER = Path(__file__).parent / "assets" / "bear_controller.json"
_FORMAT = "newton.diff_phd_bear.v1"


def load_controller(path):
    """Load weights and scene settings without loading a saved trajectory."""
    data = json.loads(Path(path).read_text())
    _validate(data)
    return data


def _validate(data):
    if not isinstance(data, dict) or data.get("format") != _FORMAT:
        raise ValueError("Unsupported bear controller checkpoint format")
    resolution = data.get("resolution")
    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution < 10:
        raise ValueError("Checkpoint resolution must be an integer of at least ten")
    dt = data.get("dt")
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("Checkpoint dt must be finite and positive")
    if data.get("frequency_hz") != FREQUENCY or data.get("amplitude") != AMPLITUDE:
        raise ValueError("Checkpoint does not match this bear's sinusoidal controller")
    weights = np.asarray(data.get("weights"), dtype=np.float64)
    if weights.shape != (8, 2) or not np.isfinite(weights).all():
        raise ValueError("Checkpoint must contain eight finite sine/cosine weight pairs")


def save_controller(path, *, weights, resolution, dt, iterations, loss, initial_loss):
    """Atomically save a controller; each accepted training update replaces it."""
    data = {
        "format": _FORMAT,
        "resolution": resolution,
        "dt": dt,
        "frequency_hz": FREQUENCY,
        "amplitude": AMPLITUDE,
        "weights": np.asarray(weights, dtype=np.float64).reshape(8, 2).tolist(),
        "training": {"iterations": iterations, "loss": loss, "initial_loss": initial_loss},
    }
    _validate(data)
    payload = json.dumps(data, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
