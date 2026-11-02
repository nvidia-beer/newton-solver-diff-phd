# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Record a complete live Hex8 cantilever trajectory for native comparison."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp

from newton_solver_diff_phd.examples._diff_phd_cantilever import CantileverSimulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", choices=("axial", "twist"), default="axial")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-capture", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    sim = CantileverSimulation(device=args.device, motion=args.motion, capture=not args.no_capture)
    setup_seconds = perf_counter() - started
    q, v = [sim.states[0].particle_q.numpy().copy()], [sim.states[0].particle_qd.numpy().copy()]
    forces, iterations, determinants, seconds = [], [], [], []
    for _ in range(sim.frames):
        start = perf_counter()
        sim.advance()
        wp.synchronize_device(sim.model.device)
        seconds.append(perf_counter() - start)
        q.append(sim.states[0].particle_q.numpy().copy())
        v.append(sim.states[0].particle_qd.numpy().copy())
        forces.append(sim.forces.copy())
        iterations.append(int(sim.solver.iterations.numpy()[0]))
        determinants.append(sim.solver.deformation_determinants(sim.states[0]).numpy())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out.with_suffix(".npz"),
        rest_positions=sim.vertices,
        hexes=sim.hexes,
        hex_materials=sim.solver.hex_materials.numpy(),
        q=q,
        v=v,
        forces=forces,
        iterations=iterations,
        determinants=determinants,
    )
    report = {
        "motion": args.motion,
        "device": str(sim.model.device),
        "frames": sim.frames,
        "dt": sim.dt,
        "complete_step_capture": sim.capture,
        "setup_seconds": setup_seconds,
        "step_seconds": seconds,
        "iterations": iterations,
        "min_jacobian": float(np.min(determinants)),
        "max_motion_m": float(np.linalg.norm(np.asarray(q) - q[0], axis=-1).max()),
        "max_pinned_error_m": float(np.max(np.abs(np.asarray(q)[:, sim.pinned] - q[0][sim.pinned]))),
        "finite": bool(np.isfinite(q).all() and np.isfinite(v).all()),
    }
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"step_seconds", "iterations"}}, indent=2))


if __name__ == "__main__":
    main()
