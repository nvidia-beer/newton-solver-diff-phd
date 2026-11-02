"""Record every state of the public Bunny scene for native-reference comparison."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp

from newton_solver_diff_phd.examples._diff_phd_bunny import (
    ASSET,
    DT,
    FRAMES,
    BunnySimulation,
    landing_loss,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--asset", type=Path, default=ASSET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-capture", action="store_true")
    args = parser.parse_args()
    started = perf_counter()
    simulation = BunnySimulation(
        asset=args.asset,
        device=args.device,
        capture=not args.no_capture,
    )
    wp.synchronize_device(simulation.model.device)
    construction = perf_counter() - started
    positions = [simulation.states[0].particle_q.numpy()]
    velocities = [simulation.states[0].particle_qd.numpy()]
    seconds, iterations, active_contacts = [], [], []
    for step in range(FRAMES):
        started = perf_counter()
        state = simulation.advance()
        wp.synchronize_device(simulation.model.device)
        seconds.append(perf_counter() - started)
        positions.append(state.particle_q.numpy())
        velocities.append(state.particle_qd.numpy())
        iterations.append(int(simulation.solver.iterations.numpy()[0]))
        active_contacts.append(int(np.count_nonzero(simulation.solver.contact_multipliers.numpy() > 0)))
        if (step + 1) % 10 == 0:
            print(f"Bunny {step + 1}/{FRAMES}: {seconds[-1]:.3f}s, {iterations[-1]} iterations", flush=True)
    positions, velocities = np.asarray(positions), np.asarray(velocities)
    rest = simulation.vertices[simulation.tets]
    rest_det = np.linalg.det(rest[:, 1:] - rest[:, :1])
    corners = positions.astype(np.float64)[:, simulation.tets]
    determinants = np.linalg.det(corners[:, :, 1:] - corners[:, :, :1]) / rest_det
    boundary_nodes = np.unique(simulation.faces)
    report = {
        "device": str(simulation.model.device),
        "capture": simulation.capture,
        "frames": FRAMES,
        "dt": DT,
        "parameters": simulation.parameters.tolist(),
        "construction_seconds": construction,
        "step_seconds": seconds,
        "setup_and_first_step_seconds": construction + seconds[0],
        "remaining_99_steps_seconds": sum(seconds[1:]),
        "iterations": iterations,
        "active_contacts": active_contacts,
        "final_center": positions[-1].astype(np.float64).mean(axis=0).tolist(),
        "loss": landing_loss(positions[-1]),
        "minimum_J": float(determinants.min()),
        "minimum_J_per_step": determinants.min(axis=1).tolist(),
        "inverted_elements": int(np.count_nonzero(determinants <= 0)),
        "finite": bool(np.isfinite(positions).all() and np.isfinite(velocities).all()),
        "minimum_registered_contact_z": float(positions[:, simulation.contact_vertices, 2].min()),
        "minimum_boundary_z": float(positions[:, boundary_nodes, 2].min()),
        "registered_contact_vertices": len(simulation.contact_vertices),
        "boundary_vertices": len(boundary_nodes),
        "precision": "Newton float32 states/materials/rest geometry; double solver algebra",
        "timing_scope": "Synchronized live steps, including convergence status checks, excluding trace copies/rendering",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        args.output / "trace.npz",
        q=positions,
        v=velocities,
        parameters=simulation.parameters,
        rest_positions=simulation.model.particle_q.numpy(),
        tet_materials=simulation.model.tet_materials.numpy(),
        tet_poses=simulation.model.tet_poses.numpy(),
        tets=simulation.tets,
    )
    if not report["finite"] or report["inverted_elements"]:
        raise AssertionError("The complete bunny landing contains an invalid state; inspect summary.json")
    print(json.dumps({key: report[key] for key in ("loss", "minimum_J", "remaining_99_steps_seconds")}))


if __name__ == "__main__":
    main()
