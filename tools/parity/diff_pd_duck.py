"""Compare the DiffPD duck example with the executed C++ reference ``pd_eigen`` (development tool).

Port side (``uv run --locked``)::

    uv run --locked python tools/parity/diff_pd_duck.py --out /tmp/diff_pd_duck \
        --reference-root ../../DiffPhD-reference-build --device cuda:0

rolls out ``example_softbody_diff_pd_duck`` and writes a fixture, then runs this
file again with ``--reference-side`` in the reference's own venv (see README.md).
The reference is DiffPhD's build; its ``pd_eigen`` path, ``arc_contact``,
``planar_contact``, gravity and the corotated/volume energies are those of
diff_pd_public 1594aa0 for homogeneous energies applied to all elements.

Both sides use the bundled ``duck.usda`` mesh (rest vertices and tetrahedra as
the port's model holds them) and ``duck_3d.py``'s ``pd_opt`` with ``use_bfgs=1``.

* Stepwise: every reference step starts from the port's float32 state of that
  step. Declared thresholds: max position difference <= 1e-10 of the rest
  bounding-box diagonal, and iteration counts equal or within one.
* Rollout: the reference's own float64 rollout from its float64 initial state
  (``DuckEnv3d.default_init_position`` and ``duck_3d.py``'s ``v0``) against the
  port's float32-state rollout; reported, not thresholded, together with the
  final duck center and the reference loss ``0.5 |center - target|^2``.
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

RELATIVE_POSITION_TOLERANCE = 1.0e-10
ITERATION_TOLERANCE = 1


def port(args):
    import newton.viewer
    import warp as wp

    from newton_solver_diff_phd.examples import example_softbody_diff_pd_duck as duck

    wp.set_device(args.device)
    example = duck.Example(newton.viewer.ViewerNull(num_frames=args.steps), argparse.Namespace())
    solver, model = example.solver, example.model
    record = {key: [] for key in ("q", "v", "x", "iterations", "seconds")}
    for _ in range(args.steps):
        record["q"].append(example.states[0].particle_q.numpy().astype(np.float64))
        record["v"].append(example.states[0].particle_qd.numpy().astype(np.float64))
        wp.synchronize()
        started = time.perf_counter()
        example.step()
        wp.synchronize()
        record["seconds"].append(time.perf_counter() - started)
        record["x"].append(solver.positions.numpy().copy())
        record["iterations"].append(int(solver.iterations.numpy()[0]))
    solver.check_status()
    scene = duck.duck_scene(example.rest)
    arc, planar = (np.asarray(parameters, dtype=np.float64) for _, parameters in scene["state_forces"])
    fixture = args.out / "fixture.npz"
    np.savez(
        fixture,
        rest=example.rest,
        tets=model.tet_indices.numpy().reshape(-1, 4),
        density=duck.DENSITY,
        youngs=duck.YOUNGS,
        poisson=duck.POISSON,
        corotated=scene["corotated_stiffness"],
        volume=scene["volume_stiffness"],
        gravity=np.asarray(scene["gravity"], dtype=np.float64),
        arc=arc,
        planar=planar,
        dt=duck.DT,
        pd_options=json.dumps(duck.PD_OPTIONS),
        q0=scene["q0"],
        v0=scene["v0"],
        target=duck.TARGET,
        **{key: np.asarray(value) for key, value in record.items()},
    )
    root = Path(args.reference_root).resolve()
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(
        [str(root / "RealSim/deps/onetbb_install/lib"), "/usr/local/cuda/lib64", env.get("LD_LIBRARY_PATH", "")]
    )
    command = [str(root / ".venv/bin/python"), "-I", __file__, "--reference-side", "--reference-root", str(root)]
    command += ["--out", str(args.out)]
    with open(args.out / "reference.log", "w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    reference = np.load(args.out / "reference.npz")
    compare(np.load(fixture), reference, args.out / "report.json")


def reference_side(args):
    sys.path.insert(0, str(Path(args.reference_root) / "python"))
    import py_diff_pd.core.py_diff_pd_core as core

    fx = np.load(args.out / "fixture.npz")
    n = 3 * len(fx["rest"])
    with tempfile.TemporaryDirectory() as folder:
        # py_diff_pd.common.tet_mesh.generate_tet_mesh
        path = os.path.join(folder, "duck.bin")
        with open(path, "wb") as f:
            f.write(struct.pack("ii", 3, 4))
            f.write(struct.pack("ii", 3, len(fx["rest"])))
            for axis in range(3):
                f.write(struct.pack(f"{len(fx['rest'])}d", *fx["rest"][:, axis]))
            f.write(struct.pack("ii", 4, len(fx["tets"])))
            for corner in range(4):
                f.write(struct.pack(f"{len(fx['tets'])}i", *fx["tets"][:, corner]))
        mesh = core.TetMesh3d()
        mesh.Initialize(path)
        deformable = core.TetDeformable()
        deformable.Initialize(path, float(fx["density"]), "none", float(fx["youngs"]), float(fx["poisson"]))
    # DuckEnv3d, in its order: gravity, arc_contact, planar_contact, corotated, volume.
    deformable.AddStateForce("gravity", [float(g) for g in fx["gravity"]])
    deformable.AddStateForce("arc_contact", [float(p) for p in fx["arc"]])
    deformable.AddStateForce("planar_contact", [float(p) for p in fx["planar"]])
    deformable.AddPdEnergy("corotated", [float(fx["corotated"])], [])
    deformable.AddPdEnergy("volume", [float(fx["volume"])], [])
    options = core.StdMap()
    # duck_3d.py pd_opt; verbose 2 prints one "PD iteration" line per iteration.
    pd = json.loads(str(fx["pd_options"]))
    for key, value in {**pd, "verbose": 2, "thread_ct": 8, "use_bfgs": 1}.items():
        options[key] = float(value)
    dt = float(fx["dt"])

    def forward(q, v):
        q_next, v_next = core.StdRealVector(n), core.StdRealVector(n)
        print("STEP", flush=True)
        started = time.perf_counter()
        deformable.PyForward(
            "pd_eigen",
            list(q.ravel()),
            list(v.ravel()),
            [],
            [0.0] * n,
            dt,
            options,
            q_next,
            v_next,
            core.StdIntVector(0),
        )
        seconds = time.perf_counter() - started
        return np.array(q_next).reshape(-1, 3), np.array(v_next).reshape(-1, 3), seconds

    stepwise, stepwise_seconds = [], []
    for q, v in zip(fx["q"], fx["v"], strict=True):
        x, _, seconds = forward(q, v)
        stepwise.append(x)
        stepwise_seconds.append(seconds)
    # DuckEnv3d.default_init_position from the reference mesh, and duck_3d.py's v0.
    rest = np.array(mesh.py_vertices()).reshape(-1, 3)
    center, radius = np.array([-0.6, 0, 7.7]), 7.5
    x_offset = np.max([-np.min(rest[:, 0]), np.max(rest[:, 0])])
    radius_adjusted = np.sqrt(radius**2 - x_offset**2)
    c, s = np.cos(np.deg2rad(70.0)), np.sin(np.deg2rad(70.0))
    q0 = rest @ np.array([[s, 0, -c], [0, 1, 0], [c, 0, s]]).T + center + radius_adjusted * np.array([c, 0, -s])
    v0 = np.zeros_like(rest) + np.array([-np.sin(np.deg2rad(70.0)), 0, -np.cos(np.deg2rad(70.0))]) * 10
    rollout, q, v = [], q0, v0
    for _ in range(len(fx["q"])):
        q, v, _ = forward(q, v)
        rollout.append(q)
    sys.stdout.flush()
    np.savez(
        args.out / "reference.npz",
        stepwise=np.asarray(stepwise),
        stepwise_seconds=np.asarray(stepwise_seconds),
        rollout=np.asarray(rollout),
        q0=q0,
        v0=v0,
    )


def iteration_counts(log: Path) -> list[int]:
    counts = []
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith("STEP"):
            counts.append(0)
        elif "PD iteration:" in line:
            counts[-1] += 1
    return counts


def compare(fx, ref, report_path):
    steps = len(fx["q"])
    scale = float(np.linalg.norm(fx["rest"].max(axis=0) - fx["rest"].min(axis=0)))
    counts = iteration_counts(report_path.parent / "reference.log")
    ref_iterations = np.asarray(counts[:steps])
    rollout_iterations = np.asarray(counts[steps : 2 * steps])
    difference = np.abs(ref["stepwise"] - fx["x"]).reshape(steps, -1).max(axis=1)
    step_size = np.abs(ref["stepwise"] - fx["q"]).reshape(steps, -1).max(axis=1)
    iteration_gap = np.abs(ref_iterations - fx["iterations"])
    port_rollout = fx["x"]
    rollout_gap = np.abs(ref["rollout"] - port_rollout).reshape(steps, -1).max(axis=1)
    target = fx["target"]
    centers = {"port": port_rollout[-1].mean(axis=0), "reference": ref["rollout"][-1].mean(axis=0)}
    report = {
        "steps": steps,
        "scale_m": scale,
        "thresholds": {"position_relative": RELATIVE_POSITION_TOLERANCE, "iterations": ITERATION_TOLERANCE},
        "stepwise": {
            "max_position_difference_m": float(difference.max()),
            "max_relative_difference": float(difference.max() / scale),
            "worst_step": int(difference.argmax()),
            "max_step_size_m": float(step_size.max()),
            "iterations_equal_steps": int(np.count_nonzero(iteration_gap == 0)),
            "max_iteration_gap": int(iteration_gap.max()),
            "port_iterations_total": int(np.sum(fx["iterations"])),
            "reference_iterations_total": int(ref_iterations.sum()),
            "passed": bool(
                difference.max() <= RELATIVE_POSITION_TOLERANCE * scale and iteration_gap.max() <= ITERATION_TOLERANCE
            ),
        },
        "rollout": {
            "initial_state_difference_m": float(np.abs(ref["q0"] - fx["q0"]).max()),
            "initial_velocity_difference_m_s": float(np.abs(ref["v0"] - fx["v0"]).max()),
            "max_position_difference_m": float(rollout_gap.max()),
            "position_difference_at_steps_m": {
                str(k): float(rollout_gap[k - 1]) for k in sorted({1, 10, 20, 50, 100, steps}) if k <= steps
            },
            "reference_iterations_total": int(rollout_iterations.sum()),
            "final_center_m": {key: value.tolist() for key, value in centers.items()},
            "target_m": target.tolist(),
            "final_center_distance_to_target_m": {
                key: float(np.linalg.norm(value - target)) for key, value in centers.items()
            },
            "loss_m2": {key: float(0.5 * np.sum((value - target) ** 2)) for key, value in centers.items()},
        },
        "timing_s": {
            "port_steps_total": float(np.sum(fx["seconds"])),
            "port_first_step_including_factor": float(fx["seconds"][0]),
            "reference_steps_total": float(np.sum(ref["stepwise_seconds"])),
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["stepwise"]["passed"]:
        raise SystemExit("Stepwise duck parity exceeded the declared thresholds")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--reference-side", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.reference_side:
        reference_side(args)
    else:
        port(args)


if __name__ == "__main__":
    main()
