# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run the original native Hex8 DiffPhD cantilever and compare saved port states.

The default executes the released CantileverEnv3d constructor unchanged, with
cantilever_3d.py's rest state, axial load, muscle energies, plane vertex energies,
25 steps and dt=0.01. --motion twist changes the initial condition and removes
axial loading; it uses 100 steps, dt=0.001 and relative tolerance 1e-6. The
released Python script does not prescribe that twist or its numerical settings.

Only source definitions needed for physics are evaluated, avoiding optional
rendering dependencies. --port-trace rebuilds the same native model with the
port's stored rest/material precision. --replay compares identical input states.
--float32-states rounds native output q/v after every step to diagnose the
precision of Newton's public state arrays; this is not the original native run.
"""

import argparse
import ast
import hashlib
import json
import os
import shutil
import struct
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

METHOD = "pd_eigen_alg_phd"
OPTIONS = {
    "max_pd_iter": 5000,
    "max_ls_iter": 10,
    "abs_tol": 1e-9,
    "rel_tol": 1e-4,
    "verbose": 0,
    "thread_ct": 4,
    "use_bfgs": 0,
    "use_acc": 1,
    "project_newton": 1,
    "use_abs": 1,
    "aa_window": 5,
}
CORNER_BITS = np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(2)])


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_definition(path, names, namespace):
    tree = ast.parse(path.read_text(), filename=str(path))
    definitions = [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    if len(definitions) != len(names):
        raise ValueError(f"Missing original definitions {names} in {path}")
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)


def original_environment(core, source_root, workdir):
    """Evaluate unchanged original physics definitions, without renderer imports."""
    namespace = {
        "np": np,
        "Path": Path,
        "os": os,
        "shutil": shutil,
        "struct": struct,
        "HexMesh3d": core.HexMesh3d,
        "HexDeformable": core.HexDeformable,
    }
    common = source_root / "python/py_diff_pd/common"
    env = source_root / "python/py_diff_pd/env"
    source_definition(common / "common.py", {"ndarray", "create_folder"}, namespace)
    source_definition(common / "hex_mesh.py", {"generate_hex_mesh"}, namespace)
    source_definition(env / "env_base.py", {"EnvBase"}, namespace)
    source_definition(env / "cantilever_env_3d.py", {"CantileverEnv3d"}, namespace)
    original = namespace["CantileverEnv3d"](
        42, workdir, {"refinement": 5, "state_force_parameters": [0, 0, 0], "heterogeneous_stiffness": True, "spp": 16}
    )
    mesh = core.HexMesh3d()
    mesh.Initialize(str(workdir / "mesh.bin"))
    asset = {
        "vertices": np.array(mesh.py_vertices()).reshape(-1, 3),
        "hexes": np.array([mesh.py_element(i) for i in range(mesh.NumOfElements())]),
        "q0": original.default_init_position().reshape(-1, 3),
        "v0": original.default_init_velocity().reshape(-1, 3),
        "pinned": np.arange(36, dtype=np.int32),
        "right": np.array(original.max_x_nodes(), dtype=np.int32),
        "stiff_elements": np.arange(7 * 25, 13 * 25, dtype=np.int32),
        "soft_elements": np.r_[np.arange(7 * 25), np.arange(13 * 25, 500)].astype(np.int32),
        "muscle_elements": np.arange(20, dtype=np.int32) * 25,
        "collision_vertices": np.array([720, 750], dtype=np.int32),
    }
    return original.deformable(), asset


def build(core, asset, workdir):
    """Rebuild native HexDeformable with shared numerical inputs and original energies."""
    vertices, hexes = asset["vertices"], asset["hexes"]
    path = workdir / "shared.bin"
    with path.open("wb") as stream:
        stream.write(struct.pack("4i", 3, 8, 3, len(vertices)))
        stream.write(np.asarray(vertices.T, dtype="=f8").tobytes())
        stream.write(struct.pack("2i", 8, len(hexes)))
        stream.write(np.asarray(hexes.T, dtype="=i4").tobytes())
    deformable = core.HexDeformable()
    deformable.Initialize(str(path), 1000.0, "none", 1e6, 0.45)
    for node in asset["pinned"]:
        for axis in range(3):
            deformable.SetDirichletBoundaryCondition(int(3 * node + axis), float(asset["q0"][node, axis]))
    deformable.AddStateForce("gravity", [0.0, 0.0, 0.0])
    poisson = 0.45
    base_mu = 1e6 / (2 * (1 + poisson))
    base_lame = 1e6 * poisson / ((1 + poisson) * (1 - 2 * poisson))
    for name, scale in (("stiff_elements", 10), ("soft_elements", 1)):
        members = asset[name].astype(int).tolist()
        mu = float(asset["mu"][members[0]]) if "mu" in asset else scale * base_mu
        lame = float(asset["lambda"][members[0]]) if "lambda" in asset else scale * base_lame
        deformable.AddPdEnergy("corotated", [2 * mu], members)
        deformable.AddPdEnergy("volume", [lame], members)
    deformable.AddPdEnergy("planar_collision", [100.0, 0.0, 0.0, 1.0, 0.1], asset["collision_vertices"].tolist())
    deformable.AddActuation(1e5, [1.0, 0.0, 0.0], asset["muscle_elements"].tolist())
    return deformable


def initial_twist(vertices, degrees=30):
    minimum, maximum = vertices.min(axis=0), vertices.max(axis=0)
    theta = np.deg2rad(degrees) * (vertices[:, 0] - minimum[0]) / (maximum[0] - minimum[0])
    center = (minimum[1:] + maximum[1:]) / 2
    offset = vertices[:, 1:] - center
    result = vertices.copy()
    result[:, 1] = center[0] + np.cos(theta) * offset[:, 0] - np.sin(theta) * offset[:, 1]
    result[:, 2] = center[1] + np.sin(theta) * offset[:, 0] + np.cos(theta) * offset[:, 1]
    return result


def hex_quadrature(vertices, hexes):
    """Original global-dx gradients and eight equal sample volumes, in source order."""
    cell = vertices[hexes]
    edges = np.stack([cell[:, 1] - cell[:, 0], cell[:, 2] - cell[:, 0], cell[:, 4] - cell[:, 0]], axis=2)
    # Match the native serial mean and std::pow convention.
    mean_volume = sum(abs(float(np.linalg.det(edge))) for edge in edges) / len(hexes)
    dx = mean_volume ** (1.0 / 3.0)
    points = ((CORNER_BITS - 0.5) / np.sqrt(3.0) + 0.5) * dx
    gradients = np.zeros((8, 8, 3))
    for sample, point in enumerate(points):
        normalized = point / dx
        for node, bits in enumerate(CORNER_BITS):
            factors = np.where(bits, normalized, 1.0 - normalized)
            for axis in range(3):
                other = [index for index in range(3) if index != axis]
                gradients[sample, node, axis] = (2 * bits[axis] - 1) * factors[other].prod() / dx
    return gradients, mean_volume / 8


def applied_force(asset, motion, step, force_steps=25):
    force = np.zeros_like(asset["vertices"])
    if motion == "axial":
        force[asset["right"]] = np.array([1, 0, 0]) * (step - 1) / force_steps * 80
    return force


def one_step(core, deformable, options, q, v, force, dt):
    q_next, v_next, active = core.StdRealVector(q.size), core.StdRealVector(q.size), core.StdIntVector(0)
    start = time.perf_counter()
    deformable.PyForward(
        METHOD,
        q.ravel().tolist(),
        v.ravel().tolist(),
        np.zeros(deformable.act_dofs()).tolist(),
        force.ravel().tolist(),
        dt,
        options,
        q_next,
        v_next,
        active,
    )
    return np.array(q_next).reshape(-1, 3), np.array(v_next).reshape(-1, 3), time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, help="Default: reference-root")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--asset", type=Path, help="Optional shared Hex8 NPZ; defaults to original source generator")
    parser.add_argument("--export-geometry", type=Path)
    parser.add_argument("--motion", choices=("axial", "twist"), default="axial")
    parser.add_argument("--steps", type=int, help="Default: 25 axial, 100 twist")
    parser.add_argument("--dt", type=float, help="Default: 0.01 axial, 0.001 twist")
    parser.add_argument("--relative-tolerance", type=float, help="Default: 1e-4 axial, 1e-6 twist")
    parser.add_argument("--port-trace", type=Path)
    parser.add_argument("--float32-states", action="store_true", help="Round native output q/v to float32 each step")
    parser.add_argument("--replay", default="", help="Comma-separated one-based steps from --port-trace")
    args = parser.parse_args()
    args.steps = args.steps if args.steps is not None else (25 if args.motion == "axial" else 100)
    if args.dt is None:
        args.dt = 0.01 if args.motion == "axial" else 0.001
    if args.relative_tolerance is None:
        args.relative_tolerance = 1e-4 if args.motion == "axial" else 1e-6
    if args.steps < 1 or args.dt <= 0:
        parser.error("steps and dt must be positive")
    source_root = args.source_root or args.reference_root
    trace = load(args.port_trace) if args.port_trace else None
    replay_steps = [int(value) for value in args.replay.split(",") if value]
    if replay_steps and (trace is None or min(replay_steps) < 1 or max(replay_steps) >= len(trace["q"])):
        parser.error("--replay requires valid one-based steps in --port-trace")
    if args.float32_states and trace is None:
        parser.error("--float32-states requires --port-trace for shared initial states and pinned values")
    sys.path.insert(0, str(args.reference_root / "python"))
    from py_diff_pd.core import py_diff_pd_core as core

    chosen_options = OPTIONS | {"rel_tol": args.relative_tolerance}
    options = core.StdMap()
    for key, value in chosen_options.items():
        options[key] = float(value)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": __doc__,
        "method": METHOD,
        "options": chosen_options,
        "motion": args.motion,
        "native_binary_sha256": digest(args.reference_root / "python/py_diff_pd/core/_py_diff_pd_core.so"),
        "source_hashes": {
            name: digest(source_root / name)
            for name in (
                "python/example/cantilever_3d.py",
                "python/py_diff_pd/env/cantilever_env_3d.py",
                "python/py_diff_pd/common/hex_mesh.py",
                "cpp/core/src/fem/hex_deformable.cpp",
            )
        },
        "port_trace_sha256": digest(args.port_trace) if args.port_trace else None,
        "asset_sha256": digest(args.asset) if args.asset else None,
        "float32_state_rounding": args.float32_states,
        "dt": args.dt,
        "requested_steps": args.steps,
        "rows": [],
    }
    with tempfile.TemporaryDirectory(prefix="diffphd-cantilever-native-") as temporary:
        workdir = Path(temporary)
        deformable, asset = original_environment(core, source_root, workdir)
        if args.asset:
            asset.update(load(args.asset))
        if args.motion == "twist":
            asset["q0"] = initial_twist(asset["vertices"])
        if trace is not None:
            asset.update(vertices=trace["rest_positions"].astype(float), hexes=trace["hexes"])
            materials = trace["hex_materials"] if "hex_materials" in trace else trace["materials"]
            asset.update(mu=materials[:, 0], **{"lambda": materials[:, 1]})
            asset["q0"], asset["v0"] = trace["q"][0].astype(float), trace["v"][0].astype(float)
        if args.asset or trace is not None:
            deformable = build(core, asset, workdir)
        if args.export_geometry:
            args.export_geometry.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.export_geometry, **asset)
        report.update(
            vertices=len(asset["vertices"]),
            hexes=len(asset["hexes"]),
            pinned=len(asset["pinned"]),
            actuation_dofs=deformable.act_dofs(),
            original_source_constructor=not (args.asset or trace is not None),
        )
        gradients, sample_volume = hex_quadrature(asset["vertices"], asset["hexes"])
        report.update(sample_volume=sample_volume, uniform_nodal_mass=sample_volume * 8 * 1000)
        output_q, output_v = [asset["q0"].copy()], [asset["v0"].copy()]
        core.AlgPhdResetFrameStats()
        for step in replay_steps or range(1, args.steps + 1):
            q, v = (trace["q"][step - 1], trace["v"][step - 1]) if replay_steps else (output_q[-1], output_v[-1])
            force = (
                trace["forces"][step - 1]
                if trace is not None and "forces" in trace
                else applied_force(asset, args.motion, step)
            )
            q_next, v_next, elapsed = one_step(core, deformable, options, q, v, force, args.dt)
            if args.float32_states:
                q_next = q_next.astype(np.float32).astype(np.float64)
                v_next = v_next.astype(np.float32).astype(np.float64)
            output_q.append(q_next)
            output_v.append(v_next)
            deformation = np.einsum("eni,snj->esij", q_next[asset["hexes"]], gradients)
            row = {
                "step": step,
                "seconds": elapsed,
                "iterations": int(core.AlgPhdFrameIters()[-1]),
                "converged": bool(core.AlgPhdFrameConverged()[-1]),
                "min_gauss_j": float(np.linalg.det(deformation).min()),
                "max_pinned_error_m": float(np.max(np.abs(q_next[asset["pinned"]] - asset["q0"][asset["pinned"]]))),
            }
            if trace is not None:
                difference = q_next - trace["q"][step]
                row.update(
                    max_nodal_position_error_m=float(np.linalg.norm(difference, axis=1).max()),
                    rms_nodal_position_error_m=float(np.sqrt(np.mean(np.sum(difference**2, axis=1)))),
                    max_nodal_velocity_error_m_s=float(np.linalg.norm(v_next - trace["v"][step], axis=1).max()),
                )
            report["rows"].append(row)
            print(json.dumps(row), flush=True)
            np.savez_compressed(
                args.out,
                q=np.array(output_q),
                v=np.array(output_v),
                steps=np.array([item["step"] for item in report["rows"]]),
            )
            args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
            if not row["converged"] or not np.isfinite(q_next).all() or row["min_gauss_j"] <= 0:
                report["failure"] = row
                break
    report["complete"] = len(report["rows"]) == len(replay_steps or range(args.steps)) and "failure" not in report
    report["forward_seconds"] = sum(row["seconds"] for row in report["rows"])
    report["min_gauss_j"] = min(row["min_gauss_j"] for row in report["rows"])
    report["max_iterations"] = max(row["iterations"] for row in report["rows"])
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2), flush=True)
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
