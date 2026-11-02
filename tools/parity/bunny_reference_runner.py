"""Execute the original DiffPhD Bunny settings in its compiled native environment.

The input archive is the original offline-prepared tetrahedral asset, including
the original contact subset and material groups. No remeshing or port solver is
used. The native C++ forward/backward API is called with bunny.py's options;
EnvBase's default per-frame adjoint clipping is reproduced and reported.
"""

import argparse
import ast
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

METHOD = "pd_eigen_alg_phd"
OPTIONS = {
    "max_pd_iter": 10000,
    "max_ls_iter": 10,
    "abs_tol": 1e-9,
    "rel_tol": 1e-6,
    "verbose": 0,
    "thread_ct": 8,
    "use_bfgs": 0,
    "use_acc": 1,
    "project_newton": 1,
    "use_abs": 1,
    "aa_window": 5,
    "friction_mu": 0.1,
    "restitution": 0,
    "skip_contact_woodbury": 0,
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_rotations(source):
    tree = ast.parse(source.read_text())
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in ("rpy_to_rotation", "rpy_to_rotation_gradient")
    ]
    scope = {"np": np, "ndarray": lambda value: np.asarray(value, dtype=np.float64)}
    exec(compile(tree, str(source), "exec"), scope)
    return scope["rpy_to_rotation"], scope["rpy_to_rotation_gradient"]


def initial_parameters():
    lower = np.array([-np.pi / 3] * 3 + [-0.03, -0.03, 0.19, -2, -2, -8])
    upper = np.array([np.pi / 3] * 3 + [0.03, 0.03, 0.21, 2, 4, -4])
    return np.random.default_rng(211).uniform(lower, upper), lower, upper


def write_mesh(path, vertices, tets):
    import struct

    with path.open("wb") as stream:
        stream.write(struct.pack("4i", 3, 4, 3, len(vertices)))
        stream.write(np.asarray(vertices.T, dtype="=f8").tobytes())
        stream.write(struct.pack("2i", 4, len(tets)))
        stream.write(np.asarray(tets.T, dtype="=i4").tobytes())


def build(core, asset, workdir, *, homogeneous=False, no_contact=False):
    vertices, tets = asset["vertices"], asset["tets"]
    path = workdir / "mesh.bin"
    write_mesh(path, vertices, tets)
    deformable = core.TetDeformable()
    deformable.Initialize(str(path), 1000.0, "none", 1e6, 0.49)
    mu = 1e6 / (2 * 1.49)
    lame = 1e6 * 0.49 / (1.49 * (1 - 2 * 0.49))
    if homogeneous:
        deformable.AddPdEnergy("corotated", [2 * mu], [])
        deformable.AddPdEnergy("volume", [lame], [])
        energies = 2
    else:
        # Preserve the original registration order, including group membership.
        for group, scale in (("stiff_elements", 10.0), ("soft_elements", 0.1)):
            elements = [int(i) for i in asset[group]]
            group_mu = float(asset["mu"][elements[0]]) if "mu" in asset else scale * mu
            group_lame = float(asset["lambda"][elements[0]]) if "lambda" in asset else scale * lame
            deformable.AddPdEnergy("corotated", [2 * group_mu], elements)
            deformable.AddPdEnergy("volume", [group_lame], elements)
        energies = 4
    deformable.AddStateForce("gravity", [0.0, 0.0, -9.81])
    if not no_contact:
        deformable.SetFrictionalBoundary("planar", [0.0, 0.0, 1.0, 0.0], [int(i) for i in asset["contact_vertices"]])
    return deformable, energies


def rollout(
    core,
    asset,
    workdir,
    x,
    rotation,
    *,
    steps,
    homogeneous=False,
    no_contact=False,
    progress=False,
    checkpoint_path=None,
):
    deformable, energies = build(core, asset, workdir, homogeneous=homogeneous, no_contact=no_contact)
    opt = core.StdMap()
    for name, value in OPTIONS.items():
        opt[name] = float(value)
    rest = asset["vertices"]
    q = ((rest - rest.mean(axis=0)) @ rotation(x[:3]).T + x[3:6]).ravel()
    v = np.broadcast_to(x[6:9], rest.shape).copy().ravel()
    q_history, v_history, active_history = [q.copy()], [v.copy()], [[]]
    force = np.zeros_like(q).tolist()
    core.AlgPhdResetFrameStats()
    seconds = []
    start = time.perf_counter()
    for step in range(steps):
        q_next, v_next = core.StdRealVector(len(q)), core.StdRealVector(len(q))
        active = core.StdIntVector(active_history[-1])
        tick = time.perf_counter()
        deformable.PyForward(METHOD, q.tolist(), v.tolist(), [], force, 0.001, opt, q_next, v_next, active)
        seconds.append(time.perf_counter() - tick)
        q, v = np.array(q_next), np.array(v_next)
        if not np.isfinite(q).all() or not np.isfinite(v).all():
            raise RuntimeError(f"Nonfinite state at step {step + 1}")
        q_history.append(q.copy())
        v_history.append(v.copy())
        active_history.append(list(active))
        if progress and (step < 3 or (step + 1) % 10 == 0):
            print(
                json.dumps({"step": step + 1, "seconds": seconds[-1], "min_z": float(q.reshape(-1, 3)[:, 2].min())}),
                flush=True,
            )
        if checkpoint_path is not None and ((step + 1) % 10 == 0 or step + 1 == steps):
            np.savez_compressed(
                checkpoint_path,
                q=np.array(q_history).reshape(-1, len(rest), 3),
                v=np.array(v_history).reshape(-1, len(rest), 3),
                iterations=np.array(core.AlgPhdFrameIters()),
                converged=np.array(core.AlgPhdFrameConverged()) == 1,
                forward_seconds=np.array(seconds),
                completed_steps=step + 1,
                requested_steps=steps,
                x=x,
            )
    com_error = q.reshape(-1, 3).mean(axis=0) - np.array([0.15, 0.15, 0.15])
    loss = 0.5 * com_error.dot(com_error) / 0.1**2
    result = {
        "q": np.array(q_history).reshape(-1, len(rest), 3),
        "v": np.array(v_history).reshape(-1, len(rest), 3),
        "loss": float(loss),
        "forward_seconds": np.array(seconds),
        "forward_elapsed_seconds": time.perf_counter() - start,
        "iterations": np.array(core.AlgPhdFrameIters()),
        "converged": np.array(core.AlgPhdFrameConverged()) == 1,
        "active_count": np.array([len(active) for active in active_history]),
    }
    return result, (deformable, energies, opt, q_history, v_history, active_history, force)


def backward(core, asset, native, result, x, rotation_gradient, *, clip_norm=1000.0):
    deformable, energies, opt, qs, vs, active, force = native
    size = len(qs[0])
    error = result["q"][-1].mean(axis=0) - np.array([0.15, 0.15, 0.15])
    dq = np.tile(error / len(asset["vertices"]) / 0.1**2, len(asset["vertices"]))
    dv = np.zeros(size)
    timings, clips, caps, suppressed = [], [], [], []
    tet = asset["vertices"][asset["tets"]]
    mass = 1000 * np.abs(np.linalg.det(tet[:, 1:] - tet[:, :1])).mean() / 6
    for step in reversed(range(len(qs) - 1)):
        outputs = [core.StdRealVector(size), core.StdRealVector(size), core.StdRealVector(0), core.StdRealVector(size)]
        outputs += [core.StdRealVector(energies), core.StdRealVector(0), core.StdRealVector(3)]
        incoming_rhs_norm = np.linalg.norm(dq + dv / 0.001)
        tick = time.perf_counter()
        deformable.PyBackward(
            METHOD,
            qs[step].tolist(),
            vs[step].tolist(),
            [],
            force,
            0.001,
            qs[step + 1].tolist(),
            vs[step + 1].tolist(),
            core.StdIntVector(active[step + 1]),
            dq.tolist(),
            dv.tolist(),
            opt,
            *outputs,
        )
        timings.append(time.perf_counter() - tick)
        dq, dv, _da, df, _dmat, _dact, _dstate = [np.array(output) for output in outputs]
        caps.append(bool(mass / 0.001**2 * np.linalg.norm(df) >= (1 - 1e-9) * (incoming_rhs_norm + 1)))
        suppressed.append(bool(incoming_rhs_norm > 0 and not np.any(df)))
        clipped = []
        for gradient in (dq, dv):
            nonfinite = not np.isfinite(gradient).all()
            if clip_norm is not None:
                np.nan_to_num(gradient, copy=False, nan=0.0, posinf=clip_norm, neginf=-clip_norm)
                norm = np.linalg.norm(gradient)
                changed = nonfinite or norm > clip_norm
                if norm > clip_norm:
                    gradient *= clip_norm / norm
            else:
                changed = False
            clipped.append(changed)
        clips.append(clipped)
    offset = asset["vertices"] - asset["vertices"].mean(axis=0)
    gradient_x = np.zeros(9)
    for i, derivative in enumerate(rotation_gradient(x[:3])):
        gradient_x[i] = (offset @ derivative.T).ravel().dot(dq)
    gradient_x[3:6] = dq.reshape(-1, 3).sum(axis=0)
    gradient_x[6:9] = dv.reshape(-1, 3).sum(axis=0)
    return {
        "gradient_x": gradient_x,
        "gradient_q0": dq.reshape(-1, 3),
        "gradient_v0": dv.reshape(-1, 3),
        "backward_seconds": np.array(timings[::-1]),
        "python_adjoint_clipped": np.array(clips[::-1]),
        "native_cascade_cap": np.array(caps[::-1]),
        "native_backbone_suppressed": np.array(suppressed[::-1]),
    }


def matching_input_replay(core, asset, trace_path, steps, out):
    """Replay selected CUDA input states with its actual stored model precision.

    Every replay starts a fresh native instance and zero warm contact forces.
    The comparison therefore reports fixed-point agreement, not equal iteration
    histories. The native rest inverse is rebuilt from the port's rest points;
    Newton's stored float32 tet inverse remains a documented representation gap.
    """
    with np.load(trace_path, allow_pickle=False) as archive:
        trace = {name: archive[name].copy() for name in archive.files}
    asset = {name: value.copy() for name, value in asset.items()}
    asset["vertices"] = trace["rest_positions"].astype(np.float64)
    asset["tets"] = trace["tets"]
    asset["mu"], asset["lambda"] = trace["tet_materials"][:, 0], trace["tet_materials"][:, 1]
    opt = core.StdMap()
    for name, value in OPTIONS.items():
        opt[name] = float(value)
    outputs_q, outputs_v, rows = [], [], []
    for step in steps:
        with tempfile.TemporaryDirectory(prefix="diffphd-bunny-replay-") as temporary:
            deformable, _energies = build(core, asset, Path(temporary))
            q, v = trace["q"][step - 1].ravel(), trace["v"][step - 1].ravel()
            q_next, v_next = core.StdRealVector(len(q)), core.StdRealVector(len(q))
            active = core.StdIntVector(0)
            core.AlgPhdResetFrameStats()
            start = time.perf_counter()
            deformable.PyForward(
                METHOD, q.tolist(), v.tolist(), [], np.zeros_like(q).tolist(), 0.001, opt, q_next, v_next, active
            )
            elapsed = time.perf_counter() - start
            q_next, v_next = np.array(q_next).reshape(-1, 3), np.array(v_next).reshape(-1, 3)
            outputs_q.append(q_next)
            outputs_v.append(v_next)
            position_error = q_next - trace["q"][step]
            velocity_error = v_next - trace["v"][step]
            row = {
                "step": step,
                "seconds": elapsed,
                "converged": bool(np.array(core.AlgPhdFrameConverged()).all()),
                "iterations": list(core.AlgPhdFrameIters()),
                "max_nodal_position_error_m": float(np.linalg.norm(position_error, axis=1).max()),
                "rms_nodal_position_error_m": float(np.sqrt(np.mean(np.sum(position_error**2, axis=1)))),
                "max_nodal_velocity_error_m_s": float(np.linalg.norm(velocity_error, axis=1).max()),
                "com_error_m": np.mean(position_error, axis=0).tolist(),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            np.savez_compressed(out, steps=np.array(steps[: len(rows)]), q=np.array(outputs_q), v=np.array(outputs_v))
            out.with_suffix(".json").write_text(
                json.dumps(
                    {"scope": matching_input_replay.__doc__, "trace_sha256": digest(trace_path), "rows": rows}, indent=2
                )
                + "\n"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--homogeneous", action="store_true")
    parser.add_argument("--no-contact", action="store_true")
    parser.add_argument("--fd", default="", help="Comma-separated initial parameter indices, or all")
    parser.add_argument("--fd-epsilon", type=float, default=1e-6)
    parser.add_argument("--replay", type=Path, help="Replay port trace input states instead of an independent rollout")
    parser.add_argument("--replay-steps", default="1,20,22,30,60,100")
    args = parser.parse_args()
    sys.path.insert(0, str(args.reference_root / "python"))
    from py_diff_pd.core import py_diff_pd_core as core

    with np.load(args.asset, allow_pickle=False) as archive:
        asset = {key: archive[key].copy() for key in archive.files}
    if args.replay is not None:
        matching_input_replay(core, asset, args.replay, [int(step) for step in args.replay_steps.split(",")], args.out)
        return
    rotation, rotation_gradient = original_rotations(args.source_root / "python/py_diff_pd/common/common.py")
    x, lower, upper = initial_parameters()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    paths = [args.asset, args.reference_root / "python/py_diff_pd/core/_py_diff_pd_core.so"]
    paths += [
        args.source_root / item
        for item in (
            "python/example/bunny.py",
            "python/py_diff_pd/env/bunny_env.py",
            "python/py_diff_pd/env/env_base.py",
            "python/py_diff_pd/common/common.py",
        )
    ]
    report = {
        "scope": __doc__,
        "method": METHOD,
        "options": OPTIONS,
        "sources_sha256": {str(path): digest(path) for path in paths},
        "steps": args.steps,
        "dt": 0.001,
        "homogeneous": args.homogeneous,
        "no_contact": args.no_contact,
        "x": x.tolist(),
    }
    with tempfile.TemporaryDirectory(prefix="diffphd-bunny-native-") as temporary:
        workdir = Path(temporary)
        result, native = rollout(
            core,
            asset,
            workdir,
            x,
            rotation,
            steps=args.steps,
            homogeneous=args.homogeneous,
            no_contact=args.no_contact,
            progress=True,
            checkpoint_path=args.out.with_name(args.out.stem + "-forward-checkpoint.npz"),
        )
        result.update(x=x, lower=lower, upper=upper)
        # Preserve a completed forward baseline even if an optional subsequent
        # backward audit or finite-difference evaluation is interrupted.
        np.savez_compressed(args.out, **result)
        if not args.forward_only:
            result.update(backward(core, asset, native, result, x, rotation_gradient))
        del native
        np.savez_compressed(args.out, **result)
        report.update(
            loss=result["loss"],
            final_com=result["q"][-1].mean(axis=0).tolist(),
            forward_seconds=float(result["forward_seconds"].sum()),
            all_converged=bool(result["converged"].all()),
            max_iterations=int(result["iterations"].max()),
        )
        if not args.forward_only:
            report.update(
                gradient_x=result["gradient_x"].tolist(),
                backward_seconds=float(result["backward_seconds"].sum()),
                python_clipped_steps=int(result["python_adjoint_clipped"].any(axis=1).sum()),
                native_capped_steps=int(result["native_cascade_cap"].sum()),
                native_suppressed_steps=int(result["native_backbone_suppressed"].sum()),
            )
        args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        indices = list(range(9)) if args.fd == "all" else [int(index) for index in args.fd.split(",") if index]
        finite_differences = []
        for index in indices:
            losses = []
            seconds = []
            for sign in (-1, 1):
                perturbed = x.copy()
                perturbed[index] += sign * args.fd_epsilon
                value, native = rollout(
                    core,
                    asset,
                    workdir,
                    perturbed,
                    rotation,
                    steps=args.steps,
                    homogeneous=args.homogeneous,
                    no_contact=args.no_contact,
                )
                losses.append(value["loss"])
                seconds.append(float(value["forward_seconds"].sum()))
                del native
            fd = (losses[1] - losses[0]) / (2 * args.fd_epsilon)
            row = {
                "parameter": index,
                "epsilon": args.fd_epsilon,
                "loss_minus": losses[0],
                "loss_plus": losses[1],
                "finite_difference": float(fd),
                "seconds": seconds,
            }
            if "gradient_x" in result:
                row.update(
                    adjoint=float(result["gradient_x"][index]),
                    absolute_error=abs(float(fd - result["gradient_x"][index])),
                )
            finite_differences.append(row)
            report["finite_differences"] = finite_differences
            args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
