"""Compare port and executed-reference parity results and write a JSON report.

    uv run --locked python tools/parity/compare.py --results OUT \
        --reference-root ../../DiffPhD-reference-build --report docs/validation/parity-reference-cuda.json

Thresholds are fixed in THRESHOLDS before any comparison and are written into
the report. Fixtures are rejected before comparison when their schema, units,
topology or definitions are not ones both sides implement.
"""

import argparse
import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path

import numpy as np

SCHEMA = "newton.diff_phd_parity.v1"
REFERENCE_REVISION = "d319435373a8a94aac2035bfe41f16f49bc1d2a1"
U32 = 2.0**-24  # unit roundoff of the port's float32 states and gradient outputs

THRESHOLDS = {
    "positions": "||q_port - q_ref||_2 <= 10 (rel_tol ||q_ref||_2 + abs_tol) + 10 k u32 (||q_ref||_2 + dt ||v_ref||_2), "
    "k = number of earlier steps whose output the port stored in float32 (0 for single steps)",
    "velocities": "position bound / dt",
    "iterations": "|it_port - it_ref| <= 1 per step (Anderson roundoff can flip a gate)",
    "converged": "equal per step",
    "active_set": "equal per step (lambda > 0)",
    "multipliers": "|lambda_port - lambda_ref| <= 5e-6 |lambda_ref| + 1e-6 max|lambda_ref| + 1e-12 "
    "(reference trace prints 6 significant digits)",
    "gaps": "max |gap_port - gap_ref| <= position bound",
    "gradients_exact": "||g_port - g_ref||_2 <= 1e-5 ||g_ref||_2 on free particles (dL/dq, dL/dv, each step's "
    "dL/df_ext, dL/du); no contact, homogeneous or single-energy per-element materials",
    "gradients_contact": "||g_port - g_ref||_2 <= 1e-4 ||g_ref||_2 (contact Woodbury: capped W_T fixed points)",
    "materials": "per reference energy: |g_port - g_ref| <= tol |g_ref| + steps u32 sum_e |g_port,e| "
    "(port returns float32 per-element values); tol as for the regime",
    "mesh_contact": "reference mesh_contact state force (zero backward on both sides): judged by the position, "
    "velocity, iteration and exact-regime gradient thresholds above; per-step forces and pair counts are reported",
    "heterogeneous_grouped": "gradients reported, not judged: different reference energy encoding "
    "(port uses one energy per family; grouped reference creates separate energies per material group)",
}


def _free_norm(a, free):
    return float(np.linalg.norm(np.asarray(a)[..., free, :]))


def _relative(error, scale):
    return error / scale if scale > 0 else (0.0 if error == 0 else float("inf"))


def _rest_inverse(fx):
    p = fx["rest_vertices"][fx["tets"]]
    return np.linalg.inv(np.stack([p[:, j] - p[:, 0] for j in (1, 2, 3)], axis=-1))


def reject(fx):
    """Reasons a fixture cannot be compared; empty when compatible."""
    reasons = []
    if str(fx.get("schema")) != SCHEMA:
        reasons.append("schema")
    if fx["tets"].ndim != 2 or fx["tets"].shape[1] != 4:
        reasons.append("only Tet4 topology")
    p = fx["rest_vertices"][fx["tets"]]
    if np.any(np.linalg.det(np.stack([p[:, j] - p[:, 0] for j in (1, 2, 3)], axis=-1)) <= 0):
        reasons.append("tetrahedra must be positively oriented")
    if str(fx["material"]) not in ("pd", "neo_hookean"):
        reasons.append("material")
    if str(fx["contact_kind"]) not in ("none", "planar", "spherical", "mesh"):
        reasons.append("contact kind")
    if "mesh_contact_parameters" in fx:
        parameters = fx["mesh_contact_parameters"]
        faces = [fx["mesh_contact_faces_a"], fx["mesh_contact_faces_b"]]
        if parameters.shape != (5,) or not np.all(np.isfinite(parameters)) or np.any(parameters < 0):
            reasons.append("mesh_contact parameters (contact_radius, kn_left, kn_right, kf, mu)")
        if any(f.ndim != 2 or f.shape[1] != 3 or not len(f) or f.min() < 0 or f.max() >= len(fx["q"]) for f in faces):
            reasons.append("mesh_contact faces")
    if "activations" in fx and "a = 1 - u" not in str(fx["activation_convention"]):
        reasons.append("activation convention")
    if "dirichlet_positions" in fx:
        targets = fx["dirichlet_positions"]
        if targets.shape != (int(fx["steps"]), len(fx["q"]), 3) or not np.all(np.isfinite(targets)):
            reasons.append("dirichlet_positions shape or finite values")
        elif not np.array_equal(targets, targets.astype(np.float32).astype(np.float64)):
            reasons.append("dirichlet_positions is not float32-representable")
    if "velocity_damping" in fx:
        damping = np.asarray(fx["velocity_damping"])
        if damping.size != 1 or not np.all(np.isfinite(damping)) or not np.all((0 <= damping) & (damping <= 1)):
            reasons.append("velocity_damping must be a finite scalar in [0,1]")
    for key in ("rest_vertices", "q", "v", "f_ext", "mu", "lambda", "gravity"):
        if not np.array_equal(fx[key], fx[key].astype(np.float32).astype(np.float64)):
            reasons.append(f"{key} is not float32-representable (Newton stores float32)")
    return reasons


def _check(name, value, bound, verdict=None):
    passed = bool(value <= bound)
    return {
        "metric": name,
        "max_error": float(value),
        "threshold": float(bound),
        "pass": passed if verdict is None else verdict,
    }


def _gradient_regime(fx, any_active):
    heterogeneous = np.ptp(fx["mu"]) > 0 or np.ptp(fx["lambda"]) > 0
    if fx["reference_material_encoding"] == "grouped" and heterogeneous:
        return "documented_deviation"
    if any_active:
        return "contact"
    return "reference" if heterogeneous else "exact"


def compare_case(fx, port, ref):
    steps, dt = int(fx["steps"]), float(fx["dt"])
    abs_tol, rel_tol = float(fx["abs_tol"]), float(fx["rel_tol"])
    free = ~fx["dirichlet"]
    contact = str(fx["contact_kind"]) != "none"
    any_active = contact and bool(ref["active"].any())
    regime = _gradient_regime(fx, any_active)
    tol = 1.0e-4 if any_active else 1.0e-5
    checks = []
    for k in range(steps):
        q, v = ref["q_next"][k], ref["v_next"][k]
        bound = 10.0 * (rel_tol * np.linalg.norm(q) + abs_tol) + 10.0 * k * U32 * (
            np.linalg.norm(q) + dt * np.linalg.norm(v)
        )
        checks.append(_check(f"step {k} positions", np.linalg.norm(port["q_next"][k] - q), bound))
        checks.append(_check(f"step {k} velocities", np.linalg.norm(port["v_next"][k] - v), bound / dt))
        checks.append(_check(f"step {k} iterations", abs(int(port["iterations"][k]) - int(ref["iterations"][k])), 1))
        same = bool(port["converged"][k]) == bool(ref["converged"][k])
        checks.append(_check(f"step {k} converged ({bool(ref['converged'][k])})", 0.0 if same else 1.0, 0.0))
        if contact:
            mismatch = int(np.sum(port["active"][k] != ref["active"][k]))
            checks.append(_check(f"step {k} active set ({int(ref['active'][k].sum())} active)", mismatch, 0))
            lam_ref, lam_port = ref["multipliers"][k], port["multipliers"][k]
            allowed = 5.0e-6 * np.abs(lam_ref) + 1.0e-6 * np.abs(lam_ref).max() + 1.0e-12
            excess = np.max(np.abs(lam_port - lam_ref) / allowed)
            checks.append(
                {
                    "metric": f"step {k} multipliers",
                    "max_error": float(np.abs(lam_port - lam_ref).max()),
                    "threshold": float(allowed.max()),
                    "max_error_over_threshold": float(excess),
                    "pass": bool(excess <= 1.0),
                }
            )
            if "gaps" in ref:
                checks.append(_check(f"step {k} gaps", np.abs(port["gaps"][k] - ref["gaps"][k]).max(), bound))
    judged = None if regime != "documented_deviation" else "reported"
    gradients = [("dL/dq", port["dL_dq"], ref["dL_dq"]), ("dL/dv", port["dL_dv"], ref["dL_dv"])]
    gradients += [(f"dL/df_ext step {k}", port["dL_df_ext"][k], ref["dL_df_ext"][k]) for k in range(steps)]
    for name, a, b in gradients:
        scale = _free_norm(b, free)
        checks.append(
            {
                **_check(f"{name} relative", _relative(_free_norm(a - b, free), scale), tol, judged),
                "reference_norm": scale,
            }
        )
    if "dL_dactivations" in ref:
        a, b = port["dL_dactivations"], ref["dL_dactivations"]
        checks.append(_check("dL/du relative", _relative(np.linalg.norm(a - b), np.linalg.norm(b)), tol, judged))
    per_element = port["dL_dmaterials_per_element"]
    for j, (kind, members) in enumerate(
        zip(ref["material_energy_kinds"], ref["material_energy_elements"], strict=True)
    ):
        # Reference energy weights: corotated and Neo-Hookean 2 mu, volume lambda.
        column, factor = (1, 1.0) if kind == "volume" else (0, 0.5)
        value = factor * per_element[members, column].sum()
        floor = steps * U32 * factor * np.abs(per_element[members, column]).sum()
        allowed = tol * abs(ref["dL_dmaterials"][j]) + floor
        entry = _check(f"dL/dw {kind} energy {j}", abs(value - ref["dL_dmaterials"][j]), allowed, judged)
        entry.update(port=float(value), reference=float(ref["dL_dmaterials"][j]))
        entry["relative"] = float(abs(value - ref["dL_dmaterials"][j]) / max(abs(ref["dL_dmaterials"][j]), 1e-300))
        checks.append(entry)
    judged_checks = [c for c in checks if isinstance(c["pass"], bool)]
    states = np.concatenate([fx["q"][None], ref["q_next"]])
    p = states[:, fx["tets"]]
    f = np.einsum("seik,ekj->seij", np.stack([p[:, :, j] - p[:, :, 0] for j in (1, 2, 3)], axis=-1), _rest_inverse(fx))
    sigma = np.linalg.svd(f, compute_uv=False)
    description = {
        "min_det_F": float(np.linalg.det(f).min()),
        "max_stretch": float(sigma.max()),
        "min_stretch": float(sigma.min()),
    }
    if contact:
        nodes = np.asarray(fx["contact_vertices"])
        normal = np.asarray(fx["contact_parameters"][:3]) / np.linalg.norm(fx["contact_parameters"][:3])
        v = ref["v_next"][:, nodes]
        tangential = np.linalg.norm(v - (v @ normal)[..., None] * normal, axis=-1)
        description["active_per_step"] = ref["active"].sum(axis=1).tolist()
        description["max_tangential_speed_active"] = [
            float(tangential[k][ref["active"][k]].max()) if ref["active"][k].any() else 0.0 for k in range(steps)
        ]
    if "mesh_contact_parameters" in fx:
        f_port, f_ref = port["mesh_contact_force"], ref["mesh_contact_force"]
        description["mesh_contact_pairs_per_step"] = {
            "a_vertices_vs_b_faces": port["mesh_contact_pairs"][:, 0].tolist(),
            "b_vertices_vs_a_faces": port["mesh_contact_pairs"][:, 1].tolist(),
        }
        description["mesh_contact_force_norm_per_step"] = [float(np.linalg.norm(f)) for f in f_ref]
        description["mesh_contact_force_difference_per_step"] = [
            float(np.linalg.norm(a - b)) for a, b in zip(f_port, f_ref, strict=True)
        ]
    return {
        "case": str(fx["case"]),
        "material": str(fx["material"]),
        "reference_material_encoding": str(fx["reference_material_encoding"]),
        "steps": steps,
        "contact": str(fx["contact_kind"]),
        "state_forces": ["gravity"] + (["mesh_contact"] if "mesh_contact_parameters" in fx else []),
        "friction": float(fx["friction"]),
        "gradient_regime": regime,
        "gradient_tolerance": tol,
        "reference_iterations": ref["iterations"].tolist(),
        "port_iterations": port["iterations"].tolist(),
        "reference_cascade_cap_active": ref["cascade_cap_active"].tolist(),
        "reference_backbone_clipped": ref["backbone_clipped"].tolist(),
        "scene": description,
        "checks": checks,
        "pass": all(c["pass"] for c in judged_checks),
    }


def _stats(seconds, warmup):
    ms = 1.0e3 * np.asarray(seconds)[warmup:]
    return {
        "median_ms": float(np.median(ms)),
        "p95_ms": float(np.percentile(ms, 95)),
        "mean_ms": float(ms.mean()),
        "samples": len(ms),
    }


def benchmark(fx, port, ref, log):
    warmup, backward_warmup = 10, 10
    result = {"scene": "BearProblem (276 nodes, 768 tets, planar frictional boundary, friction 0.7, 8 muscle groups)"}
    result.update(
        steps=int(fx["benchmark_steps"]),
        dt=float(fx["dt"]),
        tolerances={
            "abs_tol": float(fx["abs_tol"]),
            "rel_tol": float(fx["rel_tol"]),
            "max_iterations": int(fx["max_iterations"]),
        },
        excluded_warmup_steps=warmup,
        timing="synchronized wall time per step; port: wp.synchronize_device before and after step; reference: "
        "PyForward/PyBackward are synchronous (device-to-host copies), including SWIG list conversion",
        backward_window=f"taped rollout of {int(fx['backward_steps'])} steps, second repetition, steps >= {backward_warmup}",
    )
    for side, data in (("port", port), ("reference", ref)):
        iterations = np.asarray(data["forward_iterations"])[warmup:]
        result[side] = {
            "forward": _stats(data["forward_seconds"], warmup),
            "backward": _stats(data["backward_seconds"], backward_warmup),
            "iterations_median": float(np.median(iterations)),
            "iterations_p95": float(np.percentile(iterations, 95)),
            "steps_at_max_iterations": int(np.sum(iterations >= int(fx["max_iterations"]))),
        }
    result["reference"]["gpu_path"] = {
        "cuda_available_build": True,
        "gpu_verify_ok_lines": log.count("GPU S^T*S*v err=") - log.count("WRONG"),
        "batched_spmm_failed": "batched SpMM" in log,
        "used_for": "A^{-1} applications (two cuSPARSE SpMV per axis) and batched Delassus SpMM; local step, contact "
        "Newton, Anderson and the whole backward pass (CPU LDLT, by design) run on the CPU",
    }
    result["reference"]["thread_ct"] = int(ref["thread_ct"])
    result["speedup_forward_median"] = (
        result["reference"]["forward"]["median_ms"] / result["port"]["forward"]["median_ms"]
    )
    result["speedup_backward_median"] = (
        result["reference"]["backward"]["median_ms"] / result["port"]["backward"]["median_ms"]
    )
    return result


def _command(*args):
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.reference_root.resolve()
    revision = _command("git", "-C", str(root), "rev-parse", "HEAD")
    if revision != REFERENCE_REVISION:
        parser.error(f"reference revision {revision!r} does not match pinned {REFERENCE_REVISION}")
    fixtures = sorted(args.results.glob("*.fixture.npz"))
    if not fixtures:
        parser.error(f"no parity fixtures found in {args.results}")
    patch = Path(__file__).with_name("aarch64-build.patch")
    reference_binary = root / "python/py_diff_pd/core/_py_diff_pd_core.so"
    report = {
        "date": date.today().isoformat(),
        "schema": SCHEMA,
        "reference": {
            "repository": "https://github.com/Akinesia112/DiffPhD",
            "revision": revision,
            "expected_revision": REFERENCE_REVISION,
            "working_tree_changes": _command("git", "-C", str(root), "status", "--short"),
            "binary_sha256": hashlib.sha256(reference_binary.read_bytes()).hexdigest()
            if reference_binary.is_file()
            else None,
            "eigen_submodule": _command("git", "-C", str(root / "external/eigen"), "rev-parse", "HEAD"),
            "build_patch": str(patch.relative_to(Path(__file__).resolve().parents[2])),
            "build_patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
            "method": "pd_eigen_alg_phd (ForwardProjectiveDynamicsAlgPhd / BackwardProjectiveDynamicsAlgPhd)",
            "python": _command(str(root / ".venv/bin/python"), "-c", "import sys; print(sys.version.split()[0])"),
            "nvcc": (_command("/usr/local/cuda/bin/nvcc", "--version") or "").splitlines()[-1:],
            "gcc": (_command("/usr/bin/gcc", "--version") or "").splitlines()[:1],
        },
        "port": json.loads((args.results / "port_environment.json").read_text()),
        "thresholds": THRESHOLDS,
        "cases": [],
        "rejected": [],
    }
    for path in fixtures:
        fx = dict(np.load(path))
        name = str(fx["case"])
        report["port"].setdefault("revision", str(fx["port_revision"]))
        report["port"].setdefault("source_sha256", sorted({str(fx["port_source_sha256"])}))
        if str(fx["port_source_sha256"]) not in report["port"]["source_sha256"]:
            report["port"]["source_sha256"].append(str(fx["port_source_sha256"]))
        reasons = reject(fx)
        if reasons:
            report["rejected"].append({"case": name, "reasons": reasons})
            continue
        port = dict(np.load(args.results / f"{name}.port.npz"))
        ref = dict(np.load(args.results / f"{name}.reference.npz"))
        if int(fx["benchmark_steps"]):
            log = (args.results / f"{name}.log").read_text()
            report["performance"] = benchmark(fx, port, ref, log)
        else:
            report["cases"].append(compare_case(fx, port, ref))
    report["summary"] = {
        "cases": len(report["cases"]),
        "passed": [c["case"] for c in report["cases"] if c["pass"]],
        "failed": [c["case"] for c in report["cases"] if not c["pass"]],
        "gradient_comparisons_not_judged": [
            c["case"] for c in report["cases"] if c["gradient_regime"] == "documented_deviation"
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    for case in report["cases"]:
        print(f"{case['case']:40s} {'PASS' if case['pass'] else 'FAIL'} ({case['gradient_regime']})")
        for check in case["checks"]:
            if check["pass"] is False:
                print(f"    {check['metric']}: {check['max_error']:.3e} > {check['threshold']:.3e}")
    for case in report["rejected"]:
        print(f"{case['case']:40s} REJECTED: {', '.join(case['reasons'])}")
    # A successful process must mean that numerical comparisons passed. A
    # benchmark-only run supplies timings, not reference-parity evidence.
    return int(bool(report["summary"]["failed"] or report["rejected"] or not report["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
