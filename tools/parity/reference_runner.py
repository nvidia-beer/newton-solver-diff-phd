"""Run one parity fixture through the executed C++ DiffPhD reference (pd_eigen_alg_phd).

Runs in the reference's own environment, never in the port's:

    <build>/.venv/bin/python -I reference_runner.py --reference-root <build> FIXTURE OUT

``PyForward``/``PyBackward`` are called directly, without the velocity clamping
and adjoint clipping of ``py_diff_pd.env.env_base``. Normal multipliers are
read from the reference's own ``ALG_PHD_DEBUG_CONTACT_VERBOSE`` trace, which
prints six significant digits; the reference exposes them nowhere else.
"""

import argparse
import os
import struct
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

METHOD = "pd_eigen_alg_phd"


def write_mesh(path, vertices, tets):
    """py_diff_pd.common.tet_mesh.generate_tet_mesh without its optional imports."""
    with open(path, "wb") as f:
        f.write(struct.pack("i", 3))
        f.write(struct.pack("i", 4))
        f.write(struct.pack("i", 3))
        f.write(struct.pack("i", len(vertices)))
        for axis in range(3):
            for v in vertices:
                f.write(struct.pack("d", v[axis]))
        f.write(struct.pack("i", 4))
        f.write(struct.pack("i", len(tets)))
        for corner in range(4):
            for t in tets:
                f.write(struct.pack("i", int(t[corner])))


def build(core, fx, workdir):
    path = str(Path(workdir) / "mesh.bin")
    write_mesh(path, fx["rest_vertices"], fx["tets"])
    deformable = core.TetDeformable()
    deformable.Initialize(path, float(fx["density"]), "none", 0.0, 0.0)
    mu, lame = fx["mu"], fx["lambda"]
    energies = []  # (kind, element indices)
    if str(fx["material"]) == "neo_hookean":
        deformable.AddPdEnergy("pd_neohookean", list(np.column_stack([mu, lame]).reshape(-1)), [])
        energies.append(("neo_hookean", np.arange(len(mu))))
    elif str(fx["reference_material_encoding"]) == "grouped":
        # One energy pair per distinct (mu, lambda): per-group scalar stiffness.
        pairs, group = np.unique(np.column_stack([mu, lame]), axis=0, return_inverse=True)
        for g, (m, la) in enumerate(pairs):
            members = [int(i) for i in np.flatnonzero(group == g)]
            deformable.AddPdEnergy("corotated", [2.0 * m], members)
            deformable.AddPdEnergy("volume", [la], members)
            energies += [("corotated", np.array(members)), ("volume", np.array(members))]
    else:
        # Per-element stiffness arrays (InitializeElementStiffnesses).
        deformable.AddPdEnergy("corotated", list(2.0 * mu), [])
        deformable.AddPdEnergy("volume", list(lame), [])
        energies += [("corotated", np.arange(len(mu))), ("volume", np.arange(len(mu)))]
    muscles = []
    if "muscle_directions" in fx:
        for e, (direction, stiffness) in enumerate(zip(fx["muscle_directions"], fx["muscle_stiffness"], strict=True)):
            if stiffness > 0:
                deformable.AddActuation(float(stiffness), [float(c) for c in direction], [e])
                muscles.append(e)
    deformable.AddStateForce("gravity", [float(g) for g in fx["gravity"]])
    if "mesh_contact_parameters" in fx:
        faces_a, faces_b = fx["mesh_contact_faces_a"], fx["mesh_contact_faces_b"]
        deformable.AddStateForce(
            "mesh_contact",
            [float(p) for p in fx["mesh_contact_parameters"]]
            + [float(len(faces_a)), float(len(faces_b))]
            + [float(i) for i in faces_a.reshape(-1)]
            + [float(i) for i in faces_b.reshape(-1)],
        )
    for i in np.flatnonzero(fx["dirichlet"]):
        for d in range(3):
            deformable.SetDirichletBoundaryCondition(int(3 * i + d), float(fx["q"][i, d]))
    if str(fx["contact_kind"]) != "none":
        deformable.SetFrictionalBoundary(
            str(fx["contact_kind"]),
            [float(p) for p in fx["contact_parameters"]],
            [int(i) for i in fx["contact_vertices"]],
        )
    return deformable, energies, np.array(muscles, dtype=np.int64)


def options(core, fx):
    values = {
        "max_pd_iter": float(fx["max_iterations"]),
        "abs_tol": float(fx["abs_tol"]),
        "rel_tol": float(fx["rel_tol"]),
        "verbose": 0.0,
        "thread_ct": float(os.cpu_count()),
        "aa_window": float(fx["aa_window"]),
        "friction_mu": float(fx["friction"]),
        "restitution": float(fx["restitution"]),
    }
    result = core.StdMap()
    for key, value in values.items():
        result[key] = value
    return result


class CapturedStderr:
    """Collect C++ stderr (the reference's contact trace) of one call."""

    def __enter__(self):
        sys.stderr.flush()
        self.file = tempfile.TemporaryFile(mode="w+")
        self.saved = os.dup(2)
        os.dup2(self.file.fileno(), 2)
        return self

    def __exit__(self, *args):
        os.dup2(self.saved, 2)
        os.close(self.saved)
        self.file.seek(0)
        self.text = self.file.read()
        self.file.close()
        return False


def last_multipliers(text, nodes):
    """Final multipliers of the step's candidates, scattered to the registered ``nodes`` (zero elsewhere).

    The candidates are the ``init_distance[node=...]`` lines of the trace, in order; for the
    planar and spherical boundaries they are all registered vertices.
    """
    lines = [line for line in text.splitlines() if line.startswith("[ALG_PHD_DEBUG] lambda_after:")]
    candidates = [
        int(line.split("node=", 1)[1].split("]", 1)[0])
        for line in text.splitlines()
        if line.startswith("[ALG_PHD_DEBUG] init_distance[node=")
    ]
    result = np.zeros(len(nodes))
    if not lines:
        return result if not candidates else np.full(len(nodes), np.nan)
    values = np.array([float(v) for v in lines[-1].split(":", 1)[1].split(",")])
    if len(values) != len(candidates):
        raise RuntimeError("unexpected multiplier count in the reference trace")
    lookup = dict(zip(candidates, values, strict=True))
    return np.array([lookup.get(int(node), 0.0) for node in nodes])


def activations(fx, step, muscles):
    # Reference target length a = 1 - u for the contraction u of the fixture.
    return [1.0 - float(fx["activations"][step, e]) for e in muscles]


def forward(core, deformable, fx, opts, q, v, step, muscles, trace):
    set_prescribed_positions(deformable, fx, step)
    n = deformable.dofs()
    q_next, v_next, active = core.StdRealVector(n), core.StdRealVector(n), core.StdIntVector(0)
    a = activations(fx, step, muscles) if len(muscles) else []
    f = list(fx["f_ext"][step].reshape(-1))
    if trace:
        with CapturedStderr() as captured:
            deformable.PyForward(METHOD, list(q), list(v), a, f, float(fx["dt"]), opts, q_next, v_next, active)
        return np.array(q_next), np.array(v_next), list(active), captured.text
    deformable.PyForward(METHOD, list(q), list(v), a, f, float(fx["dt"]), opts, q_next, v_next, active)
    return np.array(q_next), np.array(v_next), list(active), ""


def set_prescribed_positions(deformable, fx, step):
    if "dirichlet_positions" in fx:
        for i in np.flatnonzero(fx["dirichlet"]):
            for d in range(3):
                deformable.SetDirichletBoundaryCondition(int(3 * i + d), float(fx["dirichlet_positions"][step, i, d]))


def backward(core, deformable, fx, opts, trajectory, step, muscles, dl_dq_next, dl_dv_next, energies):
    set_prescribed_positions(deformable, fx, step)
    n, acts = deformable.dofs(), deformable.act_dofs()
    q, v, q_next, v_next, active = trajectory[step]
    outputs = [core.StdRealVector(n), core.StdRealVector(n), core.StdRealVector(acts), core.StdRealVector(n)]
    outputs += [core.StdRealVector(len(energies)), core.StdRealVector(len(muscles)), core.StdRealVector(3)]
    a = activations(fx, step, muscles) if len(muscles) else []
    active_vector = core.StdIntVector(active)
    deformable.PyBackward(
        METHOD,
        list(q),
        list(v),
        a,
        list(fx["f_ext"][step].reshape(-1)),
        float(fx["dt"]),
        list(q_next),
        list(v_next),
        active_vector,
        list(dl_dq_next),
        list(dl_dv_next),
        opts,
        *outputs,
    )
    return [np.array(o) for o in outputs]


def run_case(core, fx, workdir):
    deformable, energies, muscles = build(core, fx, workdir)
    opts = options(core, fx)
    steps, dt = int(fx["steps"]), float(fx["dt"])
    contact = str(fx["contact_kind"]) != "none"
    nodes = np.asarray(fx["contact_vertices"])
    q, v = fx["q"].reshape(-1).copy(), fx["v"].reshape(-1).copy()
    trajectory, q_out, v_out, multipliers, active_sets, mesh_forces = [], [], [], [], [], []
    gravity_force = float(fx["density"]) * mean_volume(fx) * np.tile(fx["gravity"], len(fx["q"]))
    core.AlgPhdResetFrameStats()
    if contact:
        os.environ["ALG_PHD_DEBUG_CONTACT"] = "1"
        os.environ["ALG_PHD_DEBUG_CONTACT_VERBOSE"] = "1"
    for k in range(steps):
        if "mesh_contact_parameters" in fx:
            # ForwardStateForce(q, v) of this step without the gravitational state force.
            total = np.array(deformable.PyForwardStateForce(list(q), list(v)))
            mesh_forces.append((total - gravity_force).reshape(-1, 3))
        q_next, v_next, active, text = forward(core, deformable, fx, opts, q, v, k, muscles, contact)
        trajectory.append((q, v, q_next, v_next, active))
        q_out.append(q_next.reshape(-1, 3))
        physical_velocity = v_next * float(fx.get("velocity_damping", 1.0))
        v_out.append(physical_velocity.reshape(-1, 3))
        if contact:
            multipliers.append(last_multipliers(text, nodes))
            active_sets.append(np.isin(nodes, active))
        q, v = q_next, physical_velocity
    os.environ.pop("ALG_PHD_DEBUG_CONTACT", None)
    os.environ.pop("ALG_PHD_DEBUG_CONTACT_VERBOSE", None)
    result = {
        "q_next": np.stack(q_out),
        "v_next": np.stack(v_out),
        "iterations": np.array(core.AlgPhdFrameIters()),
        "converged": np.array(core.AlgPhdFrameConverged()) == 1,
    }
    if mesh_forces:
        result["mesh_contact_force"] = np.stack(mesh_forces)
    if contact:
        result["multipliers"] = np.stack(multipliers)
        result["active"] = np.stack(active_sets)
        if str(fx["contact_kind"]) == "planar":
            normal = np.asarray(fx["contact_parameters"][:3])
            result["gaps"] = (result["q_next"][:, nodes] @ normal + fx["contact_parameters"][3]) / np.linalg.norm(
                normal
            )
    # Backward through the linear objective on the final state.
    dl_dq_next, dl_dv_next = fx["objective_q"].reshape(-1), fx["objective_v"].reshape(-1)
    dl_df, dl_da = [None] * steps, [None] * steps
    dl_dmat = np.zeros(len(energies))
    cap_active, clipped = [], []
    inv_h2m = None
    for k in reversed(range(steps)):
        dl_dv_next = dl_dv_next * float(fx.get("velocity_damping", 1.0))
        dl_dq, dl_dv, da, df, dmat, _dact, _dstate = backward(
            core, deformable, fx, opts, trajectory, k, muscles, dl_dq_next, dl_dv_next, energies
        )
        if inv_h2m is None:
            mass = float(fx["density"]) * mean_volume(fx)
            inv_h2m = mass / dt**2
        # Reference safeguards: cascade cap ||m/h^2 mu|| <= ||rhs|| + 1 and the 1e6 backbone clip.
        rhs = np.linalg.norm(dl_dq_next + dl_dv_next / dt)
        injected = inv_h2m * np.linalg.norm(df)
        cap_active.append(bool(injected >= (1.0 - 1.0e-9) * (rhs + 1.0)))
        # A diverged backbone (||mu|| >= 0.999e6) skips every mu-derived term, leaving dL/df_ext = 0.
        clipped.append(bool(rhs > 0 and not np.any(df)))
        dl_df[k] = df.reshape(-1, 3)
        if len(muscles):
            full = np.zeros(len(fx["tets"]))
            full[muscles] = -da  # dL/du = -dL/da with a = 1 - u.
            dl_da[k] = full
        dl_dmat += dmat
        dl_dq_next, dl_dv_next = dl_dq, dl_dv
    result.update(
        dL_dq=dl_dq_next.reshape(-1, 3),
        dL_dv=dl_dv_next.reshape(-1, 3),
        dL_df_ext=np.stack(dl_df),
        dL_dmaterials=dl_dmat,
        material_energy_kinds=np.array([kind for kind, _ in energies]),
        material_energy_elements=np.array([np.isin(np.arange(len(fx["tets"])), e) for _, e in energies]),
        cascade_cap_active=np.array(cap_active[::-1]),
        backbone_clipped=np.array(clipped[::-1]),
    )
    if len(muscles):
        result["dL_dactivations"] = np.stack(dl_da)
    return result


def mean_volume(fx):
    p = fx["rest_vertices"][fx["tets"]]
    return float(np.mean(np.abs(np.linalg.det(np.stack([p[:, j] - p[:, 0] for j in (1, 2, 3)], axis=-1))) / 6.0))


def run_benchmark(core, fx, workdir):
    """Per-step synchronous wall times; PyForward returns after its last device copy."""
    deformable, energies, muscles = build(core, fx, workdir)
    opts = options(core, fx)
    core.AlgPhdResetFrameStats()
    q, v = fx["q"].reshape(-1).copy(), fx["v"].reshape(-1).copy()
    forward_seconds = []
    for k in range(int(fx["benchmark_steps"])):
        start = time.perf_counter()
        q, v, _active, _ = forward(core, deformable, fx, opts, q, v, k, muscles, False)
        forward_seconds.append(time.perf_counter() - start)
    iterations = np.array(core.AlgPhdFrameIters())
    converged = np.array(core.AlgPhdFrameConverged())
    steps = int(fx["backward_steps"])
    for _repeat in range(2):
        # A fresh deformable per rollout: the reference keeps warm-start multipliers per instance.
        deformable, energies, muscles = build(core, fx, workdir)
        q, v = fx["q"].reshape(-1).copy(), fx["v"].reshape(-1).copy()
        trajectory = []
        for k in range(steps):
            q_next, v_next, active, _ = forward(core, deformable, fx, opts, q, v, k, muscles, False)
            trajectory.append((q, v, q_next, v_next, active))
            q, v = q_next, v_next
        dl_dq, dl_dv = fx["objective_q"].reshape(-1), fx["objective_v"].reshape(-1)
        backward_seconds = [0.0] * steps
        for k in reversed(range(steps)):
            start = time.perf_counter()
            dl_dq, dl_dv, *_ = backward(core, deformable, fx, opts, trajectory, k, muscles, dl_dq, dl_dv, energies)
            backward_seconds[k] = time.perf_counter() - start
    return {
        "forward_seconds": np.array(forward_seconds),
        "forward_iterations": iterations,
        "forward_converged": converged,
        "backward_seconds": np.array(backward_seconds),
        "thread_ct": os.cpu_count(),
        "s_memory_bytes": core.AlgPhdSMemoryBytes(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.reference_root / "python"))
    import py_diff_pd.core.py_diff_pd_core as core

    fx = dict(np.load(args.fixture))
    if str(fx.get("schema")) != "newton.diff_phd_parity.v1":
        raise ValueError("unsupported fixture schema")
    with tempfile.TemporaryDirectory() as workdir:
        if int(fx["benchmark_steps"]):
            result = run_benchmark(core, fx, workdir)
        else:
            result = run_case(core, fx, workdir)
    sys.stdout.flush()
    np.savez(args.out, **result)


if __name__ == "__main__":
    main()
