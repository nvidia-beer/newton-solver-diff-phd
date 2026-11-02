"""Write parity fixtures, run SolverDiffPhD on them and, optionally, the executed C++ reference.

Development harness only. Every fixture follows the "Parity fixture schema" of
docs/validation.md. Inputs are rounded to float32 because Newton stores states,
rest positions and materials in float32; the reference then receives exactly the
same float64 values. Rest grids use power-of-two cells so that the float32 rest
pose inverses of Newton equal the float64 ones of the reference.

    uv run --locked python tools/parity/port_cases.py --out OUT --device cuda:0 \
        --reference-root ../../DiffPhD-reference-build
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import newton
import numpy as np
import warp as wp

from newton_solver_diff_phd import SolverDiffPhD
from newton_solver_diff_phd._src.solvers.diff_phd.contact import surface_vertices

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "newton.diff_phd_parity.v1"
TIGHT = {"abs_tol": 1.0e-12, "rel_tol": 1.0e-11, "max_iterations": 2000, "aa_window": 5}
DT = 0.004
DENSITY = 1000.0
MU, LAME = 2.0e4, 5.0e4


def f32(values):
    return np.asarray(values, dtype=np.float32).astype(np.float64)


def _grid(fix_left, contrast=1.0, height=0.5):
    """Topology and materials of a 4x2x2 grid with 0.125 m cells (exact float32 rest poses)."""
    builder = newton.ModelBuilder()
    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, height),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=4,
        dim_y=2,
        dim_z=2,
        cell_x=0.125,
        cell_y=0.125,
        cell_z=0.125,
        density=DENSITY,
        k_mu=MU,
        k_lambda=LAME,
        k_damp=0.0,
        fix_left=fix_left,
    )
    model = builder.finalize(device="cpu")
    rest = model.particle_q.numpy().astype(np.float64)
    tets = model.tet_indices.numpy().reshape(-1, 4).astype(np.int32)
    mu, lame = np.full(len(tets), MU), np.full(len(tets), LAME)
    stiff = rest[tets].mean(axis=1)[:, 0] > rest[:, 0].mean()
    mu[stiff] *= contrast
    lame[stiff] *= contrast
    return rest, tets, mu, lame, model.particle_inv_mass.numpy() == 0


def _base(name, rest, tets, mu, lame, dirichlet, *, steps=1, material="pd", seed=0):
    rng = np.random.default_rng(seed)
    n = len(rest)
    return {
        "case": name,
        "rest_vertices": rest,
        "tets": tets,
        "density": DENSITY,
        "mu": mu,
        "lambda": lame,
        "material": material,
        "dirichlet": dirichlet,
        "q": rest.copy(),
        "v": np.zeros((n, 3)),
        "f_ext": np.zeros((steps, n, 3)),
        "gravity": np.array([0.0, 0.0, -9.81]),
        "dt": DT,
        "steps": steps,
        **TIGHT,
        "contact_kind": "none",
        "contact_parameters": np.zeros(4),
        "contact_vertices": np.zeros(0, dtype=np.int32),
        "friction": 0.0,
        "restitution": 0.0,
        "objective_q": rng.standard_normal((n, 3)),
        "objective_v": 0.01 * rng.standard_normal((n, 3)),
        "reference_material_encoding": "per_element",
        "benchmark_steps": 0,
    }


def _deform(fixture, stretch, speed, force, seed):
    rng = np.random.default_rng(seed)
    rest, free = fixture["rest_vertices"], ~fixture["dirichlet"]
    center = rest.mean(axis=0)
    q = np.where(free[:, None], center + (rest - center) * np.asarray(stretch), rest)
    q[free] += 0.01 * rng.standard_normal((free.sum(), 3))
    fixture["q"] = q
    fixture["v"] = np.where(free[:, None], speed * rng.standard_normal(q.shape), 0.0)
    fixture["f_ext"] = force * rng.uniform(-1.0, 1.0, fixture["f_ext"].shape)


def _contact(name, friction, velocity, restitution=0.0):
    # The rest grid stays on dyadic coordinates; only the initial state is lifted 1 mm above the plane.
    fixture = _base(name, *_grid(False, height=0.0), steps=3, seed=5)
    fixture["q"] = fixture["rest_vertices"] + np.array([0.0, 0.0, 0.001])
    fixture["v"] = np.tile(np.asarray(velocity, dtype=np.float64), (len(fixture["q"]), 1))
    fixture.update(
        contact_kind="planar",
        contact_parameters=np.array([0.0, 0.0, 1.0, 0.0]),
        contact_vertices=surface_vertices(fixture["tets"]),
        friction=friction,
        restitution=restitution,
    )
    return fixture


_TET_FACES = np.array(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)))


def _boundary_faces(tets):
    """Faces of ``tets`` that belong to exactly one tetrahedron."""
    faces = tets[:, _TET_FACES].reshape(-1, 3)
    _, first, counts = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    return faces[np.sort(first[counts == 1])].astype(np.int32)


def _mesh_contact_poke():
    """A pinned-bottom 4x2x2 grid (A) and a free one-cell grid (B) above its top, coupled by mesh_contact.

    B rests 1/32 m above A on dyadic coordinates; its initial state is lowered 1 cm so A's top
    vertices reach B's bottom faces and B's bottom vertices reach A's top faces within the radius.
    """
    builder = newton.ModelBuilder()
    common = {"rot": wp.quat_identity(), "vel": wp.vec3(0.0), "density": DENSITY, "k_mu": MU, "k_lambda": LAME}
    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, 0.0),
        dim_x=4,
        dim_y=2,
        dim_z=2,
        cell_x=0.125,
        cell_y=0.125,
        cell_z=0.125,
        k_damp=0.0,
        fix_bottom=True,
        **common,
    )
    count_a = builder.particle_count
    builder.add_soft_grid(
        pos=wp.vec3(0.3125, 0.0625, 0.28125),
        dim_x=1,
        dim_y=1,
        dim_z=1,
        cell_x=0.125,
        cell_y=0.125,
        cell_z=0.125,
        k_damp=0.0,
        **common,
    )
    model = builder.finalize(device="cpu")
    rest = model.particle_q.numpy().astype(np.float64)
    tets = model.tet_indices.numpy().reshape(-1, 4).astype(np.int32)
    dirichlet = model.particle_inv_mass.numpy() == 0
    fixture = _base(
        "mesh_contact_poke", rest, tets, np.full(len(tets), MU), np.full(len(tets), LAME), dirichlet, steps=3, seed=9
    )
    rng = np.random.default_rng(10)
    free = ~dirichlet
    body_b = np.arange(len(rest)) >= count_a
    q = rest + np.where(free[:, None], 0.002 * rng.standard_normal(rest.shape), 0.0)
    q[body_b, 2] -= 0.01
    v = np.where(free[:, None], 0.2 * rng.standard_normal(rest.shape), 0.0)
    v[body_b] += np.array([0.5, 0.2, -0.5])
    faces = _boundary_faces(tets)
    in_b = faces >= count_a
    fixture.update(
        q=q,
        v=v,
        f_ext=0.5 * rng.uniform(-1.0, 1.0, fixture["f_ext"].shape),
        # contact_radius [m], kn_left [N/m] (A vertices vs B faces), kn_right [N/m] (B vertices vs A faces),
        # kf [N s/m], mu.
        mesh_contact_parameters=np.array([0.08, 60.0, 150.0, 3.0, 0.5]),
        mesh_contact_faces_a=faces[~in_b.any(axis=1)],
        mesh_contact_faces_b=faces[in_b.all(axis=1)],
    )
    return fixture


def _mesh_boundary_poke():
    """The ``_mesh_contact_poke`` blocks coupled by the reference MeshFrictionalBoundary instead.

    A's free surface vertices (below ``split_index``) are the candidates against B's faces, as
    the reference poke registers the die against the probe. B falls onto A's top; the boundary
    acts on the candidates only. One step: from the second step on, the port starts from the
    float32 state Newton stores, which shifts the multipliers of this stiff contact by more than
    the multiplier bound allows (positions stay within theirs); the multi-step equivalence is
    tested against the dense oracle with the same float32 round trip.
    """
    fixture = _mesh_contact_poke()
    del fixture["mesh_contact_parameters"]
    faces_left, faces_right = fixture.pop("mesh_contact_faces_a"), fixture.pop("mesh_contact_faces_b")
    count_a = int(faces_right.min())
    rng = np.random.default_rng(11)
    free = ~fixture["dirichlet"]
    body_b = np.arange(len(fixture["q"])) >= count_a
    rest = fixture["rest_vertices"]
    q = rest + np.where(free[:, None], 0.001 * rng.standard_normal(rest.shape), 0.0)
    q[body_b, 2] -= 0.03
    v = np.where(free[:, None], 0.05 * rng.standard_normal(rest.shape), 0.0)
    v[body_b] += np.array([0.0, 0.0, -0.5])
    candidates = np.unique(faces_left)
    radius = 0.01
    fixture.update(
        case="mesh_boundary_poke",
        steps=1,
        q=q,
        v=v,
        f_ext=np.zeros_like(fixture["f_ext"][:1]),
        contact_kind="mesh",
        # SetFrictionalBoundary("mesh", ...): contact_radius [m], split_index, face counts, faces_left, faces_right.
        contact_parameters=np.concatenate(
            [[radius, count_a, len(faces_left), len(faces_right)], faces_left.ravel(), faces_right.ravel()]
        ).astype(np.float64),
        contact_vertices=candidates[free[candidates]].astype(np.int32),
    )
    return fixture


def _mesh_contact_prescribed():
    """Root-prescribed probe with independent incoming velocity, as in the training demo."""
    fixture = _mesh_contact_poke()
    probe = np.unique(fixture["mesh_contact_faces_b"])
    root = probe[fixture["rest_vertices"][probe, 2] == fixture["rest_vertices"][probe, 2].max()]
    fixture["dirichlet"][root] = True
    targets = np.repeat(fixture["q"][None], fixture["steps"], axis=0)
    for k in range(fixture["steps"]):
        targets[k, root] += (k + 1) * np.array([0.001, 0.0, -0.002])
    fixture.update(
        case="mesh_contact_prescribed", material="neo_hookean", dirichlet_positions=targets, velocity_damping=0.30
    )
    return fixture


def mesh_boundary(parameters):
    """``("mesh", ...)`` parameters of SolverDiffPhD from the flat reference list."""
    radius, split, left, right = parameters[0], int(parameters[1]), int(parameters[2]), int(parameters[3])
    faces = np.asarray(parameters[4:], dtype=np.int64).reshape(-1, 3)
    if len(faces) != left + right:
        raise ValueError("mesh boundary face counts do not match")
    return float(radius), split, faces[:left], faces[left:]


def case_fixtures():
    cases = []
    fixture = _base("homogeneous_free", *_grid(False))
    _deform(fixture, (1.2, 0.85, 1.05), 0.3, 1.0, 1)
    cases.append(fixture)
    cases.append({**fixture, "case": "zero_volume_weight", "lambda": np.zeros_like(fixture["lambda"])})
    for contrast in (10.0, 100.0):
        for encoding in ("per_element", "grouped"):
            fixture = _base(f"pinned_contrast_{contrast:g}x_{encoding}", *_grid(True, contrast), seed=2)
            _deform(fixture, (1.25, 0.9, 1.1), 0.3, 1.0, 3)
            fixture["reference_material_encoding"] = encoding
            cases.append(fixture)
            if contrast == 10.0 and encoding == "per_element":
                # Keep the original capped-adjoint fixture above unchanged.
                # This additional timestep has a convergent reference adjoint.
                cases.append({**fixture, "case": "pinned_contrast_10x_per_element_small_dt", "dt": 0.001})
    fixture = _base("neo_hookean_large_deformation", *_grid(False), material="neo_hookean", seed=4)
    _deform(fixture, (1.5, 0.6, 1.25), 0.5, 1.0, 4)
    cases.append(fixture)
    cases.append(_contact("plane_impact", 0.0, (0.0, 0.0, -1.0)))
    cases.append(_contact("sticking_friction", 2.0, (0.05, 0.0, -1.0)))
    cases.append(_contact("sliding_friction", 0.1, (2.0, 0.0, -1.0)))
    fixture = _base("multistep_pinned", *_grid(True), steps=4, seed=6)
    _deform(fixture, (1.2, 0.85, 1.05), 0.3, 1.0, 6)
    cases.append(fixture)
    fixture = _base("muscle_activation", *_grid(True), steps=2, seed=7)
    _deform(fixture, (1.1, 0.95, 1.0), 0.2, 0.5, 7)
    rng = np.random.default_rng(8)
    count = len(fixture["tets"])
    fixture["muscle_directions"] = rng.standard_normal((count, 3))
    fixture["muscle_stiffness"] = np.full(count, 4.0e4)
    fixture["activations"] = rng.uniform(-0.2, 0.3, (2, count))
    cases.append(fixture)
    cases.append(_mesh_contact_poke())
    cases.append(_mesh_boundary_poke())
    cases.append(_mesh_contact_prescribed())
    return cases


def finalize_fixture(fixture):
    """Round to the float32 values Newton stores and attach metadata."""
    for key in ("rest_vertices", "mu", "lambda", "q", "v", "f_ext", "gravity", "objective_q", "objective_v"):
        fixture[key] = f32(fixture[key])
    if "activations" in fixture:
        fixture["activations"] = f32(fixture["activations"])
    if "dirichlet_positions" in fixture:
        fixture["dirichlet_positions"] = f32(fixture["dirichlet_positions"])
    fixture["schema"] = SCHEMA
    fixture["activation_convention"] = "contraction u; reference target fibre length a = 1 - u"
    fixture["port_revision"] = _git(ROOT)
    fixture["port_source_sha256"] = _source_hash()
    return fixture


def _source_hash():
    """Hash of the uncommitted package sources that produced the port results."""
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git(path):
    revision = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain"], capture_output=True, text=True).stdout
    return revision.strip() + ("+uncommitted" if dirty.strip() else "")


def build_model(fixture, device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for point, pinned in zip(fixture["rest_vertices"], fixture["dirichlet"], strict=True):
        builder.add_particle(wp.vec3(*point), wp.vec3(0.0), mass=0.0 if pinned else 1.0, radius=0.001)
    for tet, mu, lame in zip(fixture["tets"], fixture["mu"], fixture["lambda"], strict=True):
        builder.add_tetrahedron(*map(int, tet), k_mu=float(mu), k_lambda=float(lame), k_damp=0.0)
    model = builder.finalize(device=device, requires_grad=True)
    model.gravity.assign(np.asarray([fixture["gravity"]], dtype=np.float32))
    return model


def boundary_parameters(fixture):
    if str(fixture["contact_kind"]) == "mesh":
        return mesh_boundary(fixture["contact_parameters"])
    return tuple(float(p) for p in fixture["contact_parameters"])


def build_solver(fixture, model):
    options = {
        "density": float(fixture["density"]),
        "material": str(fixture["material"]),
        "max_iterations": int(fixture["max_iterations"]),
        "abs_tol": float(fixture["abs_tol"]),
        "rel_tol": float(fixture["rel_tol"]),
        "anderson_window": int(fixture["aa_window"]),
    }
    if "muscle_directions" in fixture:
        options["muscle_directions"] = fixture["muscle_directions"]
        options["muscle_stiffness"] = fixture["muscle_stiffness"]
    if str(fixture["contact_kind"]) != "none":
        options.update(
            frictional_boundary=(str(fixture["contact_kind"]), boundary_parameters(fixture)),
            frictional_vertices=np.asarray(fixture["contact_vertices"]),
            friction=float(fixture["friction"]),
            restitution=float(fixture["restitution"]),
        )
    if "mesh_contact_parameters" in fixture:
        options["mesh_contact"] = (
            *(float(p) for p in fixture["mesh_contact_parameters"]),
            np.asarray(fixture["mesh_contact_faces_a"]),
            np.asarray(fixture["mesh_contact_faces_b"]),
        )
    return SolverDiffPhD(model, **options)


@wp.kernel
def linear_loss(
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    q_weight: wp.array[wp.vec3d],
    qd_weight: wp.array[wp.vec3d],
    loss: wp.array[wp.float64],
):
    i = wp.tid()
    wp.atomic_add(loss, 0, wp.dot(wp.vec3d(q[i]), q_weight[i]) + wp.dot(wp.vec3d(qd[i]), qd_weight[i]))


@wp.kernel
def damp_velocity(velocity: wp.array[wp.vec3], damping: float, out: wp.array[wp.vec3]):
    i = wp.tid()
    out[i] = damping * velocity[i]


def _rollout(fixture, model, solver, steps, tape=None, markers=None):
    """Taped rollout from the fixture's initial state; returns states, controls and the loss."""
    device = model.device
    states = [model.state(requires_grad=True) for _ in range(steps + 1)]
    states[0].particle_q.assign(fixture["q"].astype(np.float32))
    states[0].particle_qd.assign(fixture["v"].astype(np.float32))
    controls = []
    for k in range(steps):
        states[k].particle_f.assign(fixture["f_ext"][k].astype(np.float32))
        control = model.control()
        if "activations" in fixture:
            control.tet_activations = wp.array(
                fixture["activations"][k].astype(np.float32), dtype=wp.float32, device=device, requires_grad=True
            )
        controls.append(control)
    if solver.contact_multipliers is not None:
        solver.contact_multipliers.zero_()
    loss = wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True)
    records = []
    with tape if tape is not None else _Null():
        for k in range(steps):
            if markers is not None:
                tape.record_func(lambda k=k: markers.append((k, _now(device))), [])
            targets = (
                wp.array(fixture["dirichlet_positions"][k], dtype=wp.vec3, device=device)
                if "dirichlet_positions" in fixture
                else None
            )
            raw_state = model.state(requires_grad=True) if "velocity_damping" in fixture else states[k + 1]
            solver.step(states[k], raw_state, controls[k], None, float(fixture["dt"]), dirichlet_positions=targets)
            if "velocity_damping" in fixture:
                states[k + 1].particle_q = raw_state.particle_q
                wp.launch(
                    damp_velocity,
                    model.particle_count,
                    inputs=[raw_state.particle_qd, float(fixture["velocity_damping"])],
                    outputs=[states[k + 1].particle_qd],
                    device=device,
                )
            if markers is None:
                multipliers = solver.contact_multipliers
                mesh = solver._mesh_contact
                records.append(
                    (
                        solver._x.numpy().copy(),
                        int(solver.iterations.numpy()[0]),
                        None if multipliers is None else multipliers.numpy().copy(),
                        solver.read_status(),
                        None
                        if mesh is None
                        else (
                            mesh.force.numpy().copy(),
                            # Pairs within the radius per direction (barycentric weights sum to 1 in contact).
                            int(np.any(mesh.bary_ab.numpy() != 0.0, axis=-1).sum()),
                            int(np.any(mesh.bary_ba.numpy() != 0.0, axis=-1).sum()),
                        ),
                    )
                )
        if markers is not None:
            tape.record_func(lambda: markers.append((steps, _now(device))), [])
        wp.launch(
            linear_loss,
            model.particle_count,
            inputs=[
                states[steps].particle_q,
                states[steps].particle_qd,
                wp.array(fixture["objective_q"], dtype=wp.vec3d, device=device),
                wp.array(fixture["objective_v"], dtype=wp.vec3d, device=device),
            ],
            outputs=[loss],
            device=device,
        )
    return states, controls, loss, records


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _now(device):
    wp.synchronize_device(device)
    return time.perf_counter()


def run_port(fixture, device):
    model = build_model(fixture, device)
    solver = build_solver(fixture, model)
    steps = int(fixture["steps"])
    tape = wp.Tape()
    states, controls, loss, records = _rollout(fixture, model, solver, steps, tape)
    tape.backward(loss)
    backward_status = solver.read_status()
    free = ~fixture["dirichlet"]
    q_in = [states[k].particle_q.numpy().astype(np.float64) for k in range(steps)]
    x = np.stack([r[0] for r in records])
    result = {
        "q_next": x,
        "v_next": np.stack(
            [(x[k] - q_in[k]) / fixture["dt"] * fixture.get("velocity_damping", 1.0) for k in range(steps)]
        ),
        "iterations": np.array([r[1] for r in records]),
        "converged": np.array([(r[3] & 1) == 0 for r in records]),
        "status_bits": np.array([r[3] for r in records] + [backward_status]),
        "dL_dq": states[0].particle_q.grad.numpy().astype(np.float64),
        "dL_dv": states[0].particle_qd.grad.numpy().astype(np.float64),
        "dL_df_ext": np.stack([states[k].particle_f.grad.numpy().astype(np.float64) for k in range(steps)]),
        "dL_dmaterials_per_element": model.tet_materials.grad.numpy()[:, :2].astype(np.float64),
        "loss": float(loss.numpy()[0]),
        "free": free,
    }
    if records[0][2] is not None:
        multipliers = np.stack([r[2] for r in records])
        nodes = np.asarray(fixture["contact_vertices"])
        result["multipliers"] = multipliers
        result["active"] = multipliers > 0
        if str(fixture["contact_kind"]) == "planar":
            normal = np.asarray(fixture["contact_parameters"][:3])
            result["gaps"] = (x[:, nodes] @ normal + fixture["contact_parameters"][3]) / np.linalg.norm(normal)
    if records[0][4] is not None:
        result["mesh_contact_force"] = np.stack([r[4][0] for r in records])
        result["mesh_contact_pairs"] = np.array([r[4][1:] for r in records])
    if "activations" in fixture:
        result["dL_dactivations"] = np.stack([c.tet_activations.grad.numpy().astype(np.float64) for c in controls])
    return result


def benchmark_port(fixture, device):
    """Synchronized per-step forward times and taped backward times of the fixture scene."""
    model = build_model(fixture, device)
    solver = build_solver(fixture, model)
    dt = float(fixture["dt"])
    a, b = model.state(), model.state()
    a.particle_q.assign(fixture["q"].astype(np.float32))
    a.particle_qd.assign(fixture["v"].astype(np.float32))
    control = model.control()
    control.tet_activations = wp.zeros(model.tet_count, dtype=wp.float32, device=device)
    forward, iterations, bits = [], [], 0
    for k in range(int(fixture["benchmark_steps"])):
        control.tet_activations.assign(fixture["activations"][k].astype(np.float32))
        start = _now(device)
        solver.step(a, b, control, None, dt)
        forward.append(_now(device) - start)
        iterations.append(int(solver.iterations.numpy()[0]))
        bits |= solver.read_status()
        a, b = b, a
    steps = int(fixture["backward_steps"])
    for _repeat in range(2):
        # The second repetition excludes first-use compilation and graph capture of the backward pass.
        markers = []
        tape = wp.Tape()
        _states, _controls, loss, _ = _rollout(fixture, model, solver, steps, tape, markers)
        markers.clear()
        tape.backward(loss)
        tape.zero()
        bits |= solver.read_status()
    times = dict(markers)
    backward = [times[k] - times[k + 1] for k in range(steps)]
    return {
        "forward_seconds": np.array(forward),
        "forward_iterations": np.array(iterations),
        "backward_seconds": np.array(backward),
        "status_bits": bits,
    }


def environment(device):
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "warp": wp.config.version,
        "newton": newton.__version__,
        "device": str(device),
    }
    if str(device).startswith("cuda"):
        info["device_name"] = wp.get_device(device).name
        info["cuda_driver"] = ".".join(map(str, wp._src.context.runtime.driver_version))
    return info


def run_reference(fixture_path, out_path, reference_root, log_path):
    """Execute the reference runner in its own environment as a subprocess."""
    root = Path(reference_root).resolve()
    library_path = [str(root / "RealSim/deps/onetbb_install/lib"), "/usr/local/cuda/lib64"]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(library_path + [env.get("LD_LIBRARY_PATH", "")])
    command = [
        str(root / ".venv/bin/python"),
        "-I",
        str(Path(__file__).with_name("reference_runner.py")),
        "--reference-root",
        str(root),
        str(fixture_path),
        str(out_path),
    ]
    with open(log_path, "w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cases", nargs="*", help="Case names (default: all)")
    parser.add_argument("--reference-root", type=Path, help="Built reference clone; runs its runner per fixture")
    args = parser.parse_args()
    fixtures = case_fixtures()
    known = {str(f["case"]) for f in fixtures}
    unknown = set(args.cases or ()) - known
    if unknown:
        parser.error(f"unknown parity cases: {', '.join(sorted(unknown))}")
    wp.init()
    args.out.mkdir(parents=True, exist_ok=True)
    selected = [f for f in fixtures if not args.cases or f["case"] in args.cases]
    (args.out / "port_environment.json").write_text(json.dumps(environment(args.device), indent=2) + "\n")
    for fixture in selected:
        name = fixture["case"]
        fixture = finalize_fixture(fixture)
        path = args.out / f"{name}.fixture.npz"
        np.savez(path, **fixture)
        fixture = dict(np.load(path))
        print(f"{name}: port", flush=True)
        if int(fixture["benchmark_steps"]):
            np.savez(args.out / f"{name}.port.npz", **benchmark_port(fixture, args.device))
        else:
            np.savez(args.out / f"{name}.port.npz", **run_port(fixture, args.device))
        if args.reference_root is not None:
            print(f"{name}: reference", flush=True)
            run_reference(path, args.out / f"{name}.reference.npz", args.reference_root, args.out / f"{name}.log")


if __name__ == "__main__":
    main()
