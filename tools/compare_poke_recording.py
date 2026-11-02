#!/usr/bin/env python3
"""Compare a cached poke trajectory with full recorded triangular surfaces.

No simulation, registration, remeshing or per-frame alignment is performed.
CPU Warp queries unsigned vertex-to-triangle distances in both directions.
These distances measure surface agreement, not signed penetration or a
continuous surface Hausdorff distance; vertices are not area-weighted.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp
from scipy.spatial import cKDTree

from newton_solver_diff_phd.examples._poke_recording import load_poke_recording


@wp.kernel(enable_backward=False)
def _distances(mesh: wp.uint64, points: wp.array[wp.vec3], radius: float, out: wp.array[float]):
    i = wp.tid()
    query = wp.mesh_query_point_no_sign(mesh, points[i], radius)
    out[i] = -1.0
    if query.result:
        closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        out[i] = wp.length(points[i] - closest)


def _points(values, name):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite nonempty (N,3) array")
    if np.max(np.abs(values)) >= np.finfo(np.float32).max:
        raise ValueError(f"{name} is outside Warp float32 geometry range")
    return values.astype(np.float64, copy=False)


def _faces(values, count, name):
    values = np.asarray(values)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or len(values) == 0
        or not np.issubdtype(values.dtype, np.integer)
        or np.any(values < 0)
        or np.any(values >= count)
    ):
        raise ValueError(f"{name} must contain nonempty valid triangle indices")
    return values.astype(np.int32, copy=False)


def _mesh(vertices, faces):
    return wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3, device="cpu"),
        indices=wp.array(faces.ravel(), dtype=wp.int32, device="cpu"),
    )


def _query(mesh, points, radius):
    source = wp.array(points, dtype=wp.vec3, device="cpu")
    output = wp.empty(len(points), dtype=float, device="cpu")
    wp.launch(_distances, len(points), inputs=[mesh.id, source, radius], outputs=[output], device="cpu")
    result = output.numpy().astype(np.float64)
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise ValueError("A nearest-triangle query failed; no missing distances are silently discarded")
    return result


def _stats(distances):
    return {
        "rms_m": float(np.sqrt(np.mean(distances**2))),
        "p95_m": float(np.percentile(distances, 95)),
        "max_m": float(np.max(distances)),
        "vertices": len(distances),
    }


def _metrics(forward, reverse):
    left, right = _stats(forward), _stats(reverse)
    return {
        "white_to_scan": left,
        "scan_to_white": right,
        "symmetric_rms_m": float(np.sqrt(0.5 * (left["rms_m"] ** 2 + right["rms_m"] ** 2))),
        "symmetric_p95_m": max(left["p95_m"], right["p95_m"]),
        "max_distance_m": max(left["max_m"], right["max_m"]),
    }


def _chamfer(points, target):
    forward = cKDTree(target).query(points, workers=1)[0]
    reverse = cKDTree(points).query(target, workers=1)[0]
    return float(0.5 * (np.mean(forward**2) + np.mean(reverse**2)))


def _aggregate(frames, kind):
    values = [frame[kind] for frame in frames[1:]]
    result = {
        "frames": len(values),
        "symmetric_rms_m": float(np.sqrt(np.mean([row["symmetric_rms_m"] ** 2 for row in values]))),
        "mean_frame_symmetric_p95_m": float(np.mean([row["symmetric_p95_m"] for row in values])),
        "worst_frame_symmetric_p95_m": max(row["symmetric_p95_m"] for row in values),
        "max_distance_m": max(row["max_distance_m"] for row in values),
    }
    for direction in ("white_to_scan", "scan_to_white"):
        result[f"{direction}_rms_m"] = float(np.sqrt(np.mean([row[direction]["rms_m"] ** 2 for row in values])))
    if "training_chamfer_m2" in values[0]:
        result["training_chamfer_sum_m2"] = float(sum(row["training_chamfer_m2"] for row in values))
        result["training_chamfer_rms_m"] = float(np.sqrt(result["training_chamfer_sum_m2"] / len(values)))
    return result


def compare_recording(data, positions):
    """Return JSON-compatible metrics and optional arrays for local plotting.

    ``positions`` has one (all-particle) frame per recording frame, in unchanged
    recording coordinates. Extra particles after the die are ignored by the
    surface embedding. Both inputs remain unchanged.
    """
    frame_ids = np.asarray(data["frame_ids"])
    if (
        frame_ids.ndim != 1
        or len(frame_ids) < 2
        or not np.issubdtype(frame_ids.dtype, np.integer)
        or np.any(np.diff(frame_ids) <= 0)
    ):
        raise ValueError("frame_ids must contain at least two increasing integer observation frames")
    positions = np.asarray(positions)
    die = _points(data["vertices"], "die vertices")
    if (
        positions.ndim != 3
        or positions.shape[0] != len(frame_ids)
        or positions.shape[1] < len(die)
        or positions.shape[2] != 3
        or not np.isfinite(positions).all()
    ):
        raise ValueError("positions must be finite (recording_frames, particles_at_least_die_count, 3)")
    indices, weights = np.asarray(data["surface_embedding_indices"]), np.asarray(data["surface_embedding_weights"])
    if (
        indices.ndim != 2
        or indices.shape[1] != 4
        or indices.shape != weights.shape
        or not np.issubdtype(indices.dtype, np.integer)
        or np.any(indices < 0)
        or np.any(indices >= len(die))
        or not np.isfinite(weights).all()
        or not np.allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-7)
    ):
        raise ValueError("surface embedding needs valid die indices and four finite affine weights per vertex")
    white = np.einsum("sv,fsvj->fsj", weights, positions[:, indices].astype(np.float64))
    _points(white.reshape(-1, 3), "embedded white vertices")
    faces = _faces(data["surface_faces"], len(indices), "white surface faces")
    recorded_fields = ("recorded_vertices", "recorded_vertex_offsets", "recorded_faces", "recorded_face_offsets")
    if any(name not in data for name in recorded_fields):
        raise ValueError("Full recorded triangle meshes are required; point samples are not a mesh substitute")
    scan_vertices = _points(data["recorded_vertices"], "recorded vertices")
    scan_faces = np.asarray(data["recorded_faces"])
    offsets, face_offsets = (np.asarray(data[name]) for name in recorded_fields[1::2])
    for values, offset in ((scan_vertices, offsets), (scan_faces, face_offsets)):
        if (
            offset.shape != (len(frame_ids) + 1,)
            or not np.issubdtype(offset.dtype, np.integer)
            or offset[0] != 0
            or offset[-1] != len(values)
            or np.any(np.diff(offset) <= 0)
        ):
            raise ValueError("Recorded mesh offsets must partition exactly one nonempty mesh per frame")
    sample_ids = targets = None
    if "target_vertex_indices" in data and "target_points" in data:
        all_samples = np.asarray(data["target_vertex_indices"])
        if all_samples.ndim != 2 or len(all_samples) != len(frame_ids):
            raise ValueError("Training sample indices must have one row per recording frame")
        sample_ids = all_samples[0]
        targets = np.asarray(data["target_points"])
        if (
            sample_ids.ndim != 1
            or len(sample_ids) == 0
            or not np.issubdtype(sample_ids.dtype, np.integer)
            or np.any(sample_ids < 0)
            or np.any(sample_ids >= len(indices))
            or targets.ndim != 3
            or targets.shape[0] != len(frame_ids)
            or targets.shape[1] == 0
            or targets.shape[2] != 3
            or not np.isfinite(targets).all()
        ):
            raise ValueError("Training samples must index the white surface and provide finite targets for every frame")
    frozen_mesh = _mesh(white[0], faces)
    frames, distances = (
        [],
        {key: [] for key in ("white_to_scan", "scan_to_white", "frozen_white_to_scan", "scan_to_frozen_white")},
    )
    for i, frame_id in enumerate(frame_ids):
        scan = scan_vertices[offsets[i] : offsets[i + 1]]
        triangles = _faces(scan_faces[face_offsets[i] : face_offsets[i + 1]], len(scan), f"recorded frame {i}")
        current_mesh, scan_mesh = _mesh(white[i], faces), _mesh(scan, triangles)
        extent = np.vstack((white[i], white[0], scan))
        radius = float(np.linalg.norm(np.ptp(extent, axis=0)) + 1.0)
        current = (_query(scan_mesh, white[i], radius), _query(current_mesh, scan, radius))
        frozen = (_query(scan_mesh, white[0], radius), _query(frozen_mesh, scan, radius))
        row = {
            "index": i,
            "frame_id": int(frame_id),
            "trajectory": _metrics(*current),
            "frozen_frame0": _metrics(*frozen),
        }
        if sample_ids is not None:
            row["trajectory"]["training_chamfer_m2"] = _chamfer(white[i, sample_ids], targets[i])
            row["frozen_frame0"]["training_chamfer_m2"] = _chamfer(white[0, sample_ids], targets[i])
        frames.append(row)
        for name, values in zip(distances, (*current, *frozen), strict=True):
            distances[name].append(values)
    aggregate = {kind: _aggregate(frames, kind) for kind in ("trajectory", "frozen_frame0")}
    baseline, actual = (aggregate[kind]["symmetric_rms_m"] for kind in ("frozen_frame0", "trajectory"))
    aggregate["rms_improvement_over_frozen_m"] = baseline - actual
    aggregate["rms_improvement_over_frozen_fraction"] = (baseline - actual) / baseline if baseline > 0.0 else None
    report = {
        "format": "newton.diff_phd.poke_surface_comparison.v1",
        "units": "metres, except training_chamfer_m2 and training_chamfer_sum_m2 in square metres",
        "definitions": {
            "directional": "Unsigned nearest-triangle distance from every source mesh vertex to the full opposite mesh; RMS, numpy linear p95, maximum.",
            "symmetric_rms": "sqrt(0.5 * (mean(white_to_scan^2) + mean(scan_to_white^2)))",
            "symmetric_p95": "max(p95(white_to_scan), p95(scan_to_white)); not a pooled percentile",
            "aggregate": "Frame 0 excluded; RMS gives equal weight to each remaining frame and to each direction, regardless of vertex counts.",
            "frozen_frame0": "The embedded white surface at trajectory frame 0 is held fixed in world coordinates against every recorded frame.",
            "training_chamfer": "When samples exist: 0.5*mean nearest-vertex squared distance each direction, using target_vertex_indices[0] and target_points[frame], as in fast training. Sum excludes frame 0.",
        },
        "scope": "No rigid registration, per-frame alignment, simulation or geometry modification. CPU Warp mesh queries use float32 geometry; embeddings and metric reductions use float64. Vertex sampling is not surface-area weighting, signed penetration or a continuous Hausdorff distance. No collision or tetrahedron-inversion pass can be inferred from these metrics. Frozen baseline includes initial placement error.",
        "counts": {
            "frames": len(frames),
            "die_particles": len(die),
            "trajectory_particles": positions.shape[1],
            "ignored_extra_particles": positions.shape[1] - len(die),
            "white_vertices": len(indices),
            "white_triangles": len(faces),
            "training_white_samples": None if sample_ids is None else len(sample_ids),
        },
        "frames": frames,
        "aggregate_excluding_frame0": aggregate,
    }
    arrays = {"white_vertices": white, "frame_ids": frame_ids, "recorded_vertex_offsets": offsets.copy()}
    for name, values in distances.items():
        arrays[name] = np.stack(values) if name.endswith("to_scan") else np.concatenate(values)
    return report, arrays


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--positions-key", default="positions")
    parser.add_argument("--output", type=Path, help="Write metrics JSON; otherwise print it.")
    parser.add_argument(
        "--distances-output", type=Path, help="Optional local NPZ of embedded white vertices and distances."
    )
    args = parser.parse_args(argv)
    wp.config.log_level = wp.LOG_WARNING
    data = load_poke_recording(args.recording)
    with np.load(args.trajectory, allow_pickle=False) as archive:
        if args.positions_key not in archive:
            raise ValueError(f"Trajectory does not contain positions key {args.positions_key!r}")
        positions = archive[args.positions_key].copy()
        if "frame_ids" in archive and not np.array_equal(archive["frame_ids"], data["frame_ids"]):
            raise ValueError("Trajectory frame_ids must match the recording exactly; no frame alignment is performed")
    report, arrays = compare_recording(data, positions)
    paths = {"recording": args.recording, "trajectory": args.trajectory, "comparison_tool": Path(__file__)}
    if args.recording.suffix.lower() == ".json":
        manifest = json.loads(args.recording.read_text())
        for key in ("geometry", "recording"):
            paths[f"manifest_{key}"] = args.recording.parent / manifest[key]
    report["inputs"] = {key: {"path": str(path.resolve()), "sha256": _hash(path)} for key, path in paths.items()}
    report["positions_key"] = args.positions_key
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    else:
        print(serialized, end="")
    if args.distances_output:
        args.distances_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.distances_output, **arrays)
    return report


if __name__ == "__main__":
    main()
