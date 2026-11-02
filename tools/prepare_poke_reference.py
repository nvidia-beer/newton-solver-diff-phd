# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Bake local Atlas scans with the original DiffPhD die/cylinder meshing recipe.

Offline preparation only; pyvista, tetgen, trimesh and OpenUSD are preparation
requirements, not solver runtime dependencies. Input scans and derived assets
remain user-local and are never downloaded or redistributed by this tool.
"""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boundary_faces(vertices, tets):
    faces = {}
    for tet in tets:
        for a, b, c, d in ((0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 3, 1), (1, 2, 3, 0)):
            face = tet[[a, b, c]]
            key = tuple(sorted(face))
            if key in faces:
                del faces[key]
                continue
            x = vertices[face]
            if np.dot(np.cross(x[1] - x[0], x[2] - x[0]), vertices[tet[d]] - x[0]) > 0:
                face = face[[0, 2, 1]]
            faces[key] = face
    return np.array(list(faces.values()), dtype=np.int32)


def positive_tets(vertices, tets):
    tets = np.asarray(tets[:, :4], dtype=np.int32).copy()
    invert = np.linalg.det(vertices[tets[:, 1:]] - vertices[tets[:, :1]]) < 0
    tets[invert] = tets[invert][:, [0, 2, 1, 3]]
    return tets


def prepare_geometry(source, directory):
    """Literal geometry operations from ur5_poke_demo.py, without its renderer."""
    import pyvista as pv
    import tetgen
    import trimesh

    raw = trimesh.load(str(source / "mesh-f00018.obj"), process=False)
    hull_path = directory / "hull.stl"
    trimesh.convex.convex_hull(np.asarray(raw.vertices)).export(hull_path)
    hull = pv.read(hull_path).decimate(0.85).clean()
    points = np.asarray(hull.points, dtype=np.float64)
    centroid = points.mean(axis=0)
    # rpy_to_R(90deg, 0, 0), preserving its floating cosine of pi/2.
    c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
    rotation = np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))
    rotated = (rotation @ (points - centroid).T).T + centroid
    scale = 0.30 / max(rotated.max(axis=0) - rotated.min(axis=0))
    points = rotated * scale
    translation = np.array((0.70 - points[:, 0].mean(), 0.089 - points[:, 1].mean(), -0.14 - points[:, 2].max()))
    points += translation
    hull.points = points
    volume = tetgen.TetGen(hull)
    volume.make_manifold()
    vertices, tets = volume.tetrahedralize(minratio=1.3, order=1)[:2]
    vertices = np.asarray(vertices, dtype=np.float64)
    pinned = np.flatnonzero(vertices[:, 2] <= np.quantile(vertices[:, 2], 0.10)).astype(np.int32)
    tets = positive_tets(vertices, tets)
    # The original prep_cylinder_tet exports once; tetrahedralize reloads and
    # exports a second STL with trimesh's default processing before pv.read.
    cylinder = trimesh.creation.cylinder(radius=0.01, height=0.10, sections=24)
    cylinder.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0.0, 1.0, 0.0]))
    cylinder.apply_translation([0.05, 0.0, 0.0])
    cylinder_path = directory / "cylinder.stl"
    cylinder.export(cylinder_path)
    cylinder = trimesh.load(cylinder_path)
    if not cylinder.is_watertight:
        raise ValueError("Original cylinder surface is not watertight")
    cylinder.export(cylinder_path)
    cylinder_volume = tetgen.TetGen(pv.read(cylinder_path))
    cylinder_volume.make_manifold()
    probe, probe_tets = cylinder_volume.tetrahedralize(minratio=1.2)[:2]
    # filter_unused_vertices, as in the original common.tet_mesh helper.
    used = np.unique(probe_tets[:, :4])
    remap = np.full(len(probe), -1, dtype=np.int32)
    remap[used] = np.arange(len(used))
    probe, probe_tets = np.asarray(probe, dtype=np.float64)[used], remap[probe_tets[:, :4]]
    # Existing runtime tool frame applies Ry(pi/2) to +Z-oriented local assets.
    # Inverse map (x,y,z) -> (-z,y,x) is proper and preserves topology.
    probe = probe[:, [2, 1, 0]] * [-1, 1, 1]
    probe_tets = positive_tets(probe, probe_tets)
    vertices[:, 2] += 0.5
    translation[2] += 0.5
    return vertices, tets, pinned, probe, probe_tets, raw, centroid, rotation, scale, translation


def embedding(points, vertices, tets):
    """Embed in a containing cell, retaining explicit extrapolation outside.

    Nearest centroids are a fast initial search, but elongated tetrahedra can
    contain a point despite having a distant centroid. The fallback searches
    every possible containing cell with conservative geometric bounds.
    """
    cells = vertices[tets]
    origins = vertices[tets[:, 0]]
    inverse = np.linalg.inv(np.transpose(vertices[tets[:, 1:]] - origins[:, None], (0, 2, 1)))
    centers = cells.mean(axis=1)
    tree = cKDTree(centers)
    candidates = tree.query(points, k=min(128, len(tets)))[1].reshape(len(points), min(128, len(tets)))
    indices = np.empty((len(points), 4), dtype=np.int32)
    weights = np.empty((len(points), 4), dtype=np.float64)
    for start in range(0, len(points), 256):
        end = min(start + 256, len(points))
        candidate = candidates[start:end]
        bary = np.einsum("nkij,nkj->nki", inverse[candidate], points[start:end, None] - origins[candidate])
        bary = np.concatenate((1 - bary.sum(axis=2, keepdims=True), bary), axis=2)
        # Extrapolation is explicit where the decimated original hull excludes
        # part of the scanned surface; this does not change physics geometry.
        score = np.maximum(-bary, 0.0).sum(axis=2)
        best = np.argmin(score, axis=1)
        indices[start:end] = tets[candidate[np.arange(end - start), best]]
        weights[start:end] = bary[np.arange(end - start), best]

    unresolved = np.flatnonzero(weights.min(axis=1) < -1e-10)
    if len(unresolved):
        # A convex combination of the corners cannot be farther from the
        # centroid than the most distant corner. This radius, followed by the
        # tetra AABB, excludes no containing cells; fixed nearest-k can do so.
        radius = np.linalg.norm(cells - centers[:, None], axis=2).max()
        padding = 64 * np.finfo(np.float64).eps * max(1.0, np.abs(vertices).max())
        lower, upper = cells.min(axis=1), cells.max(axis=1)
        complete = tree.query_ball_point(points[unresolved], radius + padding)
        for point_index, candidate_ids in zip(unresolved, complete, strict=True):
            candidate = np.asarray(sorted(candidate_ids), dtype=np.int32)
            point = points[point_index]
            candidate = candidate[
                np.all((point >= lower[candidate] - padding) & (point <= upper[candidate] + padding), axis=1)
            ]
            if not len(candidate):
                continue
            bary = np.einsum("nij,nj->ni", inverse[candidate], point - origins[candidate])
            bary = np.column_stack((1 - bary.sum(axis=1), bary))
            containing = np.flatnonzero(bary.min(axis=1) >= -1e-10)
            if len(containing):
                score = np.maximum(-bary[containing], 0.0).sum(axis=1)
                best = containing[np.argmin(score)]
                indices[point_index] = tets[candidate[best]]
                weights[point_index] = bary[best]
            # A genuinely outside scan vertex keeps its explicit affine
            # extrapolation; neither the scan nor the physics mesh is moved.
    error = np.linalg.norm(np.einsum("ni,nij->nj", weights, vertices[indices]) - points, axis=1).max()
    if error > 1e-12:
        raise ValueError(f"Embedding rest reconstruction error {error}")
    return indices, weights, float(error)


def quality(vertices, tets, pinned):
    x = vertices[tets]
    det = np.linalg.det(x[:, 1:] - x[:, :1])
    edge_square = sum(np.sum((x[:, a] - x[:, b]) ** 2, axis=1) for a in range(4) for b in range(a))
    ratio = 12 * (det / 2) ** (2 / 3) / edge_square
    pin_mask = np.zeros(len(vertices), dtype=bool)
    pin_mask[pinned] = True
    pin_counts = pin_mask[tets].sum(axis=1)
    return {
        "vertices": len(vertices),
        "tets": len(tets),
        "boundary_faces": len(boundary_faces(vertices, tets)),
        "pinned": len(pinned),
        "positive_volume": bool(np.all(det > 0)),
        "min_volume_m3": float(det.min() / 6),
        "mean_ratio_min": float(ratio.min()),
        "mean_ratio_p01": float(np.quantile(ratio, 0.01)),
        "three_pinned_tets": int(np.count_nonzero(pin_counts == 3)),
    }


def write_geometry(path, vertices, tets, faces, surface, surface_faces, uv, indices, weights, probe, probe_tets):
    from pxr import Sdf, Usd, UsdGeom, Vt

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, "/Die")
    for name, points, cells in (("/Die/Volume", vertices, tets), ("/Probe", probe, probe_tets)):
        prim = stage.DefinePrim(name, "TetMesh")
        prim.CreateAttribute("points", Sdf.ValueTypeNames.Point3fArray).Set(
            Vt.Vec3fArray.FromNumpy(points.astype(np.float32))
        )
        prim.CreateAttribute("newton:referencePoints", Sdf.ValueTypeNames.Point3dArray).Set(
            Vt.Vec3dArray.FromNumpy(points)
        )
        prim.CreateAttribute("tetVertexIndices", Sdf.ValueTypeNames.Int4Array).Set(Vt.Vec4iArray.FromNumpy(cells))
    volume = stage.GetPrimAtPath("/Die/Volume")
    volume.CreateAttribute("surfaceFaceVertexIndices", Sdf.ValueTypeNames.Int3Array).Set(Vt.Vec3iArray.FromNumpy(faces))
    mesh = UsdGeom.Mesh.Define(stage, "/Die/Surface")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(surface.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr([3] * len(surface_faces))
    mesh.CreateFaceVertexIndicesAttr(surface_faces.ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    prim = mesh.GetPrim()
    prim.CreateAttribute("newton:referencePoints", Sdf.ValueTypeNames.Point3dArray).Set(
        Vt.Vec3dArray.FromNumpy(surface)
    )
    prim.CreateAttribute("primvars:st", Sdf.ValueTypeNames.TexCoord2fArray).Set(
        Vt.Vec2fArray.FromNumpy(uv.astype(np.float32))
    )
    prim.CreateAttribute("newton:embeddingIndices", Sdf.ValueTypeNames.Int4Array).Set(Vt.Vec4iArray.FromNumpy(indices))
    prim.CreateAttribute("newton:embeddingWeights", Sdf.ValueTypeNames.Double4Array).Set(
        Vt.Vec4dArray.FromNumpy(weights)
    )
    stage.GetRootLayer().Save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    import pyvista as pv
    import tetgen
    import trimesh

    source = args.reference_root / "asset/mesh/atlas_dice"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="diffphd-reference-") as temporary:
        vertices, tets, pinned, probe, probe_tets, raw, centroid, rotation, scale, translation = prepare_geometry(
            source, Path(temporary)
        )

    def transform(points):
        return ((rotation @ (np.asarray(points) - centroid).T).T + centroid) * scale + translation

    surface = transform(raw.vertices)
    uv = np.asarray(raw.visual.uv) if hasattr(raw.visual, "uv") else np.zeros((len(surface), 2))
    indices, weights, embed_error = embedding(surface, vertices, tets)
    frames, targets, sample_indices, all_vertices, all_faces = [], [], [], [], []
    vertex_offsets, face_offsets, source_hashes = [0], [0], {}
    for frame in range(18, 54):
        path = source / f"mesh-f{frame:05d}.obj"
        scan = trimesh.load(path, process=False)
        points = transform(scan.vertices)
        ids = np.linspace(0, len(points) - 1, 1024, dtype=np.int32)
        frames.append(frame)
        targets.append(points[ids])
        sample_indices.append(ids)
        all_vertices.append(points)
        all_faces.append(np.asarray(scan.faces, dtype=np.int32))
        vertex_offsets.append(vertex_offsets[-1] + len(points))
        face_offsets.append(face_offsets[-1] + len(scan.faces))
        source_hashes[path.name] = digest(path)
    geometry_path = output / "die_reference.usdc"
    write_geometry(
        geometry_path,
        vertices,
        tets,
        boundary_faces(vertices, tets),
        surface,
        np.asarray(raw.faces, dtype=np.int32),
        uv,
        indices,
        weights,
        probe,
        probe_tets,
    )
    archive_path = output / "recording_reference.npz"
    np.savez_compressed(
        archive_path,
        pinned=pinned,
        frame_ids=np.array(frames, dtype=np.int32),
        target_points=np.array(targets),
        target_vertex_indices=np.array(sample_indices),
        reference_xy=np.array([0.70, 0.089]),
        reference_mesh=np.array(True),
        transform_centroid=centroid,
        transform_rotation=rotation,
        transform_scale=scale,
        transform_translation=translation,
        recorded_vertices=np.concatenate(all_vertices),
        recorded_faces=np.concatenate(all_faces),
        recorded_vertex_offsets=np.array(vertex_offsets),
        recorded_face_offsets=np.array(face_offsets),
    )
    preparation = {
        "tool": "tools/prepare_poke_reference.py",
        "recipe": "original ur5_poke_demo.py and dice_material_optimize.py geometry defaults",
        "reference_scripts_sha256": {
            n: digest(args.reference_root / "python/example" / n)
            for n in ["ur5_poke_demo.py", "dice_material_optimize.py"]
        },
        "source_mesh_sha256": source_hashes,
        "versions": {m.__name__: m.__version__ for m in [np, pv, tetgen, trimesh]},
        "die_hull_decimation": 0.85,
        "die_tetgen_minratio": 1.3,
        "bottom_quantile": 0.10,
        "world_z_offset_m": 0.5,
        "probe_radius_m": 0.01,
        "probe_length_m": 0.10,
        "probe_sections": 24,
        "probe_tetgen_minratio": 1.2,
        "tetgen_api_compatibility": "Use the first two results; recent TetGen returns optional extra arrays.",
        "die_quality": quality(vertices, tets, pinned),
        "probe_quality": quality(probe, probe_tets, np.flatnonzero(probe[:, 2] < 1e-4)),
        "embedding_max_rest_error_m": embed_error,
        "embedding_search": "Nearest 128 centroids, then complete radius/AABB containment fallback; outside vertices retain affine extrapolation.",
        "embedding_min_weight": float(weights.min()),
        "embedding_outside_fraction": float(np.mean(weights.min(axis=1) < -1e-8)),
    }
    manifest = {
        "format": "newton.diff_phd.poke_recording.v1",
        "geometry": geometry_path.name,
        "recording": archive_path.name,
        "geometry_sha256": digest(geometry_path),
        "recording_sha256": digest(archive_path),
        "notice": "Private local PokeFlex-derived data; do not redistribute, including converted data.",
        "preparation": preparation,
    }
    manifest_path = output / "recording_reference.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "preparation": preparation}, indent=2))


if __name__ == "__main__":
    main()
