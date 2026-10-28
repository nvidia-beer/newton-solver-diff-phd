# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Load local recorded scans and their static USD simulation embedding."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .usd import load_mesh


def load_poke_recording(path: str | Path) -> dict[str, np.ndarray]:
    """Read a local recording manifest or a legacy prepared NPZ archive.

    The manifest separates geometry-only USD data from measured scan samples.
    Every returned NumPy array is independently owned. No download, remeshing,
    material assignment, or simulation is performed by this reader.
    PokeFlex recordings remain subject to their original access terms.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Local PokeFlex recording not found: {path}. Private PokeFlex scans are excluded from "
            "distributions; supply your locally prepared recording with --recording PATH."
        )
    if path.suffix.lower() == ".json":
        manifest = json.loads(path.read_text())
        if manifest.get("format") != "newton.diff_phd.poke_recording.v1":
            raise ValueError("Unsupported local PokeFlex recording manifest")
        archive_path = path.parent / manifest["recording"]
        geometry_path = path.parent / manifest["geometry"]
        for field, asset in (("recording_sha256", archive_path), ("geometry_sha256", geometry_path)):
            if manifest.get(field) != hashlib.sha256(asset.read_bytes()).hexdigest():
                raise ValueError(f"Recording manifest checksum mismatch: {asset}")
        with np.load(archive_path, allow_pickle=False) as archive:
            result = {name: archive[name].copy() for name in archive.files}
        from pxr import Usd  # noqa: PLC0415 -- optional geometry dependency

        mesh = load_mesh(geometry_path)[0]
        stage = Usd.Stage.Open(str(geometry_path))
        surface = stage.GetPrimAtPath("/Die/Surface")
        volume = stage.GetPrimAtPath("/Die/Volume")
        result.update(
            vertices=mesh.vertices,
            tets=mesh.indices,
            faces=np.asarray(volume.GetAttribute("surfaceFaceVertexIndices").Get(), dtype=np.int32),
            surface_vertices=np.asarray(surface.GetAttribute("newton:referencePoints").Get(), dtype=np.float64),
            surface_faces=np.asarray(surface.GetAttribute("faceVertexIndices").Get(), dtype=np.int32).reshape(-1, 3),
            surface_uv=np.asarray(surface.GetAttribute("primvars:st").Get(), dtype=np.float64),
            surface_embedding_indices=np.asarray(surface.GetAttribute("newton:embeddingIndices").Get(), dtype=np.int32),
            surface_embedding_weights=np.asarray(
                surface.GetAttribute("newton:embeddingWeights").Get(), dtype=np.float64
            ),
        )
    else:
        with np.load(path, allow_pickle=False) as archive:
            result = {name: archive[name].copy() for name in archive.files}
    required = (
        "vertices",
        "tets",
        "faces",
        "pinned",
        "surface_vertices",
        "surface_faces",
        "surface_uv",
        "surface_embedding_indices",
        "surface_embedding_weights",
        "frame_ids",
        "target_points",
        "target_vertex_indices",
        "reference_xy",
    )
    if any(name not in result for name in required):
        raise ValueError(f"Incomplete prepared PokeFlex recording: {path}")
    for name in ("vertices", "surface_vertices", "target_points", "surface_embedding_weights"):
        if not np.isfinite(result[name]).all():
            raise ValueError(f"Nonfinite recording field {name}: {path}")
    indices = result["surface_embedding_indices"]
    if indices.shape != result["surface_embedding_weights"].shape or indices.shape != (
        len(result["surface_vertices"]),
        4,
    ):
        raise ValueError("Recorded surface embedding must have four indices and weights per vertex")
    if (
        not np.issubdtype(indices.dtype, np.integer)
        or np.any(indices < 0)
        or np.any(indices >= len(result["vertices"]))
    ):
        raise ValueError("Recorded surface embedding index outside the volume")
    recorded = ("recorded_vertices", "recorded_vertex_offsets", "recorded_faces", "recorded_face_offsets")
    if any(name in result for name in recorded):
        if any(name not in result for name in recorded):
            raise ValueError(f"Incomplete recorded scan meshes: {path}")
        for values_name, offsets_name in ((recorded[0], recorded[1]), (recorded[2], recorded[3])):
            values, offsets = result[values_name], result[offsets_name]
            if (
                values.ndim != 2
                or values.shape[1] != 3
                or offsets.shape != (len(result["frame_ids"]) + 1,)
                or not np.issubdtype(offsets.dtype, np.integer)
                or offsets[0] != 0
                or offsets[-1] != len(values)
                or np.any(np.diff(offsets) <= 0)
            ):
                raise ValueError(f"Invalid packed recorded scan meshes: {path}")
        if not np.isfinite(result["recorded_vertices"]).all() or not np.issubdtype(
            result["recorded_faces"].dtype, np.integer
        ):
            raise ValueError(f"Invalid recorded vertices or faces: {path}")
        for i, vertex_count in enumerate(np.diff(result["recorded_vertex_offsets"])):
            start, end = result["recorded_face_offsets"][i : i + 2]
            faces = result["recorded_faces"][start:end]
            if np.any(faces < 0) or np.any(faces >= vertex_count):
                raise ValueError(f"Recorded face index outside frame {i}: {path}")
    return result
