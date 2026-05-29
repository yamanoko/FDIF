"""Bake mesh -> dense SDF voxel grid at an arbitrary resolution.

Uses `trimesh.proximity.signed_distance` as the backend. trimesh is a
pure-Python (numpy-backed) package with prebuilt wheels for both
Windows and Linux and **does not need a C++ compiler**. It is slower
than C++ alternatives like pysdf / libigl, but baking is a one-time
cost per (mesh, grid_size) — results are cached on disk.

Sign convention: standard SDF — negative inside, positive outside.
(trimesh returns positive-inside / negative-outside, so we negate.)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Tuple

import numpy as np

from fdslxsdf4seg.medshapenet.paths import get_sdf_dir

# Fraction of the [-1, 1] cube the normalized mesh occupies. Leaves
# margin so rotated copies stay inside the volume.
_FILL_FRACTION = 0.8


def _normalize_mesh(verts: np.ndarray) -> np.ndarray:
    """Center the mesh at origin and rescale so its half-extent equals
    `_FILL_FRACTION` along the longest axis (fits inside [-1, 1]^3)."""
    centroid = (verts.max(axis=0) + verts.min(axis=0)) * 0.5
    centered = verts - centroid
    extent = float(np.abs(centered).max())
    if extent < 1e-8:
        return centered
    return centered * (_FILL_FRACTION / extent)


def _voxel_query_points(D: int, H: int, W: int) -> np.ndarray:
    """Return (D*H*W, 3) query points in [-1, 1]^3 as (x, y, z) triples.

    Ordering matches the SDF volume shape (D, H, W) where axis 0 is z,
    axis 1 is y, axis 2 is x — i.e. flattening uses C order.
    """
    zs = np.linspace(-1.0, 1.0, D, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, H, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, W, dtype=np.float32)
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)


def _mesh_fingerprint(verts: np.ndarray, faces: np.ndarray) -> str:
    h = hashlib.blake2b(digest_size=12)
    h.update(verts.tobytes())
    h.update(faces.tobytes())
    return h.hexdigest()


def _cache_path(
    dataset: str,
    organ_key: str,
    fingerprint: str,
    grid_size: Tuple[int, int, int],
) -> Path:
    D, H, W = grid_size
    sub = get_sdf_dir() / dataset / organ_key
    sub.mkdir(parents=True, exist_ok=True)
    return sub / f"{fingerprint}_{D}x{H}x{W}.npz"


def bake(
    verts: np.ndarray,
    faces: np.ndarray,
    grid_size: Tuple[int, int, int],
    dataset: str,
    organ_key: str,
) -> np.ndarray:
    """Return a `grid_size`-shaped SDF volume for the given mesh.

    The mesh is normalized to fit `_FILL_FRACTION * [-1, 1]^3`, then
    the SDF is sampled on a uniform `[-1, 1]^3` grid. Result is cached
    on disk keyed by (dataset, organ_key, mesh fingerprint, grid_size).

    Returns:
        np.ndarray, shape (D, H, W), dtype float32. Negative inside.
    """
    D, H, W = (int(s) for s in grid_size)
    if D < 2 or H < 2 or W < 2:
        raise ValueError(f"grid_size must be >=2 per axis, got {grid_size}")

    fingerprint = _mesh_fingerprint(verts, faces)
    cache_file = _cache_path(dataset, organ_key, fingerprint, (D, H, W))
    if cache_file.exists():
        with np.load(cache_file) as data:
            return data["sdf"].astype(np.float32, copy=False)

    sdf_vol = _compute_sdf(verts, faces, (D, H, W))
    np.savez_compressed(cache_file, sdf=sdf_vol.astype(np.float32))
    return sdf_vol


# Query points processed per trimesh call. The library allocates an
# O(points * faces) intermediate, so we chunk to bound peak RAM.
_QUERY_CHUNK = 1024


def _compute_sdf(
    verts: np.ndarray,
    faces: np.ndarray,
    grid_size: Tuple[int, int, int],
) -> np.ndarray:
    try:
        import trimesh  # type: ignore
    except ImportError as e:
        raise ImportError(
            "trimesh is required for MedShapeNet SDF baking. "
            "Install via `pip install trimesh rtree` (works on Windows and Linux)."
        ) from e

    D, H, W = grid_size
    normalized = _normalize_mesh(verts.astype(np.float32, copy=False))
    mesh = trimesh.Trimesh(
        vertices=normalized,
        faces=faces.astype(np.int32, copy=False),
        process=False,
    )
    pts = _voxel_query_points(D, H, W)

    sd = np.empty(pts.shape[0], dtype=np.float32)
    for start in range(0, pts.shape[0], _QUERY_CHUNK):
        end = min(start + _QUERY_CHUNK, pts.shape[0])
        # trimesh: positive INSIDE, negative outside.
        chunk = trimesh.proximity.signed_distance(mesh, pts[start:end])
        sd[start:end] = np.asarray(chunk, dtype=np.float32)
    sdf = -sd  # flip to standard convention (negative inside)
    return sdf.reshape(D, H, W)
