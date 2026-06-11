"""Bake mesh -> dense SDF voxel grid at an arbitrary resolution.

Backends (selectable via `FDSL_SDF_BACKEND`):

* "warp" (**default**) — NVIDIA Warp's GPU mesh **BVH**. Each query
  point does an O(log F) closest-point lookup against a bounding-volume
  hierarchy instead of scanning every triangle. A single
  `pip install warp-lang` (prebuilt wheels, no compiler), runs on CUDA
  (H100, laptop GPUs) and falls back to CPU. High-poly organ meshes that
  took minutes under brute force bake in a fraction of a second.

  Inside/outside is resolved by `FDSL_SDF_WARP_SIGN` (default
  "floodfill"): MedShapeNet meshes are non-watertight "soups", so every
  surface-based sign method (winding number, pseudonormal, and the legacy
  trimesh baseline) leaks through holes and yields broken interiors. The
  flood-fill solidifier instead seals sub-voxel holes and fills the
  enclosed region, giving clean organ masks. "normal"/"winding" remain
  available for watertight meshes.

* "torch" — pure-PyTorch, GPU-vectorized **brute force**
  (`_compute_sdf_torch`): closest-point-on-triangle + winding number,
  evaluated against *all* faces. Correct and dependency-light, but
  O(points * faces) — only practical for low-poly meshes. Used
  automatically if Warp is unavailable.

* "trimesh" — legacy CPU baker (`_compute_sdf_trimesh`), kept as a
  reference. Slowest.

Baking is a one-time cost per (mesh, grid_size) — results are cached on
disk regardless of backend.

Sign convention: standard SDF — negative inside, positive outside.
"""

from __future__ import annotations

import hashlib
import os
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


def _compute_sdf(
    verts: np.ndarray,
    faces: np.ndarray,
    grid_size: Tuple[int, int, int],
) -> np.ndarray:
    """Dispatch to the configured SDF backend.

    Backend selection (env `FDSL_SDF_BACKEND`):
        * "warp" (default): GPU mesh-BVH baker (NVIDIA Warp). Falls back
          to the torch brute-force baker if Warp can't be imported.
        * "torch": GPU-vectorized pure-PyTorch brute-force baker.
        * "trimesh": legacy CPU baker (slow; kept as reference).
    """
    backend = os.environ.get("FDSL_SDF_BACKEND", "warp").lower()
    if backend == "trimesh":
        return _compute_sdf_trimesh(verts, faces, grid_size)
    if backend == "torch":
        return _compute_sdf_torch(verts, faces, grid_size)
    try:
        return _compute_sdf_warp(verts, faces, grid_size)
    except ImportError:
        # Warp not installed — fall back to the dependency-light backend.
        return _compute_sdf_torch(verts, faces, grid_size)


# ----------------------------------------------------------------------
# GPU mesh-BVH backend (default) — NVIDIA Warp
# ----------------------------------------------------------------------

# Lazily built once: (initialized?, {sign_mode: compiled kernel}). Warp
# kernels must be defined after `import warp`, and `wp.init()` is
# comparatively expensive, so we do both a single time on first bake.
_WARP_STATE = {"ready": False, "kernels": {}}

# Faces farther than this from a query point are ignored. Our normalized
# meshes live in [-1, 1]^3, so the largest possible point->surface distance
# is well under this; the value just disables distance-based pruning.
_WARP_MAX_DIST = 1.0e6

# How inside/outside is resolved on the BVH unsigned distance:
#   "floodfill" — DEFAULT. Ignore the mesh's (unreliable) orientation
#                 entirely: treat near-surface voxels as walls and
#                 flood-fill free space from the volume border; whatever
#                 the flood can't reach is interior. Robust to the
#                 non-watertight "soup" meshes in MedShapeNet, which break
#                 every surface-based sign method. Needs the full grid in
#                 RAM (we have it) + scipy.ndimage.
#   "normal"    — angle-weighted pseudonormal of the nearest face. Cheap,
#                 reproduces the legacy trimesh.signed_distance baseline,
#                 but inherits its leak-through artifacts on open meshes.
#   "winding"   — generalized winding number. Unstable on overlapping-face
#                 meshes (winding spikes to +-3..7 in the interior).
_WARP_SIGN_MODE = os.environ.get("FDSL_SDF_WARP_SIGN", "floodfill").lower()

# Flood-fill wall thickness, in multiples of the voxel diagonal. A voxel
# closer than this to the surface is treated as a wall the flood cannot
# cross, sealing small (sub-wall) holes in non-watertight meshes.
_FLOODFILL_WALL = float(os.environ.get("FDSL_SDF_FLOODFILL_WALL", "1.5"))


def _build_warp_kernels(wp):
    """Compile the sign-mode kernels once and cache them."""

    @wp.kernel
    def _sdf_unsigned(
        mesh_id: wp.uint64,
        points: wp.array(dtype=wp.vec3),
        max_dist: wp.float32,
        epsilon: wp.float32,  # unused; keeps a uniform launch signature
        out: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        p = points[tid]
        q = wp.mesh_query_point_no_sign(mesh_id, p, max_dist)
        if q.result:
            cp = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
            out[tid] = wp.length(p - cp)
        else:
            out[tid] = max_dist

    @wp.kernel
    def _sdf_normal(
        mesh_id: wp.uint64,
        points: wp.array(dtype=wp.vec3),
        max_dist: wp.float32,
        epsilon: wp.float32,
        out: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        p = points[tid]
        q = wp.mesh_query_point_sign_normal(mesh_id, p, max_dist, epsilon)
        if q.result:
            cp = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
            d = wp.length(p - cp)
            # Warp: q.sign < 0 inside. Standard SDF: negative inside.
            s = float(1.0)
            if q.sign < 0.0:
                s = -1.0
            out[tid] = s * d
        else:
            out[tid] = max_dist

    @wp.kernel
    def _sdf_winding(
        mesh_id: wp.uint64,
        points: wp.array(dtype=wp.vec3),
        max_dist: wp.float32,
        epsilon: wp.float32,  # reused as winding `accuracy`
        out: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        p = points[tid]
        q = wp.mesh_query_point_sign_winding_number(
            mesh_id, p, max_dist, epsilon, 0.5
        )
        if q.result:
            cp = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
            d = wp.length(p - cp)
            s = float(1.0)
            if q.sign < 0.0:
                s = -1.0
            out[tid] = s * d
        else:
            out[tid] = max_dist

    return {
        "floodfill": _sdf_unsigned,  # flood-fill assigns the sign afterwards
        "normal": _sdf_normal,
        "winding": _sdf_winding,
    }


def _solidify_floodfill(
    unsigned: np.ndarray, grid_size: Tuple[int, int, int], wall_mult: float
) -> np.ndarray:
    """Turn an unsigned distance volume into a signed one via flood fill.

    Voxels nearer the surface than `wall_mult` voxel-diagonals are walls.
    Free space reachable from the volume border is "outside"; everything
    else (enclosed free space + the walls) is "inside" and gets a negative
    sign. Robust to non-watertight meshes whose holes are sub-wall sized.
    """
    from scipy import ndimage

    D, H, W = grid_size
    # Grid spacing in the normalized [-1, 1] cube (matches _voxel_query_points).
    spacing = 2.0 / (max(D, H, W) - 1)
    wall = wall_mult * np.sqrt(3.0) * spacing

    free = unsigned > wall
    labels, _ = ndimage.label(free)  # 6-connected components of free space
    border_labels = np.unique(
        np.concatenate([
            labels[0].ravel(), labels[-1].ravel(),
            labels[:, 0].ravel(), labels[:, -1].ravel(),
            labels[:, :, 0].ravel(), labels[:, :, -1].ravel(),
        ])
    )
    border_labels = border_labels[border_labels != 0]
    outside = np.isin(labels, border_labels)
    inside = ~outside  # enclosed free space + wall shell
    return np.where(inside, -unsigned, unsigned).astype(np.float32)


def _warp_sdf_kernel(sign_mode: str):
    """Init Warp (once) and return the compiled kernel for `sign_mode`."""
    import warp as wp

    if not _WARP_STATE["ready"]:
        wp.init()
        _WARP_STATE["kernels"] = _build_warp_kernels(wp)
        _WARP_STATE["ready"] = True
    return _WARP_STATE["kernels"][sign_mode]


def _orient_faces_outward(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return `faces` with winding flipped so normals point *outward*.

    Warp's winding-number sign assumes outward-facing triangles; a mesh
    wound the other way reports inside/outside swapped. The signed volume
    of a closed mesh (sum of v0 . (v1 x v2) / 6) is positive iff the
    triangles are outward-oriented, so we flip when it comes out negative.
    Translation-invariant for closed meshes; a harmless no-op heuristic
    for the rare open mesh.
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    signed_vol = float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum())
    if signed_vol < 0.0:
        return faces[:, ::-1].copy()
    return faces


def _drop_out_of_range_faces(faces: np.ndarray, n_verts: int) -> np.ndarray:
    """Return `faces` with any out-of-range (index >= n_verts or < 0) row
    removed. Raises ValueError if nothing valid remains."""
    valid = (faces >= 0).all(axis=1) & (faces < n_verts).all(axis=1)
    if valid.all():
        return faces
    faces = np.ascontiguousarray(faces[valid])
    if len(faces) == 0:
        raise ValueError(
            "mesh has no valid faces after dropping out-of-range indices "
            f"(n_verts={n_verts})"
        )
    return faces


def _compute_sdf_warp(
    verts: np.ndarray,
    faces: np.ndarray,
    grid_size: Tuple[int, int, int],
) -> np.ndarray:
    try:
        import warp as wp
    except ImportError as e:
        raise ImportError(
            "warp-lang is required for the (default) warp SDF backend. "
            "Install via `pip install warp-lang` (prebuilt wheels, no "
            "compiler; CUDA + CPU). Or set FDSL_SDF_BACKEND=torch."
        ) from e

    sign_mode = (
        _WARP_SIGN_MODE
        if _WARP_SIGN_MODE in ("floodfill", "normal", "winding")
        else "floodfill"
    )
    kernel = _warp_sdf_kernel(sign_mode)
    D, H, W = grid_size
    device = "cuda" if wp.get_cuda_device_count() > 0 else "cpu"

    normalized = _normalize_mesh(verts.astype(np.float32, copy=False))
    # Defense in depth: Warp's wp.Mesh does no bounds checking and reads
    # points[indices[i]] out of range for any face that indexes a missing
    # vertex, causing an illegal memory access (CUDA error 700) that
    # poisons the whole CUDA context. Loader-level sanitization should
    # already have stripped these, but guard here too in case bake() is
    # fed raw meshes directly.
    faces = _drop_out_of_range_faces(faces, normalized.shape[0])
    # floodfill ignores orientation; the surface-sign modes need it outward.
    if sign_mode == "floodfill":
        faces_out = faces.astype(np.int64, copy=False)
    else:
        faces_out = _orient_faces_outward(
            normalized, faces.astype(np.int64, copy=False)
        )
    pts = _voxel_query_points(D, H, W)  # (P, 3) as (x, y, z)

    mesh = wp.Mesh(
        points=wp.array(normalized, dtype=wp.vec3, device=device),
        indices=wp.array(
            faces_out.reshape(-1).astype(np.int32, copy=False),
            dtype=wp.int32,
            device=device,
        ),
        support_winding_number=(sign_mode == "winding"),
    )

    # epsilon: pseudonormal back-face tolerance ("normal") / winding
    # `accuracy` ("winding") / unused ("floodfill").
    epsilon = 1.0e-3 if sign_mode == "normal" else 2.0
    wp_points = wp.array(pts, dtype=wp.vec3, device=device)
    wp_out = wp.empty(pts.shape[0], dtype=wp.float32, device=device)
    wp.launch(
        kernel,
        dim=pts.shape[0],
        inputs=[mesh.id, wp_points, wp.float32(_WARP_MAX_DIST),
                wp.float32(epsilon), wp_out],
        device=device,
    )
    wp.synchronize_device(device)
    vol = wp_out.numpy().reshape(D, H, W).astype(np.float32)

    if sign_mode == "floodfill":
        vol = _solidify_floodfill(vol, (D, H, W), _FLOODFILL_WALL)
    return vol


# ----------------------------------------------------------------------
# GPU-vectorized backend (default)
# ----------------------------------------------------------------------

# Number of (query-point, triangle) pairs evaluated per GPU tile. Each
# pair needs a handful of (P_tile, F_tile, 3) float32 temporaries, so
# peak VRAM is ~ budget * (a few dozen) * 4 bytes. 20M ~ a few GB,
# comfortable on an H100. Override via env for smaller GPUs.
_PAIR_BUDGET = int(os.environ.get("FDSL_SDF_PAIR_BUDGET", 20_000_000))

# Cap on triangles per inner tile (keeps F_tile-shaped temporaries bounded
# even for very high-poly meshes).
_FACE_TILE = 20_000


def _select_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _closest_point_on_triangle(p, a, b, c):
    """Closest point on each triangle to each query point (Ericson, RTCD).

    Shapes broadcast: p=(P,1,3), a/b/c=(1,F,3) -> closest=(P,F,3).
    Region cases are reproduced via masked overrides applied in reverse
    priority order (vertices last so they win at corners).
    """
    import torch

    eps = 1e-12
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = (ab * ap).sum(-1)
    d2 = (ac * ap).sum(-1)

    bp = p - b
    d3 = (ab * bp).sum(-1)
    d4 = (ac * bp).sum(-1)

    cp = p - c
    d5 = (ab * cp).sum(-1)
    d6 = (ac * cp).sum(-1)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    # Interior face region (default).
    denom = 1.0 / (va + vb + vc + eps)
    v = (vb * denom).unsqueeze(-1)
    w = (vc * denom).unsqueeze(-1)
    closest = a + v * ab + w * ac

    # Edge BC.
    e_bc = (d4 - d3) + (d5 - d6)
    t_bc = ((d4 - d3) / e_bc.clamp(min=eps)).clamp(0.0, 1.0).unsqueeze(-1)
    m_bc = ((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)).unsqueeze(-1)
    closest = torch.where(m_bc, b + t_bc * (c - b), closest)

    # Edge AC.
    t_ac = (d2 / (d2 - d6).clamp(min=eps)).clamp(0.0, 1.0).unsqueeze(-1)
    m_ac = ((vb <= 0) & (d2 >= 0) & (d6 <= 0)).unsqueeze(-1)
    closest = torch.where(m_ac, a + t_ac * ac, closest)

    # Vertex C.
    m_c = ((d6 >= 0) & (d5 <= d6)).unsqueeze(-1)
    closest = torch.where(m_c, c.expand_as(closest), closest)

    # Edge AB.
    t_ab = (d1 / (d1 - d3).clamp(min=eps)).clamp(0.0, 1.0).unsqueeze(-1)
    m_ab = ((vc <= 0) & (d1 >= 0) & (d3 <= 0)).unsqueeze(-1)
    closest = torch.where(m_ab, a + t_ab * ab, closest)

    # Vertex B.
    m_b = ((d3 >= 0) & (d4 <= d3)).unsqueeze(-1)
    closest = torch.where(m_b, b.expand_as(closest), closest)

    # Vertex A.
    m_a = ((d1 <= 0) & (d2 <= 0)).unsqueeze(-1)
    closest = torch.where(m_a, a.expand_as(closest), closest)

    return closest


def _solid_angle(p, a, b, c):
    """Signed solid angle subtended by each triangle at each query point.

    Van Oosterom & Strackee formula. Shapes: p=(P,1,3), a/b/c=(1,F,3)
    -> (P,F). Summing over all faces and dividing by 4*pi gives the
    generalized winding number.
    """
    import torch

    av = a - p
    bv = b - p
    cv = c - p
    la = av.norm(dim=-1)
    lb = bv.norm(dim=-1)
    lc = cv.norm(dim=-1)

    numer = (av * torch.cross(bv, cv, dim=-1)).sum(-1)
    denom = (
        la * lb * lc
        + (av * bv).sum(-1) * lc
        + (bv * cv).sum(-1) * la
        + (cv * av).sum(-1) * lb
    )
    return 2.0 * torch.atan2(numer, denom)


def _compute_sdf_torch(
    verts: np.ndarray,
    faces: np.ndarray,
    grid_size: Tuple[int, int, int],
) -> np.ndarray:
    import math

    import torch

    D, H, W = grid_size
    device = _select_device()

    normalized = _normalize_mesh(verts.astype(np.float32, copy=False))
    v = torch.as_tensor(normalized, dtype=torch.float32, device=device)
    f = torch.as_tensor(
        faces.astype(np.int64, copy=False), dtype=torch.long, device=device
    )
    tri_a = v[f[:, 0]].unsqueeze(0)  # (1, F, 3)
    tri_b = v[f[:, 1]].unsqueeze(0)
    tri_c = v[f[:, 2]].unsqueeze(0)
    F_count = tri_a.shape[1]

    pts = torch.as_tensor(
        _voxel_query_points(D, H, W), dtype=torch.float32, device=device
    )  # (P, 3)
    P = pts.shape[0]

    f_tile = min(F_count, _FACE_TILE)
    p_tile = max(1, _PAIR_BUDGET // max(1, f_tile))

    out = torch.empty(P, dtype=torch.float32, device=device)
    four_pi = 4.0 * math.pi

    for ps in range(0, P, p_tile):
        pe = min(ps + p_tile, P)
        p = pts[ps:pe].unsqueeze(1)  # (Pc, 1, 3)

        min_d2 = torch.full((pe - ps,), float("inf"), device=device)
        wsum = torch.zeros(pe - ps, device=device)

        for fs in range(0, F_count, f_tile):
            fe = min(fs + f_tile, F_count)
            a = tri_a[:, fs:fe]
            b = tri_b[:, fs:fe]
            c = tri_c[:, fs:fe]

            closest = _closest_point_on_triangle(p, a, b, c)
            d2 = ((p - closest) ** 2).sum(-1)  # (Pc, Fc)
            min_d2 = torch.minimum(min_d2, d2.min(dim=1).values)

            wsum = wsum + _solid_angle(p, a, b, c).sum(dim=1)

        dist = torch.sqrt(min_d2)
        # |winding| ~ 1 inside, ~ 0 outside; abs() tolerates inconsistent
        # face orientation. Negative inside (standard SDF convention).
        inside = (wsum.abs() / four_pi) > 0.5
        out[ps:pe] = torch.where(inside, -dist, dist)

    return out.reshape(D, H, W).detach().to("cpu").numpy().astype(np.float32)


# ----------------------------------------------------------------------
# Legacy CPU backend (reference / fallback)
# ----------------------------------------------------------------------

# Query points processed per trimesh call. The library allocates an
# O(points * faces) intermediate, so we chunk to bound peak RAM.
_QUERY_CHUNK = 1024


def _compute_sdf_trimesh(
    verts: np.ndarray,
    faces: np.ndarray,
    grid_size: Tuple[int, int, int],
) -> np.ndarray:
    try:
        import trimesh  # type: ignore
    except ImportError as e:
        raise ImportError(
            "trimesh is required for the trimesh SDF backend. "
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
