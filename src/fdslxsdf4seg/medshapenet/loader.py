"""MedShapeNetCore loader: lazy download + memory-efficient extraction.

Three-tier loading strategy:

1. *Per-sample extracted cache* (cheap): when present, each
   (dataset, organ_key, sample_idx) lives as its own small .npz file
   on disk. `load_meshes` reads only the requested number of samples,
   so peak memory is bounded by `max_instances * per_sample_size`.

2. *Bulk .npz extraction* (one-time, heavy): if no extracted cache
   exists, we download the giant Zenodo .npz once, unpickle it,
   write per-sample files to the cache, drop the in-memory raw dict.

3. *Network download* (once per dataset): streamed into our project
   cache rather than CWD.

Tier 1 is what makes PULMONARY (1.14 GB) and ThoracicAorta_Saitta
(515 MB) usable after a one-time extraction. The bulk load in tier 2
still needs enough RAM for the giant pickle (>5 GB for PULMONARY) and
will raise a clear MemoryError if the host can't fit it — the user
can run extraction on a larger machine and copy
`cache/medshapenet/extracted/` over.

Cross-platform: Windows + Linux, no OpenGL, no GPU.
"""

from __future__ import annotations

import shutil
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from fdslxsdf4seg.medshapenet.paths import get_cache_root, get_npz_dir

# (vertices float32 (V,3), faces int32 (F,3))
Mesh = Tuple[np.ndarray, np.ndarray]

# Flat-layout datasets store meshes under `raw['mesh']`.
# Per-organ datasets keep organ dicts at the top level, each holding
# its own `mesh` sub-dict.
_PER_ORGAN_DATASETS = {"FLARE", "PULMONARY"}

_DOWNLOAD_CHUNK = 1 << 20  # 1 MiB
_FLAT_KEY = "_flat"  # cache subdir name for flat datasets


# ----------------------------------------------------------------------
# Network / pickle plumbing
# ----------------------------------------------------------------------

def _npz_file_for(dataset: str) -> Path:
    return get_npz_dir() / f"medshapenetcore_{dataset}.npz"


def _download_url(dataset: str) -> str:
    try:
        from MedShapeNetCore.__main__ import data_set_info  # type: ignore
    except ImportError as e:
        raise ImportError(
            "MedShapeNetCore is required. Install via "
            "`pip install MedShapeNetCore` (cross-platform)."
        ) from e
    entry = data_set_info.get("dataset", {}).get(dataset)
    if not entry or "url" not in entry:
        raise KeyError(
            f"Dataset {dataset!r} not found in MedShapeNetCore registry."
        )
    return entry["url"]


def _download_if_missing(dataset: str) -> Path:
    dst = _npz_file_for(dataset)
    if dst.exists():
        return dst

    try:
        import requests  # type: ignore
    except ImportError as e:
        raise ImportError(
            "requests is required for MedShapeNet downloads "
            "(installed transitively with MedShapeNetCore)."
        ) from e

    url = _download_url(dataset)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"[MedShapeNet] downloading {dataset} from {url} -> {dst}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=_DOWNLOAD_CHUNK):
                if chunk:
                    f.write(chunk)
    tmp.replace(dst)
    print(f"[MedShapeNet] downloaded {dataset} ({dst.stat().st_size / 1e6:.1f} MB)")
    return dst


def _install_pickle_aliases() -> None:
    """Make MedShapeNetCore's pickled `MyDict` resolvable on unpickle.

    The .npz files were pickled with `MyDict` defined in `__main__`
    (the package author's notebook). Without this alias, unpickling
    raises AttributeError on any other entry-point. Idempotent.
    """
    import sys

    from MedShapeNetCore.MedShapeNetCore import MyDict  # type: ignore

    main_mod = sys.modules.get("__main__")
    if main_mod is not None and not hasattr(main_mod, "MyDict"):
        main_mod.MyDict = MyDict


def _load_raw_bundle(dataset: str) -> dict:
    """Load the full .npz pickle into memory. *Heavy*: see module docstring.

    We bypass MSNLoader.load() because it hardcodes `mmap_mode='c'`
    which on Windows triggers MemoryError on the larger datasets.
    """
    _download_if_missing(dataset)
    _install_pickle_aliases()

    path = _npz_file_for(dataset)
    print(f"[MedShapeNet] loading raw bundle: {dataset} from {path}")
    with np.load(path, allow_pickle=True) as npz:
        data = npz["data"].item()
    return data


# ----------------------------------------------------------------------
# Extracted per-sample cache
# ----------------------------------------------------------------------

def _extracted_root() -> Path:
    return get_cache_root() / "extracted"


def _extracted_dir(dataset: str, organ_key: Optional[str]) -> Path:
    key = organ_key if organ_key is not None else _FLAT_KEY
    return _extracted_root() / dataset / key


def _extracted_done_marker(dataset: str) -> Path:
    return _extracted_root() / dataset / ".extraction_complete"


def _list_extracted_files(
    dataset: str, organ_key: Optional[str]
) -> List[Path]:
    d = _extracted_dir(dataset, organ_key)
    if not d.is_dir():
        return []
    return sorted(d.glob("sample_*.npz"))


def _read_extracted_mesh(path: Path) -> Optional[Mesh]:
    try:
        with np.load(path) as data:
            v = np.asarray(data["vertices"], dtype=np.float32)
            f = np.asarray(data["faces"], dtype=np.int32)
    except (KeyError, ValueError, OSError):
        return None
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        return None
    if len(v) == 0 or len(f) == 0:
        return None
    return (v, f)


def _load_from_extracted(
    dataset: str,
    organ_key: Optional[str],
    max_instances: Optional[int],
) -> List[Mesh]:
    """Load up to `max_instances` meshes from per-sample cache files."""
    files = _list_extracted_files(dataset, organ_key)
    if max_instances is not None:
        files = files[:max_instances]
    out: List[Mesh] = []
    for f in files:
        m = _read_extracted_mesh(f)
        if m is not None:
            out.append(m)
    return out


# ----------------------------------------------------------------------
# Raw-dict -> extracted-cache conversion
# ----------------------------------------------------------------------

def _dict_get(obj, key: str):
    if isinstance(obj, dict):
        return obj.get(key)
    try:
        return obj[key]
    except (KeyError, IndexError, ValueError, TypeError):
        return None


def _yield_vf_pairs(node):
    """Yield (vertices, faces) tuples from a `mesh` node, skipping bad ones."""
    verts_list = _dict_get(node, "vertices")
    faces_list = _dict_get(node, "faces")
    if verts_list is None or faces_list is None:
        return
    n = min(len(verts_list), len(faces_list))
    for i in range(n):
        v = np.asarray(verts_list[i], dtype=np.float32)
        f = np.asarray(faces_list[i], dtype=np.int32)
        if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
            continue
        if len(v) == 0 or len(f) == 0:
            continue
        yield i, v, f


def _save_organ_to_per_sample(
    dataset: str,
    organ_key: Optional[str],
    mesh_node,
) -> int:
    out_dir = _extracted_dir(dataset, organ_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for i, v, f in _yield_vf_pairs(mesh_node):
        np.savez_compressed(
            out_dir / f"sample_{i:05d}.npz", vertices=v, faces=f
        )
        count += 1
    return count


def _extract_bundle_to_cache(dataset: str) -> None:
    """Read the giant raw bundle once and emit per-sample cache files.

    Memory-heavy step: must hold the entire pickled dict in RAM. After
    extraction, the in-process raw cache is dropped immediately so
    subsequent calls only need the cheap per-sample tier.
    """
    marker = _extracted_done_marker(dataset)
    if marker.exists():
        return

    try:
        raw = _load_raw_bundle(dataset)
    except MemoryError as e:
        raise MemoryError(
            f"Not enough RAM to unpickle MedShapeNet dataset {dataset!r}. "
            "Options: (1) close other applications and retry, "
            "(2) run on a host with more RAM and copy the resulting "
            f"directory `cache/medshapenet/extracted/{dataset}/` back, "
            "(3) skip this dataset in catalog.ACTIVE_DATASETS."
        ) from e

    print(f"[MedShapeNet] extracting per-sample cache for {dataset}...")
    total = 0
    if dataset in _PER_ORGAN_DATASETS:
        # Top-level keys are organs. We persist them all so any organ
        # the catalog later references is already on disk.
        for organ in list(raw.keys()):
            organ_node = _dict_get(raw, organ)
            if organ_node is None:
                continue
            mesh_node = _dict_get(organ_node, "mesh")
            if mesh_node is None:
                continue
            n = _save_organ_to_per_sample(dataset, organ, mesh_node)
            print(f"  {organ}: {n} samples")
            total += n
    else:
        mesh_node = _dict_get(raw, "mesh")
        if mesh_node is not None:
            n = _save_organ_to_per_sample(dataset, None, mesh_node)
            print(f"  (flat): {n} samples")
            total += n

    # Drop in-memory raw to reclaim RAM before the next dataset.
    raw.clear()
    with _CACHE._lock:
        _CACHE._raw.pop(dataset, None)

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"samples_total={total}\n", encoding="utf-8")
    print(f"[MedShapeNet] extraction complete for {dataset} ({total} samples)")


def clear_extracted(dataset: str) -> None:
    """Delete the per-sample extracted cache for `dataset` (rebuild on next use)."""
    d = _extracted_root() / dataset
    if d.exists():
        shutil.rmtree(d)


# ----------------------------------------------------------------------
# In-process cache + public API
# ----------------------------------------------------------------------

class _DatasetCache:
    """In-process cache so repeated `load_meshes` calls don't re-read disk."""

    def __init__(self) -> None:
        # Held only transiently during bulk extraction.
        self._raw: Dict[str, dict] = {}
        # (dataset, organ_key) -> list of Mesh
        self._meshes: Dict[Tuple[str, Optional[str]], List[Mesh]] = {}
        self._lock = threading.Lock()

    def get_meshes(
        self, dataset: str, organ_key: Optional[str]
    ) -> List[Mesh]:
        key = (dataset, organ_key)
        with self._lock:
            if key in self._meshes:
                return self._meshes[key]
        # Ensure the extracted cache exists, then read from it.
        if not _extracted_done_marker(dataset).exists():
            _extract_bundle_to_cache(dataset)
        meshes = _load_from_extracted(dataset, organ_key, max_instances=None)
        with self._lock:
            self._meshes[key] = meshes
        return meshes


_CACHE = _DatasetCache()


def load_meshes(
    dataset: str,
    organ_key: Optional[str],
    max_instances: Optional[int] = None,
) -> List[Mesh]:
    """Return (vertices, faces) pairs for (dataset, organ_key).

    Args:
        dataset: MedShapeNetCore dataset id (e.g. "FLARE").
        organ_key: sub-organ key, or None for flat-layout datasets.
        max_instances: cap on returned meshes (None = all).

    First call for a dataset triggers a one-time heavy extraction
    (>1 GB RAM peak for some datasets). Subsequent calls are cheap.

    Returns an empty list if `organ_key` doesn't exist in `dataset`.
    """
    meshes = _CACHE.get_meshes(dataset, organ_key)
    if max_instances is not None and len(meshes) > max_instances:
        return meshes[:max_instances]
    return meshes


def organ_available(dataset: str, organ_key: Optional[str]) -> bool:
    """Whether (dataset, organ_key) yields at least one mesh.

    Uses the extracted cache when available to avoid triggering heavy
    bulk extraction just for a presence check.
    """
    if _extracted_done_marker(dataset).exists():
        return len(_list_extracted_files(dataset, organ_key)) > 0
    try:
        return len(_CACHE.get_meshes(dataset, organ_key)) > 0
    except Exception:
        return False


def ensure_extracted(dataset: str) -> None:
    """Public entry point for the CLI extractor. Triggers a one-time bulk
    extraction if the cache is missing; no-op otherwise."""
    _extract_bundle_to_cache(dataset)
