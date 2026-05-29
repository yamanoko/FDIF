"""Cross-platform path utilities for the MedShapeNet cache."""

from __future__ import annotations

import os
from pathlib import Path


def get_project_root() -> Path:
    """Return the FDIF project root (parent of `src/`).

    Falls back to the current working directory if structure is unexpected.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() or (parent / "src").is_dir():
            return parent
    return Path.cwd()


def get_cache_root() -> Path:
    """Return the cache root directory for MedShapeNet data.

    Resolution order:
        1. $FDSLXSDF4SEG_CACHE if set
        2. <project_root>/cache/medshapenet

    Works identically on Windows and Linux (uses pathlib).
    """
    env = os.environ.get("FDSLXSDF4SEG_CACHE")
    if env:
        return Path(env).expanduser().resolve()
    return get_project_root() / "cache" / "medshapenet"


def get_npz_dir() -> Path:
    """Directory where MedShapeNetCore .npz bundles are downloaded."""
    p = get_cache_root() / "npz"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_sdf_dir() -> Path:
    """Directory for baked SDF voxel caches."""
    p = get_cache_root() / "sdf"
    p.mkdir(parents=True, exist_ok=True)
    return p
