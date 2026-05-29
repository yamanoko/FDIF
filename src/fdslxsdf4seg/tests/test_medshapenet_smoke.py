"""Smoke tests for the MedShapeNet integration.

These tests exercise the integration without forcing a real download:
* registry shape and category injection are checked unconditionally
* mesh-bake -> SDFObject end-to-end is checked against a synthetic
  tetrahedron mesh, bypassing the MedShapeNetCore network path

The tests are designed to pass on both Windows and Linux without any
display server, GPU, or OpenGL context.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fdslxsdf4seg.medshapenet import (
    get_medshapenet_categories,
    get_medshapenet_primitives,
)


def test_registry_has_expected_categories():
    cats = get_medshapenet_categories()
    # Active datasets that the catalog whitelists by default.
    assert "msn_all" in cats
    assert "msn_flare" in cats
    assert "msn_asoca" in cats
    assert "msn_avt" in cats
    assert "msn_pulmonary" in cats
    assert "msn_thoracicaorta_saitta" in cats
    # KITS is reserved for finetuning evaluation and must not appear.
    assert "msn_kits" not in cats


def test_registry_primitive_names_are_namespaced():
    prims = get_medshapenet_primitives()
    assert prims, "expected at least one MedShapeNet primitive"
    for name in prims:
        assert name.startswith("msn_"), f"primitive {name!r} missing msn_ prefix"


def test_injection_into_primitive_registry():
    from fdslxsdf4seg.primitive_registry import (
        ALL_PRIMITIVES,
        PRIMITIVE_CATEGORIES,
    )

    msn_prims = get_medshapenet_primitives()
    if not msn_prims:
        pytest.skip("no MedShapeNet primitives registered")
    sample_name = next(iter(msn_prims))
    assert sample_name in ALL_PRIMITIVES
    assert "msn_all" in PRIMITIVE_CATEGORIES


# --- End-to-end bake -> SDFObject test using a synthetic mesh -----------

def _tetrahedron():
    """A unit tetrahedron mesh — small, well-defined, watertight."""
    verts = np.array(
        [
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 3, 1],
            [0, 2, 3],
            [1, 3, 2],
        ],
        dtype=np.int32,
    )
    return verts, faces


def test_bake_returns_grid_shaped_volume():
    pytest.importorskip("trimesh")
    from fdslxsdf4seg.medshapenet.mesh_to_sdf import bake

    verts, faces = _tetrahedron()
    for shape in [(16, 16, 16), (12, 20, 24)]:
        sdf = bake(verts, faces, shape, dataset="_test", organ_key="tetra")
        assert sdf.shape == shape, f"got {sdf.shape}, expected {shape}"
        assert sdf.dtype == np.float32
        # Tetrahedron straddles origin → interior exists → some negative SDF.
        assert sdf.min() < 0.0
        assert sdf.max() > 0.0


def test_medshapenet_sdfobject_returns_input_shape():
    """End-to-end: subclass MedShapeNetSDFObject with a fake mesh and check
    that `.sdf(x, y, z)` returns the same shape as the input coord tensors,
    matching the SDFObject contract used by all other primitives.
    """
    pytest.importorskip("trimesh")
    from fdslxsdf4seg.medshapenet.medshapenet_sdf import MedShapeNetSDFObject
    from fdslxsdf4seg.medshapenet import loader as msn_loader

    verts, faces = _tetrahedron()
    # Monkeypatch the loader so we don't hit the network.
    fake_meshes = [(verts, faces)]
    original = msn_loader.load_meshes
    msn_loader.load_meshes = lambda dataset, organ_key, max_instances=None: fake_meshes
    try:
        Cls = type(
            "MsnFakeTetra",
            (MedShapeNetSDFObject,),
            {"dataset": "_test", "organ_key": "tetra_obj"},
        )
        device = torch.device("cpu")
        grid_size = [16, 20, 24]
        obj = Cls(grid_size=grid_size, device=device, transform=False)

        zs = torch.linspace(-grid_size[0] / 2, grid_size[0] / 2 - 1, grid_size[0])
        ys = torch.linspace(-grid_size[1] / 2, grid_size[1] / 2 - 1, grid_size[1])
        xs = torch.linspace(-grid_size[2] / 2, grid_size[2] / 2 - 1, grid_size[2])
        Z, Y, X = torch.meshgrid(zs, ys, xs, indexing="ij")
        sdf = obj.sdf(X, Y, Z)
        assert sdf.shape == tuple(grid_size)
        assert torch.isfinite(sdf).all()
    finally:
        msn_loader.load_meshes = original
