import os

import numpy as np
import pytest
import torch

from src.fdslxsdf4seg.generate_sdf_dataset import (
    Box,
    Cylinder,
    SDFSegmentationDataset,
    Sphere,
    Torus,
    generate_and_save,
)

# テストパラメータ
grid_size = [16, 16, 16]
device = torch.device("cpu")

# --- SDF プリミティブのテスト ---


@pytest.mark.parametrize(
    "center, radius, point, expected",
    [
        ((8, 8, 8), 5.0, (8, 8, 8), -5.0),  # 中心点では -radius
        ((8, 8, 8), 5.0, (8, 8, 13), 0.0),  # 境界上ではほぼ 0
    ],
)
def test_sphere_sdf_values(center, radius, point, expected):
    sph = Sphere(grid_size, device, center=center, radius=radius)
    # 座標メッシュから特定点を取る
    z, y, x = point
    val = sph.sdf(
        torch.tensor(z, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(x, dtype=torch.float32),
    )
    assert pytest.approx(val.item(), abs=1e-3) == expected


@pytest.mark.parametrize(
    "center, half_extents, point, inside",
    [
        ((8, 8, 8), (2, 2, 2), (8, 8, 8), True),  # 中心は内部
        ((8, 8, 8), (2, 2, 2), (10, 10, 10), False),
    ],
)
def test_box_sdf_sign(center, half_extents, point, inside):
    box = Box(grid_size, device, center=center, half_extents=half_extents)
    z, y, x = point
    val = box.sdf(
        torch.tensor(z, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(x, dtype=torch.float32),
    ).item()
    if inside:
        assert val < 0
    else:
        assert val >= 0


@pytest.mark.parametrize(
    "radius, height, point, inside",
    [
        (3.0, 8.0, (8, 8, 8), True),
        (3.0, 8.0, (8, 8, 8 + 8.0 / 2 + 1), False),
    ],
)
def test_cylinder_sdf(
    center=(8, 8, 8), radius=3.0, height=8.0, point=None, inside=True
):
    cyl = Cylinder(
        grid_size, device, center=center, radius=radius, height=height, axis=2
    )
    z, y, x = point
    val = cyl.sdf(
        torch.tensor(z, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(x, dtype=torch.float32),
    ).item()
    if inside:
        assert val < 0
    else:
        assert val >= 0


@pytest.mark.parametrize(
    "major_r, minor_r, point",
    [
        (4.0, 1.0, (8, 8, 8 + 4.0 + 1.0)),  # 外側にある点
    ],
)
def test_torus_sdf_positive(major_r, minor_r, point):
    tor = Torus(
        grid_size, device, center=(8, 8, 8), major_r=major_r, minor_r=minor_r, axis=2
    )
    z, y, x = point
    val = tor.sdf(
        torch.tensor(z, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(x, dtype=torch.float32),
    ).item()
    assert val > 0


# --- データセットのテスト ---


def test_dataset_item_shape_and_range(tmp_path):
    # 小規模データセット
    ds = SDFSegmentationDataset(
        grid_size, num_volumes=1, min_objects=2, max_objects=2, device=device
    )
    x, y = ds[0]
    # x: (1, D, H, W), uint8, 値域 0-128
    assert x.dtype == np.uint8
    assert x.shape == (1, *grid_size)
    assert x.min() >= 0 and x.max() <= 128
    # y: (n_objs, D, H, W), uint8, 値は 0 or 1
    assert y.dtype == np.uint8
    assert y.shape[1:] == tuple(grid_size)
    assert set(np.unique(y)).issubset({0, 1})


def test_generate_and_save_creates_files(tmp_path):
    outdir = tmp_path / "data"
    # サンプル数2で生成
    generate_and_save(
        out_dir=str(outdir),
        grid_size=grid_size,
        num_samples=2,
        min_objects=2,
        max_objects=2,
        num_workers=0,
        seed=0,
    )
    files = sorted(os.listdir(outdir))
    assert len(files) == 2
    for fname in files:
        assert fname.endswith(".npz")
        data = np.load(outdir / fname)
        assert "x" in data and "y" in data
        x = data["x"]
        y = data["y"]
        assert x.dtype == np.uint8 and y.dtype == np.uint8
        assert x.shape[1:] == tuple(grid_size)
        assert y.shape[1:] == tuple(grid_size)
