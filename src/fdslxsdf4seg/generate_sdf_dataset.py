# file: generate_sdf_dataset.py
import argparse
import os
import random
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# --- SDFベースクラス ---
class SDFObject:
    def __init__(self, grid_size: List[int], device: torch.device):
        self.device = device
        self.grid_size = grid_size  # [D, H, W]

    def sdf(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        x, y, z: meshgrid 上の座標テンソル (shape=(D,H,W))
        戻り値: 各点の signed distance (同shape)
        """
        raise NotImplementedError

    def max_distance(self) -> float:
        """
        このオブジェクトのバウンディング半径相当
        """
        raise NotImplementedError


# --- 各種プリミティブ実装 （ランダム化ロジックをコンストラクタ内に移動） ---
class Sphere(SDFObject):
    def __init__(
        self, grid_size: List[int], device: torch.device, center=None, radius=None
    ):
        super().__init__(grid_size, device)
        D, H, W = grid_size
        # ランダム化
        if center is None:
            cz = random.uniform(W * 0.2, W * 0.8)
            cy = random.uniform(H * 0.2, H * 0.8)
            cx = random.uniform(D * 0.2, D * 0.8)
            center = (cz, cy, cx)
        if radius is None:
            radius = random.uniform(min(D, H, W) * 0.1, min(D, H, W) * 0.3)
        # パラメータ設定
        self.center = torch.tensor(center, device=device).view(3, 1, 1, 1)
        self.radius = radius

    def sdf(self, x, y, z):
        p = torch.stack([x, y, z], dim=0) - self.center
        return torch.norm(p, dim=0) - self.radius

    def max_distance(self):
        return self.radius


class Box(SDFObject):
    def __init__(
        self, grid_size: List[int], device: torch.device, center=None, half_extents=None
    ):
        super().__init__(grid_size, device)
        D, H, W = grid_size
        if center is None:
            cz = random.uniform(W * 0.2, W * 0.8)
            cy = random.uniform(H * 0.2, H * 0.8)
            cx = random.uniform(D * 0.2, D * 0.8)
            center = (cz, cy, cx)
        if half_extents is None:
            hx = random.uniform(D * 0.1, D * 0.3)
            hy = random.uniform(H * 0.1, H * 0.3)
            hz = random.uniform(W * 0.1, W * 0.3)
            half_extents = (hz, hy, hx)
        self.center = torch.tensor(center, device=device).view(3, 1, 1, 1)
        self.half = torch.tensor(half_extents, device=device).view(3, 1, 1, 1)

    def sdf(self, x, y, z):
        p = torch.stack([x, y, z], dim=0) - self.center
        q = torch.abs(p) - self.half
        outside = torch.clamp(q, min=0.0)
        inside = torch.clamp(torch.max(q, dim=0).values, max=0.0)
        return torch.norm(outside, dim=0) + inside

    def max_distance(self):
        return float(torch.norm(self.half))


class Cylinder(SDFObject):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        radius=None,
        height=None,
        axis=2,
    ):
        super().__init__(grid_size, device)
        D, H, W = grid_size
        if center is None:
            cz = random.uniform(W * 0.2, W * 0.8)
            cy = random.uniform(H * 0.2, H * 0.8)
            cx = random.uniform(D * 0.2, D * 0.8)
            center = (cz, cy, cx)
        if radius is None:
            radius = random.uniform(min(D, H) * 0.1, min(D, H) * 0.3)
        if height is None:
            height = random.uniform(W * 0.2, W * 0.6)
        self.center = torch.tensor(center, device=device).view(3, 1, 1, 1)
        self.radius = radius
        self.h = height / 2.0
        self.axis = axis

    def sdf(self, x, y, z):
        p = torch.stack([x, y, z], dim=0) - self.center
        perp = (
            torch.norm(torch.stack([p[i] for i in range(3) if i != self.axis]), dim=0)
            - self.radius
        )
        along = torch.abs(p[self.axis]) - self.h
        outside = torch.clamp(torch.stack([perp, along], dim=0), min=0.0)
        inside = torch.clamp(
            torch.max(torch.stack([perp, along], dim=0), dim=0).values, max=0.0
        )
        return torch.norm(outside, dim=0) + inside

    def max_distance(self):
        return float(max(self.h, self.radius))


class Torus(SDFObject):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        major_r=None,
        minor_r=None,
        axis=2,
    ):
        super().__init__(grid_size, device)
        D, H, W = grid_size
        if center is None:
            cz = random.uniform(W * 0.2, W * 0.8)
            cy = random.uniform(H * 0.2, H * 0.8)
            cx = random.uniform(D * 0.2, D * 0.8)
            center = (cz, cy, cx)
        if major_r is None:
            major_r = random.uniform(min(D, H) * 0.1, min(D, H) * 0.3)
        if minor_r is None:
            minor_r = major_r * random.uniform(0.3, 0.6)
        self.center = torch.tensor(center, device=device).view(3, 1, 1, 1)
        self.R = major_r
        self.r = minor_r
        self.axis = axis

    def sdf(self, x, y, z):
        p = torch.stack([x, y, z], dim=0) - self.center
        perp = (
            torch.norm(torch.stack([p[i] for i in range(3) if i != self.axis]), dim=0)
            - self.R
        )
        along = p[self.axis]
        return torch.norm(torch.stack([perp, along], dim=0), dim=0) - self.r

    def max_distance(self):
        return self.R + self.r


# --- データセット ---
class SDFSegmentationDataset(Dataset):
    def __init__(
        self,
        grid_size: List[int],
        num_volumes: int,
        min_objects: int = 2,
        max_objects: int = 5,
        device: torch.device = None,
    ):
        self.D, self.H, self.W = grid_size
        self.num_volumes = num_volumes
        self.min_o = min_objects
        self.max_o = max_objects
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        zs = torch.linspace(0, self.D - 1, self.D, device=self.device)
        ys = torch.linspace(0, self.H - 1, self.H, device=self.device)
        xs = torch.linspace(0, self.W - 1, self.W, device=self.device)
        self.Z, self.Y, self.X = torch.meshgrid(zs, ys, xs, indexing="ij")

        self.primitive_classes = [Sphere, Box, Cylinder, Torus]

    def __len__(self):
        return self.num_volumes

    def __getitem__(self, idx):
        n_objs = random.randint(self.min_o, self.max_o)
        sdfs = []
        max_ds = []

        for _ in range(n_objs):
            PrimClass = random.choice(self.primitive_classes)
            obj = PrimClass(grid_size=[self.D, self.H, self.W], device=self.device)
            s = obj.sdf(self.X, self.Y, self.Z)
            sdfs.append(s)
            max_ds.append(obj.max_distance())

        # 動的に a を決定し平均
        a_vals = [128 ** (1.0 / md) for md in max_ds]
        vals = [torch.pow(a, -sdf) for a, sdf in zip(a_vals, sdfs)]
        x_vol = torch.stack(vals, dim=0).mean(dim=0)
        x_vol = torch.clamp(x_vol, 0.0, 128.0).to(torch.uint8).unsqueeze(0)

        y_vol = torch.stack([(sdf < 0).to(torch.uint8) for sdf in sdfs], dim=0)

        return x_vol.cpu().numpy(), y_vol.cpu().numpy()


# --- データ生成＆保存 ---
def generate_and_save(
    out_dir: str,
    grid_size: List[int],
    num_samples: int,
    min_objects: int,
    max_objects: int,
    num_workers: int,
    seed: int = None,
):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    os.makedirs(out_dir, exist_ok=True)
    ds = SDFSegmentationDataset(
        grid_size=grid_size,
        num_volumes=num_samples,
        min_objects=min_objects,
        max_objects=max_objects,
    )
    loader = DataLoader(ds, batch_size=1, num_workers=num_workers, pin_memory=True)

    for i, (x, y) in enumerate(loader):
        x = x[0]
        y = y[0]
        fname = os.path.join(out_dir, f"sample_{i:05d}.npz")
        np.savez_compressed(fname, x=x, y=y)
        if i % 50 == 0:
            print(f"Saved {i + 1}/{num_samples}")

    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--D", type=int, default=64)
    p.add_argument("--H", type=int, default=64)
    p.add_argument("--W", type=int, default=64)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--min_objects", type=int, default=2)
    p.add_argument("--max_objects", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    generate_and_save(
        out_dir=args.out_dir,
        grid_size=[args.D, args.H, args.W],
        num_samples=args.num_samples,
        min_objects=args.min_objects,
        max_objects=args.max_objects,
        num_workers=args.num_workers,
        seed=args.seed,
    )
