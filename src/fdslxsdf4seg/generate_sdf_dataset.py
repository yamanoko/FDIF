# file: generate_sdf_dataset.py
import argparse
import os
import random
import time
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
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
            radius = random.uniform(min(D, H, W) * 0.3, min(D, H, W) * 0.5)
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
            radius = random.uniform(min(D, H) * 0.3, min(D, H) * 0.5)
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
    ):
        super().__init__(grid_size, device)
        D, H, W = grid_size
        if center is None:
            cz = random.uniform(W * 0.2, W * 0.8)
            cy = random.uniform(H * 0.2, H * 0.8)
            cx = random.uniform(D * 0.2, D * 0.8)
            center = (cz, cy, cx)
        if major_r is None:
            major_r = random.uniform(min(D, H) * 0.3, min(D, H) * 0.5)
        if minor_r is None:
            minor_r = major_r * random.uniform(0.3, 0.6)
        self.center = torch.tensor(center, device=device).view(3, 1, 1, 1)
        self.cx = self.center[0, :, :, :]
        self.cy = self.center[1, :, :, :]
        self.cz = self.center[2, :, :, :]
        self.R = major_r
        self.r = minor_r

    def sdf(self, x, y, z):
        new_x = x - self.cx
        new_y = y - self.cy
        new_z = z - self.cz
        q = torch.stack([new_x, new_z], dim=0).norm(dim=0) - self.R
        q = torch.stack([q, new_y], dim=0).norm(dim=0) - self.r
        return q

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

        # # 動的に a を決定し平均
        # a_vals = [128 ** (1.0 / md) for md in max_ds]
        # vals = [torch.pow(a, -sdf) for a, sdf in zip(a_vals, sdfs)]
        # x_vol = torch.stack(vals, dim=0).mean(dim=0)
        # x_vol = torch.clamp(x_vol, 0.0, 128.0).to(torch.uint8).unsqueeze(0)

        x_vol = 128.0 / (torch.pow(torch.abs(torch.stack(sdfs, dim=0)), 2.0) + 1.0)
        # x_vol = x_vol.mean(dim=0)
        x_vol = x_vol.sum(dim=0)
        x_vol = torch.clamp(x_vol, 0.0, 128.0).to(torch.uint8).unsqueeze(0)

        y_vol = torch.stack([(sdf < 0).to(torch.uint8) for sdf in sdfs], dim=0)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        return x_vol.cpu().numpy(), y_vol.cpu().numpy()


# --- データ生成＆保存 ---
def generate_and_save(
    out_dir: str,
    grid_size: List[int],
    num_samples: int,
    min_objects: int,
    max_objects: int,
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
    loader = DataLoader(ds, batch_size=1, num_workers=0)

    for i, (x, y) in enumerate(loader):
        x = x[0]
        y = y[0]
        fname = os.path.join(out_dir, f"sample_{i:05d}.npz")
        np.savez_compressed(fname, x=x, y=y)
        if i % 50 == 0:
            print(f"Saved {i + 1}/{num_samples}")

    print("Done.")


def visualize_sample(sample, output_file_name):
    x, y = sample
    # visualize x and y using voxels
    x = x.squeeze(0)  # (D, H, W)
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    ax1.voxels(x > 20, edgecolor="k", facecolors="blue", shade=False)
    # yは複数のオブジェクトマスクを持つため、各オブジェクトを異なる色で表示
    colors = np.zeros(y[0].shape + (4,), dtype=object)
    visualized_y = np.zeros(y.shape[1:], dtype=bool)
    for i in range(y.shape[0]):
        visualized_y = visualized_y | y[i]
        mask = y[i] > 0
        colors[mask, :] = plt.cm.viridis(i / y.shape[0])
    ax2.voxels(visualized_y, edgecolor="k", facecolors=colors, shade=False)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("SDF Volume")
    ax2.set_title("Object Masks")
    # 保存
    plt.tight_layout()
    # fig.savefig("sample_visualization.png")
    plt.savefig(output_file_name)
    # save a slice of the volume (only x)
    # visualize with a color bar
    slice_index = x.shape[0] // 2
    plt.figure(figsize=(5, 5))
    plt.title(f"Slice at index {slice_index}")
    plt.imshow(x[slice_index, :, :], cmap="gray", vmin=0, vmax=128)
    plt.colorbar()
    plt.savefig(output_file_name.replace(".png", "_slice.png"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str)
    p.add_argument("--D", type=int, default=64)
    p.add_argument("--H", type=int, default=64)
    p.add_argument("--W", type=int, default=64)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--min_objects", type=int, default=2)
    p.add_argument("--max_objects", type=int, default=5)
    p.add_argument("--seed", type=int, default=None)
    # p.add_argument("--visualize", action="store_true", help="Visualize a sample")
    p.add_argument(
        "--num_visualize", type=int, default=0, help="Number of samples to visualize"
    )
    args = p.parse_args()

    if not args.out_dir:
        # 出力ディレクトリが指定されていない場合は、カレントディレクトリに日付と時刻を付けて作成
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"outputs/{timestamp}"

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    # ログをoutputディレクトリに保存
    log_file = os.path.join(args.out_dir, "generation_log.txt")
    with open(log_file, "w") as f:
        f.write(f"Output directory: {args.out_dir}\n")
        f.write(f"Grid size: {args.D}x{args.H}x{args.W}\n")
        f.write(f"Number of samples: {args.num_samples}\n")
        f.write(f"Min objects per sample: {args.min_objects}\n")
        f.write(f"Max objects per sample: {args.max_objects}\n")
        if args.seed is not None:
            f.write(f"Seed: {args.seed}\n")

    print("Generating dataset with parameters:")
    print(f"  Output directory: {args.out_dir}")
    print(f"  Grid size: {args.D}x{args.H}x{args.W}")
    print(f"  Number of samples: {args.num_samples}")
    print(f"  Min objects per sample: {args.min_objects}")
    print(f"  Max objects per sample: {args.max_objects}")
    time_start = time.time()

    data_output_dir = os.path.join(args.out_dir, "data")
    os.makedirs(data_output_dir, exist_ok=True)

    generate_and_save(
        out_dir=data_output_dir,
        grid_size=[args.D, args.H, args.W],
        num_samples=args.num_samples,
        min_objects=args.min_objects,
        max_objects=args.max_objects,
        seed=args.seed,
    )

    time_end = time.time()
    print(f"Dataset generation completed in {time_end - time_start:.2f} seconds.")
    print(f"Data saved to {args.out_dir}")
    with open(log_file, "a") as f:
        f.write(
            f"Dataset generation completed in {time_end - time_start:.2f} seconds.\n"
        )
        f.write(f"Data saved to {args.out_dir}\n")
    print(f"Log saved to {log_file}")

    if args.num_visualize > 0:
        visualize_output = os.path.join(args.out_dir, "visualizations")
        os.makedirs(visualize_output, exist_ok=True)
        # load output samples at random and visualize
        output_files = [
            os.path.join(data_output_dir, f)
            for f in os.listdir(data_output_dir)
            if f.endswith(".npz")
        ]
        random.shuffle(output_files)
        for i in range(min(args.num_visualize, len(output_files))):
            sample = np.load(output_files[i])
            visualize_sample(
                (sample["x"], sample["y"]),
                output_file_name=os.path.join(
                    visualize_output,
                    f"visualization_{i:05d}.png",
                ),
            )
            print(f"Visualized sample {i + 1}/{args.num_visualize}")
