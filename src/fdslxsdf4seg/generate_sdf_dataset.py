# file: generate_sdf_dataset.py
import argparse
import json
import os
import random
import time
from typing import List

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# --- SDFベースクラス ---
class SDFObject:
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center: List[float] = None,
        transform: bool = False,
    ):
        self.device = device
        self.grid_size = grid_size  # [D, H, W]
        self.transform = transform
        T, R, S = (
            torch.eye(4, device=self.device),  # 平行移動行列
            torch.eye(4, device=self.device),  # 回転行列
            torch.eye(4, device=self.device),  # せん断行列
        )
        if center is not None:
            # 中心座標を指定する場合は平行移動行列を設定
            t_x, t_y, t_z = center
        else:
            t_x = random.uniform(-0.2, 0.2) * grid_size[0] + grid_size[0] / 2.0
            t_y = random.uniform(-0.2, 0.2) * grid_size[1] + grid_size[1] / 2.0
            t_z = random.uniform(-0.2, 0.2) * grid_size[2] + grid_size[2] / 2.0
        T = self.tranlate_matrix(t_x, t_y, t_z)
        if transform:
            # 回転角度を生成
            angle_x = random.uniform(-torch.pi, torch.pi)
            angle_y = random.uniform(-torch.pi, torch.pi)
            angle_z = random.uniform(-torch.pi, torch.pi)
            # せん断量を生成
            shx = random.uniform(-0.1, 0.1)
            shy = random.uniform(-0.1, 0.1)
            shz = random.uniform(-0.1, 0.1)
            R, S = (
                self.rotate_matrix(angle_x, angle_y, angle_z),
                self.shear_matrix(shx, shy, shz),
            )
        # 変換行列を計算
        # T: 平行移動行列, R: 回転行列, S: せん断行列
        # 変換行列は T * R * S の順で適用される
        # 変換行列は4x4の行列
        self.transform_matrix = torch.matmul(torch.matmul(T, R), S)
        # 逆変換行列を計算
        self.inv_transform_matrix = torch.inverse(self.transform_matrix)

    def sdf(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        x, y, z: meshgrid 上の座標テンソル (shape=(D,H,W))
        戻り値: 各点の signed distance (同shape)
        """
        x, y, z = self.applied_transform(x, y, z)
        return self._sdf(x, y, z)

    def _sdf(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
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

    def tranlate_matrix(self, tx, ty, tz):
        """
        X, Y, Z 軸方向の平行移動行列を生成
        tx, ty, tz: 平行移動量
        戻り値: 平行移動行列 (4x4)
        """
        T = torch.tensor(
            [
                [1, 0, 0, tx],
                [0, 1, 0, ty],
                [0, 0, 1, tz],
                [0, 0, 0, 1],
            ],
            device=self.device,
            dtype=torch.float32,
        )
        return T

    def rotate_matrix(self, angle_x, angle_y, angle_z):
        """
        X, Y, Z 軸周りの回転行列を生成
        angle_x, angle_y, angle_z: ラジアン単位の回転角度
        戻り値: 回転行列 (4x4)
        """
        import math

        Rx = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, math.cos(angle_x), -math.sin(angle_x), 0],
                [0, math.sin(angle_x), math.cos(angle_x), 0],
                [0, 0, 0, 1],
            ],
            device=self.device,
        )
        Ry = torch.tensor(
            [
                [math.cos(angle_y), 0, math.sin(angle_y), 0],
                [0, 1, 0, 0],
                [-math.sin(angle_y), 0, math.cos(angle_y), 0],
                [0, 0, 0, 1],
            ],
            device=self.device,
        )
        Rz = torch.tensor(
            [
                [math.cos(angle_z), -math.sin(angle_z), 0, 0],
                [math.sin(angle_z), math.cos(angle_z), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            device=self.device,
        )
        return torch.matmul(torch.matmul(Rz, Ry), Rx)

    def shear_matrix(self, shx, shy, shz):
        """
        X, Y, Z 軸方向のせん断行列を生成
        shx, shy, shz: せん断量
        戻り値: せん断行列 (4x4)
        """
        S = torch.tensor(
            [
                [1, shx, shy, 0],
                [shx, 1, shz, 0],
                [shy, shz, 1, 0],
                [0, 0, 0, 1],
            ],
            device=self.device,
        )
        return S

    def applied_transform(self, x, y, z):
        """
        座標 (x, y, z) に逆変換行列を適用
        戻り値: 逆変換後の座標 (x', y', z')
        """

        coords = torch.stack(
            [x.flatten(), y.flatten(), z.flatten(), torch.ones_like(x.flatten())], dim=0
        )
        coords = coords.to(self.device)
        transformed_coords = torch.matmul(self.inv_transform_matrix, coords)
        transformed_coords = transformed_coords[:3, :].view(3, *x.shape)  # (3, D, H, W)
        return transformed_coords[0], transformed_coords[1], transformed_coords[2]


# --- 各種プリミティブ実装 （ランダム化ロジックをコンストラクタ内に移動） ---
class Sphere(SDFObject):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        transform=False,
        center=None,
        radius=None,
    ):
        super().__init__(grid_size, device, center, transform)
        D, H, W = grid_size
        # ランダム化
        if radius is None:
            radius = random.uniform(min(D, H, W) * 0.3, min(D, H, W) * 0.5)
        # パラメータ設定
        self.radius = radius

    def _sdf(self, x, y, z):
        p = torch.stack([x, y, z], dim=0)
        return torch.norm(p, dim=0) - self.radius

    def max_distance(self):
        return self.radius


class Box(SDFObject):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        half_extents=None,
    ):
        super().__init__(grid_size, device, center, transform)
        D, H, W = grid_size
        if half_extents is None:
            hx = random.uniform(D * 0.1, D * 0.3)
            hy = random.uniform(H * 0.1, H * 0.3)
            hz = random.uniform(W * 0.1, W * 0.3)
            half_extents = (hz, hy, hx)
        self.half = torch.tensor(half_extents, device=device).view(3, 1, 1, 1)

    def _sdf(self, x, y, z):
        p = torch.stack([x, y, z], dim=0)
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
        transform=False,
        radius=None,
        height=None,
        axis=2,
    ):
        super().__init__(grid_size, device, center, transform)
        D, H, W = grid_size
        if radius is None:
            radius = random.uniform(min(D, H) * 0.3, min(D, H) * 0.5)
        if height is None:
            height = random.uniform(W * 0.2, W * 0.6)
        self.radius = radius
        self.h = height / 2.0
        self.axis = axis

    def _sdf(self, x, y, z):
        p = torch.stack([x, y, z], dim=0)
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
        transform=False,
        major_r=None,
        minor_r=None,
    ):
        super().__init__(grid_size, device, center, transform)
        D, H, W = grid_size
        if major_r is None:
            major_r = random.uniform(min(D, H) * 0.3, min(D, H) * 0.5)
        if minor_r is None:
            minor_r = major_r * random.uniform(0.3, 0.6)
        self.R = major_r
        self.r = minor_r

    def _sdf(self, x, y, z):
        q = torch.stack([x, z], dim=0).norm(dim=0) - self.R
        q = torch.stack([q, y], dim=0).norm(dim=0) - self.r
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
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        zs = torch.linspace(-self.D / 2, self.D / 2 - 1, self.D, device=self.device)
        ys = torch.linspace(-self.H / 2, self.H / 2 - 1, self.H, device=self.device)
        xs = torch.linspace(-self.W / 2, self.W / 2 - 1, self.W, device=self.device)
        self.Z, self.Y, self.X = torch.meshgrid(zs, ys, xs, indexing="ij")

        primitives = [Sphere, Box, Cylinder, Torus]
        # class_id, primitive_class
        # 1から始まるIDを割り当てる
        self.primitive_classes = {
            i + 1: primitive for i, primitive in enumerate(primitives)
        }
        self.min_o = max(1, min_objects)  # 最小オブジェクト数は1以上
        self.max_o = min(
            max_objects, len(self.primitive_classes)
        )  # 最大オブジェクト数はクラス数以下

    def __len__(self):
        return self.num_volumes

    def __getitem__(self, idx):
        n_objs = random.randint(self.min_o, self.max_o)
        sdfs = []
        max_ds = []
        primitive_ids = random.sample(list(self.primitive_classes.keys()), n_objs)
        for id in primitive_ids:
            PrimClass = self.primitive_classes[id]
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
        x_vol = torch.clamp(x_vol, 0.0, 128.0).to(
            torch.uint8
        )  # sdfs は各オブジェクトの SDF を保持
        # SDFが0未満の部分がどれくらいあるかを体積として、体積が大きい順にSDFを並び替える
        # 各オブジェクトのidもそれに伴って並び替える
        stacked_sdfs = torch.stack(sdfs, dim=0)
        stacked_sdfs = stacked_sdfs.view(n_objs, -1)  # (n_objs, D*H*W)
        # 各オブジェクトの体積を計算
        volumes = (stacked_sdfs < 0).sum(dim=1)  # (n_objs,)
        # 体積が大きい順にソート
        sorted_indices = torch.argsort(volumes, descending=True)
        sdfs = [sdfs[i] for i in sorted_indices.tolist()]  # (n_objs, D, H, W)
        primitive_ids = [
            primitive_ids[i] for i in sorted_indices.tolist()
        ]  # 各オブジェクトのSDFが0未満の部分がそのオブジェクトのIDとなる
        # 体積の小さいオブジェクトのIDが優先される
        y_vol = torch.zeros_like(x_vol, dtype=torch.uint8)
        for i, obj_id in enumerate(primitive_ids):
            mask = (sdfs[i] < 0).to(torch.uint8)
            # オブジェクトIDをマスクに適用
            y_vol[mask == 1] = obj_id

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
    numpy_dir = os.path.join(out_dir, "numpy")
    nii_dir = os.path.join(out_dir, "nii")
    os.makedirs(numpy_dir, exist_ok=True)
    os.makedirs(os.path.join(nii_dir, "image"), exist_ok=True)
    os.makedirs(os.path.join(nii_dir, "label"), exist_ok=True)
    ds = SDFSegmentationDataset(
        grid_size=grid_size,
        num_volumes=num_samples,
        min_objects=min_objects,
        max_objects=max_objects,
    )
    loader = DataLoader(ds, batch_size=1, num_workers=0)

    data_json_list = list()
    for i, (x, y) in enumerate(loader):
        x = x[0].numpy() if hasattr(x[0], "numpy") else x[0]
        y = y[0].numpy() if hasattr(y[0], "numpy") else y[0]
        fname = os.path.join(numpy_dir, f"sample_{i:05d}.npz")
        np.savez_compressed(fname, x=x, y=y)
        # Remove channel dimension for saving as 3D NIfTI images
        nii_x = nib.Nifti1Image(x, affine=np.eye(4))
        nii_y = nib.Nifti1Image(y, affine=np.eye(4))
        # Save the SDF volume and segmentation mask as separate .nii.gz files
        image_file = os.path.join("image", f"sample_{i:05d}_x.nii.gz")
        nib.save(nii_x, os.path.join(nii_dir, image_file))
        label_file = os.path.join("label", f"sample_{i:05d}_y.nii.gz")
        nib.save(nii_y, os.path.join(nii_dir, label_file))
        data_json_list.append(
            {
                "image": image_file,
                "label": label_file,
                "id": f"sample_{i:05d}",
            }
        )
        if i % 50 == 0:
            print(f"Saved {i + 1}/{num_samples}")
    # Save dataset metadata
    data_json_path = os.path.join(out_dir, "data.json")
    with open(data_json_path, "w") as f:
        json.dump(data_json_list, f, indent=4)
    print(f"Saved dataset metadata to {data_json_path}")
    print("Done.")


def visualize_sample(sample, output_file_name):
    x, y = sample
    # visualize x and y using voxels
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    ax1.voxels(x > 20, edgecolor="k", facecolors="blue", shade=False)
    # yは複数のオブジェクトマスクを持つため、各オブジェクトを異なる色で表示
    unique_objects = np.unique(y)
    colors = plt.cm.get_cmap("tab10", len(unique_objects))
    visualized_y = np.zeros(y.shape + (4,), dtype=np.float32)  # RGBA
    for i, obj_id in enumerate(unique_objects):
        if obj_id == 0:
            continue
        mask = y == obj_id
        color = colors(i)[:3]  # RGB
        visualized_y[mask, :3] = color  # Set RGB channels
        visualized_y[mask, 3] = 1.0  # アルファチャンネルを1に設定
    # アルファチャンネルを0に設定
    visualized_y[y == 0] = [0, 0, 0, 0]  # 背景は透明に設定
    # 色を正規化
    visualized_y = visualized_y / 255.0  # 0-1    # ボクセル表示
    # visualized_yは4次元配列 (D, H, W, 4) で、最後の次元がRGBA
    # voxels関数はボクセルが満たされているかどうかを示す3次元ブール配列と、
    # 色を指定するfacecolors配列を別々に受け取る
    filled_voxels = y > 0  # オブジェクトが存在する場所
    ax2.voxels(filled_voxels, facecolors=visualized_y, edgecolor="k", shade=False)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("SDF Volume")
    ax2.set_title("Object Masks")  # 保存
    plt.tight_layout()
    # fig.savefig("sample_visualization.png")
    plt.savefig(output_file_name)
    plt.close(fig)

    # save a slice of the volume (only x)
    # visualize with a color bar
    slice_index = x.shape[0] // 2
    plt.figure(figsize=(5, 5))
    im = plt.imshow(x[slice_index, :, :], cmap="viridis")
    plt.title(f"Slice at index {slice_index}")
    plt.colorbar(im)
    plt.savefig(output_file_name.replace(".png", "_slice.png"))
    plt.close()


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
