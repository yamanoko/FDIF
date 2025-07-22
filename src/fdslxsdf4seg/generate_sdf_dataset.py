# file: generate_sdf_dataset.py
import argparse
import json
import os
import random
import time
from typing import List

import nibabel as nib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
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
            t_x = random.uniform(-0.35, 0.35) * grid_size[0]
            t_y = random.uniform(-0.35, 0.35) * grid_size[1]
            t_z = random.uniform(-0.35, 0.35) * grid_size[2]
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
            radius = random.uniform(min(D, H, W) * 0.05, min(D, H, W) * 0.2)
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
            hx = random.uniform(D * 0.05, D * 0.2)
            hy = random.uniform(H * 0.05, H * 0.2)
            hz = random.uniform(W * 0.05, W * 0.2)
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
            radius = random.uniform(min(D, H) * 0.05, min(D, H) * 0.2)
        if height is None:
            height = random.uniform(W * 0.2, W * 0.3)
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
            major_r = random.uniform(min(D, H) * 0.1, min(D, H) * 0.3)
        if minor_r is None:
            minor_r = major_r * random.uniform(0.1, 0.3)
        self.R = major_r
        self.r = minor_r

    def _sdf(self, x, y, z):
        q = torch.stack([x, z], dim=0).norm(dim=0) - self.R
        q = torch.stack([q, y], dim=0).norm(dim=0) - self.r
        return q

    def max_distance(self):
        return self.R + self.r


class Cone(SDFObject):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius=None,
        height=None,
    ):
        super().__init__(grid_size, device, center, transform)
        D, H, W = grid_size
        if radius is None:
            radius = random.uniform(min(D, H) * 0.05, min(D, H) * 0.2)
        if height is None:
            height = random.uniform(W * 0.2, W * 0.3)
        self.radius = radius
        self.height = height

    def _sdf(self, x, y, z):
        p = torch.stack(
            [
                torch.stack([x, z], dim=0).norm(dim=0) - self.radius,
                y + (self.height / 2),
            ],
            dim=0,
        )
        e = torch.stack(
            [
                torch.ones_like(p[0]) * -self.radius,
                torch.ones_like(p[0]) * self.height,
            ],
            dim=0,
        )
        q = p - e * torch.clamp(
            torch.sum(p * e, dim=0) / torch.sum(e * e, dim=0), min=0.0, max=1.0
        )
        d = torch.norm(q, dim=0)
        max_q = torch.max(q, dim=0).values
        mask = max_q > 0.0
        min_val = torch.min(torch.stack([d, p[1]], dim=0), dim=0).values
        return torch.where(mask, d, -min_val)

    def max_distance(self):
        return float(max(self.radius, self.height))


class HexagonalPrism(SDFObject):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius=None,
        height=None,
    ):
        super().__init__(grid_size, device, center, transform)
        D, H, W = grid_size
        if radius is None:
            radius = random.uniform(min(D, H) * 0.05, min(D, H) * 0.2)
        if height is None:
            height = random.uniform(W * 0.1, W * 0.3)
        self.radius = radius
        self.height = height

    def _sdf(self, x, y, z):
        k = torch.tensor(
            [[-0.8660254], [0.5], [0.57735]], device=self.device
        )  # cos(30°), sin(30°), 1/sqrt(3)
        k = k.view(3, 1, 1, 1).expand(3, *self.grid_size)
        p = torch.stack([x, y], dim=0)
        p = torch.abs(p)
        p -= (
            2.0
            * torch.min(
                torch.stack(
                    [torch.sum(k[:2] * p, dim=0), torch.zeros_like(k[0])], dim=0
                ),
                dim=0,
            ).values
            * k[:2]
        )
        p -= torch.stack(
            [
                torch.clamp(p[0], min=-k[2] * self.radius, max=k[2] * self.radius),
                torch.ones_like(p[1]) * self.radius,
            ],
            dim=0,
        )
        perp = torch.norm(p, dim=0) * torch.sign(p[1])
        along = torch.abs(z) - self.height / 2.0
        outside = torch.clamp(torch.stack([perp, along], dim=0), min=0.0)
        inside = torch.clamp(
            torch.max(torch.stack([perp, along], dim=0), dim=0).values, max=0.0
        )
        return torch.norm(outside, dim=0) + inside

    def max_distance(self):
        return float(max(self.radius, self.height))


class SDFSegmentationDataset(Dataset):
    def __init__(
        self,
        grid_size: List[int],
        num_volumes: int,
        min_objects: int = 2,
        max_objects: int = 5,
        device: torch.device = None,
        primitives: List[str] = None,
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

        # プリミティブの名前とクラスのマッピング
        all_primitives = {
            "sphere": Sphere,
            "box": Box,
            "cylinder": Cylinder,
            "torus": Torus,
            "cone": Cone,
            "hex_prism": HexagonalPrism,
        }

        # 使用するプリミティブを選択（デフォルトは全て）
        if primitives is None:
            primitives = list(all_primitives.keys())

        # 選択されたプリミティブのみを使用
        selected_primitives = [
            all_primitives[name] for name in primitives if name in all_primitives
        ]

        # class_id, primitive_class
        # 1から始まるIDを割り当てる
        self.primitive_classes = {
            i + 1: primitive for i, primitive in enumerate(selected_primitives)
        }
        self.min_o = max(1, min_objects)  # 最小オブジェクト数は1以上
        self.max_o = max_objects  # 最大オブジェクト数の制限を削除

    def __len__(self):
        return self.num_volumes

    def __getitem__(self, idx):
        n_objs = random.randint(self.min_o, self.max_o)
        sdfs = []
        max_ds = []
        primitive_ids = random.choices(list(self.primitive_classes.keys()), k=n_objs)
        for id in primitive_ids:
            PrimClass = self.primitive_classes[id]
            obj = PrimClass(
                grid_size=[self.D, self.H, self.W], device=self.device, transform=True
            )
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
        volumes = (stacked_sdfs <= 0).sum(dim=1)  # (n_objs,)
        # 体積が大きい順にソート
        sorted_indices = torch.argsort(volumes, descending=True)
        sdfs = [sdfs[i] for i in sorted_indices.tolist()]  # (n_objs, D, H, W)
        primitive_ids = [
            primitive_ids[i] for i in sorted_indices.tolist()
        ]  # 各オブジェクトのSDFが0未満の部分がそのオブジェクトのIDとなる
        # 体積の小さいオブジェクトのIDが優先される
        y_vol = torch.zeros_like(x_vol, dtype=torch.uint8)
        for i, obj_id in enumerate(primitive_ids):
            mask = (sdfs[i] <= 0).to(torch.uint8)
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
    num_val_samples: int = 0,
    primitives: List[str] = None,
):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "image"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "label"), exist_ok=True)
    ds = SDFSegmentationDataset(
        grid_size=grid_size,
        num_volumes=num_samples + num_val_samples,
        min_objects=min_objects,
        max_objects=max_objects,
        primitives=primitives,
    )
    loader = DataLoader(ds, batch_size=1, num_workers=0)

    data_json = {}
    json_training_list = list()
    json_validation_list = list()
    for i, (x, y) in enumerate(loader):
        x = x[0].numpy() if hasattr(x[0], "numpy") else x[0]
        y = y[0].numpy() if hasattr(y[0], "numpy") else y[0]
        # Remove channel dimension for saving as 3D NIfTI images
        nii_x = nib.Nifti1Image(x, affine=np.eye(4))
        nii_y = nib.Nifti1Image(y, affine=np.eye(4))
        # Save the SDF volume and segmentation mask as separate .nii.gz files
        image_file = os.path.join(out_dir, "image", f"sample_{i:05d}_x.nii.gz")
        nib.save(nii_x, image_file)
        label_file = os.path.join(out_dir, "label", f"sample_{i:05d}_y.nii.gz")
        nib.save(nii_y, label_file)
        if i < num_samples:
            json_training_list.append(
                {
                    "image": os.path.abspath(image_file),
                    "label": os.path.abspath(label_file),
                    "id": f"sample_{i:05d}",
                }
            )
        else:
            json_validation_list.append(
                {
                    "image": os.path.abspath(image_file),
                    "label": os.path.abspath(label_file),
                    "id": f"sample_{i:05d}",
                }
            )
        if i % 50 == 0:
            print(f"Saved {i + 1}/{num_samples}")
    # Save dataset metadata
    data_json["training"] = json_training_list
    data_json["validation"] = json_validation_list
    data_json_path = os.path.join(out_dir, "data.json")
    with open(data_json_path, "w") as f:
        json.dump(data_json, f, indent=4)
    print(f"Saved dataset metadata to {data_json_path}")
    print("Done.")


def visualize_sample(sample, output_file_name):
    """
    Visualize sample using Plotly (lightweight alternative to matplotlib)
    """
    x, y = sample

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("SDF Volume", "Object Masks"),
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
    )

    # Left plot: SDF Volume visualization
    # Sample points where SDF > threshold for visualization
    threshold = 20
    z_indices, y_indices, x_indices = np.where(x > threshold)

    if len(z_indices) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=x_indices,
                y=y_indices,
                z=z_indices,
                mode="markers",
                marker=dict(size=2, color="blue", opacity=0.6),
                name="SDF > threshold",
            ),
            row=1,
            col=1,
        )

    # Right plot: Object Masks visualization
    unique_objects = np.unique(y)
    colors = px.colors.qualitative.Set1

    for i, obj_id in enumerate(unique_objects):
        if obj_id == 0:
            continue

        z_obj, y_obj, x_obj = np.where(y == obj_id)

        if len(z_obj) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=x_obj,
                    y=y_obj,
                    z=z_obj,
                    mode="markers",
                    marker=dict(size=2, color=colors[i % len(colors)], opacity=0.7),
                    name=f"Object {obj_id}",
                ),
                row=1,
                col=2,
            )

    # Update layout
    fig.update_layout(
        title="SDF Dataset Sample Visualization",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        scene2=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        showlegend=True,
        width=1000,
        height=500,
    )

    # Save as PNG (static)
    fig.write_image(output_file_name)

    # Create slice visualization
    slice_index = x.shape[0] // 2
    slice_fig = go.Figure(
        data=go.Heatmap(
            z=x[slice_index, :, :],
            colorscale="viridis",
            colorbar=dict(title="SDF Value"),
        )
    )

    slice_fig.update_layout(
        title=f"Slice at index {slice_index}", xaxis_title="X", yaxis_title="Y"
    )

    slice_filename = output_file_name.replace(".png", "_slice.png")
    slice_fig.write_image(slice_filename)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str)
    p.add_argument("--D", type=int, default=64)
    p.add_argument("--H", type=int, default=64)
    p.add_argument("--W", type=int, default=64)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--num_val_samples", type=int, default=0)
    p.add_argument("--min_objects", type=int, default=2)
    p.add_argument("--max_objects", type=int, default=5)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--primitives",
        nargs="*",
        default=["sphere", "box", "cylinder", "torus", "cone", "hex_prism"],
        choices=["sphere", "box", "cylinder", "torus", "cone", "hex_prism"],
        help="Primitive types to use for generation (default: all primitives)",
    )
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
        f.write(f"Primitives used: {', '.join(args.primitives)}\n")
        if args.seed is not None:
            f.write(f"Seed: {args.seed}\n")

    print("Generating dataset with parameters:")
    print(f"  Output directory: {args.out_dir}")
    print(f"  Grid size: {args.D}x{args.H}x{args.W}")
    print(f"  Number of samples: {args.num_samples}")
    print(f"  Min objects per sample: {args.min_objects}")
    print(f"  Max objects per sample: {args.max_objects}")
    print(f"  Primitives used: {', '.join(args.primitives)}")
    time_start = time.time()

    data_output_dir = os.path.join(args.out_dir, "data")

    generate_and_save(
        out_dir=data_output_dir,
        grid_size=[args.D, args.H, args.W],
        num_samples=args.num_samples,
        num_val_samples=args.num_val_samples,
        min_objects=args.min_objects,
        max_objects=args.max_objects,
        seed=args.seed,
        primitives=args.primitives,
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
        print(f"Visualizing {args.num_visualize} samples...")
        visualize_output = os.path.join(args.out_dir, "visualizations")
        os.makedirs(visualize_output, exist_ok=True)
        # load output samples at random and visualize
        output_json_path = os.path.join(data_output_dir, "data.json")
        with open(output_json_path, "r") as f:
            data_json = json.load(f)
        files_path = data_json["training"]
        random.shuffle(files_path)
        print(f"Total samples available for visualization: {len(files_path)}")
        for i in range(min(args.num_visualize, len(files_path))):
            file_info = files_path[i]
            image_path = file_info["image"]
            label_path = file_info["label"]
            print("loading", image_path, label_path)
            x = nib.load(image_path).get_fdata()
            y = nib.load(label_path).get_fdata()
            print(type(x), x.shape, type(y), y.shape)
            print(f"Visualizing sample {i + 1}/{args.num_visualize}...")
            visualize_sample(
                (x, y),
                output_file_name=os.path.join(
                    visualize_output,
                    f"visualization_{i:05d}.png",
                ),
            )
            print(f"Visualized sample {i + 1}/{args.num_visualize}")
