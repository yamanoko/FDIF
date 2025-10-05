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

from fdslxsdf4seg.basic_sdf import (
    ConcaveCylinder,
    Cone,
    ConeCylinder,
    ConvexCylinder,
    Cylinder,
    Octahedron,
    Sphere,
    Torus,
)
from fdslxsdf4seg.revolution.star_revolution import (
    FiveStarRevolution,
    FourStarRevolution,
    ThreeStarRevolution,
)
from fdslxsdf4seg.sector_polygon_prism.concave_sector_polygon_prism import (
    HeptagonConcavePrism,
    HexagonConcavePrism,
    NonagonConcavePrism,
    OctagonConcavePrism,
    PentagonConcavePrism,
    SquareConcavePrism,
    TriangleConcavePrism,
)
from fdslxsdf4seg.sector_polygon_prism.cone_sector_polygon_prism import (
    HeptagonConePrism,
    HexagonConePrism,
    NonagonConePrism,
    OctagonConePrism,
    PentagonConePrism,
    SquareConePrism,
    TriangleConePrism,
)
from fdslxsdf4seg.sector_polygon_prism.convex_sector_polygon_prism import (
    HeptagonConvexPrism,
    HexagonConvexPrism,
    NonagonConvexPrism,
    OctagonConvexPrism,
    PentagonConvexPrism,
    SquareConvexPrism,
    TriangleConvexPrism,
)
from fdslxsdf4seg.sector_polygon_prism.sector_polygon_prism import (
    HeptagonPrism,
    HexagonPrism,
    NonagonPrism,
    OctagonPrism,
    PentagonPrism,
    SquarePrism,
    TrianglePrism,
)
from fdslxsdf4seg.star_polygon_prism.concave_star_prism import (
    EightStarConcavePrism,
    FiveStarConcavePrism,
    SevenStarConcavePrism,
    SixStarConcavePrism,
)
from fdslxsdf4seg.star_polygon_prism.cone_star_prism import (
    EightStarConePrism,
    FiveStarConePrism,
    SevenStarConePrism,
    SixStarConePrism,
)
from fdslxsdf4seg.star_polygon_prism.convex_star_prism import (
    EightStarConvexPrism,
    FiveStarConvexPrism,
    SevenStarConvexPrism,
    SixStarConvexPrism,
)
from fdslxsdf4seg.star_polygon_prism.star_prism import (
    EightStarPrism,
    FiveStarPrism,
    SevenStarPrism,
    SixStarPrism,
)
from fdslxsdf4seg.torus.sector_polygon_torus import (
    HeptagonTorus,
    HexagonTorus,
    NonagonTorus,
    OctagonTorus,
    PentagonTorus,
    SquareTorus,
)
from fdslxsdf4seg.torus.star_torus import (
    EightStarTorus,
    FiveStarTorus,
    SevenStarTorus,
    SixStarTorus,
)


class SDFSegmentationDataset(Dataset):
    def __init__(
        self,
        grid_size: List[int],
        num_volumes: int,
        min_objects: int = 2,
        max_objects: int = 5,
        device: torch.device = None,
        primitives: List[str] = None,
        categories: List[str] = None,
        num_classes: int = None,
        transform: bool = True,
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
            "cylinder": Cylinder,
            "torus": Torus,
            "cone": Cone,
            "octahedron": Octahedron,
            "convexcylinder": ConvexCylinder,
            "concavecylinder": ConcaveCylinder,
            "conecylinder": ConeCylinder,
            # Revolution objects
            "threestarrevolution": ThreeStarRevolution,
            "fourstarrevolution": FourStarRevolution,
            "fivestarrevolution": FiveStarRevolution,
            # Sector polygon prisms
            "triangleprism": TrianglePrism,
            "squareprism": SquarePrism,
            "pentagonprism": PentagonPrism,
            "hexagonprism": HexagonPrism,
            "heptagonprism": HeptagonPrism,
            "octagonprism": OctagonPrism,
            "nonagonprism": NonagonPrism,
            # Convex sector polygon prisms
            "triangleconvexprism": TriangleConvexPrism,
            "squareconvexprism": SquareConvexPrism,
            "pentagonconvexprism": PentagonConvexPrism,
            "hexagonconvexprism": HexagonConvexPrism,
            "heptagonconvexprism": HeptagonConvexPrism,
            "octagonconvexprism": OctagonConvexPrism,
            "nonagonconvexprism": NonagonConvexPrism,
            # Concave sector polygon prisms
            "triangleconcaveprism": TriangleConcavePrism,
            "squareconcaveprism": SquareConcavePrism,
            "pentagonconcaveprism": PentagonConcavePrism,
            "hexagonconcaveprism": HexagonConcavePrism,
            "heptagonconcaveprism": HeptagonConcavePrism,
            "octagonconcaveprism": OctagonConcavePrism,
            "nonagonconcaveprism": NonagonConcavePrism,
            # Cone sector polygon prisms
            "triangleconeprism": TriangleConePrism,
            "squareconeprism": SquareConePrism,
            "pentagonconeprism": PentagonConePrism,
            "hexagonconeprism": HexagonConePrism,
            "heptagonconeprism": HeptagonConePrism,
            "octagonconeprism": OctagonConePrism,
            "nonagonconeprism": NonagonConePrism,
            # Star polygon prisms
            "fivestarprism": FiveStarPrism,
            "sixstarprism": SixStarPrism,
            "sevenstarprism": SevenStarPrism,
            "eightstarprism": EightStarPrism,
            # Star convex prisms
            "fivestarconvexprism": FiveStarConvexPrism,
            "sixstarconvexprism": SixStarConvexPrism,
            "sevenstarconvexprism": SevenStarConvexPrism,
            "eightstarconvexprism": EightStarConvexPrism,
            # Star concave prisms
            "fivestarconcaveprism": FiveStarConcavePrism,
            "sixstarconcaveprism": SixStarConcavePrism,
            "sevenstarconcaveprism": SevenStarConcavePrism,
            "eightstarconcaveprism": EightStarConcavePrism,
            # Star cone prisms
            "fivestarconeprism": FiveStarConePrism,
            "sixstarconeprism": SixStarConePrism,
            "sevenstarconeprism": SevenStarConePrism,
            "eightstarconeprism": EightStarConePrism,
            # Torus variants
            "squaretorus": SquareTorus,
            "pentagontorus": PentagonTorus,
            "hexagontorus": HexagonTorus,
            "heptagontorus": HeptagonTorus,
            "octagontorus": OctagonTorus,
            "nonagontorus": NonagonTorus,
            "fivestartorus": FiveStarTorus,
            "sixstartorus": SixStarTorus,
            "sevenstartorus": SevenStarTorus,
            "eightstartorus": EightStarTorus,
        }

        # カテゴリ別のプリミティブマッピング
        primitive_categories = {
            "basic": [
                "sphere",
                "cylinder",
                "torus",
                "cone",
                "octahedron",
                "convexcylinder",
                "concavecylinder",
                "conecylinder",
            ],
            "revolution": [
                "threestarrevolution",
                "fourstarrevolution",
                "fivestarrevolution",
            ],
            "sector_polygon_prism": [
                "triangleprism",
                "squareprism",
                "pentagonprism",
                "hexagonprism",
                "heptagonprism",
                "octagonprism",
                "nonagonprism",
            ],
            "convex_sector_polygon_prism": [
                "triangleconvexprism",
                "squareconvexprism",
                "pentagonconvexprism",
                "hexagonconvexprism",
                "heptagonconvexprism",
                "octagonconvexprism",
                "nonagonconvexprism",
            ],
            "concave_sector_polygon_prism": [
                "triangleconcaveprism",
                "squareconcaveprism",
                "pentagonconcaveprism",
                "hexagonconcaveprism",
                "heptagonconcaveprism",
                "octagonconcaveprism",
                "nonagonconcaveprism",
            ],
            "cone_sector_polygon_prism": [
                "triangleconeprism",
                "squareconeprism",
                "pentagonconeprism",
                "hexagonconeprism",
                "heptagonconeprism",
                "octagonconeprism",
                "nonagonconeprism",
            ],
            "star_polygon_prism": [
                "fivestarprism",
                "sixstarprism",
                "sevenstarprism",
                "eightstarprism",
            ],
            "star_convex_prism": [
                "fivestarconvexprism",
                "sixstarconvexprism",
                "sevenstarconvexprism",
                "eightstarconvexprism",
            ],
            "star_concave_prism": [
                "fivestarconcaveprism",
                "sixstarconcaveprism",
                "sevenstarconcaveprism",
                "eightstarconcaveprism",
            ],
            "star_cone_prism": [
                "fivestarconeprism",
                "sixstarconeprism",
                "sevenstarconeprism",
                "eightstarconeprism",
            ],
            "sector_polygon_torus": [
                "squaretorus",
                "pentagontorus",
                "hexagontorus",
                "heptagontorus",
                "octagontorus",
                "nonagontorus",
            ],
            "star_torus": [
                "fivestartorus",
                "sixstartorus",
                "sevenstartorus",
                "eightstartorus",
            ],
        }

        # 使用するプリミティブを選択（カテゴリまたは個別指定）
        if categories is not None:
            # カテゴリが指定された場合、該当するプリミティブを収集
            selected_primitive_names = []
            for category in categories:
                if category in primitive_categories:
                    selected_primitive_names.extend(primitive_categories[category])
                else:
                    print(f"Warning: Unknown category '{category}' ignored.")

            # 重複を除去
            selected_primitive_names = list(set(selected_primitive_names))
        elif primitives is not None:
            # 個別のプリミティブが指定された場合
            selected_primitive_names = primitives
        else:
            # デフォルトは全て
            selected_primitive_names = list(all_primitives.keys())

        # num_classesが指定された場合、ランダムに選択
        if num_classes is not None and num_classes > 0:
            if len(selected_primitive_names) > num_classes:
                # 指定されたクラス数にランダムに削減
                print(
                    f"Randomly selecting {num_classes} classes from {len(selected_primitive_names)} available classes."
                )
                selected_primitive_names = random.sample(
                    selected_primitive_names, num_classes
                )
            elif len(selected_primitive_names) < num_classes:
                print(
                    f"Warning: Requested {num_classes} classes, but only {len(selected_primitive_names)} available. Using all available classes."
                )

        print(
            f"Selected primitives ({len(selected_primitive_names)}): {', '.join(sorted(selected_primitive_names))}"
        )

        # 選択されたプリミティブ名を保存（ログ出力用）
        self.selected_primitive_names = selected_primitive_names

        # 選択されたプリミティブのみを使用
        selected_primitives = [
            all_primitives[name]
            for name in selected_primitive_names
            if name in all_primitives
        ]

        # class_id, primitive_class
        # 1から始まるIDを割り当てる
        self.primitive_classes = {
            i + 1: primitive for i, primitive in enumerate(selected_primitives)
        }
        self.min_o = max(1, min_objects)  # 最小オブジェクト数は1以上
        self.max_o = max_objects  # 最大オブジェクト数の制限を削除
        self.transform = transform  # 変形を適用するかどうか

    def __len__(self):
        return self.num_volumes

    def __getitem__(self, idx):
        n_objs = random.randint(self.min_o, self.max_o)
        sdfs = []
        primitive_ids = random.choices(list(self.primitive_classes.keys()), k=n_objs)
        for id in primitive_ids:
            PrimClass = self.primitive_classes[id]
            obj = PrimClass(
                grid_size=[self.D, self.H, self.W],
                device=self.device,
                transform=self.transform,
            )
            s = obj.sdf(self.X, self.Y, self.Z)
            sdfs.append(s)

        x_vol = 128.0 / (torch.pow(torch.abs(torch.stack(sdfs, dim=0)), 3.0) + 1.0)
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
    categories: List[str] = None,
    num_classes: int = None,
    log_file_path: str = None,
    transform: bool = True,
):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    os.makedirs(out_dir, exist_ok=True)

    # サブディレクトリのサイズ（100個ごと）
    batch_size = 100

    ds = SDFSegmentationDataset(
        grid_size=grid_size,
        num_volumes=num_samples + num_val_samples,
        min_objects=min_objects,
        max_objects=max_objects,
        primitives=primitives,
        categories=categories,
        num_classes=num_classes,
        transform=transform,
    )

    # 選択されたプリミティブの詳細情報を表示
    print(f"Dataset created with {len(ds.primitive_classes)} primitive classes:")
    for class_id, primitive_class in ds.primitive_classes.items():
        print(f"  Class {class_id}: {primitive_class.__name__}")

    loader = DataLoader(ds, batch_size=1, num_workers=0)

    data_json = {}
    json_training_list = list()
    json_validation_list = list()
    for i, (x, y) in enumerate(loader):
        x = x[0].numpy() if hasattr(x[0], "numpy") else x[0]
        y = y[0].numpy() if hasattr(y[0], "numpy") else y[0]

        # サブディレクトリ名を決定（batch_0000, batch_0001, ...）
        batch_idx = i // batch_size
        batch_dir = f"batch_{batch_idx:04d}"

        # サブディレクトリを作成
        image_batch_dir = os.path.join(out_dir, "image", batch_dir)
        label_batch_dir = os.path.join(out_dir, "label", batch_dir)
        os.makedirs(image_batch_dir, exist_ok=True)
        os.makedirs(label_batch_dir, exist_ok=True)

        # Remove channel dimension for saving as 3D NIfTI images
        nii_x = nib.Nifti1Image(x, affine=np.eye(4))
        nii_y = nib.Nifti1Image(y, affine=np.eye(4))
        # Save the SDF volume and segmentation mask as separate .nii.gz files
        image_file = os.path.join(image_batch_dir, f"sample_{i:05d}_x.nii.gz")
        nib.save(nii_x, image_file)
        label_file = os.path.join(label_batch_dir, f"sample_{i:05d}_y.nii.gz")
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
            print(f"Saved {i + 1}/{num_samples + num_val_samples} (batch {batch_idx})")
    # Save dataset metadata
    data_json["training"] = json_training_list
    data_json["validation"] = json_validation_list
    data_json_path = os.path.join(out_dir, "data.json")
    with open(data_json_path, "w") as f:
        json.dump(data_json, f, indent=4)
    print(f"Saved dataset metadata to {data_json_path}")

    # 選択されたプリミティブ情報をログファイルに追記
    if log_file_path and hasattr(ds, "selected_primitive_names"):
        with open(log_file_path, "a") as f:
            f.write(
                f"Actually selected primitives ({len(ds.selected_primitive_names)}): {', '.join(ds.selected_primitive_names)}\n"
            )
            f.write("Primitive class ID mapping:\n")
            for class_id, primitive_class in ds.primitive_classes.items():
                primitive_name = primitive_class.__name__
                f.write(f"  {class_id}: {primitive_name}\n")

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
        "--num_classes",
        type=int,
        default=None,
        help="Number of primitive classes to randomly select. If specified, randomly selects this many classes from the available primitives/categories.",
    )
    p.add_argument(
        "--primitives",
        nargs="*",
        default=[
            "sphere",
            "cylinder",
            "torus",
            "cone",
            "octahedron",
            "convexcylinder",
            "concavecylinder",
            "conecylinder",
            "threestarrevolution",
            "fourstarrevolution",
            "fivestarrevolution",
            "triangleprism",
            "squareprism",
            "pentagonprism",
            "hexagonprism",
            "heptagonprism",
            "octagonprism",
            "nonagonprism",
            "triangleconvexprism",
            "squareconvexprism",
            "pentagonconvexprism",
            "hexagonconvexprism",
            "heptagonconvexprism",
            "octagonconvexprism",
            "nonagonconvexprism",
            "triangleconcaveprism",
            "squareconcaveprism",
            "pentagonconcaveprism",
            "hexagonconcaveprism",
            "heptagonconcaveprism",
            "octagonconcaveprism",
            "nonagonconcaveprism",
            "triangleconeprism",
            "squareconeprism",
            "pentagonconeprism",
            "hexagonconeprism",
            "heptagonconeprism",
            "octagonconeprism",
            "nonagonconeprism",
            "fivestarprism",
            "sixstarprism",
            "sevenstarprism",
            "eightstarprism",
            "fivestarconvexprism",
            "sixstarconvexprism",
            "sevenstarconvexprism",
            "eightstarconvexprism",
            "fivestarconcaveprism",
            "sixstarconcaveprism",
            "sevenstarconcaveprism",
            "eightstarconcaveprism",
            "fivestarconeprism",
            "sixstarconeprism",
            "sevenstarconeprism",
            "eightstarconeprism",
            "squaretorus",
            "pentagontorus",
            "hexagontorus",
            "heptagontorus",
            "octagontorus",
            "nonagontorus",
            "fivestartorus",
            "sixstartorus",
            "sevenstartorus",
            "eightstartorus",
        ],
        choices=[
            "sphere",
            "cylinder",
            "torus",
            "cone",
            "octahedron",
            "convexcylinder",
            "concavecylinder",
            "conecylinder",
            "threestarrevolution",
            "fourstarrevolution",
            "fivestarrevolution",
            "triangleprism",
            "squareprism",
            "pentagonprism",
            "hexagonprism",
            "heptagonprism",
            "octagonprism",
            "nonagonprism",
            "triangleconvexprism",
            "squareconvexprism",
            "pentagonconvexprism",
            "hexagonconvexprism",
            "heptagonconvexprism",
            "octagonconvexprism",
            "nonagonconvexprism",
            "triangleconcaveprism",
            "squareconcaveprism",
            "pentagonconcaveprism",
            "hexagonconcaveprism",
            "heptagonconcaveprism",
            "octagonconcaveprism",
            "nonagonconcaveprism",
            "triangleconeprism",
            "squareconeprism",
            "pentagonconeprism",
            "hexagonconeprism",
            "heptagonconeprism",
            "octagonconeprism",
            "nonagonconeprism",
            "fivestarprism",
            "sixstarprism",
            "sevenstarprism",
            "eightstarprism",
            "fivestarconvexprism",
            "sixstarconvexprism",
            "sevenstarconvexprism",
            "eightstarconvexprism",
            "fivestarconcaveprism",
            "sixstarconcaveprism",
            "sevenstarconcaveprism",
            "eightstarconcaveprism",
            "fivestarconeprism",
            "sixstarconeprism",
            "sevenstarconeprism",
            "eightstarconeprism",
            "squaretorus",
            "pentagontorus",
            "hexagontorus",
            "heptagontorus",
            "octagontorus",
            "nonagontorus",
            "fivestartorus",
            "sixstartorus",
            "sevenstartorus",
            "eightstartorus",
        ],
        help="Primitive types to use for generation (default: all primitives)",
    )
    p.add_argument(
        "--categories",
        nargs="*",
        choices=[
            "basic",
            "revolution",
            "sector_polygon_prism",
            "convex_sector_polygon_prism",
            "concave_sector_polygon_prism",
            "cone_sector_polygon_prism",
            "star_polygon_prism",
            "star_convex_prism",
            "star_concave_prism",
            "star_cone_prism",
            "sector_polygon_torus",
            "star_torus",
        ],
        help="Primitive categories to use for generation. If specified, overrides --primitives. Multiple categories can be selected.",
    )
    p.add_argument(
        "--num_visualize", type=int, default=0, help="Number of samples to visualize"
    )
    p.add_argument(
        "--no-transform",
        action="store_true",
        help="Disable transformations for primitives (default: transformations enabled)",
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
        f.write(f"Transform enabled: {not args.no_transform}\n")
        if args.num_classes is not None:
            f.write(f"Number of classes (randomly selected): {args.num_classes}\n")
        if args.categories:
            f.write(f"Categories used: {', '.join(args.categories)}\n")
        else:
            f.write(f"Primitives used: {', '.join(args.primitives)}\n")
        if args.seed is not None:
            f.write(f"Seed: {args.seed}\n")

    print("Generating dataset with parameters:")
    print(f"  Output directory: {args.out_dir}")
    print(f"  Grid size: {args.D}x{args.H}x{args.W}")
    print(f"  Number of samples: {args.num_samples}")
    print(f"  Min objects per sample: {args.min_objects}")
    print(f"  Max objects per sample: {args.max_objects}")
    print(f"  Transform enabled: {not args.no_transform}")
    if args.num_classes is not None:
        print(f"  Number of classes (randomly selected): {args.num_classes}")
    if args.categories:
        print(f"  Categories used: {', '.join(args.categories)}")
    else:
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
        categories=args.categories,
        num_classes=args.num_classes,
        log_file_path=log_file,
        transform=not args.no_transform,
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
