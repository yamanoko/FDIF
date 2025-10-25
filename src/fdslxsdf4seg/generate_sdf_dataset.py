# file: generate_sdf_dataset.py
import argparse
import json
import os
import random
import time
from typing import List, Optional

import nibabel as nib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader, Dataset

from fdslxsdf4seg.combined_union_primitives import (
    generate_combined_union_primitives,
    generate_hybrid_combined_union_primitives,
)
from fdslxsdf4seg.hybrid_primitive import (
    create_hybrid_primitives,
    get_mapper_choices,
)
from fdslxsdf4seg.primitive_registry import (
    DEFAULT_PRIMITIVES,
    get_category_choices,
    get_primitive_choices,
    select_primitives,
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
        num_combined_unions: int = 0,
        sdf_mappers: Optional[List[str]] = None,
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

        # プリミティブの選択
        selected_primitive_names, selected_primitives_dict = select_primitives(
            primitives=primitives, categories=categories, num_classes=num_classes
        )

        # 選択されたプリミティブ名を保存（ログ出力用）
        self.selected_primitive_names = selected_primitive_names

        # 選択されたプリミティブのみを使用
        selected_primitives = list(selected_primitives_dict.values())

        # SDFマッパーの設定
        if sdf_mappers is None:
            sdf_mappers = ["inverse_cube"]  # デフォルト
        self.sdf_mappers = sdf_mappers
        self.sdf_mappers_info = sdf_mappers  # ログ出力用

        # ハイブリッドプリミティブを生成（プリミティブ × マッパーの全組み合わせ）
        self.hybrid_primitives = create_hybrid_primitives(
            selected_primitives, sdf_mappers
        )

        # CombinedObjectUnionプリミティブを生成
        self.combined_union_primitives = {}
        self.hybrid_combined_union_primitives = {}
        if num_combined_unions > 0:
            self.combined_union_primitives = generate_combined_union_primitives(
                num_combinations=num_combined_unions,
                available_primitives=selected_primitive_names,
                seed=None,  # ここでシードを設定することも可能
            )
            # CombinedUnionプリミティブ × マッパーのハイブリッドを生成
            self.hybrid_combined_union_primitives = (
                generate_hybrid_combined_union_primitives(
                    combined_union_primitives=self.combined_union_primitives,
                    mapper_names=sdf_mappers,
                )
            )

        # class_id, hybrid_primitive (ハイブリッドプリミティブ + ハイブリッドCombinedUnion)
        # 1から始まるIDを割り当てる
        self.primitive_classes = {}
        class_id = 1

        # ハイブリッドプリミティブを追加
        for hybrid_key, hybrid in self.hybrid_primitives.items():
            self.primitive_classes[class_id] = hybrid
            class_id += 1

        # ハイブリッドCombinedUnionプリミティブを追加
        for hybrid_key, hybrid in self.hybrid_combined_union_primitives.items():
            self.primitive_classes[class_id] = hybrid
            class_id += 1
        self.min_o = max(1, min_objects)  # 最小オブジェクト数は1以上
        self.max_o = max_objects  # 最大オブジェクト数の制限を削除
        self.transform = transform  # 変形を適用するかどうか

    def __len__(self):
        return self.num_volumes

    def __getitem__(self, idx):
        n_objs = random.randint(self.min_o, self.max_o)
        sdfs = []
        mappers = []  # 各プリミティブのマッパーを追跡
        primitive_ids = random.choices(list(self.primitive_classes.keys()), k=n_objs)
        for id in primitive_ids:
            hybrid = self.primitive_classes[id]

            # ハイブリッドプリミティブ（またはハイブリッドCombinedUnion）のインスタンスを作成
            obj = hybrid(
                grid_size=[self.D, self.H, self.W],
                device=self.device,
                transform=self.transform,
            )

            s = obj.sdf(self.X, self.Y, self.Z)
            sdfs.append(s)
            mappers.append(hybrid.mapper)  # 各プリミティブのマッパーを記録

        # x_volをマッパーで計算（複数のマッパーを混在させる設計）
        # 各SDFに対応するマッパーを適用して、マッピング済みのSDF値を生成
        mapped_sdfs = []
        for sdf, mapper in zip(sdfs, mappers):
            # 各SDFを個別のマッパーで処理
            mapped_sdf = mapper.apply(sdf)
            mapped_sdfs.append(mapped_sdf)

        # マッピング済みSDFを結合して、sumで合成
        x_vol = torch.stack(mapped_sdfs, dim=0).sum(dim=0)
        x_vol = torch.clamp(x_vol, 0.0, 128.0).to(
            torch.uint8
        )  # sdfs は各オブジェクトの SDF を保持
        # SDFが0未満の部分がどれくらいあるかを体積として、体積が大きい順にSDFを並び替える
        # 各オブジェクトのidもそれに伴って並び替える
        stacked_sdfs = torch.stack(sdfs, dim=0).view(n_objs, -1)  # (n_objs, D*H*W)
        # 各オブジェクトの体積を計算
        volumes = (stacked_sdfs <= 0).sum(dim=1)  # (n_objs,)
        # 体積が大きい順にソート
        sorted_indices = torch.argsort(volumes, descending=True)
        sdfs = [sdfs[i] for i in sorted_indices.tolist()]  # (n_objs, D, H, W)
        primitive_ids = [
            primitive_ids[i] for i in sorted_indices.tolist()
        ]  # 各オブジェクトのSDFが0未満の部分がそのオブジェクトのIDとなる
        # 体積の小さいオブジェクトのIDが優先される
        y_vol = torch.zeros_like(x_vol, dtype=torch.int64)
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
    num_combined_unions: int = 0,
    sdf_mappers: Optional[List[str]] = None,
    nnunet_format: bool = False,
    dataset_id: int = 999,
    dataset_name: str = "SDFSynthetic",
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
        num_combined_unions=num_combined_unions,
        sdf_mappers=sdf_mappers,
    )

    # 選択されたプリミティブの詳細情報を表示
    print(f"Dataset created with {len(ds.primitive_classes)} primitive classes:")
    for class_id, hybrid in ds.primitive_classes.items():
        # すべてがハイブリッド（ハイブリッドプリミティブまたはハイブリッドCombinedUnion）
        print(f"  Class {class_id}: {hybrid.get_display_name()}")

    loader = DataLoader(ds, batch_size=1, num_workers=0)

    data_json = {}
    json_training_list = list()
    json_validation_list = list()

    # nnUNet形式用の変数
    nnunet_labels = {"background": 0}
    if nnunet_format:
        # nnUNet形式の場合、ラベル情報を構築
        for class_id, hybrid in ds.primitive_classes.items():
            label_name = hybrid.get_display_name()
            nnunet_labels[label_name] = class_id

    for i, (x, y) in enumerate(loader):
        x = x[0].numpy() if hasattr(x[0], "numpy") else x[0]
        y = y[0].numpy() if hasattr(y[0], "numpy") else y[0]

        # Convert y to int32 for NIfTI compatibility
        y = y.astype(np.int32)

        if nnunet_format:
            # nnUNet形式でのファイル保存
            # ファイル名: {CASE_ID}_0000.nii.gz (画像), {CASE_ID}.nii.gz (ラベル)
            case_id = f"case_{i:05d}"

            # imagesTr または imagesTs ディレクトリに保存
            if i < num_samples:
                image_dir = os.path.join(out_dir, "imagesTr")
                label_dir = os.path.join(out_dir, "labelsTr")
            else:
                # validationデータは別のデータセットとして保存（imagesTs）
                image_dir = os.path.join(out_dir, "imagesTs")
                label_dir = os.path.join(out_dir, "labelsTs")

            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)

            # nnUNet形式: 画像は {CASE_ID}_0000.nii.gz
            image_file = os.path.join(image_dir, f"{case_id}_0000.nii.gz")
            # nnUNet形式: ラベルは {CASE_ID}.nii.gz
            label_file = os.path.join(label_dir, f"{case_id}.nii.gz")

            nii_x = nib.Nifti1Image(x, affine=np.eye(4))
            nii_y = nib.Nifti1Image(y, affine=np.eye(4))
            nib.save(nii_x, image_file)
            nib.save(nii_y, label_file)

            # nnUNet形式ではdataset.jsonに含めない（後で生成）
        else:
            # 既存のMONAI Decathlon形式でのファイル保存
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
            if nnunet_format:
                print(f"Saved {i + 1}/{num_samples + num_val_samples}")
            else:
                print(
                    f"Saved {i + 1}/{num_samples + num_val_samples} (batch {batch_idx})"
                )

    # dataset.jsonまたはdata.jsonの保存
    if nnunet_format:
        # nnUNet形式のdataset.jsonを生成
        nnunet_dataset_json = {
            "channel_names": {
                "0": "SDF"  # SDFボリュームは1チャネル
            },
            "labels": nnunet_labels,
            "numTraining": num_samples,
            "file_ending": ".nii.gz",
        }

        # トレーニングデータセット用のdataset.json
        dataset_json_path = os.path.join(out_dir, "dataset.json")
        with open(dataset_json_path, "w") as f:
            json.dump(nnunet_dataset_json, f, indent=4)
        print(f"Saved nnUNet dataset.json to {dataset_json_path}")

        # validationデータがある場合は、別のデータセットとして保存
        if num_val_samples > 0:
            # validation用の別ディレクトリを作成
            val_dataset_dir = out_dir.replace(
                f"Dataset{dataset_id:03d}_{dataset_name}",
                f"Dataset{dataset_id + 1:03d}_{dataset_name}_Val",
            )

            # imagesTs と labelsTs を新しいディレクトリに移動
            if os.path.exists(os.path.join(out_dir, "imagesTs")):
                import shutil

                os.makedirs(val_dataset_dir, exist_ok=True)
                shutil.move(
                    os.path.join(out_dir, "imagesTs"),
                    os.path.join(val_dataset_dir, "imagesTr"),
                )
                shutil.move(
                    os.path.join(out_dir, "labelsTs"),
                    os.path.join(val_dataset_dir, "labelsTr"),
                )

                # validation用のdataset.jsonを生成
                val_dataset_json = {
                    "channel_names": {"0": "SDF"},
                    "labels": nnunet_labels,
                    "numTraining": num_val_samples,
                    "file_ending": ".nii.gz",
                }
                val_dataset_json_path = os.path.join(val_dataset_dir, "dataset.json")
                with open(val_dataset_json_path, "w") as f:
                    json.dump(val_dataset_json, f, indent=4)
                print(f"Saved validation dataset.json to {val_dataset_json_path}")
                print(
                    f"Validation dataset created as separate dataset: {val_dataset_dir}"
                )
    else:
        # 既存のMONAI Decathlon形式のdata.jsonを保存
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
            f.write(f"SDF Mappers: {', '.join(ds.sdf_mappers_info)}\n")
            f.write("Primitive class ID mapping:\n")
            for class_id, hybrid in ds.primitive_classes.items():
                # すべてがハイブリッド
                f.write(f"  {class_id}: {hybrid.get_display_name()}\n")

            # ハイブリッドCombinedUnionプリミティブの詳細情報も追加
            if ds.hybrid_combined_union_primitives:
                f.write(
                    f"\nHybrid Combined Union Primitives ({len(ds.hybrid_combined_union_primitives)}):\n"
                )
                for (
                    hybrid_key,
                    hybrid_combined_union,
                ) in ds.hybrid_combined_union_primitives.items():
                    combined_union = hybrid_combined_union.combined_union_primitive
                    mapper_name = hybrid_combined_union.mapper.get_name()
                    f.write(f"  {hybrid_key}:\n")
                    f.write(
                        f"    Union: {combined_union.first_class.__name__} ∪ {combined_union.second_class.__name__}\n"
                    )
                    f.write(f"    Mapper: {mapper_name}\n")
                    f.write(f"    First params: {combined_union.first_params}\n")
                    f.write(f"    Second params: {combined_union.second_params}\n")

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
        default=DEFAULT_PRIMITIVES,
        choices=get_primitive_choices(),
        help="Primitive types to use for generation (default: all primitives)",
    )
    p.add_argument(
        "--categories",
        nargs="*",
        choices=get_category_choices(),
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
    p.add_argument(
        "--num_combined_unions",
        type=int,
        default=0,
        help="Number of CombinedObjectUnion primitives to automatically generate (default: 0)",
    )
    p.add_argument(
        "--sdf_mappers",
        nargs="*",
        default=None,
        choices=get_mapper_choices(),
        help=f"SDF mapper functions to use for generating x_vol. Available: {', '.join(get_mapper_choices())}. (default: inverse_cube)",
    )
    p.add_argument(
        "--nnunet_format",
        action="store_true",
        help="Generate dataset in nnUNet format instead of MONAI Decathlon format. Validation data will be saved as a separate dataset.",
    )
    p.add_argument(
        "--dataset_id",
        type=int,
        default=999,
        help="Dataset ID for nnUNet format (e.g., 999 for Dataset999_Name). Only used with --nnunet_format.",
    )
    p.add_argument(
        "--dataset_name",
        type=str,
        default="SDFSynthetic",
        help="Dataset name for nnUNet format (e.g., 'SDFSynthetic' for Dataset999_SDFSynthetic). Only used with --nnunet_format.",
    )
    args = p.parse_args()

    if not args.out_dir:
        # 出力ディレクトリが指定されていない場合
        if args.nnunet_format:
            # nnUNet形式の場合: Dataset{ID}_{Name} 形式
            args.out_dir = f"outputs/Dataset{args.dataset_id:03d}_{args.dataset_name}"
        else:
            # MONAI形式の場合: カレントディレクトリに日付と時刻を付けて作成
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            args.out_dir = f"outputs/{timestamp}"

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    # ログをoutputディレクトリに保存
    log_file = os.path.join(args.out_dir, "generation_log.txt")
    with open(log_file, "w") as f:
        f.write(f"Output directory: {args.out_dir}\n")
        f.write(
            f"Output format: {'nnUNet' if args.nnunet_format else 'MONAI Decathlon'}\n"
        )
        if args.nnunet_format:
            f.write(f"Dataset ID: {args.dataset_id:03d}\n")
            f.write(f"Dataset name: {args.dataset_name}\n")
        f.write(f"Grid size: {args.D}x{args.H}x{args.W}\n")
        f.write(f"Number of samples: {args.num_samples}\n")
        f.write(f"Min objects per sample: {args.min_objects}\n")
        f.write(f"Max objects per sample: {args.max_objects}\n")
        f.write(f"Transform enabled: {not args.no_transform}\n")
        f.write(f"Number of combined unions: {args.num_combined_unions}\n")
        if args.sdf_mappers:
            f.write(f"SDF Mappers: {', '.join(args.sdf_mappers)}\n")
        else:
            f.write("SDF Mappers: inverse_cube (default)\n")
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
    print(f"  Output format: {'nnUNet' if args.nnunet_format else 'MONAI Decathlon'}")
    if args.nnunet_format:
        print(f"  Dataset ID: {args.dataset_id:03d}")
        print(f"  Dataset name: {args.dataset_name}")
    print(f"  Grid size: {args.D}x{args.H}x{args.W}")
    print(f"  Number of samples: {args.num_samples}")
    print(f"  Min objects per sample: {args.min_objects}")
    print(f"  Max objects per sample: {args.max_objects}")
    print(f"  Transform enabled: {not args.no_transform}")
    print(f"  Number of combined unions: {args.num_combined_unions}")
    if args.sdf_mappers:
        print(f"  SDF Mappers: {', '.join(args.sdf_mappers)}")
    else:
        print("  SDF Mappers: inverse_cube (default)")
    if args.num_classes is not None:
        print(f"  Number of classes (randomly selected): {args.num_classes}")
    if args.categories:
        print(f"  Categories used: {', '.join(args.categories)}")
    else:
        print(f"  Primitives used: {', '.join(args.primitives)}")
    time_start = time.time()

    # nnUNet形式の場合、出力先はargs.out_dir直下、MONAI形式の場合はdata/サブディレクトリ
    if args.nnunet_format:
        data_output_dir = args.out_dir
    else:
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
        num_combined_unions=args.num_combined_unions,
        sdf_mappers=args.sdf_mappers,
        nnunet_format=args.nnunet_format,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
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

        if args.nnunet_format:
            # nnUNet形式の場合、imagesTrから直接読み込む
            images_dir = os.path.join(data_output_dir, "imagesTr")
            labels_dir = os.path.join(data_output_dir, "labelsTr")

            # 全画像ファイルをリスト化
            image_files = sorted(
                [f for f in os.listdir(images_dir) if f.endswith("_0000.nii.gz")]
            )
            random.shuffle(image_files)

            print(f"Total samples available for visualization: {len(image_files)}")
            for i in range(min(args.num_visualize, len(image_files))):
                image_filename = image_files[i]
                case_id = image_filename.replace("_0000.nii.gz", "")
                image_path = os.path.join(images_dir, image_filename)
                label_path = os.path.join(labels_dir, f"{case_id}.nii.gz")

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
        else:
            # MONAI形式の場合、data.jsonから読み込む
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
