# file: generate_sdf_dataset_classification.py
"""
SSL3D_classification用の分類データセット生成スクリプト

各ボリュームに1つだけのプリミティブを配置し、そのクラスIDをラベルとする。
出力形式はSSL3D_classification (nnUNet式) に合わせたBlosc2 (.b2nd) +
labelsTr.json + splits_final.json + YAML設定ファイル。

クラスの決定方法は、セグメンテーションタスク時と同様に
primitives × mappers × displacements のオプションによって決定される。
"""

import argparse
import json
import os
import random
import time
from typing import List, Optional

import blosc2
import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from sklearn.model_selection import KFold
from torch.utils.data import Dataset

from fdslxsdf4seg.combined_union_primitives import (
    generate_combined_union_primitives,
    generate_hybrid_combined_union_primitives,
)
from fdslxsdf4seg.displaced_primitive import (
    create_displaced_primitives,
    create_hybrid_displaced_primitives,
    get_displacement_choices,
)
from fdslxsdf4seg.displacement_functions import DisplacementRegistry
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
from fdslxsdf4seg.sdf_mapper import MapperRegistry


class SDFClassificationDataset(Dataset):
    """SSL3D_classification用の分類データセット

    各サンプルは1つのプリミティブのみを含み、そのクラスIDがラベルとなる。
    プリミティブはボクセル中央に配置され、grid_scaleによってボクセル内で
    相対的に大きく表示される。
    """

    def __init__(
        self,
        grid_size: List[int],
        samples_per_class: int,
        device: torch.device = None,
        primitives: List[str] = None,
        categories: List[str] = None,
        num_classes: int = None,
        transform: bool = True,
        num_combined_unions: int = 0,
        sdf_mappers: Optional[List[str]] = None,
        displacement_functions: Optional[List[str]] = None,
        mapper_as_augmentation: bool = False,
        displacement_as_augmentation: bool = False,
        grid_scale: float = 0.40,
    ):
        self.D, self.H, self.W = grid_size
        self.samples_per_class = samples_per_class
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.grid_scale = grid_scale

        # メッシュグリッドのスケーリング
        # grid_scaleを適用して座標範囲を縮小し、プリミティブを相対的に拡大する
        zs = torch.linspace(
            -self.D / 2 * grid_scale,
            self.D / 2 * grid_scale,
            self.D,
            device=self.device,
        )
        ys = torch.linspace(
            -self.H / 2 * grid_scale,
            self.H / 2 * grid_scale,
            self.H,
            device=self.device,
        )
        xs = torch.linspace(
            -self.W / 2 * grid_scale,
            self.W / 2 * grid_scale,
            self.W,
            device=self.device,
        )
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

        # Displacement関数の設定
        self.displacement_functions = displacement_functions or []
        self.displacement_functions_info = displacement_functions or []  # ログ出力用

        # Augmentationモードの設定
        self.mapper_as_augmentation = mapper_as_augmentation
        self.displacement_as_augmentation = displacement_as_augmentation

        # Augmentationモード用：マッパーとDisplacement関数のインスタンスを保持
        self.mapper_instances = [MapperRegistry.get(name) for name in sdf_mappers]
        self.displacement_instances = (
            [DisplacementRegistry.get(name) for name in displacement_functions]
            if displacement_functions
            else []
        )

        # ハイブリッドプリミティブを生成（プリミティブ × マッパーの全組み合わせ）
        if mapper_as_augmentation:
            self.hybrid_primitives = create_hybrid_primitives(
                selected_primitives,
                ["inverse_cube"],  # デフォルトマッパーのみ
            )
        else:
            self.hybrid_primitives = create_hybrid_primitives(
                selected_primitives, sdf_mappers
            )

        # DisplacedPrimitiveを生成（displacement関数が指定されている場合）
        self.displaced_primitives = {}
        self.hybrid_displaced_primitives = {}
        if displacement_functions and not displacement_as_augmentation:
            self.displaced_primitives = create_displaced_primitives(
                selected_primitives, displacement_functions
            )
            if mapper_as_augmentation:
                self.hybrid_displaced_primitives = create_hybrid_displaced_primitives(
                    selected_primitives, displacement_functions, ["inverse_cube"]
                )
            else:
                self.hybrid_displaced_primitives = create_hybrid_displaced_primitives(
                    selected_primitives, displacement_functions, sdf_mappers
                )

        # CombinedObjectUnionプリミティブを生成
        self.combined_union_primitives = {}
        self.hybrid_combined_union_primitives = {}
        if num_combined_unions > 0:
            self.combined_union_primitives = generate_combined_union_primitives(
                num_combinations=num_combined_unions,
                available_primitives=selected_primitive_names,
                seed=None,
            )
            if mapper_as_augmentation:
                self.hybrid_combined_union_primitives = (
                    generate_hybrid_combined_union_primitives(
                        combined_union_primitives=self.combined_union_primitives,
                        mapper_names=["inverse_cube"],
                    )
                )
            else:
                self.hybrid_combined_union_primitives = (
                    generate_hybrid_combined_union_primitives(
                        combined_union_primitives=self.combined_union_primitives,
                        mapper_names=sdf_mappers,
                    )
                )

        # class_idの割り当て（0から始まる、分類タスク用）
        self.primitive_classes = {}
        class_id = 0

        # ハイブリッドプリミティブを追加
        for hybrid_key, hybrid in self.hybrid_primitives.items():
            self.primitive_classes[class_id] = hybrid
            class_id += 1

        # ハイブリッドDisplacedプリミティブを追加
        for hybrid_key, hybrid in self.hybrid_displaced_primitives.items():
            self.primitive_classes[class_id] = hybrid
            class_id += 1

        # ハイブリッドCombinedUnionプリミティブを追加
        for hybrid_key, hybrid in self.hybrid_combined_union_primitives.items():
            self.primitive_classes[class_id] = hybrid
            class_id += 1

        self.transform = transform  # 回転・せん断を適用するかどうか
        self.num_total_classes = len(self.primitive_classes)

        # クラスごとに均等にサンプルを割り当てるリストを構築
        self.sample_assignments = []
        for cid in sorted(self.primitive_classes.keys()):
            for _ in range(self.samples_per_class):
                self.sample_assignments.append(cid)

    def __len__(self):
        return len(self.sample_assignments)

    def __getitem__(self, idx):
        """1つのプリミティブを生成し、(x_vol, class_id) を返す

        Returns:
            x_vol: (D, H, W) のfloat32 numpy配列（Z-Score正規化済み）
            class_id: 整数クラスID（0-indexed）
        """
        # 事前割り当てからクラスを取得（クラスごとに均等）
        class_id = self.sample_assignments[idx]
        hybrid = self.primitive_classes[class_id]

        # プリミティブを中央配置でインスタンス化（center=[0,0,0]で平行移動を無効化）
        obj = hybrid(
            grid_size=[self.D, self.H, self.W],
            device=self.device,
            transform=self.transform,
            center=[0, 0, 0],
        )

        # displacement_as_augmentationがTrueの場合、ランダムにdisplacementを適用
        if self.displacement_as_augmentation and self.displacement_instances:
            choices = self.displacement_instances + [None]
            displacement = random.choice(choices)
            if displacement is not None:
                obj.set_displacement_function(displacement.apply)

        # SDF計算
        sdf_val = obj.sdf(self.X, self.Y, self.Z)

        # mapper_as_augmentationがTrueの場合、ランダムにマッパーを選択
        if self.mapper_as_augmentation:
            mapper = random.choice(self.mapper_instances)
        else:
            mapper = hybrid.mapper

        # マッパーを適用してボリュームデータを生成
        x_vol = mapper.apply(sdf_val)
        x_vol = torch.clamp(x_vol, 0.0, 128.0).to(torch.float32)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Z-Score正規化
        x_np = x_vol.cpu().numpy()
        mean = x_np.mean()
        std = x_np.std()
        x_np = (x_np - mean) / max(std, 1e-8)
        x_np = x_np.astype(np.float32)

        return x_np, class_id


# --- Blosc2保存ユーティリティ ---
def save_blosc2(data: np.ndarray, filepath: str):
    """SSL3D_classification互換のBlosc2形式で保存

    Args:
        data: (C, D, H, W) 形状のfloat32 numpy配列
        filepath: 保存先パス（.b2ndで終わる）
    """
    cparams = {
        "codec": blosc2.Codec.ZSTD,
        "clevel": 8,
        "nthreads": 1,
    }
    blosc2.asarray(
        np.ascontiguousarray(data),
        urlpath=filepath,
        cparams=cparams,
        mmap_mode="w+",
    )


# --- クロスバリデーション分割生成 ---
def generate_crossval_split(
    identifiers: List[str], seed: int = 12345, n_splits: int = 5
) -> list:
    """SSL3D_classification互換のKFoldクロスバリデーション分割を生成

    Args:
        identifiers: サンプルIDのリスト
        seed: 乱数シード
        n_splits: フォールド数

    Returns:
        [{"train": [...], "val": [...]}, ...] のリスト
    """
    splits = []
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, val_idx in kfold.split(identifiers):
        train_keys = [identifiers[i] for i in train_idx]
        val_keys = [identifiers[i] for i in val_idx]
        splits.append({"train": train_keys, "val": val_keys})
    return splits


# --- YAML設定ファイル生成 ---
def generate_yaml_config(
    dataset_name: str,
    num_classes: int,
    patch_size: List[int],
    n_splits: int,
    output_path: str,
):
    """SSL3D_classification用のHydra YAML設定ファイルを生成

    Args:
        dataset_name: データセット名
        num_classes: クラス数
        patch_size: パッチサイズ [D, H, W]
        n_splits: クロスバリデーションのフォールド数
        output_path: YAML保存先パス
    """
    yaml_content = f"""# @package _global_
# SSL3D_classification用SDF分類データセット設定
# 生成コマンド: 生成時のgeneration_log.txtを参照
data:
  module:
    _target_: datasets.sdf_classification.SDFClassificationDataModule
    name: {dataset_name}
    data_root_dir: ${{data_dir}}
    batch_size: 1
    train_transforms:
      _target_: augmentation.policies.batchgenerators.get_training_transforms
      patch_size: ${{data.patch_size}}
      rotation_for_DA: 0.523599
      mirror_axes: [0, 1, 2]
      do_dummy_2d_data_aug: False
    test_transforms: null
  cv:
    k: {n_splits}
  num_classes: {num_classes}
  patch_size: [{patch_size[0]}, {patch_size[1]}, {patch_size[2]}]

model:
  task: 'Classification'
  cifar_size: False
  input_channels: 1
  input_dim: 3
  input_shape: ${{data.patch_size}}
  optimizer: AdamW
  lr: 0.0001
  warmstart: 20
  weight_decay: 1e-2
  label_smoothing: 0.2

trainer:
  logger:
    project: {dataset_name}
  accumulate_grad_batches: 48
  max_epochs: 400
  sync_batchnorm: True

metrics:
  - 'f1'
  - 'balanced_acc'
  - 'ap'
  - 'auroc'
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)


# --- データ生成＆保存 ---
def generate_and_save(
    out_dir: str,
    grid_size: List[int],
    samples_per_class: int,
    seed: int = None,
    primitives: List[str] = None,
    categories: List[str] = None,
    num_classes: int = None,
    log_file_path: str = None,
    transform: bool = True,
    num_combined_unions: int = 0,
    sdf_mappers: Optional[List[str]] = None,
    displacement_functions: Optional[List[str]] = None,
    mapper_as_augmentation: bool = False,
    displacement_as_augmentation: bool = False,
    grid_scale: float = 0.40,
    n_splits: int = 5,
    dataset_name: str = "sdf_classification",
):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    os.makedirs(out_dir, exist_ok=True)

    # Blosc2データ保存ディレクトリ
    data_dir = os.path.join(out_dir, "nnUNetResEncUNetLPlans_3d_fullres")
    os.makedirs(data_dir, exist_ok=True)

    ds = SDFClassificationDataset(
        grid_size=grid_size,
        samples_per_class=samples_per_class,
        primitives=primitives,
        categories=categories,
        num_classes=num_classes,
        transform=transform,
        num_combined_unions=num_combined_unions,
        sdf_mappers=sdf_mappers,
        displacement_functions=displacement_functions,
        mapper_as_augmentation=mapper_as_augmentation,
        displacement_as_augmentation=displacement_as_augmentation,
        grid_scale=grid_scale,
    )

    # 選択されたプリミティブの詳細情報を表示
    num_total_samples = len(ds)
    print(f"Dataset created with {ds.num_total_classes} classes (0-indexed):")
    print(f"  Samples per class: {samples_per_class}")
    print(f"  Total samples: {num_total_samples}")
    if mapper_as_augmentation:
        print(
            f"  (Mapper as augmentation: {', '.join(sdf_mappers or ['inverse_cube'])})"
        )
    if displacement_as_augmentation:
        print(
            f"  (Displacement as augmentation: {', '.join(displacement_functions or [])})"
        )
    for class_id, hybrid in ds.primitive_classes.items():
        print(f"  Class {class_id}: {hybrid.get_display_name()}")

    # ラベル辞書を構築
    labels_dict = {}
    all_identifiers = []

    for i in range(num_total_samples):
        x, class_id = ds[i]

        case_id = f"case_{i:05d}"
        all_identifiers.append(case_id)

        # ラベルを記録（整数クラスID）
        labels_dict[case_id] = int(class_id)

        # チャネル次元を追加: (D, H, W) -> (1, D, H, W)
        x_with_channel = x[np.newaxis, ...]

        # Blosc2形式で保存
        b2nd_path = os.path.join(data_dir, f"{case_id}.b2nd")
        save_blosc2(x_with_channel, b2nd_path)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Saved {i + 1}/{num_total_samples} (class={class_id})")

    # labelsTr.json を保存
    labels_path = os.path.join(out_dir, "labelsTr.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels_dict, f, indent=2)
    print(f"Saved labelsTr.json ({len(labels_dict)} entries)")

    # splits_final.json を保存（KFoldクロスバリデーション）
    splits = generate_crossval_split(all_identifiers, seed=12345, n_splits=n_splits)
    splits_path = os.path.join(out_dir, "splits_final.json")
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    print(f"Saved splits_final.json ({n_splits} folds)")

    # YAML設定ファイルを生成
    yaml_path = os.path.join(out_dir, f"{dataset_name}.yaml")
    generate_yaml_config(
        dataset_name=dataset_name,
        num_classes=ds.num_total_classes,
        patch_size=grid_size,
        n_splits=n_splits,
        output_path=yaml_path,
    )
    print(f"Saved YAML config to {yaml_path}")

    # クラス分布の集計
    class_counts = {}
    for cid in labels_dict.values():
        class_counts[cid] = class_counts.get(cid, 0) + 1

    # ログファイルに情報を追記
    if log_file_path:
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(
                f"\nActually selected primitives ({len(ds.selected_primitive_names)}): "
                f"{', '.join(ds.selected_primitive_names)}\n"
            )
            f.write(f"SDF Mappers: {', '.join(ds.sdf_mappers_info)}\n")
            if ds.displacement_functions_info:
                f.write(
                    f"Displacement Functions: {', '.join(ds.displacement_functions_info)}\n"
                )
            f.write(f"Grid scale: {ds.grid_scale}\n")
            f.write(f"Total classes: {ds.num_total_classes}\n")
            f.write(f"Samples per class: {samples_per_class}\n")
            f.write(f"Total samples: {num_total_samples}\n")
            f.write("Class ID mapping:\n")
            for cid, hybrid in ds.primitive_classes.items():
                f.write(f"  {cid}: {hybrid.get_display_name()}\n")

            if ds.mapper_as_augmentation:
                f.write(f"\nMapper as augmentation: {', '.join(ds.sdf_mappers_info)}\n")
            if ds.displacement_as_augmentation:
                f.write(
                    f"Displacement as augmentation: {', '.join(ds.displacement_functions_info)}\n"
                )

            if ds.hybrid_displaced_primitives:
                f.write(
                    f"\nHybrid Displaced Primitives ({len(ds.hybrid_displaced_primitives)}):\n"
                )
                for hk, hd in ds.hybrid_displaced_primitives.items():
                    pname = hd.displaced_primitive.primitive_class.__name__
                    dname = hd.displaced_primitive.displacement_function.get_name()
                    mname = hd.mapper.get_name()
                    f.write(f"  {hk}:\n")
                    f.write(f"    Primitive: {pname}\n")
                    f.write(f"    Displacement: {dname}\n")
                    f.write(f"    Mapper: {mname}\n")

            if ds.hybrid_combined_union_primitives:
                f.write(
                    f"\nHybrid Combined Union Primitives ({len(ds.hybrid_combined_union_primitives)}):\n"
                )
                for hk, hcu in ds.hybrid_combined_union_primitives.items():
                    cu = hcu.combined_union_primitive
                    mname = hcu.mapper.get_name()
                    f.write(f"  {hk}:\n")
                    f.write(
                        f"    Union: {cu.first_class.__name__} ∪ {cu.second_class.__name__}\n"
                    )
                    f.write(f"    Mapper: {mname}\n")
                    f.write(f"    First params: {cu.first_params}\n")
                    f.write(f"    Second params: {cu.second_params}\n")

            # クラス分布
            f.write("\nClass distribution:\n")
            for cid in sorted(class_counts.keys()):
                hybrid_name = ds.primitive_classes[cid].get_display_name()
                f.write(f"  Class {cid} ({hybrid_name}): {class_counts[cid]} samples\n")

            # Cross-validation情報
            f.write(f"\nCross-validation: {n_splits} folds (seed=12345)\n")

    print("Done.")


def visualize_sample(x: np.ndarray, class_id: int, class_name: str, output_file: str):
    """分類サンプルの可視化（3断面スライス表示）

    Args:
        x: (D, H, W) 形状のfloat32 numpy配列
        class_id: クラスID
        class_name: クラスの表示名
        output_file: 出力ファイルパス
    """
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            f"Axial (z={x.shape[0] // 2})",
            f"Coronal (y={x.shape[1] // 2})",
            f"Sagittal (x={x.shape[2] // 2})",
        ),
    )

    # Axial
    fig.add_trace(
        go.Heatmap(
            z=x[x.shape[0] // 2, :, :],
            colorscale="viridis",
            showscale=False,
        ),
        row=1,
        col=1,
    )
    # Coronal
    fig.add_trace(
        go.Heatmap(
            z=x[:, x.shape[1] // 2, :],
            colorscale="viridis",
            showscale=False,
        ),
        row=1,
        col=2,
    )
    # Sagittal
    fig.add_trace(
        go.Heatmap(
            z=x[:, :, x.shape[2] // 2],
            colorscale="viridis",
            showscale=True,
            colorbar=dict(title="Intensity"),
        ),
        row=1,
        col=3,
    )

    fig.update_layout(
        title=f"Class {class_id}: {class_name}",
        width=1200,
        height=400,
    )

    fig.write_image(output_file)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate SDF classification dataset for SSL3D_classification"
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/cls_<timestamp>)",
    )
    p.add_argument("--D", type=int, default=64, help="Depth of the volume")
    p.add_argument("--H", type=int, default=64, help="Height of the volume")
    p.add_argument("--W", type=int, default=64, help="Width of the volume")
    p.add_argument(
        "--samples_per_class",
        type=int,
        default=25,
        help="Number of samples to generate per class (total = samples_per_class × num_classes, default: 25)",
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of primitive classes to randomly select",
    )
    p.add_argument(
        "--primitives",
        nargs="*",
        default=DEFAULT_PRIMITIVES,
        choices=get_primitive_choices(),
        help="Primitive types to use (default: sphere, cylinder, torus, cone)",
    )
    p.add_argument(
        "--categories",
        nargs="*",
        choices=get_category_choices(),
        help="Primitive categories to use (overrides --primitives)",
    )
    p.add_argument(
        "--num_visualize",
        type=int,
        default=0,
        help="Number of samples to visualize",
    )
    p.add_argument(
        "--no-transform",
        action="store_true",
        help="Disable rotation/shear transforms (default: enabled)",
    )
    p.add_argument(
        "--num_combined_unions",
        type=int,
        default=0,
        help="Number of CombinedObjectUnion primitives to generate",
    )
    p.add_argument(
        "--sdf_mappers",
        nargs="*",
        default=None,
        choices=get_mapper_choices(),
        help=f"SDF mapper functions. Available: {', '.join(get_mapper_choices())} (default: inverse_cube)",
    )
    p.add_argument(
        "--displacement_functions",
        nargs="*",
        default=None,
        choices=get_displacement_choices(),
        help=f"Displacement functions. Available: {', '.join(get_displacement_choices())}",
    )
    p.add_argument(
        "--mapper_as_augmentation",
        action="store_true",
        help="Treat SDF mappers as augmentation (random per sample, not separate classes)",
    )
    p.add_argument(
        "--displacement_as_augmentation",
        action="store_true",
        help="Treat displacement functions as augmentation (random per sample, not separate classes)",
    )
    p.add_argument(
        "--grid_scale",
        type=float,
        default=0.40,
        help="Grid coordinate scale factor. Smaller values make primitives appear larger "
        "relative to the volume. (default: 0.40)",
    )
    p.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    p.add_argument(
        "--dataset_name",
        type=str,
        default="sdf_classification",
        help="Dataset name for YAML config and output naming",
    )
    args = p.parse_args()

    # 出力ディレクトリの決定
    if not args.out_dir:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"outputs/cls_{timestamp}"

    os.makedirs(args.out_dir, exist_ok=True)

    # ログファイル
    log_file = os.path.join(args.out_dir, "generation_log.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== SDF Classification Dataset Generation ===\n")
        f.write(f"Output directory: {args.out_dir}\n")
        f.write("Output format: Blosc2 (.b2nd) for SSL3D_classification\n")
        f.write(f"Grid size: {args.D}x{args.H}x{args.W}\n")
        f.write(f"Grid scale: {args.grid_scale}\n")
        f.write(f"Samples per class: {args.samples_per_class}\n")
        f.write(f"Transform enabled: {not args.no_transform}\n")
        f.write(f"Number of combined unions: {args.num_combined_unions}\n")
        if args.sdf_mappers:
            f.write(f"SDF Mappers: {', '.join(args.sdf_mappers)}\n")
        else:
            f.write("SDF Mappers: inverse_cube (default)\n")
        f.write(f"Mapper as augmentation: {args.mapper_as_augmentation}\n")
        if args.displacement_functions:
            f.write(
                f"Displacement Functions: {', '.join(args.displacement_functions)}\n"
            )
        f.write(f"Displacement as augmentation: {args.displacement_as_augmentation}\n")
        f.write(f"Cross-validation folds: {args.n_splits}\n")
        f.write(f"Dataset name: {args.dataset_name}\n")
        if args.num_classes is not None:
            f.write(f"Number of classes (randomly selected): {args.num_classes}\n")
        if args.categories:
            f.write(f"Categories used: {', '.join(args.categories)}\n")
        else:
            f.write(f"Primitives used: {', '.join(args.primitives)}\n")
        if args.seed is not None:
            f.write(f"Seed: {args.seed}\n")

    print("Generating classification dataset with parameters:")
    print(f"  Output directory: {args.out_dir}")
    print("  Format: Blosc2 (.b2nd) for SSL3D_classification")
    print(f"  Grid size: {args.D}x{args.H}x{args.W}")
    print(f"  Grid scale: {args.grid_scale}")
    print(f"  Samples per class: {args.samples_per_class}")
    print(f"  Transform enabled: {not args.no_transform}")
    if args.sdf_mappers:
        print(f"  SDF Mappers: {', '.join(args.sdf_mappers)}")
    else:
        print("  SDF Mappers: inverse_cube (default)")
    print(f"  Mapper as augmentation: {args.mapper_as_augmentation}")
    if args.displacement_functions:
        print(f"  Displacement Functions: {', '.join(args.displacement_functions)}")
    print(f"  Displacement as augmentation: {args.displacement_as_augmentation}")
    print(f"  Cross-validation folds: {args.n_splits}")
    print(f"  Dataset name: {args.dataset_name}")
    if args.categories:
        print(f"  Categories used: {', '.join(args.categories)}")
    else:
        print(f"  Primitives used: {', '.join(args.primitives)}")
    time_start = time.time()

    generate_and_save(
        out_dir=args.out_dir,
        grid_size=[args.D, args.H, args.W],
        samples_per_class=args.samples_per_class,
        seed=args.seed,
        primitives=args.primitives,
        categories=args.categories,
        num_classes=args.num_classes,
        log_file_path=log_file,
        transform=not args.no_transform,
        num_combined_unions=args.num_combined_unions,
        sdf_mappers=args.sdf_mappers,
        displacement_functions=args.displacement_functions,
        mapper_as_augmentation=args.mapper_as_augmentation,
        displacement_as_augmentation=args.displacement_as_augmentation,
        grid_scale=args.grid_scale,
        n_splits=args.n_splits,
        dataset_name=args.dataset_name,
    )

    time_end = time.time()
    elapsed = time_end - time_start
    print(f"Dataset generation completed in {elapsed:.2f} seconds.")
    print(f"Data saved to {args.out_dir}")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\nDataset generation completed in {elapsed:.2f} seconds.\n")
        f.write(f"Data saved to {args.out_dir}\n")
    print(f"Log saved to {log_file}")

    # 可視化
    if args.num_visualize > 0:
        print(f"Visualizing {args.num_visualize} samples...")
        vis_dir = os.path.join(args.out_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # Blosc2ファイルから読み込んで可視化
        b2nd_dir = os.path.join(args.out_dir, "nnUNetResEncUNetLPlans_3d_fullres")
        labels_path = os.path.join(args.out_dir, "labelsTr.json")
        with open(labels_path, "r") as f:
            labels = json.load(f)

        # ランダムにサンプルを選択
        case_ids = list(labels.keys())
        random.shuffle(case_ids)

        # DatasetからクラスID→名前のマッピングを取得
        ds_temp = SDFClassificationDataset(
            grid_size=[args.D, args.H, args.W],
            samples_per_class=1,
            primitives=args.primitives,
            categories=args.categories,
            num_classes=args.num_classes,
            sdf_mappers=args.sdf_mappers,
            displacement_functions=args.displacement_functions,
            mapper_as_augmentation=args.mapper_as_augmentation,
            displacement_as_augmentation=args.displacement_as_augmentation,
            grid_scale=args.grid_scale,
        )
        id_to_name = {
            cid: h.get_display_name() for cid, h in ds_temp.primitive_classes.items()
        }

        for i in range(min(args.num_visualize, len(case_ids))):
            cid_str = case_ids[i]
            cls_id = labels[cid_str]
            cls_name = id_to_name.get(cls_id, f"Unknown({cls_id})")

            # Blosc2から読み込み
            b2nd_path = os.path.join(b2nd_dir, f"{cid_str}.b2nd")
            data = blosc2.open(urlpath=b2nd_path, mode="r")
            x = np.array(data[0])  # チャネル次元を除去: (1,D,H,W) -> (D,H,W)

            print(
                f"  Visualizing {i + 1}/{args.num_visualize}: "
                f"{cid_str} (class {cls_id}: {cls_name})"
            )
            visualize_sample(
                x,
                cls_id,
                cls_name,
                os.path.join(vis_dir, f"vis_{cid_str}.png"),
            )

        print(f"Visualizations saved to {vis_dir}")
