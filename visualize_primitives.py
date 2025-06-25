#!/usr/bin/env python3
"""
各プリミティブの可視化を生成・保存するスクリプト
"""

import os

import numpy as np
import torch

from src.fdslxsdf4seg.generate_sdf_dataset import (
    Box,
    Cylinder,
    Sphere,
    Torus,
    visualize_sample,
)


def generate_primitive_visualizations(output_dir="visualize_output"):
    """各プリミティブを個別に生成し、可視化結果を保存する"""

    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)

    # グリッドサイズとデバイス設定
    grid_size = [32, 32, 32]  # より大きなサイズで詳細な可視化
    device = torch.device("cpu")

    # 座標メッシュを作成
    zs = torch.linspace(0, grid_size[0] - 1, grid_size[0], dtype=torch.float32)
    ys = torch.linspace(0, grid_size[1] - 1, grid_size[1], dtype=torch.float32)
    xs = torch.linspace(0, grid_size[2] - 1, grid_size[2], dtype=torch.float32)
    Z, Y, X = torch.meshgrid(zs, ys, xs, indexing="ij")

    # テスト対象のプリミティブ
    primitives = [
        ("Sphere", Sphere, {"center": (16.0, 16.0, 16.0), "radius": 8.0}),
        ("Box", Box, {"center": (16.0, 16.0, 16.0), "half_extents": (6.0, 6.0, 6.0)}),
        (
            "Cylinder",
            Cylinder,
            {"center": (16.0, 16.0, 16.0), "radius": 6.0, "height": 16.0},
        ),
        (
            "Torus",
            Torus,
            {"center": (16.0, 16.0, 16.0), "major_r": 8.0, "minor_r": 3.0},
        ),
    ]

    print(f"Generating primitive visualizations in '{output_dir}'...")

    for name, PrimClass, params in primitives:
        print(f"Processing {name}...")

        # ランダムシードを固定（変換を無効化）
        torch.manual_seed(42)
        np.random.seed(42)
        import random

        random.seed(42)

        # プリミティブを生成
        primitive = PrimClass(grid_size, device, **params)

        # SDFを計算
        sdf = primitive.sdf(X, Y, Z)

        # SDFを可視化用に変換（0-128の範囲）
        sdf_vis = 128.0 / (torch.pow(torch.abs(sdf), 2.0) + 1.0)
        sdf_vis = torch.clamp(sdf_vis, 0.0, 128.0).to(torch.uint8)

        # セグメンテーションマスクを作成（オブジェクトIDは1）
        mask = (sdf < 0).to(torch.uint8)

        # NumPy配列に変換
        sdf_np = sdf_vis.cpu().numpy()
        mask_np = mask.cpu().numpy()

        # 可視化を実行
        output_file = os.path.join(output_dir, f"{name.lower()}_visualization.png")
        visualize_sample((sdf_np, mask_np), output_file)

        # 統計情報を表示
        inside_count = (sdf < 0).sum().item()
        outside_count = (sdf > 0).sum().item()
        max_dist = primitive.max_distance()

        print(
            f"  {name}: inside={inside_count}, outside={outside_count}, max_dist={max_dist:.2f}"
        )
        print(f"  Saved: {output_file}")
        print(f"  Slice: {output_file.replace('.png', '_slice.png')}")

    print(f"\nAll visualizations saved in '{output_dir}' directory!")


def generate_dataset_samples(output_dir="visualize_output", num_samples=5):
    """データセットサンプルの可視化を生成"""
    from src.fdslxsdf4seg.generate_sdf_dataset import SDFSegmentationDataset

    # 出力ディレクトリを作成
    samples_dir = os.path.join(output_dir, "dataset_samples")
    os.makedirs(samples_dir, exist_ok=True)

    # グリッドサイズとデバイス設定
    grid_size = [32, 32, 32]
    device = torch.device("cpu")

    # データセットを作成
    ds = SDFSegmentationDataset(
        grid_size=grid_size,
        num_volumes=num_samples,
        min_objects=2,
        max_objects=4,
        device=device,
    )

    print(f"\nGenerating {num_samples} dataset samples in '{samples_dir}'...")

    for i in range(num_samples):
        print(f"Processing sample {i + 1}/{num_samples}...")

        x, y = ds[i]

        # 可視化を実行
        output_file = os.path.join(samples_dir, f"dataset_sample_{i:03d}.png")
        visualize_sample((x, y), output_file)

        # オブジェクト情報を表示
        unique_objects = np.unique(y)
        print(f"  Sample {i}: objects={unique_objects}")
        print(f"  Saved: {output_file}")
        print(f"  Slice: {output_file.replace('.png', '_slice.png')}")

    print(f"\nAll dataset samples saved in '{samples_dir}' directory!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate primitive and dataset visualizations"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualize_output",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--primitives",
        action="store_true",
        help="Generate individual primitive visualizations",
    )
    parser.add_argument(
        "--dataset", action="store_true", help="Generate dataset sample visualizations"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of dataset samples to generate",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate both primitives and dataset samples",
    )

    args = parser.parse_args()

    if args.all:
        args.primitives = True
        args.dataset = True

    if not (args.primitives or args.dataset):
        print("Please specify --primitives, --dataset, or --all")
        parser.print_help()
        exit(1)

    if args.primitives:
        generate_primitive_visualizations(args.output_dir)

    if args.dataset:
        generate_dataset_samples(args.output_dir, args.num_samples)
