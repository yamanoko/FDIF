#!/usr/bin/env python3
"""
各プリミティブの可視化を生成・保存するスクリプト
リファクタリング後の新しい実装に対応
"""

import os

import numpy as np
import torch

# 基本的なプリミティブ
from src.fdslxsdf4seg.basic_sdf import (
    ConcaveCylinder,
    Cone,
    ConeCylinder,
    ConvexCylinder,
    Cylinder,
    Octahedron,
    Sphere,
    Torus,
)
from src.fdslxsdf4seg.generate_sdf_dataset import (
    visualize_sample,
)

# Revolution系
from src.fdslxsdf4seg.revolution.star_revolution import (
    FiveStarRevolution,
    FourStarRevolution,
    ThreeStarRevolution,
)

# SDF Object ベースクラス
from src.fdslxsdf4seg.sdf_object import (
    SectorPolygonPrism,
)

# 凹セクターポリゴンプリズム
from src.fdslxsdf4seg.sector_polygon_prism.concave_sector_polygon_prism import (
    SquareConcavePrism,
    TriangleConcavePrism,
)

# コーンセクターポリゴンプリズム
from src.fdslxsdf4seg.sector_polygon_prism.cone_sector_polygon_prism import (
    HexagonConePrism,
    PentagonConePrism,
    SquareConePrism,
    TriangleConePrism,
)

# 凸セクターポリゴンプリズム
from src.fdslxsdf4seg.sector_polygon_prism.convex_sector_polygon_prism import (
    SquareConvexPrism,
    TriangleConvexPrism,
)

# ピラミッドセクターポリゴンプリズム
from src.fdslxsdf4seg.sector_polygon_prism.pyramid_sector_polygon_prism import (
    HexagonPyramidPrism,
    PentagonPyramidPrism,
    Pyramid,
    TrianglePyramidPrism,
)

# セクターポリゴンプリズム
from src.fdslxsdf4seg.sector_polygon_prism.sector_polygon_prism import (
    HeptagonPrism,
    HexagonPrism,
    NonagonPrism,
    OctagonPrism,
    PentagonPrism,
    SquarePrism,
    TrianglePrism,
)

# スタープリズム
from src.fdslxsdf4seg.star_polygon_prism.star_prism import (
    FiveStarPrism,
    SixStarPrism,
)

# 凸スタープリズム
# 凹スタープリズム
# コーンスタープリズム
# ピラミッドスタープリズム
# トーラス系
from src.fdslxsdf4seg.torus.sector_polygon_torus import (
    HeptagonTorus,
    HexagonTorus,
    NonagonTorus,
    OctagonTorus,
    PentagonTorus,
    SquareTorus,
)
from src.fdslxsdf4seg.torus.star_torus import (
    EightStarTorus,
    FiveStarTorus,
    SevenStarTorus,
    SixStarTorus,
)


def generate_primitive_visualizations(
    output_dir="visualize_output", primitive_type="all"
):
    """各プリミティブを個別に生成し、可視化結果を保存する

    Args:
        output_dir: 出力ディレクトリ
        primitive_type: 生成するプリミティブのタイプ ("all", "star", "basic", "polygon")
    """

    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)

    # グリッドサイズとデバイス設定
    grid_size = [64, 64, 64]  # より細かなグリッドで詳細な可視化
    device = torch.device("cpu")

    # 座標メッシュを作成
    zs = torch.linspace(0, grid_size[0] - 1, grid_size[0], dtype=torch.float32)
    ys = torch.linspace(0, grid_size[1] - 1, grid_size[1], dtype=torch.float32)
    xs = torch.linspace(0, grid_size[2] - 1, grid_size[2], dtype=torch.float32)
    Z, Y, X = torch.meshgrid(zs, ys, xs, indexing="ij")

    # 基本プリミティブ
    basic_primitives = [
        ("Sphere", Sphere, {"center": (32.0, 32.0, 32.0), "radius": 16.0}),
        (
            "Torus",
            Torus,
            {"center": (32.0, 32.0, 32.0), "major_r": 16.0, "minor_r": 6.0},
        ),
        (
            "Cone",
            Cone,
            {"center": (32.0, 32.0, 32.0), "radius": 12.0, "height": 32.0},
        ),
        (
            "Octahedron",
            Octahedron,
            {"center": (32.0, 32.0, 32.0), "size": 15.0},
        ),
        (
            "Cylinder",
            Cylinder,
            {"center": (32.0, 32.0, 32.0), "radius": 12.0, "height": 32.0},
        ),
        (
            "ConvexCylinder",
            ConvexCylinder,
            {
                "center": (32.0, 32.0, 32.0),
                "radius": 12.0,
                "height": 32.0,
                "second_scale": 1.5,
                "neck": 5.0,
            },
        ),
        (
            "ConcaveCylinder",
            ConcaveCylinder,
            {
                "center": (32.0, 32.0, 32.0),
                "radius": 12.0,
                "height": 32.0,
                "second_scale": 0.5,
                "neck": 5.0,
            },
        ),
        (
            "ConeCylinder",
            ConeCylinder,
            {
                "center": (32.0, 32.0, 32.0),
                "radius": 12.0,
                "height": 32.0,
                "second_scale": 0.5,
            },
        ),
    ]

    # セクターポリゴンプリズム
    sector_polygon_primitives = [
        (
            "SectorPolygonPrism",
            SectorPolygonPrism,
            {
                "center": (32.0, 32.0, 32.0),
                "n": 8,
                "r1": 10.0,
                "r2": 12.0,
                "height": 32.0,
                "seed": 42,
            },
        ),
        (
            "TrianglePrism",
            TrianglePrism,
            {
                "center": (32.0, 32.0, 32.0),
                "r1": 10.0,
                "r2": 12.0,
                "height": 32.0,
                "seed": 42,
            },
        ),
        (
            "SquarePrism",
            SquarePrism,
            {
                "center": (32.0, 32.0, 32.0),
                "r1": 10.0,
                "r2": 12.0,
                "height": 32.0,
                "seed": 42,
            },
        ),
        (
            "PentagonPrism",
            PentagonPrism,
            {
                "center": (32.0, 32.0, 32.0),
                "r1": 10.0,
                "r2": 12.0,
                "height": 32.0,
                "seed": 42,
            },
        ),
        (
            "HexagonPrism",
            HexagonPrism,
            {
                "center": (32.0, 32.0, 32.0),
                "r1": 10.0,
                "r2": 12.0,
                "height": 32.0,
                "seed": 42,
            },
        ),
        (
            "HeptagonPrism",
            HeptagonPrism,
            {
                "center": (32.0, 32.0, 32.0),
                "r1": 10.0,
                "r2": 12.0,
                "height": 32.0,
                "seed": 42,
            },
        ),
        (
            "OctagonPrism",
            OctagonPrism,
            {
                "center": (32.0, 32.0, 32.0),
                "r1": 10.0,
                "r2": 12.0,
                "height": 32.0,
                "seed": 42,
            },
        ),
        (
            "NonagonPrism",
            NonagonPrism,
            {
                "center": (32.0, 32.0, 32.0),
                "r1": 10.0,
                "r2": 12.0,
                "height": 32.0,
                "seed": 42,
            },
        ),
    ]

    # ピラミッドプリズム
    pyramid_primitives = [
        (
            "Pyramid",
            Pyramid,
            {
                "center": (32.0, 32.0, 32.0),
                "r1": 8.0,
                "r2": 12.0,
                "height": 25.0,
                "seed": 42,
            },
        ),
        (
            "TrianglePyramidPrism",
            TrianglePyramidPrism,
            {
                "center": (32.0, 32.0, 32.0),
                "r1": 8.0,
                "r2": 12.0,
                "height": 25.0,
                "seed": 42,
            },
        ),
        (
            "PentagonPyramidPrism",
            PentagonPyramidPrism,
            {
                "center": (32.0, 32.0, 32.0),
                "r1": 8.0,
                "r2": 12.0,
                "height": 25.0,
                "seed": 42,
            },
        ),
        (
            "HexagonPyramidPrism",
            HexagonPyramidPrism,
            {
                "center": (32.0, 32.0, 32.0),
                "r1": 8.0,
                "r2": 12.0,
                "height": 25.0,
                "seed": 42,
            },
        ),
    ]

    # コーンプリズム
    cone_primitives = [
        (
            "TriangleConePrism",
            TriangleConePrism,
            {
                "center": (32.0, 32.0, 32.0),
                "r1": 8.0,
                "r2": 12.0,
                "height": 32.0,
                "second_scale": 0.3,
                "seed": 42,
            },
        ),
        (
            "SquareConePrism",
            SquareConePrism,
            {
                "center": (32.0, 32.0, 32.0),
                "r1": 8.0,
                "r2": 12.0,
                "height": 32.0,
                "second_scale": 0.3,
                "seed": 42,
            },
        ),
        (
            "PentagonConePrism",
            PentagonConePrism,
            {
                "center": (32.0, 32.0, 32.0),
                "r1": 8.0,
                "r2": 12.0,
                "height": 32.0,
                "second_scale": 0.3,
                "seed": 42,
            },
        ),
        (
            "HexagonConePrism",
            HexagonConePrism,
            {
                "center": (32.0, 32.0, 32.0),
                "r1": 8.0,
                "r2": 12.0,
                "height": 32.0,
                "second_scale": 0.3,
                "seed": 42,
            },
        ),
    ]

    # 凸プリズム
    convex_primitives = [
        (
            "TriangleConvexPrism",
            TriangleConvexPrism,
            {
                "center": (32.0, 32.0, 32.0),
                "r1": 8.0,
                "r2": 12.0,
                "height": 32.0,
                "second_scale": 1.5,
                "neck": 0.0,
                "seed": 42,
            },
        ),
        (
            "SquareConvexPrism",
            SquareConvexPrism,
            {
                "center": (32.0, 32.0, 32.0),
                "r1": 8.0,
                "r2": 12.0,
                "height": 32.0,
                "second_scale": 1.5,
                "neck": 0.0,
                "seed": 42,
            },
        ),
    ]

    # 凹プリズム
    concave_primitives = [
        (
            "TriangleConcavePrism",
            TriangleConcavePrism,
            {
                "center": (32.0, 32.0, 32.0),
                "r1": 8.0,
                "r2": 12.0,
                "height": 32.0,
                "second_scale": 0.5,
                "neck": 0.0,
                "seed": 42,
            },
        ),
        (
            "SquareConcavePrism",
            SquareConcavePrism,
            {
                "center": (32.0, 32.0, 32.0),
                "r1": 8.0,
                "r2": 12.0,
                "height": 32.0,
                "second_scale": 0.5,
                "neck": 0.0,
                "seed": 42,
            },
        ),
    ]

    # スタープリズム
    star_primitives = [
        (
            "FiveStarPrism",
            FiveStarPrism,
            {
                "center": (32.0, 32.0, 32.0),
                "radius": 12.0,
                "w": 0.5,
                "height": 32.0,
                "seed": 42,
            },
        ),
        (
            "SixStarPrism",
            SixStarPrism,
            {
                "center": (32.0, 32.0, 32.0),
                "radius": 12.0,
                "w": 0.5,
                "height": 32.0,
                "seed": 42,
            },
        ),
    ]

    # トーラスプリミティブ
    torus_primitives = [
        # セクターポリゴントーラス
        (
            "SquareTorus",
            SquareTorus,
            {
                "center": (32.0, 32.0, 32.0),
                "major_r": 20.0,
                "minor_r": 8.0,
            },
        ),
        (
            "PentagonTorus",
            PentagonTorus,
            {
                "center": (32.0, 32.0, 32.0),
                "major_r": 20.0,
                "minor_r": 8.0,
            },
        ),
        (
            "HexagonTorus",
            HexagonTorus,
            {
                "center": (32.0, 32.0, 32.0),
                "major_r": 20.0,
                "minor_r": 8.0,
            },
        ),
        (
            "HeptagonTorus",
            HeptagonTorus,
            {
                "center": (32.0, 32.0, 32.0),
                "major_r": 20.0,
                "minor_r": 8.0,
            },
        ),
        (
            "OctagonTorus",
            OctagonTorus,
            {
                "center": (32.0, 32.0, 32.0),
                "major_r": 20.0,
                "minor_r": 8.0,
            },
        ),
        (
            "NonagonTorus",
            NonagonTorus,
            {
                "center": (32.0, 32.0, 32.0),
                "major_r": 20.0,
                "minor_r": 8.0,
            },
        ),
        # スタートーラス
        (
            "FiveStarTorus",
            FiveStarTorus,
            {
                "center": (32.0, 32.0, 32.0),
                "major_r": 20.0,
                "minor_r": 8.0,
                "w": 0.5,
            },
        ),
        (
            "SixStarTorus",
            SixStarTorus,
            {
                "center": (32.0, 32.0, 32.0),
                "major_r": 20.0,
                "minor_r": 8.0,
                "w": 0.5,
            },
        ),
        (
            "SevenStarTorus",
            SevenStarTorus,
            {
                "center": (32.0, 32.0, 32.0),
                "major_r": 20.0,
                "minor_r": 8.0,
                "w": 0.5,
            },
        ),
        (
            "EightStarTorus",
            EightStarTorus,
            {
                "center": (32.0, 32.0, 32.0),
                "major_r": 20.0,
                "minor_r": 8.0,
                "w": 0.5,
            },
        ),
    ]

    # Revolution（回転体）プリミティブ
    revolution_primitives = [
        (
            "ThreeStarRevolution",
            ThreeStarRevolution,
            {
                "center": (32.0, 32.0, 32.0),
                "radius": 8.0,
                "w": 0.5,
            },
        ),
        (
            "FourStarRevolution",
            FourStarRevolution,
            {
                "center": (32.0, 32.0, 32.0),
                "radius": 8.0,
                "w": 0.5,
            },
        ),
        (
            "FiveStarRevolution",
            FiveStarRevolution,
            {
                "center": (32.0, 32.0, 32.0),
                "radius": 8.0,
                "w": 0.5,
            },
        ),
    ]

    # プリミティブタイプに応じて選択
    if primitive_type == "star":
        selected_primitives = star_primitives
        print(
            f"Generating {len(selected_primitives)} star primitive visualizations in '{output_dir}'..."
        )
    elif primitive_type == "basic":
        selected_primitives = basic_primitives
        print(
            f"Generating {len(selected_primitives)} basic primitive visualizations in '{output_dir}'..."
        )
    elif primitive_type == "polygon":
        selected_primitives = (
            sector_polygon_primitives
            + pyramid_primitives
            + cone_primitives
            + convex_primitives
            + concave_primitives
        )
        print(
            f"Generating {len(selected_primitives)} polygon primitive visualizations in '{output_dir}'..."
        )
    elif primitive_type == "torus":
        selected_primitives = torus_primitives
        print(
            f"Generating {len(selected_primitives)} torus primitive visualizations in '{output_dir}'..."
        )
    elif primitive_type == "revolution":
        selected_primitives = revolution_primitives
        print(
            f"Generating {len(selected_primitives)} revolution primitive visualizations in '{output_dir}'..."
        )
    else:  # "all" or default
        selected_primitives = (
            basic_primitives
            + sector_polygon_primitives
            + pyramid_primitives
            + cone_primitives
            + convex_primitives
            + concave_primitives
            + star_primitives
            + torus_primitives
            + revolution_primitives
        )
        print(
            f"Generating {len(selected_primitives)} primitive visualizations in '{output_dir}'..."
        )

    for name, PrimClass, params in selected_primitives:
        print(f"Processing {name}...")

        try:
            # ランダムシードを固定（変換を無効化）
            torch.manual_seed(20)
            np.random.seed(20)
            import random

            random.seed(20)

            # プリミティブを生成
            primitive = PrimClass(grid_size, device, **params)

            # SDFを計算
            sdf = primitive.sdf(X, Y, Z)

            # SDFを可視化用に変換（0-128の範囲）
            sdf_vis = 128.0 / (torch.pow(torch.abs(sdf), 2.0) + 1.0)
            sdf_vis = torch.clamp(sdf_vis, 0.0, 128.0).to(torch.uint8)

            # セグメンテーションマスクを作成（オブジェクトIDは1）
            mask = (sdf <= 0).to(torch.uint8)

            # NumPy配列に変換
            sdf_np = sdf_vis.cpu().numpy()
            mask_np = mask.cpu().numpy()

            # 可視化を実行
            output_file = os.path.join(output_dir, f"{name.lower()}_visualization.png")
            visualize_sample((sdf_np, mask_np), output_file)

            # 統計情報を表示
            inside_count = (sdf < 0).sum().item()
            outside_count = (sdf > 0).sum().item()

            print(f"  {name}: inside={inside_count}, outside={outside_count}")
            print(f"  Saved: {output_file}")
            print(f"  Slice: {output_file.replace('.png', '_slice.png')}")

        except Exception as e:
            print(f"  Error processing {name}: {e}")
            continue

    print(f"\nAll visualizations saved in '{output_dir}' directory!")


def generate_dataset_samples(output_dir="visualize_output", num_samples=5):
    """データセットサンプルの可視化を生成"""
    from src.fdslxsdf4seg.generate_sdf_dataset import SDFSegmentationDataset

    # 出力ディレクトリを作成
    samples_dir = os.path.join(output_dir, "dataset_samples")
    os.makedirs(samples_dir, exist_ok=True)

    # グリッドサイズとデバイス設定
    grid_size = [96, 96, 96]
    device = torch.device("cpu")

    # データセットを作成
    ds = SDFSegmentationDataset(
        grid_size=grid_size,
        num_volumes=num_samples,
        min_objects=20,
        max_objects=20,
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
        "--primitive_type",
        type=str,
        choices=["all", "star", "basic", "polygon", "torus", "revolution"],
        default="all",
        help="Type of primitives to generate (all/star/basic/polygon/torus/revolution)",
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
        generate_primitive_visualizations(args.output_dir, args.primitive_type)

    if args.dataset:
        generate_dataset_samples(args.output_dir, args.num_samples)
