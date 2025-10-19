# Dataset Visualization Guide

## Overview

`visualize_primitives.py`スクリプトが大幅に強化されました。`generate_sdf_dataset.py`の`SDFSegmentationDataset`の最新機能をサポートし、コマンドライン引数で細かく設定できるようになりました。

## New Features

### 1. ハイブリッドプリミティブ対応
- プリミティブとSDFマッパーの組み合わせをサポート
- デフォルト: `inverse_cube`マッパー

### 2. CombinedObjectUnion プリミティブ
- 複数のプリミティブを組み合わせた新しいプリミティブを自動生成
- `--num_combined_unions`パラメータで生成数を指定

### 3. 柔軟なプリミティブ選択
- 個別プリミティブの指定: `--dataset_primitives`
- カテゴリベースの選択: `--dataset_categories`
- ランダム選択: `--dataset_num_classes`

### 4. SDFマッパー設定
- 複数のマッパーを指定可能: `--sdf_mappers`

### 5. サンプルパラメータ制御
- グリッドサイズ: `--grid_size D H W`
- オブジェクト数: `--min_objects`, `--max_objects`
- 変形の有効/無効: `--no_transform`

### 6. デバイス選択
- CUDA使用: `--cuda` (利用可能な場合)

## Usage Examples

### 基本的な使用方法

```bash
# デフォルト設定でサンプル生成
uv run python visualize_primitives.py --dataset --num_samples 5

# グリッドサイズを指定
uv run python visualize_primitives.py --dataset --num_samples 5 --grid_size 128 128 128

# オブジェクト数を制御
uv run python visualize_primitives.py --dataset --num_samples 5 --min_objects 3 --max_objects 8
```

### プリミティブ選択

```bash
# 特定のプリミティブのみを使用
uv run python visualize_primitives.py --dataset --num_samples 5 \
  --dataset_primitives Sphere Torus Cone

# カテゴリベースで選択
uv run python visualize_primitives.py --dataset --num_samples 5 \
  --dataset_categories basic

# ランダムに5個のクラスを選択
uv run python visualize_primitives.py --dataset --num_samples 5 \
  --dataset_num_classes 5
```

### CombinedObjectUnion

```bash
# 2つのCombinedObjectUnionプリミティブを追加
uv run python visualize_primitives.py --dataset --num_samples 5 \
  --num_combined_unions 2

# basicカテゴリ + CombinedObjectUnion
uv run python visualize_primitives.py --dataset --num_samples 5 \
  --dataset_categories basic \
  --num_combined_unions 1
```

### SDFマッパー設定

```bash
# 特定のマッパーを指定
uv run python visualize_primitives.py --dataset --num_samples 5 \
  --sdf_mappers inverse_cube

# 複数のマッパー（複数指定可能）
uv run python visualize_primitives.py --dataset --num_samples 5 \
  --sdf_mappers inverse_cube exponential_mapper
```

### その他のオプション

```bash
# 変形を無効化
uv run python visualize_primitives.py --dataset --num_samples 5 \
  --no_transform

# CUDAを使用（利用可能な場合）
uv run python visualize_primitives.py --dataset --num_samples 5 \
  --cuda

# カスタム出力ディレクトリ
uv run python visualize_primitives.py --dataset --num_samples 5 \
  --output_dir my_output
```

### 複合的な例

```bash
# Basicカテゴリ + CombinedObjectUnion + カスタムグリッドサイズ
uv run python visualize_primitives.py --dataset --num_samples 10 \
  --dataset_categories basic \
  --num_combined_unions 2 \
  --grid_size 128 128 128 \
  --min_objects 2 \
  --max_objects 5

# 特定プリミティブ + 複数マッパー + 変形無効
uv run python visualize_primitives.py --dataset --num_samples 5 \
  --dataset_primitives Sphere Torus Cone \
  --sdf_mappers inverse_cube \
  --no_transform \
  --min_objects 1 \
  --max_objects 3

# 複数カテゴリ + CombinedObjectUnion + CUDA
uv run python visualize_primitives.py --dataset --num_samples 20 \
  --dataset_categories basic sector_polygon_prism \
  --num_combined_unions 3 \
  --cuda \
  --output_dir advanced_output
```

## Available Primitive Categories

以下のカテゴリから選択可能です（`--dataset_categories`で指定）:

- `basic`: 基本プリミティブ (Sphere, Torus, Cone, Cylinder, etc.)
- `sector_polygon_prism`: セクターポリゴンプリズム
- `convex_sector_polygon_prism`: 凸プリズム
- `concave_sector_polygon_prism`: 凹プリズム
- `cone_sector_polygon_prism`: コーンプリズム
- `star_polygon_prism`: スタープリズム
- `torus_shapes`: トーラス形状
- `revolution`: 回転形状
- `onioned_sector_polygon_prism`: オニオン形状（セクター）
- `onioned_star_polygon_prism`: オニオン形状（スター）
- `union`: ユニオン形状

## Command Line Arguments

```
--dataset                     Generate dataset sample visualizations
--num_samples NUM_SAMPLES     Number of dataset samples to generate (default: 5)
--grid_size D H W             Grid size for samples [D H W] (default: 64 64 64)
--min_objects MIN_OBJECTS     Minimum objects per sample (default: 2)
--max_objects MAX_OBJECTS     Maximum objects per sample (default: 5)
--dataset_primitives PRIMS    Specific primitives to use
--dataset_categories CATS     Primitive categories to use
--dataset_num_classes NUM     Number of classes to randomly select
--num_combined_unions NUM     CombinedObjectUnion primitives count (default: 0)
--sdf_mappers MAPPERS         SDF mapper functions to use
--no_transform                Disable transformations
--cuda                        Use CUDA if available
--output_dir OUTPUT_DIR       Output directory (default: visualize_output)
```

## Output Structure

生成されたサンプルは以下の構造で保存されます：

```
visualize_output/
└── dataset_samples/
    ├── dataset_sample_000.png          # 2D可視化（合成SDF）
    ├── dataset_sample_000_slice.png    # 2Dスライス
    ├── dataset_sample_001.png
    ├── dataset_sample_001_slice.png
    └── ...
```

## Integration with SDFSegmentationDataset

`visualize_primitives.py`の`generate_dataset_samples`関数は、以下の`SDFSegmentationDataset`パラメータをサポートしています：

- `grid_size`: グリッドの3次元サイズ
- `num_volumes`: 生成するボリューム数
- `min_objects`, `max_objects`: 各サンプルのオブジェクト数範囲
- `device`: 計算デバイス
- `primitives`: 使用するプリミティブリスト
- `categories`: 使用するカテゴリリスト
- `num_classes`: ランダムに選択するクラス数
- `transform`: 変形の有効/無効
- `num_combined_unions`: CombinedObjectUnionプリミティブの数
- `sdf_mappers`: SDFマッパーリスト

## Notes

1. **プリミティブ名の大文字小文字**: コマンドラインでプリミティブを指定する際は、大文字で指定可能（内部的に小文字に変換）

2. **デフォルトマッパー**: `--sdf_mappers`を指定しない場合、`inverse_cube`がデフォルトマッパーとして使用されます

3. **パフォーマンス**: CombinedObjectUnionプリミティブの数を増やすと、生成時間が増加します

4. **メモリ使用量**: 大きなグリッドサイズを使用する場合、CUDAメモリが必要になる場合があります

## Example Output Log

```
Generating 5 dataset samples in 'visualize_output\dataset_samples'...
  Grid size: [64, 64, 64]
  Objects per sample: 2-5
  Combined unions: 0
  Transform: True
Selected primitives (125): ... (all primitives)

Dataset created with 125 primitive classes:
  Class 1: Sphere + inverse_cube
  Class 2: Cylinder + inverse_cube
  ... (more classes)

Generating 5 samples...
Processing sample 1/5...
  Sample 0: 2 objects, IDs=[24, 77]
  Saved: visualize_output\dataset_samples\dataset_sample_000.png
  Slice: visualize_output\dataset_samples\dataset_sample_000_slice.png
...

All dataset samples saved in 'visualize_output\dataset_samples' directory!
```
