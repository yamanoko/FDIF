# generate_sdf_dataset.py

## 概要

`generate_sdf_dataset.py`は、3DセグメンテーションタスクのためのSDF（Signed Distance Function）ベースの合成データセットを生成するPythonスクリプトです。複数の3Dプリミティブ（球、箱、円柱、トーラス）を含む3Dボリュームデータとそのセグメンテーションマスクを自動生成し、医療画像解析やコンピュータビジョンの研究に活用できます。

**2つの出力形式をサポート**:
- **MONAI Decathlon形式** (デフォルト): MONAIフレームワーク用
- **nnUNet形式**: nnUNetフレームワーク用（`--nnunet_format`フラグで有効化）

## 主要機能

### 1. 3Dプリミティブの生成
- **球（Sphere）**: ランダムな半径の球体
- **箱（Box）**: ランダムなサイズの直方体
- **円柱（Cylinder）**: ランダムな半径と高さの円柱
- **トーラス（Torus）**: ランダムな大半径・小半径のトーラス

### 2. プリミティブ選択機能
- **全プリミティブ使用**: デフォルトで4種類すべてのプリミティブを使用
- **選択的使用**: 特定のプリミティブのみを選択して使用可能
- **複数インスタンス**: 同じプリミティブの複数インスタンスを生成可能
- **動的クラス数**: 選択したプリミティブ数に応じてセグメンテーションクラス数が変動

### 3. 変換機能
- **平行移動**: オブジェクトのランダムな位置配置
- **回転**: X、Y、Z軸周りのランダム回転
- **せん断**: ランダムなせん断変形

### 4. データセット生成
- 複数のオブジェクトを含む3Dボリュームの生成
- SDF値に基づく強度画像の生成
- オブジェクトIDベースのセグメンテーションマスクの生成
- NIfTI形式での保存

### 5. 可視化機能
- Plotlyを使用した3D可視化
- スライス画像の生成
- インタラクティブなHTML出力

## 使用法

### 基本的な使用方法

```bash
python generate_sdf_dataset.py --out_dir ./output --num_samples 100
```

### パラメータ詳細

| パラメータ | デフォルト値 | 説明 |
|-----------|-------------|------|
| `--out_dir` | 自動生成 | 出力ディレクトリのパス |
| `--D` | 64 | ボリュームの深度（Z軸） |
| `--H` | 64 | ボリュームの高さ（Y軸） |
| `--W` | 64 | ボリュームの幅（X軸） |
| `--num_samples` | 200 | 生成するトレーニングサンプル数 |
| `--num_val_samples` | 0 | 生成する検証サンプル数 |
| `--min_objects` | 2 | 1サンプルあたりの最小オブジェクト数 |
| `--max_objects` | 5 | 1サンプルあたりの最大オブジェクト数 |
| `--primitives` | 全プリミティブ | 使用するプリミティブ（`sphere`, `box`, `cylinder`, `torus`） |
| `--seed` | None | 乱数シード（再現性のため） |
| `--num_visualize` | 0 | 可視化するサンプル数 |
| `--nnunet_format` | False | nnUNet形式でデータセットを生成 |
| `--dataset_id` | 999 | nnUNet形式のデータセットID（例: 999で`Dataset999_Name`） |
| `--dataset_name` | "SDFSynthetic" | nnUNet形式のデータセット名 |

### 使用例

#### 1. 基本的なデータセット生成（MONAI Decathlon形式）
```bash
uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --out_dir ./synthetic_dataset \
    --D 96 --H 96 --W 96 \
    --num_samples 500 \
    --num_val_samples 100 \
    --min_objects 2 \
    --max_objects 4 \
    --seed 42
```

#### 2. nnUNet形式でデータセット生成
```bash
uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --nnunet_format \
    --dataset_id 999 \
    --dataset_name SDFSynthetic \
    --D 96 --H 96 --W 96 \
    --num_samples 500 \
    --num_val_samples 100 \
    --min_objects 2 \
    --max_objects 4 \
    --seed 42
```

**nnUNet形式の特徴**:
- トレーニングデータは `Dataset999_SDFSynthetic/imagesTr/` と `labelsTr/` に保存
- 検証データは別データセット `Dataset1000_SDFSynthetic_Val/imagesTr/` と `labelsTr/` として保存
- 画像ファイル名: `case_00000_0000.nii.gz` (nnUNet命名規則に準拠)
- ラベルファイル名: `case_00000.nii.gz`
- `dataset.json` にメタデータ（チャネル名、ラベル、サンプル数）を含む

#### 3. 可視化付きデータセット生成
```bash
uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --out_dir ./dataset_with_viz \
    --num_samples 100 \
    --num_visualize 10 \
    --seed 42
```

#### 4. 特定のプリミティブのみを使用
```bash
# 球体とボックスのみを使用
uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --out_dir ./sphere_box_dataset \
    --primitives sphere box \
    --num_samples 300 \
    --max_objects 6

# 単一プリミティブ（円柱のみ）
uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --out_dir ./cylinder_only_dataset \
    --primitives cylinder \
    --num_samples 200 \
    --max_objects 8
```

#### 5. 段階的学習用データセット
```bash
# レベル1: 単一プリミティブ
uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --out_dir ./level1_simple \
    --primitives sphere \
    --max_objects 3 \
    --num_samples 200

# レベル2: 複数プリミティブ
uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --out_dir ./level2_medium \
    --primitives sphere box \
    --max_objects 5 \
    --num_samples 300

# レベル3: 全プリミティブ
uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --out_dir ./level3_complex \
    --primitives sphere box cylinder torus \
    --max_objects 8 \
    --num_samples 500
```

## 出力構造

データセット生成後、以下の構造でファイルが出力されます：

### MONAI Decathlon形式（デフォルト）

```
output_directory/
├── generation_log.txt              # 生成ログ
├── data/
│   ├── data.json                   # データセットメタデータ
│   ├── image/                      # SDF強度画像
│   │   ├── batch_0000/
│   │   │   ├── sample_00000_x.nii.gz
│   │   │   └── ...
│   │   └── ...
│   └── label/                      # セグメンテーションマスク
│       ├── batch_0000/
│       │   ├── sample_00000_y.nii.gz
│       │   └── ...
│       └── ...
└── visualizations/                 # 可視化結果（--num_visualize > 0の場合）
    ├── visualization_00000.png
    ├── visualization_00000_slice.png
    └── ...
```

### nnUNet形式（--nnunet_formatを指定した場合）

```
Dataset999_SDFSynthetic/            # トレーニングデータセット
├── dataset.json                     # nnUNetメタデータ
├── imagesTr/                        # トレーニング画像
│   ├── case_00000_0000.nii.gz
│   ├── case_00001_0000.nii.gz
│   └── ...
├── labelsTr/                        # トレーニングラベル
│   ├── case_00000.nii.gz
│   ├── case_00001.nii.gz
│   └── ...
├── generation_log.txt               # 生成ログ
└── visualizations/                  # 可視化結果（--num_visualize > 0の場合）
    └── ...

Dataset1000_SDFSynthetic_Val/        # 検証データセット（--num_val_samples > 0の場合）
├── dataset.json                     # nnUNetメタデータ
├── imagesTr/                        # 検証画像（nnUNet形式ではimageTrとして保存）
│   ├── case_00500_0000.nii.gz
│   ├── case_00501_0000.nii.gz
│   └── ...
└── labelsTr/                        # 検証ラベル
    ├── case_00500.nii.gz
    ├── case_00501.nii.gz
    └── ...
```

## データフォーマット

### MONAI Decathlon形式: data.json
```json
{
    "training": [
        {
            "image": "絶対パス/sample_00000_x.nii.gz",
            "label": "絶対パス/sample_00000_y.nii.gz", 
            "id": "sample_00000"
        }
    ],
    "validation": [...]
}
```

### nnUNet形式: dataset.json
```json
{
    "channel_names": {
        "0": "SDF"
    },
    "labels": {
        "background": 0,
        "Sphere_inverse_cube": 1,
        "Cylinder_inverse_cube": 2,
        "Box_inverse_cube": 3,
        "Torus_inverse_cube": 4
    },
    "numTraining": 500,
    "file_ending": ".nii.gz"
}
```

**nnUNet形式の特徴**:
- `channel_names`: 入力チャネル（この場合はSDFボリュームの1チャネル）
- `labels`: 各セグメンテーションクラスの名前とID（ハイブリッドプリミティブ名を使用）
- `numTraining`: トレーニングサンプル数
- `file_ending`: ファイル拡張子（`.nii.gz`）

### 画像データ
- **MONAI形式 - 画像ファイル（*_x.nii.gz）**: SDF値から計算された強度値（0-128の範囲）
- **MONAI形式 - ラベルファイル（*_y.nii.gz）**: オブジェクトID（0=背景、1-N=各プリミティブ、Nは選択したプリミティブ数）
- **nnUNet形式 - 画像ファイル（*_0000.nii.gz）**: SDF値から計算された強度値（0-128の範囲）
- **nnUNet形式 - ラベルファイル（*.nii.gz）**: オブジェクトID（0=背景、1-N=各ハイブリッドプリミティブ）

### プリミティブIDマッピング
選択したプリミティブに応じて、以下のIDが割り当てられます：
- **全プリミティブ使用時**: 1=Sphere, 2=Box, 3=Cylinder, 4=Torus
- **部分選択時**: 選択順序に応じて1から順番に割り当て

例：`--primitives box torus` の場合
- 1=Box, 2=Torus

## 技術的詳細

### SDF計算
各プリミティブのSDF関数：
- **球**: `|p| - radius`
- **箱**: 箱の境界からの距離
- **円柱**: 軸方向と半径方向の距離の組み合わせ
- **トーラス**: 主軸からの距離とチューブ半径の組み合わせ

### 強度値計算
```python
x_vol = 128.0 / (torch.pow(torch.abs(torch.stack(sdfs, dim=0)), 2.0) + 1.0)
x_vol = x_vol.sum(dim=0)
x_vol = torch.clamp(x_vol, 0.0, 128.0).to(torch.uint8)
```

### セグメンテーション優先度
複数のオブジェクトが重複する場合、体積の大きいオブジェクトが優先されます。

## 依存関係

- torch
- nibabel
- numpy
- plotly
- argparse

## 注意事項

1. **GPU使用**: CUDAが利用可能な場合、自動的にGPUを使用します
2. **メモリ使用量**: 大きなグリッドサイズや多数のサンプル生成時はメモリ使用量に注意
3. **ファイルパス**: 出力されるdata.jsonには絶対パスが記録されます
4. **可視化**: 可視化機能を使用する場合、plotlyとkaleido（画像出力用）が必要です

## トラブルシューティング

### よくある問題

1. **メモリ不足**: グリッドサイズやバッチサイズを小さくしてください
2. **可視化エラー**: kaleido パッケージをインストールしてください: `pip install kaleido`
3. **CUDA エラー**: GPU メモリ不足の場合、CPUモードで実行されます

### パフォーマンス最適化

- 大量のデータ生成時は `--num_visualize 0` を設定
- バッチ処理でメモリ使用量を制御
- SSDストレージの使用を推奨
