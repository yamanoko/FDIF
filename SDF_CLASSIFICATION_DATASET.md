# SDF Classification Dataset for SSL3D_classification

## 概要

FDSLxSDF4Segの SDF プリミティブシステムを使用して、[SSL3D_classification](https://github.com/constantinulrich/SSL3D_classification) の学習パイプラインに直接読み込み可能な分類データセットを生成するためのドキュメント。

各ボリュームには**1つのプリミティブ**のみが含まれ、そのプリミティブの種類（×マッパー×Displacement の組み合わせ）がクラスIDとなる。

---

## データ構造

```
<data_root_dir>/
├── nnUNetResEncUNetLPlans_3d_fullres/
│   ├── case_00000.b2nd      # Blosc2圧縮, shape: (1, D, H, W), float32
│   ├── case_00001.b2nd
│   ├── case_00002.b2nd
│   └── ...
├── labelsTr.json             # ラベル辞書
├── splits_final.json         # KFoldクロスバリデーション分割
├── <dataset_name>.yaml       # SSL3D_classification用Hydra設定ファイル
└── generation_log.txt        # 生成パラメータ・クラスマッピングログ
```

### ファイル形式の詳細

#### Blosc2ファイル (.b2nd)

| 項目 | 値 |
|---|---|
| 形状 | `(1, D, H, W)` — チャネル次元1を含む4D配列 |
| データ型 | `float32` |
| 圧縮 | ZSTD, clevel=8 |
| 正規化 | Z-Score正規化済み `(x - mean) / max(std, 1e-8)` |

SSL3D_classificationの `Blosc2IO.load()` でそのまま読み込み可能。

#### labelsTr.json

```json
{
  "case_00000": 3,
  "case_00001": 0,
  "case_00002": 1,
  ...
}
```

- **キー**: サンプルID文字列（Blosc2ファイル名から拡張子を除いたもの）
- **値**: 整数クラスID（0-indexed）

SSL3D_classificationの `AbideData` (nnUNet式) と互換。ただしラベル値は直接整数であり、`[key, value]` リスト形式ではない点に注意（`abide_1mm_cropped_160_new` 形式と同じ）。

#### splits_final.json

```json
[
  {
    "train": ["case_00000", "case_00002", "case_00005", ...],
    "val": ["case_00001", "case_00003", ...]
  },
  {
    "train": ["case_00001", "case_00003", ...],
    "val": ["case_00000", "case_00002", ...]
  },
  ...
]
```

- TrainデータとValデータは別々に生成され、全foldで同一のtrain/val分割を使用
- デフォルト: 5 folds
- アクセス: `splits[fold]["train"]` / `splits[fold]["val"]`

---

## クラス決定ロジック

クラスIDはセグメンテーションタスクと同じ仕組みで決定される。プリミティブ、マッパー、Displacement関数の組み合わせに応じてクラス数が変化する。

### 基本パターン

| オプション | クラス構成 | クラス数の計算 |
|---|---|---|
| `--primitives sphere cylinder` | Sphere+mapper, Cylinder+mapper | primitives × mappers |
| `--sdf_mappers inverse_cube linear` | 各プリミティブ×各マッパー | primitives × mappers |
| `--displacement_functions sine perlin` | 上記 + 各プリミティブ×各Displacement×各マッパー | P×M + P×D×M |
| `--mapper_as_augmentation` | マッパーはクラスに寄与しない | primitives (+ P×D if disp) |
| `--displacement_as_augmentation` | Displacementはクラスに寄与しない | P×M |

### クラスIDの割り当て順序

1. **ハイブリッドプリミティブ** (primitive × mapper): ID 0, 1, 2, ...
2. **ハイブリッドDisplacedプリミティブ** (primitive × displacement × mapper): 続きのID
3. **CombinedUnionプリミティブ** (union × mapper): 続きのID

### 例

```bash
--primitives sphere cylinder --sdf_mappers inverse_cube linear
```

| Class ID | 名前 |
|---|---|
| 0 | Sphere + inverse_cube |
| 1 | Sphere + linear |
| 2 | Cylinder + inverse_cube |
| 3 | Cylinder + linear |

→ 合計4クラス

```bash
--primitives sphere cylinder --sdf_mappers inverse_cube --displacement_functions perlin
```

| Class ID | 名前 |
|---|---|
| 0 | Sphere + inverse_cube |
| 1 | Cylinder + inverse_cube |
| 2 | Sphere + displacement(perlin) + inverse_cube |
| 3 | Cylinder + displacement(perlin) + inverse_cube |

→ 合計4クラス (2×1 + 2×1×1)

---

## grid_scale パラメータ

セグメンテーション用データセットでは、プリミティブは複数配置するために相対的に小さいサイズで生成される。分類タスクでは1つのプリミティブだけを配置するため、ボクセル空間内でプリミティブを拡大する必要がある。

`grid_scale` はメッシュグリッドの座標範囲を制御する:

- **セグメンテーション版**: `linspace(-D/2, D/2-1, D)` → 座標範囲 = `[-32, 31]` (D=64時)
- **分類版**: `linspace(-D/2 * scale, D/2 * scale, D)` → 座標範囲 = `[-14.4, 14.4]` (D=64, scale=0.45時)

座標範囲が小さくなることで、同じパラメータのプリミティブがボクセル空間内で相対的に大きく見える。

| grid_scale | 効果 |
|---|---|
| 1.0 | セグメンテーション版と同じサイズ（小さい） |
| 0.5 | プリミティブが2倍に拡大 |
| **0.45** (デフォルト) | プリミティブが約2.22倍に拡大 |
| 0.35 | プリミティブが約2.86倍に拡大 |
| 0.25 | プリミティブが4倍に拡大 |

---

## Z-Score正規化

SSL3D_classificationの前処理パイプラインでは、Z-Score正規化はリサンプリング前に適用され、学習時のAugmentationには含まれない。本データセットでは前処理済みデータとして保存するため、生成時にZ-Score正規化を適用している:

```python
mean = x.mean()
std = x.std()
x_normalized = (x - mean) / max(std, 1e-8)
```

---

## プリミティブの配置

- **中央配置**: `center=[0, 0, 0]` を渡して平行移動を無効化
- **回転・せん断**: `transform=True` (デフォルト) で適用。回転角は `[-π, π]`、せん断量は `[-0.1, 0.1]`
- **平行移動なし**: プリミティブは常にボリュームの中心に配置される

---

## SSL3D_classification側のDataLoader実装ガイド

### 必要なファイル

1. `datasets/sdf_classification.py` — Dataset + DataModule
2. `cli_configs/data/<dataset_name>.yaml` — 生成された YAML を配置

### Dataset クラスの実装例

```python
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from utils.io import Blosc2IO  # SSL3D_classificationのBlosc2IOユーティリティ


class SDFClassificationData(Dataset):
    def __init__(self, root, split, fold, transform=None):
        super().__init__()
        self.img_dir = Path(root) / "nnUNetResEncUNetLPlans_3d_fullres"
        label_file = Path(root) / "labelsTr.json"
        split_file = Path(root) / "splits_final.json"

        with open(split_file) as f:
            self.img_files = json.load(f)[fold][
                "train" if split == "train" else "val"
            ]

        with open(label_file) as f:
            labels = json.load(f)
        # ラベルは直接整数値
        self.labels = [labels[i] for i in self.img_files]

        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img, _ = Blosc2IO.load(
            self.img_dir / (self.img_files[idx] + ".b2nd"), mode="r"
        )
        if self.transform:
            img = self.transform(
                **{"image": torch.from_numpy(img[...])}
            )["image"]
        else:
            img = torch.from_numpy(img[...])
        return img, self.labels[idx]
```

### DataModule クラスの実装例

```python
from datasets.base import BaseDataModule


class SDFClassificationDataModule(BaseDataModule):
    def __init__(self, **params):
        super().__init__(**params)

    def setup(self, stage: str):
        self.train_dataset = SDFClassificationData(
            self.data_path, split="train",
            transform=self.train_transforms, fold=self.fold
        )
        self.val_dataset = SDFClassificationData(
            self.data_path, split="val",
            transform=self.test_transforms, fold=self.fold
        )
```

### `__getitem__` の返り値

| 項目 | 値 |
|---|---|
| `img` | `torch.Tensor`, shape `(1, D, H, W)`, dtype `float32` |
| `label` | `int` (0-indexed クラスID) |

---

## コマンド例

### 基本的な生成

```bash
uv run python src/fdslxsdf4seg/generate_sdf_dataset_classification.py \
    --out_dir outputs/cls_basic \
    --D 96 --H 96 --W 96 \
    --samples_per_class 50 \
    --primitives sphere cylinder torus cone \
    --num_visualize 5
```

→ 4プリミティブ × 1マッパー = 4クラス × 50 = 200サンプル

### Displacement + 複数マッパー

```bash
uv run python src/fdslxsdf4seg/generate_sdf_dataset_classification.py \
    --out_dir outputs/cls_disp \
    --D 96 --H 96 --W 96 \
    --samples_per_class 100 \
    --primitives sphere cylinder \
    --sdf_mappers inverse_cube linear \
    --displacement_functions perlin turbulence \
    --grid_scale 0.45 \
    --num_visualize 10
```

→ (2×2) + (2×2×2) = 4 + 8 = 12クラス × 100 = 1200サンプル

### Augmentationモード

```bash
uv run python src/fdslxsdf4seg/generate_sdf_dataset_classification.py \
    --out_dir outputs/cls_aug \
    --D 64 --H 64 --W 64 \
    --samples_per_class 50 \
    --primitives sphere cylinder torus \
    --sdf_mappers inverse_cube linear exponential \
    --mapper_as_augmentation \
    --displacement_functions perlin sine \
    --displacement_as_augmentation \
    --num_visualize 5
```

→ 3クラス × 50 = 150サンプル（マッパーとDisplacementはランダム選択されるがクラスに影響しない）
