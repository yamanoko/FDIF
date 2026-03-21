# SDF Classification Dataset for SSL3D_classification

## Overview

Documentation for generating classification datasets that can be directly loaded into the [SSL3D_classification](https://github.com/constantinulrich/SSL3D_classification) training pipeline, using the FDSLxSDF4Seg SDF primitive system.

Each volume contains **a single primitive**, and the class ID is determined by the combination of primitive type, mapper, and displacement function.

---

## Data Structure

```
<data_root_dir>/
├── nnUNetResEncUNetLPlans_3d_fullres/
│   ├── case_00000.b2nd      # Blosc2 compressed, shape: (1, D, H, W), float32
│   ├── case_00001.b2nd
│   ├── case_00002.b2nd
│   └── ...
├── labelsTr.json             # Label dictionary
├── splits_final.json         # K-Fold cross-validation splits
├── <dataset_name>.yaml       # Hydra config file for SSL3D_classification
└── generation_log.txt        # Generation parameters and class mapping log
```

### File Format Details

#### Blosc2 Files (.b2nd)

| Item | Value |
|------|-------|
| Shape | `(1, D, H, W)` — 4D array including 1 channel dimension |
| Data type | `float32` |
| Compression | ZSTD, clevel=8 |
| Normalization | Z-Score normalized: `(x - mean) / max(std, 1e-8)` |

Directly loadable by SSL3D_classification's `Blosc2IO.load()`.

#### labelsTr.json

```json
{
  "case_00000": 3,
  "case_00001": 0,
  "case_00002": 1,
  ...
}
```

- **Key**: Sample ID string (Blosc2 filename without extension)
- **Value**: Integer class ID (0-indexed)

Compatible with SSL3D_classification's `AbideData` (nnUNet-style). Note that label values are directly integers, not `[key, value]` list format (same as the `abide_1mm_cropped_160_new` format).

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

- Training and validation data are generated separately, and all folds use the same train/val split
- Default: 5 folds
- Access: `splits[fold]["train"]` / `splits[fold]["val"]`

---

## Class Determination Logic

Class IDs are determined using the same system as the segmentation task. The number of classes changes depending on the combination of primitives, mappers, and displacement functions.

### Basic Patterns

| Option | Class Composition | Class Count Formula |
|--------|------------------|-------------------|
| `--primitives sphere cylinder` | Sphere+mapper, Cylinder+mapper | primitives x mappers |
| `--sdf_mappers inverse_cube linear` | Each primitive x each mapper | primitives x mappers |
| `--displacement_functions sine perlin` | Above + each primitive x each displacement x each mapper | P x M + P x D x M |
| `--mapper_as_augmentation` | Mapper does not contribute to classes | primitives (+ P x D if disp) |
| `--displacement_as_augmentation` | Displacement does not contribute to classes | P x M |

### Class ID Assignment Order

1. **Hybrid Primitives** (primitive x mapper): ID 0, 1, 2, ...
2. **Hybrid Displaced Primitives** (primitive x displacement x mapper): Continuing IDs
3. **CombinedUnion Primitives** (union x mapper): Continuing IDs

### Examples

```bash
--primitives sphere cylinder --sdf_mappers inverse_cube linear
```

| Class ID | Name |
|----------|------|
| 0 | Sphere + inverse_cube |
| 1 | Sphere + linear |
| 2 | Cylinder + inverse_cube |
| 3 | Cylinder + linear |

→ Total: 4 classes

```bash
--primitives sphere cylinder --sdf_mappers inverse_cube --displacement_functions perlin
```

| Class ID | Name |
|----------|------|
| 0 | Sphere + inverse_cube |
| 1 | Cylinder + inverse_cube |
| 2 | Sphere + displacement(perlin) + inverse_cube |
| 3 | Cylinder + displacement(perlin) + inverse_cube |

→ Total: 4 classes (2x1 + 2x1x1)

---

## grid_scale Parameter

In the segmentation dataset, primitives are generated at relatively small sizes to allow multiple placements. For classification tasks, only a single primitive is placed, so it needs to be enlarged within the voxel space.

`grid_scale` controls the coordinate range of the mesh grid:

- **Segmentation version**: `linspace(-D/2, D/2-1, D)` → coordinate range = `[-32, 31]` (when D=64)
- **Classification version**: `linspace(-D/2 * scale, D/2 * scale, D)` → coordinate range = `[-14.4, 14.4]` (when D=64, scale=0.45)

A smaller coordinate range makes the same-parameter primitive appear relatively larger in voxel space.

| grid_scale | Effect |
|------------|--------|
| 1.0 | Same size as segmentation version (small) |
| 0.5 | Primitive appears 2x larger |
| **0.45** (default) | Primitive appears ~2.22x larger |
| 0.35 | Primitive appears ~2.86x larger |
| 0.25 | Primitive appears 4x larger |

---

## Z-Score Normalization

In SSL3D_classification's preprocessing pipeline, Z-Score normalization is applied before resampling and is not included in training augmentation. This dataset applies Z-Score normalization at generation time, saving pre-processed data:

```python
mean = x.mean()
std = x.std()
x_normalized = (x - mean) / max(std, 1e-8)
```

---

## Primitive Placement

- **Center placement**: Pass `center=[0, 0, 0]` to disable translation
- **Rotation/shear**: Applied with `transform=True` (default). Rotation angle in `[-pi, pi]`, shear amount in `[-0.1, 0.1]`
- **No translation**: The primitive is always placed at the center of the volume

---

## SSL3D_classification DataLoader Implementation Guide

### Required Files

1. `datasets/sdf_classification.py` — Dataset + DataModule
2. `cli_configs/data/<dataset_name>.yaml` — Place the generated YAML here

### Dataset Class Example

```python
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from utils.io import Blosc2IO  # SSL3D_classification's Blosc2IO utility


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
        # Labels are direct integer values
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

### DataModule Class Implementation Example

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

### `__getitem__` Return Values

| Item | Value |
|------|-------|
| `img` | `torch.Tensor`, shape `(1, D, H, W)`, dtype `float32` |
| `label` | `int` (0-indexed class ID) |

---

## Command Examples

### Basic Generation

```bash
uv run python src/fdslxsdf4seg/generate_sdf_dataset_classification.py \
    --out_dir outputs/cls_basic \
    --D 96 --H 96 --W 96 \
    --samples_per_class 50 \
    --primitives sphere cylinder torus cone \
    --num_visualize 5
```

→ 4 primitives x 1 mapper = 4 classes x 50 = 200 samples

### Displacement + Multiple Mappers

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

→ (2x2) + (2x2x2) = 4 + 8 = 12 classes x 100 = 1200 samples

### Augmentation Mode

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

→ 3 classes x 50 = 150 samples (mapper and displacement are randomly selected but do not affect class ID)
