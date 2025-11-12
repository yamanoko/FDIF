# FDSLxSDF4Seg AI Development Guide

## 重要な指示 / Important Instructions

**日本語で応答してください。** このプロジェクトは日本語での開発を前提としています。コードや進捗の説明には日本語で応答してください。

### 開発時の必須ルール

1. **コード変更時の文書更新**: コードの追加・変更を行った際は、必ずこの `copilot-instructions.md` ファイルも更新してください。新しいパターン、関数、クラスを追加した場合は、適切なセクションに説明を追加します。

2. **Pythonコードの実行**: このプロジェクトでは **`uv`** パッケージマネージャーを使用します。Python実行時は以下のパターンに従ってください：
   ```bash
   # スクリプト実行
   uv run python src/fdslxsdf4seg/generate_sdf_dataset.py [args]
   
   # パッケージインストール
   uv pip install [package]
   
   # 依存関係の追加
   uv add [package]
   ```

3. **コードスタイル**: Ruffフォーマッター設定に従い、88文字制限、ダブルクォートを使用してください。

---

## Project Overview

**FDSLxSDF4Seg** is a framework for 3D medical image segmentation using SDF (Signed Distance Function)-based synthetic dataset generation. The project combines procedural 3D geometry with deep learning to train segmentation models (VNet, UNETR, SwinUNETR) on synthetic data that can transfer to real medical imaging tasks.

**Core Workflow**: Generate synthetic SDF volumes → Train segmentation models → Fine-tune on real data (BTCV)

## Architecture: The Hybrid Primitive System

The codebase's key innovation is a **three-layer primitive composition system**:

1. **Base Primitives** (`basic_sdf.py`, `sector_polygon_prism/`, `star_polygon_prism/`): ~100+ geometric shapes (Sphere, Cylinder, Torus, polygon prisms, etc.) that implement `SDFObject` base class with `_sdf(x, y, z)` method
2. **SDF Mappers** (`sdf_mapper.py`): Transform functions (InverseCubeMapper, ExponentialMapper, LinearMapper) that convert SDF distance values to intensity values
3. **Hybrid Primitives** (`hybrid_primitive.py`): Cross-product of primitives × mappers, creating unique segmentation classes. Each `HybridPrimitive` wraps a primitive class + mapper, with unique names like `"Cylinder_inverse_cube"`

**Why this matters**: The system can generate hundreds of distinct classes from base primitives. Each `HybridPrimitive` represents a different "tissue type" in synthetic data, allowing flexible class counts without hardcoding shapes.

### Key Data Flow
```
SDFObject._sdf() → raw distance field
→ SDFMapper.apply() → intensity volume
→ PyTorch Dataset → .nii.gz files
→ MONAI training pipeline
```

## Critical Patterns & Conventions

### SDF Object Hierarchy
- All primitives inherit from `SDFObject` (`sdf_object.py:12`)
- Override `_sdf(x, y, z)` to define shape geometry
- Transform matrix system: `T * R * S` (translate, rotate, shear) applied via `applied_transform()`
- Prism-based shapes inherit from `_PrismBase`, `_ConvexPrismBase`, `_ConcavePrismBase`, or `_ConePrismBase` for 2D→3D extrusion

### Registry Pattern
- `primitive_registry.py` maintains `ALL_PRIMITIVES` dict and categorized groups
- Use `select_primitives(primitives=[], categories=[], num_classes=N)` to filter shapes
- Categories: `"basic"`, `"sector_polygon"`, `"star_polygon"`, `"revolution"`, `"onioned_*"`
- `DEFAULT_PRIMITIVES` = `[Sphere, Cylinder, Torus, Cone]` for backward compatibility

### Dataset Generation (`generate_sdf_dataset.py`)
- `SDFSegmentationDataset` creates volumes with 2-5 objects per sample
- **Output channels = num_unique_hybrid_primitives + 1 (background)**
- Generated files: `imagesTr/`, `labelsTr/`, `data.json` (MONAI-compatible Decathlon format)
- Visualization outputs: Plotly 3D HTML + slice PNGs in `visualizations/`

### Training Pipeline (`training.py`)
- **Critical**: Match `--out_channel` to dataset's class count (read from generation logs)
- Real data uses `CacheDataset` with aggressive transforms; synthetic uses `Dataset`
- Sliding window inference: `roi_size = spatial_size` from `--grid_size`
- Pretrained model loading: `--pretrained_model` + `--pretraining_out_channel` (adjusts final layer if class count differs)

## Command Patterns

### Generate Synthetic Dataset
```bash
uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --out_dir ./outputs/my_dataset \
    --D 96 --H 96 --W 96 \
    --num_samples 500 --num_val_samples 100 \
    --primitives sphere box cylinder \
    --sdf_mappers inverse_cube linear \
    --num_visualize 10
```
→ Creates 3 primitives × 2 mappers = 6 classes + background = `--out_channel 7`

**nnUNet Format**:
```bash
uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --nnunet_format \
    --dataset_id 999 \
    --dataset_name SDFSynthetic \
    --D 96 --H 96 --W 96 \
    --num_samples 500 --num_val_samples 100 \
    --primitives sphere box cylinder \
    --sdf_mappers inverse_cube linear \
    --num_visualize 10
```
→ Creates Dataset999_SDFSynthetic with nnUNet-compatible structure
→ Validation data saved as separate Dataset1000_SDFSynthetic_Val

### Train Model
```bash
uv run python src/fdslxsdf4seg/training.py \
    --data_json_path ./outputs/my_dataset/data/data.json \
    --model_name swin_unetr \
    --out_channel 7 \
    --grid_size 96 96 96 \
    --max_iterations 30000
```

### Visualize Primitives
```bash
uv run python visualize_primitives.py --variations \
    --variation_primitive Cylinder --num_variations 9
```

### Fine-tune on Real Data
```bash
uv run python src/fdslxsdf4seg/training.py \
    --data_json_path ./BTCV/dataset.json \
    --model_name unetr \
    --pretrained_model ./training_output/vnet/model_best.pth \
    --pretraining_out_channel 7 \
    --out_channel 14 \
    --is_real_data \
    --grid_size 96 96 96
```

## Development Environment

### Dependencies (pyproject.toml)
- **PyTorch**: CUDA 12.8 via custom index (uses `uv` for package management)
- **MONAI**: 3D medical imaging transforms/models
- **Plotly + Kaleido**: Interactive 3D visualizations
- **nibabel**: NIfTI format I/O

### Code Quality
- Ruff formatter: 88 char lines, double quotes
- Type hints encouraged (see `py.typed` marker)
- Japanese comments common in original code—preserve or translate consistently

## File Organization

```
src/fdslxsdf4seg/
├── basic_sdf.py              # Core shapes: Sphere, Torus, Cylinder, etc.
├── sdf_object.py             # SDFObject base + transform logic
├── sdf_mapper.py             # Intensity mapping functions + MapperRegistry
├── hybrid_primitive.py       # Primitive × Mapper cross-product
├── primitive_registry.py     # ALL_PRIMITIVES catalog + selection logic
├── combined_union_primitives.py  # Union of two primitives (experimental)
├── generate_sdf_dataset.py   # Main dataset generation script
├── training.py               # MONAI-based training loop
├── lr_scheduler.py           # LinearWarmupCosineAnnealingLR
├── visualize_training_metrics.py  # Plot Dice scores from experiments
└── {sector_polygon_prism/, star_polygon_prism/, etc.}  # Shape variants
```

## Testing & Debugging

- No formal test suite—validation via visual inspection of `visualizations/` output
- Check `generation_log.txt` in output dirs for class mappings
- Use `--num_visualize 5` to generate sample visualizations during dataset creation
- For training issues: check `training_log.txt` and Dice metric plots

## Common Pitfalls

1. **Class count mismatch**: Always verify `out_channel = len(hybrid_primitives) + 1`
2. **CUDA OOM**: Reduce `--batch_size`, `--grid_size`, or use `feature_size=24` for SwinUNETR
3. **Transform flag**: `transform=True` in `SDFObject` enables rotation/shear—needed for diverse synthetic data
4. **NIfTI orientation**: Real data uses `Orientationd(axcodes="RAS")` preprocessing
5. **Mapper naming**: Use exact names from `MapperRegistry.MAPPERS` dict keys
6. **Visualization errors**: `visualize_training_metrics.py` uses safe annotation positioning (offset-based, not multiplication) to prevent matplotlib "Image size too large" errors. Annotation positions are calculated using data range offsets instead of coordinate multiplication to avoid extreme values when data is very small.

## Bug Fixes & Known Issues

## Extending the System

### Add New Primitive
1. Create class in appropriate module (e.g., `basic_sdf.py`)
2. Inherit from `SDFObject` or `_PrismBase`
3. Implement `_sdf(x, y, z)` returning distance field
4. Register in `primitive_registry.py` → `ALL_PRIMITIVES`

### Add New Mapper
1. Create class in `sdf_mapper.py` inheriting `SDFMapper`
2. Implement `apply(sdfs)` and `get_name()`
3. Register in `MapperRegistry.MAPPERS`

### Custom Dataset Config
Edit `select_primitives()` call in `SDFSegmentationDataset.__init__` to use custom primitive lists or categories.

## Resources

- **Documentation**: `generate_sdf_dataset.md`, `training.md`, `VISUALIZATION_README.md`, `VARIATIONS_README.md`
- **Real data setup**: `BTCV/make_data_json.py` creates Decathlon-format JSON from medical scans
- **Experiment tracking**: Check `outputs/YYYYMMDD_HHMMSS/` timestamped directories
