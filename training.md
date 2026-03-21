# training.py

## Overview

`training.py` is a Python script for training 3D segmentation models using the MONAI framework. It supports modern 3D segmentation architectures such as VNet, UNETR, and SwinUNETR, and works with both real and synthetic data. Fine-tuning from pre-trained models is also supported.

## Key Features

### 1. Supported Models
- **VNet**: V-shaped network for 3D medical image segmentation
- **UNETR**: Vision Transformer-based U-shaped network
- **SwinUNETR**: Swin Transformer-based U-shaped network

### 2. Data Processing
- **Real data support**: Medical image datasets such as BTCV
- **Synthetic data support**: SDF-generated datasets
- **Multi-modality support**: Multiple channel inputs (e.g., DWI+ADC for ISLES)
- **Data augmentation**: Flip, rotation, intensity shift, etc.
- **Automatic preprocessing**: Normalization, cropping, resampling

### 3. Training Features
- **Mixed precision training**: Speedup via CUDA AMP
- **Sliding window inference**: Efficient processing of large images
- **Metrics tracking**: Performance evaluation via Dice coefficient
- **Model saving**: Automatic saving of the best model

### 4. Pre-training Support
- Load pre-trained models
- Transfer learning with output layer adjustment
- Supports both fine-tuning and training from scratch

## Usage

### Basic Usage

```bash
python training.py \
    --data_json_path ./data/data.json \
    --model_name vnet \
    --out_channel 5
```

### Parameter Reference

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--data_json_path` | Yes | - | Path to dataset JSON file |
| `--model_name` | Yes | - | Model name (vnet/unetr/swin_unetr) |
| `--is_real_data` | - | False | Flag to use real data |
| `--pretrained_model` | - | None | Path to pre-trained model |
| `--pretraining_out_channel` | - | 14 | Output channels of the pre-trained model |
| `--grid_size` | - | [96,96,96] | Input grid size |
| `--out_channel` | - | 14 | Output channels (number of classes including background) |
| `--in_channels` | - | 1 | Input channels (number of modalities) |
| `--feature_size` | - | Auto | Feature size |
| `--batch_size` | - | 1 | Batch size |
| `--max_iterations` | - | 30000 | Maximum training iterations |
| `--out_dir` | - | Auto-generated | Output directory |

### Setting the Output Channel Count

The output channel count varies depending on primitive selection:

| Primitive Configuration | Output Channels | Example |
|------------------------|----------------|---------|
| Single primitive | 2 | `--primitives sphere --out_channel 2` |
| 2 primitives | 3 | `--primitives sphere box --out_channel 3` |
| 3 primitives | 4 | `--primitives sphere box cylinder --out_channel 4` |
| All primitives | 5 | `--primitives sphere box cylinder torus --out_channel 5` |
| Real data (BTCV) | 14 | `--is_real_data --out_channel 14` |

### Examples

#### 1. Basic Training on Synthetic Data
```bash
python training.py \
    --data_json_path ./synthetic_dataset/data/data.json \
    --model_name vnet \
    --out_channel 5 \
    --grid_size 64 64 64 \
    --max_iterations 10000
```

#### 2. Training on Real Data
```bash
python training.py \
    --data_json_path ./BTCV/dataset.json \
    --model_name swin_unetr \
    --is_real_data \
    --out_channel 14 \
    --feature_size 48 \
    --batch_size 2
```

#### 3. Fine-Tuning from a Pre-Trained Model
```bash
python training.py \
    --data_json_path ./data/data.json \
    --model_name unetr \
    --pretrained_model ./pretrained/model_best.pth \
    --pretraining_out_channel 14 \
    --out_channel 5 \
    --grid_size 96 96 96
```

#### 4. Training on High-Resolution Data
```bash
python training.py \
    --data_json_path ./data/data.json \
    --model_name swin_unetr \
    --grid_size 128 128 128 \
    --feature_size 48 \
    --batch_size 1 \
    --max_iterations 50000
```

#### 5. Training on Specific Primitive Data
```bash
# Single primitive data (2 classes: background + primitive)
python training.py \
    --data_json_path ./sphere_only_dataset/data/data.json \
    --model_name vnet \
    --out_channel 2 \
    --grid_size 64 64 64

# Multiple primitive data (3 classes: background + 2 primitives)
python training.py \
    --data_json_path ./sphere_box_dataset/data/data.json \
    --model_name swin_unetr \
    --out_channel 3 \
    --grid_size 96 96 96
```

#### 6. Multi-Modality Training (e.g., ISLES DWI+ADC)
```bash
# 2-modality (DWI+ADC for ISLES data)
python training.py \
    --data_json_path ~/ISLES_fdsls4seg/data.json \
    --model_name vnet \
    --in_channels 2 \
    --out_channel 2 \
    --is_real_data \
    --grid_size 96 96 96 \
    --max_iterations 30000

# Multi-modality + SwinUNETR
python training.py \
    --data_json_path ~/ISLES_fdsls4seg/data.json \
    --model_name swin_unetr \
    --in_channels 2 \
    --out_channel 2 \
    --is_real_data \
    --feature_size 48 \
    --grid_size 96 96 96
```

## Multi-Modality Input

### Overview
The training script supports medical images with multiple modalities (e.g., DWI, ADC, T1, T2, etc.).

### Input Image Format
Multi-modality data must be stored as 4D NIfTI files:
- **Shape**: (C, D, H, W) where C = number of modalities
- **Examples**:
  - ISLES (DWI + ADC): (2, 96, 112, 96)
  - T1 + T2 + FLAIR: (3, 256, 256, 128)

### Parameter Settings
```bash
--in_channels 2      # DWI + ADC
--in_channels 3      # T1 + T2 + FLAIR
--in_channels 4      # 4 modalities
```

### Usage Example
```bash
# Training on ISLES dataset (DWI + ADC)
python training.py \
    --data_json_path ~/ISLES_fdsls4seg/data.json \
    --model_name vnet \
    --in_channels 2 \
    --out_channel 2 \
    --is_real_data \
    --grid_size 96 96 96
```

### Important Notes
1. All input images must have the same number of channels
2. Channel ordering must be consistent (e.g., always DWI then ADC)
3. Ensure channels are merged correctly during dataset conversion
4. When using a pre-trained model, `--in_channels` must match the value used during training

### Dataset Conversion Example
Converting ISLES-2022 to FDSLxSDF4Seg format:
```bash
# 1. Convert to nnUNet format
python convert_isles_nnunet.py \
    -i /path/to/ISLES-2022 \
    -o $nnUNet_raw

# 2. Convert to FDSLxSDF4Seg format (channel merging)
python convert_nnunet_to_fdsls4seg.py \
    --input $nnUNet_raw/Dataset500_ISLES2022 \
    --output ~/ISLES_fdsls4seg

# 3. Train with 2 channels
python training.py \
    --data_json_path ~/ISLES_fdsls4seg/data.json \
    --model_name vnet \
    --in_channels 2 \
    --out_channel 2 \
    --is_real_data
```

## Data Requirements

### Dataset JSON Format
```json
{
    "training": [
        {
            "image": "path/to/image.nii.gz",
            "label": "path/to/label.nii.gz",
            "id": "sample_id"
        }
    ],
    "validation": [...]
}
```

### Image Format
- **File format**: NIfTI (.nii.gz)
- **Dimensions**: 
  - Single modality: 3D (D x H x W)
  - Multiple modalities: 4D (C x D x H x W) where C = `--in_channels`
- **Images**: Grayscale intensity values (normalization recommended)
- **Labels**: Integer class IDs (0=background, 1–N=each class)

### Class Count Based on Primitive Selection

The class count of a generated dataset depends on the primitives used:

```bash
# Example: Using only sphere and box
# Classes: 0=background, 1=sphere, 2=box → 3 classes
--out_channel 3

# Example: Using all primitives
# Classes: 0=background, 1=sphere, 2=box, 3=cylinder, 4=torus → 5 classes
--out_channel 5
```

## Output Structure

After training completes, the following files are generated:

```
output_directory/
├── training_log.txt           # Training logs and configuration
├── best_metric_model.pth      # Best performance model
└── (additional files may be added in the future)
```

## Model Details

### VNet
- **Features**: V-shaped architecture based on 3D convolutions
- **Use case**: Medium-scale 3D segmentation
- **Memory**: Relatively lightweight

### UNETR
- **Features**: Vision Transformer encoder + CNN decoder
- **Use case**: High-accuracy segmentation
- **Parameters**: `feature_size` (default: 16)

### SwinUNETR
- **Features**: Hierarchical architecture based on Swin Transformer
- **Use case**: Highest-accuracy segmentation
- **Parameters**: `feature_size` (default: 48)

## Data Augmentation

### For Real Data
- Scale intensity normalization (-175 to 250 → 0 to 1)
- Foreground crop
- Orientation normalization (RAS)
- Spatial resampling (1.5 x 1.5 x 2.0 mm)

### Common Augmentations
- Random crop (considering positive/negative samples)
- Random flip (10% probability per axis)
- Random 90-degree rotation (10% probability)
- Random intensity shift (50% probability, ±10%)

## Training Configuration

### Optimization
- **Optimizer**: AdamW
- **Learning rate**: 1e-4
- **Weight decay**: 1e-5
- **Mixed precision**: Using CUDA AMP

### Loss Function
- **DiceCELoss**: Dice loss + Cross Entropy loss
- **One-hot conversion**: Applied automatically
- **Softmax**: Applied automatically

### Evaluation
- **Metric**: Dice coefficient
- **Evaluation frequency**: Every 500 iterations
- **Inference method**: Sliding window (overlap 4)

## Performance Optimization

### Memory Optimization
1. **Batch size adjustment**: Set to 1–2 depending on GPU memory
2. **Grid size**: Adjust between 64–128 based on memory constraints
3. **Cache settings**: Adjust data loader cache size

### Speed Optimization
1. **Mixed precision**: Enabled automatically
2. **Data loader**: Multi-processing
3. **Cache dataset**: Accelerates frequently accessed data

## Dependencies

### Required Packages
- torch
- monai
- tqdm
- argparse

### Recommended Environment
- CUDA-capable GPU
- 16 GB+ GPU memory (for high-resolution data)
- SSD storage

## Troubleshooting

### Common Issues

#### 1. Out of Memory
```
Solution:
- Reduce batch_size to 1
- Reduce grid_size (e.g., 64x64x64)
- Set num_workers to 0
```

#### 2. Not Converging
```
Solution:
- Lower the learning rate (e.g., 1e-5)
- Run more iterations
- Adjust data augmentation
```

#### 3. Pre-trained Model Loading Error
```
Solution:
- Set pretraining_out_channel correctly
- Verify model architecture match
- Check path accuracy
```

### Debugging Tips

1. **Small-scale experiments**: Verify with a small number of samples first
2. **Check logs**: Verify configuration in training_log.txt
3. **GPU memory monitoring**: Use nvidia-smi to check memory usage

## Benchmarks and Expected Performance

### Synthetic Data (SDF)
- **Data size**: 64³
- **Classes**: 2–5 (number of primitives + 1)
- **Expected Dice**: 0.85–0.95
- **Training time**: 1–2 hours (RTX 3080)

### Real Data (BTCV)
- **Data size**: 96³
- **Classes**: 14
- **Expected Dice**: 0.75–0.85
- **Training time**: 8–12 hours (RTX 3080)

### Expected Performance by Complexity Level

| Level | Primitives | Classes | Expected Dice | Training Time |
|-------|-----------|---------|---------------|---------------|
| Level 1 | 1 type | 2 | 0.90–0.95 | 30 min |
| Level 2 | 2 types | 3 | 0.88–0.93 | 45 min |
| Level 3 | 3 types | 4 | 0.87–0.92 | 1 hour |
| Level 4 | 4 types | 5 | 0.85–0.90 | 1.5 hours |
