# generate_sdf_dataset.py

## Overview

`generate_sdf_dataset.py` is a Python script for generating SDF (Signed Distance Function)-based synthetic datasets for 3D segmentation tasks. It automatically generates 3D volume data containing multiple 3D primitives (sphere, box, cylinder, torus, and 100+ other shapes) along with their segmentation masks.

**Two output formats are supported**:
- **MONAI Decathlon format** (default): For the MONAI framework
- **nnUNet format**: For the nnUNet framework (enabled with the `--nnunet_format` flag)

## Key Features

### 1. 3D Primitive Generation
- **Sphere**: Spheres with random radii
- **Box**: Rectangular boxes with random dimensions
- **Cylinder**: Cylinders with random radii and heights
- **Torus**: Torii with random major/minor radii
- **100+ additional shapes**: Polygon prisms, star prisms, revolution shapes, onioned variants, etc.

### 2. Primitive Selection
- **Use all primitives**: All 4 default primitives are used by default
- **Selective use**: Choose specific primitives by name or category
- **Multiple instances**: Generate multiple instances of the same primitive per sample
- **Dynamic class count**: Segmentation class count varies based on selected primitives, mappers, and displacements

### 3. Spatial Transforms
- **Translation**: Random object placement within the volume
- **Rotation**: Random rotation around X, Y, Z axes
- **Shear**: Random shear deformation

### 4. Dataset Generation
- Generate 3D volumes containing multiple objects
- SDF-based intensity image generation
- Object ID-based segmentation mask generation
- NIfTI format output

### 5. Visualization
- 3D visualization using Plotly
- Slice image generation
- Interactive HTML output

## Usage

### Basic Usage

```bash
python generate_sdf_dataset.py --out_dir ./output --num_samples 100
```

### Parameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--out_dir` | Auto-generated | Output directory path |
| `--D` | 64 | Volume depth (Z-axis) |
| `--H` | 64 | Volume height (Y-axis) |
| `--W` | 64 | Volume width (X-axis) |
| `--num_samples` | 200 | Number of training samples to generate |
| `--num_val_samples` | 0 | Number of validation samples to generate |
| `--min_objects` | 2 | Minimum objects per sample |
| `--max_objects` | 5 | Maximum objects per sample |
| `--primitives` | All primitives | Primitives to use (e.g., `sphere`, `box`, `cylinder`, `torus`) |
| `--seed` | None | Random seed for reproducibility |
| `--num_visualize` | 0 | Number of samples to visualize |
| `--nnunet_format` | False | Generate dataset in nnUNet format |
| `--dataset_id` | 999 | nnUNet dataset ID (e.g., 999 → `Dataset999_Name`) |
| `--dataset_name` | "SDFSynthetic" | nnUNet dataset name |

### Examples

#### 1. Basic Dataset Generation (MONAI Decathlon Format)
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

#### 2. nnUNet Format Dataset Generation
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

**nnUNet format details**:
- Training data is saved to `Dataset999_SDFSynthetic/imagesTr/` and `labelsTr/`
- Validation data is saved as a separate dataset `Dataset1000_SDFSynthetic_Val/imagesTr/` and `labelsTr/`
- Image filenames: `case_00000_0000.nii.gz` (following nnUNet naming conventions)
- Label filenames: `case_00000.nii.gz`
- `dataset.json` contains metadata (channel names, labels, sample count)

#### 3. Dataset Generation with Visualization
```bash
uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --out_dir ./dataset_with_viz \
    --num_samples 100 \
    --num_visualize 10 \
    --seed 42
```

#### 4. Using Specific Primitives Only
```bash
# Use only sphere and box
uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --out_dir ./sphere_box_dataset \
    --primitives sphere box \
    --num_samples 300 \
    --max_objects 6

# Single primitive (cylinder only)
uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --out_dir ./cylinder_only_dataset \
    --primitives cylinder \
    --num_samples 200 \
    --max_objects 8
```

#### 5. Progressive Complexity Datasets
```bash
# Level 1: Single primitive
uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --out_dir ./level1_simple \
    --primitives sphere \
    --max_objects 3 \
    --num_samples 200

# Level 2: Multiple primitives
uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --out_dir ./level2_medium \
    --primitives sphere box \
    --max_objects 5 \
    --num_samples 300

# Level 3: All primitives
uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --out_dir ./level3_complex \
    --primitives sphere box cylinder torus \
    --max_objects 8 \
    --num_samples 500
```

## Output Structure

### MONAI Decathlon Format (Default)

```
output_directory/
├── generation_log.txt              # Generation log
├── data/
│   ├── data.json                   # Dataset metadata
│   ├── image/                      # SDF intensity images
│   │   ├── batch_0000/
│   │   │   ├── sample_00000_x.nii.gz
│   │   │   └── ...
│   │   └── ...
│   └── label/                      # Segmentation masks
│       ├── batch_0000/
│       │   ├── sample_00000_y.nii.gz
│       │   └── ...
│       └── ...
└── visualizations/                 # Visualization results (if --num_visualize > 0)
    ├── visualization_00000.png
    ├── visualization_00000_slice.png
    └── ...
```

### nnUNet Format (when --nnunet_format is specified)

```
Dataset999_SDFSynthetic/            # Training dataset
├── dataset.json                     # nnUNet metadata
├── imagesTr/                        # Training images
│   ├── case_00000_0000.nii.gz
│   ├── case_00001_0000.nii.gz
│   └── ...
├── labelsTr/                        # Training labels
│   ├── case_00000.nii.gz
│   ├── case_00001.nii.gz
│   └── ...
├── generation_log.txt               # Generation log
└── visualizations/                  # Visualization results (if --num_visualize > 0)
    └── ...

Dataset1000_SDFSynthetic_Val/        # Validation dataset (if --num_val_samples > 0)
├── dataset.json                     # nnUNet metadata
├── imagesTr/                        # Validation images (saved as imagesTr per nnUNet convention)
│   ├── case_00500_0000.nii.gz
│   ├── case_00501_0000.nii.gz
│   └── ...
└── labelsTr/                        # Validation labels
    ├── case_00500.nii.gz
    ├── case_00501.nii.gz
    └── ...
```

## Data Format

### MONAI Decathlon Format: data.json
```json
{
    "training": [
        {
            "image": "/absolute/path/sample_00000_x.nii.gz",
            "label": "/absolute/path/sample_00000_y.nii.gz", 
            "id": "sample_00000"
        }
    ],
    "validation": [...]
}
```

### nnUNet Format: dataset.json
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

**nnUNet format details**:
- `channel_names`: Input channels (a single SDF volume channel in this case)
- `labels`: Segmentation class name-to-ID mapping (using hybrid primitive names)
- `numTraining`: Number of training samples
- `file_ending`: File extension (`.nii.gz`)

### Image Data
- **MONAI format - Image files (*_x.nii.gz)**: Intensity values computed from SDF values (range 0–128)
- **MONAI format - Label files (*_y.nii.gz)**: Object IDs (0=background, 1–N=each primitive, N = number of selected primitives)
- **nnUNet format - Image files (*_0000.nii.gz)**: Intensity values computed from SDF values (range 0–128)
- **nnUNet format - Label files (*.nii.gz)**: Object IDs (0=background, 1–N=each hybrid primitive)

### Primitive ID Mapping
IDs are assigned based on the selected primitives:
- **All primitives**: 1=Sphere, 2=Box, 3=Cylinder, 4=Torus
- **Partial selection**: IDs are assigned sequentially starting from 1 in selection order

Example: `--primitives box torus`
- 1=Box, 2=Torus

## Technical Details

### SDF Computation
SDF functions for each primitive:
- **Sphere**: `|p| - radius`
- **Box**: Distance from box boundary
- **Cylinder**: Combination of axial and radial distances
- **Torus**: Combination of distance from main axis and tube radius

### Intensity Computation
```python
x_vol = 128.0 / (torch.pow(torch.abs(torch.stack(sdfs, dim=0)), 2.0) + 1.0)
x_vol = x_vol.sum(dim=0)
x_vol = torch.clamp(x_vol, 0.0, 128.0).to(torch.uint8)
```

### Segmentation Priority
When multiple objects overlap, the object with larger volume takes priority.

## Dependencies

- torch
- nibabel
- numpy
- plotly
- argparse

## Notes

1. **GPU usage**: Automatically uses GPU when CUDA is available
2. **Memory usage**: Be mindful of memory usage with large grid sizes or many samples
3. **File paths**: Absolute paths are recorded in the output data.json
4. **Visualization**: Plotly and kaleido (for image export) are required for the visualization feature

## Troubleshooting

### Common Issues

1. **Out of memory**: Reduce grid size or batch size
2. **Visualization errors**: Install the kaleido package: `pip install kaleido`
3. **CUDA errors**: Falls back to CPU mode if GPU memory is insufficient

### Performance Optimization

- Set `--num_visualize 0` for large-scale data generation
- Batch processing controls memory usage
- SSD storage is recommended
