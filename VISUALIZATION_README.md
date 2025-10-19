# Visualization Module Documentation

This module provides comprehensive tools for visualizing SDF (Signed Distance Field) primitives and comparing training metrics from deep learning experiments.

## Overview

The visualization module consists of two main components:

1. **Primitive Visualization** (`visualize_primitives.py`) - Generate and visualize SDF primitives
2. **Training Metrics Visualization** (`visualize_training_metrics.py`) - Compare training experiments

---

## Part 1: Primitive Visualization

### Features

The primitive visualization module provides advanced visualization tools for SDF primitives:

#### 3D Mesh Visualization
- **Marching Cubes Algorithm**: Converts SDF data to 3D surface meshes
- **Interactive HTML Output**: Plotly-based interactive 3D visualizations
- **Static PNG Export**: Optional static image generation (requires kaleido)
- **Semi-transparent Rendering**: Better visibility of shape surfaces

#### Batch Visualization
- Generate visualizations for multiple primitives at once
- Support for all primitive types in the system
- Configurable output formats

#### Primitive Variations
- Generate and analyze multiple variations of a single primitive
- Combined grid visualization of variations
- 2D slice visualization alongside 3D mesh views

#### Multi-Primitive Comparison
- Combine multiple primitive visualizations in a single HTML page
- Iframe-based layout for interactive browsing
- Direct links to individual visualizations

### Supported Primitives

#### Basic Primitives
- **Sphere**
- **Torus**
- **Cone**
- **Octahedron**
- **Cylinder** (including Convex and Concave variants)

#### Polygon Prisms
- **Sector Polygons**: Triangle, Square, Pentagon, Hexagon, Heptagon, Octagon, Nonagon
- **Cone Prisms**: Triangle, Square, Pentagon, Hexagon
- **Convex Prisms**: Triangle, Square
- **Concave Prisms**: Triangle, Square

#### Star Primitives
- **Star Prisms**: Five-pointed, Six-pointed
- **Star Torii**: Five, Six, Seven, Eight-pointed stars
- **Star Revolutions**: Three, Four, Five-pointed stars

#### Complex Shapes
- **Onioned Primitives**: Concentric layered versions of sector and star polygons
- **Revolutions**: Star-based rotational shapes
- **Unions**: Combined shapes (e.g., Sphere+Cylinder, Sphere+Torus)

### Usage

#### Generate All Primitive Visualizations
```bash
python visualize_primitives.py
```

#### Generate Specific Primitive Types
```python
from visualize_primitives import generate_primitive_visualizations

# Generate only basic primitives
generate_primitive_visualizations(
    output_dir="visualize_output",
    primitive_type="basic",
    enable_3d=True
)

# Generate all primitives with auto-combining
generate_primitive_visualizations(
    output_dir="visualize_output",
    primitive_type="all",
    enable_3d=True,
    auto_combine=True
)
```

#### Generate Single Primitive Variations
```python
from visualize_primitives import generate_primitive_variations

# Generate 8 variations of Sphere with 3D mesh
generate_primitive_variations(
    output_dir="visualize_output",
    primitive_name="Sphere",
    num_variations=8,
    enable_3d=True
)
```

#### Combine Multiple Primitive Visualizations
```python
from visualize_primitives import combine_3d_visualizations

# Create a combined HTML page with multiple primitives
result = combine_3d_visualizations(
    primitive_names=["Sphere", "Torus", "Cone", "Cylinder"],
    output_dir="visualize_output",
    viz_type="mesh"
)
```

#### Get All Available Primitive Names
```python
from visualize_primitives import get_all_primitive_names

primitives = get_all_primitive_names()
print(f"Total primitives available: {len(primitives)}")
for prim in primitives:
    print(f"  - {prim}")
```

### Output Files

#### 3D Mesh Visualization
- `{primitive_name}_visualization_mesh.html`: Interactive 3D mesh (Plotly)
- `{primitive_name}_visualization_mesh.png`: Static 3D mesh image (if kaleido installed)

#### Variation Analysis
- `variations/{primitive_name}_var_01.png`: Individual variation 2D slice
- `variations/{primitive_name}_variations_combined.png`: Combined variation grid
- `variations/{primitive_name}_variations_3d_combined.png`: Combined 3D meshes (if enabled)

#### Combined Visualizations
- `combined/combined_mesh_primitives_{timestamp}.html`: Multi-primitive HTML page
- `combined/combined_3d_primitives_{timestamp}.png`: Combined 3D mesh grid

### Advanced Options

#### Custom Grid Configuration
```python
generate_primitive_visualizations(
    output_dir="visualize_output",
    primitive_type="all",
    enable_3d=True,
    auto_combine=True,
    grid_cols=3,  # 3 columns in combined images
    combine_3d=True  # Also combine 3D meshes
)
```

#### Enable/Disable Features
```python
# Mesh visualization only (faster, more compatible)
visualize_primitive_marching_cubes(
    sdf_data=sdf_array,
    output_file="output.html",
    name="MyPrimitive"
)
```

### Requirements

#### Core Requirements
- `torch`: SDF computation
- `plotly`: Interactive 3D visualization
- `scikit-image`: Marching Cubes algorithm
- `numpy`: Array operations

#### Optional Requirements
- `kaleido`: Static image export (PNG/JPG)
- `Pillow`: Image combining operations

Install optional dependencies:
```bash
pip install kaleido Pillow
```

### Technical Details

#### SDF to Mesh Conversion
The module uses the **Marching Cubes** algorithm to convert SDF grids to 3D surface meshes:
1. Evaluate SDF on a 3D grid (default 64×64×64)
2. Find isosurface at SDF level = 0
3. Generate triangular mesh vertices and faces
4. Visualize with Plotly's Mesh3d

#### Grid Generation
```python
grid_size = [64, 64, 64]
zs = torch.linspace(0, grid_size[0] - 1, grid_size[0])
ys = torch.linspace(0, grid_size[1] - 1, grid_size[1])
xs = torch.linspace(0, grid_size[2] - 1, grid_size[2])
Z, Y, X = torch.meshgrid(zs, ys, xs, indexing="ij")
```

#### 3D Mesh Export
```python
# Interactive HTML
fig.write_html("output.html")

# Static PNG (requires kaleido)
fig.write_image("output.png", width=800, height=600)
```

---

## Part 2: Training Metrics Visualization

### Features

#### Single Run Visualization
- Training loss over time
- Validation Dice score progression  
- Best performance annotations
- Statistical summary

#### Multiple Runs Comparison
- Side-by-side loss comparison
- Dice score comparison with best points highlighted
- Convergence analysis
- Performance summary table
- Overall best performance identification

### Usage

#### Single Training Run
Visualize metrics from a single training experiment:

```bash
python src/fdslxsdf4seg/visualize_training_metrics.py --output_dir training_output/vnet/training_from_scratch/20250713_123456
```

#### Multiple Training Runs Comparison
Compare multiple training experiments:

```bash
python src/fdslxsdf4seg/visualize_training_metrics.py --output_dirs \
    training_output/vnet/training_from_scratch/20250713_123456 \
    training_output/unetr/training_from_scratch/20250713_134567 \
    training_output/swin_unetr/fine_tuning/20250713_145678
```

#### Advanced Options

##### Save plots to specific directory:
```bash
python src/fdslxsdf4seg/visualize_training_metrics.py \
    --output_dirs run1/ run2/ run3/ \
    --save_to comparison_results/
```

##### Summary only (no plots):
```bash
python src/fdslxsdf4seg/visualize_training_metrics.py \
    --output_dirs run1/ run2/ run3/ \
    --no_plot
```

### Output Files

#### Single Run Mode
- `training_metrics_plot.png`: Combined loss and Dice score plots
- `training_loss_individual.png`: Individual training loss plot
- `validation_dice_individual.png`: Individual validation Dice score plot

#### Multiple Runs Mode
- `training_metrics_comparison.png`: Combined side-by-side comparison plots
- `training_loss_comparison_individual.png`: Individual training loss comparison
- `validation_dice_comparison_individual.png`: Individual validation Dice comparison
- `convergence_analysis.png`: Convergence rate analysis

### Generated Visualizations

#### 1. Training Metrics Comparison
Shows training loss and validation Dice scores for all compared runs with:
- Different colors for each run
- Best performance points marked with stars
- Legends identifying each experiment

#### 2. Convergence Analysis
Provides insights into training dynamics:
- Smoothed training loss (log scale) to assess convergence rate
- Validation Dice progression with markers showing evaluation points

#### 3. Summary Table
Console output showing:
- Best Dice score achieved by each run
- Final Dice score and training loss
- Step at which best performance was achieved
- Overall best performing run identification

### Requirements

The script requires the following files to be present in each training output directory:
- `training_loss.npy`: NumPy array of training losses
- `validation_dice.npy`: NumPy array of validation Dice scores  
- `steps.npy`: NumPy array of corresponding training steps

These files are automatically generated by the updated `training.py` script.

### Example Output

```
================================================================================
TRAINING RUNS COMPARISON SUMMARY
================================================================================
Run Name                  Best Dice    Final Dice   Final Loss   Best Step
--------------------------------------------------------------------------------
20250713_123456          0.823450     0.820123     0.234567     15000     
20250713_134567          0.834567     0.831234     0.223456     18000     
20250713_145678          0.845678     0.842345     0.212345     12000     
--------------------------------------------------------------------------------
Best overall performance: 20250713_145678 (Dice: 0.845678)
================================================================================
```

---

## Integration and Workflows

### Workflow 1: Complete Primitive Analysis
```python
from visualize_primitives import (
    generate_primitive_variations,
    get_all_primitive_names,
)

# Get all primitives
all_prims = get_all_primitive_names()

# Generate variations for key primitives
key_primitives = ["Sphere", "Torus", "FiveStarPrism", "OnionedSquarePrism"]
for prim_name in key_primitives:
    if prim_name in all_prims:
        generate_primitive_variations(
            output_dir="visualize_output",
            primitive_name=prim_name,
            num_variations=6,
            enable_3d=True
        )
```

### Workflow 2: Batch Visualization and Comparison
```python
from visualize_primitives import (
    generate_primitive_visualizations,
    combine_3d_visualizations,
)

# Generate all primitives
generate_primitive_visualizations(
    output_dir="visualize_output",
    primitive_type="all",
    enable_3d=True,
    auto_combine=True
)

# Create specific comparison groups
sphere_comparisons = ["Sphere", "OnionedCylinder", "SphereCylinderUnion"]
combine_3d_visualizations(
    primitive_names=sphere_comparisons,
    output_dir="visualize_output",
    viz_type="mesh"
)
```

---

## Troubleshooting

### Missing Dependencies
If you see import errors, install the required packages:
```bash
pip install torch plotly scikit-image numpy
pip install kaleido Pillow  # Optional but recommended
```

### PNG Export Not Working
Static PNG export requires `kaleido`. Install with:
```bash
pip install kaleido
```

### Large Memory Usage
For very large grids or many primitives:
- Reduce `grid_size` parameter
- Process primitives in batches
- Use `enable_3d=False` for initial testing

### Visualization Not Displaying
- Check that output files are created in `visualize_output/`
- Ensure HTML files are opened in a modern web browser
- For iframe-based pages, ensure all linked files are in correct relative paths
