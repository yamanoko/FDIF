# Primitive Variation Visualization

This feature visualizes the same primitive multiple times to demonstrate how much variation it exhibits during generation. The resulting images are arranged side by side and output as a single combined image.

## Usage

### Basic Usage

```bash
# Generate 6 variations of the Sphere primitive
python visualize_primitives.py --variations

# Generate variations for a specific primitive
python visualize_primitives.py --variations --variation_primitive FiveStarPrism

# Specify the number of variations
python visualize_primitives.py --variations --variation_primitive Cylinder --num_variations 9

# Specify an output directory
python visualize_primitives.py --variations --output_dir my_output --variation_primitive Torus
```

### Options

- `--variations`: Enable variation analysis mode
- `--variation_primitive`: Name of the primitive to analyze (default: "Sphere")
- `--num_variations`: Number of variations to generate (default: 6)
- `--output_dir`: Output directory (default: "visualize_output")
- `--3d`: Also enable 3D visualization (normally disabled in variation analysis)

## Supported Primitives

### Basic Primitives
- `Sphere` - Sphere
- `Torus` - Torus
- `Cone` - Cone
- `Octahedron` - Octahedron
- `Cylinder` - Cylinder
- `ConvexCylinder` - Convex Cylinder
- `ConcaveCylinder` - Concave Cylinder
- `ConeCylinder` - Cone Cylinder

### Sector Polygon Prisms
- `TrianglePrism` - Triangle Prism
- `SquarePrism` - Square Prism
- `PentagonPrism` - Pentagon Prism
- `HexagonPrism` - Hexagon Prism
- `HeptagonPrism` - Heptagon Prism
- `OctagonPrism` - Octagon Prism

### Cone Prisms
- `TriangleConePrism` - Triangle Cone Prism
- `SquareConePrism` - Square Cone Prism
- `PentagonConePrism` - Pentagon Cone Prism
- `HexagonConePrism` - Hexagon Cone Prism

### Convex / Concave Prisms
- `TriangleConvexPrism` - Triangle Convex Prism
- `SquareConvexPrism` - Square Convex Prism
- `TriangleConcavePrism` - Triangle Concave Prism
- `SquareConcavePrism` - Square Concave Prism

### Star Prisms
- `FiveStarPrism` - Five-pointed Star Prism
- `SixStarPrism` - Six-pointed Star Prism

### Torus Variants
- `SquareTorus` - Square Torus
- `PentagonTorus` - Pentagon Torus
- `HexagonTorus` - Hexagon Torus
- `FiveStarTorus` - Five-pointed Star Torus
- `SixStarTorus` - Six-pointed Star Torus

### Revolution Shapes (Solids of Revolution)
- `ThreeStarRevolution` - Three-pointed Star Revolution
- `FourStarRevolution` - Four-pointed Star Revolution
- `FiveStarRevolution` - Five-pointed Star Revolution

### Onioned Shapes (Concentric Shell Shapes)
- `OnionedCylinder` - Onioned Cylinder
- `OnionedTrianglePrism` - Onioned Triangle Prism
- `OnionedSquarePrism` - Onioned Square Prism
- `OnionedFiveStarPrism` - Onioned Five-pointed Star Prism

### Union Shapes (Combined Shapes)
- `SphereCylinderUnion` - Sphere + Cylinder Union
- `SphereTriangleUnion` - Sphere + Triangle Union
- `FiveStarRevolutionCylinderUnion` - Five-pointed Star Revolution + Cylinder Union
- `FiveStarRevolutionPentagonUnion` - Five-pointed Star Revolution + Pentagon Union

## Output

### File Structure
```
visualize_output/
└── variations/
    ├── sphere_var_01.png              # Individual variation images
    ├── sphere_var_02.png
    ├── ...
    ├── sphere_var_06.png
    ├── sphere_var_01_slice.png        # Slice views
    ├── sphere_var_02_slice.png
    ├── ...
    └── sphere_variations_combined.png # Combined comparison image
```

### Combined Image Features
- All variations are arranged in a single image
- Each variation is numbered
- The primitive name and sample count are shown in the title
- Clean grid layout for easy comparison

## Examples

### Basic Variation Analysis
```bash
# Basic variation check for Sphere
python visualize_primitives.py --variations

# Detailed analysis with more variations
python visualize_primitives.py --variations --num_variations 12 --variation_primitive FiveStarPrism
```

### Complex Primitive Analysis
```bash
# Variations of Union primitives
python visualize_primitives.py --variations --variation_primitive FiveStarRevolutionCylinderUnion --num_variations 9

# Onioned primitive analysis
python visualize_primitives.py --variations --variation_primitive OnionedFiveStarPrism --num_variations 6
```

## Dependencies

The following library is required for the variation image combining feature:

```bash
pip install Pillow
```

If Pillow is not installed, individual images will still be generated, but the combined image will not be created.

## Statistics

During execution, the following statistics are displayed for each variation:
- `inside`: Number of voxels inside the primitive
- `outside`: Number of voxels outside the primitive

This allows you to numerically verify differences in size and shape across variations.

## Notes

1. **Randomness**: Some primitives include random parameters, so different variations may be generated each time you run the script.

2. **Memory Usage**: Be mindful of memory usage when generating a large number of variations.

3. **Execution Time**: Execution time may increase depending on the complexity of the primitive and the number of variations.

4. **3D Visualization**: 3D visualization is normally disabled in variation analysis, but it can be enabled with the `--3d` option. Note that enabling 3D visualization with many variations will significantly increase processing time.