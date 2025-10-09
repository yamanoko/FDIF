# file: primitive_registry.py
"""
Primitive registry for SDF dataset generation.
Contains all available primitives and their categorization.
"""

from fdslxsdf4seg.basic_sdf import (
    ConcaveCylinder,
    Cone,
    ConeCylinder,
    ConvexCylinder,
    Cylinder,
    Octahedron,
    Sphere,
    Torus,
)
from fdslxsdf4seg.onioned_prism.onioned_sector_polygon_prism import (
    OnionedHeptagonConcavePrism,
    OnionedHeptagonConePrism,
    OnionedHeptagonConvexPrism,
    OnionedHeptagonPrism,
    OnionedHexagonConcavePrism,
    OnionedHexagonConePrism,
    OnionedHexagonConvexPrism,
    OnionedHexagonPrism,
    OnionedNonagonConcavePrism,
    OnionedNonagonConePrism,
    OnionedNonagonConvexPrism,
    OnionedNonagonPrism,
    OnionedOctagonConcavePrism,
    OnionedOctagonConePrism,
    OnionedOctagonConvexPrism,
    OnionedOctagonPrism,
    OnionedPentagonConcavePrism,
    OnionedPentagonConePrism,
    OnionedPentagonConvexPrism,
    OnionedPentagonPrism,
    OnionedSquareConcavePrism,
    OnionedSquareConePrism,
    OnionedSquareConvexPrism,
    OnionedSquarePrism,
    OnionedTriangleConcavePrism,
    OnionedTriangleConePrism,
    OnionedTriangleConvexPrism,
    OnionedTrianglePrism,
)
from fdslxsdf4seg.onioned_prism.onioned_star_polygon_prism import (
    OnionedEightStarConcavePrism,
    OnionedEightStarConePrism,
    OnionedEightStarConvexPrism,
    OnionedEightStarPrism,
    OnionedFiveStarConcavePrism,
    OnionedFiveStarConePrism,
    OnionedFiveStarConvexPrism,
    OnionedFiveStarPrism,
    OnionedSevenStarConcavePrism,
    OnionedSevenStarConePrism,
    OnionedSevenStarConvexPrism,
    OnionedSevenStarPrism,
    OnionedSixStarConcavePrism,
    OnionedSixStarConePrism,
    OnionedSixStarConvexPrism,
    OnionedSixStarPrism,
)
from fdslxsdf4seg.revolution.star_revolution import (
    FiveStarRevolution,
    FourStarRevolution,
    ThreeStarRevolution,
)
from fdslxsdf4seg.sector_polygon_prism.concave_sector_polygon_prism import (
    HeptagonConcavePrism,
    HexagonConcavePrism,
    NonagonConcavePrism,
    OctagonConcavePrism,
    PentagonConcavePrism,
    SquareConcavePrism,
    TriangleConcavePrism,
)
from fdslxsdf4seg.sector_polygon_prism.cone_sector_polygon_prism import (
    HeptagonConePrism,
    HexagonConePrism,
    NonagonConePrism,
    OctagonConePrism,
    PentagonConePrism,
    SquareConePrism,
    TriangleConePrism,
)
from fdslxsdf4seg.sector_polygon_prism.convex_sector_polygon_prism import (
    HeptagonConvexPrism,
    HexagonConvexPrism,
    NonagonConvexPrism,
    OctagonConvexPrism,
    PentagonConvexPrism,
    SquareConvexPrism,
    TriangleConvexPrism,
)
from fdslxsdf4seg.sector_polygon_prism.sector_polygon_prism import (
    HeptagonPrism,
    HexagonPrism,
    NonagonPrism,
    OctagonPrism,
    PentagonPrism,
    SquarePrism,
    TrianglePrism,
)
from fdslxsdf4seg.star_polygon_prism.concave_star_prism import (
    EightStarConcavePrism,
    FiveStarConcavePrism,
    SevenStarConcavePrism,
    SixStarConcavePrism,
)
from fdslxsdf4seg.star_polygon_prism.cone_star_prism import (
    EightStarConePrism,
    FiveStarConePrism,
    SevenStarConePrism,
    SixStarConePrism,
)
from fdslxsdf4seg.star_polygon_prism.convex_star_prism import (
    EightStarConvexPrism,
    FiveStarConvexPrism,
    SevenStarConvexPrism,
    SixStarConvexPrism,
)
from fdslxsdf4seg.star_polygon_prism.star_prism import (
    EightStarPrism,
    FiveStarPrism,
    SevenStarPrism,
    SixStarPrism,
)
from fdslxsdf4seg.torus.sector_polygon_torus import (
    HeptagonTorus,
    HexagonTorus,
    NonagonTorus,
    OctagonTorus,
    PentagonTorus,
    SquareTorus,
)
from fdslxsdf4seg.torus.star_torus import (
    EightStarTorus,
    FiveStarTorus,
    SevenStarTorus,
    SixStarTorus,
)
from fdslxsdf4seg.union.sphere_tube import (
    FiveStarRevolutionCylinderUnion,
    FiveStarRevolutionPentagonUnion,
    FiveStarRevolutionSquareUnion,
    FiveStarRevolutionTriangleUnion,
    FourStarRevolutionCylinderUnion,
    FourStarRevolutionPentagonUnion,
    FourStarRevolutionSquareUnion,
    FourStarRevolutionTriangleUnion,
    SphereCylinderUnion,
    SpherePentagonUnion,
    SphereSquareUnion,
    SphereTriangleUnion,
    ThreeStarRevolutionCylinderUnion,
    ThreeStarRevolutionPentagonUnion,
    ThreeStarRevolutionSquareUnion,
    ThreeStarRevolutionTriangleUnion,
)

# プリミティブの名前とクラスのマッピング
ALL_PRIMITIVES = {
    "sphere": Sphere,
    "cylinder": Cylinder,
    "torus": Torus,
    "cone": Cone,
    "octahedron": Octahedron,
    "convexcylinder": ConvexCylinder,
    "concavecylinder": ConcaveCylinder,
    "conecylinder": ConeCylinder,
    # Revolution objects
    "threestarrevolution": ThreeStarRevolution,
    "fourstarrevolution": FourStarRevolution,
    "fivestarrevolution": FiveStarRevolution,
    # Sector polygon prisms
    "triangleprism": TrianglePrism,
    "squareprism": SquarePrism,
    "pentagonprism": PentagonPrism,
    "hexagonprism": HexagonPrism,
    "heptagonprism": HeptagonPrism,
    "octagonprism": OctagonPrism,
    "nonagonprism": NonagonPrism,
    # Convex sector polygon prisms
    "triangleconvexprism": TriangleConvexPrism,
    "squareconvexprism": SquareConvexPrism,
    "pentagonconvexprism": PentagonConvexPrism,
    "hexagonconvexprism": HexagonConvexPrism,
    "heptagonconvexprism": HeptagonConvexPrism,
    "octagonconvexprism": OctagonConvexPrism,
    "nonagonconvexprism": NonagonConvexPrism,
    # Concave sector polygon prisms
    "triangleconcaveprism": TriangleConcavePrism,
    "squareconcaveprism": SquareConcavePrism,
    "pentagonconcaveprism": PentagonConcavePrism,
    "hexagonconcaveprism": HexagonConcavePrism,
    "heptagonconcaveprism": HeptagonConcavePrism,
    "octagonconcaveprism": OctagonConcavePrism,
    "nonagonconcaveprism": NonagonConcavePrism,
    # Cone sector polygon prisms
    "triangleconeprism": TriangleConePrism,
    "squareconeprism": SquareConePrism,
    "pentagonconeprism": PentagonConePrism,
    "hexagonconeprism": HexagonConePrism,
    "heptagonconeprism": HeptagonConePrism,
    "octagonconeprism": OctagonConePrism,
    "nonagonconeprism": NonagonConePrism,
    # Star polygon prisms
    "fivestarprism": FiveStarPrism,
    "sixstarprism": SixStarPrism,
    "sevenstarprism": SevenStarPrism,
    "eightstarprism": EightStarPrism,
    # Star convex prisms
    "fivestarconvexprism": FiveStarConvexPrism,
    "sixstarconvexprism": SixStarConvexPrism,
    "sevenstarconvexprism": SevenStarConvexPrism,
    "eightstarconvexprism": EightStarConvexPrism,
    # Star concave prisms
    "fivestarconcaveprism": FiveStarConcavePrism,
    "sixstarconcaveprism": SixStarConcavePrism,
    "sevenstarconcaveprism": SevenStarConcavePrism,
    "eightstarconcaveprism": EightStarConcavePrism,
    # Star cone prisms
    "fivestarconeprism": FiveStarConePrism,
    "sixstarconeprism": SixStarConePrism,
    "sevenstarconeprism": SevenStarConePrism,
    "eightstarconeprism": EightStarConePrism,
    # Torus variants
    "squaretorus": SquareTorus,
    "pentagontorus": PentagonTorus,
    "hexagontorus": HexagonTorus,
    "heptagontorus": HeptagonTorus,
    "octagontorus": OctagonTorus,
    "nonagontorus": NonagonTorus,
    "fivestartorus": FiveStarTorus,
    "sixstartorus": SixStarTorus,
    "sevenstartorus": SevenStarTorus,
    "eightstartorus": EightStarTorus,
    # Onioned sector polygon prisms
    "onionedtriangleprism": OnionedTrianglePrism,
    "onionedSquareprism": OnionedSquarePrism,
    "onionedpentagonprism": OnionedPentagonPrism,
    "onionedchexagonprism": OnionedHexagonPrism,
    "onionedcheptagonprism": OnionedHeptagonPrism,
    "onionedoctagonprism": OnionedOctagonPrism,
    "onionednonagonprism": OnionedNonagonPrism,
    # Onioned convex sector polygon prisms
    "onionedtriangleconvexprism": OnionedTriangleConvexPrism,
    "onionedSquareconvexprism": OnionedSquareConvexPrism,
    "onionedpentagonconvexprism": OnionedPentagonConvexPrism,
    "onionedchexagonconvexprism": OnionedHexagonConvexPrism,
    "onionedcheptagonconvexprism": OnionedHeptagonConvexPrism,
    "onionedoctagonconvexprism": OnionedOctagonConvexPrism,
    "onionednonagonconvexprism": OnionedNonagonConvexPrism,
    # Onioned concave sector polygon prisms
    "onionedtriangleconcaveprism": OnionedTriangleConcavePrism,
    "onionedSquareconcaveprism": OnionedSquareConcavePrism,
    "onionedpentagonconcaveprism": OnionedPentagonConcavePrism,
    "onionedchexagonconcaveprism": OnionedHexagonConcavePrism,
    "onionedcheptagonconcaveprism": OnionedHeptagonConcavePrism,
    "onionedoctagonconcaveprism": OnionedOctagonConcavePrism,
    "onionednonagonconcaveprism": OnionedNonagonConcavePrism,
    # Onioned cone sector polygon prisms
    "onionedtriangleconeprism": OnionedTriangleConePrism,
    "onionedSquareconeprism": OnionedSquareConePrism,
    "onionedpentagonconeprism": OnionedPentagonConePrism,
    "onionedchexagonconeprism": OnionedHexagonConePrism,
    "onionedcheptagonconeprism": OnionedHeptagonConePrism,
    "onionedoctagonconeprism": OnionedOctagonConePrism,
    "onionednonagonconeprism": OnionedNonagonConePrism,
    # Onioned star polygon prisms
    "onionedfivestarprism": OnionedFiveStarPrism,
    "onionedixstarprism": OnionedSixStarPrism,
    "onionedsevenstarprism": OnionedSevenStarPrism,
    "onionedeightstarprism": OnionedEightStarPrism,
    # Onioned star convex prisms
    "onionedfivestarconvexprism": OnionedFiveStarConvexPrism,
    "onionedixstarconvexprism": OnionedSixStarConvexPrism,
    "onionedsevenstarconvexprism": OnionedSevenStarConvexPrism,
    "onionedeightstarconvexprism": OnionedEightStarConvexPrism,
    # Onioned star concave prisms
    "onionedfvestarconcaveprism": OnionedFiveStarConcavePrism,
    "onionedixstarconcaveprism": OnionedSixStarConcavePrism,
    "onionedsevenstar concaveprism": OnionedSevenStarConcavePrism,
    "onionedeightstar concaveprism": OnionedEightStarConcavePrism,
    # Onioned star cone prisms
    "onionedfvestarconeprism": OnionedFiveStarConePrism,
    "onionedixstarconeprism": OnionedSixStarConePrism,
    "onionedsevenstar coneprism": OnionedSevenStarConePrism,
    "onionedeightstarconeprism": OnionedEightStarConePrism,
    # Union objects
    "spheretriangleunion": SphereTriangleUnion,
    "spheresquareunion": SphereSquareUnion,
    "spherepentagonunion": SpherePentagonUnion,
    "spherecylinderunion": SphereCylinderUnion,
    "threestarrevolutiontriangleunion": ThreeStarRevolutionTriangleUnion,
    "threestarrevolutionsquareunion": ThreeStarRevolutionSquareUnion,
    "threestarrevolutionpentagonunion": ThreeStarRevolutionPentagonUnion,
    "threestarrevolutioncylinderunion": ThreeStarRevolutionCylinderUnion,
    "fourstarrevolutiontriangleunion": FourStarRevolutionTriangleUnion,
    "fourstarrevolutionsquareunion": FourStarRevolutionSquareUnion,
    "fourstarrevolutionpentagonunion": FourStarRevolutionPentagonUnion,
    "fourstarrevolutioncylinderunion": FourStarRevolutionCylinderUnion,
    "fivestarrevolutiontriangleunion": FiveStarRevolutionTriangleUnion,
    "fivestarrevolutionsquareunion": FiveStarRevolutionSquareUnion,
    "fivestarrevolutionpentagonunion": FiveStarRevolutionPentagonUnion,
    "fivestarrevolutioncylinderunion": FiveStarRevolutionCylinderUnion,
}


# カテゴリ別のプリミティブマッピング
PRIMITIVE_CATEGORIES = {
    "basic": [
        "sphere",
        "cylinder",
        "torus",
        "cone",
        "octahedron",
        "convexcylinder",
        "concavecylinder",
        "conecylinder",
    ],
    "revolution": [
        "threestarrevolution",
        "fourstarrevolution",
        "fivestarrevolution",
    ],
    "sector_polygon_prism": [
        "triangleprism",
        "squareprism",
        "pentagonprism",
        "hexagonprism",
        "heptagonprism",
        "octagonprism",
        "nonagonprism",
    ],
    "convex_sector_polygon_prism": [
        "triangleconvexprism",
        "squareconvexprism",
        "pentagonconvexprism",
        "hexagonconvexprism",
        "heptagonconvexprism",
        "octagonconvexprism",
        "nonagonconvexprism",
    ],
    "concave_sector_polygon_prism": [
        "triangleconcaveprism",
        "squareconcaveprism",
        "pentagonconcaveprism",
        "hexagonconcaveprism",
        "heptagonconcaveprism",
        "octagonconcaveprism",
        "nonagonconcaveprism",
    ],
    "cone_sector_polygon_prism": [
        "triangleconeprism",
        "squareconeprism",
        "pentagonconeprism",
        "hexagonconeprism",
        "heptagonconeprism",
        "octagonconeprism",
        "nonagonconeprism",
    ],
    "star_polygon_prism": [
        "fivestarprism",
        "sixstarprism",
        "sevenstarprism",
        "eightstarprism",
    ],
    "star_convex_prism": [
        "fivestarconvexprism",
        "sixstarconvexprism",
        "sevenstarconvexprism",
        "eightstarconvexprism",
    ],
    "star_concave_prism": [
        "fivestarconcaveprism",
        "sixstarconcaveprism",
        "sevenstarconcaveprism",
        "eightstarconcaveprism",
    ],
    "star_cone_prism": [
        "fivestarconeprism",
        "sixstarconeprism",
        "sevenstarconeprism",
        "eightstarconeprism",
    ],
    "sector_polygon_torus": [
        "squaretorus",
        "pentagontorus",
        "hexagontorus",
        "heptagontorus",
        "octagontorus",
        "nonagontorus",
    ],
    "star_torus": [
        "fivestartorus",
        "sixstartorus",
        "sevenstartorus",
        "eightstartorus",
    ],
    "onioned_sector_polygon_prism": [
        "onionedtriangleprism",
        "onionedSquareprism",
        "onionedpentagonprism",
        "onionedchexagonprism",
        "onionedcheptagonprism",
        "onionedoctagonprism",
        "onionednonagonprism",
    ],
    "onioned_convex_sector_polygon_prism": [
        "onionedtriangleconvexprism",
        "onionedSquareconvexprism",
        "onionedpentagonconvexprism",
        "onionedchexagonconvexprism",
        "onionedcheptagonconvexprism",
        "onionedoctagonconvexprism",
        "onionednonagonconvexprism",
    ],
    "onioned_concave_sector_polygon_prism": [
        "onionedtriangleconcaveprism",
        "onionedSquareconcaveprism",
        "onionedpentagonconcaveprism",
        "onionedchexagonconcaveprism",
        "onionedcheptagonconcaveprism",
        "onionedoctagonconcaveprism",
        "onionednonagonconcaveprism",
    ],
    "onioned_cone_sector_polygon_prism": [
        "onionedtriangleconeprism",
        "onionedSquareconeprism",
        "onionedpentagonconeprism",
        "onionedchexagonconeprism",
        "onionedcheptagonconeprism",
        "onionedoctagonconeprism",
        "onionednonagonconeprism",
    ],
    "onioned_star_polygon_prism": [
        "onionedfivestarprism",
        "onionedixstarprism",
        "onionedsevenstarprism",
        "onionedeightstarprism",
    ],
    "onioned_star_convex_prism": [
        "onionedfivestarconvexprism",
        "onionedixstarconvexprism",
        "onionedsevenstarconvexprism",
        "onionedeightstarconvexprism",
    ],
    "onioned_star_concave_prism": [
        "onionedfvestarconcaveprism",
        "onionedixstarconcaveprism",
        "onionedsevenstar concaveprism",
        "onionedeightstar concaveprism",
    ],
    "onioned_star_cone_prism": [
        "onionedfvestarconeprism",
        "onionedixstarconeprism",
        "onionedsevenstar coneprism",
        "onionedeightstarconeprism",
    ],
    "union": [
        "spheretriangleunion",
        "spheresquareunion",
        "spherepentagonunion",
        "spherecylinderunion",
        "threestarrevolutiontriangleunion",
        "threestarrevolutionsquareunion",
        "threestarrevolutionpentagonunion",
        "threestarrevolutioncylinderunion",
        "fourstarrevolutiontriangleunion",
        "fourstarrevolutionsquareunion",
        "fourstarrevolutionpentagonunion",
        "fourstarrevolutioncylinderunion",
        "fivestarrevolutiontriangleunion",
        "fivestarrevolutionsquareunion",
        "fivestarrevolutionpentagonunion",
        "fivestarrevolutioncylinderunion",
    ],
}


# デフォルトで使用するプリミティブのリスト
DEFAULT_PRIMITIVES = [
    "sphere",
    "cylinder",
    "torus",
    "cone",
    "octahedron",
    "convexcylinder",
    "concavecylinder",
    "conecylinder",
    "threestarrevolution",
    "fourstarrevolution",
    "fivestarrevolution",
    "triangleprism",
    "squareprism",
    "pentagonprism",
    "hexagonprism",
    "heptagonprism",
    "octagonprism",
    "nonagonprism",
    "triangleconvexprism",
    "squareconvexprism",
    "pentagonconvexprism",
    "hexagonconvexprism",
    "heptagonconvexprism",
    "octagonconvexprism",
    "nonagonconvexprism",
    "triangleconcaveprism",
    "squareconcaveprism",
    "pentagonconcaveprism",
    "hexagonconcaveprism",
    "heptagonconcaveprism",
    "octagonconcaveprism",
    "nonagonconcaveprism",
    "triangleconeprism",
    "squareconeprism",
    "pentagonconeprism",
    "hexagonconeprism",
    "heptagonconeprism",
    "octagonconeprism",
    "nonagonconeprism",
    "fivestarprism",
    "sixstarprism",
    "sevenstarprism",
    "eightstarprism",
    "fivestarconvexprism",
    "sixstarconvexprism",
    "sevenstarconvexprism",
    "eightstarconvexprism",
    "fivestarconcaveprism",
    "sixstarconcaveprism",
    "sevenstarconcaveprism",
    "eightstarconcaveprism",
    "fivestarconeprism",
    "sixstarconeprism",
    "sevenstarconeprism",
    "eightstarconeprism",
    "squaretorus",
    "pentagontorus",
    "hexagontorus",
    "heptagontorus",
    "octagontorus",
    "nonagontorus",
    "fivestartorus",
    "sixstartorus",
    "sevenstartorus",
    "eightstartorus",
    # Onioned primitives
    "onionedtriangleprism",
    "onionedSquareprism",
    "onionedpentagonprism",
    "onionedchexagonprism",
    "onionedcheptagonprism",
    "onionedoctagonprism",
    "onionednonagonprism",
    "onionedtriangleconvexprism",
    "onionedSquareconvexprism",
    "onionedpentagonconvexprism",
    "onionedchexagonconvexprism",
    "onionedcheptagonconvexprism",
    "onionedoctagonconvexprism",
    "onionednonagonconvexprism",
    "onionedtriangleconcaveprism",
    "onionedSquareconcaveprism",
    "onionedpentagonconcaveprism",
    "onionedchexagonconcaveprism",
    "onionedcheptagonconcaveprism",
    "onionedoctagonconcaveprism",
    "onionednonagonconcaveprism",
    "onionedtriangleconeprism",
    "onionedSquareconeprism",
    "onionedpentagonconeprism",
    "onionedchexagonconeprism",
    "onionedcheptagonconeprism",
    "onionedoctagonconeprism",
    "onionednonagonconeprism",
    "onionedfivestarprism",
    "onionedixstarprism",
    "onionedsevenstarprism",
    "onionedeightstarprism",
    "onionedfivestarconvexprism",
    "onionedixstarconvexprism",
    "onionedsevenstarconvexprism",
    "onionedeightstarconvexprism",
    "onionedfvestarconcaveprism",
    "onionedixstarconcaveprism",
    "onionedsevenstar concaveprism",
    "onionedeightstar concaveprism",
    "onionedfvestarconeprism",
    "onionedixstarconeprism",
    "onionedsevenstar coneprism",
    "onionedeightstarconeprism",
    # Union primitives
    "spheretriangleunion",
    "spheresquareunion",
    "spherepentagonunion",
    "spherecylinderunion",
    "threestarrevolutiontriangleunion",
    "threestarrevolutionsquareunion",
    "threestarrevolutionpentagonunion",
    "threestarrevolutioncylinderunion",
    "fourstarrevolutiontriangleunion",
    "fourstarrevolutionsquareunion",
    "fourstarrevolutionpentagonunion",
    "fourstarrevolutioncylinderunion",
    "fivestarrevolutiontriangleunion",
    "fivestarrevolutionsquareunion",
    "fivestarrevolutionpentagonunion",
    "fivestarrevolutioncylinderunion",
]


def get_primitive_choices():
    """Get all available primitive choices for argument parser."""
    return list(ALL_PRIMITIVES.keys())


def get_category_choices():
    """Get all available category choices for argument parser."""
    return list(PRIMITIVE_CATEGORIES.keys())


def select_primitives(primitives=None, categories=None, num_classes=None):
    """
    Select primitives based on the given criteria.

    Args:
        primitives: List of specific primitive names to use
        categories: List of category names to use
        num_classes: Number of classes to randomly select

    Returns:
        tuple: (selected_primitive_names, selected_primitives_dict)
    """
    import random

    # 使用するプリミティブを選択（カテゴリまたは個別指定）
    if categories is not None:
        # カテゴリが指定された場合、該当するプリミティブを収集
        selected_primitive_names = []
        for category in categories:
            if category in PRIMITIVE_CATEGORIES:
                selected_primitive_names.extend(PRIMITIVE_CATEGORIES[category])
            else:
                print(f"Warning: Unknown category '{category}' ignored.")

        # 重複を除去
        selected_primitive_names = list(set(selected_primitive_names))
    elif primitives is not None:
        # 個別のプリミティブが指定された場合
        selected_primitive_names = primitives
    else:
        # デフォルトは全て
        selected_primitive_names = list(ALL_PRIMITIVES.keys())

    # num_classesが指定された場合、ランダムに選択
    if num_classes is not None and num_classes > 0:
        if len(selected_primitive_names) > num_classes:
            # 指定されたクラス数にランダムに削減
            print(
                f"Randomly selecting {num_classes} classes from {len(selected_primitive_names)} available classes."
            )
            selected_primitive_names = random.sample(
                selected_primitive_names, num_classes
            )
        elif len(selected_primitive_names) < num_classes:
            print(
                f"Warning: Requested {num_classes} classes, but only {len(selected_primitive_names)} available. Using all available classes."
            )

    print(
        f"Selected primitives ({len(selected_primitive_names)}): {', '.join(sorted(selected_primitive_names))}"
    )

    # 選択されたプリミティブのみを使用
    selected_primitives = {
        name: ALL_PRIMITIVES[name]
        for name in selected_primitive_names
        if name in ALL_PRIMITIVES
    }

    return selected_primitive_names, selected_primitives
