"""MedShapeNet integration for FDIF.

Provides SDFObject-compatible wrappers around real medical mesh data
from MedShapeNetCore. Designed to run on both Windows and Linux
without OpenGL dependencies (uses pysdf as the SDF backend).
"""

from fdslxsdf4seg.medshapenet.registry import (
    inject,
    get_medshapenet_categories,
    get_medshapenet_primitives,
)

__all__ = ["inject", "get_medshapenet_categories", "get_medshapenet_primitives"]
