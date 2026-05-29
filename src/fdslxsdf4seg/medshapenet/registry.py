"""Dynamic class generation + registry injection.

For every catalog entry we generate a subclass of `MedShapeNetSDFObject`
with `dataset` and `organ_key` bound as class attributes. Those classes
behave like any other FDIF primitive (callable as `Cls(grid_size, device, ...)`),
so they slot into `ALL_PRIMITIVES` and `PRIMITIVE_CATEGORIES` without any
special-casing downstream.
"""

from __future__ import annotations

from typing import Dict, List, Type

from fdslxsdf4seg.medshapenet.catalog import (
    CatalogEntry,
    build_catalog,
    build_categories,
)
from fdslxsdf4seg.medshapenet.medshapenet_sdf import MedShapeNetSDFObject


def _make_class(entry: CatalogEntry) -> Type[MedShapeNetSDFObject]:
    """Build a subclass of MedShapeNetSDFObject with bound dataset/organ_key."""
    return type(
        entry.class_name,
        (MedShapeNetSDFObject,),
        {
            "dataset": entry.dataset,
            "organ_key": entry.organ_key,
            "__doc__": (
                f"MedShapeNet primitive: dataset={entry.dataset}, "
                f"organ_key={entry.organ_key}."
            ),
        },
    )


_CATALOG: List[CatalogEntry] = build_catalog()
_PRIMITIVES: Dict[str, Type[MedShapeNetSDFObject]] = {
    e.primitive_name: _make_class(e) for e in _CATALOG
}
_CATEGORIES: Dict[str, List[str]] = build_categories(_CATALOG)


def get_medshapenet_primitives() -> Dict[str, Type[MedShapeNetSDFObject]]:
    """Return {primitive_name: class} for all MedShapeNet entries."""
    return dict(_PRIMITIVES)


def get_medshapenet_categories() -> Dict[str, List[str]]:
    """Return {category_name: [primitive_name, ...]} for MedShapeNet."""
    return {k: list(v) for k, v in _CATEGORIES.items()}


def inject(
    all_primitives: Dict[str, type],
    primitive_categories: Dict[str, List[str]],
) -> None:
    """Merge MedShapeNet primitives + categories into existing FDIF registries.

    Idempotent: re-running overwrites prior MedShapeNet entries but does
    not duplicate them.
    """
    for name, cls in _PRIMITIVES.items():
        all_primitives[name] = cls
    for cat, names in _CATEGORIES.items():
        primitive_categories[cat] = list(names)
