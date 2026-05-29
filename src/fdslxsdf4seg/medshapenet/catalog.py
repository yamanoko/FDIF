"""Catalog of MedShapeNetCore datasets and organ keys exposed as FDIF classes.

KITS is intentionally excluded (reserved for finetuning evaluation).
ACDC and AMOS are not present in MedShapeNetCore at all.

Each catalog entry maps a stable FDIF primitive name (e.g. "msn_liver")
to its source (dataset_name, organ_key). For datasets that contain a
single anatomy per sample (no organ-key dict), organ_key is None and
the loader uses every sample in the bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class CatalogEntry:
    primitive_name: str          # e.g. "msn_liver"
    class_name: str              # e.g. "MsnLiver" (dynamic class name)
    dataset: str                 # MedShapeNetCore dataset id
    organ_key: Optional[str]     # sub-organ key inside the .npz, or None
    category: str                # FDIF category tag, e.g. "msn_flare"


# Whitelist of datasets we use. Order matters only for display.
#
# All these datasets work via the loader's per-sample extracted cache
# (`fdslxsdf4seg.medshapenet.loader._extract_bundle_to_cache`). The
# *first* time each dataset is touched, the full Zenodo .npz must be
# unpickled into RAM once. After that, only small per-sample files are
# loaded on demand.
#
# Approximate peak RAM during the one-time extraction:
#   ASOCA, AVT  -> < 500 MB     (small)
#   FLARE       -> ~ 2-3 GB
#   ThoracicAorta_Saitta -> ~ 3-5 GB
#   PULMONARY   -> ~ 5-8 GB     (largest)
#
# If a host can't fit the one-time extraction, run
#   python -m fdslxsdf4seg.medshapenet.extract <dataset>
# on a bigger machine and copy `cache/medshapenet/extracted/<dataset>/`
# back. Day-to-day data generation does not need that headroom.
ACTIVE_DATASETS: List[str] = [
    "ASOCA",                # 41.8 MB  /  40 samples  / coronary arteries
    "AVT",                  # 115 MB   /  42 samples  / aortic vessel tree
    "ThoracicAorta_Saitta", # 515 MB   / 500 samples  / thoracic aorta
    "FLARE",                # 555 MB   / 650 samples  / 13 abdominal organs
    "PULMONARY",            # 1.14 GB  / 2397 samples / airway + lung vessels
]

# Datasets explicitly excluded (reserved for finetuning evaluation).
# Adding any of these to ACTIVE_DATASETS would risk train/test leakage.
EXCLUDED_DATASETS: List[str] = [
    "KITS",
]


# Per-dataset organ keys, reconciled against actual .npz contents.
# The loader will silently skip entries that turn out to be missing.
DATASET_ORGAN_KEYS: Dict[str, List[Optional[str]]] = {
    # FLARE22: 13 abdominal organs (verified against the .npz top-level keys).
    "FLARE": [
        "liver",
        "right_kidney",
        "left_kidney",
        "spleen",
        "pancreas",
        "aorta",
        "inferior_vena_cava",
        "right_adrenal_gland",
        "left_adrenal_gland",
        "gallbladder",
        "esophagus",
        "stomach",
        "duodenum",
    ],
    # PULMONARY: organ keys per MedShapeNet info — airway / artery / vein / lung.
    "PULMONARY": [
        "airway",
        "artery",
        "vein",
        "lung",
    ],
    # Single-anatomy datasets: one mesh per sample, no organ-key dict.
    "ASOCA": [None],                  # coronary arteries
    "ThoracicAorta_Saitta": [None],   # thoracic aorta
    "AVT": [None],                    # aortic vessel tree
}


# Friendly display names per organ key (for class names and logs).
ORGAN_DISPLAY: Dict[str, str] = {
    "liver": "Liver",
    "right_kidney": "RightKidney",
    "left_kidney": "LeftKidney",
    "spleen": "Spleen",
    "pancreas": "Pancreas",
    "aorta": "Aorta",
    "inferior_vena_cava": "InferiorVenaCava",
    "right_adrenal_gland": "RightAdrenalGland",
    "left_adrenal_gland": "LeftAdrenalGland",
    "gallbladder": "Gallbladder",
    "esophagus": "Esophagus",
    "stomach": "Stomach",
    "duodenum": "Duodenum",
    "airway": "Airway",
    "pulmonary_artery": "PulmonaryArtery",
    "pulmonary_vein": "PulmonaryVein",
    "vein": "Vein",
    "artery": "Artery",
}

DATASET_DISPLAY: Dict[str, str] = {
    "ASOCA": "CoronaryArtery",
    "ThoracicAorta_Saitta": "ThoracicAorta",
    "AVT": "AorticVesselTree",
}


def _category_for_dataset(dataset: str) -> str:
    return f"msn_{dataset.lower()}"


def build_catalog() -> List[CatalogEntry]:
    """Build the full list of catalog entries from declarations above."""
    entries: List[CatalogEntry] = []
    for dataset in ACTIVE_DATASETS:
        organ_keys = DATASET_ORGAN_KEYS.get(dataset, [None])
        for organ_key in organ_keys:
            if organ_key is None:
                display = DATASET_DISPLAY.get(dataset, dataset)
                primitive_name = f"msn_{display.lower()}"
                class_name = f"Msn{display}"
            else:
                display = ORGAN_DISPLAY.get(
                    organ_key,
                    "".join(part.capitalize() for part in organ_key.split("_")),
                )
                primitive_name = f"msn_{organ_key}"
                class_name = f"Msn{display}"
            entries.append(
                CatalogEntry(
                    primitive_name=primitive_name,
                    class_name=class_name,
                    dataset=dataset,
                    organ_key=organ_key,
                    category=_category_for_dataset(dataset),
                )
            )
    return entries


def build_categories(entries: List[CatalogEntry]) -> Dict[str, List[str]]:
    """Group primitive names by category tag, plus a global `msn_all` bucket."""
    by_cat: Dict[str, List[str]] = {}
    for e in entries:
        by_cat.setdefault(e.category, []).append(e.primitive_name)
    by_cat["msn_all"] = [e.primitive_name for e in entries]
    return by_cat
