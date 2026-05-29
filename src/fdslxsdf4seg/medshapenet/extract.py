"""CLI to pre-extract MedShapeNet dataset bundles to the per-sample cache.

Use this when the runtime host is too small for the one-time pickle
load (e.g. PULMONARY needs ~5-8 GB peak RAM). Run on a larger host,
then copy `cache/medshapenet/extracted/<dataset>/` to the small host.

Usage:
    python -m fdslxsdf4seg.medshapenet.extract <dataset> [<dataset> ...]
    python -m fdslxsdf4seg.medshapenet.extract --all
    python -m fdslxsdf4seg.medshapenet.extract --list
    python -m fdslxsdf4seg.medshapenet.extract --force <dataset>
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from fdslxsdf4seg.medshapenet import catalog as _catalog
from fdslxsdf4seg.medshapenet.loader import (
    clear_extracted,
    ensure_extracted,
)


def _all_known_datasets() -> List[str]:
    return list(_catalog.ACTIVE_DATASETS)


def _resolve_targets(args: argparse.Namespace) -> List[str]:
    if args.all:
        return _all_known_datasets()
    if not args.datasets:
        print(
            "Error: specify at least one dataset, or use --all / --list.",
            file=sys.stderr,
        )
        sys.exit(2)
    known = set(_all_known_datasets())
    unknown = [d for d in args.datasets if d not in known]
    if unknown:
        print(
            f"Error: unknown dataset(s): {unknown}. "
            f"Known: {sorted(known)}.",
            file=sys.stderr,
        )
        sys.exit(2)
    return list(args.datasets)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m fdslxsdf4seg.medshapenet.extract",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Dataset id(s) to extract (e.g. PULMONARY ThoracicAorta_Saitta).",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Extract every dataset in catalog.ACTIVE_DATASETS.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print the active dataset list and exit.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Delete an existing extracted cache and rebuild it.",
    )
    args = parser.parse_args(argv)

    if args.list:
        for d in _all_known_datasets():
            print(d)
        return 0

    targets = _resolve_targets(args)
    print(f"Extracting datasets: {targets}")
    for d in targets:
        if args.force:
            print(f"[force] clearing existing extracted cache for {d}")
            clear_extracted(d)
        try:
            ensure_extracted(d)
        except MemoryError as e:
            print(f"[ERROR] {d}: {e}", file=sys.stderr)
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {d}: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
    print("All requested extractions complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
