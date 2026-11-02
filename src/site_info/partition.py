"""Partition a target site list against an already-extracted one.

The combined corpus overlaps the corpus whose gridded features were extracted in
January: 165 of its 226 sites carry site codes that already have a row in every
``data/raw/extracted_data/*/sapwood/`` product, and 61 do not.  Because every one
of those products is a plain per-site row concatenation keyed on ``site_name``,
extracting only the 61 and concatenating is *equal to* a full re-extraction, not
an approximation of one.

This module names that partition once so the delta run and the later assembly
agree on it by construction rather than by two people computing it the same way.

Usage
-----
    python -m src.site_info.partition \
        --target   data/raw/0.1.5/0.1.5/csv/sapwood_combined/site_info.csv \
        --reference data/raw/0.1.5/0.1.5/csv/sapwood/site_info.csv \
        --write-new sapwood_combined_new_sites.txt
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SITE_NAME_COLUMN = "site_name"
ENCODING = "utf-8-sig"


class PartitionError(RuntimeError):
    """The two site lists could not be compared."""


@dataclass(frozen=True)
class Partition:
    """How a target site list sits against an already-extracted reference."""

    shared: tuple[str, ...]
    """In both: reuse the reference's extracted rows."""

    new: tuple[str, ...]
    """In the target only: these need a fresh extraction."""

    reference_only: tuple[str, ...]
    """In the reference only: extracted rows with nowhere to go in the target."""

    def summary(self) -> str:
        return (
            f"target = {len(self.shared) + len(self.new)} "
            f"({len(self.shared)} reusable + {len(self.new)} to extract); "
            f"reference-only = {len(self.reference_only)}"
        )


def site_names(csv_path: Path) -> list[str]:
    """Read the ``site_name`` column of a site_info-shaped CSV.

    Reads with ``utf-8-sig`` and strips whitespace: these files have been
    written on Windows, so a trailing ``\\r`` on the last column is routine and
    turns every set comparison silently empty.
    """
    path = Path(csv_path)
    if not path.exists():
        raise PartitionError(f"no such file: {path}")

    frame = pd.read_csv(path, encoding=ENCODING)
    if SITE_NAME_COLUMN not in frame.columns:
        raise PartitionError(f"{path} has no {SITE_NAME_COLUMN!r} column (columns: {list(frame.columns)})")

    names = [str(name).strip() for name in frame[SITE_NAME_COLUMN]]
    blank = [i for i, name in enumerate(names) if not name or name == "nan"]
    if blank:
        raise PartitionError(f"{path}: blank {SITE_NAME_COLUMN} at rows {blank[:5]}")

    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise PartitionError(f"{path}: duplicate {SITE_NAME_COLUMN}: {duplicated}")

    return names


def partition(target: Sequence[str], reference: Sequence[str]) -> Partition:
    """Split ``target`` into what ``reference`` already covers and what it does not."""
    target_set = set(target)
    reference_set = set(reference)
    return Partition(
        shared=tuple(sorted(target_set & reference_set)),
        new=tuple(sorted(target_set - reference_set)),
        reference_only=tuple(sorted(reference_set - target_set)),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", required=True, help="site_info.csv to extract for")
    parser.add_argument("--reference", required=True, help="site_info.csv already extracted")
    parser.add_argument("--write-new", default=None, help="write the to-extract site codes here")
    args = parser.parse_args(argv)

    target = site_names(Path(args.target))
    reference = site_names(Path(args.reference))
    result = partition(target, reference)

    print(f"target    : {args.target} ({len(target)} sites)")
    print(f"reference : {args.reference} ({len(reference)} sites)")
    print(f"\n{result.summary()}\n")
    print(f"to extract ({len(result.new)}):")
    for name in result.new:
        print(f"  {name}")
    if result.reference_only:
        print(f"\nreference-only ({len(result.reference_only)}):")
        for name in result.reference_only:
            print(f"  {name}")

    if args.write_new:
        out = Path(args.write_new)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(result.new) + "\n", encoding="utf-8")
        print(f"\nwrote {len(result.new)} site codes to {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
