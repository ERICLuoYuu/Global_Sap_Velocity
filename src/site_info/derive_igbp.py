"""Fill in ``si_igbp`` for the sites the archive never gave one.

``si_igbp`` is absent at exactly the 23 internal sites, so ``pft_resolution``
falls back to MODIS there -- and MODIS is least reliable at the canopies those
sites happen to have.  This module derives the class from the stand and species
metadata the providers already filled in (see ``igbp_from_metadata``) and offers
it as a tier between the provider value and the raster.

Two ways to use it, deliberately separate
-----------------------------------------
``derive_for_scale`` is a pure read.  The merge calls it and passes the result
to ``pft_resolution.resolve_pft`` as the ``derived`` argument, so the value is
recomputed from metadata on every run.  Nothing is stored, so nothing can go
stale and a Stage-1 regeneration has nothing to revert -- which is the failure
mode the Koppen ocean repair taught (a patcher-only fix that the next
extraction run silently undid).

``--apply`` is the other way: it writes the derived code into ``si_igbp`` in the
site metadata, for the consumers that read that column directly rather than
going through the merge (``src/paper_replicate``, ``sap_statistics_analyzer``,
``gleam_diagnostics``).  It refuses to touch a site whose provider value is
present, re-reads every file afterwards and fails if any OTHER field moved.

Usage
-----
    # report only, nothing written
    python -m src.site_info.derive_igbp --scale sapwood_combined

    # also write si_igbp into the site metadata, and a provenance table
    python -m src.site_info.derive_igbp --scale sapwood_combined --apply \
        --report outputs/site_info/igbp_provenance.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import sys

from src.site_info.igbp_from_metadata import (
    TIER_NONE,
    derive_igbp,
    is_absent,
    summarise,
)

#: What ``derive_for_scale`` reports for a site the provider already labelled.
TIER_PROVIDER = "provider"

SITE_SUFFIX = "_site_md.csv"
IGBP_COLUMN = "si_igbp"
FOREST_TYPE_COLUMN = "st_forest_type"
SPECIES_NAME_COLUMN = "sp_name"
SPECIES_HABIT_COLUMN = "sp_leaf_habit"

#: Written beside each site_md the first time --apply changes it.
BACKUP_SUFFIX = ".pre_igbp_apply"

#: csv writes multi-line and comma-bearing fields back with quotes; the site
#: metadata carries both (contact names, LTER remarks).  QUOTE_MINIMAL
#: reproduces exactly the quoting those fields require, and the post-write
#: verification proves it rather than assuming it.
_QUOTING = csv.QUOTE_MINIMAL


class ApplyRefusedError(RuntimeError):
    """A write would have changed something it was not asked to change."""


def _read_table(path: str) -> tuple[list[str], list[dict]]:
    """``(fieldnames, rows)`` with every value as text and nothing coerced."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        return fieldnames, list(reader)


def _write_table(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=_QUOTING)
        writer.writeheader()
        writer.writerows(rows)


def site_codes(csv_dir: str) -> list[str]:
    """Every site in a scale directory, by exact suffix strip.

    Never a prefix glob: ``CHE_DAV`` is a prefix of ``CHE_DAV_SEE`` and
    ``SWE_NOR`` of ``SWE_NOR_ST1_AF1``.
    """
    paths = glob.glob(os.path.join(csv_dir, "*" + SITE_SUFFIX))
    return sorted(os.path.basename(p)[: -len(SITE_SUFFIX)] for p in paths)


def _first_value(path: str, column: str) -> str:
    """The first row of a one-row metadata file, or "" if it has no such column."""
    if not os.path.exists(path):
        return ""
    fieldnames, rows = _read_table(path)
    if column not in fieldnames or not rows:
        return ""
    return str(rows[0].get(column) or "").strip()


def _species_rows(csv_dir: str, site: str) -> list[tuple[str, str]]:
    path = os.path.join(csv_dir, f"{site}_species_md.csv")
    if not os.path.exists(path):
        return []
    fieldnames, rows = _read_table(path)
    if SPECIES_NAME_COLUMN not in fieldnames:
        return []
    pairs = []
    for row in rows:
        name = str(row.get(SPECIES_NAME_COLUMN) or "").strip()
        if is_absent(name):
            continue
        pairs.append((name, str(row.get(SPECIES_HABIT_COLUMN) or "").strip()))
    return pairs


def derive_for_scale(csv_dir: str) -> dict[str, tuple[str | None, str, str]]:
    """``{site: (code, tier, evidence)}`` for every site in a scale directory.

    Sites the provider already labelled come back with their own value and tier
    ``"provider"``, so a caller can tell at a glance which population it is
    looking at.  Pure read -- opens nothing for writing.
    """
    resolved: dict[str, tuple[str | None, str, str]] = {}
    for site in site_codes(csv_dir):
        provider = _first_value(os.path.join(csv_dir, site + SITE_SUFFIX), IGBP_COLUMN)
        if not is_absent(provider):
            resolved[site] = (provider, TIER_PROVIDER, "si_igbp supplied by the archive")
            continue
        forest_type = _first_value(os.path.join(csv_dir, f"{site}_stand_md.csv"), FOREST_TYPE_COLUMN)
        resolved[site] = derive_igbp(forest_type, _species_rows(csv_dir, site))
    return resolved


def derived_only(rows: dict[str, tuple[str | None, str, str]]) -> dict[str, str]:
    """``{site: code}`` for the sites this module actually decided.

    Shaped for ``pft_resolution.resolve_pft``: provider sites are excluded
    because the provider value already wins there, and unresolved sites are
    excluded so the MODIS fallback can still act as the last resort.
    """
    return {
        site: code
        for site, (code, tier, _) in rows.items()
        if code is not None and tier not in (TIER_PROVIDER, TIER_NONE)
    }


def apply_to_site_md(csv_dir: str, rows: dict[str, tuple[str | None, str, str]]) -> list[tuple[str, str]]:
    """Write derived codes into ``si_igbp``; return the ``(site, code)`` written.

    Refuses on three counts, all of which mean the caller is not doing what it
    thinks: a site whose provider value is present, a file with no ``si_igbp``
    column, and any field other than ``si_igbp`` differing after the write.
    """
    planned = derived_only(rows)
    written: list[tuple[str, str]] = []

    for site, code in sorted(planned.items()):
        path = os.path.join(csv_dir, site + SITE_SUFFIX)
        fieldnames, table = _read_table(path)
        if IGBP_COLUMN not in fieldnames:
            raise ApplyRefusedError(f"{path} has no {IGBP_COLUMN!r} column")
        if not table:
            raise ApplyRefusedError(f"{path} has a header but no row")

        before = [dict(row) for row in table]
        current = str(table[0].get(IGBP_COLUMN) or "").strip()
        if current == code:
            continue
        if not is_absent(current):
            raise ApplyRefusedError(
                f"{site} already carries {IGBP_COLUMN}={current!r} -- refusing to overwrite a provider value"
            )

        # Snapshot once, then never again: a second run must not overwrite a
        # good backup with an already-modified file.
        backup = path + BACKUP_SUFFIX
        if not os.path.exists(backup):
            shutil.copy2(path, backup)

        table[0][IGBP_COLUMN] = code
        _write_table(path, fieldnames, table)

        # Verify rather than trust: csv re-quotes on write, and a field with an
        # embedded comma or newline is exactly where that could go wrong.
        after_fields, after = _read_table(path)
        if after_fields != fieldnames or len(after) != len(before):
            raise ApplyRefusedError(f"{path}: shape changed on write")
        for index, (old_row, new_row) in enumerate(zip(before, after)):
            for column in fieldnames:
                if column == IGBP_COLUMN and index == 0:
                    continue
                if (old_row.get(column) or "") != (new_row.get(column) or ""):
                    raise ApplyRefusedError(
                        f"{path}: {column!r} changed on write ({old_row.get(column)!r} -> {new_row.get(column)!r})"
                    )
        written.append((site, code))
    return written


def write_report(path: str, rows: dict[str, tuple[str | None, str, str]]) -> None:
    """A provenance table: which tier decided each site, and on what evidence."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, quoting=_QUOTING)
        writer.writerow(["site_name", "si_igbp", "tier", "evidence"])
        for site in sorted(rows):
            code, tier, evidence = rows[site]
            writer.writerow([site, code or "", tier, evidence])


def default_report_path(scale: str) -> str:
    """Where the provenance table lands when ``--report`` is not given.

    Deterministic and under the outputs root, not wherever the caller happened
    to be standing.  The derived codes are recomputed at merge time and stored
    nowhere else, so without this file the only record of what a run decided --
    and on what evidence -- would be the stdout of that run.
    """
    os.environ.setdefault("ANALYSIS_SCALE", scale)
    from path_config import PathConfig

    return str(PathConfig().base_output_dir / "site_info" / scale / "igbp_provenance.csv")


def _default_csv_dir(scale: str) -> str:
    """The scale directory PathConfig points at, so the CLI needs no path."""
    os.environ.setdefault("ANALYSIS_SCALE", scale)
    from path_config import PathConfig

    return str(PathConfig().site_info_path.parent)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scale", default="sapwood_combined")
    parser.add_argument("--csv-dir", default=None, help="overrides --scale")
    parser.add_argument("--apply", action="store_true", help="write si_igbp into site_md")
    parser.add_argument(
        "--report",
        default=None,
        help="provenance CSV; defaults to a path under the outputs root",
    )
    args = parser.parse_args(argv)

    csv_dir = args.csv_dir or _default_csv_dir(args.scale)
    if not os.path.isdir(csv_dir):
        parser.error(f"not a directory: {csv_dir}")
    print(f"scale directory: {csv_dir}")

    rows = derive_for_scale(csv_dir)
    needed = {site: value for site, value in rows.items() if value[1] != TIER_PROVIDER}
    print(f"sites: {len(rows)}   provider-labelled: {len(rows) - len(needed)}   derived here: {len(needed)}")
    print("")
    for line in summarise(needed):
        print(line)

    unresolved = sorted(site for site, value in needed.items() if value[0] is None)
    if unresolved:
        print("")
        print(f"UNRESOLVED (these would still fall back to MODIS): {unresolved}")

    report = args.report or default_report_path(args.scale)
    write_report(report, rows)
    print("")
    print(f"provenance written: {report}")

    if args.apply:
        written = apply_to_site_md(csv_dir, rows)
        print(f"\nsi_igbp written at {len(written)} site(s)")
        for site, code in written:
            print(f"  {site} -> {code}")

    return 1 if unresolved else 0


if __name__ == "__main__":
    sys.exit(main())
