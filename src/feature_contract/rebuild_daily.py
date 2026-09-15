"""Rebuild a merged daily tree under the servable feature contract.

Reads a Stage-4 config directory (``<config>/daily/*_daily.csv`` plus ``RUN_PROVENANCE.json``)
and writes a sibling tree in which ONLY the normalised soil-moisture families change::

    volumetric_soil_water_layer_L{,_min,_max,_sum}          raw / sigma_clim
    volumetric_soil_water_layer_L_zscore{,_min,_max,_sum}    (raw - mu_clim) / sigma_clim

Every other cell is copied as TEXT, byte for byte. It is never parsed and re-written:
pandas' default float parser is not round-trip exact (it reads ``'206.45833333333334'`` as
``206.45833333333331``) and its writer shortens digits, so a ``read_csv``/``to_csv`` cycle
moved 4-54 of 223 untouched columns per site by 1-2 ulp (job 46722293). Copying text keeps
the rebuilt tree identical to the source for ANY reader, the trainer's included.

The rewritten cells are computed from raw values parsed exactly (``float_precision=
"round_trip"``) and written with Python's shortest round-trip ``repr``; NaN is written empty,
as pandas writes a missing value. After each file is written, its untouched cells are
compared with the source as strings.

Exactness of the values: both transforms are affine in raw, so they commute with the daily
mean, min and max the merge took over hourly values. The ``_sum`` of the z-score needs the
hour count, ``n = raw_sum / raw_mean``; where ``raw_mean`` is 0 it is undefined (NaN).

The tree is built under ``<dst>.partial`` and renamed at the end, so a crash never leaves a
destination that looks complete.

CLI::

    python -m src.feature_contract.rebuild_daily --src <config_dir> --dst <new_config_dir> --cube <cube.nc>
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from src.feature_contract.definitions import (
    ERA5_HOURLY_MEAN_PRECIP_COL,
    PROVENANCE_FILE,
    PROVENANCE_KEY,
    SERVABLE_CONTRACT,
    SERVABLE_RAIN_EVENT_MM_PER_DAY,
    SIDECAR_FILE,
)
from src.feature_contract.swvl_climatology import (
    DEFAULT_MAX_RADIUS_CELLS,
    MIN_VALID_SIGMA,
    CellStats,
    load_climatology,
    lookup_cell,
    normalise_swvl,
)

LOGGER = logging.getLogger("rebuild_daily")

CONTRACT_NAME = SERVABLE_CONTRACT
LAYERS = (1, 2, 3, 4)
STAT_SUFFIXES = ("", "_min", "_max", "_sum")
SWVL = "volumetric_soil_water_layer_{layer}"
LAT_COL = "latitude_x"
LON_COL = "longitude_x"
DAILY_SUFFIX = "_daily.csv"
AGGREGATE_MARKER = "all_biomes_merged"
CELLS_FILE = "feature_contract_cells.csv"
MANIFEST_FILE = "feature_contract_manifest.json"
PARTIAL_SUFFIX = ".partial"
CSV_LINETERMINATOR = "\n"


def site_coordinates(df: pd.DataFrame, site: str):
    """Median site latitude/longitude -- the same reduction the trainer applies."""
    for col in (LAT_COL, LON_COL):
        if col not in df.columns or df[col].dropna().empty:
            raise ValueError(f"{site}: no usable {col!r} column")
    return float(df[LAT_COL].median()), float(df[LON_COL].median())


def touched_columns(columns: Sequence[str], layers: Sequence[int]) -> list[str]:
    """The bare and z-score family columns that will be rewritten: those whose raw companion exists."""
    present = set(columns)
    touched = []
    for layer in layers:
        base = SWVL.format(layer=layer)
        for suffix in STAT_SUFFIXES:
            if f"{base}_raw{suffix}" not in present:
                continue
            for col in (f"{base}{suffix}", f"{base}_zscore{suffix}"):
                if col in present:
                    touched.append(col)
    return touched


def orphan_family_columns(columns: Sequence[str], layers: Sequence[int]) -> list[str]:
    """Family columns with no raw companion; they would silently keep their legacy per-site values."""
    present = set(columns)
    orphans = []
    for layer in layers:
        base = SWVL.format(layer=layer)
        for suffix in STAT_SUFFIXES:
            if f"{base}_raw{suffix}" in present:
                continue
            orphans.extend(col for col in (f"{base}{suffix}", f"{base}_zscore{suffix}") if col in present)
    return orphans


def rebuild_site_frame(df: pd.DataFrame, cells: dict[int, CellStats], site: str) -> pd.DataFrame:
    """Return a new frame with the soil-moisture families renormalised by the climatology."""
    out = df.copy()
    for layer, cell in cells.items():
        base = SWVL.format(layer=layer)
        raw_mean_col = f"{base}_raw"
        if raw_mean_col not in df.columns:
            raise ValueError(f"{site}: missing {raw_mean_col!r}; cannot renormalise layer {layer}")
        raw_mean = df[raw_mean_col]
        for suffix in STAT_SUFFIXES:
            raw_col = f"{base}_raw{suffix}"
            if raw_col not in df.columns:
                continue
            raw = df[raw_col]
            if suffix == "_sum":
                bare, _ = normalise_swvl(raw, cell.sigma, cell.mean)
                n_hours = raw / raw_mean.where(raw_mean != 0)
                zscore = (raw - n_hours * cell.mean) / cell.sigma
            else:
                bare, zscore = normalise_swvl(raw, cell.sigma, cell.mean)
            for col, values in ((f"{base}{suffix}", bare), (f"{base}_zscore{suffix}", zscore)):
                if col in out.columns:
                    out[col] = values
    return out


def format_cell(value) -> str:
    """Shortest round-trip text for a float; empty for NaN, as pandas writes a missing value."""
    number = float(value)
    return "" if math.isnan(number) else repr(number)


def _read_rows(path: Path) -> list[list[str]]:
    with open(path, newline="") as fh:
        return list(csv.reader(fh))


def splice_csv(src_path: Path, dst_path: Path, new_values: pd.DataFrame, site: str) -> int:
    """Copy ``src_path`` to ``dst_path`` as text, replacing only the columns of ``new_values``.

    ``new_values`` must hold one row per data line of the file, in file order.
    Returns the number of data rows written.
    """
    rows = _read_rows(src_path)
    if not rows:
        raise ValueError(f"{site}: {src_path} is empty")
    header = rows[0]
    if len(set(header)) != len(header):
        raise ValueError(f"{site}: duplicate column names in the header; cannot address columns by name")
    index = {name: j for j, name in enumerate(header)}
    missing = [c for c in new_values.columns if c not in index]
    if missing:
        raise ValueError(f"{site}: columns not in the file header: {missing}")
    n_data = len(rows) - 1
    if n_data != len(new_values):
        raise ValueError(f"{site}: file has {n_data} data rows, the parsed frame has {len(new_values)}")

    targets = [(index[c], [format_cell(v) for v in new_values[c].to_numpy()]) for c in new_values.columns]
    with open(dst_path, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator=CSV_LINETERMINATOR)
        writer.writerow(header)
        for k in range(n_data):
            row = rows[k + 1]
            if len(row) != len(header):
                raise ValueError(f"{site}: data row {k + 1} has {len(row)} fields, the header has {len(header)}")
            row = list(row)
            for j, cells in targets:
                row[j] = cells[k]
            writer.writerow(row)
    return n_data


def verify_untouched_text(src_path: Path, dst_path: Path, touched: Sequence[str], site: str) -> None:
    """Raise unless every cell outside ``touched`` is the same string in both files."""
    src_rows, dst_rows = _read_rows(src_path), _read_rows(dst_path)
    if src_rows[0] != dst_rows[0]:
        raise RuntimeError(f"{site}: the header changed during the rebuild")
    if len(src_rows) != len(dst_rows):
        raise RuntimeError(f"{site}: row count changed ({len(src_rows)} -> {len(dst_rows)})")
    skip = {j for j, name in enumerate(src_rows[0]) if name in set(touched)}
    keep = [j for j in range(len(src_rows[0])) if j not in skip]
    for k in range(1, len(src_rows)):
        before = [src_rows[k][j] for j in keep]
        after = [dst_rows[k][j] for j in keep]
        if before != after:
            col = next(src_rows[0][j] for j in keep if src_rows[k][j] != dst_rows[k][j])
            raise RuntimeError(f"{site}: untouched cell {col!r} changed text in data row {k}")


def _read_provenance(config_dir: Path) -> dict:
    path = config_dir / PROVENANCE_FILE
    sidecar = config_dir / SIDECAR_FILE
    if sidecar.exists():
        # Checked before the provenance block, not instead of it: between the merge and the sweep
        # that folds the sidecar in, RUN_PROVENANCE.json exists WITHOUT the block, so the block
        # check alone would let an already-servable tree through and quietly square the sigma.
        raise ValueError(
            f"{sidecar} is present: the Stage-4 merge already built this tree under the servable "
            "contract. Rebuilding it would normalise values that are already normalised -- the bare "
            "column would become raw/sigma/sigma -- and nothing downstream could tell. Rebuild the "
            "legacy tree instead, or use this tree as it is."
        )
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; the source tree must carry its merge provenance")
    with open(path) as fh:
        provenance = json.load(fh)
    if PROVENANCE_KEY in provenance:
        raise ValueError(f"{path} already has a {PROVENANCE_KEY!r} block; rebuild from the legacy tree")
    return provenance


def _contract_block(clim, src: Path, layers: Sequence[int], max_radius_cells: int, cell_rows, git_sha) -> dict:
    return {
        "name": CONTRACT_NAME,
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "source_tree": str(src),
        "swvl": {
            "cube": clim.path,
            "cube_sha256": clim.sha256,
            "cube_attrs": clim.attrs,
            "layers": list(layers),
            "definition": {"bare": "raw / sigma_clim", "zscore": "(raw - mu_clim) / sigma_clim"},
            "clip": None,
            "min_valid_sigma": MIN_VALID_SIGMA,
            "max_radius_cells": max_radius_cells,
            "n_fallback_cells": int(sum(1 for row in cell_rows if row["fallback"])),
            "untouched_cells": "copied as text, verified string-identical per file",
        },
        "precip_memory": {
            "source_column": ERA5_HOURLY_MEAN_PRECIP_COL,
            "units": "daily mean of hourly metres, x 24 x 1000 -> mm/day; also the servable precip",
            "rain_event_mm_per_day": SERVABLE_RAIN_EVENT_MM_PER_DAY,
            "where": "feature_engineering under --FEATURE_CONTRACT servable; the tree already carries the column",
        },
    }


def rebuild_tree(
    src_config_dir,
    dst_config_dir,
    cube_path,
    layers: Sequence[int] = LAYERS,
    max_radius_cells: int = DEFAULT_MAX_RADIUS_CELLS,
    git_sha: str | None = None,
) -> dict:
    """Write a servable copy of ``src_config_dir`` at ``dst_config_dir``; return the manifest."""
    src, dst = Path(src_config_dir), Path(dst_config_dir)
    staged = dst.with_name(dst.name + PARTIAL_SUFFIX)
    for target in (dst, staged):
        if target.exists():
            raise FileExistsError(f"{target} exists; refusing to overwrite a built tree")

    files = sorted(p for p in (src / "daily").glob(f"*{DAILY_SUFFIX}") if AGGREGATE_MARKER not in p.name)
    if not files:
        raise FileNotFoundError(f"no site files under {src / 'daily'}")
    provenance = _read_provenance(src)
    clim = load_climatology(cube_path)

    (staged / "daily").mkdir(parents=True)
    cell_rows, n_rows = [], 0
    for path in files:
        site = path.name[: -len(DAILY_SUFFIX)]
        df = pd.read_csv(path, low_memory=False, float_precision="round_trip")
        lat, lon = site_coordinates(df, site)
        cells = {}
        for layer in layers:
            try:
                cells[layer] = lookup_cell(clim, lat, lon, layer, max_radius_cells)
            except LookupError as exc:
                raise LookupError(f"{site}: {exc}") from exc
            cell_rows.append({"site": site, "site_lat": lat, "site_lon": lon, **asdict(cells[layer])})
        out = rebuild_site_frame(df, cells, site)
        orphans = orphan_family_columns(df.columns, layers)
        if orphans:
            raise ValueError(f"{site}: no _raw companion for {orphans}; they would keep legacy per-site values")
        touched = touched_columns(df.columns, layers)
        dst_file = staged / "daily" / path.name
        n_rows += splice_csv(path, dst_file, out[touched], site)
        verify_untouched_text(path, dst_file, touched, site)
        fallbacks = [layer for layer, cell in cells.items() if cell.fallback]
        LOGGER.info("%s rows=%d fallback_layers=%s", site, len(df), fallbacks or "-")

    not_copied_dirs = sorted(p.name for p in src.iterdir() if p.is_dir() and p.name != "daily")
    if not_copied_dirs:
        LOGGER.warning("not copied (only daily/ is rebuilt): %s", not_copied_dirs)
    for extra in src.iterdir():
        if extra.is_file() and extra.name != PROVENANCE_FILE:
            shutil.copyfile(extra, staged / extra.name)
    provenance[PROVENANCE_KEY] = _contract_block(clim, src, layers, max_radius_cells, cell_rows, git_sha)
    with open(staged / PROVENANCE_FILE, "w") as fh:
        json.dump(provenance, fh, indent=2)
    pd.DataFrame(cell_rows).to_csv(staged / CELLS_FILE, index=False)

    manifest = {
        "dst": str(dst),
        "n_sites": len(files),
        "n_rows": n_rows,
        "n_fallback_cells": provenance[PROVENANCE_KEY]["swvl"]["n_fallback_cells"],
        "not_copied_dirs": not_copied_dirs,
    }
    with open(staged / MANIFEST_FILE, "w") as fh:
        json.dump(manifest, fh, indent=2)
    staged.rename(dst)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="Source config dir (holds daily/ and RUN_PROVENANCE.json)")
    parser.add_argument("--dst", required=True, help="New config dir; must not exist")
    parser.add_argument("--cube", required=True, help="swvl_sigma_climatology_v2_1995_2019.nc")
    parser.add_argument("--layers", default="1,2,3,4", help="Comma-separated soil layers")
    parser.add_argument("--max-radius-cells", type=int, default=DEFAULT_MAX_RADIUS_CELLS)
    parser.add_argument("--git-sha", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    layers = tuple(int(x) for x in args.layers.split(","))
    manifest = rebuild_tree(args.src, args.dst, args.cube, layers, args.max_radius_cells, args.git_sha)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
