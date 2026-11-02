"""Han 2025 rooting depth, 2015 map: the native 0.05 degree pixel, tree types only, with a 1-pixel neighbour rule."""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
from src.static_features.rasters import pixel_index, pixel_with_reach
from src.static_features.results import NO_DATA, ExtractionFailure, ProductResult, RunContext

TREE_TYPES = ("BETTr", "BETTe", "BDTTr", "BDTTe", "BDTBo", "NETTe", "NETBo", "NDT")
OTHER_TYPES = ("BESTe", "BDSTe", "BDSBo", "C3G", "C4G")
FILE_NAME = "Global_RD_{kind}_{plant}_2015.nc"
REACH = 1

# The archive is a real RAR and Palma has no unrar, so the copy unpacked with Windows tar is the source.
SOURCE_DIR = Path("/scratch/tmp/yluo2/han_probe/data/History")
ARCHIVE_MD5 = "a3ddf1a20334ada08b5525cecd82cb3c"
PROVENANCE = {
    "source": "https://zenodo.org/records/15361409, History.rar",
    "version": "Han et al. 2025, 2015 map; later years in the record are future scenarios",
    "checksum": {"History.rar": f"md5:{ARCHIVE_MD5}"},
    "time_window": "2015",
    "method": (
        "Native pixel holding a tree type; if it holds none, the nearest tree pixel within 1. Grass and shrub depths are "
        "not used for a tree measurement. No pixel on Earth holds more than one of the 13 types."
    ),
}


@dataclass(frozen=True)
class HanGrids:
    """Row-first, trees only: depth and uncertainty are NaN and type_index is -1 where no tree type is mapped."""

    depth: np.ndarray
    uncertainty: np.ndarray
    type_index: np.ndarray


def _read_depth(path: Path) -> np.ndarray:
    with netCDF4.Dataset(path) as dataset:
        return np.ma.filled(dataset.variables["depth"][:].astype("f4"), np.nan)


def load_grids(directory: Path) -> HanGrids:
    """Join the plant types into one map per variable, stopping if any two of the 13 share a pixel."""
    directory = Path(directory)
    occupied = depth = uncertainty = type_index = None
    for position, plant in enumerate(TREE_TYPES + OTHER_TYPES):
        grid = _read_depth(directory / FILE_NAME.format(kind="EST", plant=plant))
        if occupied is None:
            occupied = np.zeros(grid.shape, dtype=bool)
            depth = np.full(grid.shape, np.nan, dtype="f4")
            uncertainty = np.full(grid.shape, np.nan, dtype="f4")
            type_index = np.full(grid.shape, -1, dtype="i1")
        filled = ~np.isnan(grid)
        shared = int((occupied & filled).sum())
        if shared:
            raise ExtractionFailure(
                f"han: {plant} shares {shared} pixels with another plant type; one type per pixel no longer holds"
            )
        occupied |= filled
        if plant in TREE_TYPES:
            depth[filled] = grid[filled]
            uncertainty[filled] = _read_depth(directory / FILE_NAME.format(kind="UI", plant=plant))[filled]
            type_index[filled] = position
    return HanGrids(depth, uncertainty, type_index)


def _is_mapped(block: np.ndarray) -> np.ndarray:
    return ~np.isnan(block)


def extract(context: RunContext) -> ProductResult:
    grids = load_grids(context.raw_root / "han_2025" / "History")
    n_rows, n_cols = grids.depth.shape
    step = 180.0 / n_rows

    def read_block(row_start: int, row_stop: int, col_start: int, col_stop: int) -> np.ndarray:
        return grids.depth[row_start:row_stop, col_start:col_stop]

    values, status = [], []
    for site in context.sites.itertuples(index=False):
        row = pixel_index(site.lat, 90.0, -step, n_rows)
        col = pixel_index(site.lon, -180.0, step, n_cols)
        hit = pixel_with_reach(read_block, grids.depth.shape, row, col, REACH, _is_mapped)
        if hit.row is None:
            values.append((math.nan, math.nan, None))
            status.append((NO_DATA, NO_DATA, NO_DATA))
            continue
        spread = float(grids.uncertainty[hit.row, hit.col])
        plant = TREE_TYPES[grids.type_index[hit.row, hit.col]]
        values.append((hit.value, spread, plant))
        status.append((hit.status, NO_DATA if math.isnan(spread) else hit.status, hit.status))

    columns = ["root_depth_han", "root_depth_han_uncertainty", "root_depth_han_type"]
    index = pd.Index(context.sites["site_name"], name="site_name")
    return ProductResult(
        product="han",
        values=pd.DataFrame(values, index=index, columns=columns),
        status=pd.DataFrame(status, index=index, columns=columns),
        coverage=pd.DataFrame(index=index),
        by_year=None,
        extras={},
        provenance=PROVENANCE,
    )


def fetch_inputs(context: RunContext) -> None:
    target = context.raw_root / "han_2025" / "History"
    names = [FILE_NAME.format(kind=kind, plant=plant) for kind in ("EST", "UI") for plant in TREE_TYPES + OTHER_TYPES]
    absent = [name for name in names if not (SOURCE_DIR / name).exists()]
    if absent:
        raise ExtractionFailure(
            f"han: {len(absent)} of {len(names)} files are absent from {SOURCE_DIR}, e.g. {absent[0]}"
        )
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        if not (target / name).exists():
            temporary = target / f".{name}.partial"
            shutil.copy2(SOURCE_DIR / name, temporary)
            os.replace(temporary, target / name)
    record = {"archive": "History.rar", "md5": ARCHIVE_MD5, "unpacked_from": str(SOURCE_DIR)}
    (target.parent / "source.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
