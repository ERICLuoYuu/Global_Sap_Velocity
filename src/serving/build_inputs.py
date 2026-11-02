"""Stage B: the served grid and cells, and the per-cell inputs other than ERA5-Land.

    python -m src.serving.build_inputs cells --store S --mask M --cube C [--statics P]
    python -m src.serving.build_inputs solar --store S --year Y [--processes P]
    python -m src.serving.build_inputs lai   --store S --year Y --lai_dir D

``cells`` writes ``grid.json`` and ``cells.parquet``: every woody cell of the mask, its PFT, and the
soil-moisture climatology a site at the cell centre would have read. ``--statics`` joins a table of
static layers keyed by grid ``row``/``col`` (only :data:`STATIC_COLUMNS` names are accepted).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.serving.base_columns import STATIC_COLUMNS
from src.serving.store import Store, save_grid, write_json, year_days

logger = logging.getLogger("build_inputs")

KEY = ["row", "col"]


def attach_statics(cells: pd.DataFrame, statics: pd.DataFrame) -> pd.DataFrame:
    """``cells`` with the static layers of ``statics`` joined on grid row/col; refuses unknown or duplicate keys."""
    missing_key = [name for name in KEY if name not in statics.columns]
    if missing_key:
        raise KeyError(f"statics table lacks key column(s) {missing_key}")
    unknown = [name for name in statics.columns if name not in KEY and name not in STATIC_COLUMNS]
    if unknown:
        raise ValueError(f"statics table holds non-static column(s) {unknown}")
    clash = [name for name in statics.columns if name not in KEY and name in cells.columns]
    if clash:
        raise ValueError(f"cells already hold {clash}")
    return cells.merge(statics, on=KEY, how="left", validate="one_to_one")


def within_bbox(cells: pd.DataFrame, bbox) -> pd.DataFrame:
    """The cells whose centres lie in ``(west, south, east, north)``, in their original order."""
    west, south, east, north = (float(value) for value in bbox)
    if not (west < east and south < north):
        raise ValueError(f"bbox {bbox} is not (west, south, east, north)")
    inside = cells["longitude"].between(west, east) & cells["latitude"].between(south, north)
    if not inside.any():
        raise ValueError(f"no cell lies in bbox {bbox}")
    return cells.loc[inside].reset_index(drop=True)


def build_cells(mask, cube, statics_path=None, bbox=None) -> tuple:
    from src.feature_contract.swvl_climatology import load_climatology  # noqa: PLC0415
    from src.serving.grid import climatology_for_cells, forest_cells, grid_from_raster  # noqa: PLC0415

    grid = grid_from_raster(mask)
    cells = forest_cells(mask)
    if bbox is not None:
        cells = within_bbox(cells, bbox)
    cells = pd.concat([cells, climatology_for_cells(cells, load_climatology(cube))], axis=1)
    if statics_path:
        cells = attach_statics(cells, pd.read_parquet(statics_path))
    return grid, cells


def latitude_rows(cells: pd.DataFrame) -> tuple:
    """The distinct grid rows holding a cell, ascending, and each row's latitude."""
    table = cells.groupby("row")["latitude"].agg(["min", "max"])
    if not np.array_equal(table["min"].to_numpy(), table["max"].to_numpy()):
        raise ValueError("cells of one grid row disagree on latitude")
    return table.index.to_numpy(dtype=np.int64), table["min"].to_numpy(dtype=float)


def _save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npy")
    np.save(temporary, array)
    os.replace(temporary, path)


def build_solar(store: Store, year: int, processes: int) -> Path:
    from src.serving.solar_grid import solar_year  # noqa: PLC0415

    rows, latitudes = latitude_rows(pd.read_parquet(store.cells, columns=["row", "latitude"]))
    result = solar_year(year_days(year), latitudes, processes)
    result["rows"] = rows
    path = Path(store.solar(year))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    with open(temporary, "wb") as handle:
        np.savez(handle, **result)
    os.replace(temporary, path)
    return path


def build_lai(store: Store, year: int, lai_dir) -> Path:
    from src.serving.lai import lai_for_cells  # noqa: PLC0415

    cells = pd.read_parquet(store.cells, columns=["latitude", "longitude"])
    values = lai_for_cells(year_days(year), lai_dir, cells["latitude"], cells["longitude"])
    empty_days = int(np.isnan(values).all(axis=1).sum())
    if empty_days:
        logger.warning("%d day(s) of %d have no GlobMap composite within the match tolerance", empty_days, year)
    path = Path(store.lai(year))
    _save_npy(path, values)
    return path


def _cells_command(args) -> None:
    store = Store(Path(args.store))
    grid, cells = build_cells(args.mask, args.cube, args.statics, args.bbox)
    Path(store.root).mkdir(parents=True, exist_ok=True)
    save_grid(store, grid)
    temporary = Path(store.cells).with_name("cells.tmp.parquet")
    cells.to_parquet(temporary, index=False)
    os.replace(temporary, store.cells)
    summary = {
        "mask": str(args.mask),
        "cube": str(args.cube),
        "statics": str(args.statics) if args.statics else None,
        "bbox": args.bbox,
        "cells": len(cells),
        "cells_per_pft": cells["pft"].value_counts().to_dict(),
        "cells_without_climatology": {
            name: int(cells[name].isna().sum()) for name in cells.columns if name.startswith("swvl_sigma_")
        },
        "static_columns": [name for name in STATIC_COLUMNS if name in cells.columns],
    }
    write_json(Path(store.root) / "cells.json", summary)
    logger.info("%s", summary)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    cells = commands.add_parser("cells")
    cells.add_argument("--store", required=True)
    cells.add_argument("--mask", required=True)
    cells.add_argument("--cube", required=True)
    cells.add_argument("--statics", default=None)
    cells.add_argument("--bbox", type=float, nargs=4, default=None, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    solar = commands.add_parser("solar")
    solar.add_argument("--store", required=True)
    solar.add_argument("--year", type=int, required=True)
    solar.add_argument("--processes", type=int, default=1)
    lai = commands.add_parser("lai")
    lai.add_argument("--store", required=True)
    lai.add_argument("--year", type=int, required=True)
    lai.add_argument("--lai_dir", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.command == "cells":
        _cells_command(args)
    elif args.command == "solar":
        logger.info("wrote %s", build_solar(Store(Path(args.store)), args.year, args.processes))
    else:
        logger.info("wrote %s", build_lai(Store(Path(args.store)), args.year, args.lai_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
