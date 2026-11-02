"""Join the parts into the files the merge reads, the new columns, quality, the long table and provenance.

python -m src.static_features.assemble --scale <scale>. Everything is built in a temporary directory first, so a
failure leaves any earlier output as it was.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import subprocess  # nosec B404
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from src.static_features.results import (
    BY_YEAR_COLUMNS,
    ExtractionFailure,
    ProductResult,
    RunContext,
    read_part,
    validate,
)
from src.static_features.run import DEFAULT_PROJECT, PRODUCTS
from src.static_features.sites import COORDINATE_CORRECTIONS, site_list_hash

LOGGER = logging.getLogger(__name__)

SECTORS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
SITE_COLUMNS = ["start_date", "end_date", "lat", "lon", "site_name"]
ENV_COLUMNS = [
    *SITE_COLUMNS,
    "canopy_height_m",
    "elevation_m",
    "slope_deg",
    "aspect_deg",
    "aspect_standardized",
    "aspect_category",
    "aspect_sin",
    "aspect_cos",
    *[f"bio{n}" for n in range(1, 20)],
    "koppen_class",
    "koppen_name",
    "slope_deg_centre",
]
SOIL_PROPERTIES = ("clay", "sand", "silt", "bdod", "cfvo", "soc", "phh2o", "nitrogen")
SOIL_DEPTHS = ("0_5", "5_15", "15_30", "30_60", "60_100", "100_200")
SOIL_VALUE_COLUMNS = [f"{prop}_{depth}" for prop in SOIL_PROPERTIES for depth in SOIL_DEPTHS]
SOIL_COLUMNS = [
    *SITE_COLUMNS,
    "data_source",
    "offset_distance_m",
    "snapped_lat",
    "snapped_lon",
    *SOIL_VALUE_COLUMNS,
    "texture_class_0_5",
]
SOIL_DATA_SOURCE = "earth_engine_1km"
STAND_AGE_COLUMNS = [*SITE_COLUMNS, "stand_age_TC000", "stand_age_TC010", "stand_age_TC020", "stand_age_TC030"]
FAN_COLUMNS = [
    "site_name",
    "root_depth",
    "infiltration_depth",
    "deep_drainage",
    "drainage_frequency",
    "regolith_flush_rate",
    "gw_residence_time",
    "root_depth_mask_valid",
    "root_depth_fallback_px",
    "root_depth_continent",
    "root_depth_lat_pixel",
    "root_depth_lon_pixel",
]
NEW_PRODUCTS = ("han", "yang", "lusk", "liang", "lesiv", "cci", "gami", "tree_cover", "evi", "phenology")
NEW_FROM_OLD_PRODUCTS = ("canopy_height_p95_m", "land_share")
LUSK_QUALITY_SUFFIXES = ("_cov", "_aoa")
NOTES = {
    "slope_deg_centre": "Empty: a 1 km cell has no centre-pixel slope, so the column stays for the merge's header only.",
    "aspect_deg": "aspect_deg and aspect_standardized are the bearing of the mean aspect vector, empty where it is (0, 0).",
}


def compass_sector(aspect_deg: float) -> str:
    """N spans 337.5 to 22.5, so the bins are offset by half a sector before binning."""
    return SECTORS[int(((aspect_deg % 360) + 22.5) // 45) % 8]


def aspect_report(sin: float, cos: float) -> tuple[float | None, str | None]:
    """The bearing and sector of a mean aspect vector. A cell of flat pixels has no direction, so none is made up."""
    if math.isnan(sin) or math.isnan(cos) or (sin == 0.0 and cos == 0.0):
        return None, None
    bearing = math.degrees(math.atan2(sin, cos)) % 360
    return bearing, compass_sector(bearing)


def _read_parts(context: RunContext, products: Sequence[str]) -> dict[str, ProductResult]:
    expected = site_list_hash(context.sites)
    names = list(context.sites["site_name"])
    parts = {}
    for product in products:
        part, found = read_part(context.parts_root, product)
        if found != expected:
            raise ExtractionFailure(f"{product}: the part was built from another site list; rerun the product")
        validate(part, names)
        parts[product] = part
    return parts


def _refuse_shared_columns(parts: dict[str, ProductResult]) -> None:
    owners: dict[str, list[str]] = {}
    for product, part in parts.items():
        for column in part.values.columns:
            owners.setdefault(column, []).append(product)
    shared = {column: names for column, names in owners.items() if len(names) > 1}
    if shared:
        raise ExtractionFailure(f"columns come from two products: {shared}")


def _with_site_columns(sites: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([sites[SITE_COLUMNS].reset_index(drop=True), values.reset_index(drop=True)], axis=1)


def _select(table: pd.DataFrame, columns: list[str], name: str) -> pd.DataFrame:
    absent = [column for column in columns if column not in table.columns]
    if absent:
        raise ExtractionFailure(f"{name}: no part holds {absent}")
    return table[columns]


def _env_table(sites: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    vectors = _select(values, ["aspect_sin", "aspect_cos"], "site_info_with_env_data.csv")
    reports = [aspect_report(row.aspect_sin, row.aspect_cos) for row in vectors.itertuples(index=False)]
    bearings = pd.Series([report[0] for report in reports], dtype=float, index=values.index)
    table = values.assign(
        aspect_deg=bearings,
        aspect_standardized=bearings,
        aspect_category=[report[1] for report in reports],
        slope_deg_centre=np.nan,
    )
    return _select(_with_site_columns(sites, table), ENV_COLUMNS, "site_info_with_env_data.csv")


def _soil_table(sites: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    table = _with_site_columns(sites, values)
    table = table.assign(
        data_source=SOIL_DATA_SOURCE, offset_distance_m=0, snapped_lat=table["lat"], snapped_lon=table["lon"]
    )
    return _select(table, SOIL_COLUMNS, "soilgrids_data.csv")


def _new_table(parts: dict[str, ProductResult], values: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for product in NEW_PRODUCTS
        if product in parts
        for column in parts[product].values.columns
        if not (product == "lusk" and column.endswith(LUSK_QUALITY_SUFFIXES))
    ]
    return _select(values, [*columns, *NEW_FROM_OLD_PRODUCTS], "new_static_features.csv").reset_index()


def _lusk_quality_table(parts: dict[str, ProductResult], values: pd.DataFrame) -> pd.DataFrame:
    """How far to trust each trait, kept out of the merge: it describes the Lusk maps, not the site."""
    columns = [column for column in parts["lusk"].values.columns if column.endswith(LUSK_QUALITY_SUFFIXES)]
    return _select(values, columns, "lusk_quality.csv").reset_index()


def _quality_table(parts: dict[str, ProductResult], index: pd.Index) -> pd.DataFrame:
    columns = {}
    for part in parts.values():
        for column in part.values.columns:
            columns[f"{column}_status"] = part.status[column].reindex(index)
            if column in part.coverage.columns:
                columns[f"{column}_coverage"] = part.coverage[column].reindex(index)
    return pd.DataFrame(columns, index=index).reset_index()


def _long_table(parts: dict[str, ProductResult]) -> pd.DataFrame:
    tables = [part.by_year for part in parts.values() if part.by_year is not None]
    return pd.concat(tables, ignore_index=True)[list(BY_YEAR_COLUMNS)]


def _tables(sites: pd.DataFrame, parts: dict[str, ProductResult]) -> dict[str, pd.DataFrame]:
    index = pd.Index(sites["site_name"], name="site_name")
    values = pd.concat([part.values.reindex(index) for part in parts.values()], axis=1)
    tables = {
        "site_info_with_env_data": _env_table(sites, values),
        "soilgrids_data": _soil_table(sites, values),
        "stand_age_data": _select(_with_site_columns(sites, values), STAND_AGE_COLUMNS, "stand_age_data.csv"),
        "site_info_with_root_depth_fan2017": _select(values.reset_index(), FAN_COLUMNS, "fan file"),
        "new_static_features": _new_table(parts, values),
        "static_quality": _quality_table(parts, index),
        "static_features_by_year": _long_table(parts),
    }
    if "lusk" in parts:
        tables["lusk_quality"] = _lusk_quality_table(parts, values)
    for part in parts.values():
        tables.update(part.extras)
    return tables


def _provenance(parts: dict[str, ProductResult], site_hash: str, git_commit: str | None) -> dict[str, Any]:
    return {
        "code_commit": git_commit,
        "site_list_hash": site_hash,
        "coordinate_corrections": COORDINATE_CORRECTIONS,
        "products": {product: part.provenance for product, part in parts.items()},
        "notes": NOTES,
    }


def assemble(context: RunContext, products: Sequence[str] = tuple(PRODUCTS), git_commit: str | None = None) -> Path:
    parts = _read_parts(context, products)
    _refuse_shared_columns(parts)
    tables = _tables(context.sites, parts)
    provenance = _provenance(parts, site_list_hash(context.sites), git_commit)

    output = context.output_root
    output.mkdir(parents=True, exist_ok=True)
    for stale in output.glob(".tmp_assemble_*"):
        shutil.rmtree(stale)  # left by a job that was killed; one job at a time
    temp = output / f".tmp_assemble_{os.getpid()}"
    temp.mkdir()
    try:
        for name, table in tables.items():
            table.to_csv(temp / f"{name}.csv", index=False)
        with open(temp / "provenance.json", "w", encoding="utf-8") as handle:
            json.dump(provenance, handle, indent=2, sort_keys=True)
        for path in sorted(temp.iterdir()):
            os.replace(path, output / path.name)
    finally:
        shutil.rmtree(temp, ignore_errors=True)
    return output


def _current_commit() -> str | None:
    completed = subprocess.run(  # nosec B603 B607
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", required=True)
    args = parser.parse_args(argv)

    context = RunContext.for_scale(args.scale, DEFAULT_PROJECT)
    try:
        output = assemble(context, git_commit=_current_commit())
    except ExtractionFailure as failure:
        LOGGER.error("%s", failure)
        return 1
    LOGGER.info("assembled %d sites into %s", len(context.sites), output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
