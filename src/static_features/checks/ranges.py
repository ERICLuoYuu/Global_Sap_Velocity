"""Physical ranges and the coverage report: python -m src.static_features.checks.ranges --scale <scale>.

Exits 1 and lists every value outside its range, for the owner to read. Writes out_of_range.csv, coverage_report.csv and
low_land_sites.csv to checks/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
from src.static_features.assemble import SOIL_DEPTHS
from src.static_features.results import ExtractionFailure, RunContext, read_table
from src.static_features.run import DEFAULT_PROJECT, PRODUCTS

LOGGER = logging.getLogger(__name__)

VALUE_FILES = ("site_info_with_env_data.csv", "soilgrids_data.csv", "new_static_features.csv")
PHENOLOGY_DATES = ("greenup", "midgreenup", "maturity", "peak", "senescence", "midgreendown", "dormancy")
OUT_OF_RANGE_COLUMNS = ["site_name", "column", "value", "low", "high"]
SHARE_PREFIX = "fm_share_"
SHARE_TOLERANCE = 1e-9
RANGES: dict[str, tuple[float, float]] = {
    "elevation_m": (-450, 9000),
    "slope_deg": (0, 90),
    "aspect_sin": (-1, 1),
    "aspect_cos": (-1, 1),
    "canopy_height_m": (0, 60),
    "canopy_height_p95_m": (0, 100),
    "land_share": (0, 1),
    "phen_cycle_share": (0, 1),
    "root_depth_han": (0, 100),
    "wood_density": (0.1, 1.5),
    "tree_richness_per_ha": (1, 250),
    "agb_mg_ha": (0, 1500),
    "agb_sd": (0, 1500),
    "forest_age_gami": (0, 300),
    "tree_cover_pct": (0, 100),
    "evi_mean": (-0.2, 1),
    "evi_amplitude": (-0.2, 1),
    "evi_season_amplitude": (0, 1),
    "evi_season_r2": (0, 1),
    "evi_peak_cos_w": (-1, 1),
    "evi_peak_sin_w": (-1, 1),
    "phen_evi2_min": (-0.2, 1),
    "phen_evi2_amplitude": (0, 1),
    "phen_season_days": (0, 730),
    "phen_midseason_days": (0, 730),
    "phen_num_cycles": (0, 2),
    "phen_years_used": (0, 20),
    "evi_years_used": (0, 20),
    "root_depth_lusk": (0, 100),
    "wood_density_lusk": (0.1, 1.5),
    **{f"{SHARE_PREFIX}{code}": (0, 1) for code in (11, 20, 31, 32, 40, 53)},
    **{f"phen_{date}_{part}": (-1, 1) for date in PHENOLOGY_DATES for part in ("sin", "cos")},
    **{
        f"{prop}_{depth}": bounds
        for prop, bounds in (
            ("clay", (0, 100)),
            ("sand", (0, 100)),
            ("silt", (0, 100)),
            ("cfvo", (0, 100)),
            ("bdod", (0, 2.2)),
            ("soc", (0, 1000)),
            ("phh2o", (3, 11)),
            ("nitrogen", (0, 100)),
        )
        for depth in SOIL_DEPTHS
    },
}


def out_of_range(values: pd.DataFrame) -> pd.DataFrame:
    """One row per value outside its column's range, and per site whose management shares sum above 1. NaN is fine."""
    rows = []
    for column in values.columns:
        if column in RANGES:
            low, high = RANGES[column]
            outside = values[column][(values[column] < low) | (values[column] > high)]
            rows.extend((site, column, float(value), low, high) for site, value in outside.items())
    shares = values[[column for column in values.columns if column.startswith(SHARE_PREFIX)]].sum(axis=1)
    rows.extend(
        (site, "fm_share_sum", float(value), 0, 1) for site, value in shares[shares > 1 + SHARE_TOLERANCE].items()
    )
    return pd.DataFrame(rows, columns=OUT_OF_RANGE_COLUMNS)


def coverage_report(quality: pd.DataFrame, column_products: Mapping[str, str]) -> pd.DataFrame:
    """Per product and reason: how many columns have that reason, and the fewest and most sites among them."""
    suffix = "_status"
    rows = []
    for name in quality.columns:
        if name.endswith(suffix):
            column = name[: -len(suffix)]
            counts = quality[name].value_counts()
            rows.extend((column_products[column], reason, column, int(sites)) for reason, sites in counts.items())
    long = pd.DataFrame(rows, columns=["product", "reason", "column", "sites"])
    return long.groupby(["product", "reason"], as_index=False).agg(
        columns=("column", "size"), sites_min=("sites", "min"), sites_max=("sites", "max")
    )


def low_land_sites(values: pd.DataFrame, threshold: float = 0.5) -> list[str]:
    return sorted(values.index[values["land_share"] < threshold])


def _column_products(parts_root: Path) -> dict[str, str]:
    products = {}
    for product in PRODUCTS:
        for column in pd.read_csv(parts_root / product / "status.csv", nrows=0).columns:
            if column != "site_name":
                products[column] = product
    return products


def _read_values(output: Path) -> pd.DataFrame:
    frames = [read_table(output / name, index_col="site_name") for name in VALUE_FILES]
    return pd.concat([frame[[column for column in frame.columns if column in RANGES]] for frame in frames], axis=1)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", required=True)
    args = parser.parse_args(argv)

    context = RunContext.for_scale(args.scale, DEFAULT_PROJECT)
    output = context.output_root
    try:
        values = _read_values(output)
        quality = read_table(output / "static_quality.csv")
        coverage = coverage_report(quality, _column_products(context.parts_root))
    except (ExtractionFailure, OSError, KeyError) as failure:
        LOGGER.error("%s", failure)
        return 1

    outside = out_of_range(values)
    low_land = low_land_sites(values)
    checks = output / "checks"
    checks.mkdir(parents=True, exist_ok=True)
    outside.to_csv(checks / "out_of_range.csv", index=False)
    coverage.to_csv(checks / "coverage_report.csv", index=False)
    pd.DataFrame({"site_name": low_land}).to_csv(checks / "low_land_sites.csv", index=False)
    LOGGER.info("sites with under half their cell on land: %s", low_land)
    for row in outside.itertuples(index=False):
        LOGGER.error("%s %s = %s is outside %s to %s", row.site_name, row.column, row.value, row.low, row.high)
    return 1 if len(outside) else 0


if __name__ == "__main__":
    sys.exit(main())
