"""Changed products against what they replace: r and the largest movers per column. A report, not a gate.

python -m src.static_features.checks.compare --scale <scale>. Writes checks/compare_summary.csv and compare_movers.csv.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from src.static_features.assemble import SOIL_VALUE_COLUMNS
from src.static_features.results import ExtractionFailure, RunContext, read_table
from src.static_features.run import DEFAULT_PROJECT

LOGGER = logging.getLogger(__name__)

ENV_FILE = "site_info_with_env_data.csv"
SOIL_FILE = "soilgrids_data.csv"
ENV_COLUMNS = ["elevation_m", "slope_deg", "aspect_sin", "aspect_cos", "canopy_height_m"]
PROBE_FILE = Path("probes") / "canopy_height_ground" / "canopy_stats_sites.csv"
PROBE_COLUMN = "metasq1km_h"
MOVER_COLUMNS = ["site_name", "old", "new", "change"]


def column_report(old: pd.Series, new: pd.Series, top: int = 10) -> dict[str, Any]:
    """Over the sites holding both values: their number, r, the median change and the largest changes, biggest first."""
    joined = pd.concat([old.rename("old"), new.rename("new")], axis=1, join="inner").dropna()
    change = joined["new"] - joined["old"]
    varies = len(joined) > 1 and joined["old"].std() > 0 and joined["new"].std() > 0
    movers = [
        {
            "site_name": site,
            "old": float(joined.loc[site, "old"]),
            "new": float(joined.loc[site, "new"]),
            "change": float(change[site]),
        }
        for site in change.abs().sort_values(ascending=False, kind="mergesort").index[:top]
    ]
    return {
        "n": len(joined),
        "r": float(joined["old"].corr(joined["new"])) if varies else math.nan,
        "median_abs_change": float(change.abs().median()) if len(joined) else math.nan,
        "largest_movers": movers,
    }


def _reports(context: RunContext, scale: str, probe_root: str | None) -> list[tuple[str, str, dict[str, Any]]]:
    output = context.output_root
    extracted = output.parents[1]
    env = read_table(output / ENV_FILE, index_col="site_name")
    soil = read_table(output / SOIL_FILE, index_col="site_name")
    old_env = read_table(extracted / "env_site_data" / scale / ENV_FILE, index_col="site_name")
    old_soil = read_table(extracted / "terrain_site_data" / scale / SOIL_FILE, index_col="site_name")
    probe_base = output.parents[4] if probe_root is None else Path(probe_root)
    probe = read_table(probe_base / PROBE_FILE, index_col="site_name")

    reports = [(ENV_FILE, column, column_report(old_env[column], env[column])) for column in ENV_COLUMNS]
    reports.extend((SOIL_FILE, column, column_report(old_soil[column], soil[column])) for column in SOIL_VALUE_COLUMNS)
    against = f"{PROBE_FILE.name} {PROBE_COLUMN}"
    reports.append((against, "canopy_height_m", column_report(probe[PROBE_COLUMN], env["canopy_height_m"])))
    return reports


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", required=True)
    parser.add_argument("--probe-root", help="The checkout holding probes/, where the data lives apart from the code.")
    args = parser.parse_args(argv)

    context = RunContext.for_scale(args.scale, DEFAULT_PROJECT)
    try:
        reports = _reports(context, args.scale, args.probe_root)
    except (ExtractionFailure, OSError, KeyError) as failure:
        LOGGER.error("%s", failure)
        return 1

    summary, movers = [], []
    for against, column, report in reports:
        summary.append((against, column, report["n"], report["r"], report["median_abs_change"]))
        movers.extend({"against": against, "column": column, **mover} for mover in report["largest_movers"])
        LOGGER.info(
            "%s %s: n=%d r=%.3f median change %.3f",
            against,
            column,
            report["n"],
            report["r"],
            report["median_abs_change"],
        )
    checks = context.output_root / "checks"
    checks.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary, columns=["against", "column", "n", "r", "median_abs_change"]).to_csv(
        checks / "compare_summary.csv", index=False
    )
    pd.DataFrame(movers, columns=["against", "column", *MOVER_COLUMNS]).to_csv(
        checks / "compare_movers.csv", index=False
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
