"""A merge on the earlier static files against one on the 1 km features: same rows, same dynamic values, new columns.

python -m src.static_features.checks.merge_parity --scale <scale> --before <tree> --after <tree>. Each tree holds
daily/<site>_daily.csv. Writes checks/merge_parity.csv; exits 1 on anything but static columns moving. A static value
that becomes empty is not a move: it stops the check unless the site and column are named in EXPECTED_EMPTIED.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from src.static_features.results import RunContext, read_table
from src.static_features.run import DEFAULT_PROJECT

LOGGER = logging.getLogger(__name__)

KEY = ["site_name", "TIMESTAMP"]
#: The merge writes this beside the site files. It pools every site, so its keys repeat.
AGGREGATE = "all_biomes_merged_daily.csv"
#: Site -> the static columns that may become empty there, and why. Still reported on every run.
EXPECTED_EMPTIED: dict[str, dict[str, Any]] = {
    "CHN_YUN_YUN": {
        "columns": (
            *(f"soil_{prop}{depth}" for prop in ("soc", "bdod") for depth in ("", "_0_5", "_5_15", "_15_30", "_30_60", "_60_100")),
            "soil_theta_wp",
            "soil_theta_fc",
            "soil_theta_sat",
        ),
        "reason": "The cell is mostly water. SoilGrids on Earth Engine holds no soc or bdod there, and the old REST ring "
        "search reached beyond the cell for them.",
    },
}
STATIC_EXACT = frozenset(
    {
        "elevation",
        "slope",
        "aspect_sin",
        "aspect_cos",
        "mean_annual_temp",
        "mean_annual_precip",
        "temp_seasonality",
        "precip_seasonality",
        "canopy_height",
        "stand_age",
        "root_depth",
        "infiltration_depth",
        "deep_drainage",
        "drainage_frequency",
        "regolith_flush_rate",
        "gw_residence_time",
    }
)


def static_columns_of(columns: Iterable[str]) -> list[str]:
    """The merge's static columns among ``columns``. ``soil_temperature_level_*`` is ERA5 and moves daily."""
    return [
        column
        for column in columns
        if column in STATIC_EXACT or (column.startswith("soil_") and not column.startswith("soil_temperature_level_"))
    ]


def _rows_differing(old: pd.Series, new: pd.Series) -> int:
    return int(((old != new) & ~(old.isna() & new.isna())).sum())


def compare(old: pd.DataFrame, new: pd.DataFrame, static_columns: Iterable[str]) -> dict[str, Any]:
    """Rows are matched on site and timestamp, never on position. Both empty counts as the same."""
    old_keyed, new_keyed = old.set_index(KEY), new.set_index(KEY)
    shared = old_keyed.index.intersection(new_keyed.index)
    lost = len(old_keyed.index.difference(new_keyed.index))
    gained = len(new_keyed.index.difference(old_keyed.index))
    static = set(static_columns)
    differing: dict[bool, dict[str, int]] = {True: {}, False: {}}
    emptied: dict[str, int] = {}
    for column in old_keyed.columns.intersection(new_keyed.columns, sort=False):
        before, after = old_keyed[column].reindex(shared), new_keyed[column].reindex(shared)
        rows = _rows_differing(before, after)
        if rows:
            differing[column in static][column] = rows
        became_empty = int((before.notna() & after.isna()).sum())
        if became_empty and column in static:
            emptied[column] = became_empty
    return {
        "same_keys": lost == 0 and gained == 0,
        "lost_keys": lost,
        "gained_keys": gained,
        "added_columns": [column for column in new_keyed.columns if column not in old_keyed.columns],
        "removed_columns": [column for column in old_keyed.columns if column not in new_keyed.columns],
        "dynamic_columns_differing": differing[False],
        "static_columns_differing": differing[True],
        "static_columns_emptied": emptied,
    }


def _daily_files(tree: str) -> dict[str, Path]:
    return {path.name: path for path in sorted((Path(tree) / "daily").glob("*_daily.csv")) if path.name != AGGREGATE}


def _report_rows(tallies: dict[str, Counter]) -> pd.DataFrame:
    rows = [
        {"column": column, "kind": kind, "rows": count}
        for kind, tally in tallies.items()
        for column, count in sorted(tally.items())
    ]
    return pd.DataFrame(rows, columns=["column", "kind", "rows"])


def _compare_trees(before: dict[str, Path], after: dict[str, Path], expected: set[str]) -> tuple[pd.DataFrame, list[str]]:
    problems: list[str] = []
    one_side = sorted(set(before) ^ set(after))
    if one_side:
        problems.append(f"on one side only: {one_side}")
    kinds = ("dynamic_differs", "removed", "missing", "unexpected", "static_differs", "added", "emptied")
    emptied_unnamed: list[str] = []
    tallies: dict[str, Counter] = {kind: Counter() for kind in kinds}
    for name in sorted(set(before) & set(after)):
        old, new = read_table(before[name]), read_table(after[name])
        report = compare(old, new, static_columns_of(old.columns))
        if not report["same_keys"]:
            problems.append(f"{name}: lost {report['lost_keys']} keys, gained {report['gained_keys']}")
        tallies["dynamic_differs"].update(report["dynamic_columns_differing"])
        tallies["static_differs"].update(report["static_columns_differing"])
        tallies["emptied"].update(report["static_columns_emptied"])
        site = name[: -len("_daily.csv")]
        entry = EXPECTED_EMPTIED.get(site, {"columns": (), "reason": ""})
        for column in report["static_columns_emptied"]:
            if column in entry["columns"]:
                LOGGER.warning("%s: %s became empty, named: %s", site, column, entry["reason"])
            else:
                emptied_unnamed.append(f"{site}: {column}")
        tallies["removed"].update(dict.fromkeys(report["removed_columns"], len(old)))
        added = set(report["added_columns"])
        tallies["added"].update(dict.fromkeys(added & expected, len(new)))
        tallies["unexpected"].update(dict.fromkeys(added - expected, len(new)))
        tallies["missing"].update(dict.fromkeys(expected - added, len(new)))
    for kind in ("dynamic_differs", "removed", "missing", "unexpected"):
        if tallies[kind]:
            problems.append(f"{kind}: {sorted(tallies[kind])}")
    if emptied_unnamed:
        problems.append(f"static values became empty at {emptied_unnamed}")
    return _report_rows(tallies), problems


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", required=True)
    parser.add_argument("--before", required=True, help="The merge tree on the earlier static files.")
    parser.add_argument("--after", required=True, help="The merge tree on the 1 km features.")
    args = parser.parse_args(argv)

    context = RunContext.for_scale(args.scale, DEFAULT_PROJECT)
    output = context.output_root
    before, after = _daily_files(args.before), _daily_files(args.after)
    if not before or not after:
        LOGGER.error("no daily files: before holds %d, after holds %d", len(before), len(after))
        return 1

    expected = set(read_table(output / "new_static_features.csv", nrows=0).columns) - {"site_name"}
    table, problems = _compare_trees(before, after, expected)
    checks = output / "checks"
    checks.mkdir(parents=True, exist_ok=True)
    table.to_csv(checks / "merge_parity.csv", index=False)
    moved = table.loc[table["kind"] == "static_differs", ["column", "rows"]].to_numpy().tolist()
    LOGGER.info("%d site files compared; static columns that moved, with rows: %s", len(set(before) & set(after)), moved)
    for problem in problems:
        LOGGER.error("%s", problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
