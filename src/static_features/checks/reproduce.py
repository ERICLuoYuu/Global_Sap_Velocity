"""Unchanged products must reproduce the current files: python -m src.static_features.checks.reproduce.

A site that was empty and now has a value is a swallowed failure in today's code, not a pass. It stops the check and
goes to the owner, because it changes the merge's st_age fallback. A difference the owner has seen is named below, is
still reported on every run, and does not stop the check; anything else does.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from src.static_features.results import ExtractionFailure, RunContext, read_part
from src.static_features.run import DEFAULT_PROJECT
from src.static_features.sites import site_list_hash

LOGGER = logging.getLogger(__name__)

MATCH = "match"
DIFFERS = "differs"
EMPTY_TO_VALUE = "empty_to_value"
VALUE_TO_EMPTY = "value_to_empty"

_DEVILLA = {
    "outcomes": (DIFFERS, EMPTY_TO_VALUE),
    "reason": "Coordinate corrected, 4b9eaf8. The earlier point is water, with no BGI pixel, so the merge fell back on "
    "ground st_age 41; the forest pixel gives 54-56 years. Reported to the owner on 2026-11-03.",
}

_GUIANA = {
    "outcomes": (DIFFERS, EMPTY_TO_VALUE),
    "products": ("fan",),
    "reason": "Fan read the NAMERICA file, the first box holding the site, which has no land there; the reader now "
    "goes on to the SAMERICA box, whose file holds the site's own pixel (owner, 2026-11-04).",
}

#: Sites allowed to differ from the current files, the outcomes allowed there, and why. An entry naming products
#: allows the difference in those products only.
EXPECTED_DIFFERENCES = {
    "CHE_DAV_SEE": {"outcomes": (DIFFERS,), "reason": "Coordinate corrected, 4b9eaf8."},
    "CHN_YUN_YUN": {
        "outcomes": (DIFFERS,),
        "reason": "WorldClim neighbour: the current file came from an earlier corner-first search (ledger, Task 6).",
    },
    "GBR_DEV_CON": _DEVILLA,
    "GBR_DEV_DRO": _DEVILLA,
    "GUF_GUY_GUY": _GUIANA,
    "GUF_GUY_ST2": _GUIANA,
}

#: Columns a product gained after the current files were written: nothing to compare them with. Any other column the
#: current file lacks stops the check, so a renamed column cannot pass unseen.
NEW_COLUMNS = {"stand_age": ("stand_age_neighbour",)}

CURRENT_FILES = {
    "worldclim": ("env_site_data", "site_info_with_env_data.csv"),
    "stand_age": ("terrain_site_data", "stand_age_data.csv"),
    "fan": ("root_depth_site_data", "site_info_with_root_depth_fan2017.csv"),
}


def classify(old: pd.Series, new: pd.Series, rtol: float = 1e-6) -> pd.Series:
    """Per row: match, differs, empty_to_value or value_to_empty. Both empty is a match. Text compares exactly."""
    old = old.reset_index(drop=True)
    new = new.reset_index(drop=True)
    old_empty, new_empty = old.isna(), new.isna()
    if pd.api.types.is_numeric_dtype(old) and pd.api.types.is_numeric_dtype(new):
        old_numbers, new_numbers = old.astype(float), new.astype(float)
        same = (old_numbers - new_numbers).abs() <= rtol * np.maximum(old_numbers.abs(), new_numbers.abs())
    else:
        same = old.astype(str) == new.astype(str)

    outcome = pd.Series(DIFFERS, index=old.index, dtype=object)
    outcome[same & ~old_empty & ~new_empty] = MATCH
    outcome[old_empty & new_empty] = MATCH
    outcome[old_empty & ~new_empty] = EMPTY_TO_VALUE
    outcome[~old_empty & new_empty] = VALUE_TO_EMPTY
    return outcome


def _compare_product(product: str, context: RunContext, scale: str) -> pd.DataFrame:
    part, site_hash = read_part(context.parts_root, product)
    if site_hash != site_list_hash(context.sites):
        raise ExtractionFailure(f"{product}: the part was built from another site list; rerun the product")
    directory, name = CURRENT_FILES[product]
    current_path: Path = context.output_root.parents[1] / directory / scale / name
    current = pd.read_csv(current_path, float_precision="round_trip", keep_default_na=False, na_values=[""])
    current = current.set_index("site_name").reindex(part.values.index)

    absent = [column for column in part.values.columns if column not in current.columns]
    unnamed = sorted(set(absent) - set(NEW_COLUMNS.get(product, ())))
    if unnamed:
        raise ExtractionFailure(f"{product}: {current_path} holds no {unnamed}, and they are not named as new")
    for column in absent:
        LOGGER.info("%s: %s is new since the current files, not compared", product, column)

    rows = []
    for column in [column for column in part.values.columns if column in current.columns]:
        outcome = classify(current[column], part.values[column])
        rows.append(
            pd.DataFrame(
                {
                    "product": product,
                    "column": column,
                    "site_name": part.values.index,
                    "old": current[column].to_numpy(),
                    "new": part.values[column].to_numpy(),
                    "outcome": outcome.to_numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", required=True)
    args = parser.parse_args(argv)

    context = RunContext.for_scale(args.scale, DEFAULT_PROJECT)
    try:
        table = pd.concat(
            [_compare_product(product, context, args.scale) for product in CURRENT_FILES], ignore_index=True
        )
    except ExtractionFailure as failure:
        LOGGER.error("%s", failure)
        return 1

    summary = table.groupby(["product", "column", "outcome"]).size().unstack(fill_value=0).reset_index()
    checks = context.output_root / "checks"
    checks.mkdir(parents=True, exist_ok=True)
    summary.to_csv(checks / "reproduce_summary.csv", index=False)
    different = table[table["outcome"] != MATCH]
    different.to_csv(checks / "reproduce.csv", index=False)
    LOGGER.info("columns compared: %d, sites: %d", table["column"].nunique(), table["site_name"].nunique())

    if different.empty:
        LOGGER.info("every site matches in every column")
        return 0
    unexpected = 0
    for row in different.itertuples(index=False):
        entry = EXPECTED_DIFFERENCES.get(row.site_name)
        detail = f"{row.outcome} {row.column} {row.site_name}: {row.product} ({row.old} -> {row.new})"
        if entry is not None and row.outcome in entry["outcomes"] and row.product in entry.get("products", (row.product,)):
            LOGGER.warning("%s, named: %s", detail, entry["reason"])
        else:
            LOGGER.error("%s", detail)
            unexpected += 1
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
