"""Agreement between sources of one quantity, overall and by si_igbp. A report, not a gate: both sources are kept.

python -m src.static_features.checks.agreement --scale <scale>. Writes checks/agreement.csv. Sites sharing a pixel are
collapsed first, so duplicates cannot inflate the fit.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from src.static_features.results import ExtractionFailure, RunContext, read_table
from src.static_features.run import DEFAULT_PROJECT

LOGGER = logging.getLogger(__name__)

PAIRS = (
    ("root_depth_lusk", "root_depth_han"),
    ("root_depth_lusk", "root_depth"),
    ("root_depth_han", "root_depth"),
    ("wood_density_lusk", "wood_density"),
)
EXPECTED_UNITS = {"root_depth_lusk": "m", "wood_density_lusk": "g cm3"}  # the files' spelling of g cm-3
STATISTICS = ("pearson", "spearman", "bias", "rmse")
DRAWS = 2000
ALL_SITES = "all"


def check_units(provenance: Mapping[str, Any]) -> None:
    """Lusk's unit tags come from its files, so a wrong one stops the comparison before any number is reported."""
    traits = provenance["products"]["lusk"]["traits"]
    units = {trait["column"]: trait["unit"] for trait in traits.values()}
    wrong = {column: units.get(column) for column, unit in EXPECTED_UNITS.items() if units.get(column) != unit}
    if wrong:
        raise ExtractionFailure(f"agreement: Lusk's unit tags read {wrong}, expected {EXPECTED_UNITS}")


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def _statistics(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    difference = a - b
    return (
        _pearson(a, b),
        _pearson(rankdata(a), rankdata(b)),
        float(difference.mean()),
        math.sqrt(float((difference**2).mean())),
    )


def _interval(draws: np.ndarray) -> tuple[float, float]:
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return math.nan, math.nan
    return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def pair_stats(a: pd.Series, b: pd.Series, seed: int = 42, draws: int = 2000) -> dict[str, float]:
    """Pearson and Spearman r, bias of a - b, and RMSE, each with a 2.5 to 97.5 bootstrap interval over distinct pairs."""
    pairs = pd.DataFrame({"a": a, "b": b}).dropna().drop_duplicates()
    n = len(pairs)
    if n == 0:
        empty = {name: math.nan for name in STATISTICS}
        return {
            "n": 0,
            **empty,
            **{f"{name}_low": math.nan for name in STATISTICS},
            **{f"{name}_high": math.nan for name in STATISTICS},
        }
    a_values, b_values = pairs["a"].to_numpy(), pairs["b"].to_numpy()
    stats: dict[str, float] = {"n": n}
    estimate = _statistics(a_values, b_values)

    rng = np.random.default_rng(seed)
    resampled = np.empty((draws, len(STATISTICS)))
    for draw in range(draws):
        chosen = rng.integers(0, n, n)
        resampled[draw] = _statistics(a_values[chosen], b_values[chosen])
    for position, name in enumerate(STATISTICS):
        low, high = _interval(resampled[:, position])
        stats[name], stats[f"{name}_low"], stats[f"{name}_high"] = estimate[position], low, high
    return stats


def report(frame: pd.DataFrame, pairs: Sequence[tuple[str, str]], igbp: pd.Series, draws: int = 2000) -> pd.DataFrame:
    """One row per pair, overall and per class. A site without a class counts only overall."""
    rows = []
    groups = sorted(igbp.dropna().unique())
    for first, second in pairs:
        rows.append(
            {"a": first, "b": second, "group": ALL_SITES, **pair_stats(frame[first], frame[second], draws=draws)}
        )
        for group in groups:
            members = igbp.index[igbp == group]
            stats = pair_stats(frame.loc[members, first], frame.loc[members, second], draws=draws)
            rows.append({"a": first, "b": second, "group": group, **stats})
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", required=True)
    args = parser.parse_args(argv)

    context = RunContext.for_scale(args.scale, DEFAULT_PROJECT)
    output = context.output_root
    try:
        check_units(json.loads((output / "provenance.json").read_text(encoding="utf-8")))
        new = read_table(output / "new_static_features.csv", index_col="site_name")
        fan = read_table(output / "site_info_with_root_depth_fan2017.csv", index_col="site_name")
    except (ExtractionFailure, OSError, KeyError) as failure:
        LOGGER.error("%s", failure)
        return 1

    table = report(new.join(fan["root_depth"]), PAIRS, context.sites.set_index("site_name")["si_igbp"], draws=DRAWS)
    checks = output / "checks"
    checks.mkdir(parents=True, exist_ok=True)
    table.to_csv(checks / "agreement.csv", index=False)
    for row in table[table["group"] == ALL_SITES].itertuples(index=False):
        LOGGER.info(
            "%s vs %s: n=%d r=%.3f rho=%.3f bias=%.3f rmse=%.3f",
            row.a,
            row.b,
            row.n,
            row.pearson,
            row.spearman,
            row.bias,
            row.rmse,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
