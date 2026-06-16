# src/sm_vpd_decoupling/aggregate.py
"""Aggregate per-site decoupled effects into site tables and dominance summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.sm_vpd_decoupling.decoupling import MIN_BIN_COUNT, decouple_site
from src.sm_vpd_decoupling.sensitivity import sm_sensitivity

_CARRY = ("biome", "pft", "aridity", "canopy_height", "lat", "lon")


def decouple_all_sites(
    table: pd.DataFrame,
    sm_col: str,
    response: str,
    n_bins: int,
    min_valid_days: int,
    vpd_col: str = "vpd",
    site_col: str = "site_name",
    min_bin_count: int = MIN_BIN_COUNT,
) -> pd.DataFrame:
    """Per-site effects + sensitivity for sites with >= ``min_valid_days`` valid rows."""
    rows: list[dict] = []
    for site, g in table.groupby(site_col, sort=True):
        valid = g.dropna(subset=[vpd_col, sm_col, response])
        if len(valid) < min_valid_days:
            continue
        eff = decouple_site(valid, response, n_bins, vpd_col, sm_col, min_bin_count)
        sens = sm_sensitivity(valid, response, n_bins, vpd_col, sm_col, min_bin_count)
        rec: dict = {site_col: site, "n_days": int(len(valid)), **eff, "sensitivity": sens}
        for c in _CARRY:
            if c in g.columns:
                vals = g[c].dropna()
                rec[c] = vals.iloc[0] if len(vals) else np.nan
        rows.append(rec)
    return pd.DataFrame.from_records(rows)


def dominance_summary(effects: pd.DataFrame) -> tuple[float, int]:
    """(% of valid sites where |sm_given_vpd| > |vpd_given_sm|, n valid sites).

    A site is valid only if both effects are finite.
    """
    valid = effects.dropna(subset=["sm_given_vpd", "vpd_given_sm"])
    n = len(valid)
    if n == 0:
        return float("nan"), 0
    sm_wins = (valid["sm_given_vpd"].abs() > valid["vpd_given_sm"].abs()).sum()
    return float(sm_wins) / n * 100.0, n
