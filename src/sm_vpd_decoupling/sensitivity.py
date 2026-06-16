# src/sm_vpd_decoupling/sensitivity.py
"""Standardized sensitivity of the response to SM (Liu et al. 2020).

delta(Resp)/delta(SM) per 0.1 m3/m3, computed WITHIN VPD bins (so the SM-VPD
coupling is broken) and averaged over populated VPD bins. Removes the SM-range
effect so the sensitivity is comparable across sites.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.sm_vpd_decoupling.decoupling import (
    MIN_BIN_COUNT,
    MIN_COND_BINS,
    assign_percentile_bins,
)

SM_STEP = 0.1  # m3/m3


def sm_sensitivity(
    site_df: pd.DataFrame,
    response: str,
    n_bins: int,
    vpd_col: str = "vpd",
    sm_col: str = "sm",
    min_bin_count: int = MIN_BIN_COUNT,
    min_cond_bins: int = MIN_COND_BINS,
) -> float:
    """Mean over VPD bins of [d(resp)/d(SM)] * 0.1, using the highest/lowest
    populated SM bins within each VPD bin (Liu approach i, slope form)."""
    df = site_df.copy()
    df["_vpd_bin"] = assign_percentile_bins(df[vpd_col], n_bins)
    df["_sm_bin"] = assign_percentile_bins(df[sm_col], n_bins)
    df = df.dropna(subset=["_vpd_bin", "_sm_bin", response, sm_col])

    grouped = df.groupby(["_vpd_bin", "_sm_bin"])
    resp_mean = grouped[response].mean()
    sm_mean = grouped[sm_col].mean()
    cell_n = grouped.size()
    keep = cell_n >= min_bin_count
    resp_mean = resp_mean[keep]
    sm_mean = sm_mean[keep]

    slopes: list[float] = []
    for _vpd_val, sub in resp_mean.groupby(level=0):
        sub = sub.droplevel(0)
        if sub.index.nunique() < 2:
            continue
        hi_bin, lo_bin = sub.index.max(), sub.index.min()
        sm_hi = float(sm_mean.loc[(_vpd_val, hi_bin)])
        sm_lo = float(sm_mean.loc[(_vpd_val, lo_bin)])
        if sm_hi == sm_lo:
            continue
        slope = (float(sub.loc[hi_bin]) - float(sub.loc[lo_bin])) / (sm_hi - sm_lo)
        slopes.append(slope * SM_STEP)
    if len(slopes) < min_cond_bins:
        return float("nan")
    return float(np.mean(slopes))
