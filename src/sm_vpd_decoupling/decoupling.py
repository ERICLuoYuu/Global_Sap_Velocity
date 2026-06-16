# src/sm_vpd_decoupling/decoupling.py
"""Nested percentile-binning decoupling of SM and VPD (Liu et al. 2020, Eqs. 1-2).

For each site, bin SM and VPD into per-site percentiles, then measure each
driver's effect on the response WITHIN bins of the other (where residual SM-VPD
correlation is ~0):

  dResp(SM|VPD) = mean over populated VPD bins of [resp(lowest SM bin)
                  - resp(highest SM bin)]      (low SM is the stressor; Eq. 2)
  dResp(VPD|SM) = mean over populated SM bins of [resp(highest VPD bin)
                  - resp(lowest VPD bin)]      (Eq. 1)

A driver cell counts only with >= MIN_BIN_COUNT points; a site effect needs
>= MIN_COND_BINS populated conditioning bins, else NaN (reported insufficient).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_BIN_COUNT = 3
MIN_COND_BINS = 2


def assign_percentile_bins(values: pd.Series, n_bins: int) -> pd.Series:
    """Per-series percentile bin index (0..n_bins-1) via ``qcut`` (duplicates dropped).

    Low-variance series collapse to fewer bins; a single-value series -> bin 0.
    """
    v = pd.to_numeric(values, errors="coerce")
    try:
        codes = pd.qcut(v, q=n_bins, labels=False, duplicates="drop")
    except (ValueError, IndexError):
        codes = pd.Series(np.nan, index=v.index)
    codes = pd.Series(np.asarray(codes, dtype="float64"), index=v.index)
    if codes.isna().all():
        codes = pd.Series(np.where(v.notna(), 0.0, np.nan), index=v.index)
    return codes


def decouple_effect(
    df: pd.DataFrame,
    driver_bin: str,
    cond_bin: str,
    response: str,
    low_minus_high: bool = False,
    min_bin_count: int = MIN_BIN_COUNT,
    min_cond_bins: int = MIN_COND_BINS,
) -> float:
    """Mean over populated conditioning bins of (high - low) driver-bin response.

    ``low_minus_high=True`` flips the sign (low - high), used for SM where low SM
    is the stressor (Liu Eq. 2). Cells with < ``min_bin_count`` points are dropped;
    < ``min_cond_bins`` populated conditioning bins -> NaN.
    """
    grouped = df.groupby([cond_bin, driver_bin])[response]
    cell_mean = grouped.mean()
    cell_n = grouped.size()
    cell_mean = cell_mean[cell_n >= min_bin_count]
    effects: list[float] = []
    for _cond_val, sub in cell_mean.groupby(level=0):
        sub = sub.droplevel(0)
        if sub.index.nunique() < 2:
            continue
        high = float(sub.loc[sub.index.max()])
        low = float(sub.loc[sub.index.min()])
        effects.append(low - high if low_minus_high else high - low)
    if len(effects) < min_cond_bins:
        return float("nan")
    return float(np.mean(effects))


def decouple_site(
    site_df: pd.DataFrame,
    response: str,
    n_bins: int,
    vpd_col: str = "vpd",
    sm_col: str = "sm",
    min_bin_count: int = MIN_BIN_COUNT,
) -> dict[str, float]:
    """Both decoupled effects for one site's site-day records (VPD-vs-SM pair)."""
    df = site_df.copy()
    df["_vpd_bin"] = assign_percentile_bins(df[vpd_col], n_bins)
    df["_sm_bin"] = assign_percentile_bins(df[sm_col], n_bins)
    df = df.dropna(subset=["_vpd_bin", "_sm_bin", response])
    return {
        "sm_given_vpd": decouple_effect(
            df, "_sm_bin", "_vpd_bin", response, low_minus_high=True, min_bin_count=min_bin_count
        ),
        "vpd_given_sm": decouple_effect(
            df, "_vpd_bin", "_sm_bin", response, low_minus_high=False, min_bin_count=min_bin_count
        ),
    }
