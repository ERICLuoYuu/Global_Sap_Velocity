"""Binned-percentile driver decoupling (Liu et al. 2024, GRL — SI Text S2).

Breaks the VPD–Tair–SM multicollinearity by per-site decile binning, then isolates
each driver's effect on ΔSF while holding another constant:

- ``ΔSF(VPD|Tair)`` = mean over populated Tair bins of [ΔSF at highest − lowest populated VPD bin]
- ``ΔSF(Tair|VPD)`` = mean over populated VPD bins of [high Tair − low Tair]
- ``ΔSF(VPD|SM)``  = mean over populated SM bins of [high VPD − low VPD]
- ``ΔSF(SM|VPD)``  = mean over populated VPD bins of [**low SM − high SM**]  (Eq. 2: low SM is the stressor)

The high/low bins are the highest/lowest *populated* bins within each conditioning bin
(not fixed deciles), matching Eq. 1–2. Effects are computed per site, then aggregated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def assign_deciles(values: pd.Series, n_bins: int = 10) -> pd.Series:
    """Per-series decile bin index (0..n_bins-1) via percentile thresholds.

    Uses ``qcut`` with ``duplicates="drop"`` so low-variance series collapse to
    fewer bins instead of raising; a degenerate (single-value) series → all bin 0.
    """
    v = pd.to_numeric(values, errors="coerce")
    try:
        codes = pd.qcut(v, q=n_bins, labels=False, duplicates="drop")
    except (ValueError, IndexError):
        codes = pd.Series(np.nan, index=v.index)
    codes = pd.Series(np.asarray(codes, dtype="float64"), index=v.index)
    # A degenerate (≤1 unique value) series makes qcut return all-NaN without
    # raising; collapse the finite observations to a single bin (no gradient).
    if codes.isna().all():
        codes = pd.Series(np.where(v.notna(), 0.0, np.nan), index=v.index)
    return codes


def decouple_effect(
    df: pd.DataFrame,
    driver_bin: str,
    cond_bin: str,
    response: str,
    low_minus_high: bool = False,
) -> float:
    """Mean over populated conditioning bins of (high − low) driver-bin response.

    With ``low_minus_high=True`` the sign is flipped (low − high) — used for the SM
    effect, where *low* soil moisture is the stressor (Liu Eq. 2). Conditioning bins
    containing fewer than two distinct driver bins are skipped.
    """
    cell_means = df.groupby([cond_bin, driver_bin])[response].mean()
    effects: list[float] = []
    for _cond_val, sub in cell_means.groupby(level=0):
        sub = sub.droplevel(0)
        if sub.index.nunique() < 2:
            continue
        high = float(sub.loc[sub.index.max()])
        low = float(sub.loc[sub.index.min()])
        effects.append(low - high if low_minus_high else high - low)
    return float(np.mean(effects)) if effects else float("nan")


def decouple_site(
    site_df: pd.DataFrame,
    response: str = "delta_sf",
    n_bins: int = 10,
    vpd_col: str = "vpd",
    tair_col: str = "tair",
    sm_col: str = "sm",
) -> dict[str, float]:
    """All four decoupled effects for a single site's site-day records."""
    df = site_df.copy()
    df["_vpd_bin"] = assign_deciles(df[vpd_col], n_bins)
    df["_tair_bin"] = assign_deciles(df[tair_col], n_bins)
    df["_sm_bin"] = assign_deciles(df[sm_col], n_bins)
    df = df.dropna(subset=["_vpd_bin", "_tair_bin", "_sm_bin", response])
    return {
        "vpd_given_tair": decouple_effect(df, "_vpd_bin", "_tair_bin", response),
        "tair_given_vpd": decouple_effect(df, "_tair_bin", "_vpd_bin", response),
        "vpd_given_sm": decouple_effect(df, "_vpd_bin", "_sm_bin", response),
        "sm_given_vpd": decouple_effect(df, "_sm_bin", "_vpd_bin", response, low_minus_high=True),
    }


def decouple_all_sites(
    table: pd.DataFrame,
    site_col: str = "site_name",
    response: str = "delta_sf",
    n_bins: int = 10,
    min_valid_days: int = 120,
    vpd_col: str = "vpd",
    tair_col: str = "tair",
    sm_col: str = "sm",
) -> pd.DataFrame:
    """Per-site decoupled effects for sites with ≥ ``min_valid_days`` valid records.

    Returns one row per qualifying site with the four effects plus a carried PFT
    (first non-null) for downstream stratification. Sites below the threshold are
    skipped (reported by the caller).
    """
    needed = [vpd_col, tair_col, sm_col, response]
    rows: list[dict] = []
    for site, g in table.groupby(site_col, sort=True):
        valid = g.dropna(subset=needed)
        if len(valid) < min_valid_days:
            continue
        effects = decouple_site(
            valid, response=response, n_bins=n_bins, vpd_col=vpd_col, tair_col=tair_col, sm_col=sm_col
        )
        rec: dict = {site_col: site, "n_days": int(len(valid)), **effects}
        if "pft" in g.columns:
            pft = g["pft"].dropna()
            rec["pft"] = pft.iloc[0] if len(pft) else np.nan
        rows.append(rec)
    return pd.DataFrame.from_records(rows)
