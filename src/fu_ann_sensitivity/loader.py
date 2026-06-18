# src/fu_ann_sensitivity/loader.py
"""Adapt the sm_vpd_decoupling site-day table to the Fu et al. 2022 ANN inputs.

Reuses ``sm_vpd_decoupling.loader.load_table`` (same Liu active-day filter:
Tair>15, VPD>0.5, PPFD>500 ~= SW_in>250 W m-2 -- matching Fu's growing-season
screen) and adds:
  * per-site z-scoring of predictors/response (so a +/-1 SD perturbation = +/-1);
  * Fu's site-type exclusion (cropland & wetland: management / fluctuating water
    table -- paper Methods lines 855-862, 905).
"""

from __future__ import annotations

import logging

import pandas as pd

from src.sm_vpd_decoupling.loader import load_table as _load_table  # re-exported

logger = logging.getLogger(__name__)

PREDICTORS = ["tair", "vpd", "sm", "ppfd"]
# Fu et al. 2022 excluded cropland & wetland sites. PFT codes per CLAUDE.md
# feature list (WET present; CRO/CROP guarded in case they ever appear).
EXCLUDE_PFT = ("CRO", "CROP", "WET")


def zscore_per_site(df: pd.DataFrame, cols, response: str, site_col: str = "site_name") -> pd.DataFrame:
    """Append ``<col>_z`` columns, z-scored within each site (ddof=0).

    Zero-variance columns within a site yield NaN (caught downstream). ``response``
    is accepted explicitly so callers state intent; it must be included in ``cols``.
    """
    out = df.copy()
    for c in cols:
        g = out.groupby(site_col)[c]
        mu = g.transform("mean")
        sd = g.transform("std", ddof=0)
        out[f"{c}_z"] = (out[c] - mu) / sd.where(sd > 0)
    return out


def exclude_site_types(table: pd.DataFrame, exclude_pft=EXCLUDE_PFT) -> pd.DataFrame:
    """Drop cropland/wetland rows (Fu exclusion). No-op if no 'pft' column or empty list."""
    if "pft" not in table.columns or not exclude_pft:
        return table
    mask = table["pft"].astype(str).str.upper().isin({p.upper() for p in exclude_pft})
    if mask.any():
        dropped = sorted(table.loc[mask, "site_name"].unique())
        logger.info("Excluding %d cropland/wetland sites: %s", len(dropped), dropped)
    return table[~mask].copy()


def load_table(*args, exclude_pft=EXCLUDE_PFT, **kwargs):
    """Sibling loader + Fu cropland/wetland exclusion (pass ``exclude_pft=()`` to keep all)."""
    return exclude_site_types(_load_table(*args, **kwargs), exclude_pft=exclude_pft)
