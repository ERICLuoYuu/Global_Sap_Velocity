"""Random-forest ±1 SD perturbation sensitivity (Liu et al. 2024, SI Text S3).

Model-based robustness cross-check for the binned decoupling: fit ``n_models`` random
forests (each on a ``train_frac`` split of ONE pooled, all-sites table — matching Liu's
"60% of the data … 100 RF models") of ΔSF on the climate/vegetation predictors, then
perturb one predictor by ±1 SD and measure the mean change in predicted ΔSF.
VPD/Tair/radiation/LAI are increased by 1 SD; SM is *decreased* by 1 SD (low SM is the
stressor), matching Eq. 3.

``sd_scope`` sets the size of the 1-SD nudge:
  * ``"global"`` — one SD per predictor over the whole pooled dataset (literal text reading).
  * ``"site"``   — each row is nudged by its OWN site's within-site SD (per-pixel-consistent
    with the decoupling). This matters because SM varies far more BETWEEN sites
    (desert→rainforest) than WITHIN a site (soil is buffered), so a global-SD nudge injects
    the between-site climate gradient into SM's "sensitivity" and over-ranks it; the
    within-site nudge isolates the temporal sensitivity the per-pixel decoupling measures.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

DEFAULT_PREDICTORS = ("vpd", "tair", "sm", "rad", "lai")


def rf_sensitivity(
    table: pd.DataFrame,
    predictors: Iterable[str] = DEFAULT_PREDICTORS,
    response: str = "delta_sf",
    n_models: int = 100,
    train_frac: float = 0.6,
    decrease_for: Iterable[str] = ("sm",),
    n_estimators: int = 100,
    random_state: int = 42,
    min_samples: int = 50,
    site_col: str = "site_name",
    sd_scope: str = "global",
) -> dict[str, float]:
    """Mean predicted-ΔSF change per predictor after a ±1 SD perturbation.

    ``sd_scope="site"`` nudges each row by its within-site SD (per-pixel sensitivity),
    falling back to the global SD for sites with a degenerate (NaN/zero) SD. Returns a
    predictor→sensitivity dict (NaN for every predictor if fewer than ``min_samples``
    complete rows are available).
    """
    predictors = [p for p in predictors if p in table.columns]
    decrease_for = set(decrease_for)
    per_site = sd_scope == "site" and site_col in table.columns
    cols = [*predictors, response, site_col] if per_site else [*predictors, response]
    data = table[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=[*predictors, response])
    if len(data) < min_samples:
        return {p: float("nan") for p in predictors}

    x = data[predictors].to_numpy(dtype=float)
    y = data[response].to_numpy(dtype=float)
    gsd = x.std(axis=0, ddof=0)
    if per_site:
        # per-row step = that row's within-site SD; fall back to the global SD where a
        # site has a NaN (single obs) or zero SD so it still gets a meaningful nudge.
        site_sd = data.groupby(site_col)[predictors].transform("std").to_numpy(dtype=float)
        step = np.where(~np.isfinite(site_sd) | (site_sd == 0.0), gsd, site_sd)
    else:
        step = np.broadcast_to(gsd, x.shape)
    signs = np.array([-1.0 if p in decrease_for else 1.0 for p in predictors])
    accum: dict[str, list[float]] = {p: [] for p in predictors}

    idx = np.arange(len(x))
    for m in range(n_models):
        seed = random_state + m
        i_tr, i_te = train_test_split(idx, train_size=train_frac, random_state=seed)
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
        rf.fit(x[i_tr], y[i_tr])
        base = rf.predict(x[i_te])
        for j, p in enumerate(predictors):
            perturbed = x[i_te].copy()
            perturbed[:, j] += signs[j] * step[i_te, j]
            accum[p].append(float(np.mean(rf.predict(perturbed) - base)))

    return {p: float(np.mean(v)) for p, v in accum.items()}
