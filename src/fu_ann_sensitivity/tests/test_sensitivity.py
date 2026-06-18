"""Tests for the Eq.-10 perturbation sensitivity and percentile binning (Task 2).

Sensitivities follow Fu et al. 2022's DRYNESS-STRESS convention:
  * SWC sensitivity = response to a -1 SD SWC step (drying)
  * VPD sensitivity = response to a +1 SD VPD step
so a response that rises with both sm and vpd yields d_sm < 0 and d_vpd > 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.fu_ann_sensitivity.ann import train_ensemble
from src.fu_ann_sensitivity.sensitivity import (
    bin_map,
    site_row_sensitivities,
    site_sensitivity_maps,
)

PREDICTORS = ["tair", "vpd", "sm", "ppfd"]


def _fit_synthetic(n=800, seed=1, sm_coef=1.0, vpd_coef=1.0):
    rng = np.random.default_rng(seed)
    vpd = rng.normal(size=n)
    sm = rng.normal(size=n)
    tair = rng.normal(size=n)
    ppfd = rng.normal(size=n)
    y = sm_coef * sm + vpd_coef * vpd + 0.1 * rng.normal(size=n)
    X = np.column_stack([tair, vpd, sm, ppfd])
    models, _ = train_ensemble(X, y, n_repeats=3, seed=7)
    df = pd.DataFrame({"tair": tair, "vpd": vpd, "sm": sm, "ppfd": ppfd})
    return models, df


def test_dryness_stress_sign_convention():
    # Response rises with BOTH sm and vpd -> the two dryness-stress legs must have
    # OPPOSITE signs (catches a both-axes-perturbed-the-same-way bug).
    models, df = _fit_synthetic(sm_coef=1.0, vpd_coef=1.0)
    sm_map, vpd_map = site_sensitivity_maps(models, df, n_bins=5, predictors=PREDICTORS)
    assert np.nanmedian(sm_map) < 0  # drying reduces a response that rises with sm
    assert np.nanmedian(vpd_map) > 0  # response rises with vpd
    assert sm_map.shape == (5, 5)
    assert vpd_map.shape == (5, 5)


def test_row_sensitivities_reused_across_bin_counts():
    models, df = _fit_synthetic()
    d_sm, d_vpd = site_row_sensitivities(models, df, PREDICTORS)
    # Per-model per-row: shape (n_models, n_rows) so binning medians across the ANNs.
    assert d_sm.shape == (len(models), len(df))
    m5 = bin_map(d_sm, df["sm"].to_numpy(), df["vpd"].to_numpy(), 5)
    m10 = bin_map(d_sm, df["sm"].to_numpy(), df["vpd"].to_numpy(), 10)
    assert m5.shape == (5, 5)
    assert m10.shape == (10, 10)
    # Same underlying per-row sensitivities -> consistent overall median sign.
    assert np.sign(np.nanmedian(m5)) == np.sign(np.nanmedian(m10))


def test_empty_cells_are_nan():
    models, df = _fit_synthetic(n=60)
    # 10 bins on 60 points -> many empty cells must be NaN, not 0.
    m = bin_map(np.ones(len(df)), df["sm"].to_numpy(), df["vpd"].to_numpy(), 10)
    assert np.isnan(m).any()
