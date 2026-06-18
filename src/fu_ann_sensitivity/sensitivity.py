# src/fu_ann_sensitivity/sensitivity.py
"""Eq.-10 perturbation sensitivity on the trained ANN, binned by SWC x VPD.

Inputs are z-scored, so +/-1 SD is +/-1.0. Sensitivities use Fu et al. 2022's
DRYNESS-STRESS convention, so signs match the paper (Fig. 2 caption: "positive
signs mean GPP increases when SWC becomes drier") AND the sibling
sm_vpd_decoupling module (which reports resp(dry)-resp(wet) and resp(highVPD)-
resp(lowVPD)):

  * VPD sensitivity = response to a +1 SD VPD increase:   [R(vpd+1) - R(base)] / 1
  * SWC sensitivity = response to a +1 SD SWC DECREASE (drying):
        [R(swc-1) - R(base)] / 1   (= -dR/dSWC)

Sign: NEGATIVE => response reduced under stress (drier soil / drier air);
POSITIVE for SWC => response rises as soil dries (Fu's high-SWC compensation).
The two axes are perturbed in OPPOSITE directions because increasing dryness
means LOWER SWC but HIGHER VPD.

Median nesting follows Fu's stated order (Methods, Eq. 10 paragraph + "the median
of these were used at each site"): PER ANN, take the within-bin median of per-row
sensitivities; THEN take the median across the 5 ANNs. So per-row sensitivities
are kept per-model (shape (n_models, n_rows)); ``bin_map`` / ``bin_1d`` bin each
model then median across models. Computed once per site and reused across bin
counts (5x5 and 10x10).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.fu_ann_sensitivity.ann import ensemble_predict
from src.sm_vpd_decoupling.decoupling import assign_percentile_bins

SIGMA = 1.0


def _nanmedian_quiet(arr: np.ndarray, axis: int) -> np.ndarray:
    """``np.nanmedian`` that does not warn on all-NaN slices (empty bins -> NaN by design)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN slice encountered", RuntimeWarning)
        return np.nanmedian(arr, axis=axis)


def _row_sensitivity(models, X: np.ndarray, col: int, step: float) -> np.ndarray:
    """Per-MODEL per-row [R_m(x + step*e_col) - R_m(x)] / |step|. Shape (n_models, n_rows).

    ``step`` carries the stress direction: -SIGMA for SWC (drying), +SIGMA for VPD.
    Dividing by the magnitude |step| keeps "negative = stress reduces response".
    Cross-model reduction is deferred to binning (Fu's median order).
    """
    x_pert = X.copy()
    x_pert[:, col] += step
    rows = [(ensemble_predict([m], x_pert) - ensemble_predict([m], X)) / abs(step) for m in models]
    return np.stack(rows, axis=0)


def site_row_sensitivities(models, df: pd.DataFrame, predictors, sm_col: str = "sm", vpd_col: str = "vpd"):
    """Per-model per-row dryness-stress sensitivities (d_sm, d_vpd), each (n_models, n_rows).

    d_sm uses a -1 SD (drying) step; d_vpd uses a +1 SD step.
    """
    X = df[predictors].to_numpy(dtype=np.float64)
    sm_idx, vpd_idx = predictors.index(sm_col), predictors.index(vpd_col)
    d_sm = _row_sensitivity(models, X, sm_idx, -SIGMA)
    d_vpd = _row_sensitivity(models, X, vpd_idx, +SIGMA)
    return d_sm, d_vpd


def bin_map(values: np.ndarray, sm_vals, vpd_vals, n_bins: int) -> np.ndarray:
    """Median into an n_bins x n_bins SWC(axis0) x VPD(axis1) percentile grid.

    ``values`` is (n_rows,) or (n_models, n_rows): each model's within-cell median
    is taken, then medianed across models (Fu's order). Empty cells -> NaN.
    """
    values = np.atleast_2d(np.asarray(values, dtype=np.float64))
    sm_bin = assign_percentile_bins(pd.Series(np.asarray(sm_vals)), n_bins).to_numpy()
    vpd_bin = assign_percentile_bins(pd.Series(np.asarray(vpd_vals)), n_bins).to_numpy()
    per_model = np.full((values.shape[0], n_bins, n_bins), np.nan)
    for k in range(values.shape[0]):
        for i in range(n_bins):
            for j in range(n_bins):
                cell = (sm_bin == i) & (vpd_bin == j)
                if cell.sum() > 0:
                    per_model[k, i, j] = np.nanmedian(values[k, cell])
    return _nanmedian_quiet(per_model, axis=0)


def bin_1d(values: np.ndarray, axis_vals, n_bins: int) -> np.ndarray:
    """Median into n_bins along ONE axis (for Fig 3). Same per-model-then-across rule."""
    values = np.atleast_2d(np.asarray(values, dtype=np.float64))
    a_bin = assign_percentile_bins(pd.Series(np.asarray(axis_vals)), n_bins).to_numpy()
    per_model = np.full((values.shape[0], n_bins), np.nan)
    for k in range(values.shape[0]):
        for i in range(n_bins):
            cell = a_bin == i
            if cell.sum() > 0:
                per_model[k, i] = np.nanmedian(values[k, cell])
    return _nanmedian_quiet(per_model, axis=0)


def site_sensitivity_maps(models, df: pd.DataFrame, n_bins: int, predictors, sm_col: str = "sm", vpd_col: str = "vpd"):
    """Convenience: (sm_map, vpd_map) at one bin count. Empty cells -> NaN."""
    d_sm, d_vpd = site_row_sensitivities(models, df, predictors, sm_col, vpd_col)
    sm_vals, vpd_vals = df[sm_col].to_numpy(), df[vpd_col].to_numpy()
    return (bin_map(d_sm, sm_vals, vpd_vals, n_bins), bin_map(d_vpd, sm_vals, vpd_vals, n_bins))
