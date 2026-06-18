# src/fu_ann_sensitivity/aggregate.py
"""Cross-site aggregation of per-site Fu sensitivities + significance testing.

Two-stage so the (expensive) ANN training/perturbation runs ONCE per site and
both bin counts (5x5, 10x10) reuse it:

  fit_site_sensitivities(...)  -> per-site bundles (train ANN, perturb), drop r<0.5
  aggregate_at_nbins(per_site, n_bins) -> 2-D maps + per-cell t-tests (Fig 2) and
                                          1-D dual-leg tables + per-bin t-tests (Fig 3)

Median nesting (per-ANN within-bin median, then across the 5 ANNs) is handled
inside ``bin_map``/``bin_1d``; this module then medians across sites and runs the
across-site 1-sample t-tests Fu uses for significance (Methods lines 1065-1066).
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from src.fu_ann_sensitivity.ann import train_ensemble
from src.fu_ann_sensitivity.sensitivity import bin_1d, bin_map, site_row_sensitivities

logger = logging.getLogger(__name__)

MIN_SITES_TTEST = 5  # min finite per-site values before a bin's t-test is meaningful
P_THRESHOLD = 0.05


def fit_site_sensitivities(
    table: pd.DataFrame,
    response: str,
    sm_col: str,
    vpd_col: str,
    predictors,
    min_valid_days: int,
    n_repeats: int,
    r_threshold: float = 0.5,
    site_col: str = "site_name",
    pft_col: str = "pft",
    seed: int = 0,
):
    """Per site: train the ANN ensemble and compute dryness-stress sensitivities.

    Columns named here are the z-scored columns. Sites with fewer than
    ``min_valid_days`` complete rows, or pred-vs-obs (test-set) r < ``r_threshold``,
    are skipped. Returns ``(per_site, n_dropped, median_r)`` where each bundle holds
    per-model per-row ``d_sm``/``d_vpd`` (shape (n_models, n_rows)) and the binning
    axes ``sm_vals``/``vpd_vals``.
    """
    needed = list(dict.fromkeys([response, *predictors, sm_col, vpd_col]))
    per_site, dropped, rs = [], 0, []
    for site, sdf in table.groupby(site_col):
        sub = sdf.dropna(subset=needed)
        if len(sub) < min_valid_days:
            continue
        X = sub[predictors].to_numpy(dtype=np.float64)
        y = sub[response].to_numpy(dtype=np.float64)
        models, r = train_ensemble(X, y, n_repeats=n_repeats, seed=seed)
        rs.append(r)
        if not np.isfinite(r) or r < r_threshold:
            dropped += 1
            continue
        d_sm, d_vpd = site_row_sensitivities(models, sub, predictors, sm_col=sm_col, vpd_col=vpd_col)
        per_site.append(
            {
                "site": site,
                "pft": str(sub[pft_col].iloc[0]) if pft_col in sub.columns else "NA",
                "d_sm": d_sm,
                "d_vpd": d_vpd,
                "sm_vals": sub[sm_col].to_numpy(dtype=np.float64),
                "vpd_vals": sub[vpd_col].to_numpy(dtype=np.float64),
                "r": r,
                "n_days": int(len(sub)),
            }
        )
    median_r = float(np.median(rs)) if rs else float("nan")
    logger.info("fit_site_sensitivities: kept %d sites, dropped %d (median r=%.3f)", len(per_site), dropped, median_r)
    return per_site, dropped, median_r


def _sig_mask(stack: np.ndarray, min_sites: int) -> np.ndarray:
    """Per-cell 1-sample t-test (H0: mean=0) over the leading (site) axis.

    ``stack`` shape (n_sites, *cell_shape). A cell is significant when it has
    >= ``min_sites`` finite values and the two-sided p < P_THRESHOLD.
    """
    cell_shape = stack.shape[1:]
    flat = stack.reshape(stack.shape[0], -1)
    sig = np.zeros(flat.shape[1], dtype=bool)
    for c in range(flat.shape[1]):
        vals = flat[:, c]
        vals = vals[np.isfinite(vals)]
        if len(vals) < min_sites:
            continue
        if np.ptp(vals) == 0:
            # Zero variance across sites: t -> inf, so significant iff the
            # (constant) mean is non-zero. scipy would return NaN here.
            sig[c] = vals[0] != 0.0
        else:
            sig[c] = stats.ttest_1samp(vals, 0.0).pvalue < P_THRESHOLD
    return sig.reshape(cell_shape)


def _nanmedian_quiet(stack: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN slice encountered", RuntimeWarning)
        return np.nanmedian(stack, axis=0)


def _nanpercentile_quiet(stack: np.ndarray, q: float) -> np.ndarray:
    """``np.nanpercentile`` over the site axis, silent on all-NaN bins (NaN by design)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN slice encountered", RuntimeWarning)
        return np.nanpercentile(stack, q, axis=0)


def _leg_table(per_site, axis_key: str, n_bins: int, min_sites: int) -> pd.DataFrame:
    """1-D dual-leg table: both sensitivities binned along ONE axis (Fig 3).

    ``axis_key`` is 'sm_vals' (Fig 3a, x = SWC bin) or 'vpd_vals' (Fig 3b, x = VPD bin).
    """
    sm_stack = np.stack([bin_1d(p["d_sm"], p[axis_key], n_bins) for p in per_site], axis=0)
    vpd_stack = np.stack([bin_1d(p["d_vpd"], p[axis_key], n_bins) for p in per_site], axis=0)
    axis_name = "sm_bin" if axis_key == "sm_vals" else "vpd_bin"
    out = {axis_name: np.arange(n_bins)}
    for leg, stack in [("d_sm", sm_stack), ("d_vpd", vpd_stack)]:
        out[f"{leg}_median"] = _nanmedian_quiet(stack)
        out[f"{leg}_q25"] = _nanpercentile_quiet(stack, 25)
        out[f"{leg}_q75"] = _nanpercentile_quiet(stack, 75)
        out[f"{leg}_sig"] = _sig_mask(stack, min_sites)
    return pd.DataFrame(out)


def aggregate_at_nbins(per_site, n_bins: int, min_sites_ttest: int = MIN_SITES_TTEST) -> dict:
    """Bundle the cross-site result for ONE bin count (no retraining)."""
    if not per_site:
        raise ValueError("aggregate_at_nbins: no sites passed the fit stage")
    sm_stack = np.stack([bin_map(p["d_sm"], p["sm_vals"], p["vpd_vals"], n_bins) for p in per_site], axis=0)
    vpd_stack = np.stack([bin_map(p["d_vpd"], p["sm_vals"], p["vpd_vals"], n_bins) for p in per_site], axis=0)
    return {
        "n_bins": n_bins,
        "n_sites": len(per_site),
        "sm_map": _nanmedian_quiet(sm_stack),
        "vpd_map": _nanmedian_quiet(vpd_stack),
        "sm_sig": _sig_mask(sm_stack, min_sites_ttest),
        "vpd_sig": _sig_mask(vpd_stack, min_sites_ttest),
        "by_sm_bin": _leg_table(per_site, "sm_vals", n_bins, min_sites_ttest),
        "by_vpd_bin": _leg_table(per_site, "vpd_vals", n_bins, min_sites_ttest),
    }
