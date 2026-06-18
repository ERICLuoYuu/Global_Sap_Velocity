"""Phase 5 Round 1 - edge cases & boundary values."""

from __future__ import annotations

import numpy as np

from src.fu_ann_sensitivity.aggregate import aggregate_at_nbins
from src.fu_ann_sensitivity.ann import ensemble_predict, train_ensemble
from src.fu_ann_sensitivity.sensitivity import bin_map


def test_train_ensemble_tiny_data_no_crash():
    # n=8 -> test split has 2 rows (< 3), so r falls back to 0.0 but nothing crashes.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 4))
    y = rng.normal(size=8)
    models, r = train_ensemble(X, y, n_repeats=2, seed=0)
    assert len(models) == 2
    assert np.isfinite(r)
    assert ensemble_predict(models, X).shape == (8,)


def test_bin_map_single_bin_is_overall_median():
    vals = np.array([1.0, 3.0, 5.0])
    m = bin_map(vals, [0.1, 0.2, 0.3], [0.5, 0.6, 0.7], n_bins=1)
    assert m.shape == (1, 1)
    assert np.isclose(m[0, 0], 3.0)


def test_bin_map_constant_sm_collapses_to_one_row():
    n = 30
    vals = np.ones(n)
    sm = np.full(n, 0.25)  # constant SWC -> every row in SWC bin 0
    vpd = np.linspace(0.0, 1.0, n)
    m = bin_map(vals, sm, vpd, n_bins=5)
    assert m.shape == (5, 5)
    assert np.isfinite(m[0]).any()  # only the single populated SWC row has values
    assert np.isnan(m[1:]).all()  # the rest are NaN, never 0


def test_aggregate_single_site_below_ttest_threshold():
    rng = np.random.default_rng(0)
    n = 200
    site = {
        "site": "s0",
        "pft": "ENF",
        "d_sm": np.full((1, n), -0.4),
        "d_vpd": np.full((1, n), 0.4),
        "sm_vals": rng.normal(size=n),
        "vpd_vals": rng.normal(size=n),
        "r": 0.9,
        "n_days": n,
    }
    agg = aggregate_at_nbins([site], n_bins=5)
    assert agg["n_sites"] == 1
    assert np.nanmedian(agg["sm_map"]) < 0
    assert not agg["sm_sig"].any()  # 1 site < MIN_SITES_TTEST -> never significant
