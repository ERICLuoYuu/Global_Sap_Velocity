# src/sm_vpd_decoupling/tests/test_decoupling.py
from __future__ import annotations

import numpy as np
import pandas as pd
from src.sm_vpd_decoupling.decoupling import (
    assign_percentile_bins,
    decouple_effect,
    decouple_site,
)


def _synthetic(driver: str, n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """SM and VPD fully decoupled (independent uniforms) so both axes populate.
    `driver` selects which one the response depends on (response = that driver)."""
    rng = np.random.default_rng(seed)
    sm = rng.uniform(0.0, 1.0, n)
    vpd = rng.uniform(0.5, 3.0, n)
    resp = sm.copy() if driver == "sm" else (vpd - 0.5) / 2.5
    return pd.DataFrame({"sm": sm, "vpd": vpd, "resp": resp})


def test_assign_percentile_bins_basic():
    s = pd.Series(np.arange(100.0))
    codes = assign_percentile_bins(s, 5)
    assert codes.notna().all()
    assert sorted(codes.unique()) == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_assign_percentile_bins_degenerate_all_same():
    s = pd.Series([1.0] * 50)
    codes = assign_percentile_bins(s, 5)
    # Single-value series collapses to one bin (no gradient), no exception.
    assert set(codes.dropna().unique()).issubset({0.0})


def test_sm_dominant_case():
    df = _synthetic("sm")
    eff = decouple_site(df, response="resp", n_bins=5, vpd_col="vpd", sm_col="sm")
    # Low SM is the stressor -> sm_given_vpd strongly negative.
    assert eff["sm_given_vpd"] < -0.5
    # VPD has no effect on response -> ~0.
    assert abs(eff["vpd_given_sm"]) < 0.15
    assert abs(eff["sm_given_vpd"]) > abs(eff["vpd_given_sm"])


def test_vpd_dominant_case():
    df = _synthetic("vpd")
    eff = decouple_site(df, response="resp", n_bins=5, vpd_col="vpd", sm_col="sm")
    assert eff["vpd_given_sm"] > 0.5
    assert abs(eff["sm_given_vpd"]) < 0.15
    assert abs(eff["vpd_given_sm"]) > abs(eff["sm_given_vpd"])


def test_min_cond_bins_returns_nan():
    # Only one VPD value -> for vpd_given_sm there are <2 driver bins everywhere,
    # and for sm_given_vpd only one conditioning (VPD) bin -> < MIN_COND_BINS.
    df = pd.DataFrame({"sm": np.linspace(0, 1, 100), "vpd": np.ones(100), "resp": np.linspace(0, 1, 100)})
    eff = decouple_site(df, response="resp", n_bins=5, vpd_col="vpd", sm_col="sm")
    assert np.isnan(eff["sm_given_vpd"])
    assert np.isnan(eff["vpd_given_sm"])


def test_min_bin_count_drops_sparse_cells():
    df = _synthetic("sm", n=40)  # ~1-2 points per 5x5 cell
    eff = decouple_site(df, response="resp", n_bins=5, vpd_col="vpd", sm_col="sm", min_bin_count=3)
    # Too sparse: most cells < 3 points -> NaN (not a misleading number).
    assert np.isnan(eff["sm_given_vpd"]) or np.isnan(eff["vpd_given_sm"])
