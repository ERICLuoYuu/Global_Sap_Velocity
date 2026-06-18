"""Tests for cross-site aggregation + per-cell/per-bin t-tests (Task 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.fu_ann_sensitivity.aggregate import (
    aggregate_at_nbins,
    fit_site_sensitivities,
)

PREDICTORS_Z = ["tair_z", "vpd_z", "sm_z", "ppfd_z"]


def _fake_site(site, n=400, d_sm_val=-0.5, d_vpd_val=0.5, seed=0):
    """A per-site bundle with KNOWN-sign sensitivities (1 pseudo-model)."""
    rng = np.random.default_rng(seed)
    return {
        "site": site,
        "pft": "ENF",
        "d_sm": np.full((1, n), d_sm_val),
        "d_vpd": np.full((1, n), d_vpd_val),
        "sm_vals": rng.normal(size=n),
        "vpd_vals": rng.normal(size=n),
        "r": 0.9,
        "n_days": n,
    }


# ---- aggregate_at_nbins (pure) ----------------------------------------------


def test_maps_shape_and_dryness_sign():
    per_site = [_fake_site(f"s{i}", seed=i) for i in range(6)]
    agg = aggregate_at_nbins(per_site, n_bins=5)
    assert agg["sm_map"].shape == (5, 5)
    assert agg["vpd_map"].shape == (5, 5)
    assert agg["sm_sig"].shape == (5, 5)
    assert agg["vpd_sig"].shape == (5, 5)
    assert agg["n_sites"] == 6
    # Response rises with both -> drying reduces it (sm<0); rises with vpd (vpd>0).
    assert np.nanmedian(agg["sm_map"]) < 0
    assert np.nanmedian(agg["vpd_map"]) > 0


def test_dual_leg_dataframes_schema_and_significance():
    per_site = [_fake_site(f"s{i}", seed=i) for i in range(6)]
    agg = aggregate_at_nbins(per_site, n_bins=5, min_sites_ttest=3)
    for tbl_key, axis_col in [("by_sm_bin", "sm_bin"), ("by_vpd_bin", "vpd_bin")]:
        tbl = agg[tbl_key]
        assert len(tbl) == 5  # one row per percentile bin
        for col in (
            axis_col,
            "d_sm_median",
            "d_sm_q25",
            "d_sm_q75",
            "d_sm_sig",
            "d_vpd_median",
            "d_vpd_q25",
            "d_vpd_q75",
            "d_vpd_sig",
        ):
            assert col in tbl.columns
    # 6 sites all at -0.5 -> SWC leg significantly different from zero.
    assert agg["by_sm_bin"]["d_sm_sig"].any()


# ---- fit_site_sensitivities (drops low-r sites) -----------------------------


def _zscored_multisite_table():
    rng = np.random.default_rng(1)
    frames = []
    for site, signal in [("good1", True), ("good2", True), ("noise", False)]:
        n = 200
        tair = rng.normal(size=n)
        vpd = rng.normal(size=n)
        sm = rng.normal(size=n)
        ppfd = rng.normal(size=n)
        e = (0.8 * sm + 0.8 * vpd + 0.1 * rng.normal(size=n)) if signal else rng.normal(size=n)
        frames.append(
            pd.DataFrame(
                {
                    "site_name": [site] * n,
                    "pft": ["ENF"] * n,
                    "tair_z": tair,
                    "vpd_z": vpd,
                    "sm_z": sm,
                    "ppfd_z": ppfd,
                    "E_z": e,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_fit_drops_low_r_site():
    per_site, n_dropped, median_r = fit_site_sensitivities(
        _zscored_multisite_table(),
        response="E_z",
        sm_col="sm_z",
        vpd_col="vpd_z",
        predictors=PREDICTORS_Z,
        min_valid_days=20,
        n_repeats=2,
        r_threshold=0.5,
    )
    kept = {p["site"] for p in per_site}
    assert "noise" not in kept  # pure-noise site fails r<0.5
    assert n_dropped >= 1
    for p in per_site:
        assert p["d_sm"].shape == (2, 200)  # (n_models, n_rows)


def test_fit_skips_sites_below_min_days():
    df = _zscored_multisite_table()
    per_site, _, _ = fit_site_sensitivities(
        df,
        response="E_z",
        sm_col="sm_z",
        vpd_col="vpd_z",
        predictors=PREDICTORS_Z,
        min_valid_days=10_000,
        n_repeats=2,
    )
    assert per_site == []
