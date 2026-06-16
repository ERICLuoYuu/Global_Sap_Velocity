# src/sm_vpd_decoupling/tests/test_edge_cases.py
"""Phase 5 Round 1 — edge-case and boundary-value tests.

All expected values are verified against the actual source code before assertion.
Synthetic data uses seeded np.random.default_rng per the module's test conventions.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.sm_vpd_decoupling.aggregate import dominance_summary
from src.sm_vpd_decoupling.conductance import ETA, T0_K, canopy_conductance, sfd_to_kg_m2_s
from src.sm_vpd_decoupling.decoupling import (
    MIN_BIN_COUNT,
    MIN_COND_BINS,
    assign_percentile_bins,
    decouple_effect,
)
from src.sm_vpd_decoupling.loader import normalize_per_site, resolve_ppfd, root_zone_sm

# ---------------------------------------------------------------------------
# conductance.canopy_conductance
# ---------------------------------------------------------------------------


def test_canopy_conductance_negative_vpd_scalar_is_nan():
    """Negative VPD (scalar path, vpd <= 0 branch) must return NaN, not inf."""
    result = canopy_conductance(1.0, 20.0, -0.5, 0.0)
    assert np.isnan(result)


def test_canopy_conductance_negative_vpd_series_produces_nan_for_negative_row():
    """Negative VPD inside a Series must yield NaN for that row only (vpd.where(vpd > 0))."""
    sv = pd.Series([1.0, 1.0])
    tair = pd.Series([20.0, 20.0])
    vpd = pd.Series([-0.5, 1.0])
    alt = pd.Series([0.0, 0.0])
    out = canopy_conductance(sv, tair, vpd, alt)
    assert np.isnan(out.iloc[0]), "negative VPD row must be NaN"
    assert np.isfinite(out.iloc[1]), "positive VPD row must be finite"


def test_canopy_conductance_nan_tair_propagates():
    """NaN in tair propagates to NaN Gc (no special-casing of NaN temperature)."""
    result = canopy_conductance(1.0, float("nan"), 1.0, 0.0)
    assert np.isnan(result)


def test_canopy_conductance_nan_tair_series_propagates_per_row():
    """Series NaN tair propagates NaN only for its row; valid rows remain finite."""
    sv = pd.Series([1.0, 1.0])
    tair = pd.Series([float("nan"), 20.0])
    vpd = pd.Series([1.0, 1.0])
    alt = pd.Series([0.0, 0.0])
    out = canopy_conductance(sv, tair, vpd, alt)
    assert np.isnan(out.iloc[0])
    assert np.isfinite(out.iloc[1])


def test_canopy_conductance_large_altitude_is_finite_and_correct():
    """Very large altitude (10 000 m) stays finite; exp(0.00012*10000) ≈ 3.32."""
    h = 10_000.0
    t, vpd, sv = 20.0, 1.0, 1.0
    sfd = sfd_to_kg_m2_s(sv)
    expected = (115.8 + 0.4236 * t) * (sfd / vpd) * (ETA * T0_K / (T0_K + t)) * math.exp(0.00012 * h)
    got = canopy_conductance(sv, t, vpd, h)
    assert np.isfinite(got)
    assert math.isclose(got, expected, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# decoupling.assign_percentile_bins
# ---------------------------------------------------------------------------


def test_assign_percentile_bins_n_bins_larger_than_unique_values_collapses():
    """n_bins=10 with only 3 unique values: qcut(duplicates='drop') collapses to 3 bins."""
    s = pd.Series([1.0, 2.0, 3.0])
    codes = assign_percentile_bins(s, 10)
    # Fewer than 10 unique codes are produced — no crash, all non-NaN.
    assert codes.notna().all()
    assert codes.nunique() <= 3


def test_assign_percentile_bins_n_bins_1_puts_all_in_bin_zero():
    """n_bins=1: single quantile cut assigns everything to code 0."""
    s = pd.Series(np.arange(10.0))
    codes = assign_percentile_bins(s, 1)
    assert codes.notna().all()
    assert set(codes.unique()) == {0.0}


# ---------------------------------------------------------------------------
# decoupling.decouple_effect — boundary counts
# ---------------------------------------------------------------------------


def _two_by_two_df(points_per_cell: int, resp_by_sm: bool = True) -> pd.DataFrame:
    """2 VPD-conditioning bins × 2 SM-driver bins with exact points_per_cell each.

    resp_by_sm=True  → response == sm_bin value (SM-dominant signal).
    """
    rows = []
    for vpd_b in [0.0, 1.0]:
        for sm_b in [0.0, 1.0]:
            for _ in range(points_per_cell):
                rows.append(
                    {"_vpd_bin": float(vpd_b), "_sm_bin": float(sm_b), "resp": float(sm_b) if resp_by_sm else 0.5}
                )
    return pd.DataFrame(rows)


def test_decouple_effect_exactly_min_bin_count_cells_are_kept():
    """Cells with exactly MIN_BIN_COUNT points are kept (inclusive boundary).

    With 2 conditioning bins × 2 driver bins, all cells at the boundary count,
    so len(effects) == 2 >= MIN_COND_BINS → finite result.
    """
    df = _two_by_two_df(MIN_BIN_COUNT)
    result = decouple_effect(df, "_sm_bin", "_vpd_bin", "resp", low_minus_high=True, min_bin_count=MIN_BIN_COUNT)
    assert np.isfinite(result), f"Expected finite result but got {result}"


def test_decouple_effect_exactly_min_cond_bins_returns_finite():
    """Exactly MIN_COND_BINS (2) populated conditioning bins returns a finite number."""
    df = _two_by_two_df(MIN_BIN_COUNT)
    result = decouple_effect(df, "_sm_bin", "_vpd_bin", "resp", low_minus_high=True, min_cond_bins=MIN_COND_BINS)
    assert np.isfinite(result)


def test_decouple_effect_one_cond_bin_returns_nan():
    """Only 1 populated conditioning bin < MIN_COND_BINS (2) → NaN."""
    rows = []
    for sm_b in [0.0, 1.0]:
        for _ in range(MIN_BIN_COUNT):
            rows.append({"_vpd_bin": 0.0, "_sm_bin": float(sm_b), "resp": float(sm_b)})
    df = pd.DataFrame(rows)
    result = decouple_effect(df, "_sm_bin", "_vpd_bin", "resp", low_minus_high=True)
    assert np.isnan(result)


def test_decouple_effect_all_nan_response_returns_nan():
    """All-NaN response means every cell mean is NaN → groupby drops → NaN result."""
    df = _two_by_two_df(MIN_BIN_COUNT)
    df["resp"] = np.nan
    result = decouple_effect(df, "_sm_bin", "_vpd_bin", "resp", low_minus_high=True)
    assert np.isnan(result)


# ---------------------------------------------------------------------------
# loader.normalize_per_site
# ---------------------------------------------------------------------------


def test_normalize_per_site_all_nan_returns_all_nan():
    """All-NaN series: anchor is NaN → function returns all-NaN series (no crash)."""
    s = pd.Series([np.nan] * 5)
    out = normalize_per_site(s)
    assert out.isna().all()


def test_normalize_per_site_single_non_nan_normalizes_to_one():
    """Single non-NaN value: anchor equals that value, normalized result == 1.0 there."""
    s = pd.Series([np.nan, np.nan, 7.0, np.nan])
    out = normalize_per_site(s)
    # NaN positions remain NaN.
    assert np.isnan(out.iloc[0])
    assert np.isnan(out.iloc[1])
    assert np.isnan(out.iloc[3])
    # The single valid value normalizes to 1.0 (anchor = mean([7.0]) = 7.0).
    assert math.isclose(out.iloc[2], 1.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# loader.root_zone_sm
# ---------------------------------------------------------------------------


def test_root_zone_sm_nan_in_one_layer_propagates_to_that_row_only():
    """NaN in swvl2 for row 0 makes row 0 NaN; row 1 (all valid) stays finite."""
    s1 = pd.Series([0.2, 0.2])
    s2 = pd.Series([np.nan, 0.3])
    s3 = pd.Series([0.4, 0.4])
    out = root_zone_sm(s1, s2, s3)
    assert np.isnan(out.iloc[0]), "row with NaN layer must be NaN"
    assert np.isfinite(out.iloc[1]), "row with all-valid layers must be finite"


# ---------------------------------------------------------------------------
# loader.resolve_ppfd
# ---------------------------------------------------------------------------


def test_resolve_ppfd_no_relevant_columns_returns_none_source_and_all_nan():
    """DataFrame with no ppfd/sw/era5 columns: source='none', series all-NaN."""
    df = pd.DataFrame({"other_col": [1.0, 2.0, 3.0]})
    ppfd, source = resolve_ppfd(df)
    assert source == "none"
    assert ppfd.isna().all()


# ---------------------------------------------------------------------------
# aggregate.dominance_summary
# ---------------------------------------------------------------------------


def test_dominance_summary_empty_dataframe_returns_nan_and_zero():
    """Empty effects DataFrame (0 rows): returns (nan, 0)."""
    empty = pd.DataFrame({"sm_given_vpd": pd.Series([], dtype=float), "vpd_given_sm": pd.Series([], dtype=float)})
    pct, n = dominance_summary(empty)
    assert n == 0
    assert np.isnan(pct)


def test_dominance_summary_all_nan_effects_returns_nan_and_zero():
    """All-NaN sm_given_vpd/vpd_given_sm: no valid sites → (nan, 0)."""
    nan_df = pd.DataFrame({"sm_given_vpd": [np.nan, np.nan], "vpd_given_sm": [np.nan, np.nan]})
    pct, n = dominance_summary(nan_df)
    assert n == 0
    assert np.isnan(pct)
