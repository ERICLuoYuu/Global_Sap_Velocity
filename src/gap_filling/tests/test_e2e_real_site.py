"""E2E test: run the full gap-filling pipeline on real site data.

Sites:
  - AUS_WOM: Australian eucalyptus woodland (~50k hourly, 11 trees)
  - DEU_HIN_OAK: German oak forest (~26k hourly, 8 trees)

Tests cover:
  - Structural integrity (shape, index, columns)
  - Gap reduction
  - Value preservation
  - Physical plausibility (bounds, diurnal, nighttime)
  - Statistical consistency (distribution, autocorrelation, outlier injection)
  - Cross-tree consistency (inter-correlation)
  - Environmental coherence (VPD, radiation correlation)
  - Gap boundary quality (discontinuities, smooth transitions)
  - Cross-biome generalization
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

from src.gap_filling.config import GapFillingConfig
from src.gap_filling.detector import GapDetector
from src.gap_filling.filler import GapFiller

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA = Path("outputs/processed_data/sapwood")

_SITES = {
    "AUS_WOM": {
        "sap": _DATA / "sap" / "outliers_removed" / "AUS_WOM_sapf_data_outliers_removed.csv",
        "env": _DATA / "env" / "outliers_removed" / "AUS_WOM_env_data_outliers_removed.csv",
    },
    "DEU_HIN_OAK": {
        "sap": _DATA / "sap" / "outliers_removed" / "DEU_HIN_OAK_sapf_data_outliers_removed.csv",
        "env": _DATA / "env" / "outliers_removed" / "DEU_HIN_OAK_env_data_outliers_removed.csv",
    },
}


def _site_available(site: str) -> bool:
    return _SITES[site]["sap"].exists() and _SITES[site]["env"].exists()


def _load(site: str, kind: str) -> pd.DataFrame:
    df = pd.read_csv(_SITES[site][kind], parse_dates=["TIMESTAMP"], index_col="TIMESTAMP")
    drop_cols = [c for c in df.columns if "TIMESTAMP" in c.upper()]
    return df.drop(columns=drop_cols, errors="ignore").apply(pd.to_numeric, errors="coerce")


skip_no_aus = pytest.mark.skipif(not _site_available("AUS_WOM"), reason="AUS_WOM data not available")
skip_no_deu = pytest.mark.skipif(not _site_available("DEU_HIN_OAK"), reason="DEU_HIN_OAK data not available")


# ---- Helpers --------------------------------------------------------------


def _gap_mask(original: pd.Series, filled: pd.Series) -> pd.Series:
    """Boolean mask: True where original was NaN but filled is not."""
    return original.isna() & filled.notna()


def _boundary_indices(original: pd.Series) -> list[int]:
    """Integer positions of gap start/end boundaries."""
    is_nan = original.isna().values
    boundaries = []
    for i in range(1, len(is_nan)):
        if is_nan[i] != is_nan[i - 1]:
            boundaries.append(i)
    return boundaries


# ===========================================================================
# AUS_WOM --- primary site tests
# ===========================================================================


@skip_no_aus
class TestRealSiteE2E:
    """Structural integrity tests on AUS_WOM."""

    @pytest.fixture(scope="class")
    def sap_df(self) -> pd.DataFrame:
        return _load("AUS_WOM", "sap")

    @pytest.fixture(scope="class")
    def env_df(self) -> pd.DataFrame:
        return _load("AUS_WOM", "env")

    @pytest.fixture(scope="class")
    def filler(self) -> GapFiller:
        return GapFiller(GapFillingConfig(time_scale="hourly"), target="sap")

    @pytest.fixture(scope="class")
    def filled_df(self, filler, sap_df, env_df) -> pd.DataFrame:
        return filler.fill_dataframe(sap_df, env_df=env_df)

    # --- shape & index ---

    def test_output_shape_matches_input(self, sap_df, filled_df):
        assert filled_df.shape == sap_df.shape

    def test_index_preserved(self, sap_df, filled_df):
        pd.testing.assert_index_equal(filled_df.index, sap_df.index)

    def test_column_order_preserved(self, sap_df, filled_df):
        assert list(filled_df.columns) == list(sap_df.columns)

    # --- gap reduction ---

    def test_total_nans_reduced(self, sap_df, filled_df):
        orig_nans = sap_df.isna().sum().sum()
        filled_nans = filled_df.isna().sum().sum()
        assert filled_nans <= orig_nans, f"NaN increased: {orig_nans} -> {filled_nans}"

    def test_at_least_one_column_improved(self, sap_df, filled_df):
        improved = (filled_df.isna().sum() < sap_df.isna().sum()).any()
        assert improved, "No column had NaN reduction"

    def test_per_column_nan_report(self, sap_df, filled_df):
        for col in sap_df.columns:
            orig = sap_df[col].isna().sum()
            after = filled_df[col].isna().sum()
            pct = (1 - after / orig) * 100 if orig > 0 else 0
            print(f"{col}: {orig} -> {after} ({pct:.1f}% filled)")

    # --- value preservation ---

    def test_original_values_preserved(self, sap_df, filled_df):
        mask = sap_df.notna()
        np.testing.assert_array_almost_equal(filled_df.values[mask.values], sap_df.values[mask.values], decimal=6)

    def test_no_negative_values(self, filled_df):
        neg_count = (filled_df.select_dtypes(include=[np.number]) < 0).sum().sum()
        assert neg_count == 0, f"Found {neg_count} negative values"

    # --- detector ---

    def test_detector_finds_gaps(self, sap_df):
        any_gaps = any(GapDetector.detect(sap_df[c]) for c in sap_df.columns)
        assert any_gaps, "No gaps detected"

    def test_single_column_fill(self, filler, sap_df, env_df):
        col = sap_df.columns[0]
        filled = filler.fill_column(sap_df[col], env_df=env_df)
        assert len(filled) == len(sap_df[col])
        assert filled.isna().sum() <= sap_df[col].isna().sum()
        mask = sap_df[col].notna()
        np.testing.assert_array_almost_equal(filled[mask].values, sap_df[col][mask].values, decimal=6)


# ===========================================================================
# Physical plausibility
# ===========================================================================


@skip_no_aus
class TestPhysicalPlausibility:
    """Sap velocity must obey physical constraints."""

    @pytest.fixture(scope="class")
    def data(self):
        sap = _load("AUS_WOM", "sap")
        env = _load("AUS_WOM", "env")
        filler = GapFiller(GapFillingConfig(time_scale="hourly"), target="sap")
        filled = filler.fill_dataframe(sap, env_df=env)
        return sap, env, filled

    def test_value_range(self, data):
        """Filled values within plausible sap velocity range [0, 150]."""
        _, _, filled = data
        max_val = filled.max().max()
        assert max_val <= 150, f"Max filled value {max_val} exceeds physical bound 150"

    def test_diurnal_pattern(self, data):
        """Daytime mean sap flow > nighttime mean (using ext_rad as proxy)."""
        _, env, filled = data
        if "ext_rad" not in env.columns:
            pytest.skip("ext_rad not available")
        day_mask = env["ext_rad"] > 0
        night_mask = env["ext_rad"] == 0
        common = filled.index.intersection(env.index)
        for col in filled.columns[:3]:
            day_vals = filled.loc[common][col][day_mask.reindex(common, fill_value=False)]
            night_vals = filled.loc[common][col][night_mask.reindex(common, fill_value=False)]
            if len(day_vals.dropna()) > 100 and len(night_vals.dropna()) > 100:
                assert day_vals.mean() > night_vals.mean(), (
                    f"{col}: daytime mean {day_vals.mean():.2f} <= nighttime {night_vals.mean():.2f}"
                )

    def test_nighttime_values_low(self, data):
        """Night-time filled values should be low (< 50th percentile of daytime)."""
        _, env, filled = data
        if "ext_rad" not in env.columns:
            pytest.skip("ext_rad not available")
        night_mask = env["ext_rad"] == 0
        day_mask = env["ext_rad"] > 0
        common = filled.index.intersection(env.index)
        for col in filled.columns[:3]:
            day_vals = filled.loc[common][col][day_mask.reindex(common, fill_value=False)].dropna()
            night_vals = filled.loc[common][col][night_mask.reindex(common, fill_value=False)].dropna()
            if len(day_vals) > 100 and len(night_vals) > 100:
                day_p50 = day_vals.quantile(0.5)
                night_mean = night_vals.mean()
                assert night_mean < day_p50, f"{col}: night mean {night_mean:.2f} >= day median {day_p50:.2f}"


# ===========================================================================
# Statistical consistency
# ===========================================================================


@skip_no_aus
class TestStatisticalConsistency:
    """Filled values should be statistically consistent with observed."""

    @pytest.fixture(scope="class")
    def data(self):
        sap = _load("AUS_WOM", "sap")
        env = _load("AUS_WOM", "env")
        filler = GapFiller(GapFillingConfig(time_scale="hourly"), target="sap")
        filled = filler.fill_dataframe(sap, env_df=env)
        return sap, env, filled

    def test_distribution_similarity(self, data):
        """KS test: filled-only values vs observed -- print p-values."""
        sap, _, filled = data
        for col in sap.columns[:3]:
            mask = _gap_mask(sap[col], filled[col])
            observed = sap[col].dropna().values
            filled_vals = filled[col][mask].dropna().values
            if len(filled_vals) < 10 or len(observed) < 10:
                continue
            _, p_val = sp_stats.ks_2samp(observed, filled_vals)
            print(f"{col}: KS p={p_val:.4f}")

    def test_no_outlier_injection(self, data):
        """Filled values should not exceed 2x the observed max."""
        sap, _, filled = data
        for col in sap.columns:
            obs_max = sap[col].max()
            if pd.isna(obs_max) or obs_max == 0:
                continue
            mask = _gap_mask(sap[col], filled[col])
            filled_vals = filled[col][mask]
            if filled_vals.empty:
                continue
            filled_max = filled_vals.max()
            assert filled_max <= 2 * obs_max, f"{col}: filled max {filled_max:.2f} > 2x observed max {obs_max:.2f}"

    def test_mean_within_range(self, data):
        """Mean of filled values within 0.2x to 5x the observed mean."""
        sap, _, filled = data
        for col in sap.columns:
            obs_mean = sap[col].dropna().mean()
            if pd.isna(obs_mean) or obs_mean < 0.01:
                continue
            mask = _gap_mask(sap[col], filled[col])
            filled_vals = filled[col][mask].dropna()
            if len(filled_vals) < 10:
                continue
            filled_mean = filled_vals.mean()
            assert 0.2 * obs_mean <= filled_mean <= 5 * obs_mean, (
                f"{col}: filled mean {filled_mean:.2f} outside [0.2x, 5x] observed mean {obs_mean:.2f}"
            )

    def test_autocorrelation_preserved(self, data):
        """Lag-1 autocorrelation of filled series within 0.5x of original."""
        sap, _, filled = data
        for col in sap.columns[:3]:
            obs_clean = sap[col].dropna()
            if len(obs_clean) < 100:
                continue
            acf_orig = obs_clean.autocorr(lag=1)
            acf_filled = filled[col].autocorr(lag=1)
            if pd.isna(acf_orig) or pd.isna(acf_filled):
                continue
            print(f"{col}: acf_orig={acf_orig:.3f}, acf_filled={acf_filled:.3f}")
            assert acf_filled >= 0.5 * acf_orig, f"{col}: acf dropped from {acf_orig:.3f} to {acf_filled:.3f}"


# ===========================================================================
# Cross-tree consistency
# ===========================================================================


@skip_no_aus
class TestCrossTreeConsistency:
    """Co-located trees should remain correlated after filling."""

    @pytest.fixture(scope="class")
    def data(self):
        sap = _load("AUS_WOM", "sap")
        env = _load("AUS_WOM", "env")
        filler = GapFiller(GapFillingConfig(time_scale="hourly"), target="sap")
        filled = filler.fill_dataframe(sap, env_df=env)
        return sap, env, filled

    def test_inter_tree_correlation_maintained(self, data):
        """Mean pairwise correlation should not drop by more than 50%."""
        sap, _, filled = data
        orig_corr = sap.corr().values
        filled_corr = filled.corr().values
        mask = ~np.eye(orig_corr.shape[0], dtype=bool)
        orig_mean = np.nanmean(orig_corr[mask])
        filled_mean = np.nanmean(filled_corr[mask])
        print(f"Mean inter-tree corr: orig={orig_mean:.3f}, filled={filled_mean:.3f}")
        assert filled_mean >= 0.5 * orig_mean, (
            f"Inter-tree correlation dropped too much: {orig_mean:.3f} -> {filled_mean:.3f}"
        )

    def test_pairwise_correlation_sign_preserved(self, data):
        """Sign of pairwise correlations should be preserved."""
        sap, _, filled = data
        orig_corr = sap.corr()
        filled_corr = filled.corr()
        for i, c1 in enumerate(sap.columns):
            for j, c2 in enumerate(sap.columns):
                if i >= j:
                    continue
                oc = orig_corr.loc[c1, c2]
                fc = filled_corr.loc[c1, c2]
                if pd.notna(oc) and pd.notna(fc) and abs(oc) > 0.3:
                    assert np.sign(oc) == np.sign(fc), f"{c1}-{c2}: correlation sign flipped {oc:.3f} -> {fc:.3f}"


# ===========================================================================
# Environmental coherence
# ===========================================================================


@skip_no_aus
class TestEnvironmentalCoherence:
    """Filled values should correlate with environmental drivers."""

    @pytest.fixture(scope="class")
    def data(self):
        sap = _load("AUS_WOM", "sap")
        env = _load("AUS_WOM", "env")
        filler = GapFiller(GapFillingConfig(time_scale="hourly"), target="sap")
        filled = filler.fill_dataframe(sap, env_df=env)
        return sap, env, filled

    def test_vpd_positive_correlation(self, data):
        """Filled sap velocity should correlate positively with VPD."""
        _, env, filled = data
        if "vpd" not in env.columns:
            pytest.skip("vpd not available")
        common = filled.index.intersection(env.index)
        col = filled.columns[0]
        merged = pd.DataFrame({"sap": filled.loc[common, col], "vpd": env.loc[common, "vpd"]}).dropna()
        if len(merged) < 100:
            pytest.skip("Not enough overlapping data")
        corr = merged["sap"].corr(merged["vpd"])
        print(f"{col} vs vpd: r={corr:.3f}")
        assert corr > 0, f"Expected positive VPD correlation, got {corr:.3f}"

    def test_radiation_positive_correlation(self, data):
        """Filled sap velocity should correlate positively with sw_in."""
        _, env, filled = data
        if "sw_in" not in env.columns:
            pytest.skip("sw_in not available")
        common = filled.index.intersection(env.index)
        col = filled.columns[0]
        merged = pd.DataFrame({"sap": filled.loc[common, col], "sw_in": env.loc[common, "sw_in"]}).dropna()
        if len(merged) < 100:
            pytest.skip("Not enough overlapping data")
        corr = merged["sap"].corr(merged["sw_in"])
        print(f"{col} vs sw_in: r={corr:.3f}")
        assert corr > 0, f"Expected positive radiation correlation, got {corr:.3f}"

    def test_temperature_response(self, data):
        """Sap flow during cold periods (ta < 5C) should be lower than warm (ta > 20C)."""
        _, env, filled = data
        if "ta" not in env.columns:
            pytest.skip("ta not available")
        common = filled.index.intersection(env.index)
        col = filled.columns[0]
        ta = env.loc[common, "ta"]
        sap_vals = filled.loc[common, col]
        cold = sap_vals[ta < 5].dropna()
        warm = sap_vals[ta > 20].dropna()
        if len(cold) < 50 or len(warm) < 50:
            pytest.skip("Not enough cold/warm data")
        print(f"{col}: cold mean={cold.mean():.2f}, warm mean={warm.mean():.2f}")
        assert cold.mean() < warm.mean(), f"Cold mean {cold.mean():.2f} >= warm mean {warm.mean():.2f}"


# ===========================================================================
# Gap boundary quality
# ===========================================================================


@skip_no_aus
class TestGapBoundaryQuality:
    """No large discontinuities at gap start/end boundaries."""

    @pytest.fixture(scope="class")
    def data(self):
        sap = _load("AUS_WOM", "sap")
        env = _load("AUS_WOM", "env")
        filler = GapFiller(GapFillingConfig(time_scale="hourly"), target="sap")
        filled = filler.fill_dataframe(sap, env_df=env)
        return sap, env, filled

    def test_no_extreme_discontinuities(self, data):
        """Jump at gap boundaries should not exceed 10x the typical step size."""
        sap, _, filled = data
        col = sap.columns[0]
        series = filled[col]
        boundaries = _boundary_indices(sap[col])
        if len(boundaries) < 2:
            pytest.skip("Not enough gap boundaries")

        obs = sap[col].dropna()
        typical_step = obs.diff().abs().median()
        if typical_step < 0.001:
            typical_step = 0.001

        extreme_jumps = 0
        for idx in boundaries:
            if idx >= len(series) or idx < 1:
                continue
            jump = abs(series.iloc[idx] - series.iloc[idx - 1])
            if pd.notna(jump) and jump > 10 * typical_step:
                extreme_jumps += 1

        pct_extreme = extreme_jumps / max(len(boundaries), 1)
        print(f"{col}: {extreme_jumps}/{len(boundaries)} extreme jumps ({pct_extreme:.1%})")
        assert pct_extreme < 0.20, f"{col}: {pct_extreme:.1%} of boundaries have extreme jumps (>10x typical step)"

    def test_rolling_std_at_boundaries(self, data):
        """Rolling std at gap boundaries should not spike excessively."""
        sap, _, filled = data
        col = sap.columns[0]
        series = filled[col]
        boundaries = _boundary_indices(sap[col])
        if len(boundaries) < 2:
            pytest.skip("Not enough gap boundaries")

        rolling_std = series.rolling(window=6, center=True, min_periods=3).std()
        global_std = series.std()
        if pd.isna(global_std) or global_std < 0.001:
            pytest.skip("Series has no variance")

        boundary_stds = [
            rolling_std.iloc[i] for i in boundaries if i < len(rolling_std) and pd.notna(rolling_std.iloc[i])
        ]
        if not boundary_stds:
            pytest.skip("No valid boundary stds")

        median_boundary_std = np.median(boundary_stds)
        ratio = median_boundary_std / global_std
        print(f"{col}: boundary rolling std / global std = {ratio:.2f}")
        assert ratio < 3.0, f"Boundary variability too high: {ratio:.2f}x global std"


# ===========================================================================
# Cross-biome generalization (DEU_HIN_OAK)
# ===========================================================================


@skip_no_deu
class TestCrossBiomeGeneralization:
    """Same pipeline on a European oak site should also work."""

    @pytest.fixture(scope="class")
    def sap_df(self) -> pd.DataFrame:
        return _load("DEU_HIN_OAK", "sap")

    @pytest.fixture(scope="class")
    def env_df(self) -> pd.DataFrame:
        return _load("DEU_HIN_OAK", "env")

    @pytest.fixture(scope="class")
    def filler(self) -> GapFiller:
        return GapFiller(GapFillingConfig(time_scale="hourly"), target="sap")

    @pytest.fixture(scope="class")
    def filled_df(self, filler, sap_df, env_df) -> pd.DataFrame:
        return filler.fill_dataframe(sap_df, env_df=env_df)

    def test_shape_preserved(self, sap_df, filled_df):
        assert filled_df.shape == sap_df.shape

    def test_nans_reduced(self, sap_df, filled_df):
        assert filled_df.isna().sum().sum() <= sap_df.isna().sum().sum()

    def test_original_values_preserved(self, sap_df, filled_df):
        mask = sap_df.notna()
        np.testing.assert_array_almost_equal(filled_df.values[mask.values], sap_df.values[mask.values], decimal=6)

    def test_no_negative_values(self, filled_df):
        assert (filled_df.select_dtypes(include=[np.number]) < 0).sum().sum() == 0

    def test_at_least_one_column_improved(self, sap_df, filled_df):
        improved = (filled_df.isna().sum() < sap_df.isna().sum()).any()
        assert improved

    def test_value_range_plausible(self, filled_df):
        assert filled_df.max().max() <= 150

    def test_per_column_report(self, sap_df, filled_df):
        for col in sap_df.columns:
            orig = sap_df[col].isna().sum()
            after = filled_df[col].isna().sum()
            pct = (1 - after / orig) * 100 if orig > 0 else 0
            print(f"{col}: {orig} -> {after} ({pct:.1f}% filled)")

    def test_vpd_correlation(self, filled_df, env_df):
        """Even on a different biome, VPD should correlate positively."""
        if "vpd" not in env_df.columns:
            pytest.skip("vpd not available")
        common = filled_df.index.intersection(env_df.index)
        col = filled_df.columns[0]
        merged = pd.DataFrame({"sap": filled_df.loc[common, col], "vpd": env_df.loc[common, "vpd"]}).dropna()
        if len(merged) < 100:
            pytest.skip("Not enough data")
        corr = merged["sap"].corr(merged["vpd"])
        print(f"{col} vs vpd: r={corr:.3f}")
        assert corr > 0

    def test_diurnal_pattern(self, filled_df, env_df):
        """Daytime mean > nighttime mean."""
        if "ext_rad" not in env_df.columns:
            pytest.skip("ext_rad not available")
        common = filled_df.index.intersection(env_df.index)
        day_mask = env_df.loc[common, "ext_rad"] > 0
        col = filled_df.columns[0]
        day_mean = filled_df.loc[common][col][day_mask].dropna().mean()
        night_mean = filled_df.loc[common][col][~day_mask].dropna().mean()
        if pd.isna(day_mean) or pd.isna(night_mean):
            pytest.skip("Not enough day/night data")
        print(f"{col}: day={day_mean:.2f}, night={night_mean:.2f}")
        assert day_mean > night_mean
