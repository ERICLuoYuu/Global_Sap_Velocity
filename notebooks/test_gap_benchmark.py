"""
Tests for gap_benchmark.py -- correctness, regression, and performance.

Run with:
    pytest notebooks/test_gap_benchmark.py -v
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# -- Setup project imports ---------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / ".venv"))
sys.path.insert(0, str(ROOT))

from path_config import PathConfig

_paths = PathConfig(
    scale="sapwood",
    base_data_dir=str(ROOT / "data"),
    base_output_dir=str(ROOT / "outputs"),
)
SAP_DIR = _paths.sap_outliers_removed_dir
SITE_MD_DIR = _paths.raw_csv_dir


# -- Import the module under test -------------------------------------------
from notebooks.gap_benchmark import (  # noqa: E402
    _benchmark_one_segment,
    _find_gap_ranges,
    _infer_site_name,
    compute_metrics,
    detect_gaps_in_series,
    fill_cubic,
    fill_linear,
    fill_mdv,
    fill_nearest,
    fill_rolling_mean,
    get_all_qualifying_segments,
    inject_gaps_replicated,
    load_site_metadata,
)

# ============================================================================
# CORRECTNESS TESTS
# ============================================================================


class TestInferSiteName:
    """_infer_site_name must extract full site codes, not truncate to 2 parts."""

    def test_two_part_code(self):
        assert _infer_site_name("ARG_TRE_sapf_data_outliers_removed") == "ARG_TRE"

    def test_three_part_code(self):
        assert _infer_site_name("FIN_HYY_SME_sapf_data_outliers_removed") == "FIN_HYY_SME"

    def test_four_part_code(self):
        assert _infer_site_name("AUS_CAR_THI_00F_sapf_data_outliers_removed") == "AUS_CAR_THI_00F"

    def test_no_suffix(self):
        assert _infer_site_name("ARG_TRE") == "ARG_TRE"


class TestBiomeMapping:
    """Biome metadata must be correctly loaded -- no Unknown for real sites."""

    def test_known_site_has_biome(self):
        md = load_site_metadata("FIN_HYY_SME")
        assert md["si_biome"] != "Unknown", f"Got: {md['si_biome']}"

    def test_no_unknown_biomes_for_existing_sites(self):
        md_files = sorted(SITE_MD_DIR.glob("*_site_md.csv"))
        if not md_files:
            pytest.skip("No site metadata files found")
        unknown = []
        for f in md_files:
            site = f.stem.replace("_site_md", "")
            md = load_site_metadata(site)
            if md["si_biome"] == "Unknown":
                unknown.append(site)
        assert len(unknown) == 0, f"{len(unknown)} sites Unknown: {unknown[:10]}"

    def test_multiple_biomes_represented(self):
        md_files = sorted(SITE_MD_DIR.glob("*_site_md.csv"))
        if not md_files:
            pytest.skip("No site metadata files found")
        biomes = {load_site_metadata(f.stem.replace("_site_md", ""))["si_biome"] for f in md_files}
        biomes.discard("Unknown")
        assert len(biomes) >= 5, f"Only {len(biomes)} biomes: {biomes}"


class TestCSVPathResolution:
    """All SAP files must roundtrip through _infer_site_name."""

    def test_all_sap_files_resolvable(self):
        sap_files = sorted(SAP_DIR.glob("*outliers_removed.csv"))
        if not sap_files:
            pytest.skip("No SAP data files found")
        unresolvable = []
        for f in sap_files:
            site = _infer_site_name(f.stem)
            candidates = sorted(SAP_DIR.glob(f"{site}_sapf_data_outliers_removed.csv"))
            if not candidates:
                unresolvable.append(site)
        assert len(unresolvable) == 0, f"Unresolvable: {unresolvable[:10]}"

    def test_site_name_roundtrip(self):
        sap_files = sorted(SAP_DIR.glob("*outliers_removed.csv"))
        if not sap_files:
            pytest.skip("No SAP data files found")
        for f in sap_files[:20]:
            site = _infer_site_name(f.stem)
            expected = f"{site}_sapf_data_outliers_removed.csv"
            assert f.name == expected, f"Roundtrip fail: {f.name} -> {site} -> {expected}"


class TestSegmentDeduplication:
    """No duplicate site names after _infer_site_name."""

    def test_no_duplicate_site_names(self):
        sap_files = sorted(SAP_DIR.glob("*outliers_removed.csv"))
        if not sap_files:
            pytest.skip("No SAP data files found")
        names = [_infer_site_name(f.stem) for f in sap_files]
        dupes = [s for s in names if names.count(s) > 1]
        assert len(dupes) == 0, f"Duplicates: {set(dupes)}"


# ============================================================================
# REGRESSION TESTS
# ============================================================================


class TestGapFillingRegression:
    @pytest.fixture
    def synthetic_series(self):
        rng = np.random.default_rng(42)
        n = 2000
        t = np.arange(n)
        signal = 5.0 + 3.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.3, n)
        signal = np.clip(signal, 0, None)
        idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
        return pd.Series(signal, index=idx, name="synthetic_sap")

    def test_linear_exact_on_linear_data(self):
        n = 100
        idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
        s = pd.Series(np.linspace(1, 10, n), index=idx)
        gapped = s.copy()
        gapped.iloc[40:50] = np.nan
        filled = fill_linear(gapped)
        np.testing.assert_allclose(filled.iloc[40:50], s.iloc[40:50], atol=1e-10)

    def test_fill_methods_no_nans(self, synthetic_series):
        s = synthetic_series.copy()
        s.iloc[100:124] = np.nan
        for name, fn in [
            ("linear", fill_linear),
            ("cubic", fill_cubic),
            ("nearest", fill_nearest),
            ("rolling", fill_rolling_mean),
        ]:
            filled = fn(s.copy())
            assert filled.notna().all(), f"{name} left NaN values"

    def test_fill_methods_non_negative(self, synthetic_series):
        s = synthetic_series.copy()
        s.iloc[100:124] = np.nan
        for name, fn in [
            ("linear", fill_linear),
            ("cubic", fill_cubic),
            ("nearest", fill_nearest),
            ("rolling", fill_rolling_mean),
        ]:
            filled = fn(s.copy())
            assert (filled >= 0).all(), f"{name} produced negative values"

    def test_compute_metrics_perfect(self):
        true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m = compute_metrics(true, true.copy())
        assert m["r2"] == pytest.approx(1.0)
        assert m["rmse"] == pytest.approx(0.0, abs=1e-12)

    def test_compute_metrics_known_values(self):
        true = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.1, 2.2, 2.8])
        m = compute_metrics(true, pred)
        assert m["rmse"] == pytest.approx(0.1732, abs=1e-3)


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestEdgeCases:
    def test_empty_series_gap_detection(self):
        assert detect_gaps_in_series(pd.Series(dtype=float)) == []

    def test_inject_gaps_too_short(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="h", tz="UTC")
        s = pd.Series(np.ones(10), index=idx)
        result = inject_gaps_replicated(s, 5, 3, np.random.default_rng(42))
        assert len(result) == 0

    def test_qualifying_segments_short_series(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="h", tz="UTC")
        df = pd.DataFrame({"col1": np.ones(100)}, index=idx)
        assert len(get_all_qualifying_segments(df, min_len=200)) == 0


# ============================================================================
# PERFORMANCE BENCHMARKS
# ============================================================================


class TestPerformance:
    @pytest.fixture
    def long_series(self):
        rng = np.random.default_rng(42)
        n = 8000
        t = np.arange(n)
        signal = np.clip(
            5.0 + 3.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.3, n),
            0,
            None,
        )
        idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
        return pd.Series(signal, index=idx)

    def test_cubic_under_30s(self, long_series):
        reps = inject_gaps_replicated(long_series, 24, 5, np.random.default_rng(42))
        t0 = time.time()
        for gapped, _ in reps:
            fill_cubic(gapped.copy())
        elapsed = time.time() - t0
        assert elapsed < 30.0, f"A_cubic took {elapsed:.1f}s (budget: 30s)"

    def test_mdv_under_30s(self, long_series):
        reps = inject_gaps_replicated(long_series, 24, 5, np.random.default_rng(42))
        t0 = time.time()
        for gapped, _ in reps:
            fill_mdv(gapped.copy())
        elapsed = time.time() - t0
        assert elapsed < 30.0, f"B_mdv took {elapsed:.1f}s (budget: 30s)"


# ============================================================================
# NEW: CUBIC LOCAL-WINDOW AND WORKER TESTS
# ============================================================================


class TestCubicLocalWindow:
    """Verify cubic local-window scoping produces correct results."""

    def test_find_gap_ranges_single(self):
        s = pd.Series([1, 2, np.nan, np.nan, 5, 6])
        gaps = _find_gap_ranges(s)
        assert gaps == [(2, 4)]

    def test_find_gap_ranges_multiple(self):
        s = pd.Series([1, np.nan, 3, np.nan, np.nan, 6])
        gaps = _find_gap_ranges(s)
        assert gaps == [(1, 2), (3, 5)]

    def test_find_gap_ranges_no_gaps(self):
        s = pd.Series([1, 2, 3, 4])
        assert _find_gap_ranges(s) == []

    def test_cubic_matches_linear_quality(self):
        """Cubic should be at least as good as linear for smooth data."""
        rng = np.random.default_rng(42)
        n = 2000
        t = np.arange(n)
        signal = np.clip(5.0 + 3.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.1, n), 0, None)
        idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
        s = pd.Series(signal, index=idx)
        gapped = s.copy()
        gapped.iloc[100:112] = np.nan
        cubic_filled = fill_cubic(gapped.copy())
        linear_filled = fill_linear(gapped.copy())
        cubic_err = np.abs(cubic_filled.iloc[100:112] - s.iloc[100:112]).mean()
        linear_err = np.abs(linear_filled.iloc[100:112] - s.iloc[100:112]).mean()
        # Cubic should be comparable or better
        assert cubic_err < linear_err * 2.0, f"Cubic error {cubic_err:.4f} much worse than linear {linear_err:.4f}"


class TestBenchmarkWorker:
    """Test the _benchmark_one_segment worker function."""

    @pytest.fixture
    def synthetic_segment(self):
        rng = np.random.default_rng(42)
        n = 2000
        t = np.arange(n)
        signal = np.clip(5.0 + 3.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.3, n), 0, None)
        idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
        gt = pd.Series(signal, index=idx, name="synthetic")
        return {
            "hourly": gt,
            "daily": gt.resample("D").mean(),
            "site": "TEST_SITE",
            "col": "synthetic",
            "biome": "ENF",
            "env_hourly": None,
            "env_daily": None,
        }

    def test_worker_returns_results(self, synthetic_segment):
        results = _benchmark_one_segment(
            key="TEST__synthetic",
            sdata=synthetic_segment,
            scale="hourly",
            method_name="A_linear",
            method_fn=fill_linear,
            gap_sizes=[24],
            n_reps=3,
            rng_seed=42,
        )
        assert len(results) > 0
        assert all(r["method"] == "A_linear" for r in results)
        assert all(r["time_scale"] == "hourly" for r in results)

    def test_worker_no_nans_in_metrics(self, synthetic_segment):
        results = _benchmark_one_segment(
            key="TEST__synthetic",
            sdata=synthetic_segment,
            scale="hourly",
            method_name="A_linear",
            method_fn=fill_linear,
            gap_sizes=[6],
            n_reps=3,
            rng_seed=42,
        )
        for r in results:
            assert not np.isnan(r["r2"]), f"NaN R² in result: {r}"

    def test_worker_skips_short_segment(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="h", tz="UTC")
        sdata = {"hourly": pd.Series(np.ones(10), index=idx), "site": "X", "col": "y"}
        results = _benchmark_one_segment("X__y", sdata, "hourly", "A_linear", fill_linear, [6], 3, 42)
        assert results == []
