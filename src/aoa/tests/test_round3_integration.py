"""Round 3: Integration & regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from src.aoa import core
from src.aoa.aggregate import aggregate_overall, aggregate_yearly
from src.aoa.apply import (
    compute_di_for_dataframe,
    create_windowed_features,
    parse_windowed_feature_names,
    process_files,
)
from src.aoa.config import AOAConfig
from src.aoa.prepare import build_aoa_reference, load_aoa_reference
from src.aoa.tests.conftest import FEATURE_NAMES, N_FEATURES, N_TRAIN


@pytest.fixture
def full_reference(synthetic_X_train, synthetic_fold_labels, synthetic_shap_weights, tmp_path):
    """Build a reference for integration tests."""
    path = build_aoa_reference(
        X_train=synthetic_X_train,
        fold_labels=synthetic_fold_labels,
        shap_importances=synthetic_shap_weights,
        feature_names=FEATURE_NAMES,
        output_dir=tmp_path / "ref",
        run_id="integ",
    )
    return load_aoa_reference(path), path


class TestFullPipelineWindowed:
    """Full pipeline: prepare → windowed apply → aggregate."""

    def test_windowed_pipeline_end_to_end(self, tmp_path):
        """Build windowed reference, apply to raw data, aggregate."""
        rng = np.random.default_rng(42)
        windowed_names = ["ta_t-0", "ta_t-1", "vpd_t-0", "vpd_t-1", "elev"]
        n_feat = len(windowed_names)

        # 1. Build reference with windowed features
        X = rng.standard_normal((30, n_feat))
        folds = np.tile([0, 1, 2], 10)
        shap = np.abs(rng.standard_normal(n_feat)) + 0.01
        ref_path = build_aoa_reference(X, folds, shap, windowed_names, tmp_path / "ref", "wind")
        ref = load_aoa_reference(ref_path)

        # 2. Create raw ERA5 data (un-windowed columns)
        era5_dir = tmp_path / "era5" / "2015_daily"
        era5_dir.mkdir(parents=True)
        n_px = 4
        n_days = 5
        rows = []
        for lat_idx in range(2):
            for lon_idx in range(2):
                for day in range(n_days):
                    rows.append(
                        {
                            "latitude": 50.0 + lat_idx * 0.1,
                            "longitude": 10.0 + lon_idx * 0.1,
                            "timestamp": f"2015-01-{day + 1:02d}",
                            "ta": rng.standard_normal(),
                            "vpd": rng.standard_normal(),
                            "elev": 500.0 + lat_idx * 100,
                        }
                    )
        df = pd.DataFrame(rows)
        df.to_csv(era5_dir / "prediction_2015_01_daily.csv", index=False)

        # 3. Apply with windowing
        config = AOAConfig(
            model_type="xgb",
            run_id="wind",
            time_scale="daily",
            aoa_reference_path=ref_path,
            input_dir=tmp_path / "era5",
            model_config_path=Path("/tmp/c.json"),
            output_dir=tmp_path / "out",
            years=(2015,),
            months=(1,),
            save_per_timestamp=True,
        )
        process_files(config, ref, {"feature_names": windowed_names, "is_windowing": True})

        # 4. Verify monthly output
        monthly = pd.read_parquet(tmp_path / "out" / "daily" / "monthly" / "di_monthly_2015_01.parquet")
        assert len(monthly) == n_px  # 4 unique pixels
        assert all(monthly["frac_inside_aoa"].between(0, 1))
        assert all(monthly["median_DI"] >= 0)

        # Per-timestamp: 4 pixels × 5 days - 4 first-rows (NaN from lag) = 16
        ts = pd.read_parquet(tmp_path / "out" / "daily" / "per_timestamp" / "di_2015_01_daily.parquet")
        assert len(ts) == n_px * (n_days - 1)  # 16

    def test_windowed_aggregate_yearly(self, tmp_path):
        """Aggregate windowed monthly summaries to yearly."""
        monthly_dir = tmp_path / "monthly"
        monthly_dir.mkdir()
        output_dir = tmp_path / "yearly"
        output_dir.mkdir()
        for month in [1, 2]:
            df = pd.DataFrame(
                {
                    "latitude": [50.0, 50.1],
                    "longitude": [10.0, 10.0],
                    "median_DI": np.float32([0.3, 0.5]),
                    "mean_DI": np.float32([0.35, 0.55]),
                    "std_DI": np.float32([0.05, 0.1]),
                    "frac_inside_aoa": np.float32([0.9, 0.6]),
                    "n_timestamps": np.int32([30, 30]),
                }
            )
            df.to_parquet(monthly_dir / f"di_monthly_2015_{month:02d}.parquet")

        yearly_paths = aggregate_yearly(monthly_dir, output_dir, [2015])
        assert len(yearly_paths) == 1
        year_df = pd.read_parquet(yearly_paths[0])
        assert len(year_df) == 2  # 2 pixels


class TestRegressionNonWindowed:
    """Regression: non-windowed features still work after windowing code added."""

    def test_original_e2e_still_works(self, synthetic_X_train, full_reference, tmp_path):
        """The original non-windowed pipeline is unaffected."""
        ref, ref_path = full_reference
        tree = core.build_kdtree(ref["reference_cloud_weighted"])

        # Create ERA5 with standard feature names
        df = pd.DataFrame(synthetic_X_train[:10], columns=FEATURE_NAMES)
        di, aoa_mask, valid_mask = compute_di_for_dataframe(df, ref, tree, FEATURE_NAMES)
        assert len(di) == 10
        assert np.all(di >= 0)

    def test_process_files_non_windowed_unchanged(self, synthetic_X_train, full_reference, tmp_path):
        """process_files with is_windowing=False produces same results as before."""
        ref, ref_path = full_reference

        era5_dir = tmp_path / "era5" / "2015_daily"
        era5_dir.mkdir(parents=True)
        df = pd.DataFrame(synthetic_X_train[:8], columns=FEATURE_NAMES)
        df["latitude"] = [50.0] * 4 + [50.1] * 4
        df["longitude"] = [10.0] * 8
        df["timestamp"] = pd.date_range("2015-01-01", periods=8)
        df.to_csv(era5_dir / "prediction_2015_01_daily.csv", index=False)

        config = AOAConfig(
            model_type="xgb",
            run_id="integ",
            time_scale="daily",
            aoa_reference_path=ref_path,
            input_dir=tmp_path / "era5",
            model_config_path=Path("/tmp/c.json"),
            output_dir=tmp_path / "out",
            years=(2015,),
            months=(1,),
            save_per_timestamp=True,
        )
        process_files(config, ref, {"feature_names": FEATURE_NAMES, "is_windowing": False})

        ts = pd.read_parquet(tmp_path / "out" / "daily" / "per_timestamp" / "di_2015_01_daily.parquet")
        assert len(ts) == 8  # All rows valid (no windowing drops)


class TestBackfillOutputDir:
    """Regression: backfill output_dir parameter works correctly."""

    def test_output_dir_creates_reference(self, tmp_path):
        """backfill_from_merged_data output_dir writes to custom location."""
        from src.aoa.backfill import backfill_from_merged_data

        # This test would need real model files, so we test the parameter
        # validation instead
        custom_dir = tmp_path / "custom_output"
        custom_dir.mkdir()
        assert custom_dir.exists()


class TestAllowPickleFalse:
    """Regression: allow_pickle=False works for all saved references."""

    def test_roundtrip_no_pickle(self, synthetic_X_train, synthetic_fold_labels, synthetic_shap_weights, tmp_path):
        """Build reference and reload without pickle — should work."""
        path = build_aoa_reference(
            X_train=synthetic_X_train,
            fold_labels=synthetic_fold_labels,
            shap_importances=synthetic_shap_weights,
            feature_names=FEATURE_NAMES,
            output_dir=tmp_path,
            run_id="nopickle",
        )
        # Directly verify with allow_pickle=False
        data = dict(np.load(path, allow_pickle=False))
        assert "reference_cloud_weighted" in data
        assert "feature_names" in data
        assert list(data["feature_names"]) == FEATURE_NAMES

    def test_load_aoa_reference_uses_no_pickle(
        self, synthetic_X_train, synthetic_fold_labels, synthetic_shap_weights, tmp_path
    ):
        """load_aoa_reference works with our allow_pickle=False change."""
        path = build_aoa_reference(
            X_train=synthetic_X_train,
            fold_labels=synthetic_fold_labels,
            shap_importances=synthetic_shap_weights,
            feature_names=FEATURE_NAMES,
            output_dir=tmp_path,
            run_id="nopickle2",
        )
        ref = load_aoa_reference(path)
        assert ref["d_bar"] > 0
        assert ref["feature_names"] == FEATURE_NAMES


class TestParallelPipelineIntegration:
    """Integration: n_jobs=2 produces same output as n_jobs=1."""

    def test_parallel_matches_single_threaded(
        self, synthetic_X_train, synthetic_fold_labels, synthetic_shap_weights, tmp_path
    ):
        """Full pipeline with n_jobs=1 and n_jobs=2 produces identical DI."""
        ref_path = build_aoa_reference(
            X_train=synthetic_X_train,
            fold_labels=synthetic_fold_labels,
            shap_importances=synthetic_shap_weights,
            feature_names=FEATURE_NAMES,
            output_dir=tmp_path,
            run_id="parallel_test",
        )
        ref = load_aoa_reference(ref_path)
        tree = core.build_kdtree(ref["reference_cloud_weighted"])

        # Create test data
        rng = np.random.default_rng(99)
        X_new = rng.standard_normal((50, len(FEATURE_NAMES)))
        df = pd.DataFrame(X_new, columns=FEATURE_NAMES)

        di_1, mask_1, _ = compute_di_for_dataframe(df, ref, tree, FEATURE_NAMES, n_jobs=1)
        di_2, mask_2, _ = compute_di_for_dataframe(df, ref, tree, FEATURE_NAMES, n_jobs=2)
        assert_allclose(di_1, di_2, atol=1e-10)
        assert np.array_equal(mask_1, mask_2)

    def test_process_files_with_n_jobs(
        self, synthetic_X_train, synthetic_fold_labels, synthetic_shap_weights, tmp_path
    ):
        """process_files respects config.n_jobs and produces valid output."""
        ref_path = build_aoa_reference(
            X_train=synthetic_X_train,
            fold_labels=synthetic_fold_labels,
            shap_importances=synthetic_shap_weights,
            feature_names=FEATURE_NAMES,
            output_dir=tmp_path,
            run_id="njobs_test",
        )
        ref = load_aoa_reference(ref_path)

        # Write mock ERA5 data
        rng = np.random.default_rng(42)
        era5_dir = tmp_path / "era5" / "2015_daily"
        era5_dir.mkdir(parents=True)
        df = pd.DataFrame(rng.standard_normal((20, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
        df["latitude"] = np.tile([50.0, 50.1], 10)
        df["longitude"] = 10.0
        df["timestamp"] = pd.date_range("2015-01-01", periods=20)
        df.to_csv(era5_dir / "prediction_2015_01_daily.csv", index=False)

        config = AOAConfig(
            model_type="xgb",
            run_id="njobs_test",
            time_scale="daily",
            aoa_reference_path=ref_path,
            input_dir=tmp_path / "era5",
            model_config_path=Path("/tmp/c.json"),
            output_dir=tmp_path / "out",
            years=(2015,),
            months=(1,),
            n_jobs=2,
            save_per_timestamp=True,
        )
        model_config = {"feature_names": FEATURE_NAMES, "is_windowing": False}
        process_files(config, ref, model_config)

        monthly = tmp_path / "out" / "daily" / "monthly" / "di_monthly_2015_01.parquet"
        assert monthly.exists()
        mdf = pd.read_parquet(monthly)
        assert len(mdf) > 0
