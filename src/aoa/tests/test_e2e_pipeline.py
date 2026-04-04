"""End-to-end pipeline test: prepare → apply → aggregate → visualize."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from src.aoa import core
from src.aoa.aggregate import (
    aggregate_climatological,
    aggregate_overall,
    aggregate_yearly,
)
from src.aoa.apply import (
    compute_di_for_dataframe,
    compute_monthly_summary,
    load_input_file,
    process_files,
    write_aoa_meta,
)
from src.aoa.config import AOAConfig
from src.aoa.prepare import build_aoa_reference, load_aoa_reference
from src.aoa.tests.conftest import FEATURE_NAMES, N_FEATURES


def _make_era5_csv(path: Path, feature_names: list[str], rng, n_days: int = 30):
    """Create a mock ERA5 CSV with realistic structure."""
    n_pixels = 6  # 3 lats x 2 lons
    n = n_pixels * n_days
    lats = np.tile([50.0, 50.1, 50.2, 50.0, 50.1, 50.2], n_days)
    lons = np.tile([10.0, 10.0, 10.0, 10.1, 10.1, 10.1], n_days)
    timestamps = np.repeat(pd.date_range("2015-01-01", periods=n_days, freq="D"), n_pixels)
    data = {"latitude": lats, "longitude": lons, "timestamp": timestamps}
    for feat in feature_names:
        data[feat] = rng.standard_normal(n)
    # Add extra cols like real ERA5 files
    data["name"] = "mock_site"
    data["solar_timestamp"] = timestamps
    df = pd.DataFrame(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


class TestE2EPipeline:
    """Full pipeline: prepare → apply → aggregate → visualize."""

    def test_full_pipeline(self, rng, tmp_path):
        # --- 1. PREPARE: Build training reference ---
        n_train = 100
        X_train = rng.standard_normal((n_train, N_FEATURES))
        # Make PFT columns one-hot
        for i in range(n_train):
            X_train[i, -3:] = 0.0
            X_train[i, -3 + rng.integers(0, 3)] = 1.0

        fold_labels = np.tile(np.arange(5), n_train // 5)
        shap_weights = np.abs(rng.standard_normal(N_FEATURES)) + 0.01

        ref_path = build_aoa_reference(
            X_train=X_train,
            fold_labels=fold_labels,
            shap_importances=shap_weights,
            feature_names=FEATURE_NAMES,
            output_dir=tmp_path / "model",
            run_id="e2e_test",
        )
        assert ref_path.exists()
        ref = load_aoa_reference(ref_path)

        # --- 2. APPLY: Process mock ERA5 files ---
        era5_base = tmp_path / "era5"
        for year in [2015, 2016]:
            for month in [1, 6]:
                csv_path = era5_base / f"{year}_daily" / f"prediction_{year}_{month:02d}_daily.csv"
                _make_era5_csv(csv_path, FEATURE_NAMES, rng, n_days=10)

        config = AOAConfig(
            model_type="xgb",
            run_id="e2e_test",
            time_scale="daily",
            aoa_reference_path=ref_path,
            input_dir=era5_base,
            model_config_path=tmp_path / "config.json",
            output_dir=tmp_path / "aoa_output",
            years=(2015, 2016),
            months=(1, 6),
            save_per_timestamp=True,
        )
        model_config = {"feature_names": FEATURE_NAMES}
        process_files(config, ref, model_config)

        # Verify apply outputs
        monthly_dir = tmp_path / "aoa_output" / "daily" / "monthly"
        ts_dir = tmp_path / "aoa_output" / "daily" / "per_timestamp"
        monthly_files = sorted(monthly_dir.glob("*.parquet"))
        ts_files = sorted(ts_dir.glob("*.parquet"))
        assert len(monthly_files) == 4  # 2 years x 2 months
        assert len(ts_files) == 4

        # Check monthly summary content
        mdf = pd.read_parquet(monthly_files[0])
        assert set(mdf.columns) >= {
            "latitude",
            "longitude",
            "median_DI",
            "mean_DI",
            "std_DI",
            "frac_inside_aoa",
            "n_timestamps",
        }
        assert len(mdf) == 6  # 6 unique pixels
        assert (mdf["median_DI"] >= 0).all()
        assert (mdf["frac_inside_aoa"] >= 0).all()
        assert (mdf["frac_inside_aoa"] <= 1).all()

        # Check per-timestamp content
        tdf = pd.read_parquet(ts_files[0])
        assert set(tdf.columns) >= {
            "latitude",
            "longitude",
            "timestamp",
            "DI",
            "aoa_mask",
        }
        assert len(tdf) == 60  # 6 pixels x 10 days
        assert (tdf["DI"] >= 0).all()

        # Check aoa_meta.json
        meta_path = tmp_path / "aoa_output" / "daily" / "aoa_meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["run_id"] == "e2e_test"
        assert meta["n_features"] == N_FEATURES

        # --- 3. AGGREGATE: yearly, climatological, overall ---
        yearly_paths = aggregate_yearly(
            monthly_dir,
            tmp_path / "aoa_output" / "daily" / "yearly",
            [2015, 2016],
        )
        assert len(yearly_paths) == 2

        clim_paths = aggregate_climatological(
            monthly_dir,
            tmp_path / "aoa_output" / "daily" / "climatological",
        )
        assert len(clim_paths) == 2  # Jan and Jun

        overall_path = aggregate_overall(
            monthly_dir,
            tmp_path / "aoa_output" / "daily" / "di_overall.parquet",
        )
        assert overall_path.exists()

        # Verify aggregation content
        yearly_df = pd.read_parquet(yearly_paths[0])
        assert len(yearly_df) == 6  # 6 pixels
        assert yearly_df["n_timestamps"].sum() == 120  # 6 pixels x 2 months x 10 days

        overall_df = pd.read_parquet(overall_path)
        assert len(overall_df) == 6
        assert overall_df["n_timestamps"].sum() == 240  # 6 pixels x 4 months x 10 days

        # --- 4. VISUALIZE: GeoTIFF generation ---
        try:
            import rasterio

            from src.aoa.visualize import parquet_to_geotiff

            tif_path = tmp_path / "test_map.tif"
            parquet_to_geotiff(
                overall_df,
                tif_path,
                di_col="median_DI",
                threshold=ref["threshold"],
            )
            assert tif_path.exists()

            with rasterio.open(tif_path) as ds:
                assert ds.crs.to_epsg() == 4326
                assert ds.count == 2
                di_band = ds.read(1)
                mask_band = ds.read(2)
                valid_di = di_band[~np.isnan(di_band)]
                assert len(valid_di) == 6
                assert (valid_di >= 0).all()
                valid_mask = mask_band[~np.isnan(mask_band)]
                assert set(valid_mask.tolist()).issubset({0.0, 1.0})
        except ImportError:
            pytest.skip("rasterio not installed")

    def test_pipeline_with_nan_rows(self, rng, tmp_path):
        """Pipeline handles NaN in prediction data gracefully."""
        n_train = 50
        X_train = rng.standard_normal((n_train, N_FEATURES))
        for i in range(n_train):
            X_train[i, -3:] = 0.0
            X_train[i, -3 + rng.integers(0, 3)] = 1.0

        fold_labels = np.tile(np.arange(5), 10)
        shap_weights = np.abs(rng.standard_normal(N_FEATURES)) + 0.01

        ref_path = build_aoa_reference(
            X_train=X_train,
            fold_labels=fold_labels,
            shap_importances=shap_weights,
            feature_names=FEATURE_NAMES,
            output_dir=tmp_path / "model",
            run_id="nan_test",
        )
        ref = load_aoa_reference(ref_path)

        # Create ERA5 with some NaN rows
        era5_dir = tmp_path / "era5" / "2015_daily"
        era5_dir.mkdir(parents=True)
        df = pd.DataFrame(rng.standard_normal((20, N_FEATURES)), columns=FEATURE_NAMES)
        df["latitude"] = [50.0] * 10 + [50.1] * 10
        df["longitude"] = [10.0] * 20
        df["timestamp"] = pd.date_range("2015-01-01", periods=20)
        # Inject NaN
        df.iloc[3, 0] = np.nan
        df.iloc[7, 2] = np.nan
        df.to_csv(era5_dir / "prediction_2015_01_daily.csv", index=False)

        config = AOAConfig(
            model_type="xgb",
            run_id="nan_test",
            time_scale="daily",
            aoa_reference_path=ref_path,
            input_dir=tmp_path / "era5",
            model_config_path=tmp_path / "c.json",
            output_dir=tmp_path / "out",
            years=(2015,),
            months=(1,),
        )
        process_files(config, ref, {"feature_names": FEATURE_NAMES})

        monthly = tmp_path / "out" / "daily" / "monthly" / "di_monthly_2015_01.parquet"
        assert monthly.exists()
        mdf = pd.read_parquet(monthly)
        # 18 valid rows across 2 pixels
        assert mdf["n_timestamps"].sum() == 18

    def test_pipeline_empty_file_skipped(self, rng, tmp_path):
        """Pipeline skips files where all features are NaN."""
        n_train = 50
        X_train = rng.standard_normal((n_train, N_FEATURES))
        for i in range(n_train):
            X_train[i, -3:] = 0.0
            X_train[i, -3 + rng.integers(0, 3)] = 1.0

        fold_labels = np.tile(np.arange(5), 10)
        shap_weights = np.abs(rng.standard_normal(N_FEATURES)) + 0.01

        ref_path = build_aoa_reference(
            X_train=X_train,
            fold_labels=fold_labels,
            shap_importances=shap_weights,
            feature_names=FEATURE_NAMES,
            output_dir=tmp_path / "model",
            run_id="empty_test",
        )
        ref = load_aoa_reference(ref_path)

        # All-NaN ERA5 file
        era5_dir = tmp_path / "era5" / "2015_daily"
        era5_dir.mkdir(parents=True)
        df = pd.DataFrame(np.full((5, N_FEATURES), np.nan), columns=FEATURE_NAMES)
        df["latitude"] = 50.0
        df["longitude"] = 10.0
        df["timestamp"] = pd.date_range("2015-01-01", periods=5)
        df.to_csv(era5_dir / "prediction_2015_01_daily.csv", index=False)

        config = AOAConfig(
            model_type="xgb",
            run_id="empty_test",
            time_scale="daily",
            aoa_reference_path=ref_path,
            input_dir=tmp_path / "era5",
            model_config_path=tmp_path / "c.json",
            output_dir=tmp_path / "out",
            years=(2015,),
            months=(1,),
        )
        process_files(config, ref, {"feature_names": FEATURE_NAMES})

        # No monthly output should be produced
        monthly_dir = tmp_path / "out" / "daily" / "monthly"
        monthly_files = list(monthly_dir.glob("*.parquet"))
        assert len(monthly_files) == 0
