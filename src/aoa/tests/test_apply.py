"""Tests for apply.py (M4)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from src.aoa import core
from src.aoa.apply import (
    compute_di_for_dataframe,
    compute_monthly_summary,
    discover_era5_files,
    load_input_file,
    load_model_config,
    parse_range,
    select_and_validate_features,
    write_aoa_meta,
)
from src.aoa.config import AOAConfig
from src.aoa.prepare import build_aoa_reference, load_aoa_reference
from src.aoa.tests.conftest import FEATURE_NAMES, N_FEATURES, N_TRAIN


@pytest.fixture
def reference_and_tree(synthetic_X_train, synthetic_fold_labels, synthetic_shap_weights, tmp_path):
    """Build a reference NPZ and return (reference_dict, tree, path)."""
    path = build_aoa_reference(
        X_train=synthetic_X_train,
        fold_labels=synthetic_fold_labels,
        shap_importances=synthetic_shap_weights,
        feature_names=FEATURE_NAMES,
        output_dir=tmp_path,
        run_id="test",
    )
    ref = load_aoa_reference(path)
    tree = core.build_kdtree(ref["reference_cloud_weighted"])
    return ref, tree, path


@pytest.fixture
def model_config_json(tmp_path):
    """Write a mock model config JSON."""
    config = {
        "model_type": "xgb",
        "run_id": "test",
        "feature_names": FEATURE_NAMES,
        "data_info": {"IS_WINDOWING": False, "input_width": 2, "shift": 1},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


class TestParseRange:
    def test_range_format(self):
        assert parse_range("1995-2018") == list(range(1995, 2019))

    def test_comma_format(self):
        assert parse_range("2015,2016,2017") == [2015, 2016, 2017]

    def test_single_value(self):
        assert parse_range("2020") == [2020]


class TestLoadModelConfig:
    def test_loads_correctly(self, model_config_json):
        config = load_model_config(model_config_json)
        assert config["model_type"] == "xgb"
        assert config["run_id"] == "test"
        assert config["feature_names"] == FEATURE_NAMES
        assert config["is_windowing"] is False


class TestLoadInputFile:
    def test_csv_loading(self, era5_like_df, tmp_path):
        csv_path = tmp_path / "test.csv"
        era5_like_df.to_csv(csv_path, index=False)
        df = load_input_file(csv_path)
        assert len(df) == len(era5_like_df)

    def test_parquet_loading(self, era5_like_df, tmp_path):
        pq_path = tmp_path / "test.parquet"
        era5_like_df.to_parquet(pq_path, index=False)
        df = load_input_file(pq_path)
        assert len(df) == len(era5_like_df)

    def test_duplicate_column_dedup(self, tmp_path):
        csv_path = tmp_path / "dup.csv"
        csv_path.write_text("a,b,a\n1,2,3\n4,5,6\n")
        df = load_input_file(csv_path)
        assert not df.columns.duplicated().any()

    def test_unsupported_format_raises(self, tmp_path):
        bad = tmp_path / "test.xlsx"
        bad.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported"):
            load_input_file(bad)


class TestSelectAndValidateFeatures:
    def test_reorders_columns(self, era5_like_df):
        ref_names = list(reversed(FEATURE_NAMES))
        result = select_and_validate_features(era5_like_df, FEATURE_NAMES, ref_names)
        assert list(result.columns) == ref_names

    def test_missing_feature_raises(self, era5_like_df):
        ref_names = FEATURE_NAMES + ["missing_feat"]
        with pytest.raises(ValueError, match="Missing"):
            select_and_validate_features(era5_like_df, ref_names, ref_names)


class TestComputeDIForDataframe:
    def test_prepare_apply_roundtrip(self, synthetic_X_train, reference_and_tree):
        ref, tree, _ = reference_and_tree
        df = pd.DataFrame(synthetic_X_train, columns=FEATURE_NAMES)
        di, aoa_mask, valid_mask = compute_di_for_dataframe(df, ref, tree, FEATURE_NAMES)
        assert len(di) == N_TRAIN
        assert len(aoa_mask) == N_TRAIN
        assert np.all(di >= 0)

    def test_training_point_near_zero(self, synthetic_X_train, reference_and_tree):
        ref, tree, _ = reference_and_tree
        single = synthetic_X_train[:1]
        df = pd.DataFrame(single, columns=FEATURE_NAMES)
        di, _, _ = compute_di_for_dataframe(df, ref, tree, FEATURE_NAMES)
        assert di[0] < 0.5  # should be very close to 0

    def test_nan_rows_dropped(self, reference_and_tree):
        ref, tree, _ = reference_and_tree
        data = np.ones((10, N_FEATURES))
        data[3, 0] = np.nan
        data[7, 2] = np.nan
        df = pd.DataFrame(data, columns=FEATURE_NAMES)
        di, aoa_mask, valid_mask = compute_di_for_dataframe(df, ref, tree, FEATURE_NAMES)
        assert len(di) == 8
        assert (~valid_mask).sum() == 2

    def test_all_nan_returns_empty(self, reference_and_tree):
        ref, tree, _ = reference_and_tree
        data = np.full((5, N_FEATURES), np.nan)
        df = pd.DataFrame(data, columns=FEATURE_NAMES)
        di, aoa_mask, valid_mask = compute_di_for_dataframe(df, ref, tree, FEATURE_NAMES)
        assert len(di) == 0

    def test_csv_parquet_equivalence(self, era5_like_df, reference_and_tree, tmp_path):
        ref, tree, _ = reference_and_tree
        csv_path = tmp_path / "test.csv"
        pq_path = tmp_path / "test.parquet"
        era5_like_df.to_csv(csv_path, index=False)
        era5_like_df.to_parquet(pq_path, index=False)
        df_csv = load_input_file(csv_path)
        df_pq = load_input_file(pq_path)
        di_csv, _, _ = compute_di_for_dataframe(df_csv, ref, tree, FEATURE_NAMES)
        di_pq, _, _ = compute_di_for_dataframe(df_pq, ref, tree, FEATURE_NAMES)
        assert_allclose(di_csv, di_pq, atol=1e-5)


class TestDiscoverERA5Files:
    def test_finds_matching_files(self, tmp_path):
        d = tmp_path / "2015_daily"
        d.mkdir()
        (d / "prediction_2015_01_daily.csv").write_text("a,b\n1,2\n")
        (d / "prediction_2015_02_daily.csv").write_text("a,b\n1,2\n")
        files = discover_era5_files(tmp_path, "daily", (2015,), (1, 2))
        assert len(files) == 2

    def test_missing_year_dir_skips(self, tmp_path):
        files = discover_era5_files(tmp_path, "daily", (2099,), (1,))
        assert len(files) == 0

    def test_month_filter(self, tmp_path):
        d = tmp_path / "2015_daily"
        d.mkdir()
        (d / "prediction_2015_01_daily.csv").write_text("a,b\n1,2\n")
        (d / "prediction_2015_06_daily.csv").write_text("a,b\n1,2\n")
        files = discover_era5_files(tmp_path, "daily", (2015,), (6,))
        assert len(files) == 1
        assert "06" in files[0].name


class TestComputeMonthlySummary:
    def test_aggregates_correctly(self):
        df = pd.DataFrame(
            {
                "latitude": [50.0, 50.0, 50.0, 50.1, 50.1],
                "longitude": [10.0, 10.0, 10.0, 10.0, 10.0],
                "timestamp": pd.date_range("2015-01-01", periods=5),
                "DI": [0.1, 0.2, 0.3, 0.5, 0.6],
                "aoa_mask": [True, True, True, False, False],
            }
        )
        summary = compute_monthly_summary(df)
        assert len(summary) == 2
        row_50 = summary[summary["latitude"] == 50.0].iloc[0]
        assert_allclose(row_50["median_DI"], 0.2, atol=1e-5)
        assert_allclose(row_50["frac_inside_aoa"], 1.0)
        assert row_50["n_timestamps"] == 3


class TestWriteAOAMeta:
    def test_writes_valid_json(self, reference_and_tree, tmp_path):
        ref, _, _ = reference_and_tree
        config = AOAConfig(
            model_type="xgb",
            run_id="test",
            time_scale="daily",
            aoa_reference_path=Path("/tmp/ref.npz"),
            input_dir=Path("/tmp/input"),
            model_config_path=Path("/tmp/config.json"),
            output_dir=tmp_path,
            years=(2015,),
        )
        write_aoa_meta(config, ref, tmp_path)
        meta_path = tmp_path / "aoa_meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["run_id"] == "test"
        assert meta["threshold"] == ref["threshold"]
        assert meta["d_bar"] == ref["d_bar"]
        assert "created_at" in meta
        assert meta["feature_names"] == FEATURE_NAMES
