"""Round 1: Edge cases & boundary values for windowed + CAST features."""

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
    create_windowed_features,
    generate_slurm_script,
    parse_windowed_feature_names,
)
from src.aoa.config import AOAConfig
from src.aoa.prepare import build_aoa_reference, load_aoa_reference
from src.aoa.tests.conftest import FEATURE_NAMES, N_FEATURES


class TestWindowedEdgeCases:
    """Edge cases for windowed feature parsing and creation."""

    def test_empty_feature_list(self):
        """Empty feature list produces empty dynamic and static."""
        dynamic, static = parse_windowed_feature_names([])
        assert dynamic == {}
        assert static == []

    def test_high_lag_values(self):
        """Handles large lag indices (t-10, t-99)."""
        names = ["ta_t-0", "ta_t-10", "ta_t-99"]
        dynamic, static = parse_windowed_feature_names(names)
        assert sorted(dynamic["ta"]) == [0, 10, 99]

    def test_feature_name_with_hyphen(self):
        """Feature names containing hyphens don't confuse the parser."""
        # "precip-rate_t-1" should parse as base="precip-rate", lag=1
        names = ["precip-rate_t-0", "precip-rate_t-1", "elevation"]
        dynamic, static = parse_windowed_feature_names(names)
        assert "precip-rate" in dynamic
        assert sorted(dynamic["precip-rate"]) == [0, 1]
        assert static == ["elevation"]

    def test_feature_name_with_underscore_t(self):
        """Feature like 'soil_temp_t-0' parses correctly."""
        names = ["soil_temp_t-0", "soil_temp_t-1"]
        dynamic, static = parse_windowed_feature_names(names)
        assert "soil_temp" in dynamic
        assert sorted(dynamic["soil_temp"]) == [0, 1]

    def test_single_pixel_all_nan_windowed(self):
        """Single pixel where all lagged values are NaN (first timestamp only)."""
        df = pd.DataFrame(
            {
                "latitude": [50.0],
                "longitude": [10.0],
                "timestamp": ["2015-01-01"],
                "ta": [1.0],
                "elevation": [500.0],
            }
        )
        dynamic = {"ta": [0, 1]}
        static = ["elevation"]
        result = create_windowed_features(df, dynamic, static)
        # Only 1 row, t-1 should be NaN
        assert np.isnan(result["ta_t-1"].iloc[0])
        assert result["ta_t-0"].iloc[0] == 1.0

    def test_windowed_two_pixels_independent_lags(self):
        """Lag creation is independent per pixel — pixel B's lag shouldn't use pixel A's data."""
        df = pd.DataFrame(
            {
                "latitude": [50.0, 50.0, 60.0, 60.0],
                "longitude": [10.0, 10.0, 20.0, 20.0],
                "timestamp": ["2015-01-01", "2015-01-02", "2015-01-01", "2015-01-02"],
                "ta": [1.0, 2.0, 100.0, 200.0],
            }
        )
        dynamic = {"ta": [0, 1]}
        static = []
        result = create_windowed_features(df, dynamic, static)
        # Check pixel (60, 20): t-1 of day 2 should be 100 (not 2)
        pixel_60 = result[(result["latitude"] == 60.0) & (result["timestamp"] == "2015-01-02")]
        assert pixel_60["ta_t-1"].iloc[0] == 100.0


class TestMinimalTrainingData:
    """Boundary: minimum viable training data (2 samples, 2 folds)."""

    def test_two_samples_two_folds(self, tmp_path):
        """Smallest possible training set: 2 samples in 2 folds."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        folds = np.array([0, 1])
        shap = np.array([0.5, 0.5])
        names = ["f1", "f2"]

        path = build_aoa_reference(X, folds, shap, names, tmp_path, "mini")
        ref = load_aoa_reference(path)
        assert ref["d_bar"] > 0
        assert ref["threshold"] > 0
        assert len(ref["training_di"]) == 2

    def test_single_feature(self, tmp_path):
        """Works with a single feature."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 1))
        folds = np.tile([0, 1], 10)
        shap = np.array([1.0])
        names = ["only_feature"]

        path = build_aoa_reference(X, folds, shap, names, tmp_path, "1feat")
        ref = load_aoa_reference(path)
        tree = core.build_kdtree(ref["reference_cloud_weighted"])
        di = core.compute_prediction_di(
            core.apply_importance_weights(
                core.standardize_features(X[:3], ref["feature_means"], ref["feature_stds"]),
                ref["feature_weights"],
            ),
            tree,
            ref["d_bar"],
        )
        assert len(di) == 3
        assert np.all(di >= 0)


class TestSlurmPathQuoting:
    """Verify SLURM script paths are quoted (review fix)."""

    def test_paths_are_quoted(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = AOAConfig(
            model_type="xgb",
            run_id="test",
            time_scale="daily",
            aoa_reference_path=Path("/path with spaces/ref.npz"),
            input_dir=Path("/another path/input"),
            model_config_path=Path("/yet another/config.json"),
            output_dir=Path("/output dir/aoa"),
            years=(2015,),
            months=(1,),
        )
        generate_slurm_script(config, None)
        script = (tmp_path / "job_aoa_apply_test.sh").read_text()
        assert '"/path with spaces/ref.npz"' in script
        assert '"/another path/input"' in script
        assert '"/yet another/config.json"' in script
        assert '"/output dir/aoa"' in script
