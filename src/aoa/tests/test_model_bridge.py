"""Tests for model bridge — loading artifacts and reconstructing CV folds for AOA."""

import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.aoa.model_bridge import (
    load_aoa_arrays,
    load_model_config,
    load_shap_weights,
    reconstruct_fold_indices,
    save_aoa_arrays,
)


class TestLoadModelConfig:
    """Tests for loading and validating model config JSON."""

    def test_loads_valid_config(self, tmp_path):
        """Should load config and return dict with required keys."""
        config = {
            "model_type": "xgb",
            "run_id": "test_run",
            "cv_results": {"n_folds": 10},
            "feature_names": ["a", "b", "c"],
            "random_seed": 42,
            "split_type": "spatial_stratified",
            "data_info": {"n_samples": 100, "n_features": 3},
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        result = load_model_config(config_path)

        assert result["run_id"] == "test_run"
        assert result["cv_results"]["n_folds"] == 10
        assert result["feature_names"] == ["a", "b", "c"]

    def test_missing_file_raises(self, tmp_path):
        """Should raise FileNotFoundError for missing config."""
        with pytest.raises(FileNotFoundError):
            load_model_config(tmp_path / "nonexistent.json")

    def test_missing_required_keys_raises(self, tmp_path):
        """Should raise ValueError if required keys are missing."""
        config = {"model_type": "xgb"}  # missing feature_names, etc.
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="missing required"):
            load_model_config(config_path)


class TestLoadShapWeights:
    """Tests for loading SHAP feature importance as AOA weights."""

    def test_loads_and_aligns_to_feature_names(self, tmp_path):
        """Weights should be ordered to match feature_names list."""
        csv_content = "feature,importance,category,unit\nb,0.3,Static,\na,0.5,Static,\nc,0.2,Static,\n"
        csv_path = tmp_path / "shap_feature_importance.csv"
        csv_path.write_text(csv_content)

        feature_names = ["a", "b", "c"]
        weights = load_shap_weights(csv_path, feature_names)

        assert weights.shape == (3,)
        # a=0.5, b=0.3, c=0.2 → ordered by feature_names
        np.testing.assert_allclose(weights, [0.5, 0.3, 0.2])

    def test_normalizes_to_sum_one(self, tmp_path):
        """Weights should be normalized so they sum to 1."""
        csv_content = "feature,importance,category,unit\na,10.0,S,\nb,20.0,S,\nc,30.0,S,\n"
        csv_path = tmp_path / "shap_feature_importance.csv"
        csv_path.write_text(csv_content)

        weights = load_shap_weights(csv_path, ["a", "b", "c"])

        assert weights.sum() == pytest.approx(1.0)

    def test_missing_feature_raises(self, tmp_path):
        """Should raise if CSV doesn't contain all required features."""
        csv_content = "feature,importance,category,unit\na,0.5,S,\nb,0.3,S,\n"
        csv_path = tmp_path / "shap_feature_importance.csv"
        csv_path.write_text(csv_content)

        with pytest.raises(ValueError, match="missing.*features"):
            load_shap_weights(csv_path, ["a", "b", "c"])


class TestReconstructFoldIndices:
    """Tests for reconstructing CV fold indices from groups and PFTs."""

    def test_produces_correct_number_of_folds(self):
        """Should produce n_folds fold index arrays."""
        rng = np.random.default_rng(42)
        n = 200
        groups = rng.integers(0, 10, size=n)
        pfts = rng.integers(0, 3, size=n)

        folds = reconstruct_fold_indices(
            n_samples=n,
            groups=groups,
            pfts_encoded=pfts,
            n_folds=10,
            random_seed=42,
            split_type="spatial_stratified",
        )

        assert len(folds) == 10

    def test_folds_cover_all_indices(self):
        """Union of all fold test indices should cover all samples."""
        rng = np.random.default_rng(42)
        n = 200
        groups = rng.integers(0, 10, size=n)
        pfts = rng.integers(0, 3, size=n)

        folds = reconstruct_fold_indices(
            n_samples=n,
            groups=groups,
            pfts_encoded=pfts,
            n_folds=10,
            random_seed=42,
            split_type="spatial_stratified",
        )

        all_indices = np.concatenate(folds)
        assert len(np.unique(all_indices)) == n

    def test_folds_are_disjoint(self):
        """No sample should appear in more than one fold."""
        rng = np.random.default_rng(42)
        n = 200
        groups = rng.integers(0, 10, size=n)
        pfts = rng.integers(0, 3, size=n)

        folds = reconstruct_fold_indices(
            n_samples=n,
            groups=groups,
            pfts_encoded=pfts,
            n_folds=10,
            random_seed=42,
            split_type="spatial_stratified",
        )

        all_indices = np.concatenate(folds)
        assert len(all_indices) == len(np.unique(all_indices))

    def test_deterministic_with_same_seed(self):
        """Same inputs + seed should produce identical folds."""
        rng = np.random.default_rng(42)
        n = 200
        groups = rng.integers(0, 10, size=n)
        pfts = rng.integers(0, 3, size=n)

        folds1 = reconstruct_fold_indices(n, groups, pfts, 10, 42, "spatial_stratified")
        folds2 = reconstruct_fold_indices(n, groups, pfts, 10, 42, "spatial_stratified")

        for f1, f2 in zip(folds1, folds2):
            np.testing.assert_array_equal(f1, f2)

    def test_group_kfold_fallback(self):
        """split_type != 'spatial_stratified' should use GroupKFold."""
        rng = np.random.default_rng(42)
        n = 200
        groups = rng.integers(0, 10, size=n)
        pfts = rng.integers(0, 3, size=n)

        folds = reconstruct_fold_indices(n, groups, pfts, 10, 42, "grouped")

        assert len(folds) == 10
        all_indices = np.concatenate(folds)
        assert len(np.unique(all_indices)) == n


class TestSaveLoadAoaArrays:
    """Tests for saving and loading AOA arrays to/from disk."""

    def test_roundtrip_preserves_data(self, tmp_path):
        """save then load should return identical arrays."""
        rng = np.random.default_rng(99)
        X = rng.random((50, 5))
        groups = rng.integers(0, 10, size=50)
        pfts = rng.integers(0, 4, size=50)

        save_aoa_arrays(tmp_path, "test_run", X, groups, pfts)
        X_loaded, groups_loaded, pfts_loaded = load_aoa_arrays(tmp_path, "test_run")

        np.testing.assert_array_equal(X_loaded, X)
        np.testing.assert_array_equal(groups_loaded, groups)
        np.testing.assert_array_equal(pfts_loaded, pfts)

    def test_creates_expected_files(self, tmp_path):
        """Should create three .npy files with correct naming."""
        X = np.zeros((10, 3))
        groups = np.zeros(10)
        pfts = np.zeros(10)

        save_aoa_arrays(tmp_path, "my_run", X, groups, pfts)

        assert (tmp_path / "aoa_X_train_my_run.npy").exists()
        assert (tmp_path / "aoa_groups_my_run.npy").exists()
        assert (tmp_path / "aoa_pfts_encoded_my_run.npy").exists()

    def test_load_missing_files_raises(self, tmp_path):
        """Should raise FileNotFoundError if .npy files don't exist."""
        with pytest.raises(FileNotFoundError):
            load_aoa_arrays(tmp_path, "nonexistent")
