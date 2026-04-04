"""Tests for prepare.py (M2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.aoa.prepare import REQUIRED_NPZ_KEYS, build_aoa_reference, load_aoa_reference
from src.aoa.tests.conftest import FEATURE_NAMES, N_FEATURES, N_FOLDS, N_TRAIN


class TestBuildReferenceRoundtrip:
    def test_roundtrip(
        self,
        synthetic_X_train,
        synthetic_fold_labels,
        synthetic_shap_weights,
        tmp_path,
    ):
        path = build_aoa_reference(
            X_train=synthetic_X_train,
            fold_labels=synthetic_fold_labels,
            shap_importances=synthetic_shap_weights,
            feature_names=FEATURE_NAMES,
            output_dir=tmp_path,
            run_id="test",
        )
        assert path.exists()
        ref = load_aoa_reference(path)
        assert ref["reference_cloud_weighted"].shape == (N_TRAIN, N_FEATURES)
        assert ref["feature_means"].shape == (N_FEATURES,)
        assert ref["feature_stds"].shape == (N_FEATURES,)
        assert ref["feature_weights"].shape == (N_FEATURES,)
        assert ref["training_di"].shape == (N_TRAIN,)
        assert ref["fold_assignments"].shape == (N_TRAIN,)
        assert len(ref["feature_names"]) == N_FEATURES

    def test_all_keys_present(
        self,
        synthetic_X_train,
        synthetic_fold_labels,
        synthetic_shap_weights,
        tmp_path,
    ):
        path = build_aoa_reference(
            X_train=synthetic_X_train,
            fold_labels=synthetic_fold_labels,
            shap_importances=synthetic_shap_weights,
            feature_names=FEATURE_NAMES,
            output_dir=tmp_path,
            run_id="test",
        )
        data = dict(np.load(path, allow_pickle=True))
        assert set(data.keys()) >= REQUIRED_NPZ_KEYS

    def test_values_consistent(
        self,
        synthetic_X_train,
        synthetic_fold_labels,
        synthetic_shap_weights,
        tmp_path,
    ):
        path = build_aoa_reference(
            X_train=synthetic_X_train,
            fold_labels=synthetic_fold_labels,
            shap_importances=synthetic_shap_weights,
            feature_names=FEATURE_NAMES,
            output_dir=tmp_path,
            run_id="test",
        )
        ref = load_aoa_reference(path)
        assert ref["d_bar"] > 0
        assert ref["threshold"] >= 0
        assert ref["threshold"] <= np.max(ref["training_di"]) + 1e-10

    def test_feature_names_preserved(
        self,
        synthetic_X_train,
        synthetic_fold_labels,
        synthetic_shap_weights,
        tmp_path,
    ):
        path = build_aoa_reference(
            X_train=synthetic_X_train,
            fold_labels=synthetic_fold_labels,
            shap_importances=synthetic_shap_weights,
            feature_names=FEATURE_NAMES,
            output_dir=tmp_path,
            run_id="test",
        )
        ref = load_aoa_reference(path)
        assert ref["feature_names"] == FEATURE_NAMES


class TestBuildReferenceValidation:
    def test_rejects_nan(
        self,
        synthetic_fold_labels,
        synthetic_shap_weights,
        tmp_path,
    ):
        X = np.ones((N_TRAIN, N_FEATURES))
        X[5, 2] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            build_aoa_reference(
                X_train=X,
                fold_labels=synthetic_fold_labels,
                shap_importances=synthetic_shap_weights,
                feature_names=FEATURE_NAMES,
                output_dir=tmp_path,
                run_id="test",
            )

    def test_rejects_shape_mismatch(
        self,
        synthetic_X_train,
        synthetic_shap_weights,
        tmp_path,
    ):
        wrong_folds = np.array([0, 1, 2])  # wrong length
        with pytest.raises(ValueError, match="fold_labels"):
            build_aoa_reference(
                X_train=synthetic_X_train,
                fold_labels=wrong_folds,
                shap_importances=synthetic_shap_weights,
                feature_names=FEATURE_NAMES,
                output_dir=tmp_path,
                run_id="test",
            )

    def test_rejects_nan_shap(
        self,
        synthetic_X_train,
        synthetic_fold_labels,
        tmp_path,
    ):
        bad_shap = np.ones(N_FEATURES)
        bad_shap[0] = np.nan
        with pytest.raises(ValueError, match="shap_importances"):
            build_aoa_reference(
                X_train=synthetic_X_train,
                fold_labels=synthetic_fold_labels,
                shap_importances=bad_shap,
                feature_names=FEATURE_NAMES,
                output_dir=tmp_path,
                run_id="test",
            )


class TestDBarMethodSample:
    def test_sample_method_runs(
        self,
        synthetic_X_train,
        synthetic_fold_labels,
        synthetic_shap_weights,
        tmp_path,
    ):
        path = build_aoa_reference(
            X_train=synthetic_X_train,
            fold_labels=synthetic_fold_labels,
            shap_importances=synthetic_shap_weights,
            feature_names=FEATURE_NAMES,
            output_dir=tmp_path,
            run_id="test_sample",
            d_bar_method="sample:20",
        )
        ref = load_aoa_reference(path)
        assert ref["d_bar"] > 0
        assert ref["d_bar_method"] == "sample:20"


class TestLoadValidation:
    def test_missing_keys_raises(self, tmp_path):
        bad_path = tmp_path / "bad.npz"
        np.savez(bad_path, d_bar=1.0, threshold=0.5)
        with pytest.raises(ValueError, match="missing keys"):
            load_aoa_reference(bad_path)
