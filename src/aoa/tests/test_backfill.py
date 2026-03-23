"""Tests for backfill.py (M3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from src.aoa.backfill import (
    _create_spatial_groups,
    _reconstruct_fold_labels,
    backfill_from_saved_arrays,
)
from src.aoa.prepare import load_aoa_reference
from src.aoa.tests.conftest import FEATURE_NAMES, N_FEATURES, N_FOLDS, N_TRAIN


class TestReconstructFoldLabels:
    def test_deterministic(self):
        """Same inputs produce same fold labels."""
        groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        pfts = np.array(["ENF"] * 6 + ["DBF"] * 6)
        labels1 = _reconstruct_fold_labels(groups, pfts, n_splits=3)
        labels2 = _reconstruct_fold_labels(groups, pfts, n_splits=3)
        assert_allclose(labels1, labels2)

    def test_all_folds_assigned(self):
        groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        pfts = np.array(["ENF"] * 4 + ["DBF"] * 6)
        labels = _reconstruct_fold_labels(groups, pfts, n_splits=3)
        assert (labels >= 0).all()
        assert len(np.unique(labels)) == 3

    def test_consistent_within_group(self):
        """All samples in same group should be in same fold."""
        groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3])
        pfts = np.array(["ENF"] * 6 + ["DBF"] * 4)
        labels = _reconstruct_fold_labels(groups, pfts, n_splits=3)
        for g in np.unique(groups):
            fold_set = set(labels[groups == g])
            assert len(fold_set) == 1, f"Group {g} split across folds: {fold_set}"


class TestCreateSpatialGroups:
    def test_same_location_same_group(self):
        lats = np.array([50.0, 50.0, 51.0])
        lons = np.array([10.0, 10.0, 11.0])
        groups = _create_spatial_groups(lats, lons)
        assert groups[0] == groups[1]

    def test_different_locations_different_groups(self):
        lats = np.array([0.0, 50.0])
        lons = np.array([0.0, 100.0])
        groups = _create_spatial_groups(lats, lons)
        assert groups[0] != groups[1]


class TestBackfillFromSavedArrays:
    def test_produces_valid_reference(
        self,
        synthetic_X_train,
        synthetic_fold_labels,
        synthetic_shap_weights,
        tmp_path,
        rng,
    ):
        """Round-trip: save arrays, backfill, load reference."""
        run_id = "test_bf"
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # Save mock training data as parquet
        df = pd.DataFrame(synthetic_X_train, columns=FEATURE_NAMES)
        df.to_parquet(model_dir / f"FINAL_X_train_{run_id}.parquet")

        # Save mock CV folds
        np.savez(
            model_dir / f"FINAL_cv_folds_{run_id}.npz",
            fold_labels=synthetic_fold_labels,
        )

        # Save mock SHAP CSV
        shap_csv = tmp_path / "shap.csv"
        shap_df = pd.DataFrame(
            {
                "feature": FEATURE_NAMES,
                "importance": synthetic_shap_weights,
            }
        )
        shap_df.to_csv(shap_csv, index=False)

        path = backfill_from_saved_arrays(
            model_dir,
            run_id,
            shap_csv,
            FEATURE_NAMES,
        )
        assert path.exists()
        ref = load_aoa_reference(path)
        assert ref["d_bar"] > 0
        assert ref["threshold"] >= 0
        assert len(ref["feature_names"]) == N_FEATURES

    def test_missing_x_train_raises(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        np.savez(model_dir / "FINAL_cv_folds_test.npz", fold_labels=np.array([0]))
        with pytest.raises(FileNotFoundError, match="Training data"):
            backfill_from_saved_arrays(
                model_dir,
                "test",
                tmp_path / "shap.csv",
                ["a"],
            )

    def test_missing_folds_raises(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        pd.DataFrame({"a": [1]}).to_parquet(model_dir / "FINAL_X_train_test.parquet")
        with pytest.raises(FileNotFoundError, match="CV folds"):
            backfill_from_saved_arrays(
                model_dir,
                "test",
                tmp_path / "shap.csv",
                ["a"],
            )
