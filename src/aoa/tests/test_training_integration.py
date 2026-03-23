"""Tests for M7 training integration — AOA reference generation in training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from src.aoa.prepare import REQUIRED_NPZ_KEYS, build_aoa_reference, load_aoa_reference
from src.aoa.tests.conftest import FEATURE_NAMES, N_FEATURES, N_TRAIN


class TestTrainingIntegrationPattern:
    """Test the code pattern that M7 adds to the training script."""

    def test_fold_label_derivation(self, rng):
        """The fold-label derivation pattern works as expected."""
        from sklearn.model_selection import StratifiedGroupKFold

        n = 100
        groups = np.repeat(np.arange(10), 10)
        pfts = np.array(["ENF"] * 50 + ["DBF"] * 50)
        pfts_encoded, _ = pd.factorize(pfts)

        outer_cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
        fold_labels = np.full(n, -1, dtype=int)
        X_dummy = np.zeros((n, 1))
        for fold_idx, (_, test_idx) in enumerate(outer_cv.split(X_dummy, pfts_encoded, groups)):
            fold_labels[test_idx] = fold_idx

        assert (fold_labels >= 0).all()
        assert len(np.unique(fold_labels)) == 10

    def test_shap_csv_to_importances(self, tmp_path):
        """SHAP CSV parsing matches expected pattern."""
        shap_df = pd.DataFrame(
            {
                "feature": FEATURE_NAMES,
                "importance": np.abs(np.arange(N_FEATURES, dtype=float)) + 0.01,
            }
        )
        csv_path = tmp_path / "shap.csv"
        shap_df.to_csv(csv_path, index=False)

        loaded = pd.read_csv(csv_path)
        importances = loaded.set_index("feature")["importance"].reindex(FEATURE_NAMES).values
        assert not np.any(np.isnan(importances))
        assert len(importances) == N_FEATURES

    def test_full_integration_pattern(
        self,
        synthetic_X_train,
        synthetic_shap_weights,
        tmp_path,
        rng,
    ):
        """Simulate the M7 code path: derive folds + call build_aoa_reference."""
        from sklearn.model_selection import StratifiedGroupKFold

        X_all_records = synthetic_X_train
        n = len(X_all_records)
        groups_all_records = np.repeat(np.arange(5), n // 5)
        pfts_all_records = np.array(["ENF"] * (n // 2) + ["DBF"] * (n - n // 2))

        pfts_encoded, _ = pd.factorize(pfts_all_records)
        outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        fold_labels = np.full(n, -1, dtype=int)
        X_dummy = np.zeros((n, 1))
        for fold_idx, (_, test_idx) in enumerate(outer_cv.split(X_dummy, pfts_encoded, groups_all_records)):
            fold_labels[test_idx] = fold_idx

        shap_importances = synthetic_shap_weights
        run_id = "test_integration"
        model_dir = tmp_path / "model"

        path = build_aoa_reference(
            X_train=X_all_records,
            fold_labels=fold_labels,
            shap_importances=shap_importances,
            feature_names=FEATURE_NAMES,
            output_dir=model_dir,
            run_id=run_id,
        )

        assert path.exists()
        ref = load_aoa_reference(path)
        assert set(ref.keys()) >= set(REQUIRED_NPZ_KEYS)
        assert ref["reference_cloud_weighted"].shape == (n, N_FEATURES)

        # Also save backups (simulating the M7 pattern)
        np.savez(
            model_dir / f"FINAL_cv_folds_{run_id}.npz",
            fold_labels=fold_labels,
            spatial_groups=groups_all_records,
            pfts=pfts_all_records,
        )
        pd.DataFrame(X_all_records, columns=FEATURE_NAMES).to_parquet(
            model_dir / f"FINAL_X_train_{run_id}.parquet",
        )

        assert (model_dir / f"FINAL_cv_folds_{run_id}.npz").exists()
        assert (model_dir / f"FINAL_X_train_{run_id}.parquet").exists()
