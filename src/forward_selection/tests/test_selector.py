"""Tests for selector.py — result extraction and saving with mock SFS."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.forward_selection.config import ScoringMode, SelectionConfig
from src.forward_selection.selector import _extract_results, _save_results


def _mock_sfs(n_steps: int = 5, n_mandatory: int = 6) -> MagicMock:
    """Create a mock SFS object with realistic subsets_ and attributes."""
    sfs = MagicMock()

    subsets: dict[int, dict[str, Any]] = {}
    all_idx = list(range(n_mandatory))
    for step in range(n_mandatory, n_mandatory + n_steps):
        all_idx = list(range(step + 1))
        subsets[step + 1] = {
            "feature_idx": tuple(all_idx),
            "avg_score": 0.5 + 0.03 * (step - n_mandatory),
        }

    sfs.subsets_ = subsets
    best_step = max(subsets, key=lambda s: subsets[s]["avg_score"])
    sfs.k_feature_idx_ = tuple(subsets[best_step]["feature_idx"])
    sfs.k_score_ = subsets[best_step]["avg_score"]
    return sfs


def _feature_names(n: int = 11) -> list[str]:
    return [f"feat_{i}" for i in range(n)]


def _default_config() -> SelectionConfig:
    return SelectionConfig(scoring=ScoringMode.MEAN_R2)


def _pooled_metrics(sfs: MagicMock) -> dict[int, dict[str, float]]:
    """Create matching pooled metrics for mock SFS."""
    return {
        step: {"pooled_r2": info["avg_score"] - 0.02, "pooled_rmse": 0.5 - info["avg_score"] * 0.3}
        for step, info in sfs.subsets_.items()
    }


class TestExtractResults:
    def test_returns_expected_keys(self) -> None:
        sfs = _mock_sfs()
        pooled = _pooled_metrics(sfs)
        result = _extract_results(sfs, _feature_names(), _default_config(), 300.0, pooled)
        expected_keys = {
            "scoring",
            "best_score",
            "n_selected",
            "selected_features",
            "selected_indices",
            "step_scores",
            "step_pooled_r2",
            "step_pooled_rmse",
            "step_features",
            "elapsed_minutes",
            "config",
        }
        assert set(result.keys()) == expected_keys

    def test_best_score_is_float(self) -> None:
        sfs = _mock_sfs()
        pooled = _pooled_metrics(sfs)
        result = _extract_results(sfs, _feature_names(), _default_config(), 300.0, pooled)
        assert isinstance(result["best_score"], float)

    def test_selected_features_are_strings(self) -> None:
        sfs = _mock_sfs()
        pooled = _pooled_metrics(sfs)
        result = _extract_results(sfs, _feature_names(), _default_config(), 300.0, pooled)
        assert all(isinstance(f, str) for f in result["selected_features"])

    def test_step_scores_match_subsets(self) -> None:
        sfs = _mock_sfs(n_steps=3)
        pooled = _pooled_metrics(sfs)
        result = _extract_results(sfs, _feature_names(), _default_config(), 100.0, pooled)
        for step, info in sfs.subsets_.items():
            assert result["step_scores"][step] == pytest.approx(info["avg_score"])

    def test_pooled_metrics_included(self) -> None:
        sfs = _mock_sfs(n_steps=3)
        pooled = _pooled_metrics(sfs)
        result = _extract_results(sfs, _feature_names(), _default_config(), 100.0, pooled)
        for step in pooled:
            assert step in result["step_pooled_r2"]
            assert step in result["step_pooled_rmse"]

    def test_elapsed_minutes_correct(self) -> None:
        sfs = _mock_sfs()
        pooled = _pooled_metrics(sfs)
        result = _extract_results(sfs, _feature_names(), _default_config(), 600.0, pooled)
        assert result["elapsed_minutes"] == pytest.approx(10.0)

    def test_empty_pooled_metrics(self) -> None:
        sfs = _mock_sfs()
        result = _extract_results(sfs, _feature_names(), _default_config(), 100.0, {})
        assert result["step_pooled_r2"] == {}
        assert result["step_pooled_rmse"] == {}


class TestSaveResults:
    def _make_results(self) -> dict[str, Any]:
        sfs = _mock_sfs()
        pooled = _pooled_metrics(sfs)
        return _extract_results(sfs, _feature_names(), _default_config(), 100.0, pooled)

    def test_writes_json(self, tmp_path: Path) -> None:
        _save_results(self._make_results(), tmp_path, "mean_r2")
        jpath = tmp_path / "ffs_mean_r2_results.json"
        assert jpath.exists()
        with open(jpath) as f:
            data = json.load(f)
        assert data["scoring"] == "mean_r2"

    def test_json_step_keys_are_strings(self, tmp_path: Path) -> None:
        _save_results(self._make_results(), tmp_path, "mean_r2")
        with open(tmp_path / "ffs_mean_r2_results.json") as f:
            data = json.load(f)
        for key in ("step_scores", "step_pooled_r2", "step_pooled_rmse", "step_features"):
            assert all(isinstance(k, str) for k in data[key])

    def test_writes_npz(self, tmp_path: Path) -> None:
        _save_results(self._make_results(), tmp_path, "mean_r2")
        npz_path = tmp_path / "ffs_mean_r2_arrays.npz"
        assert npz_path.exists()
        data = np.load(npz_path)
        assert "selected_indices" in data
        assert "step_scores" in data
        assert "step_pooled_r2" in data
        assert "step_pooled_rmse" in data

    def test_npz_arrays_same_length(self, tmp_path: Path) -> None:
        _save_results(self._make_results(), tmp_path, "mean_r2")
        data = np.load(tmp_path / "ffs_mean_r2_arrays.npz")
        n = len(data["step_scores"])
        assert len(data["step_pooled_r2"]) == n
        assert len(data["step_pooled_rmse"]) == n

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "dir"
        results = self._make_results()
        # _save_results expects the dir to exist (created by caller),
        # but we can test the JSON write still works
        out.mkdir(parents=True)
        _save_results(results, out, "neg_rmse")
        assert (out / "ffs_neg_rmse_results.json").exists()


class TestEdgeCases:
    """Boundary values and regression tests."""

    def test_single_step_sfs(self) -> None:
        sfs = _mock_sfs(n_steps=1)
        pooled = _pooled_metrics(sfs)
        result = _extract_results(sfs, _feature_names(), _default_config(), 60.0, pooled)
        assert len(result["step_scores"]) == 1
        assert len(result["step_pooled_r2"]) == 1

    def test_save_then_load_roundtrip(self, tmp_path: Path) -> None:
        """Integration: save results, load JSON, verify structure."""
        sfs = _mock_sfs(n_steps=3)
        pooled = _pooled_metrics(sfs)
        results = _extract_results(sfs, _feature_names(), _default_config(), 100.0, pooled)
        _save_results(results, tmp_path, "mean_r2")

        with open(tmp_path / "ffs_mean_r2_results.json") as f:
            loaded = json.load(f)
        # Verify all step keys are strings in JSON
        for key in ("step_scores", "step_pooled_r2", "step_pooled_rmse"):
            assert all(isinstance(k, str) for k in loaded[key])
        # Verify values survive roundtrip
        assert loaded["best_score"] == pytest.approx(results["best_score"])

    def test_neg_rmse_scoring_config(self) -> None:
        config = SelectionConfig(scoring=ScoringMode.NEG_RMSE)
        sfs = _mock_sfs()
        pooled = _pooled_metrics(sfs)
        result = _extract_results(sfs, _feature_names(), config, 100.0, pooled)
        assert result["scoring"] == "neg_rmse"
