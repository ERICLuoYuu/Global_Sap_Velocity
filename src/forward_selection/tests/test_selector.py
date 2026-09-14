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
            "step_pooled_r2_log",
            "step_pooled_rmse_log",
            "step_mean_fold_r2_log",
            "step_mean_fold_r2_orig",
            "step_features",
            "metric_scales",
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


class TestPooledMetricScales:
    """The two headline numbers must not be reported on different scales in silence.

    `step_scores` comes from mlxtend, which fits and scores on log1p(y).
    `step_pooled_r2` was back-transformed with expm1 and scored on the original
    scale. Adjacent keys in one JSON, different targets -- and on the real corpus
    that accounted for 0.109 of the 0.164 apparent mean-vs-pooled gap, two thirds
    of it, while reading as a cross-validation effect.

    The fix does not pick a winner. It records BOTH scales for BOTH aggregations
    so any comparison drawn from the file is like-for-like by construction.
    """

    @staticmethod
    def _skewed_problem(n: int = 120):
        """A right-skewed target, so the two scales genuinely disagree."""
        from sklearn.linear_model import LinearRegression

        rng = np.random.RandomState(0)
        x = rng.randn(n, 3)
        # Lognormal target with SUBSTANTIAL irreducible noise. Both halves matter:
        # skew puts the raw-scale variance in a tail, and the noise stops the
        # model fitting it. A near-perfect fit scores ~1.0 on BOTH scales and the
        # comparison is vacuous -- an earlier version of this fixture used
        # expm1(linear), which log1p inverts exactly, and the test could not fail.
        y_orig = np.exp(1.5 + 0.8 * x[:, 0] + 1.2 * rng.randn(n))
        y_log = np.log1p(y_orig)
        cv = [
            (np.arange(n)[np.arange(n) % 3 != k], np.arange(n)[np.arange(n) % 3 == k])
            for k in range(3)
        ]
        sfs = MagicMock()
        sfs.subsets_ = {3: {"feature_idx": (0, 1, 2), "avg_score": 0.5}}
        return sfs, x, y_log, y_orig, cv, LinearRegression()

    def test_reports_both_scales_for_both_aggregations(self) -> None:
        from src.forward_selection.selector import _compute_pooled_metrics

        sfs, x, y_log, y_orig, cv, est = self._skewed_problem()
        out = _compute_pooled_metrics(sfs, x, y_log, cv, est, 1, y_orig=y_orig)

        entry = out[3]
        assert {
            "pooled_r2",
            "pooled_rmse",
            "pooled_r2_log",
            "pooled_rmse_log",
            "mean_fold_r2_log",
            "mean_fold_r2_orig",
        } <= set(entry)

    def test_the_two_scales_actually_differ(self) -> None:
        """Negative control: if these matched, the test above would prove nothing."""
        from src.forward_selection.selector import _compute_pooled_metrics

        sfs, x, y_log, y_orig, cv, est = self._skewed_problem()
        entry = _compute_pooled_metrics(sfs, x, y_log, cv, est, 1, y_orig=y_orig)[3]
        assert abs(entry["pooled_r2"] - entry["pooled_r2_log"]) > 0.01

    def test_mean_fold_log_matches_hand_computation(self) -> None:
        """mean_fold_r2_log must be the UNWEIGHTED mean of per-fold R2 on log."""
        from sklearn.base import clone
        from sklearn.metrics import r2_score
        from sklearn.model_selection import cross_val_predict

        from src.forward_selection.selector import _compute_pooled_metrics

        sfs, x, y_log, y_orig, cv, est = self._skewed_problem()
        entry = _compute_pooled_metrics(sfs, x, y_log, cv, est, 1, y_orig=y_orig)[3]

        oof = cross_val_predict(clone(est), x, y_log, cv=cv, n_jobs=1)
        by_fold = [r2_score(y_log[te], oof[te]) for _, te in cv]
        assert entry["mean_fold_r2_log"] == pytest.approx(float(np.mean(by_fold)), abs=1e-9)

    def test_pooled_is_not_the_mean_of_folds(self) -> None:
        """Pooled scores one concatenated vector; it is not the per-fold average."""
        from src.forward_selection.selector import _compute_pooled_metrics

        sfs, x, y_log, y_orig, cv, est = self._skewed_problem()
        entry = _compute_pooled_metrics(sfs, x, y_log, cv, est, 1, y_orig=y_orig)[3]
        assert entry["pooled_r2_log"] != entry["mean_fold_r2_log"]

    def test_without_y_orig_both_scales_collapse(self) -> None:
        """No back-transform requested -> the 'original' keys are the log ones."""
        from src.forward_selection.selector import _compute_pooled_metrics

        sfs, x, y_log, _y_orig, cv, est = self._skewed_problem()
        entry = _compute_pooled_metrics(sfs, x, y_log, cv, est, 1, y_orig=None)[3]
        assert entry["pooled_r2"] == pytest.approx(entry["pooled_r2_log"])
        assert entry["mean_fold_r2_orig"] == pytest.approx(entry["mean_fold_r2_log"])


class TestMetricScales:
    """Every metric key must carry a machine-readable statement of its scale."""

    def test_names_the_scale_of_each_metric(self) -> None:
        sfs = _mock_sfs()
        result = _extract_results(sfs, _feature_names(), _default_config(), 300.0, _pooled_metrics(sfs))
        scales = result["metric_scales"]
        assert scales["step_scores"] == "log1p"
        assert scales["step_pooled_r2"] == "original"
        assert scales["step_pooled_r2_log"] == "log1p"
        assert scales["step_mean_fold_r2_orig"] == "original"

    def test_tolerates_pooled_dicts_without_the_new_keys(self) -> None:
        """Old pooled dicts (two keys) must still extract, leaving the rest empty."""
        sfs = _mock_sfs()
        result = _extract_results(sfs, _feature_names(), _default_config(), 300.0, _pooled_metrics(sfs))
        assert result["step_pooled_r2"]
        assert result["step_pooled_r2_log"] == {}

    def test_new_keys_survive_the_json_roundtrip(self, tmp_path: Path) -> None:
        sfs = _mock_sfs()
        pooled = {
            step: {
                "pooled_r2": 0.4,
                "pooled_rmse": 3.5,
                "pooled_r2_log": 0.57,
                "pooled_rmse_log": 0.42,
                "mean_fold_r2_log": 0.58,
                "mean_fold_r2_orig": 0.47,
            }
            for step in sfs.subsets_
        }
        result = _extract_results(sfs, _feature_names(), _default_config(), 300.0, pooled)
        _save_results(result, tmp_path, "mean_r2")
        with open(tmp_path / "ffs_mean_r2_results.json") as fh:
            loaded = json.load(fh)
        step = sorted(loaded["step_pooled_r2_log"])[0]
        assert loaded["step_pooled_r2_log"][step] == pytest.approx(0.57)
        assert loaded["step_mean_fold_r2_orig"][step] == pytest.approx(0.47)
        assert loaded["metric_scales"]["step_pooled_r2"] == "original"
