"""Tests for config.py — dataclasses and enums."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.forward_selection.config import HyperparamConfig, ScoringMode, SelectionConfig


class TestScoringMode:
    def test_only_two_modes(self) -> None:
        assert len(ScoringMode) == 2

    def test_mean_r2_value(self) -> None:
        assert ScoringMode.MEAN_R2.value == "mean_r2"

    def test_neg_rmse_value(self) -> None:
        assert ScoringMode.NEG_RMSE.value == "neg_rmse"

    def test_construct_from_string(self) -> None:
        assert ScoringMode("mean_r2") is ScoringMode.MEAN_R2

    def test_invalid_string_raises(self) -> None:
        with pytest.raises(ValueError):
            ScoringMode("pooled_r2")


class TestHyperparamConfig:
    def test_defaults(self) -> None:
        cfg = HyperparamConfig()
        assert cfg.n_estimators == 1000
        assert cfg.learning_rate == 0.01
        assert cfg.max_depth == 10
        assert cfg.reg_lambda == 10.0

    def test_frozen(self) -> None:
        cfg = HyperparamConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.n_estimators = 500  # type: ignore[misc]

    def test_to_xgb_params_keys(self) -> None:
        params = HyperparamConfig().to_xgb_params()
        expected_keys = {
            "n_estimators",
            "learning_rate",
            "max_depth",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "gamma",
            "reg_alpha",
            "reg_lambda",
            "n_jobs",
            "random_state",
            "verbosity",
        }
        assert set(params.keys()) == expected_keys

    def test_to_xgb_params_custom_njobs(self) -> None:
        params = HyperparamConfig().to_xgb_params(n_jobs=16, random_state=123)
        assert params["n_jobs"] == 16
        assert params["random_state"] == 123

    def test_to_xgb_params_verbosity_zero(self) -> None:
        params = HyperparamConfig().to_xgb_params()
        assert params["verbosity"] == 0


class TestSelectionConfig:
    def test_defaults(self) -> None:
        cfg = SelectionConfig(scoring=ScoringMode.MEAN_R2)
        assert cfg.n_splits == 10
        assert cfg.random_seed == 42
        assert cfg.n_jobs_sfs == 32
        assert cfg.n_jobs_xgb == 4
        assert cfg.k_features == "best"
        assert cfg.forward is True
        assert cfg.floating is False

    def test_frozen(self) -> None:
        cfg = SelectionConfig(scoring=ScoringMode.MEAN_R2)
        with pytest.raises(FrozenInstanceError):
            cfg.n_splits = 5  # type: ignore[misc]

    def test_output_dir_default_is_path(self) -> None:
        cfg = SelectionConfig(scoring=ScoringMode.NEG_RMSE)
        assert isinstance(cfg.output_dir, Path)
