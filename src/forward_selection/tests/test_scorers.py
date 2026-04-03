"""Tests for scorers.py."""

import pytest

from src.forward_selection.config import ScoringMode
from src.forward_selection.scorers import get_scorer


class TestGetScorer:
    def test_mean_r2_returns_string(self) -> None:
        result = get_scorer(ScoringMode.MEAN_R2)
        assert result == "r2"

    def test_neg_rmse_returns_string(self) -> None:
        result = get_scorer(ScoringMode.NEG_RMSE)
        assert result == "neg_root_mean_squared_error"

    def test_string_input(self) -> None:
        result = get_scorer("mean_r2")
        assert result == "r2"

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            get_scorer("invalid_scorer")
