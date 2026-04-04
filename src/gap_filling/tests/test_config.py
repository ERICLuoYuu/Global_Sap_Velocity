from pathlib import Path

import pytest


def test_config_defaults():
    from src.gap_filling.config import GapFillingConfig

    cfg = GapFillingConfig()
    assert cfg.metric == "r2_pooled"
    assert cfg.threshold == 0.9
    assert cfg.buffer_hours == 500
    assert cfg.min_data_ml == 500
    assert cfg.min_data_dl == 2000
    assert cfg.prefer_non_env_below_h == 6
    assert cfg.time_scale == "hourly"


def test_config_override():
    from src.gap_filling.config import GapFillingConfig

    cfg = GapFillingConfig(threshold=0.85, min_data_dl=3000)
    assert cfg.threshold == 0.85
    assert cfg.min_data_dl == 3000
    assert cfg.metric == "r2_pooled"


def test_config_lookup_paths():
    from src.gap_filling.config import GapFillingConfig

    cfg = GapFillingConfig()
    assert "benchmark_results_full.csv" in str(cfg.lookup_csv_sap)
    assert "benchmark_results_full_env.csv" in str(cfg.lookup_csv_env)


def test_config_eligible_groups():
    from src.gap_filling.config import GapFillingConfig

    cfg = GapFillingConfig()
    # < min_data_ml: only A, B
    assert cfg.eligible_groups(300) == ["A", "B"]
    # >= min_data_ml, < min_data_dl: A, B, C, Ce
    assert cfg.eligible_groups(800) == ["A", "B", "C", "Ce"]
    # >= min_data_dl: all
    assert cfg.eligible_groups(3000) == ["A", "B", "C", "Ce", "D", "De"]


def test_config_frozen():
    from src.gap_filling.config import GapFillingConfig

    cfg = GapFillingConfig()
    with pytest.raises(AttributeError):
        cfg.threshold = 0.5
