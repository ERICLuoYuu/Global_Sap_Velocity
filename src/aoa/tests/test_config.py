"""Tests for AOAConfig dataclass (M0)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from src.aoa.config import AOAConfig


def _valid_kwargs():
    """Minimal valid AOAConfig kwargs."""
    return dict(
        model_type="xgb",
        run_id="test_run",
        time_scale="daily",
        aoa_reference_path=Path("/tmp/ref.npz"),
        input_dir=Path("/tmp/input"),
        model_config_path=Path("/tmp/config.json"),
        output_dir=Path("/tmp/output"),
    )


class TestAOAConfigHappyPath:
    def test_instantiates_with_valid_args(self):
        config = AOAConfig(**_valid_kwargs())
        assert config.model_type == "xgb"
        assert config.run_id == "test_run"
        assert config.iqr_multiplier == 1.5
        assert config.batch_size == 500_000

    def test_auto_resolve_save_per_timestamp_daily(self):
        config = AOAConfig(**_valid_kwargs())
        assert config.save_per_timestamp is True

    def test_auto_resolve_save_per_timestamp_hourly(self):
        kw = _valid_kwargs()
        kw["time_scale"] = "hourly"
        config = AOAConfig(**kw)
        assert config.save_per_timestamp is False

    def test_explicit_save_per_timestamp_overrides_auto(self):
        kw = _valid_kwargs()
        kw["save_per_timestamp"] = False
        config = AOAConfig(**kw)
        assert config.save_per_timestamp is False

    def test_default_months_all_12(self):
        config = AOAConfig(**_valid_kwargs())
        assert config.months == tuple(range(1, 13))

    def test_custom_years_immutable(self):
        kw = _valid_kwargs()
        kw["years"] = (2015, 2016, 2017)
        config = AOAConfig(**kw)
        assert config.years == (2015, 2016, 2017)


class TestAOAConfigValidation:
    def test_invalid_time_scale_raises(self):
        kw = _valid_kwargs()
        kw["time_scale"] = "weekly"
        with pytest.raises(ValueError, match="time_scale"):
            AOAConfig(**kw)

    def test_negative_iqr_multiplier_raises(self):
        kw = _valid_kwargs()
        kw["iqr_multiplier"] = -1.0
        with pytest.raises(ValueError, match="iqr_multiplier"):
            AOAConfig(**kw)

    def test_zero_iqr_multiplier_raises(self):
        kw = _valid_kwargs()
        kw["iqr_multiplier"] = 0.0
        with pytest.raises(ValueError, match="iqr_multiplier"):
            AOAConfig(**kw)

    def test_invalid_map_format_raises(self):
        kw = _valid_kwargs()
        kw["map_format"] = "png"
        with pytest.raises(ValueError, match="map_format"):
            AOAConfig(**kw)


class TestAOAConfigImmutability:
    def test_frozen_rejects_mutation(self):
        config = AOAConfig(**_valid_kwargs())
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.batch_size = 100

    def test_frozen_rejects_new_attribute(self):
        config = AOAConfig(**_valid_kwargs())
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.new_field = "oops"
