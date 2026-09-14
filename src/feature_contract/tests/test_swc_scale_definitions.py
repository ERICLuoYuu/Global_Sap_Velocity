"""The soil-moisture scale switch: which scales exist, and how the level columns are replaced."""

import numpy as np
import pandas as pd
import pytest
from src.feature_contract.definitions import (
    DEFAULT_SWC_SCALE,
    SWC_SCALES,
    rescale_swc_levels,
    validate_swc_scale,
)

L1 = "volumetric_soil_water_layer_1"
SUFFIXES = ["", "_min", "_max", "_sum"]


def _frame():
    columns = {"vpd": [1.0, 2.0]}
    for offset, suffix in enumerate(SUFFIXES):
        columns[L1 + suffix] = [10.0 + offset, 11.0 + offset]
        columns[L1 + "_zscore" + suffix] = [-1.0 - offset, 1.0 + offset]
        columns[L1 + "_raw" + suffix] = [0.2 + offset, 0.3 + offset]
    return pd.DataFrame(columns)


class TestValidateSwcScale:
    def test_the_scales_and_the_default(self):
        assert SWC_SCALES == ("ratio", "zscore", "raw")
        assert DEFAULT_SWC_SCALE == "ratio"

    @pytest.mark.parametrize("scale", SWC_SCALES)
    def test_every_scale_is_accepted_under_servable(self, scale):
        assert validate_swc_scale(scale, "servable") == scale

    def test_ratio_is_accepted_under_site(self):
        assert validate_swc_scale("ratio", "site") == "ratio"

    @pytest.mark.parametrize("scale", ["zscore", "raw"])
    def test_other_scales_are_refused_under_site(self, scale):
        with pytest.raises(ValueError, match="servable"):
            validate_swc_scale(scale, "site")

    def test_an_unknown_scale_is_refused(self):
        with pytest.raises(ValueError, match="swc scale"):
            validate_swc_scale("minmax", "servable")


class TestRescaleSwcLevels:
    def test_ratio_returns_the_frame_itself(self):
        df = _frame()
        assert rescale_swc_levels(df, "ratio") is df

    @pytest.mark.parametrize("scale,companion", [("zscore", "_zscore"), ("raw", "_raw")])
    def test_every_level_and_stat_column_takes_its_companion(self, scale, companion):
        df = _frame()
        out = rescale_swc_levels(df, scale)
        for suffix in SUFFIXES:
            np.testing.assert_array_equal(out[L1 + suffix].to_numpy(), df[L1 + companion + suffix].to_numpy())

    @pytest.mark.parametrize("scale", ["zscore", "raw"])
    def test_companions_and_other_columns_are_untouched(self, scale):
        df = _frame()
        out = rescale_swc_levels(df, scale)
        assert list(out.columns) == list(df.columns)
        for col in df.columns:
            if col.startswith(L1) and ("_zscore" in col or "_raw" in col) or col == "vpd":
                pd.testing.assert_series_equal(out[col], df[col])

    @pytest.mark.parametrize("scale", ["zscore", "raw"])
    def test_the_input_is_not_mutated(self, scale):
        df = _frame()
        before = df.copy()
        rescale_swc_levels(df, scale)
        pd.testing.assert_frame_equal(df, before)

    def test_a_level_column_without_its_companion_raises(self):
        df = _frame().drop(columns=[L1 + "_zscore_min"])
        with pytest.raises(KeyError, match="_zscore_min"):
            rescale_swc_levels(df, "zscore")

    def test_a_frame_without_soil_moisture_passes_through(self):
        df = pd.DataFrame({"vpd": [1.0, 2.0]})
        pd.testing.assert_frame_equal(rescale_swc_levels(df, "raw"), df)

    def test_an_unknown_scale_raises(self):
        with pytest.raises(ValueError, match="swc scale"):
            rescale_swc_levels(_frame(), "minmax")
