"""--SWC_SCALE inside feature engineering: one scale for every soil-moisture feature."""

import numpy as np
import pandas as pd
import pytest

from src.hyperparameter_optimization.feature_engineering import (
    apply_all_feature_engineering,
    apply_feature_engineering,
)

TP = "total_precipitation_hourly_sum"
L1 = "volumetric_soil_water_layer_1"
L3 = "volumetric_soil_water_layer_3"
OUTSIDE_SOIL_MOISTURE = ["ta_lag1d", "ta_change_1d", "vpd_log", "vpd_roll14d_std", "precip_sum_7d"]


def _frame():
    rng = np.random.RandomState(2)
    n = 20
    return pd.DataFrame(
        {
            "vpd": rng.uniform(0.5, 3.0, n),
            "ta": rng.uniform(5.0, 30.0, n),
            "sw_in": rng.uniform(50.0, 400.0, n),
            "precip": np.zeros(n),
            TP: np.full(n, 0.004),
            L1: rng.uniform(4.0, 8.0, n),
            L1 + "_zscore": rng.uniform(-2.0, 2.0, n),
            L1 + "_raw": rng.uniform(0.1, 0.4, n),
            L3: rng.uniform(4.0, 8.0, n),
            L3 + "_zscore": rng.uniform(-2.0, 2.0, n),
            L3 + "_raw": rng.uniform(0.1, 0.4, n),
            "latitude": np.full(n, 51.0),
        }
    )


@pytest.mark.parametrize("scale,companion", [("zscore", "_zscore"), ("raw", "_raw")])
def test_every_soil_moisture_feature_shares_the_scale(scale, companion):
    df = _frame()
    out, _ = apply_all_feature_engineering(df, contract="servable", swc_scale=scale)
    level = df[L1 + companion]
    np.testing.assert_allclose(out[L1].to_numpy(), level.to_numpy(), rtol=1e-6)
    np.testing.assert_allclose(out["swc_lag3d"].to_numpy(), level.shift(3).to_numpy(), rtol=1e-6)
    np.testing.assert_allclose(out["vpd_x_swc"].to_numpy(), (df["vpd"] * level).to_numpy(), rtol=1e-6)
    np.testing.assert_allclose(out["ta_x_swc"].to_numpy(), (df["ta"] * level).to_numpy(), rtol=1e-6)
    np.testing.assert_allclose(out["swc_layer3_norm"].to_numpy(), df[L3 + companion].to_numpy(), rtol=1e-6)


def test_ratio_is_the_default_and_keeps_the_rebuilt_split():
    df = _frame()
    default, _ = apply_all_feature_engineering(df, contract="servable")
    ratio, _ = apply_all_feature_engineering(df, contract="servable", swc_scale="ratio")
    pd.testing.assert_frame_equal(default, ratio)
    np.testing.assert_array_equal(ratio[L1].to_numpy(), df[L1].to_numpy())
    np.testing.assert_array_equal(ratio["swc_layer3_norm"].to_numpy(), df[L3 + "_zscore"].to_numpy())


def test_features_outside_soil_moisture_do_not_depend_on_the_scale():
    df = _frame()
    ratio, _ = apply_all_feature_engineering(df, contract="servable", swc_scale="ratio")
    for scale in ("zscore", "raw"):
        other, _ = apply_all_feature_engineering(df, contract="servable", swc_scale=scale)
        for col in OUTSIDE_SOIL_MOISTURE:
            pd.testing.assert_series_equal(ratio[col], other[col])


def test_the_companion_columns_survive_unchanged():
    df = _frame()
    out, _ = apply_all_feature_engineering(df, contract="servable", swc_scale="raw")
    for col in (L1 + "_raw", L1 + "_zscore", L3 + "_raw", L3 + "_zscore"):
        np.testing.assert_array_equal(out[col].to_numpy(), df[col].to_numpy())


def test_the_same_feature_names_are_emitted_under_every_scale():
    df = _frame()
    names = {
        scale: set(apply_all_feature_engineering(df, contract="servable", swc_scale=scale)[1])
        for scale in ("ratio", "zscore", "raw")
    }
    assert names["ratio"] == names["zscore"] == names["raw"]


@pytest.mark.parametrize("scale", ["zscore", "raw"])
def test_a_non_ratio_scale_is_refused_under_the_site_contract(scale):
    with pytest.raises(ValueError, match="servable"):
        apply_all_feature_engineering(_frame(), contract="site", swc_scale=scale)
    with pytest.raises(ValueError, match="servable"):
        apply_feature_engineering(_frame(), ["swc_memory"], contract="site", swc_scale=scale)


def test_an_unknown_scale_is_refused():
    with pytest.raises(ValueError, match="swc scale"):
        apply_all_feature_engineering(_frame(), contract="servable", swc_scale="minmax")
