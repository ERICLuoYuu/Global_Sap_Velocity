"""Under the servable contract, ``precip`` is ERA5-Land daily precipitation in mm/day.

A 0.1 deg prediction cell has no rain gauge, and the prediction pipeline supplies ERA5-Land daily
precipitation as a daily TOTAL in mm. Under ``site`` the column stays the gauge's daily MEAN of hourly
mm, 24 times smaller on a complete day, so a servable model trained on it would be fed values 24x the
training scale. Everything built from ``precip`` (``precip_lag1d``, precipitation memory) follows.

The total is the daily mean of the hourly ERA5 values times 24, not the merge's daily sum: the sum
counts only the hours present in the merged frame, and 2.1% of sap-days have fewer than 24.
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_contract.definitions import ERA5_HOURLY_MEAN_PRECIP_COL, era5_hourly_mean_precip_mm_per_day
from src.hyperparameter_optimization.feature_engineering import (
    apply_all_feature_engineering,
    apply_feature_engineering,
)

TP_MEAN = "total_precipitation_hourly"  # metres per hour, mean of the hours present
TP_SUM = "total_precipitation_hourly_sum"  # metres, sum of the hours present
MM_PER_DAY = [0.0, 12.0, 0.0, 3.5, 2.0]


def _frame():
    # Gauge and ERA5 disagree on purpose: the gauge is a daily mean (mm/h) of an unrelated series.
    return pd.DataFrame(
        {
            "precip": [0.5, 0.0, 0.25, 0.0, 1.0],
            TP_MEAN: [v / 24000.0 for v in MM_PER_DAY],
            TP_SUM: [v / 1000.0 for v in MM_PER_DAY],
            "ta": [10.0, 11.0, 12.0, 13.0, 14.0],
        }
    )


class TestConversion:
    def test_the_column_is_the_hourly_mean(self):
        assert ERA5_HOURLY_MEAN_PRECIP_COL == TP_MEAN

    def test_metres_per_hour_to_mm_per_day(self):
        out = era5_hourly_mean_precip_mm_per_day(pd.Series([0.0005, 0.0]))
        np.testing.assert_allclose(out.to_numpy(), [12.0, 0.0])

    def test_negative_reanalysis_noise_is_clipped_and_a_day_without_hours_stays_missing(self):
        out = era5_hourly_mean_precip_mm_per_day(pd.Series([-1e-9, np.nan]))
        assert out.iloc[0] == 0.0
        assert np.isnan(out.iloc[1])


class TestServablePrecipIsEra5MmPerDay:
    def test_precip_is_replaced_by_era5_in_mm_per_day(self):
        out, _ = apply_feature_engineering(_frame(), [], contract="servable")
        np.testing.assert_allclose(out["precip"].to_numpy(), MM_PER_DAY)

    def test_a_day_short_of_hours_is_not_undercounted(self):
        # 12 of 24 hours present at 0.5 mm/h: the merge's sum says 6 mm, the day had 12.
        df = pd.DataFrame({TP_MEAN: [0.0005], TP_SUM: [0.006]})
        out, _ = apply_all_feature_engineering(df, contract="servable")
        np.testing.assert_allclose(out["precip"].to_numpy(), [12.0])
        np.testing.assert_allclose(out["precip_sum_3d"].to_numpy(), [12.0])

    def test_the_lag_follows_the_replaced_column(self):
        out, _ = apply_feature_engineering(_frame(), ["lags_1d"], contract="servable")
        np.testing.assert_allclose(out["precip_lag1d"].to_numpy()[1:], MM_PER_DAY[:-1])
        assert np.isnan(out["precip_lag1d"].iloc[0])

    def test_precip_memory_is_the_rolling_sum_of_the_same_column(self):
        out, _ = apply_all_feature_engineering(_frame(), contract="servable")
        np.testing.assert_allclose(
            out["precip_sum_3d"].to_numpy(), out["precip"].rolling(3, min_periods=1).sum().to_numpy()
        )

    def test_a_site_without_a_gauge_gets_precip_from_era5(self):
        out, feats = apply_all_feature_engineering(_frame().drop(columns="precip"), contract="servable")
        np.testing.assert_allclose(out["precip"].to_numpy(), MM_PER_DAY)
        assert "precip_lag1d" in feats

    def test_without_era5_the_gauge_does_not_leak_in(self):
        out, feats = apply_all_feature_engineering(_frame().drop(columns=[TP_MEAN]), contract="servable")
        assert "precip" not in out.columns
        assert "precip_lag1d" not in out.columns
        assert "precip_lag1d" not in feats
        assert "precip_sum_3d" not in feats

    def test_the_input_frame_is_not_modified(self):
        df = _frame()
        before = df.copy()
        apply_all_feature_engineering(df, contract="servable")
        pd.testing.assert_frame_equal(df, before)

    def test_hourly_is_not_defined(self):
        with pytest.raises(NotImplementedError):
            apply_feature_engineering(_frame(), ["lags_1d"], time_scale="hourly", contract="servable")


class TestSitePrecipIsUnchanged:
    def test_site_keeps_the_gauge_column(self):
        df = _frame()
        out, _ = apply_all_feature_engineering(df, contract="site")
        pd.testing.assert_series_equal(out["precip"], df["precip"])
        np.testing.assert_allclose(out["precip_lag1d"].to_numpy()[1:], df["precip"].to_numpy()[:-1])

    def test_site_without_a_gauge_does_not_borrow_era5(self):
        out, _ = apply_all_feature_engineering(_frame().drop(columns="precip"), contract="site")
        assert "precip" not in out.columns
