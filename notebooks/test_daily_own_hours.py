"""Daily statistics from each source's own hours in the Stage-4 merge.

The merge aggregates the day over the hours in which sap AND env observed. ERA5-Land has every hour and a
rain gauge has its own record, so both are re-aggregated over those, a statistic is missing rather than
partial when the day is incomplete, and the gauge's daily ``precip`` is a total in mm/day.
"""

import numpy as np
import pandas as pd
import pytest

from notebooks.merge_gap_filled_hourly_orginal import (
    add_swc_representations,
    complete_day_statistics,
    era5_daily_for_site,
    gauge_daily_totals,
    gauge_reporting_minutes,
    normalise_timestamps,
    overwrite_daily_columns,
    round_instants,
    swc_site_moments,
)

DAY = pd.Timestamp("2015-06-01", tz="UTC")
SWC1 = "volumetric_soil_water_layer_1"


def _hours(n_days=1):
    return pd.date_range(DAY, periods=24 * n_days, freq="h")


def _frame(index, **columns):
    return pd.DataFrame(columns, index=pd.DatetimeIndex(index, name="TIMESTAMP"))


def _lit(stamps):
    return np.isin(pd.DatetimeIndex(stamps).hour, range(6, 18))


def _one(stamp):
    return pd.DatetimeIndex([stamp], tz="UTC")


class TestTimestamps:
    def test_a_second_early_reading_joins_its_own_hour(self):
        frame = _frame(_one("2015-06-01 01:59:59"), precip=[1.0])
        assert frame.index.floor("h")[0].hour == 1  # the defect
        assert round_instants(frame).index.floor("h")[0].hour == 2

    def test_midnight_stored_as_235959_joins_the_next_day(self):
        frame = _frame(_one("2015-06-01 23:59:59"), precip=[1.0])
        assert round_instants(frame).index.floor("D")[0] == DAY + pd.Timedelta(days=1)

    def test_an_instant_duplicated_by_rounding_is_counted_once(self):
        index = pd.DatetimeIndex(["2015-06-01 01:59:59", "2015-06-01 02:00:00"], tz="UTC")
        out = normalise_timestamps(_frame(index, precip=[1.0, 2.0]))
        assert len(out) == 1
        assert out["precip"].iloc[0] == 1.0

    def test_the_input_is_not_modified_and_the_index_keeps_its_name(self):
        frame = _frame(_one("2015-06-01 01:59:59"), precip=[1.0])
        out = normalise_timestamps(frame)
        assert frame.index[0].second == 59
        assert out.index.name == "TIMESTAMP"


class TestCompleteDayStatistics:
    def test_a_complete_day(self):
        out = complete_day_statistics(_frame(_hours(), x=np.arange(1.0, 25.0)), ["x"])
        np.testing.assert_allclose(out.loc[DAY, ["x", "x_min", "x_max", "x_sum"]].to_numpy(float), [12.5, 1, 24, 300])

    def test_a_day_short_of_one_hour_is_missing_not_short(self):
        out = complete_day_statistics(_frame(_hours()[:-1], x=np.arange(1.0, 24.0)), ["x"])
        assert out.loc[DAY].isna().all()  # a plain daily sum reads 276

    def test_a_missing_value_counts_as_a_missing_hour(self):
        values = np.arange(1.0, 25.0)
        values[5] = np.nan
        out = complete_day_statistics(_frame(_hours(), x=values), ["x"])
        assert np.isnan(out.loc[DAY, "x_sum"])

    def test_columns_are_judged_separately(self):
        y = np.ones(24)
        y[0] = np.nan
        out = complete_day_statistics(_frame(_hours(), x=np.ones(24), y=y), ["x", "y"])
        assert out.loc[DAY, "x_sum"] == 24.0
        assert np.isnan(out.loc[DAY, "y_sum"])

    def test_the_statistics_of_a_column_sit_together(self):
        out = complete_day_statistics(_frame(_hours(), x=np.ones(24), y=np.ones(24)), ["x", "y"])
        assert list(out.columns) == ["x", "x_min", "x_max", "x_sum", "y", "y_min", "y_max", "y_sum"]

    def test_keep_hours_needs_only_the_retained_hours(self):
        index = _hours()
        values = np.where(_lit(index), 10.0, 1.0)
        night_gap = _frame(index, x=values).drop(index[:3])
        out = complete_day_statistics(night_gap, ["x"], keep_hours=_lit)
        np.testing.assert_allclose(out.loc[DAY, ["x", "x_sum"]].to_numpy(float), [10.0, 120.0])
        day_gap = _frame(index, x=values).drop(index[12])
        assert np.isnan(complete_day_statistics(day_gap, ["x"], keep_hours=_lit).loc[DAY, "x_sum"])

    def test_duplicate_hours_raise(self):
        index = _hours().append(_hours()[:1])
        with pytest.raises(ValueError):
            complete_day_statistics(_frame(index, x=np.ones(25)), ["x"])

    def test_an_empty_frame_gives_the_columns_and_no_rows(self):
        out = complete_day_statistics(_frame(_hours()[:0], x=[]), ["x"])
        assert out.empty
        assert list(out.columns) == ["x", "x_min", "x_max", "x_sum"]


class TestSwcRepresentations:
    def test_the_legacy_formulas_and_column_order(self):
        raw = pd.Series([0.2, 0.3, 0.4, 0.5])
        frame = pd.DataFrame({SWC1: raw})
        out = add_swc_representations(frame, swc_site_moments(frame), "S")
        np.testing.assert_allclose(out[SWC1 + "_raw"], raw)
        np.testing.assert_allclose(out[SWC1], raw / raw.std())
        np.testing.assert_allclose(out[SWC1 + "_zscore"], (raw - raw.mean()) / raw.std())
        assert list(out.columns) == [SWC1, SWC1 + "_raw", SWC1 + "_zscore"]
        np.testing.assert_allclose(frame[SWC1], raw)  # input not modified

    def test_near_zero_sigma_writes_zero(self):
        frame = pd.DataFrame({SWC1: [0.3, 0.3]})
        out = add_swc_representations(frame, swc_site_moments(frame), "S", warn=False)
        assert (out[SWC1] == 0.0).all()
        assert (out[SWC1 + "_zscore"] == 0.0).all()

    def test_the_moments_come_from_the_caller(self):
        frame = pd.DataFrame({SWC1: [0.3, 0.4]})
        out = add_swc_representations(frame, {SWC1: (0.2, 0.1)}, "S")
        np.testing.assert_allclose(out[SWC1 + "_zscore"], [1.0, 2.0])


class TestEra5DailyForSite:
    @staticmethod
    def _site(n_days=2):
        index = _hours(n_days)
        return pd.DataFrame(
            {
                "site_id": 7,
                "site_name": "S",
                "latitude": 50.0,
                "longitude": 8.0,
                "TIMESTAMP": index,
                "total_precipitation_hourly": 1e-4,
                "soil_temperature_level_1": 280.0 + index.hour.to_numpy(float),  # first 280, mean 291.5
                SWC1: 0.3 + 0.01 * (index.hour.to_numpy() % 2),  # mean 0.305
            }
        )

    def test_a_total_covers_the_whole_utc_day(self):
        out = era5_daily_for_site(self._site(), {}, "S")
        np.testing.assert_allclose(out["total_precipitation_hourly_sum"].to_numpy(), [0.0024, 0.0024])

    def test_soil_temperature_is_the_daily_mean_not_the_first_hour(self):
        out = era5_daily_for_site(self._site(), {}, "S")
        np.testing.assert_allclose(out["soil_temperature_level_1"].to_numpy(), [291.5, 291.5])
        np.testing.assert_allclose(out["soil_temperature_level_1_min"].to_numpy(), [280.0, 280.0])

    def test_identifier_columns_are_not_aggregated(self):
        out = era5_daily_for_site(self._site(), {}, "S")
        assert not {"site_id", "site_id_sum", "latitude", "longitude_max"} & set(out.columns)

    def test_soil_moisture_uses_the_moments_it_is_given(self):
        out = era5_daily_for_site(self._site(), {SWC1: (0.3, 0.01)}, "S")
        np.testing.assert_allclose(out[SWC1 + "_raw"].to_numpy(), [0.305, 0.305])
        np.testing.assert_allclose(out[SWC1].to_numpy(), [30.5, 30.5])
        np.testing.assert_allclose(out[SWC1 + "_zscore"].to_numpy(), [0.5, 0.5])

    def test_a_partial_first_day_of_the_extract_is_missing(self):
        out = era5_daily_for_site(self._site().iloc[10:], {}, "S")
        assert np.isnan(out["total_precipitation_hourly_sum"].iloc[0])
        assert out["total_precipitation_hourly_sum"].iloc[1] == pytest.approx(0.0024)


class TestGaugeDailyTotals:
    @staticmethod
    def _hourly_gauge(n_days=2, value=0.5):
        index = _hours(n_days)
        return _frame(index, precip=np.full(len(index), value), ta=np.ones(len(index)))

    def test_a_complete_day_is_its_total_in_mm_per_day(self):
        out = gauge_daily_totals(self._hourly_gauge(), "precip")
        np.testing.assert_allclose(out["precip"].to_numpy(), [12.0, 12.0])
        np.testing.assert_allclose(out["precip_sum"].to_numpy(), [12.0, 12.0])
        np.testing.assert_allclose(out["precip_coverage"].to_numpy(), [1.0, 1.0])

    def test_a_day_with_a_missing_hour_is_missing(self):
        env = self._hourly_gauge()
        env.iloc[5, 0] = np.nan
        out = gauge_daily_totals(env, "precip")
        assert np.isnan(out["precip"].iloc[0])
        assert out["precip"].iloc[1] == 12.0
        assert out["precip_coverage"].iloc[0] == pytest.approx(23 / 24)

    def test_a_day_without_readings_is_missing_not_dry(self):
        env = self._hourly_gauge(n_days=3)
        env.iloc[24:48, 0] = np.nan
        out = gauge_daily_totals(env, "precip")
        assert np.isnan(out["precip"].iloc[1])
        assert out["precip_coverage"].iloc[1] == 0.0

    def test_a_half_hourly_gauge(self):
        index = pd.date_range(DAY, periods=48, freq="30min")
        out = gauge_daily_totals(_frame(index, precip=np.full(48, 0.25)), "precip")
        assert out["precip"].iloc[0] == pytest.approx(12.0)
        assert out["precip_max"].iloc[0] == pytest.approx(0.5)  # the largest hourly total

    def test_a_daily_gauge_is_complete_with_its_one_reading(self):
        index = pd.DatetimeIndex(["2015-06-01 06:00", "2015-06-02 06:00", "2015-06-03 06:00"], tz="UTC")
        out = gauge_daily_totals(_frame(index, precip=[3.0, 0.0, 7.0]), "precip")
        np.testing.assert_allclose(out["precip"].to_numpy(), [3.0, 0.0, 7.0])

    def test_a_twelve_hourly_gauge_needs_both_readings(self):
        index = pd.date_range("2015-06-01 03:00", periods=6, freq="12h", tz="UTC")
        out = gauge_daily_totals(_frame(index, precip=[1.0, 2.0, 3.0, np.nan, 5.0, 6.0]), "precip")
        np.testing.assert_allclose(out["precip"].to_numpy()[[0, 2]], [3.0, 11.0])
        assert np.isnan(out["precip"].iloc[1])

    def test_keep_hours_scopes_an_hourly_gauge(self):
        env = self._hourly_gauge(n_days=1)
        env.iloc[2, 0] = np.nan  # a night hour
        out = gauge_daily_totals(env, "precip", keep_hours=_lit)
        assert out["precip"].iloc[0] == pytest.approx(6.0)

    def test_the_reporting_interval(self):
        index = pd.date_range(DAY, periods=10, freq="10min")
        assert gauge_reporting_minutes(pd.Series(1.0, index=index)) == 10.0
        assert np.isnan(gauge_reporting_minutes(pd.Series([1.0], index=index[:1])))

    def test_no_readings_gives_the_columns_and_no_rows(self):
        env = self._hourly_gauge()
        env["precip"] = np.nan
        out = gauge_daily_totals(env, "precip")
        assert out.empty
        assert "precip_coverage" in out.columns


class TestOverwriteDailyColumns:
    def test_values_follow_the_day_and_every_tree_gets_them(self):
        days = pd.date_range(DAY, periods=2, freq="D")
        df_daily = pd.DataFrame(
            {"TIMESTAMP": days.repeat(2), "pl_code": ["A", "B"] * 2, "x": 0.0, "sap_velocity": [1.0, 2.0, 3.0, 4.0]}
        )
        daily = pd.DataFrame({"x": [5.0], "x_new": [6.0]}, index=days[:1])
        out = overwrite_daily_columns(df_daily, daily)
        np.testing.assert_allclose(out["x"].to_numpy(), [5.0, 5.0, np.nan, np.nan])
        np.testing.assert_allclose(out["sap_velocity"].to_numpy(), [1.0, 2.0, 3.0, 4.0])
        assert list(out.columns) == ["TIMESTAMP", "pl_code", "x", "sap_velocity", "x_new"]
        assert (df_daily["x"] == 0.0).all()  # input not modified
