"""Daily env statistics from env's own hours (plan WP-B1): the support rule, radiation scaling, night fill.

Failure modes F12 (dawn hour taken for dark), F13 (scaling), F14 (polar night and day). Radiation inputs are
built with a uniform clearness index, kt x the hour's TOA, so the value a correct estimator must return is
known exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.env_daily.support import (
    ENV_CONTINUOUS_COLS,
    MIN_ENV_HOURS,
    MIN_TOA_COVERAGE,
    env_daily_statistics,
    hourly_toa,
)
from src.env_derive.ext_rad import extraterrestrial_radiation

LAT, LON = 45.0, 0.0
POLAR_LAT, POLAR_LON = 69.5, 25.0
SUMMER = pd.Timestamp("2015-06-21", tz="UTC")
WINTER = pd.Timestamp("2015-12-21", tz="UTC")


def hours(day, n_days=1):
    return pd.date_range(day, periods=24 * n_days, freq="h", name="TIMESTAMP")


def toa_and_dark(day, lat=LAT, lon=LON):
    toa, dark = hourly_toa(pd.DatetimeIndex([day]), lat, lon)
    return toa[0], dark[0]


def clear_sky(day, lat=LAT, lon=LON, kt=0.6):
    """Hourly sw_in with a uniform clearness index: kt times the hour's TOA, zero in the dark."""
    toa, _dark = toa_and_dark(day, lat, lon)
    return pd.Series(kt * toa, index=hours(day), name="sw_in")


def env(**columns):
    first = next(iter(columns.values()))
    return pd.DataFrame(columns, index=first.index)


def row(stats, day):
    return stats.loc[pd.Timestamp(day)]


def morning_to_evening(stamps):
    return np.isin(pd.DatetimeIndex(stamps).hour, range(6, 18))


def sunrise_minute(day, lat, lon):
    minutes = pd.date_range(day, periods=24 * 60, freq="min")
    lit = extraterrestrial_radiation(minutes, lat, lon).to_numpy() > 0
    return int(np.flatnonzero(lit)[0])


class TestHourlyToa:
    @pytest.mark.parametrize(
        "lat, day",
        [(LAT, SUMMER), (LAT, WINTER), (-33.0, SUMMER), (POLAR_LAT, SUMMER), (POLAR_LAT, WINTER)],
    )
    def test_an_hour_is_dark_exactly_when_the_sun_is_down_across_its_whole_window(self, lat, day):
        _toa, dark = toa_and_dark(day, lat, POLAR_LON)
        for hour in range(24):
            start = day + pd.Timedelta(hours=hour, minutes=-30)
            minutes = pd.date_range(start, start + pd.Timedelta(minutes=89), freq="min")
            sun_down = (extraterrestrial_radiation(minutes, lat, POLAR_LON).to_numpy() <= 0).all()
            assert bool(dark[hour]) == bool(sun_down), (lat, day, hour)

    def test_the_hours_average_to_the_served_daily_value(self):
        from src.serving.era5_daily import ext_rad_daily

        days = pd.DatetimeIndex([SUMMER, WINTER])
        toa, _dark = hourly_toa(days, LAT, LON)
        np.testing.assert_allclose(toa.mean(axis=1), ext_rad_daily(days, LAT, LON).to_numpy(), rtol=0.01)

    def test_the_arrays_are_days_by_24_hours(self):
        toa, dark = hourly_toa(pd.DatetimeIndex([SUMMER, WINTER]), LAT, LON)
        assert toa.shape == dark.shape == (2, 24)
        assert dark.dtype == bool


class TestCompleteDay:
    def test_a_complete_continuous_day_is_its_plain_statistics(self):
        ta = pd.Series(np.linspace(10.0, 21.5, 24), index=hours(SUMMER))
        got = row(env_daily_statistics(env(ta=ta), LAT, LON), SUMMER)
        assert got["ta"] == pytest.approx(ta.mean())
        assert got["ta_min"] == pytest.approx(ta.min())
        assert got["ta_max"] == pytest.approx(ta.max())
        assert got["ta_sum"] == pytest.approx(ta.sum())
        assert got["ta_hours"] == 24

    def test_a_complete_radiation_day_is_its_plain_mean(self):
        sw = clear_sky(SUMMER) + np.random.default_rng(1).uniform(-3.0, 3.0, 24)
        got = row(env_daily_statistics(env(sw_in=sw), LAT, LON), SUMMER)
        assert got["sw_in"] == pytest.approx(sw.mean(), rel=1e-12, abs=1e-9)
        assert got["sw_in_sum"] == pytest.approx(sw.sum(), rel=1e-12, abs=1e-9)
        assert got["sw_in_toa_coverage"] == pytest.approx(1.0)
        assert got["sw_in_hours"] == 24


class TestNightFill:
    def test_night_na_gives_the_24_hour_mean_not_the_daylight_mean(self):
        sw = clear_sky(SUMMER)
        _toa, dark = toa_and_dark(SUMMER)
        shipped = sw.where(~dark)  # the archive ships NA at night
        truth = sw.sum() / 24.0
        got = row(env_daily_statistics(env(sw_in=shipped), LAT, LON), SUMMER)
        assert got["sw_in"] == pytest.approx(truth, rel=1e-9)
        assert got["sw_in_hours"] == 24
        assert shipped.mean() > 1.2 * truth  # the defect: a daylight mean read as a daily one

    def test_measured_night_values_are_kept_including_negatives(self):
        sw = clear_sky(SUMMER)
        _toa, dark = toa_and_dark(SUMMER)
        measured = sw.where(~dark, -2.0)
        got = row(env_daily_statistics(env(sw_in=measured), LAT, LON), SUMMER)
        assert got["sw_in"] == pytest.approx(measured.mean(), rel=1e-9)
        assert got["sw_in_min"] == -2.0

    def test_an_hour_whose_window_catches_sunrise_is_missing_not_dark(self):
        # Sunrise is placed at :50 of an hour by moving the site west, 4 minutes of clock per degree.
        base = sunrise_minute(SUMMER, LAT, 0.0)
        lon = -((50 - base % 60) % 60) / 4.0
        minute = sunrise_minute(SUMMER, LAT, lon)
        assert 48 <= minute % 60 <= 52
        hour = minute // 60
        _toa, dark = toa_and_dark(SUMMER, LAT, lon)
        assert dark[hour - 1] and not dark[hour]
        gappy = clear_sky(SUMMER, lon=lon)
        gappy.iloc[hour] = np.nan
        got = row(env_daily_statistics(env(sw_in=gappy), LAT, lon), SUMMER)
        assert got["sw_in_hours"] == 23


class TestScaling:
    def test_two_missing_daylight_hours_with_uniform_clearness_recover_the_truth(self):
        sw = clear_sky(SUMMER)
        gappy = sw.copy()
        gappy.iloc[[9, 16]] = np.nan
        got = row(env_daily_statistics(env(sw_in=gappy), LAT, LON), SUMMER)
        assert got["sw_in"] == pytest.approx(sw.sum() / 24.0, rel=0.01)
        assert MIN_TOA_COVERAGE <= got["sw_in_toa_coverage"] < 1.0

    def test_scaling_beats_averaging_the_hours_that_are_left(self):
        sw = clear_sky(SUMMER)
        _toa, dark = toa_and_dark(SUMMER)
        truth = sw.sum() / 24.0
        gappy = sw.copy()
        gappy.iloc[[8, 16]] = np.nan
        nightfill_mean = gappy.where(~dark, 0.0).mean()
        got = row(env_daily_statistics(env(sw_in=gappy), LAT, LON), SUMMER)["sw_in"]
        assert abs(got - truth) < abs(nightfill_mean - truth)

    def test_coverage_below_the_threshold_gives_nan(self):
        gappy = clear_sky(SUMMER)
        gappy.iloc[9:15] = np.nan
        got = row(env_daily_statistics(env(sw_in=gappy), LAT, LON), SUMMER)
        assert all(np.isnan(got[c]) for c in ("sw_in", "sw_in_min", "sw_in_max", "sw_in_sum"))
        assert got["sw_in_toa_coverage"] < MIN_TOA_COVERAGE

    def test_the_coverage_column_and_the_validity_decision_never_disagree(self):
        # The column a reader trusts and the rule the code applies are computed apart from each other.
        # Whatever the gap, a finite mean and a coverage at or above the threshold must be the same event.
        for width in range(1, 9):
            gappy = clear_sky(SUMMER)
            gappy.iloc[10 : 10 + width] = np.nan
            got = row(env_daily_statistics(env(sw_in=gappy), LAT, LON), SUMMER)
            assert np.isfinite(got["sw_in"]) == (got["sw_in_toa_coverage"] >= MIN_TOA_COVERAGE), width

    def test_a_day_whose_only_readings_are_dark_is_missing_not_zero(self):
        # Every lit hour absent, every dark hour present: the scaling divides by the energy of the hours
        # that were present, which here is nothing at all.
        sw = clear_sky(SUMMER)
        _toa, dark = toa_and_dark(SUMMER)
        sw[~dark] = np.nan
        got = row(env_daily_statistics(env(sw_in=sw), LAT, LON), SUMMER)
        assert got["sw_in_toa_coverage"] == 0.0
        assert all(np.isnan(got[c]) for c in ("sw_in", "sw_in_min", "sw_in_max", "sw_in_sum"))

    def test_a_site_at_the_date_line_still_cuts_its_days_on_utc(self):
        # Solar noon there falls near 00:00 UTC, so a day cut on the sun would split every one of these
        # readings across two rows. The daily frame is UTC, as ERA5 and the served product are.
        far_east = 179.9
        stats = env_daily_statistics(env(sw_in=clear_sky(SUMMER, LAT, far_east)), LAT, far_east)
        assert list(stats.index) == [SUMMER]
        assert np.isfinite(row(stats, SUMMER)["sw_in"])

    def test_min_and_max_need_twenty_filled_hours_even_when_the_mean_is_valid(self):
        sw = clear_sky(SUMMER)
        _toa, dark = toa_and_dark(SUMMER)
        lit = np.flatnonzero(~dark)
        gappy = sw.copy()
        gappy.iloc[[lit[0], lit[1], lit[-1], lit[-2], lit[-3]]] = np.nan  # dawn and dusk: little energy
        got = row(env_daily_statistics(env(sw_in=gappy), LAT, LON), SUMMER)
        assert got["sw_in_hours"] == MIN_ENV_HOURS - 1
        assert got["sw_in_toa_coverage"] >= MIN_TOA_COVERAGE
        assert np.isfinite(got["sw_in"])
        assert np.isnan(got["sw_in_min"]) and np.isnan(got["sw_in_max"])

    def test_ppfd_in_follows_the_radiation_rule(self):
        ppfd = 2.1 * clear_sky(SUMMER)
        _toa, dark = toa_and_dark(SUMMER)
        got = row(env_daily_statistics(env(ppfd_in=ppfd.where(~dark)), LAT, LON), SUMMER)
        assert got["ppfd_in"] == pytest.approx(ppfd.sum() / 24.0, rel=1e-9)
        assert got["ppfd_in_toa_coverage"] == pytest.approx(1.0)


class TestPolar:
    def test_a_polar_night_day_with_nothing_measured_is_zero(self):
        index = hours(WINTER)
        frame = env(sw_in=pd.Series(np.nan, index=index), ta=pd.Series(-20.0, index=index))
        got = row(env_daily_statistics(frame, POLAR_LAT, POLAR_LON), WINTER)
        assert got["sw_in"] == 0.0
        assert got["sw_in_toa_coverage"] == 1.0
        assert got["sw_in_hours"] == 24

    def test_a_polar_day_is_never_night_filled(self):
        _toa, dark = toa_and_dark(SUMMER, POLAR_LAT, POLAR_LON)
        assert not dark.any()
        sw = clear_sky(SUMMER, POLAR_LAT, POLAR_LON)
        gappy = sw.copy()
        gappy.iloc[0] = np.nan
        got = row(env_daily_statistics(env(sw_in=gappy), POLAR_LAT, POLAR_LON), SUMMER)
        assert got["sw_in_hours"] == 23
        assert got["sw_in"] == pytest.approx(sw.sum() / 24.0, rel=1e-9)


class TestContinuousRule:
    @pytest.mark.parametrize("present", [MIN_ENV_HOURS - 1, MIN_ENV_HOURS])
    def test_twenty_hours_is_the_threshold(self, present):
        gappy = pd.Series(np.arange(24, dtype=float), index=hours(SUMMER))
        gappy.iloc[present:] = np.nan
        got = row(env_daily_statistics(env(ta=gappy), LAT, LON), SUMMER)
        assert got["ta_hours"] == present
        if present < MIN_ENV_HOURS:
            assert all(np.isnan(got[c]) for c in ("ta", "ta_min", "ta_max", "ta_sum"))
        else:
            assert got["ta"] == pytest.approx(gappy.mean())
            assert got["ta_sum"] == pytest.approx(gappy.mean() * 24)
            assert got["ta_max"] == present - 1

    def test_every_continuous_column_follows_the_rule(self):
        index = hours(SUMMER)
        frame = pd.DataFrame({col: pd.Series(1.5, index=index) for col in ENV_CONTINUOUS_COLS})
        got = row(env_daily_statistics(frame, LAT, LON), SUMMER)
        for col in ENV_CONTINUOUS_COLS:
            assert got[col] == 1.5
            assert got[f"{col}_hours"] == 24

    def test_netrad_is_left_alone(self):
        index = hours(SUMMER)
        frame = env(ta=pd.Series(1.0, index=index), netrad=pd.Series(5.0, index=index))
        stats = env_daily_statistics(frame, LAT, LON)
        assert not [col for col in stats.columns if col.startswith("netrad")]


class TestKeepHours:
    @pytest.mark.parametrize("present, valid", [(12, True), (10, True), (9, False)])
    def test_expected_hours_are_the_kept_hours_and_the_threshold_scales(self, present, valid):
        gappy = pd.Series(np.arange(24, dtype=float), index=hours(SUMMER))
        kept = np.flatnonzero(morning_to_evening(gappy.index))
        gappy.iloc[kept[present:]] = np.nan
        got = row(env_daily_statistics(env(ta=gappy), LAT, LON, keep_hours=morning_to_evening), SUMMER)
        assert got["env_expected_hours"] == 12
        assert got["ta_hours"] == present
        assert bool(np.isfinite(got["ta"])) == valid
        if valid:
            assert got["ta"] == pytest.approx(gappy.iloc[kept].mean())
            assert got["ta_sum"] == pytest.approx(gappy.iloc[kept].mean() * 12)

    def test_radiation_is_scaled_over_the_kept_hours_only(self):
        sw = clear_sky(SUMMER)
        kept = np.flatnonzero(morning_to_evening(sw.index))
        gappy = sw.copy()
        gappy.iloc[12] = np.nan
        got = row(env_daily_statistics(env(sw_in=gappy), LAT, LON, keep_hours=morning_to_evening), SUMMER)
        assert got["sw_in"] == pytest.approx(sw.iloc[kept].mean(), rel=1e-9)


class TestShapeAndSafety:
    def test_the_columns_written_for_each_kind_of_variable(self):
        index = hours(SUMMER)
        stats = env_daily_statistics(env(ta=pd.Series(1.0, index=index), sw_in=clear_sky(SUMMER)), LAT, LON)
        assert set(stats.columns) == {
            "ta",
            "ta_min",
            "ta_max",
            "ta_sum",
            "ta_hours",
            "sw_in",
            "sw_in_min",
            "sw_in_max",
            "sw_in_sum",
            "sw_in_hours",
            "sw_in_toa_coverage",
            "env_expected_hours",
        }

    def test_one_row_per_utc_day_holding_an_env_hour(self):
        index = hours(SUMMER).append(hours(SUMMER + pd.Timedelta(days=2)))
        stats = env_daily_statistics(env(ta=pd.Series(1.0, index=index)), LAT, LON)
        assert list(stats.index) == [SUMMER, SUMMER + pd.Timedelta(days=2)]
        assert str(stats.index.tz) == "UTC"

    def test_a_naive_hourly_index_is_read_as_utc(self):
        sw = clear_sky(SUMMER)
        naive = env(sw_in=sw.set_axis(sw.index.tz_localize(None)))
        pd.testing.assert_frame_equal(
            env_daily_statistics(naive, LAT, LON), env_daily_statistics(env(sw_in=sw), LAT, LON)
        )

    def test_the_input_frame_is_not_modified(self):
        frame = env(sw_in=clear_sky(SUMMER).where(lambda s: s > 0))
        before = frame.copy(deep=True)
        env_daily_statistics(frame, LAT, LON)
        pd.testing.assert_frame_equal(frame, before)

    def test_a_duplicated_hour_raises(self):
        index = hours(SUMMER)
        frame = env(ta=pd.Series(1.0, index=index.append(index[:1])))
        with pytest.raises(ValueError):
            env_daily_statistics(frame, LAT, LON)

    def test_a_sub_hourly_stamp_raises(self):
        index = pd.DatetimeIndex([SUMMER, SUMMER + pd.Timedelta(minutes=30)], name="TIMESTAMP")
        with pytest.raises(ValueError):
            env_daily_statistics(env(ta=pd.Series(1.0, index=index)), LAT, LON)

    def test_an_empty_frame_gives_an_empty_result(self):
        empty = pd.DataFrame({"ta": pd.Series([], dtype=float)}, index=pd.DatetimeIndex([], tz="UTC"))
        assert len(env_daily_statistics(empty, LAT, LON)) == 0
