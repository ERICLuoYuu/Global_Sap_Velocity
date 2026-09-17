"""Daily ext_rad is the served geometry (plan WP-B2, acceptance criterion AC8)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.env_daily.support import env_daily_statistics, ext_rad_daily_statistics, hourly_toa
from src.env_derive.ext_rad import extraterrestrial_radiation
from src.serving.era5_daily import EXT_RAD_STEP_MINUTES, ext_rad_daily

DAYS = pd.DatetimeIndex(["2015-01-01", "2015-06-21", "2015-09-22", "2015-12-21"], tz="UTC")
KT = 0.7


def lit(stamps):
    return np.isin(pd.DatetimeIndex(stamps).hour, range(6, 18))


def clear_sky(days, lat, lon):
    """Hourly sw_in at a constant clearness over every hour of ``days``, as aggregate_env_to_hourly writes it."""
    toa, _dark = hourly_toa(days, lat, lon)
    # Day by day: the days need not be contiguous, and the TOA rows are theirs.
    index = days.repeat(24) + pd.to_timedelta(np.tile(np.arange(24), len(days)), unit="h")
    return pd.DataFrame({"sw_in": KT * toa.ravel()}, index=index.rename("TIMESTAMP"))


@pytest.mark.parametrize("lat, lon", [(45.0, 0.0), (-51.6, -72.3), (69.5, 25.0), (0.5, 120.0)])
def test_the_daily_mean_is_the_served_value(lat, lon):
    got = ext_rad_daily_statistics(DAYS, lat, lon)
    served = ext_rad_daily(DAYS, lat, lon)
    assert np.max(np.abs(got["ext_rad"].to_numpy() - served.to_numpy())) <= 1e-9


def test_min_max_and_sum_come_from_the_same_instants():
    got = ext_rad_daily_statistics(DAYS[:1], 45.0, 0.0).iloc[0]
    instants = DAYS[0] + pd.to_timedelta(np.arange(0, 24 * 60, EXT_RAD_STEP_MINUTES), unit="min")
    values = extraterrestrial_radiation(instants, 45.0, 0.0).to_numpy()
    assert got["ext_rad_min"] == pytest.approx(values.min())
    assert got["ext_rad_max"] == pytest.approx(values.max())
    assert got["ext_rad_sum"] == pytest.approx(24.0 * values.mean())


def test_repeated_days_and_times_of_day_collapse_to_one_row_per_utc_day():
    stamps = pd.DatetimeIndex(["2015-06-21 00:00", "2015-06-21 00:00", "2015-06-22 13:00"], tz="UTC")
    got = ext_rad_daily_statistics(stamps, 45.0, 0.0)
    assert list(got.index) == list(pd.DatetimeIndex(["2015-06-21", "2015-06-22"], tz="UTC"))


def test_naive_days_are_utc():
    naive = ext_rad_daily_statistics(DAYS.tz_localize(None), 45.0, 0.0)
    pd.testing.assert_frame_equal(naive, ext_rad_daily_statistics(DAYS, 45.0, 0.0))


def test_the_columns_are_the_four_daily_statistics():
    got = ext_rad_daily_statistics(DAYS, 45.0, 0.0)
    assert list(got.columns) == ["ext_rad", "ext_rad_min", "ext_rad_max", "ext_rad_sum"]


def test_no_days_give_an_empty_frame():
    got = ext_rad_daily_statistics(pd.DatetimeIndex([], tz="UTC"), 45.0, 0.0)
    assert len(got) == 0
    assert list(got.columns) == ["ext_rad", "ext_rad_min", "ext_rad_max", "ext_rad_sum"]


def test_full_day_clearness_against_the_scaled_sw_in_is_within_half_a_percent():
    # The served mean samples 48 instants, the sw_in rule four per hour: two Riemann sums of one curve.
    days = pd.DatetimeIndex(["2015-03-21", "2015-06-21"], tz="UTC")
    sw_in = env_daily_statistics(clear_sky(days, 45.0, 0.0), 45.0, 0.0)["sw_in"]
    ratio = (sw_in / ext_rad_daily_statistics(days, 45.0, 0.0)["ext_rad"]).to_numpy()
    np.testing.assert_allclose(ratio, KT, rtol=5e-3)


class TestDaytimeOnly:
    """``--daytime-only`` scopes every env statistic to the kept hours; ext_rad follows, as the sw_in rule does."""

    def test_the_mean_is_the_kept_hours_toa_over_the_hours_the_day_expects(self):
        got = ext_rad_daily_statistics(DAYS, 45.0, 0.0, keep_hours=lit)
        toa, _dark = hourly_toa(DAYS, 45.0, 0.0)
        kept = toa[:, 6:18]
        np.testing.assert_allclose(got["ext_rad"].to_numpy(), kept.mean(axis=1), rtol=0, atol=1e-9)
        np.testing.assert_allclose(got["ext_rad_min"].to_numpy(), kept.min(axis=1), rtol=0, atol=1e-9)
        np.testing.assert_allclose(got["ext_rad_max"].to_numpy(), kept.max(axis=1), rtol=0, atol=1e-9)
        np.testing.assert_allclose(got["ext_rad_sum"].to_numpy(), 12.0 * kept.mean(axis=1), rtol=0, atol=1e-9)

    @pytest.mark.parametrize("lat, lon", [(45.0, 0.0), (-33.9, 151.2), (69.5, 25.0)])
    def test_clearness_against_the_scaled_sw_in_is_exact(self, lat, lon):
        sw_in = env_daily_statistics(clear_sky(DAYS, lat, lon), lat, lon, keep_hours=lit)["sw_in"]
        ext_rad = ext_rad_daily_statistics(DAYS, lat, lon, keep_hours=lit)["ext_rad"]
        lit_days = ext_rad.to_numpy() > 0
        np.testing.assert_allclose((sw_in / ext_rad).to_numpy()[lit_days], KT, rtol=1e-12)

    def test_a_single_kept_hour_is_that_hour_over_a_day_that_expects_one(self):
        # The narrowest daylight scope there is. Mean and sum must still be the same statement.
        day = DAYS[[1]]  # the solstice
        noon_only = lambda stamps: pd.DatetimeIndex(stamps).hour == 12  # noqa: E731 - one line, read here
        stats = ext_rad_daily_statistics(day, 45.0, 0.0, keep_hours=noon_only)
        toa, _dark = hourly_toa(day, 45.0, 0.0)
        assert stats["ext_rad"].iloc[0] == pytest.approx(toa[0][12])
        assert stats["ext_rad_sum"].iloc[0] == pytest.approx(stats["ext_rad"].iloc[0])
        assert stats["ext_rad_min"].iloc[0] == pytest.approx(stats["ext_rad_max"].iloc[0])

    def test_a_day_without_a_kept_hour_is_missing(self):
        got = ext_rad_daily_statistics(DAYS, 45.0, 0.0, keep_hours=lambda stamps: np.zeros(len(stamps), dtype=bool))
        assert got.isna().all().all()

    def test_the_hours_are_asked_of_keep_hours_at_the_top_of_each_hour(self):
        asked = []

        def keep(stamps):
            asked.append(pd.DatetimeIndex(stamps))
            return lit(stamps)

        ext_rad_daily_statistics(DAYS[:1], 45.0, 0.0, keep_hours=keep)
        assert len(asked) == 1
        assert list(asked[0]) == list(pd.date_range(DAYS[0], periods=24, freq="h"))
