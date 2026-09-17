"""Measuring a clock from a site's own data (plan WP-A4; failure modes F2, F10).

Every series here is built in TRUE time and then stamped late: the reading at label ``t`` was taken at
``t - offset``, which is the convention the registry and the gate use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.clock_offsets.detect import (
    Estimate,
    clear_day_lag,
    correlation_lag,
    edge_shift,
    pair_lag,
    reference_lag,
    toa_for,
)

LAT, LON = 45.0, 5.0
CLEARNESS = 0.75
REGIMES = ("clear", "random", "noon", "afternoon", "fog")


def grid(days=200, start="2015-03-01", step="30min"):
    per_day = 48 if step == "30min" else 24
    return pd.date_range(pd.Timestamp(start, tz="UTC"), periods=int(days * per_day), freq=step)


def cloud(stamps, kind, seed=7):
    """A cloud factor on TRUE time. Every regime leaves one day in four untouched, so clear days exist."""
    if kind == "clear":
        return np.ones(len(stamps))
    hour = np.asarray(stamps.hour + stamps.minute / 60.0, dtype=float)
    day = np.asarray((stamps.normalize() - stamps.normalize()[0]).days, dtype=int)
    cloudy = (day % 4) != 0
    if kind == "random":
        return np.where(cloudy, np.random.default_rng(seed).uniform(0.2, 1.0, len(stamps)), 1.0)
    if kind == "noon":
        return np.where(cloudy & (np.abs(hour - 12.0) < 2.0), 0.15, 1.0)
    if kind == "afternoon":
        return np.where(cloudy & (hour > 13.0), 0.1, 1.0)
    if kind == "fog":
        return np.where(cloudy & (hour < 9.0), 0.1, 1.0)
    raise ValueError(kind)


def late_by(offset, kind="clear", lat=LAT, lon=LON, clearness=CLEARNESS, **kwargs):
    """sw_in stamped ``offset`` hours late: the value at label t was taken at t - offset."""
    index = grid(**kwargs)
    true_time = index - pd.Timedelta(hours=offset)
    values = clearness * toa_for(true_time, lat, lon).to_numpy() * cloud(true_time, kind)
    return pd.Series(values, index=index)


class TestClearDayLag:
    @pytest.mark.parametrize("offset", [0.0, 3.0, -2.5, 5.0])
    def test_a_clear_record_is_measured_within_a_quarter_hour(self, offset):
        found = clear_day_lag(late_by(offset), LAT, LON, reference_hours=offset)
        assert found, found.reason
        assert abs(found.hours - offset) <= 0.25

    @pytest.mark.parametrize("kind", REGIMES)
    def test_cloud_cannot_move_it(self, kind):
        found = clear_day_lag(late_by(3.0, kind), LAT, LON, reference_hours=3.0)
        assert found, found.reason
        assert abs(found.hours - 3.0) <= 0.25, kind

    def test_the_all_day_correlation_is_the_one_afternoon_storms_bias(self):
        # Why the gate asks for clear days: the same record, both ways.
        storms = late_by(3.0, "afternoon")
        all_day = correlation_lag(storms.to_numpy(), toa_for(storms.index, LAT, LON).to_numpy())
        clear = clear_day_lag(storms, LAT, LON, reference_hours=3.0)
        assert abs(all_day.hours - 3.0) > 0.25
        assert abs(clear.hours - 3.0) <= 0.25

    def test_a_dead_sensor_gives_a_reason_not_a_number(self):
        found = clear_day_lag(late_by(3.0, clearness=0.02), LAT, LON)
        assert not found
        assert "dead" in found.reason

    def test_a_record_too_short_to_have_clear_days_gives_a_reason(self):
        found = clear_day_lag(late_by(3.0, days=20), LAT, LON)
        assert not found
        assert "clear days" in found.reason

    def test_a_polar_night_gives_a_reason(self):
        dark = late_by(3.0, lat=80.0, lon=0.0, days=60, start="2015-11-20")
        found = clear_day_lag(dark, 80.0, 0.0)
        assert not found
        assert "polar night" in found.reason


class TestEdgeShift:
    @pytest.mark.parametrize("offset", [0.0, 3.0, -2.5])
    def test_the_daylight_window_finds_the_offset(self, offset):
        found = edge_shift(late_by(offset), LAT, LON)
        assert found, found.reason
        assert abs(found.hours - offset) <= 0.25

    @pytest.mark.parametrize("kind", REGIMES)
    def test_cloud_that_does_not_reach_the_threshold_cannot_move_it(self, kind):
        found = edge_shift(late_by(3.0, kind), LAT, LON)
        assert found, found.reason
        assert abs(found.hours - 3.0) <= 0.25, kind

    def test_a_polar_day_has_no_edges_to_read(self):
        midnight_sun = late_by(3.0, lat=69.5, lon=25.0, days=45, start="2015-06-01")
        found = edge_shift(midnight_sun, 69.5, 25.0)
        assert not found
        assert "daylight window" in found.reason

    def test_a_short_record_gives_a_reason_with_the_days_it_had(self):
        found = edge_shift(late_by(3.0, days=10), LAT, LON)
        assert not found
        assert found.sample <= 10


class TestFinerGrids:
    def test_a_quarter_hour_record_is_resampled_and_still_measured(self):
        # SAPFLUXNET ships 15-minute records as well as half-hourly ones; the estimators put everything on
        # one grid first, so the answer must not depend on which the site used.
        index = pd.date_range(pd.Timestamp("2015-03-01", tz="UTC"), periods=200 * 96, freq="15min")
        values = CLEARNESS * toa_for(index - pd.Timedelta(hours=3.0), LAT, LON).to_numpy()
        found = clear_day_lag(pd.Series(values, index=index), LAT, LON, reference_hours=3.0)
        assert found, found.reason
        assert abs(found.hours - 3.0) <= 0.25


class TestPairLag:
    def test_a_shared_clock_measures_the_physiological_delay_alone(self):
        sw_in = late_by(0.0)
        sap = sw_in.shift(3)  # sap follows radiation by 1.5 h
        found = pair_lag(sap, sw_in)
        assert abs(found.hours - 1.5) <= 0.25

    def test_a_sap_clock_ahead_of_the_env_clock_adds_to_it(self):
        sw_in = late_by(0.0)
        sap = pd.Series(sw_in.shift(3).to_numpy(), index=sw_in.index + pd.Timedelta(hours=5))
        found = pair_lag(sap, sw_in)
        assert abs(found.hours - 6.5) <= 0.25


class TestReferenceLag:
    def test_an_hourly_reference_finds_the_offset(self):
        sw_in = late_by(3.0)
        hourly = grid(step="h")
        era5 = pd.Series(CLEARNESS * toa_for(hourly, LAT, LON).to_numpy(), index=hourly)
        found = reference_lag(sw_in, era5)
        assert abs(found.hours - 3.0) <= 0.25


class TestCorrelationLag:
    def test_it_resolves_a_quarter_hour_on_a_half_hour_grid(self):
        values = late_by(0.25)
        found = correlation_lag(values.to_numpy(), toa_for(values.index, LAT, LON).to_numpy())
        assert abs(found.hours - 0.25) <= 0.1

    def test_series_of_different_lengths_raise(self):
        with pytest.raises(ValueError):
            correlation_lag(np.zeros(10), np.zeros(11))

    def test_too_few_pairs_gives_a_reason(self):
        values = late_by(3.0, days=3)
        found = correlation_lag(values.to_numpy(), toa_for(values.index, LAT, LON).to_numpy())
        assert not found
        assert "usable pairs" in found.reason

    def test_a_flat_series_has_no_peak_to_find(self):
        index = grid(days=30)
        found = correlation_lag(np.ones(len(index)), toa_for(index, LAT, LON).to_numpy())
        assert not found

    def test_an_estimate_without_a_number_is_falsy(self):
        assert not Estimate(reason="nothing to measure")
        assert Estimate(hours=0.0)
