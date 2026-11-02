from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.env_derive.humidity import saturation_vapour_pressure
from src.env_derive.ppfd import PPFD_PER_SW_IN
from src.serving import era5_daily as ed

CELLS = {layer: SimpleNamespace(mean=0.25, sigma=0.05) for layer in (1, 2, 3, 4)}


def _hourly(days=2, start="2010-06-01"):
    index = pd.date_range(start, periods=24 * days, freq="h")
    frame = pd.DataFrame({band: np.ones(len(index)) for band in ed.HOURLY_BANDS}, index=index)
    frame[ed.TEMPERATURE] = 293.15
    frame[ed.DEWPOINT] = 283.15
    frame[ed.WIND_U] = 3.0
    frame[ed.WIND_V] = 4.0
    frame[ed.SHORTWAVE] = 3600.0 * 200.0
    frame[ed.PRECIPITATION] = 1e-4
    frame[ed.POTENTIAL_EVAPORATION] = -2e-4
    frame[ed.SURFACE_PRESSURE] = 95000.0
    for column in ed.SOIL_TEMPERATURE:
        frame[column] = 280.0
    for column in ed.SOIL_WATER:
        frame[column] = 0.3
    return frame


def test_daily_means_carry_training_names_and_units():
    daily = ed.daily_base_columns(_hourly(), CELLS)
    row = daily.iloc[0]
    assert row["ta"] == pytest.approx(20.0)
    assert row["ws"] == pytest.approx(5.0)
    assert row["sw_in"] == pytest.approx(200.0)
    assert row["ppfd_in"] == pytest.approx(200.0 * PPFD_PER_SW_IN)
    assert row["surface_pressure"] == pytest.approx(95000.0)
    assert row["soil_temperature_level_3"] == pytest.approx(280.0)
    assert row["volumetric_soil_water_layer_2_raw"] == pytest.approx(0.3)
    assert row["volumetric_soil_water_layer_2_zscore"] == pytest.approx(1.0)
    assert row["total_precipitation_hourly"] == pytest.approx(1e-4)
    assert row["total_precipitation_hourly_sum"] == pytest.approx(24e-4)
    assert row["potential_evaporation_hourly_sum"] == pytest.approx(-48e-4)
    assert row["prcip/PET"] == pytest.approx(0.5)
    assert str(daily.index.tz) == "UTC"
    assert "potential_evaporation_hourly" not in daily.columns


def test_vpd_is_hourly_then_averaged_not_from_daily_means():
    hourly = _hourly(days=1)
    hourly[ed.TEMPERATURE] = np.where(np.arange(24) % 2 == 0, 283.15, 303.15)
    hourly[ed.DEWPOINT] = 278.15
    daily = ed.daily_base_columns(hourly, CELLS).iloc[0]
    es_dew = saturation_vapour_pressure(pd.Series([5.0])).iloc[0]
    hourly_vpd = (saturation_vapour_pressure(pd.Series([10.0, 30.0])) - es_dew).mean()
    from_means = saturation_vapour_pressure(pd.Series([20.0])).iloc[0] - es_dew
    assert daily["vpd"] == pytest.approx(hourly_vpd)
    assert daily["vpd"] > from_means
    assert daily["ta_min"] == pytest.approx(10.0)
    assert daily["ta_max"] == pytest.approx(30.0)
    assert daily["vpd_max"] > daily["vpd_min"]


def test_supersaturated_hour_reads_rh_100_and_no_deficit():
    hourly = _hourly(days=1)
    hourly[ed.DEWPOINT] = hourly[ed.TEMPERATURE] + 1.0
    daily = ed.daily_base_columns(hourly, CELLS).iloc[0]
    assert daily["rh"] == pytest.approx(100.0)
    assert daily["vpd"] == pytest.approx(0.0)


def test_day_short_of_an_hour_is_missing_and_other_days_are_not():
    hourly = _hourly(days=2).drop(pd.Timestamp("2010-06-02 05:00"))
    daily = ed.daily_base_columns(hourly, CELLS)
    assert daily.iloc[0].notna().all()
    assert daily.iloc[1].isna().all()


def test_missing_value_in_one_band_blanks_only_its_columns():
    hourly = _hourly(days=1)
    hourly.iloc[3, hourly.columns.get_loc(ed.PRECIPITATION)] = np.nan
    daily = ed.daily_base_columns(hourly, CELLS).iloc[0]
    assert np.isnan(daily["total_precipitation_hourly_sum"])
    assert np.isnan(daily["prcip/PET"])
    assert daily["ta"] == pytest.approx(20.0)


@pytest.mark.parametrize(
    "mutate, error",
    [
        (lambda f: pd.concat([f, f.iloc[[0]]]), ValueError),
        (lambda f: f.set_axis(f.index + pd.Timedelta(minutes=30), axis=0), ValueError),
        (lambda f: f.drop(columns=ed.SHORTWAVE), KeyError),
    ],
    ids=["duplicate-hour", "off-the-hour", "missing-band"],
)
def test_malformed_hourly_input_is_refused(mutate, error):
    with pytest.raises(error):
        ed.daily_base_columns(mutate(_hourly()), CELLS)


def test_zero_potential_evaporation_uses_the_merge_floor():
    hourly = _hourly(days=1)
    hourly[ed.POTENTIAL_EVAPORATION] = 0.0
    assert ed.daily_base_columns(hourly, CELLS).iloc[0]["prcip/PET"] == pytest.approx(ed.PRECIP_PET_MAX)


def test_invalid_climatology_sigma_is_refused():
    cells = dict(CELLS)
    cells[3] = SimpleNamespace(mean=0.25, sigma=0.0)
    with pytest.raises(ValueError):
        ed.daily_base_columns(_hourly(), cells)


def test_day_length_polar_day_polar_night_and_equinox():
    days = pd.DatetimeIndex(["2010-06-21", "2010-12-21"], tz="UTC")
    assert ed.day_length_daily(days, 80.0, 0.0).tolist() == [24.0, 0.0]
    equinox = ed.day_length_daily(pd.DatetimeIndex(["2010-03-20"], tz="UTC"), 0.0, 0.0).iloc[0]
    assert 12.0 <= equinox <= 12.3


def _fao56_daily_ra_wm2(latitude, day_of_year):
    phi = np.radians(latitude)
    dr = 1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365)
    delta = 0.409 * np.sin(2 * np.pi * day_of_year / 365 - 1.39)
    ws = np.arccos(-np.tan(phi) * np.tan(delta))
    ra_mj = 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))
    return ra_mj * 1e6 / 86400


def test_ext_rad_daily_matches_the_fao56_daily_integral():
    days = pd.DatetimeIndex(["2010-06-21", "2010-01-15"], tz="UTC")
    got = ed.ext_rad_daily(days, 45.0, 7.0)
    for value, doy in zip(got, days.dayofyear):
        assert value == pytest.approx(_fao56_daily_ra_wm2(45.0, doy), rel=0.015)
    night = ed.ext_rad_daily(pd.DatetimeIndex(["2010-12-21"], tz="UTC"), 80.0, 0.0).iloc[0]
    assert night == pytest.approx(0.0, abs=1e-9)


def test_solar_timestamp_is_longitude_plus_equation_of_time():
    days = pd.DatetimeIndex(["2010-02-11", "2010-11-03"], tz="UTC")
    greenwich = (ed.solar_timestamp(days, 0.0) - days).total_seconds().to_numpy()
    east = (ed.solar_timestamp(days, 15.0) - days).total_seconds().to_numpy()
    assert (np.abs(greenwich) <= 17 * 60).all()
    np.testing.assert_allclose(east - greenwich, 3600.0, atol=1e-3)


def test_days_off_midnight_are_refused():
    with pytest.raises(ValueError):
        ed.ext_rad_daily(pd.DatetimeIndex(["2010-06-21 06:00"], tz="UTC"), 45.0, 7.0)
