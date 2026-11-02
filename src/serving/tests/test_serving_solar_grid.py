import numpy as np
import pandas as pd
import pytest

from src.serving.era5_daily import day_length_daily, ext_rad_daily
from src.serving.solar_grid import N_BANDS, band_centres, band_of, cell_solar

SPRING = pd.date_range("2015-03-01", "2015-06-30", freq="D", tz="UTC")
MINUTE = 1.0 / 60.0


def test_band_of_edges_and_centres():
    np.testing.assert_array_equal(band_of([-180.0, -150.05, -149.95, 0.0, 179.95]), [0, 0, 1, 6, 11])
    assert band_centres()[6] == 15.0 and len(band_centres()) == N_BANDS


HIGH_LATITUDE_DEG = 60.0
DAY_LENGTH_MAX_ERROR_H = 0.15


@pytest.mark.parametrize("latitude", [45.0, 67.0, -35.0])
def test_a_band_centre_stands_in_for_any_longitude_of_its_band(latitude):
    edge, centre = 29.95, 15.0
    exact = ext_rad_daily(SPRING, latitude, edge).to_numpy()
    banded = ext_rad_daily(SPRING, latitude, centre).to_numpy()
    assert np.abs(exact - banded).max() <= 0.1 * np.abs(np.diff(exact)).max()

    exact = day_length_daily(SPRING, latitude, edge).to_numpy()
    banded = day_length_daily(SPRING, latitude, centre).to_numpy()
    error = np.abs(exact - banded)
    assert error.max() <= DAY_LENGTH_MAX_ERROR_H
    if abs(latitude) < HIGH_LATITUDE_DEG:
        assert error.max() <= 0.1 * np.abs(np.diff(exact)).max() + MINUTE


def test_site_altitude_does_not_move_day_length():
    at_sea_level = day_length_daily(SPRING, 60.0, 15.0, 0.0).to_numpy()
    on_a_mountain = day_length_daily(SPRING, 60.0, 15.0, 5000.0).to_numpy()
    assert np.abs(on_a_mountain - at_sea_level).max() <= MINUTE


def test_cell_solar_reads_the_row_and_band_of_each_cell():
    table = np.arange(2 * 3 * N_BANDS, dtype=np.float32).reshape(2, 3, N_BANDS)
    solar = {"ext_rad": table, "day_length": table + 100}
    ext_rad, day_length = cell_solar(solar, [2, 0], [15.0, -179.95])
    np.testing.assert_array_equal(ext_rad[:, 0], table[:, 2, 6])
    np.testing.assert_array_equal(ext_rad[:, 1], table[:, 0, 0])
    np.testing.assert_array_equal(day_length, ext_rad + 100)
