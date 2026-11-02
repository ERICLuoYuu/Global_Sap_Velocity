from __future__ import annotations

from datetime import date

import netCDF4
import numpy as np
import pandas as pd
import pytest
from src.static_features.products import cci
from src.static_features.products.cci import extract, fetch_inputs, window_value
from src.static_features.rasters import PixelHit
from src.static_features.results import NEIGHBOUR, NO_DATA, OK, ExtractionFailure, RunContext, validate

FILL = -32768
SHAPE = (18, 36)  # 10 degree pixels, so the site at (45, 5) is row 4, col 18
YEARS = (2009, 2010, 2011, 2012, 2013)


def _write(tmp_path, agb=None, agb_sd=None, years=YEARS, north_first=True):
    """agb, agb_sd: year -> {(row, col): raw}. Everything else is fill."""
    directory = tmp_path / "esa_cci_biomass_v7"
    directory.mkdir()
    path = directory / "ESACCI-BIOMASS-L4-AGB-MERGED-1000m-fv7.0.nc"
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("time", len(years))
        dataset.createDimension("lat", SHAPE[0])
        dataset.createDimension("lon", SHAPE[1])
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = "days since 1990-01-01"
        time[:] = [(date(year, 1, 1) - date(1990, 1, 1)).days for year in years]
        lat = dataset.createVariable("lat", "f8", ("lat",))
        lat[:] = np.arange(85.0, -90.0, -10.0) if north_first else np.arange(-85.0, 90.0, 10.0)
        dataset.createVariable("lon", "f8", ("lon",))[:] = np.arange(-175.0, 180.0, 10.0)
        for name, points in (("agb", agb or {}), ("agb_sd", agb_sd or {})):
            variable = dataset.createVariable(name, "i2", ("time", "lat", "lon"), fill_value=FILL)
            grid = np.full((len(years), *SHAPE), FILL, dtype="i2")
            for position, year in enumerate(years):
                for (row, col), raw in points.get(year, {}).items():
                    grid[position, row, col] = raw
            variable[:] = grid
    return path


def _context(tmp_path, lat=45.0, lon=5.0):
    sites = pd.DataFrame({"site_name": ["A"], "lat": [lat], "lon": [lon]})
    return RunContext(sites=sites, raw_root=tmp_path, output_root=tmp_path / "out", project="p")


def _own(year_values):
    return {year: {(4, 18): value} for year, value in year_values.items()}


def test_cci_window_mean_and_status():
    assert window_value(
        [PixelHit(10.0, OK, 1, 1), PixelHit(20.0, OK, 1, 1), PixelHit(np.nan, NO_DATA, None, None)]
    ) == (15.0, OK)
    assert window_value([PixelHit(10.0, OK, 1, 1), PixelHit(20.0, NEIGHBOUR, 1, 2)]) == (15.0, NEIGHBOUR)
    value, status = window_value([PixelHit(np.nan, NO_DATA, None, None)] * 3)
    assert np.isnan(value) and status == NO_DATA


def test_cci_takes_the_mean_over_2010_to_2012(tmp_path):
    _write(tmp_path, _own({2010: 100, 2011: 200, 2012: 600}), _own({2010: 10, 2011: 20, 2012: 30}))
    result = extract(_context(tmp_path))

    assert result.values.loc["A", "agb_mg_ha"] == pytest.approx(300.0)
    assert result.values.loc["A", "agb_sd"] == pytest.approx(20.0)
    assert result.status.loc["A", "agb_mg_ha"] == OK
    validate(result, ["A"])


def test_cci_zero_is_a_value(tmp_path):
    _write(tmp_path, _own({2010: 0, 2011: 0, 2012: 0}), _own({2010: 0, 2011: 0, 2012: 0}))
    result = extract(_context(tmp_path))

    assert result.values.loc["A", "agb_mg_ha"] == 0.0
    assert result.status.loc["A", "agb_mg_ha"] == OK


def test_cci_fill_takes_the_nearest_valid_pixel_within_three(tmp_path):
    _write(tmp_path, {2010: {(4, 21): 50}, 2011: {(4, 21): 50}, 2012: {(4, 21): 50}})
    result = extract(_context(tmp_path))

    assert result.values.loc["A", "agb_mg_ha"] == 50.0
    assert result.status.loc["A", "agb_mg_ha"] == NEIGHBOUR
    assert result.status.loc["A", "agb_sd"] == NO_DATA


def test_cci_beyond_three_pixels_is_no_data(tmp_path):
    _write(tmp_path, {2010: {(4, 22): 50}, 2011: {(4, 22): 50}, 2012: {(4, 22): 50}})
    result = extract(_context(tmp_path))

    assert np.isnan(result.values.loc["A", "agb_mg_ha"])
    assert result.status.loc["A", "agb_mg_ha"] == NO_DATA
    validate(result, ["A"])


def test_cci_year_outside_window_goes_to_by_year_only(tmp_path):
    _write(tmp_path, _own({2009: 900, 2010: 100, 2011: 100, 2012: 100, 2013: 900}))
    result = extract(_context(tmp_path))

    assert result.values.loc["A", "agb_mg_ha"] == 100.0
    table = result.by_year[result.by_year["column"] == "agb_mg_ha"].set_index("year")
    assert sorted(table.index) == list(YEARS)
    assert table.loc[2013, "value"] == 900.0
    assert table.loc[2009, "status"] == OK
    assert set(result.by_year["column"]) == {"agb_mg_ha", "agb_sd"}
    validate(result, ["A"])


def test_cci_refuses_file_without_the_window_years(tmp_path):
    _write(tmp_path, years=(2009, 2010, 2011))
    with pytest.raises(ExtractionFailure, match="2012"):
        extract(_context(tmp_path))


def test_cci_refuses_a_latitude_axis_it_does_not_expect(tmp_path):
    _write(tmp_path, north_first=False)
    with pytest.raises(ExtractionFailure, match="latitude"):
        extract(_context(tmp_path))


def test_fetch_inputs_pins_url_and_size(tmp_path, monkeypatch):
    requested = []
    monkeypatch.setattr(cci, "fetch", lambda item, directory: requested.append((item, directory)))
    fetch_inputs(_context(tmp_path))

    item, directory = requested[0]
    assert item.url.endswith("/v7.0/netcdf/ESACCI-BIOMASS-L4-AGB-MERGED-1000m-fv7.0.nc")
    assert item.size == 2983287549
    assert directory == tmp_path / "esa_cci_biomass_v7"
