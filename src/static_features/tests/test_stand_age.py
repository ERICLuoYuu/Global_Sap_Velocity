from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest
from src.static_features.products.stand_age import age_value, extract
from src.static_features.results import NO_DATA, OK, ExtractionFailure, RunContext, validate

LATS = np.array([10.5, 9.5, 8.5, 7.5])
LONS = np.array([0.5, 1.5, 2.5])
VARIABLES = ("ForestAge_TC000", "ForestAge_TC010", "ForestAge_TC020", "ForestAge_TC030")


def _write_file(tmp_path, grids=None, lon_first=False):
    directory = tmp_path / "stand_age"
    directory.mkdir(exist_ok=True)
    with h5py.File(directory / "20260101_BGIForestAgeMPIBGC1.0.0.nc", "w") as handle:
        handle["latitude"] = LATS
        handle["longitude"] = LONS
        for position, name in enumerate(VARIABLES):
            grid = np.full((4, 3), 100.0 + position) if grids is None else grids[position]
            handle[name] = grid.T if lon_first else grid


def _context(tmp_path, lat=9.4, lon=1.6):
    sites = pd.DataFrame({"site_name": ["A"], "lat": [lat], "lon": [lon]})
    return RunContext(sites=sites, raw_root=tmp_path, output_root=tmp_path / "out", project="p")


def test_age_value_boundaries():
    assert age_value(500.0) == 500.0
    assert age_value(1.5) == 1.5
    assert np.isnan(age_value(0.0)) and np.isnan(age_value(500.5)) and np.isnan(age_value(np.nan))
    assert np.isnan(age_value(-9999.0))


def test_stand_age_reads_each_threshold_at_the_nearest_node(tmp_path):
    grid = np.full((4, 3), 50.0)
    grid[1, 1] = 77.0  # latitude 9.5, longitude 1.5 is the node nearest (9.4, 1.6)
    _write_file(tmp_path, grids=[grid, grid + 1, grid + 2, grid + 3])
    result = extract(_context(tmp_path))

    assert result.values.loc["A", "stand_age_TC000"] == 77.0
    assert result.values.loc["A", "stand_age_TC030"] == 80.0
    assert result.status.loc["A", "stand_age_TC010"] == OK
    assert list(result.values.columns) == [
        "stand_age_TC000", "stand_age_TC010", "stand_age_TC020", "stand_age_TC030",
    ]  # fmt: skip
    validate(result, ["A"])


def test_stand_age_reads_a_longitude_first_file_too(tmp_path):
    grid = np.full((4, 3), 50.0)
    grid[1, 1] = 77.0
    _write_file(tmp_path, grids=[grid] * 4, lon_first=True)
    assert extract(_context(tmp_path)).values.loc["A", "stand_age_TC000"] == 77.0


def test_empty_node_is_no_data_with_no_search(tmp_path):
    # A filled value would displace the ground st_age in the merge, so a valid neighbour must not be used.
    grid = np.full((4, 3), 50.0)
    grid[1, 1] = np.nan
    _write_file(tmp_path, grids=[grid] * 4)
    result = extract(_context(tmp_path))

    assert np.isnan(result.values.loc["A", "stand_age_TC000"])
    assert result.status.loc["A", "stand_age_TC000"] == NO_DATA
    validate(result, ["A"])


def test_thresholds_are_empty_independently(tmp_path):
    full = np.full((4, 3), 50.0)
    holed = full.copy()
    holed[1, 1] = 0.0
    _write_file(tmp_path, grids=[full, holed, full, full])
    result = extract(_context(tmp_path))

    assert result.status.loc["A", "stand_age_TC000"] == OK
    assert result.status.loc["A", "stand_age_TC010"] == NO_DATA


def test_missing_file_stops_naming_the_directory(tmp_path):
    with pytest.raises(ExtractionFailure, match="stand_age"):
        extract(_context(tmp_path))
