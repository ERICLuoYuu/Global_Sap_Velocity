from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest
from src.static_features.neighbours import BORROWED_COLUMNS, KM_PER_DEGREE
from src.static_features.products.stand_age import NEIGHBOUR_COLUMN, age_value, extract
from src.static_features.results import NEIGHBOUR, NO_DATA, OK, ExtractionFailure, RunContext, validate

LATS = np.array([10.5, 9.5, 8.5, 7.5])
LONS = np.array([0.5, 1.5, 2.5])
VARIABLES = ("ForestAge_TC000", "ForestAge_TC010", "ForestAge_TC020", "ForestAge_TC030")
# A grid of about 1.1 km, for the neighbour search; the site sits on node (2, 0).
FINE_LATS = np.array([9.52, 9.51, 9.50, 9.49])
FINE_LONS = np.array([1.50, 1.51, 1.52])


def _write_file(tmp_path, grids=None, lon_first=False, lats=LATS, lons=LONS):
    directory = tmp_path / "stand_age"
    directory.mkdir(exist_ok=True)
    with h5py.File(directory / "20260101_BGIForestAgeMPIBGC1.0.0.nc", "w") as handle:
        handle["latitude"] = lats
        handle["longitude"] = lons
        for position, name in enumerate(VARIABLES):
            grid = np.full((len(lats), len(lons)), 100.0 + position) if grids is None else grids[position]
            handle[name] = grid.T if lon_first else grid


def _context(tmp_path, lat=9.4, lon=1.6):
    sites = pd.DataFrame({"site_name": ["A"], "lat": [lat], "lon": [lon]})
    return RunContext(sites=sites, raw_root=tmp_path, output_root=tmp_path / "out", project="p")


def _fine_grids(own_empty=True):
    """Every node invalid but (1, 0), 1.1 km north of the site, which holds two thresholds."""
    grids = [np.zeros((4, 3)) for _ in VARIABLES]
    grids[0][1, 0], grids[1][1, 0] = 40.0, 50.0
    grids[2][0, 2] = 80.0  # farther, and must not be taken
    if not own_empty:
        grids[3][2, 0] = 120.0
    return grids


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
        "stand_age_TC000", "stand_age_TC010", "stand_age_TC020", "stand_age_TC030", "stand_age_neighbour",
    ]  # fmt: skip
    validate(result, ["A"])


def test_stand_age_reads_a_longitude_first_file_too(tmp_path):
    grid = np.full((4, 3), 50.0)
    grid[1, 1] = 77.0
    _write_file(tmp_path, grids=[grid] * 4, lon_first=True)
    assert extract(_context(tmp_path)).values.loc["A", "stand_age_TC000"] == 77.0


def test_an_empty_own_node_keeps_its_thresholds_empty(tmp_path):
    # The merge reads these first and the ground st_age next, so a borrowed value must never land in them.
    _write_file(tmp_path, grids=_fine_grids(), lats=FINE_LATS, lons=FINE_LONS)
    result = extract(_context(tmp_path, lat=9.50, lon=1.50))

    for threshold in ("TC000", "TC010", "TC020", "TC030"):
        assert np.isnan(result.values.loc["A", f"stand_age_{threshold}"])
        assert result.status.loc["A", f"stand_age_{threshold}"] == NO_DATA
    validate(result, ["A"])


def test_an_empty_own_node_borrows_the_mean_of_the_nearest_node_holding_an_age(tmp_path):
    _write_file(tmp_path, grids=_fine_grids(), lats=FINE_LATS, lons=FINE_LONS)
    result = extract(_context(tmp_path, lat=9.50, lon=1.50))

    assert result.values.loc["A", NEIGHBOUR_COLUMN] == 45.0
    assert result.status.loc["A", NEIGHBOUR_COLUMN] == NEIGHBOUR
    table = result.extras["stand_age_neighbours"]
    assert list(table.columns) == BORROWED_COLUMNS
    assert table.iloc[0]["column"] == NEIGHBOUR_COLUMN
    assert table.iloc[0]["distance_km"] == pytest.approx(0.01 * KM_PER_DEGREE)
    assert table.iloc[0]["source_lat"] == 9.51


def test_an_own_node_with_any_threshold_borrows_nothing(tmp_path):
    _write_file(tmp_path, grids=_fine_grids(own_empty=False), lats=FINE_LATS, lons=FINE_LONS)
    result = extract(_context(tmp_path, lat=9.50, lon=1.50))

    assert result.values.loc["A", "stand_age_TC030"] == 120.0
    assert np.isnan(result.values.loc["A", NEIGHBOUR_COLUMN])
    assert result.status.loc["A", NEIGHBOUR_COLUMN] == NO_DATA
    assert result.extras["stand_age_neighbours"].empty


def test_nothing_is_borrowed_from_beyond_25_km(tmp_path):
    grid = np.full((4, 3), 50.0)  # nodes a degree apart: the nearest is 111 km away
    grid[1, 1] = np.nan
    _write_file(tmp_path, grids=[grid] * 4)
    result = extract(_context(tmp_path, lat=9.5, lon=1.5))

    assert np.isnan(result.values.loc["A", "stand_age_TC000"])
    assert np.isnan(result.values.loc["A", NEIGHBOUR_COLUMN])
    assert result.status.loc["A", NEIGHBOUR_COLUMN] == NO_DATA
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
