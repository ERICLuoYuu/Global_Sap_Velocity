from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from src.static_features.fetch import Download
from src.static_features.grid import cell_of
from src.static_features.neighbours import BORROWED_COLUMNS, KM_PER_DEGREE
from src.static_features.products import lesiv
from src.static_features.products.lesiv import cell_shares, extract, fetch_inputs
from src.static_features.results import NEIGHBOUR, NO_DATA, OK, ExtractionFailure, RunContext, validate

# 240 pixels per degree from (10.0, 50.1). The site at (50.0525, 10.0525) has cell bounds 10.05..10.0583 and
# 50.05..50.0583, so the pixels centred in it are rows 10-11, cols 12-13.
LAT, LON = 50.0525, 10.0525


def _write(tmp_path, cell, background=0, nodata=None, others=()):
    directory = tmp_path / "lesiv_2022"
    directory.mkdir()
    grid = np.full((24, 24), background, dtype="uint8")
    grid[10:12, 12:14] = cell
    for rows, cols, value in others:  # other cells, as pixel slices
        grid[rows, cols] = value
    profile = {
        "driver": "GTiff", "height": 24, "width": 24, "count": 1, "dtype": "uint8", "crs": "EPSG:4326",
        "transform": from_origin(10.0, 50.1, 1 / 240, 1 / 240), "nodata": nodata,
    }  # fmt: skip
    with rasterio.open(directory / "FML_v3-2_with-colorbar.tif", "w", **profile) as dst:
        dst.write(grid, 1)


def _context(tmp_path, lat=LAT, lon=LON):
    row, col = cell_of(lat, lon)
    sites = pd.DataFrame({"site_name": ["A"], "lat": [lat], "lon": [lon], "row": [row], "col": [col]})
    return RunContext(sites=sites, raw_root=tmp_path, output_root=tmp_path / "out", project="p")


def test_lesiv_shares_hand_computed():
    block = np.array([11, 11, 11, 11, 31, 31, 31, 128, 128, 128])
    shares, dominant = cell_shares(block, None)
    assert shares[11] == 0.4 and shares[31] == 0.3 and shares[53] == 0.0 and dominant == 11


def test_lesiv_shares_of_a_two_dimensional_block():
    shares, dominant = cell_shares(np.array([[20, 20], [40, 0]]), None)
    assert shares[20] == 0.5 and shares[40] == 0.25 and dominant == 20


def test_lesiv_tie_takes_lowest_code():
    shares, dominant = cell_shares(np.array([32, 32, 53, 53]), None)
    assert shares[32] == shares[53] == 0.5
    assert dominant == 32


def test_lesiv_no_forest_has_zero_shares_and_no_dominant():
    shares, dominant = cell_shares(np.array([0, 128, 0, 128]), None)
    assert set(shares) == {11, 20, 31, 32, 40, 53}
    assert all(share == 0.0 for share in shares.values())
    assert dominant is None


def test_lesiv_unknown_value_stops():
    with pytest.raises(ValueError, match="77"):
        cell_shares(np.array([11, 77]), None)


def test_lesiv_declared_nodata_is_allowed():
    shares, dominant = cell_shares(np.array([11, 255, 255, 255]), 255)
    assert shares[11] == 0.25 and dominant == 11


def test_lesiv_reads_the_pixels_centred_in_the_cell(tmp_path):
    # The surrounding pixels hold another class, so a window off by one pixel would change every number.
    _write(tmp_path, [[11, 11], [31, 128]], background=53)
    result = extract(_context(tmp_path))

    assert result.values.loc["A", "fm_share_11"] == 0.5
    assert result.values.loc["A", "fm_share_31"] == 0.25
    assert result.values.loc["A", "fm_share_53"] == 0.0
    assert result.values.loc["A", "fm_dominant"] == 11
    assert result.status.loc["A", "fm_share_11"] == OK
    assert result.status.loc["A", "fm_dominant"] == OK
    assert result.coverage.loc["A", "fm_dominant"] == 0.75
    validate(result, ["A"])


def test_lesiv_columns(tmp_path):
    _write(tmp_path, 11)
    result = extract(_context(tmp_path))
    assert list(result.values.columns) == [f"fm_share_{code}" for code in (11, 20, 31, 32, 40, 53)] + ["fm_dominant"]


def test_lesiv_cell_without_forest_is_ok_at_zero_with_no_dominant(tmp_path):
    _write(tmp_path, 0)
    result = extract(_context(tmp_path))

    assert result.values.loc["A", "fm_share_20"] == 0.0
    assert result.status.loc["A", "fm_share_20"] == OK
    assert np.isnan(result.values.loc["A", "fm_dominant"])
    assert result.status.loc["A", "fm_dominant"] == NO_DATA
    assert result.coverage.loc["A", "fm_dominant"] == 0.0
    validate(result, ["A"])


def test_lesiv_unknown_value_in_the_cell_stops_naming_the_site(tmp_path):
    _write(tmp_path, [[11, 77], [11, 11]])
    with pytest.raises(ExtractionFailure, match="A.*77"):
        extract(_context(tmp_path))


def test_lesiv_cell_outside_the_raster_stops_naming_the_site(tmp_path):
    _write(tmp_path, 11)
    with pytest.raises(ExtractionFailure, match="A"):
        extract(_context(tmp_path, lat=20.0, lon=10.0525))


def test_fetch_inputs_takes_only_the_map_from_the_record(tmp_path, monkeypatch):
    items = [
        Download("u1", "FML_v3-2_with-colorbar.tif", size=1),
        Download("u2", "FML_v3-2.tif", size=1),
    ]
    fetched = []
    monkeypatch.setattr(lesiv, "zenodo_downloads", lambda record_id, directory: items)
    monkeypatch.setattr(lesiv, "fetch", lambda item, directory: fetched.append(item.name))
    fetch_inputs(_context(tmp_path))
    assert fetched == ["FML_v3-2_with-colorbar.tif"]


def test_fetch_inputs_stops_when_the_record_lacks_the_map(tmp_path, monkeypatch):
    monkeypatch.setattr(lesiv, "zenodo_downloads", lambda record_id, directory: [Download("u2", "other.tif", size=1)])
    with pytest.raises(ExtractionFailure, match="FML_v3-2_with-colorbar.tif"):
        fetch_inputs(_context(tmp_path))


def test_lesiv_cell_shares_of_an_empty_block_is_refused():
    with pytest.raises(ValueError):
        cell_shares(np.empty((0, 0), dtype="uint8"), None)


def test_lesiv_cell_without_forest_borrows_the_class_of_the_nearest_forest_cell(tmp_path):
    # The cell east of the site's is 0.7 km away and holds 31; two cells north, 2 km away, holds 11.
    _write(tmp_path, 0, others=[(slice(10, 12), slice(14, 16), 31), (slice(6, 8), slice(12, 14), 11)])
    result = extract(_context(tmp_path))

    assert result.values.loc["A", "fm_dominant"] == 31
    assert result.status.loc["A", "fm_dominant"] == NEIGHBOUR
    assert result.values.loc["A", "fm_share_31"] == 0.0  # the shares describe the site's own cell
    assert result.status.loc["A", "fm_share_31"] == OK
    assert result.coverage.loc["A", "fm_dominant"] == 0.0
    table = result.extras["lesiv_neighbours"]
    assert list(table.columns) == BORROWED_COLUMNS
    # The east cell is centred 0.01 deg east of the site and 1/600 deg north of it.
    east_km = 0.01 * KM_PER_DEGREE * np.cos(np.radians(LAT))
    assert table.iloc[0]["distance_km"] == pytest.approx(np.hypot(east_km, (50.0541667 - LAT) * KM_PER_DEGREE), abs=1e-3)
    validate(result, ["A"])


def test_lesiv_cell_with_forest_borrows_nothing(tmp_path):
    _write(tmp_path, 11, others=[(slice(10, 12), slice(14, 16), 31)])
    result = extract(_context(tmp_path))

    assert result.values.loc["A", "fm_dominant"] == 11
    assert result.status.loc["A", "fm_dominant"] == OK
    assert result.extras["lesiv_neighbours"].empty


def test_lesiv_unknown_value_in_a_neighbour_cell_stops(tmp_path):
    _write(tmp_path, 0, others=[(slice(10, 12), slice(14, 16), 77)])
    with pytest.raises(ExtractionFailure, match="77"):
        extract(_context(tmp_path))
