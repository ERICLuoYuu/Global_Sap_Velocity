from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from src.static_features.products import liang
from src.static_features.products.liang import extract, fetch_inputs
from src.static_features.results import NEIGHBOUR, NO_DATA, OK, RunContext, validate

NODATA = -3.4e38


def _write(tmp_path, points):
    """A global grid of 10 degree pixels, so the site at (45, 5) is row 4, col 18."""
    directory = tmp_path / "liang_2022"
    directory.mkdir()
    grid = np.full((18, 36), NODATA, dtype="float32")
    for (row, col), value in points.items():
        grid[row, col] = value
    profile = {
        "driver": "GTiff", "height": 18, "width": 36, "count": 1, "dtype": "float32",
        "crs": "EPSG:4326", "transform": from_origin(-180, 90, 10, 10), "nodata": NODATA,
    }  # fmt: skip
    with rasterio.open(directory / "S_mean_raster.tif", "w", **profile) as dst:
        dst.write(grid, 1)


def _context(tmp_path, lat=45.0, lon=5.0):
    sites = pd.DataFrame({"site_name": ["A"], "lat": [lat], "lon": [lon]})
    return RunContext(sites=sites, raw_root=tmp_path, output_root=tmp_path / "out", project="p")


def test_liang_own_pixel_is_ok(tmp_path):
    _write(tmp_path, {(4, 18): 42.5, (4, 19): 7.0})
    result = extract(_context(tmp_path))

    assert result.values.loc["A", "tree_richness_per_ha"] == 42.5
    assert result.status.loc["A", "tree_richness_per_ha"] == OK
    validate(result, ["A"])


def test_liang_declared_nodata_takes_the_neighbour(tmp_path):
    _write(tmp_path, {(5, 18): 7.0})
    result = extract(_context(tmp_path))

    assert result.values.loc["A", "tree_richness_per_ha"] == 7.0
    assert result.status.loc["A", "tree_richness_per_ha"] == NEIGHBOUR


def test_liang_zero_is_not_a_richness(tmp_path):
    _write(tmp_path, {(4, 18): 0.0, (4, 19): 3.0})
    result = extract(_context(tmp_path))

    assert result.values.loc["A", "tree_richness_per_ha"] == 3.0
    assert result.status.loc["A", "tree_richness_per_ha"] == NEIGHBOUR


def test_liang_beyond_one_pixel_is_no_data(tmp_path):
    _write(tmp_path, {(4, 20): 7.0})
    result = extract(_context(tmp_path))

    assert np.isnan(result.values.loc["A", "tree_richness_per_ha"])
    assert result.status.loc["A", "tree_richness_per_ha"] == NO_DATA
    validate(result, ["A"])


def test_liang_wraps_the_dateline(tmp_path):
    _write(tmp_path, {(4, 35): 7.0})
    result = extract(_context(tmp_path, lat=45.0, lon=-175.0))
    assert result.status.loc["A", "tree_richness_per_ha"] == NEIGHBOUR


def test_fetch_inputs_pins_the_file_and_md5(tmp_path, monkeypatch):
    requested = []
    monkeypatch.setattr(liang, "fetch", lambda item, directory: requested.append((item, directory)))
    fetch_inputs(_context(tmp_path))

    item, directory = requested[0]
    assert item.url == "https://ndownloader.figshare.com/files/35133814"
    assert item.name == "S_mean_raster.tif"
    assert item.md5 == "bf745222a79918df7e52e4b785ccd993"
    assert directory == tmp_path / "liang_2022"
