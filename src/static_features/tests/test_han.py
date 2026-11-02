from __future__ import annotations

import json

import netCDF4
import numpy as np
import pandas as pd
import pytest
from src.static_features.products import han
from src.static_features.products.han import OTHER_TYPES, TREE_TYPES, extract, fetch_inputs, load_grids
from src.static_features.results import NEIGHBOUR, NO_DATA, OK, ExtractionFailure, RunContext, validate

MISSING = -32768.0
SHAPE = (18, 36)  # 10 degree pixels, so the site at (45, 5) is pixel (4, 18)


def _write(directory, kind, plant, points):
    grid = np.full(SHAPE, MISSING, dtype="f4")
    for (row, col), value in points.items():
        grid[row, col] = value
    with netCDF4.Dataset(directory / f"Global_RD_{kind}_{plant}_2015.nc", "w") as dataset:
        dataset.createDimension("lat", SHAPE[0])
        dataset.createDimension("lon", SHAPE[1])
        variable = dataset.createVariable("depth", "f4", ("lat", "lon"))
        variable.setncattr("missing_value", MISSING)
        variable[:] = grid


def _history(tmp_path, estimate=None, uncertainty=None):
    """estimate: plant -> {(row, col): depth}; uncertainty defaults to the depth / 10 at the same pixels."""
    directory = tmp_path / "han_2025" / "History"
    directory.mkdir(parents=True)
    estimate = estimate or {}
    for plant in TREE_TYPES + OTHER_TYPES:
        points = estimate.get(plant, {})
        _write(directory, "EST", plant, points)
        spread = (uncertainty or {}).get(plant, {point: value / 10 for point, value in points.items()})
        _write(directory, "UI", plant, spread)
    return directory


def _context(tmp_path, lat=45.0, lon=5.0, name="A"):
    sites = pd.DataFrame({"site_name": [name], "lat": [lat], "lon": [lon]})
    return RunContext(sites=sites, raw_root=tmp_path, output_root=tmp_path / "out", project="p")


def test_han_tree_pixel_is_ok_with_uncertainty_and_type(tmp_path):
    _history(tmp_path, {"BETTr": {(4, 18): 7.5}}, {"BETTr": {(4, 18): 1.5}})
    result = extract(_context(tmp_path))

    assert result.values.loc["A", "root_depth_han"] == 7.5
    assert result.values.loc["A", "root_depth_han_uncertainty"] == 1.5
    assert result.values.loc["A", "root_depth_han_type"] == "BETTr"
    assert result.status.loc["A", "root_depth_han"] == OK
    assert result.status.loc["A", "root_depth_han_type"] == OK
    validate(result, ["A"])


def test_han_grass_pixel_is_skipped_for_a_neighbouring_tree(tmp_path):
    _history(tmp_path, {"C3G": {(4, 18): 2.0}, "NETBo": {(4, 19): 9.0}})
    result = extract(_context(tmp_path))

    assert result.values.loc["A", "root_depth_han"] == 9.0
    assert result.values.loc["A", "root_depth_han_uncertainty"] == pytest.approx(0.9)
    assert result.values.loc["A", "root_depth_han_type"] == "NETBo"
    assert result.status.loc["A", "root_depth_han"] == NEIGHBOUR
    assert result.status.loc["A", "root_depth_han_type"] == NEIGHBOUR


def test_han_no_tree_within_one_pixel_is_no_data(tmp_path):
    _history(tmp_path, {"NETBo": {(4, 20): 9.0}})
    result = extract(_context(tmp_path))

    assert np.isnan(result.values.loc["A", "root_depth_han"])
    assert pd.isna(result.values.loc["A", "root_depth_han_type"])
    assert result.status.loc["A", "root_depth_han_type"] == NO_DATA
    validate(result, ["A"])


def test_han_grass_alone_is_no_data(tmp_path):
    _history(tmp_path, {"C4G": {(4, 18): 2.0}})
    assert extract(_context(tmp_path)).status.loc["A", "root_depth_han"] == NO_DATA


def test_han_two_types_on_one_pixel_stops(tmp_path):
    _history(tmp_path, {"BETTr": {(4, 18): 7.5}, "BDTBo": {(4, 18): 3.0}})
    with pytest.raises(ExtractionFailure, match="BDTBo"):
        extract(_context(tmp_path))


def test_han_tree_and_grass_on_one_pixel_stops(tmp_path):
    _history(tmp_path, {"BETTr": {(4, 18): 7.5}, "C3G": {(4, 18): 3.0}})
    with pytest.raises(ExtractionFailure, match="C3G"):
        extract(_context(tmp_path))


def test_han_wraps_the_dateline(tmp_path):
    _history(tmp_path, {"NDT": {(4, 35): 6.0}})
    result = extract(_context(tmp_path, lat=45.0, lon=-175.0))

    assert result.values.loc["A", "root_depth_han"] == 6.0
    assert result.status.loc["A", "root_depth_han"] == NEIGHBOUR


def test_load_grids_holds_trees_only(tmp_path):
    directory = _history(tmp_path, {"BDTTe": {(1, 2): 4.0}, "BESTe": {(3, 4): 5.0}})
    grids = load_grids(directory)

    assert grids.depth[1, 2] == 4.0
    assert np.isnan(grids.depth[3, 4])
    assert grids.type_index[3, 4] == -1
    assert TREE_TYPES[grids.type_index[1, 2]] == "BDTTe"


def test_empty_uncertainty_is_no_data_for_that_column_only(tmp_path):
    _history(tmp_path, {"BETTr": {(4, 18): 7.5}}, {"BETTr": {}})
    result = extract(_context(tmp_path))

    assert result.status.loc["A", "root_depth_han"] == OK
    assert result.status.loc["A", "root_depth_han_uncertainty"] == NO_DATA
    validate(result, ["A"])


def test_fetch_copies_the_26_files_and_records_the_archive_md5(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    for kind in ("EST", "UI"):
        for plant in TREE_TYPES + OTHER_TYPES:
            (source / f"Global_RD_{kind}_{plant}_2015.nc").write_bytes(b"x")
    monkeypatch.setattr(han, "SOURCE_DIR", source)
    raw = tmp_path / "raw"
    raw.mkdir()
    context = RunContext(sites=pd.DataFrame(), raw_root=raw, output_root=tmp_path / "out", project="p")

    fetch_inputs(context)

    assert len(list((raw / "han_2025" / "History").glob("*.nc"))) == 26
    assert json.loads((raw / "han_2025" / "source.json").read_text(encoding="utf-8"))["md5"] == han.ARCHIVE_MD5


def test_fetch_stops_when_the_unpacked_copy_is_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(han, "SOURCE_DIR", tmp_path / "nowhere")
    context = RunContext(sites=pd.DataFrame(), raw_root=tmp_path, output_root=tmp_path / "out", project="p")
    with pytest.raises(ExtractionFailure, match="nowhere"):
        fetch_inputs(context)
