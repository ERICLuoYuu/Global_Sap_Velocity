from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.static_features.products.fan import COLUMNS, extract, status_from_fallback
from src.static_features.results import NEIGHBOUR, NO_DATA, OK, ExtractionFailure, RunContext, validate

from src.Extractors import extract_root_depth_fan2017 as reader


def _context(tmp_path, names=("A",)):
    sites = pd.DataFrame({"site_name": list(names), "lat": [1.0] * len(names), "lon": [2.0] * len(names)})
    return RunContext(sites=sites, raw_root=tmp_path, output_root=tmp_path / "out", project="p")


def _result(site, fallback=0, values=None, continent="EURASIA", lat_pixel=1.5, lon_pixel=2.5):
    filled = dict.fromkeys(reader.FEATURE_COLUMNS, 3.5) if values is None else values
    return reader.RootDepthResult(
        site_name=site,
        lat_site=1.0,
        lon_site=2.0,
        lat_pixel=lat_pixel,
        lon_pixel=lon_pixel,
        continent=continent,
        mask_valid=fallback == 0,
        fallback_radius_px=fallback,
        values=filled,
    )


def _fake_reader(monkeypatch, results):
    caches = []

    def fetch(lat, lon, site_name="unknown", *, dataset_cache=None, **kwargs):
        caches.append(dataset_cache)
        return results[site_name]

    monkeypatch.setattr(reader, "fetch_all_variables_at_point", fetch)
    return caches


def test_fan_fallback_to_status():
    assert (status_from_fallback(0), status_from_fallback(2), status_from_fallback(-1)) == (
        "ok",
        "neighbour",
        "no_data",
    )


def test_fan_columns_are_the_current_files(tmp_path, monkeypatch):
    _fake_reader(monkeypatch, {"A": _result("A")})
    result = extract(_context(tmp_path))

    assert list(result.values.columns) == [
        "root_depth", "infiltration_depth", "deep_drainage", "drainage_frequency", "regolith_flush_rate",
        "gw_residence_time", "root_depth_mask_valid", "root_depth_fallback_px", "root_depth_continent",
        "root_depth_lat_pixel", "root_depth_lon_pixel",
    ]  # fmt: skip
    assert list(result.values.columns) == list(COLUMNS)
    assert result.values.loc["A", "root_depth"] == 3.5
    assert result.status.loc["A", "root_depth"] == OK
    validate(result, ["A"])


def test_fan_nearby_pixel_is_neighbour(tmp_path, monkeypatch):
    _fake_reader(monkeypatch, {"A": _result("A", fallback=2)})
    result = extract(_context(tmp_path))

    assert result.status.loc["A", "gw_residence_time"] == NEIGHBOUR
    assert result.values.loc["A", "root_depth_fallback_px"] == 2
    validate(result, ["A"])


def test_fan_outside_every_continent_is_no_data(tmp_path, monkeypatch):
    empty = dict.fromkeys(reader.FEATURE_COLUMNS)
    _fake_reader(
        monkeypatch,
        {"A": _result("A", fallback=-1, values=empty, continent=None, lat_pixel=float("nan"), lon_pixel=float("nan"))},
    )
    result = extract(_context(tmp_path))

    assert np.isnan(result.values.loc["A", "root_depth"])
    assert result.status.loc["A", "root_depth"] == NO_DATA
    assert result.values.loc["A", "root_depth_fallback_px"] == -1
    assert pd.isna(result.values.loc["A", "root_depth_continent"])
    validate(result, ["A"])


def test_one_empty_variable_at_a_valid_pixel_is_no_data_for_that_column_only(tmp_path, monkeypatch):
    values = dict.fromkeys(reader.FEATURE_COLUMNS, 3.5)
    values["deep_drainage"] = None
    _fake_reader(monkeypatch, {"A": _result("A", values=values)})
    result = extract(_context(tmp_path))

    assert result.status.loc["A", "deep_drainage"] == NO_DATA
    assert result.status.loc["A", "root_depth"] == OK
    validate(result, ["A"])


def test_reader_exception_stops_naming_the_site(tmp_path, monkeypatch):
    def fetch(lat, lon, site_name="unknown", *, dataset_cache=None, **kwargs):
        if site_name == "B":
            raise OSError("NetCDF: DAP failure")
        return _result(site_name)

    monkeypatch.setattr(reader, "fetch_all_variables_at_point", fetch)
    with pytest.raises(ExtractionFailure, match="fan.*B.*DAP failure"):
        extract(_context(tmp_path, names=("A", "B")))


def test_one_dataset_cache_is_shared_across_sites(tmp_path, monkeypatch):
    caches = _fake_reader(monkeypatch, {"A": _result("A"), "B": _result("B")})
    extract(_context(tmp_path, names=("A", "B")))
    assert len(caches) == 2 and caches[0] is caches[1]
