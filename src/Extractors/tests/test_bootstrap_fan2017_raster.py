"""Unit tests for ``src.Extractors.bootstrap_fan2017_raster``.

Hermetic — no OPeNDAP. Synthetic continent datasets injected via
monkeypatching ``open_etdepth`` cover:
  * TargetGrid spans the union of bboxes at exactly 0.1°.
  * Area-mean of mask==1 source pixels (sign-flipped).
  * ``min_valid_fraction`` gate masks coast/edge cells.
  * First-match-wins composition across overlapping continents.
  * NetCDF round-trip preserves attrs and per-cell provenance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from src.Extractors import bootstrap_fan2017_raster as mod
from src.Extractors.extract_root_depth_fan2017 import CONTINENT_BBOXES, ContinentBBox

pytestmark = pytest.mark.unit


# -- TargetGrid --------------------------------------------------------------


def test_target_grid_spans_union_at_0p1_deg():
    grid = mod.TargetGrid.union_of_bboxes(CONTINENT_BBOXES)
    spacing_lat = np.diff(grid.lats)
    spacing_lon = np.diff(grid.lons)
    np.testing.assert_allclose(spacing_lat, 0.1, atol=1e-4)
    np.testing.assert_allclose(spacing_lon, 0.1, atol=1e-4)
    # Must cover EURASIA's lat range at minimum.
    assert grid.lats.min() <= 0.0
    assert grid.lats.max() >= 83.0


def test_target_grid_centres_on_multiples_of_0p1_deg():
    bboxes = (ContinentBBox("EURASIA", lat_min=10.05, lat_max=10.45, lon_min=20.05, lon_max=20.45),)
    grid = mod.TargetGrid.union_of_bboxes(bboxes)
    # Cell centres should snap to multiples of 0.1.
    for lat in grid.lats:
        assert abs(round(float(lat) * 10) / 10 - float(lat)) < 1e-5
    for lon in grid.lons:
        assert abs(round(float(lon) * 10) / 10 - float(lon)) < 1e-5


# -- Synthetic continent dataset --------------------------------------------


def _make_synthetic_continent(
    *,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    native_res_deg: float = 1.0 / 120,
    fill_value: float,
    mask_pattern: str = "all_valid",
) -> xr.Dataset:
    """Build an in-memory ETDEPTH-shaped dataset.

    ``fill_value`` is the *raw* (pre-sign-flip) ETDEPTH value, so passing -2.0
    decodes to root_depth = +2.0 m after the bootstrap's sign flip.
    """
    n_lat = int(round((lat_max - lat_min) / native_res_deg))
    n_lon = int(round((lon_max - lon_min) / native_res_deg))
    lats = np.linspace(lat_min, lat_max, n_lat, endpoint=False).astype(np.float64)
    lons = np.linspace(lon_min, lon_max, n_lon, endpoint=False).astype(np.float64)

    et = np.full((1, n_lat, n_lon), fill_value, dtype=np.float32)
    mask = np.ones((n_lat, n_lon), dtype=np.int8)

    if mask_pattern == "checkerboard":
        ii, jj = np.indices(mask.shape)
        mask[(ii + jj) % 2 == 0] = 0
    elif mask_pattern == "left_half_invalid":
        mask[:, : n_lon // 2] = 0
    elif mask_pattern == "all_invalid":
        mask[:] = 0

    return xr.Dataset(
        data_vars={
            "ETDEPTH": (("time", "lat", "lon"), et),
            "mask": (("lat", "lon"), mask),
        },
        coords={
            "time": [0],
            "lat": lats,
            "lon": lons,
        },
    )


# -- regrid_continent --------------------------------------------------------


def test_regrid_continent_area_mean_with_sign_flip(monkeypatch):
    """Uniform synthetic continent → all valid cells equal sign-flipped value."""
    bbox = ContinentBBox("EURASIA", lat_min=10.0, lat_max=10.5, lon_min=20.0, lon_max=20.5)

    def fake_opener(continent: str) -> xr.Dataset:
        return _make_synthetic_continent(
            lat_min=10.0,
            lat_max=10.5,
            lon_min=20.0,
            lon_max=20.5,
            fill_value=-2.0,  # raw below-surface z-value
        )

    monkeypatch.setattr(mod, "open_etdepth", fake_opener)

    target = mod.TargetGrid.union_of_bboxes((bbox,))
    rd, n_valid = mod.regrid_continent(
        "EURASIA",
        bbox,
        target,
        min_valid_fraction=0.25,
    )

    finite_mask = np.isfinite(rd)
    assert finite_mask.any(), "expected at least some filled cells"
    # sign=-1 flips -2.0 → +2.0
    np.testing.assert_allclose(rd[finite_mask], 2.0, atol=1e-5)
    assert (n_valid[finite_mask] > 0).all()


def test_regrid_continent_min_valid_fraction_gate(monkeypatch):
    """left_half_invalid mask → cells in left half should be NaN."""
    bbox = ContinentBBox("EURASIA", lat_min=10.0, lat_max=10.4, lon_min=20.0, lon_max=20.4)

    def fake_opener(continent: str) -> xr.Dataset:
        return _make_synthetic_continent(
            lat_min=10.0,
            lat_max=10.4,
            lon_min=20.0,
            lon_max=20.4,
            fill_value=-1.5,
            mask_pattern="left_half_invalid",
        )

    monkeypatch.setattr(mod, "open_etdepth", fake_opener)

    target = mod.TargetGrid.union_of_bboxes((bbox,))
    rd, _ = mod.regrid_continent(
        "EURASIA",
        bbox,
        target,
        min_valid_fraction=0.5,
    )

    # Right-half cells (lon >= 20.2) should be filled; left-half NaN.
    for j, lon in enumerate(target.lons):
        if not (bbox.lon_min <= lon <= bbox.lon_max):
            continue
        col = rd[:, j]
        if lon < 20.2:
            assert np.isnan(col).all(), f"lon {lon} should be masked"
        else:
            assert np.isfinite(col).any(), f"lon {lon} should have at least one valid cell"


def test_regrid_continent_skips_cells_outside_bbox(monkeypatch):
    bbox = ContinentBBox("EURASIA", lat_min=50.0, lat_max=50.4, lon_min=0.0, lon_max=0.4)

    def fake_opener(continent: str) -> xr.Dataset:
        return _make_synthetic_continent(
            lat_min=50.0,
            lat_max=50.4,
            lon_min=0.0,
            lon_max=0.4,
            fill_value=-3.0,
        )

    monkeypatch.setattr(mod, "open_etdepth", fake_opener)

    # Target grid covers a wider area, but only cells inside bbox should fill.
    target = mod.TargetGrid.union_of_bboxes(
        (ContinentBBox("EURASIA", lat_min=49.0, lat_max=51.0, lon_min=-1.0, lon_max=1.0),)
    )
    rd, _ = mod.regrid_continent("EURASIA", bbox, target, min_valid_fraction=0.25)

    for i, lat in enumerate(target.lats):
        for j, lon in enumerate(target.lons):
            in_bbox = bbox.lat_min <= lat <= bbox.lat_max and bbox.lon_min <= lon <= bbox.lon_max
            if not in_bbox:
                assert np.isnan(rd[i, j]), f"cell ({lat}, {lon}) outside bbox should be NaN"


# -- composite_continents ----------------------------------------------------


def test_composite_continents_first_match_wins(monkeypatch):
    """Two continents overlap → the earlier one in CONTINENT_BBOXES wins."""
    captured_calls = []

    # Use a tiny pair of overlapping bboxes.
    fake_bboxes = (
        ContinentBBox("EURASIA", lat_min=0.0, lat_max=0.4, lon_min=0.0, lon_max=0.4),
        ContinentBBox("AFRICA", lat_min=0.0, lat_max=0.4, lon_min=0.0, lon_max=0.4),
    )
    monkeypatch.setattr(mod, "CONTINENT_BBOXES", fake_bboxes)

    def fake_opener(continent: str) -> xr.Dataset:
        captured_calls.append(continent)
        # EURASIA fills with -1.0 (→ +1.0 m); AFRICA fills with -9.0 (→ +9.0 m).
        # Distinguishable so the test can prove which one wins.
        fill = -1.0 if continent == "EURASIA" else -9.0
        return _make_synthetic_continent(
            lat_min=0.0,
            lat_max=0.4,
            lon_min=0.0,
            lon_max=0.4,
            fill_value=fill,
        )

    monkeypatch.setattr(mod, "open_etdepth", fake_opener)

    target = mod.TargetGrid.union_of_bboxes(fake_bboxes)
    rd, n_valid, sc = mod.composite_continents(target, min_valid_fraction=0.25)

    # Both continents called; EURASIA's value (1.0) should win.
    assert "EURASIA" in captured_calls and "AFRICA" in captured_calls
    finite = np.isfinite(rd)
    assert finite.any()
    np.testing.assert_allclose(rd[finite], 1.0, atol=1e-5)
    # source_continent should be 2 (EURASIA's code) where filled.
    assert (sc[finite] == mod.CONTINENT_CODES["EURASIA"]).all()


def test_composite_continents_marks_unfilled_cells_with_minus_one(monkeypatch):
    fake_bboxes = (ContinentBBox("EURASIA", lat_min=0.0, lat_max=0.2, lon_min=0.0, lon_max=0.2),)
    monkeypatch.setattr(mod, "CONTINENT_BBOXES", fake_bboxes)

    def fake_opener(continent: str) -> xr.Dataset:
        # All-invalid mask → no cells fill.
        return _make_synthetic_continent(
            lat_min=0.0,
            lat_max=0.2,
            lon_min=0.0,
            lon_max=0.2,
            fill_value=-2.0,
            mask_pattern="all_invalid",
        )

    monkeypatch.setattr(mod, "open_etdepth", fake_opener)

    target = mod.TargetGrid.union_of_bboxes(
        (ContinentBBox("WIDE", lat_min=-1.0, lat_max=1.0, lon_min=-1.0, lon_max=1.0),)
    )
    rd, _, sc = mod.composite_continents(target, min_valid_fraction=0.25)

    assert np.isnan(rd).all()
    assert (sc == -1).all()


# -- NetCDF round-trip -------------------------------------------------------


def test_write_netcdf_preserves_contract(tmp_path: Path):
    target = mod.TargetGrid(
        lats=np.array([10.0, 10.1], dtype=np.float32),
        lons=np.array([20.0, 20.1], dtype=np.float32),
    )
    root_depth = np.array([[1.5, 2.0], [np.nan, 2.5]], dtype=np.float32)
    n_valid = np.array([[100, 144], [0, 144]], dtype=np.int32)
    source_continent = np.array([[2, 2], [-1, 2]], dtype=np.int8)

    out_path = tmp_path / "raster.nc"
    mod.write_netcdf(
        out_path,
        target,
        root_depth,
        n_valid,
        source_continent,
        min_valid_fraction=0.25,
        version="v1",
    )

    ds = xr.open_dataset(out_path)
    try:
        assert "root_depth" in ds.data_vars
        assert "n_valid_source_pixels" in ds.data_vars
        assert "source_continent" in ds.data_vars
        assert ds.attrs["min_valid_fraction"] == pytest.approx(0.25)
        assert ds.attrs["version"] == "v1"
        assert "Fan" in ds.attrs["source"]
        np.testing.assert_array_equal(ds["lat"].values, target.lats)
        np.testing.assert_allclose(
            ds["root_depth"].values[0],
            root_depth[0],
            atol=1e-6,
        )
        # NaN cell stays NaN after round-trip.
        assert np.isnan(ds["root_depth"].values[1, 0])
        # Provenance encodes correctly.
        assert ds["source_continent"].values[1, 0] == -1
    finally:
        ds.close()
