import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.serving.statics import (
    SOIL_DEPTHS_CM,
    SOIL_PROPERTIES,
    STAND_AGE_VARIABLES,
    WORLDCLIM_BIO,
    destination_point,
    mean_over_thresholds,
    nearest_valid_values,
    ring_search,
    soil_column,
    soil_search_steps,
    soil_statics,
    stand_age_by_threshold,
    weighted_soil_properties,
    worldclim_values,
)

MERGE_SCRIPT = Path("notebooks/merge_gap_filled_hourly_orginal.py")


@pytest.fixture(scope="module")
def merge():
    spec = importlib.util.spec_from_file_location("merge_orginal_under_test", MERGE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _per_site(data, nodata, py, px, radius=3):
    """``extract_raster_value``'s search, one point at a time, as the extractor loops it."""
    height, width = data.shape
    row_start, row_stop = max(py - radius, 0), min(py + radius + 1, height)
    col_start, col_stop = max(px - radius, 0), min(px + radius + 1, width)
    if row_stop <= row_start or col_stop <= col_start:
        return np.nan
    best_distance, best_value = None, None
    for ny in range(row_start, row_stop):
        for nx in range(col_start, col_stop):
            value = data[ny, nx]
            if value != value or (nodata is not None and value == nodata):
                continue
            distance = math.hypot(ny - py, nx - px)
            if best_distance is None or distance < best_distance:
                best_distance, best_value = distance, value
    return np.nan if best_value is None else float(best_value)


def _holey_raster(seed=0, shape=(20, 30), nodata=-9999.0):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=shape).astype(np.float32)
    data[rng.random(shape) < 0.35] = nodata
    data[rng.random(shape) < 0.1] = np.nan
    data[5:12, 10:18] = nodata
    return data


def test_nearest_valid_values_is_the_per_site_search_everywhere():
    data = _holey_raster()
    rng = np.random.default_rng(1)
    rows, cols = rng.integers(-5, 25, 400), rng.integers(-5, 35, 400)
    expected = [_per_site(data, -9999.0, int(r), int(c)) for r, c in zip(rows, cols)]  # noqa: B905
    np.testing.assert_array_equal(nearest_valid_values(data, -9999.0, rows, cols), np.asarray(expected))


def test_worldclim_values_read_each_bio_file_at_the_containing_pixel(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    transform = from_origin(-180.0, 90.0, 12.0, 9.0)
    rasters = {}
    for k, bio in enumerate(WORLDCLIM_BIO.values()):
        data = _holey_raster(seed=10 + k)
        rasters[bio] = data
        with rasterio.open(
            tmp_path / f"wc2.1_30s_bio_{bio}.tif", "w", driver="GTiff", height=20, width=30, count=1,
            dtype="float32", crs="EPSG:4326", transform=transform, nodata=-9999.0,
        ) as dst:
            dst.write(data, 1)
    rng = np.random.default_rng(2)
    lats, lons = rng.uniform(-89, 89, 50), rng.uniform(-179, 179, 50)
    got = worldclim_values(lats, lons, tmp_path)
    rows, cols = rasterio.transform.rowcol(transform, lons, lats)
    for column, bio in WORLDCLIM_BIO.items():
        expected = [_per_site(rasters[bio], -9999.0, int(r), int(c)) for r, c in zip(rows, cols)]  # noqa: B905
        np.testing.assert_array_equal(got[column], np.asarray(expected))


@pytest.mark.parametrize("chunked", [False, True])
def test_stand_age_takes_the_nearest_node_per_threshold_and_averages_the_valid_ones(tmp_path, chunked):
    h5py = pytest.importorskip("h5py")
    lats, lons = np.linspace(60.0, 50.0, 11), np.linspace(0.0, 19.0, 20)
    rng = np.random.default_rng(3)
    grids = {}
    path = tmp_path / "x_BGIForestAge_test.nc"
    with h5py.File(path, "w") as handle:
        handle["latitude"], handle["longitude"] = lats, lons
        for name in STAND_AGE_VARIABLES:
            grid = rng.uniform(-50, 650, size=(11, 20)).astype(np.float32)
            grid[rng.random((11, 20)) < 0.2] = np.nan
            grids[name] = grid
            if chunked:  # the BGI file's layout: gzip + shuffle chunks spanning several rows and columns
                handle.create_dataset(name, data=grid, chunks=(4, 6), compression="gzip", compression_opts=5, shuffle=True)
            else:
                handle[name] = grid
    points_lat, points_lon = rng.uniform(49.0, 61.0, 60), rng.uniform(-1.0, 20.0, 60)
    got = stand_age_by_threshold(points_lat, points_lon, path)
    table = {}
    for name in STAND_AGE_VARIABLES:
        expected = []
        for lat, lon in zip(points_lat, points_lon):  # noqa: B905
            value = float(grids[name][int(np.abs(lats - lat).argmin()), int(np.abs(lons - lon).argmin())])
            expected.append(value if (not np.isnan(value) and 0 < value <= 500) else np.nan)
        np.testing.assert_array_equal(got[name], expected)
        table[f"stand_age_{name[-5:]}"] = expected
    np.testing.assert_allclose(mean_over_thresholds(got), pd.DataFrame(table).mean(axis=1), rtol=1e-12)


def _soil_rows(n=80, seed=4):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            soil_column(prop, top, bottom): rng.uniform(0, 60 if prop != "bdod" else 1.6, n)
            for prop in SOIL_PROPERTIES
            for top, bottom in SOIL_DEPTHS_CM
        }
    )
    for column in frame.columns:
        frame.loc[rng.random(n) < 0.15, column] = np.nan
    frame.loc[0, [soil_column("sand", t, b) for t, b in SOIL_DEPTHS_CM]] = np.nan
    return frame


def test_weighted_soil_properties_are_the_merge_function(merge):
    frame = _soil_rows()
    got = weighted_soil_properties({column: frame[column].to_numpy() for column in frame.columns})
    for i in range(len(frame)):
        expected = merge.get_weighted_soil_props(frame.iloc[i])
        for prop in SOIL_PROPERTIES:
            np.testing.assert_allclose(got[prop][i], expected[prop], rtol=1e-12, equal_nan=True)


def test_soil_statics_are_the_merge_hydraulics(merge):
    frame = _soil_rows(seed=5)
    weighted = weighted_soil_properties({column: frame[column].to_numpy() for column in frame.columns})
    got = soil_statics(weighted)
    for i in range(len(frame)):
        sand, clay, soc, cfvo = (weighted[p][i] for p in ("sand", "clay", "soc", "cfvo"))
        if np.isnan(sand) or np.isnan(clay):
            assert np.isnan(got["soil_theta_fc"][i])
            continue
        wp, fc, sat = merge.calculate_soil_hydraulics(
            sand=sand / 100.0, clay=clay / 100.0, organic_matter=(soc / 10.0) * 1.72, coarse_fragments_vol_percent=cfvo
        )
        np.testing.assert_allclose([got["soil_theta_wp"][i], got["soil_theta_fc"][i], got["soil_theta_sat"][i]], [wp, fc, sat], rtol=1e-12)
        assert got["soil_sand"][i] == sand and got["soil_cfvo"][i] == cfvo


def _extractor_destination(lat, lon, distance_m, bearing_deg):
    """``SoilGridsExtractor._calculate_destination_point`` as written (returns lon, lat)."""
    R = 6378137.0
    lat_rad, lon_rad, bearing_rad = math.radians(lat), math.radians(lon), math.radians(bearing_deg)
    new_lat_rad = math.asin(math.sin(lat_rad) * math.cos(distance_m / R) + math.cos(lat_rad) * math.sin(distance_m / R) * math.cos(bearing_rad))
    new_lon_rad = lon_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(distance_m / R) * math.cos(lat_rad), math.cos(distance_m / R) - math.sin(lat_rad) * math.sin(new_lat_rad)
    )
    return math.degrees(new_lon_rad), math.degrees(new_lat_rad)


def test_destination_point_is_the_extractor_formula():
    rng = np.random.default_rng(6)
    lats, lons = rng.uniform(-60, 78, 40), rng.uniform(-180, 180, 40)
    for distance, bearing in soil_search_steps():
        got_lat, got_lon = destination_point(lats, lons, distance, bearing)
        expected = [_extractor_destination(a, b, distance, bearing) for a, b in zip(lats, lons)]  # noqa: B905
        np.testing.assert_allclose(got_lon, [e[0] for e in expected], rtol=0, atol=1e-9)
        np.testing.assert_allclose(got_lat, [e[1] for e in expected], rtol=0, atol=1e-9)


def test_soil_search_steps_are_the_extractor_order():
    steps = soil_search_steps()
    assert steps[:9] == [(250, 0), (250, 45), (250, 90), (250, 135), (250, 180), (250, 225), (250, 270), (250, 315), (500, 0)]
    assert len(steps) == 32 and steps[-1] == (1000, 315)


def test_ring_search_takes_the_first_step_with_data_as_the_extractor_does():
    sand, clay = soil_column("sand", 0, 5), soil_column("clay", 0, 5)
    lats, lons = np.array([50.0, 10.0, -30.0, 65.0]), np.array([7.0, 120.0, -60.0, 179.99])
    field = {}

    def key(lat, lon):
        return round(float(lat), 6), round(float(lon), 6)

    def put(i, distance, bearing, values):
        if distance == 0:
            field[key(lats[i], lons[i])] = values
            return
        lat, lon = destination_point(lats[i : i + 1], lons[i : i + 1], distance, bearing)
        field[key(lat[0], lon[0])] = values

    put(0, 0, 0, (40.0, 20.0))
    put(0, 250, 0, (99.0, 99.0))  # never read: the point itself has data
    put(1, 250, 180, (11.0, 12.0))
    put(1, 250, 90, (13.0, 14.0))  # the earlier bearing at the same distance wins
    put(1, 500, 0, (15.0, 16.0))
    put(2, 750, 315, (np.nan, 30.0))  # clay alone counts as data
    put(2, 1000, 0, (31.0, 32.0))
    calls = []

    def sample(a, b):
        calls.append(len(a))
        values = [field.get(key(x, y), (np.nan, np.nan)) for x, y in zip(a, b)]  # noqa: B905
        return {sand: np.array([v[0] for v in values]), clay: np.array([v[1] for v in values])}

    raw = sample(lats, lons)
    before = {column: values.copy() for column, values in raw.items()}
    got, distance = ring_search(raw, lats, lons, sample)
    np.testing.assert_array_equal(distance, [0, 250, 750, -1])
    np.testing.assert_array_equal(got[sand], [40.0, 13.0, np.nan, np.nan])
    np.testing.assert_array_equal(got[clay], [20.0, 14.0, 30.0, np.nan])
    for column in raw:
        np.testing.assert_array_equal(raw[column], before[column])
    assert calls[1:] == [3, 3, 3] + [2] * 5 + [2] * 8 + [2] * 8 + [1] * 8
