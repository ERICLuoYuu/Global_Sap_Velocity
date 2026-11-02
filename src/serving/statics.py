"""Static layers for served cells, each read the way the site extraction read it for training.

Every static column of the training tree comes from a per-site extraction (``src/Extractors``). A served
cell reads the same product the same way, at the cell centre:

=====================  ==================================================================================
elevation, slope       ASTER GDEM (Earth Engine), mean over a 250 m buffer at ``scale=30``; slope is
                       ``ee.Terrain.slope`` of it (``extract_canopyheight.py``)
aspect_sin, aspect_cos the single DEM pixel containing the point at the DEM's native scale, flat-guarded
                       (``terrain_aspect.py``, called here)
canopy_height          Meta canopy height (Earth Engine), mean over a 250 m buffer at ``scale=1.2``
mean_annual_temp, ...  WorldClim v2.1 30 s bio1/bio12/bio4/bio15, factor 1: the pixel containing the point,
                       else the nearest valid pixel within 3 (``extract_raster_value``)
stand_age              BGI forest age: per tree-cover threshold TC000..TC030 the nearest grid node, raw
                       HDF5 value kept in (0, 500]; the mean over thresholds that have one
                       (``extract_stand_age.py`` and the merge's ``load_stand_age_map``)
soil_*                 SoilGrids 2.0 ``mean``, thickness-weighted over 0-100 cm, Saxton-Rawls hydraulics
                       (the merge's ``get_weighted_soil_props`` and ``calculate_soil_hydraulics``).
                       Training queried the ISRIC REST service; a cell reads the Earth Engine copy
                       (``projects/soilgrids-isric``, regridded to Mollweide 250 m) at the pixel holding
                       the centre and, where that has no value, repeats the REST extractor's search: 250 m
                       steps to 1 km, 8 bearings each, the first point with data. "Data" is any property at
                       any depth read here; the extractor also counted silt, pH, nitrogen and 100-200 cm.
=====================  ==================================================================================

A cell's static value is therefore a point value at its centre, not an 11 km average: that is the quantity
the model was trained on. ``probes/serving/probe_statics_parity.py`` measures every layer against the
training extraction at the sites.

    python -m src.serving.statics --store S --layer worldclim|stand_age|terrain|canopy|soil [--threads T]
    python -m src.serving.statics --store S --layer combine
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Mapping, Optional

import numpy as np
import pandas as pd

from src.serving.grid import nearest_index

logger = logging.getLogger("statics")

WORLDCLIM_DIR = Path("data/raw/grided/spatial_features/bio_30s")
WORLDCLIM_BIO = {"mean_annual_temp": 1, "mean_annual_precip": 12, "temp_seasonality": 4, "precip_seasonality": 15}
WORLDCLIM_SEARCH_RADIUS_PX = 3

STAND_AGE_DIR = Path("data/raw/grided/stand_age")
STAND_AGE_VARIABLES = tuple(f"ForestAge_TC{threshold:03d}" for threshold in (0, 10, 20, 30))
STAND_AGE_MAX_YEARS = 500.0
STAND_AGE_CHUNK_CACHE_BYTES = 512 * 2**20  # > one band of chunks (15 x 1440 x 2880 float32 = 249 MB)
STAND_AGE_CHUNK_CACHE_SLOTS = 10007

SOIL_PROPERTIES = ("sand", "clay", "soc", "bdod", "cfvo")
SOIL_DEPTHS_CM = ((0, 5), (5, 15), (15, 30), (30, 60), (60, 100))
SOIL_CONVERSION = {"sand": 0.1, "clay": 0.1, "soc": 0.1, "bdod": 0.01, "cfvo": 0.1}
ORGANIC_MATTER_PER_SOC = 1.72
SOIL_SEARCH_STEP_M = 250
SOIL_SEARCH_RADIUS_M = 1000
SOIL_SEARCH_BEARINGS_DEG = (0, 45, 90, 135, 180, 225, 270, 315)
EXTRACTOR_EARTH_RADIUS_M = 6378137.0

GEE_PROJECT = "era5download-447713"
ASTER_GDEM = "projects/sat-io/open-datasets/ASTER/GDEM"
META_CANOPY_HEIGHT = "projects/sat-io/open-datasets/facebook/meta-canopy-height"
SOILGRIDS_IMAGE = "projects/soilgrids-isric/{prop}_mean"
BUFFER_M = 250
TERRAIN_SCALE_M = 30
CANOPY_SCALE_M = 1.2

LAYER_COLUMNS = {
    "worldclim": tuple(WORLDCLIM_BIO),
    "stand_age": ("stand_age",),
    "terrain": ("elevation", "slope", "aspect_sin", "aspect_cos"),
    "canopy": ("canopy_height",),
    "soil": ("soil_sand", "soil_clay", "soil_soc", "soil_cfvo", "soil_theta_wp", "soil_theta_fc", "soil_theta_sat"),
}
DEFAULT_CHUNK = {"terrain": 500, "canopy": 25, "soil": 2000}


# ---------------------------------------------------------------- local rasters


def _nearest_first_offsets(radius: int) -> list:
    """Window offsets by distance; ties keep the row-major scan order the per-site loop visits them in."""
    offsets = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    return sorted(offsets, key=lambda offset: math.hypot(*offset))


def _valid(values: np.ndarray, nodata) -> np.ndarray:
    valid = ~np.isnan(values.astype(float))
    if nodata is not None:
        valid &= values != nodata
    return valid


def nearest_valid_values(data: np.ndarray, nodata, rows, cols, radius: int = WORLDCLIM_SEARCH_RADIUS_PX) -> np.ndarray:
    """``extract_raster_value`` for many points: the containing pixel, else the nearest valid one within ``radius``."""
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    height, width = data.shape
    out = np.full(len(rows), np.nan)
    unresolved = np.ones(len(rows), dtype=bool)
    for dy, dx in _nearest_first_offsets(radius):
        r, c = rows + dy, cols + dx
        candidates = np.flatnonzero(unresolved & (r >= 0) & (r < height) & (c >= 0) & (c < width))
        if len(candidates) == 0:
            continue
        values = data[r[candidates], c[candidates]]
        valid = _valid(values, nodata)
        out[candidates[valid]] = values[valid]
        unresolved[candidates[valid]] = False
        if not unresolved.any():
            break
    return out


def worldclim_values(latitudes, longitudes, directory=WORLDCLIM_DIR) -> dict:
    import rasterio  # noqa: PLC0415

    lat = np.asarray(latitudes, dtype=float)
    lon = np.asarray(longitudes, dtype=float)
    out = {}
    for column, bio in WORLDCLIM_BIO.items():
        with rasterio.open(Path(directory) / f"wc2.1_30s_bio_{bio}.tif") as src:
            rows, cols = rasterio.transform.rowcol(src.transform, lon, lat)
            out[column] = nearest_valid_values(src.read(1), src.nodata, rows, cols)
    return out


def stand_age_path(directory=STAND_AGE_DIR) -> Path:
    candidates = sorted(Path(directory).glob("*BGIForestAge*.nc"))
    if not candidates:
        raise FileNotFoundError(f"no *BGIForestAge*.nc under {directory}")
    return candidates[0]


def stand_age_by_threshold(latitudes, longitudes, path=None) -> dict:
    """Per BGI tree-cover threshold, the raw value at the nearest grid node; NaN outside (0, 500]."""
    import h5py  # noqa: PLC0415 -- the extractor reads the raw HDF5 datasets, so this does too

    lat = np.asarray(latitudes, dtype=float)
    lon = np.asarray(longitudes, dtype=float)
    out = {}
    # The datasets are gzip chunks of (1440, 2880); a row read decompresses the band of chunks it crosses.
    # h5py's default 1 MB cache holds none of them, so every row would decompress its band again.
    with h5py.File(path or stand_age_path(), "r", rdcc_nbytes=STAND_AGE_CHUNK_CACHE_BYTES, rdcc_nslots=STAND_AGE_CHUNK_CACHE_SLOTS) as handle:
        lats = np.asarray(handle["latitude"][:], dtype=float).ravel()
        lons = np.asarray(handle["longitude"][:], dtype=float).ravel()
        i, j = nearest_index(lats, lat), nearest_index(lons, lon)
        for name in STAND_AGE_VARIABLES:
            dataset = handle[name]
            if dataset.shape != (len(lats), len(lons)):
                raise ValueError(f"{name} has shape {dataset.shape}, expected (latitude, longitude)")
            values = np.empty(len(lat))
            for row in np.unique(i):
                chosen = i == row
                values[chosen] = np.asarray(dataset[row, :], dtype=float)[j[chosen]]
            with np.errstate(invalid="ignore"):
                values[~((values > 0) & (values <= STAND_AGE_MAX_YEARS))] = np.nan
            out[name] = values
    return out


def mean_over_thresholds(by_threshold: Mapping) -> np.ndarray:
    stacked = np.vstack([by_threshold[name] for name in STAND_AGE_VARIABLES])
    count = np.isfinite(stacked).sum(axis=0)
    total = np.nansum(stacked, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


# ---------------------------------------------------------------- soil


def soil_column(prop: str, top: int, bottom: int) -> str:
    """A per-depth column as ``soilgrids_data.csv`` names it: ``sand_0_5``."""
    return f"{prop}_{top}_{bottom}"


def soil_band(prop: str, top: int, bottom: int) -> str:
    """A per-depth band as SoilGrids names it: ``sand_0-5cm_mean``."""
    return f"{prop}_{top}-{bottom}cm_mean"


def weighted_soil_properties(raw: Mapping) -> dict:
    """``get_weighted_soil_props`` over arrays: a depth counts where its sand is present, and any other
    property missing at a counted depth contributes 0."""
    n = len(np.asarray(raw[soil_column("sand", *SOIL_DEPTHS_CM[0])]))
    thickness_total = np.zeros(n)
    sums = {prop: np.zeros(n) for prop in SOIL_PROPERTIES}
    for top, bottom in SOIL_DEPTHS_CM:
        thickness = float(bottom - top)
        present = np.isfinite(np.asarray(raw[soil_column("sand", top, bottom)], dtype=float))
        thickness_total += np.where(present, thickness, 0.0)
        for prop in SOIL_PROPERTIES:
            values = np.nan_to_num(np.asarray(raw.get(soil_column(prop, top, bottom), np.zeros(n)), dtype=float), nan=0.0)
            sums[prop] += np.where(present, values * thickness, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {prop: np.where(thickness_total > 0, sums[prop] / thickness_total, np.nan) for prop in SOIL_PROPERTIES}


def soil_statics(weighted: Mapping) -> dict:
    """The tree's soil columns from weighted 0-100 cm properties (sand/clay/cfvo in %, soc in g/kg)."""
    from src.hyperparameter_optimization.feature_engineering import calculate_soil_hydraulics_sr2006  # noqa: PLC0415

    sand, clay = np.asarray(weighted["sand"], dtype=float), np.asarray(weighted["clay"], dtype=float)
    soc, cfvo = np.asarray(weighted["soc"], dtype=float), np.asarray(weighted["cfvo"], dtype=float)
    with np.errstate(invalid="ignore"):
        wp, fc, sat = calculate_soil_hydraulics_sr2006(
            sand=sand / 100.0,
            clay=clay / 100.0,
            organic_matter=(soc / 10.0) * ORGANIC_MATTER_PER_SOC,
            coarse_fragments_vol_percent=cfvo,
        )
    usable = np.isfinite(sand) & np.isfinite(clay)
    out = {"soil_sand": sand, "soil_clay": clay, "soil_soc": soc, "soil_cfvo": cfvo}
    for name, values in (("soil_theta_wp", wp), ("soil_theta_fc", fc), ("soil_theta_sat", sat)):
        out[name] = np.where(usable, np.asarray(values, dtype=float), np.nan)
    return out


# ---------------------------------------------------------------- Earth Engine


def _with_retries(request):
    from src.serving.era5_gee_daily import _with_retries as retry  # noqa: PLC0415

    return retry(request)


def _points(lats, lons, keys, buffer_m: Optional[float]):
    import ee  # noqa: PLC0415

    features = []
    for key, lat, lon in zip(keys, lats, lons):  # noqa: B905
        geometry = ee.Geometry.Point([float(lon), float(lat)])
        if buffer_m:
            geometry = geometry.buffer(buffer_m)
        features.append(ee.Feature(geometry, {"cell": int(key)}))
    return ee.FeatureCollection(features)


def _by_key(info: dict, keys, prop: str) -> np.ndarray:
    found = {feature["properties"]["cell"]: feature["properties"].get(prop) for feature in info.get("features", [])}
    return np.array([np.nan if found.get(int(key)) is None else float(found[int(key)]) for key in keys])


def buffered_mean(image, lats, lons, keys, scale: float, buffer_m: float = BUFFER_M) -> np.ndarray:
    """The mean of a one-band ``image`` over a ``buffer_m`` circle round each point, at ``scale``."""
    import ee  # noqa: PLC0415

    reduced = image.reduceRegions(collection=_points(lats, lons, keys, buffer_m), reducer=ee.Reducer.mean(), scale=scale)
    return _by_key(_with_retries(reduced.getInfo), keys, "mean")


def terrain_values(lats, lons, keys) -> dict:
    import ee  # noqa: PLC0415

    from src.Extractors.terrain_aspect import centre_pixel_terrain, native_scale, standardize_aspect, terrain_bands  # noqa: PLC0415

    elevation = ee.Image(ASTER_GDEM)
    out = {
        "elevation": buffered_mean(elevation.select(["b1"]), lats, lons, keys, TERRAIN_SCALE_M),
        "slope": buffered_mean(ee.Terrain.slope(elevation).select(["slope"]), lats, lons, keys, TERRAIN_SCALE_M),
    }
    points = [(int(key), float(lon), float(lat)) for key, lat, lon in zip(keys, lats, lons)]  # noqa: B905
    centre = _with_retries(lambda: centre_pixel_terrain(terrain_bands(elevation), points, native_scale(elevation), key="cell"))
    sines, cosines = [], []
    for key in keys:
        slope_deg, aspect_deg = centre.get(int(key), (None, None))
        standard = standardize_aspect(aspect_deg, slope_deg)
        sines.append(np.nan if standard["aspect_sin"] is None else standard["aspect_sin"])
        cosines.append(np.nan if standard["aspect_cos"] is None else standard["aspect_cos"])
    out["aspect_sin"] = np.asarray(sines, dtype=float)
    out["aspect_cos"] = np.asarray(cosines, dtype=float)
    return out


def canopy_values(lats, lons, keys, scale: float = CANOPY_SCALE_M) -> dict:
    import ee  # noqa: PLC0415

    image = ee.ImageCollection(META_CANOPY_HEIGHT).mosaic().select(["cover_code"])
    return {"canopy_height": buffered_mean(image, lats, lons, keys, scale)}


def soil_band_groups() -> list:
    """SoilGrids bands grouped by native grid (the asset holds more than one), each group one image."""
    import ee  # noqa: PLC0415

    groups = {}
    for prop in SOIL_PROPERTIES:
        image = ee.Image(SOILGRIDS_IMAGE.format(prop=prop))
        for top, bottom in SOIL_DEPTHS_CM:
            band = soil_band(prop, top, bottom)
            projection = _with_retries(image.select([band]).projection().getInfo)
            key = (projection.get("wkt") or projection.get("crs"), tuple(projection["transform"]))
            groups.setdefault(key, []).append((prop, top, bottom, image.select([band])))
    return [members for members in groups.values()]


def soilgrids_raw(lats, lons, keys, groups) -> dict:
    """Per-depth SoilGrids values in the units ``soilgrids_data.csv`` holds (the pixel containing each point)."""
    import ee  # noqa: PLC0415

    out = {}
    collection = _points(lats, lons, keys, None)
    for members in groups:
        image = ee.Image.cat([member[3] for member in members])
        projection = members[0][3].projection()
        info = _with_retries(image.reduceRegions(collection=collection, reducer=ee.Reducer.first(), crs=projection).getInfo)
        for prop, top, bottom, _ in members:
            name = soil_band(prop, top, bottom) if len(members) > 1 else "first"
            out[soil_column(prop, top, bottom)] = _by_key(info, keys, name) * SOIL_CONVERSION[prop]
    return out


def destination_point(lats, lons, distance_m: float, bearing_deg: float) -> tuple:
    """``SoilGridsExtractor._calculate_destination_point`` over arrays, returned as (lat, lon)."""
    lat = np.radians(np.asarray(lats, dtype=float))
    lon = np.radians(np.asarray(lons, dtype=float))
    bearing = math.radians(bearing_deg)
    angle = distance_m / EXTRACTOR_EARTH_RADIUS_M
    new_lat = np.arcsin(np.sin(lat) * math.cos(angle) + np.cos(lat) * math.sin(angle) * math.cos(bearing))
    new_lon = lon + np.arctan2(math.sin(bearing) * math.sin(angle) * np.cos(lat), math.cos(angle) - np.sin(lat) * np.sin(new_lat))
    return np.degrees(new_lat), np.degrees(new_lon)


def soil_search_steps() -> list:
    """(distance, bearing) in the order the REST extractor tries them when the point itself has no data."""
    distances = range(SOIL_SEARCH_STEP_M, SOIL_SEARCH_RADIUS_M + 1, SOIL_SEARCH_STEP_M)
    return [(distance, bearing) for distance in distances for bearing in SOIL_SEARCH_BEARINGS_DEG]


def soil_has_data(raw: Mapping) -> np.ndarray:
    """``_check_data_validity`` per point: any property at any depth present."""
    return np.isfinite(np.vstack([np.asarray(values, dtype=float) for values in raw.values()])).any(axis=0)


def ring_search(raw: Mapping, lats, lons, sample: Callable) -> tuple:
    """Where a point has no SoilGrids data, the values at the first search step that has, as the extractor searches.

    ``sample(lats, lons) -> column -> values`` reads points. Returns the raw columns with found values in (a new
    mapping) and the distance per point: 0 at the point itself, the step distance, or -1 where nothing within 1 km.
    """
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    out = {column: np.asarray(values, dtype=float).copy() for column, values in raw.items()}
    distance = np.where(soil_has_data(raw), 0.0, -1.0)
    for step_distance, bearing in soil_search_steps():
        pending = np.flatnonzero(distance < 0)
        if len(pending) == 0:
            break
        step_lats, step_lons = destination_point(lats[pending], lons[pending], step_distance, bearing)
        found = sample(step_lats, step_lons)
        hit = soil_has_data(found)
        for column in out:
            out[column][pending[hit]] = np.asarray(found[column], dtype=float)[hit]
        distance[pending[hit]] = step_distance
    return out, distance


# ---------------------------------------------------------------- runner


def run_in_chunks(function: Callable, lats, lons, keys, chunk: int, threads: int, label: str) -> dict:
    """``function(lats, lons, keys) -> column -> values`` over chunks of points, threaded, stitched in order."""
    n = len(keys)
    bounds = [(start, min(start + chunk, n)) for start in range(0, n, chunk)]
    out = {}
    began = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = {pool.submit(function, lats[a:b], lons[a:b], keys[a:b]): (a, b) for a, b in bounds}
        for future in as_completed(futures):
            a, b = futures[future]
            for column, values in future.result().items():
                out.setdefault(column, np.full(n, np.nan))[a:b] = values
            done += 1
            if done % max(1, len(bounds) // 20) == 0 or done == len(bounds):
                logger.info("%s: %d/%d chunks (%.0f s)", label, done, len(bounds), time.time() - began)
    return out


def layer_values(layer: str, cells: pd.DataFrame, threads: int, chunk: Optional[int], canopy_scale: float) -> dict:
    lats = cells["latitude"].to_numpy(dtype=float)
    lons = cells["longitude"].to_numpy(dtype=float)
    keys = np.arange(len(cells))
    if layer == "worldclim":
        return worldclim_values(lats, lons)
    if layer == "stand_age":
        return {"stand_age": mean_over_thresholds(stand_age_by_threshold(lats, lons))}
    import ee  # noqa: PLC0415

    ee.Initialize(project=GEE_PROJECT)
    size = chunk or DEFAULT_CHUNK[layer]
    if layer == "terrain":
        return run_in_chunks(terrain_values, lats, lons, keys, size, threads, layer)
    if layer == "canopy":
        return run_in_chunks(lambda a, b, k: canopy_values(a, b, k, canopy_scale), lats, lons, keys, size, threads, layer)
    if layer == "soil":
        groups = soil_band_groups()

        def sample(a, b):
            return run_in_chunks(lambda x, y, k: soilgrids_raw(x, y, k, groups), a, b, np.arange(len(a)), size, threads, layer)

        raw, distance = ring_search(sample(lats, lons), lats, lons, sample)
        steps, counts = np.unique(distance, return_counts=True)
        logger.info("soil search distance (m) -> cells: %s", {float(d): int(c) for d, c in zip(steps, counts)})  # noqa: B905
        return soil_statics(weighted_soil_properties(raw))
    raise ValueError(f"unknown layer {layer!r}")


def _layer_path(store_root: Path, layer: str) -> Path:
    return Path(store_root) / "statics" / f"{layer}.parquet"


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def combine(store_root: Path) -> Path:
    """Join every built layer on grid row/col into ``statics/statics.parquet``."""
    frames = [pd.read_parquet(_layer_path(store_root, layer)) for layer in LAYER_COLUMNS if _layer_path(store_root, layer).exists()]
    if not frames:
        raise FileNotFoundError(f"no static layer built under {Path(store_root) / 'statics'}")
    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.merge(frame, on=["row", "col"], how="outer", validate="one_to_one")
    path = Path(store_root) / "statics" / "statics.parquet"
    _write_parquet(combined, path)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", required=True)
    parser.add_argument("--layer", required=True, choices=[*LAYER_COLUMNS, "combine"])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=None)
    parser.add_argument("--canopy_scale", type=float, default=CANOPY_SCALE_M)
    parser.add_argument("--limit", type=int, default=None, help="only the first N cells (a timing run)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = Path(args.store)
    if args.layer == "combine":
        logger.info("wrote %s", combine(root))
        return 0
    cells = pd.read_parquet(root / "cells.parquet", columns=["row", "col", "latitude", "longitude"])
    if args.limit:
        cells = cells.iloc[: args.limit]
    began = time.time()
    values = layer_values(args.layer, cells, args.threads, args.chunk, args.canopy_scale)
    frame = cells[["row", "col"]].copy()
    for column in LAYER_COLUMNS[args.layer]:
        frame[column] = values[column]
    missing = {column: int(frame[column].isna().sum()) for column in LAYER_COLUMNS[args.layer]}
    suffix = f".limit{args.limit}" if args.limit else ""
    path = _layer_path(root, args.layer + suffix)
    _write_parquet(frame, path)
    logger.info("wrote %s: %d cells in %.0f s; missing per column %s", path, len(frame), time.time() - began, missing)
    return 0


if __name__ == "__main__":
    sys.exit(main())
