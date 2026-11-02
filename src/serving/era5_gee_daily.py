"""ERA5-Land daily base columns computed in Earth Engine from ``ECMWF/ERA5_LAND/HOURLY``.

The hourly derivations and daily statistics of :mod:`src.serving.era5_daily`, written as Earth Engine
image algebra, so one day of a whole grid is one server-side computation instead of 24 hourly downloads.
``era5_daily`` stays the reference: ``probes/serving/probe_gee_daily_parity.py`` samples
:func:`daily_image` at training sites and compares it with ``era5_daily`` applied to the site extract,
which read the same collection.

The daily image keeps ERA5-Land's native projection as its default, so sampling it at a point reads
the native pixel the hourly site extract read.
"""

from __future__ import annotations

import random
import time

import ee
import numpy as np
import pandas as pd

from src.env_derive.ppfd import PPFD_PER_SW_IN
from src.feature_contract.definitions import SWC_LAYERS, swc_level_col
from src.serving.era5_daily import (
    DEWPOINT,
    HOURS_PER_DAY,
    KELVIN_OFFSET,
    PET_FLOOR_M,
    POTENTIAL_EVAPORATION,
    PRECIP_PET_MAX,
    PRECIPITATION,
    SECONDS_PER_HOUR,
    SHORTWAVE,
    SOIL_TEMPERATURE,
    SOIL_WATER,
    SUM_ONLY,
    SUMMED,
    SURFACE_PRESSURE,
    TEMPERATURE,
    WIND_U,
    WIND_V,
    WITH_EXTREMES,
)

COLLECTION = "ECMWF/ERA5_LAND/HOURLY"
#: The site extract's sampling scale (``src/Extractors/extract_climatedata_gee.py``), ERA5-Land's native grid.
ERA5_NATIVE_SCALE_M = 11132.0

#: Earth Engine band names cannot hold "/", so ``prcip/PET`` travels under this name.
PRECIP_OVER_PET_BAND = "prcip_over_PET"
BAND_TO_COLUMN = {PRECIP_OVER_PET_BAND: "prcip/PET"}

#: computePixels returns masked pixels as zeros; masked pixels are filled with this and read back as NaN.
MISSING_VALUE = -9999.0
MAX_ATTEMPTS = 6
BACKOFF_SECONDS = 15.0

HOURLY_DRIVER_BANDS = (
    "ta",
    "vpd",
    "rh",
    "ws",
    "sw_in",
    "ppfd_in",
    PRECIPITATION,
    POTENTIAL_EVAPORATION,
    SURFACE_PRESSURE,
    *SOIL_TEMPERATURE,
    *(swc_level_col(layer) for layer in SWC_LAYERS),
)
MEAN_BANDS = tuple(band for band in HOURLY_DRIVER_BANDS if band not in SUM_ONLY)
EXTREME_BANDS = tuple(f"{band}_{stat}" for band in WITH_EXTREMES for stat in ("min", "max"))
SUM_BANDS = tuple(f"{band}_sum" for band in SUMMED)
DAILY_BANDS = (*MEAN_BANDS, *EXTREME_BANDS, *SUM_BANDS, PRECIP_OVER_PET_BAND)


def _saturation_vapour_pressure(celsius: ee.Image) -> ee.Image:
    """FAO-56 Tetens, kPa: :func:`src.env_derive.humidity.saturation_vapour_pressure` in image algebra."""
    return celsius.expression("0.6108 * exp(17.27 * t / (t + 237.3))", {"t": celsius})


def hourly_drivers(image: ee.Image) -> ee.Image:
    """One hourly ERA5-Land image -> the hourly driver bands of :func:`src.serving.era5_daily.hourly_drivers`."""
    ta = image.select(TEMPERATURE).subtract(KELVIN_OFFSET)
    dewpoint = image.select(DEWPOINT).subtract(KELVIN_OFFSET)
    saturation = _saturation_vapour_pressure(ta)
    rh = _saturation_vapour_pressure(dewpoint).divide(saturation).multiply(100.0).clamp(0.0, 100.0)
    vpd = saturation.multiply(ee.Image.constant(1.0).subtract(rh.divide(100.0)))
    sw_in = image.select(SHORTWAVE).divide(SECONDS_PER_HOUR).max(0.0)
    bands = [
        ta.rename("ta"),
        vpd.rename("vpd"),
        rh.rename("rh"),
        image.select(WIND_U).hypot(image.select(WIND_V)).rename("ws"),
        sw_in.rename("sw_in"),
        sw_in.multiply(PPFD_PER_SW_IN).rename("ppfd_in"),
        image.select(PRECIPITATION),
        image.select(POTENTIAL_EVAPORATION),
        image.select(SURFACE_PRESSURE),
        *(image.select(band) for band in SOIL_TEMPERATURE),
        *(image.select(source).rename(swc_level_col(layer)) for layer, source in zip(SWC_LAYERS, SOIL_WATER)),
    ]
    return ee.Image.cat(bands).copyProperties(image, ["system:time_start"])


def _day_start(day) -> ee.Date:
    stamp = pd.Timestamp(day)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    if stamp != stamp.normalize():
        raise ValueError(f"a day starts at UTC midnight, got {stamp}")
    return ee.Date(stamp.strftime("%Y-%m-%dT00:00:00Z"))


def native_projection(day) -> ee.Projection:
    start = _day_start(day)
    first = ee.Image(ee.ImageCollection(COLLECTION).filterDate(start, start.advance(1, "day")).first())
    return first.select(TEMPERATURE).projection()


def daily_image(day) -> ee.Image:
    """Every ERA5-Land base column of one UTC day, as bands :data:`DAILY_BANDS`.

    A band is masked at a pixel unless all 24 hours of that day hold a value, the local rule.
    """
    start = _day_start(day)
    drivers = ee.ImageCollection(COLLECTION).filterDate(start, start.advance(1, "day")).map(hourly_drivers)
    complete = drivers.select(list(HOURLY_DRIVER_BANDS)).reduce(ee.Reducer.count()).eq(HOURS_PER_DAY)
    complete = complete.rename(list(HOURLY_DRIVER_BANDS))

    means = drivers.select(list(MEAN_BANDS)).reduce(ee.Reducer.mean()).rename(list(MEAN_BANDS))
    means = means.updateMask(complete.select(list(MEAN_BANDS)))
    extremes = drivers.select(list(WITH_EXTREMES)).reduce(ee.Reducer.minMax()).select(list(EXTREME_BANDS))
    # select(sources, new_names): each driver's mask taken twice (min and max) under distinct names.
    extremes = extremes.updateMask(
        complete.select([band.rsplit("_", 1)[0] for band in EXTREME_BANDS], list(EXTREME_BANDS))
    )
    sums = drivers.select(list(SUMMED)).reduce(ee.Reducer.sum()).rename(list(SUM_BANDS))
    sums = sums.updateMask(complete.select(list(SUMMED)))

    pet = sums.select(f"{POTENTIAL_EVAPORATION}_sum").abs().max(PET_FLOOR_M)
    ratio = sums.select(f"{PRECIPITATION}_sum").divide(pet).clamp(0.0, PRECIP_PET_MAX).rename(PRECIP_OVER_PET_BAND)
    image = ee.Image.cat([means, extremes, sums, ratio]).select(list(DAILY_BANDS))
    return image.setDefaultProjection(native_projection(day))


def sample_points(day, points) -> pd.DataFrame:
    """:func:`daily_image` at ``points`` (iterable of ``(point_id, latitude, longitude)``), one row per point.

    Columns carry base-column names (``prcip/PET``, not the Earth Engine band name).
    """
    features = [ee.Feature(ee.Geometry.Point([float(lon), float(lat)]), {"point_id": str(pid)}) for pid, lat, lon in points]
    sampled = daily_image(day).reduceRegions(
        collection=ee.FeatureCollection(features), reducer=ee.Reducer.first(), scale=ERA5_NATIVE_SCALE_M
    )
    rows = [feature["properties"] for feature in sampled.getInfo()["features"]]
    frame = pd.DataFrame(rows).rename(columns=BAND_TO_COLUMN)
    frame["day"] = pd.Timestamp(day).tz_localize("UTC") if pd.Timestamp(day).tzinfo is None else pd.Timestamp(day)
    return frame


TRUNCATED_PAYLOAD = "EOF: reading array data"


def _is_transient(exc: BaseException) -> bool:
    """A refusal, or a pixel payload Earth Engine cut short.

    A short read arrives as numpy's ``ValueError``, not an :class:`ee.EEException`, because the request
    was accepted and the transport failed mid-stream; retrying it costs one call, whereas letting it
    escape ends the whole year. Any other error is this code's own defect and must surface at once.
    """
    if isinstance(exc, ee.EEException):
        return True
    return isinstance(exc, ValueError) and TRUNCATED_PAYLOAD in str(exc)


def _with_retries(request):
    """Call ``request()``; retry Earth Engine's transient failures with jittered backoff, then raise."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return request()
        except Exception as exc:  # noqa: BLE001 - _is_transient decides; anything else re-raises at once
            if not _is_transient(exc) or attempt == MAX_ATTEMPTS:
                raise
            wait = BACKOFF_SECONDS * attempt * (1.0 + random.random())
            print(f"  Earth Engine attempt {attempt} failed ({exc}); retrying in {wait:.0f} s", flush=True)
            time.sleep(wait)
    raise AssertionError("unreachable")


def fetch_tile(day, grid, row0: int, col0: int, height: int, width: int) -> dict:
    """One window of :func:`daily_image` on ``grid`` (a :class:`src.serving.grid.Grid`), no resampling
    when ``grid`` is a crop of the native grid. Column name -> float32 array (height, width), NaN where masked.
    """
    image = daily_image(day).toFloat().unmask(MISSING_VALUE, False)
    request = {
        "expression": image,
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": int(width), "height": int(height)},
            "affineTransform": grid.affine(row0, col0),
            "crsCode": grid.crs,
        },
    }
    array = _with_retries(lambda: ee.data.computePixels(request))
    out = {}
    for band in DAILY_BANDS:
        values = np.asarray(array[band], dtype=np.float32)
        out[BAND_TO_COLUMN.get(band, band)] = np.where(values == MISSING_VALUE, np.float32(np.nan), values)
    return out
