"""ERA5-Land hourly -> the daily base columns of the merged training tree, and the solar columns.

The merge (``notebooks/merge_gap_filled_hourly_orginal.py``) builds one row per UTC day from hourly
values: the bare column is the mean, ``_min``/``_max``/``_sum`` the other statistics, and a statistic is
NaN unless all 24 hours hold a value (``complete_day_statistics``). The same rule is applied here to
ERA5-Land's own hours.

The hourly derivations follow the training columns, not the April prediction scripts:

* ``vpd`` and ``rh`` are computed PER HOUR from 2 m temperature and dewpoint, then aggregated. A daily
  deficit from the daily mean temperature and dewpoint (what ``process_era5land_cds.py`` did) reads low,
  because saturation vapour pressure is convex in temperature.
* ``ws`` is the mean of the hourly speed, not the speed of the mean wind vector.
* ``sw_in`` is the day's mean irradiance in W m-2, from the hourly downward shortwave in J m-2.
* ``ppfd_in`` is ``sw_in`` x :data:`src.env_derive.ppfd.PPFD_PER_SW_IN`, the archive's own factor.
* ``prcip/PET`` is the merge's daily ratio of the two daily sums, floored and clipped as the merge does.

The soil-moisture level is the daily mean of the hourly values, and the anomaly is that mean against the
per-cell climatology, as :mod:`src.feature_contract.rebuild_daily` writes the servable tree.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from src.env_derive.ext_rad import extraterrestrial_radiation
from src.env_derive.humidity import saturation_vapour_pressure, vapour_pressure_deficit
from src.env_derive.ppfd import PPFD_PER_SW_IN
from src.feature_contract.definitions import SWC_LAYERS, swc_anomaly_col, swc_level_col
from src.feature_contract.swvl_climatology import normalise_swvl

KELVIN_OFFSET = 273.15
SECONDS_PER_HOUR = 3600.0
HOURS_PER_DAY = 24
MINUTES_PER_DAY = 1440
MINUTES_PER_DEGREE_LONGITUDE = 4.0

#: The merge's ``prcip/PET``: ``pet = where(|pet| < 1e-10, 1e-10, |pet|)``, ``clip(precip / pet, 0, 10)``.
PET_FLOOR_M = 1e-10
PRECIP_PET_MAX = 10.0

#: ``ext_rad`` is averaged over instants this many minutes apart, the corpus's common env step.
EXT_RAD_STEP_MINUTES = 30

TEMPERATURE = "temperature_2m"
DEWPOINT = "dewpoint_temperature_2m"
WIND_U = "u_component_of_wind_10m"
WIND_V = "v_component_of_wind_10m"
SHORTWAVE = "surface_solar_radiation_downwards_hourly"
PRECIPITATION = "total_precipitation_hourly"
POTENTIAL_EVAPORATION = "potential_evaporation_hourly"
SURFACE_PRESSURE = "surface_pressure"
SOIL_TEMPERATURE = tuple(f"soil_temperature_level_{layer}" for layer in SWC_LAYERS)
SOIL_WATER = tuple(f"volumetric_soil_water_layer_{layer}" for layer in SWC_LAYERS)

#: ERA5-Land hourly band names, as ``ECMWF/ERA5_LAND/HOURLY`` and the site extract spell them.
HOURLY_BANDS = (
    TEMPERATURE,
    DEWPOINT,
    WIND_U,
    WIND_V,
    SHORTWAVE,
    PRECIPITATION,
    POTENTIAL_EVAPORATION,
    SURFACE_PRESSURE,
    *SOIL_TEMPERATURE,
    *SOIL_WATER,
)

#: Hourly driver columns whose daily minimum and maximum are base columns too.
WITH_EXTREMES = ("ta", "vpd", "rh")
#: Hourly driver columns whose daily sum is a base column (``<name>_sum``).
SUMMED = (PRECIPITATION, POTENTIAL_EVAPORATION)
#: Hourly driver columns that have no bare daily base column (only their sum is used).
SUM_ONLY = (POTENTIAL_EVAPORATION,)


def _utc_hours(index) -> pd.DatetimeIndex:
    hours = pd.DatetimeIndex(index)
    hours = hours.tz_localize("UTC") if hours.tz is None else hours.tz_convert("UTC")
    if hours.hasnans:
        raise ValueError("hourly ERA5-Land index holds NaT")
    if hours.has_duplicates:
        raise ValueError("hourly ERA5-Land needs one row per hour; the index has duplicates")
    off_hour = hours[hours != hours.floor("h")]
    if len(off_hour):
        raise ValueError(f"ERA5-Land hours sit on the hour; {len(off_hour)} do not, first {off_hour[0]}")
    return hours


def hourly_drivers(hourly: pd.DataFrame) -> pd.DataFrame:
    """ERA5-Land hourly bands -> hourly columns under the training names and units, index UTC hours.

    A naive index is read as UTC, the convention of the site extract.
    """
    missing = [band for band in HOURLY_BANDS if band not in hourly.columns]
    if missing:
        raise KeyError(f"hourly ERA5-Land frame lacks band(s) {missing}")
    frame = hourly.set_axis(_utc_hours(hourly.index), axis=0).sort_index()

    ta = frame[TEMPERATURE] - KELVIN_OFFSET
    dewpoint = frame[DEWPOINT] - KELVIN_OFFSET
    rh = (100.0 * saturation_vapour_pressure(dewpoint) / saturation_vapour_pressure(ta)).clip(0.0, 100.0)
    sw_in = (frame[SHORTWAVE] / SECONDS_PER_HOUR).clip(lower=0.0)

    columns = {
        "ta": ta,
        "vpd": vapour_pressure_deficit(ta, rh),
        "rh": rh,
        "ws": np.hypot(frame[WIND_U], frame[WIND_V]),
        "sw_in": sw_in,
        "ppfd_in": sw_in * PPFD_PER_SW_IN,
        PRECIPITATION: frame[PRECIPITATION],
        POTENTIAL_EVAPORATION: frame[POTENTIAL_EVAPORATION],
        SURFACE_PRESSURE: frame[SURFACE_PRESSURE],
    }
    for column in SOIL_TEMPERATURE:
        columns[column] = frame[column]
    for layer, column in zip(SWC_LAYERS, SOIL_WATER):
        columns[swc_level_col(layer)] = frame[column]
    return pd.DataFrame(columns, index=frame.index)


def daily_statistics(drivers: pd.DataFrame) -> pd.DataFrame:
    """One row per UTC day: means under the bare names, extremes, sums, and ``prcip/PET``.

    A statistic is NaN unless all 24 hours of that day hold a value of that column.
    """
    day = drivers.resample("D")
    complete = day.count().ge(HOURS_PER_DAY)
    means = day.mean().where(complete)
    minima = day.min().where(complete)
    maxima = day.max().where(complete)
    sums = day.sum(min_count=1).where(complete)

    out = means.drop(columns=list(SUM_ONLY))
    for column in WITH_EXTREMES:
        out[f"{column}_min"] = minima[column]
        out[f"{column}_max"] = maxima[column]
    for column in SUMMED:
        out[f"{column}_sum"] = sums[column]

    pet = out[f"{POTENTIAL_EVAPORATION}_sum"].abs()
    pet = pet.mask(pet < PET_FLOOR_M, PET_FLOOR_M)
    out["prcip/PET"] = (out[f"{PRECIPITATION}_sum"] / pet).clip(0.0, PRECIP_PET_MAX)
    return out


def add_swvl_anomaly(daily: pd.DataFrame, cells: Mapping) -> pd.DataFrame:
    """Return a copy with ``volumetric_soil_water_layer_L_zscore`` for every layer.

    ``cells`` maps each layer to an object with ``mean`` and ``sigma`` (a
    :class:`src.feature_contract.swvl_climatology.CellStats`).
    """
    out = daily.copy()
    for layer in SWC_LAYERS:
        if layer not in cells:
            raise KeyError(f"no climatology cell for soil-moisture layer {layer}")
        cell = cells[layer]
        _, zscore = normalise_swvl(out[swc_level_col(layer)], cell.sigma, cell.mean)
        out[swc_anomaly_col(layer)] = zscore
    return out


def daily_base_columns(hourly: pd.DataFrame, cells: Mapping) -> pd.DataFrame:
    """Every ERA5-Land base column of one place, one row per UTC day (index at UTC midnight)."""
    return add_swvl_anomaly(daily_statistics(hourly_drivers(hourly)), cells)


def _utc_days(days) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(days)
    index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    if index.hasnans:
        raise ValueError("day index holds NaT")
    if (index != index.normalize()).any():
        raise ValueError("days must sit at UTC midnight")
    return index


def _instants(days: pd.DatetimeIndex, offsets_minutes: np.ndarray) -> pd.DatetimeIndex:
    grid = days.tz_convert(None).values[:, None] + offsets_minutes.astype("timedelta64[m]")[None, :]
    return pd.DatetimeIndex(grid.ravel()).tz_localize("UTC")


def ext_rad_daily(days, latitude: float, longitude: float, step_minutes: int = EXT_RAD_STEP_MINUTES) -> pd.Series:
    """Daily mean extraterrestrial radiation, W m-2: :func:`extraterrestrial_radiation` averaged over the day.

    The merge's daily ``ext_rad`` is the mean of the site's sub-hourly column over the UTC day.
    """
    index = _utc_days(days)
    offsets = np.arange(0, MINUTES_PER_DAY, step_minutes)
    values = extraterrestrial_radiation(_instants(index, offsets), latitude, longitude).to_numpy(dtype=float)
    return pd.Series(values.reshape(len(index), len(offsets)).mean(axis=1), index=index, name="ext_rad")


def day_length_daily(
    days, latitude: float, longitude: float, elevation: float = 0.0, chunk_days: int = 366
) -> pd.Series:
    """Day length in hours, as the merge computes it.

    Solar elevation at the 1441 minute stamps from UTC midnight to the next midnight; the minutes above
    the horizon over 60, and exactly 24 or 0 when the sun never sets or never rises.
    """
    import pvlib  # noqa: PLC0415 -- heavy, and only this function needs it

    index = _utc_days(days)
    altitude = float(elevation) if np.isfinite(elevation) else 0.0
    offsets = np.arange(MINUTES_PER_DAY + 1)
    hours = np.empty(len(index), dtype=float)
    for start in range(0, len(index), chunk_days):
        chunk = index[start : start + chunk_days]
        position = pvlib.solarposition.get_solarposition(
            _instants(chunk, offsets), latitude, longitude, altitude=altitude
        )
        above = position["elevation"].to_numpy(dtype=float).reshape(len(chunk), len(offsets)) > 0
        values = above.sum(axis=1) / 60.0
        values[above.all(axis=1)] = 24.0
        values[~above.any(axis=1)] = 0.0
        hours[start : start + len(chunk)] = values
    return pd.Series(hours, index=index, name="day_length")


def solar_timestamp(days, longitude: float) -> pd.DatetimeIndex:
    """The apparent solar time of each day's first UTC hour: UTC + longitude x 4 min + equation of time.

    The daily tree's ``solar_TIMESTAMP`` is the solar stamp of the first merged hour, which is 00 UTC on
    a full day. The trainer reads it as the time index, so it shapes every cyclical time feature.
    """
    import pvlib  # noqa: PLC0415

    index = _utc_days(days)
    equation_of_time = pvlib.solarposition.equation_of_time_spencer71(index.dayofyear.to_numpy())
    offset = pd.to_timedelta(longitude * MINUTES_PER_DEGREE_LONGITUDE + np.asarray(equation_of_time), unit="m")
    return index + offset
