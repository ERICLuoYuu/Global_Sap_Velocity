"""Daily env statistics from env's OWN hours, under an explicit support rule.

Until 2026-09-16 the merge took every env daily statistic over the hours in which sap AND env both held a
row, with no completeness rule at all. Two defects followed (Palma 46752793, 46757405, 46757568):

* The archive ships ``sw_in = NA`` at night at several sites, so a skip-NaN mean was a DAYLIGHT mean set
  against a 24-hour ``ext_rad``: +20 % to +97 % at eight sites, and 2,141 site-days on which surface
  shortwave exceeded top-of-atmosphere radiation.
* A day on which sap or env missed hours was averaged over whatever remained: ``ext_rad`` p95 errors of
  200-415 W m-2 at four sites, POL_TUC ``ta`` 2.4 K off its own complete-day mean.

The rule (user decision 2026-09-16, plan WP-B1):

* ``ta``, ``vpd``, ``rh``, ``ws`` -- mean, minimum and maximum over the hours holding a value, written only
  when at least :data:`MIN_ENV_HOURS` of the day's 24 do. ``_sum`` is the mean times the expected hours.
* ``sw_in``, ``ppfd_in`` -- a missing hour is a known ZERO when computed top-of-atmosphere radiation (TOA)
  is zero across ``[h - 30 min, h + 59 min]``; the margin covers loggers that stamp an interval's end. The
  day's mean is clearness-scaled, ``sum(present) / TOA(present) * TOA(day) / expected``, and written only
  when the present hours carry at least :data:`MIN_TOA_COVERAGE` of the day's TOA energy. On a complete day
  that is exactly the plain mean. Measured night values -- a pyranometer's small negative offsets included
  -- are kept; only a missing value is filled. ``_min`` / ``_max`` need :data:`MIN_ENV_HOURS` filled hours.
  A day with no TOA at all (polar night) is the plain mean of its filled hours: 0.0 with nothing measured.
* ``netrad`` is deliberately not covered: no feature, contract or serving path reads it (user, 2026-09-16).
* ``ext_rad`` is geometry, :func:`ext_rad_daily_statistics` -- the served definition, not the shipped column,
  which is displaced by 2-11 h at 15 sites.

Measured on the 2026-09-16 servable tree: rows with every core env feature valid 101.1 % of before (env's own
hours rescue days the sap join had emptied); knockout on complete days, sw_in error 1.3 % median (p95 7.2 %)
with two random daylight hours missing, against 4.2 % (p95 16 %) for night-fill-and-average; ``ta`` 0.15 K.

Under ``--daytime-only`` (``keep_hours``) a day expects only its kept hours; the hour threshold scales with
them and the TOA energy is the kept hours' energy.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.env_derive.ext_rad import extraterrestrial_radiation

RULE_VERSION = 1
MIN_ENV_HOURS = 20
MIN_TOA_COVERAGE = 0.8
HOURS_PER_DAY = 24
MINUTES_PER_DAY = 1440

ENV_CONTINUOUS_COLS = ("ta", "vpd", "rh", "ws")
ENV_RADIATION_COLS = ("sw_in", "ppfd_in")

#: Minutes after the hour sampled to decide an hour is dark: its window is [h - 30, h + 59].
DARK_SAMPLE_MINUTES = (-30, -15, 0, 15, 30, 45, 59)

#: Minutes after the hour whose TOA averages to the hour's TOA; a subset of :data:`DARK_SAMPLE_MINUTES`.
BODY_SAMPLE_MINUTES = (0, 15, 30, 45)

EXPECTED_HOURS_COL = "env_expected_hours"
EXT_RAD_COLUMNS = ("ext_rad", "ext_rad_min", "ext_rad_max", "ext_rad_sum")


def hourly_toa(days, latitude: float, longitude: float) -> Tuple[np.ndarray, np.ndarray]:
    """Each hour's TOA (W m-2, mean of :data:`BODY_SAMPLE_MINUTES`) and whether it is dark, as days x 24."""
    index = _utc_days(days)
    offsets = np.asarray(DARK_SAMPLE_MINUTES)
    minutes = (np.arange(HOURS_PER_DAY)[:, None] * 60 + offsets[None, :]).ravel()
    stamps = index.repeat(len(minutes)) + pd.to_timedelta(np.tile(minutes, len(index)), unit="min")
    values = extraterrestrial_radiation(stamps, latitude, longitude).to_numpy(dtype=float)
    values = values.reshape(len(index), HOURS_PER_DAY, len(offsets))
    toa = values[:, :, np.isin(offsets, BODY_SAMPLE_MINUTES)].mean(axis=2)
    dark = (values <= 0.0).all(axis=2)
    return toa, dark


def env_daily_statistics(
    env_hourly: pd.DataFrame,
    latitude: float,
    longitude: float,
    keep_hours: Optional[Callable] = None,
) -> pd.DataFrame:
    """Daily statistics of env's own hours, one row per UTC day that holds at least one env hour.

    ``env_hourly`` is ``aggregate_env_to_hourly`` output: one row per whole hour, naive (read as UTC) or
    tz-aware. Per covered column the result carries the mean under the bare name, ``_min``, ``_max``,
    ``_sum`` and ``_hours``; radiation adds ``_toa_coverage``. ``env_expected_hours`` is the hours each day
    expects. Values without the support the module docstring describes are NaN. The input is not modified.
    """
    frame = _hourly_frame(env_hourly)
    continuous = [col for col in ENV_CONTINUOUS_COLS if col in frame.columns]
    radiation = [col for col in ENV_RADIATION_COLS if col in frame.columns]
    columns = _output_columns(continuous, radiation)
    if frame.empty or not columns[:-1]:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], tz="UTC"), dtype=float)

    days = frame.index.floor("D").unique()
    grid = _hour_grid(days)
    kept = _kept_hours(grid, keep_hours, len(days))
    expected = kept.sum(axis=1)
    required = (expected * MIN_ENV_HOURS + HOURS_PER_DAY - 1) // HOURS_PER_DAY
    values = frame.reindex(grid)

    out: Dict[str, np.ndarray] = {}
    for col in continuous:
        out.update(_continuous(col, _day_hours(values[col]), kept, expected, required))
    if radiation:
        toa, dark = hourly_toa(days, latitude, longitude)
        for col in radiation:
            out.update(_radiation(col, _day_hours(values[col]), toa, dark, kept, expected, required))
    out[EXPECTED_HOURS_COL] = expected
    return pd.DataFrame(out, index=days)[columns]


def ext_rad_daily_statistics(
    days,
    latitude: float,
    longitude: float,
    keep_hours: Optional[Callable] = None,
) -> pd.DataFrame:
    """Daily ``ext_rad`` as it is SERVED, with ``_min`` / ``_max`` over the same instants, ``_sum`` = 24 x mean.

    ``days`` may repeat or carry a time of day (a plant-level frame's ``TIMESTAMP``); the result has one row
    per UTC day. The mean is :func:`src.serving.era5_daily.ext_rad_daily` itself, so the trained feature and
    the served one are the same function of the same instants.

    Under ``keep_hours`` (``--daytime-only``) every env statistic covers the kept hours only, and so does this
    one: the mean is the kept hours' TOA -- the denominator of the clearness-scaled ``sw_in`` -- over the hours
    the day expects, ``_min`` / ``_max`` run over those hours, ``_sum`` is the mean times the expected hours,
    and a day without a kept hour is missing. Nothing served is daylight-scoped, so there is no served value
    to match; daily ``sw_in / ext_rad`` stays the clearness of the hours the tree describes.
    """
    from src.serving.era5_daily import EXT_RAD_STEP_MINUTES, ext_rad_daily  # noqa: PLC0415 -- heavy import

    index = _utc_days(days)
    if len(index) == 0:
        return pd.DataFrame(columns=list(EXT_RAD_COLUMNS), index=index, dtype=float)
    if keep_hours is not None:
        return _kept_ext_rad(index, latitude, longitude, keep_hours)
    mean = ext_rad_daily(index, latitude, longitude).to_numpy(dtype=float)
    offsets = np.arange(0, MINUTES_PER_DAY, EXT_RAD_STEP_MINUTES)
    stamps = index.repeat(len(offsets)) + pd.to_timedelta(np.tile(offsets, len(index)), unit="min")
    instants = extraterrestrial_radiation(stamps, latitude, longitude).to_numpy(dtype=float)
    instants = instants.reshape(len(index), len(offsets))
    stats = {
        "ext_rad": mean,
        "ext_rad_min": instants.min(axis=1),
        "ext_rad_max": instants.max(axis=1),
        "ext_rad_sum": HOURS_PER_DAY * mean,
    }
    return pd.DataFrame(stats, index=index)[list(EXT_RAD_COLUMNS)]


def _kept_ext_rad(index: pd.DatetimeIndex, latitude: float, longitude: float, keep_hours: Callable) -> pd.DataFrame:
    toa, _dark = hourly_toa(index, latitude, longitude)
    kept = _kept_hours(_hour_grid(index), keep_hours, len(index))
    expected = kept.sum(axis=1)
    has_hours = expected > 0
    total = np.where(kept, toa, 0.0).sum(axis=1)
    mean = np.divide(total, expected, out=np.full(len(index), np.nan), where=has_hours)
    low, high = _extremes(toa, kept)
    stats = {
        "ext_rad": mean,
        "ext_rad_min": np.where(has_hours, low, np.nan),
        "ext_rad_max": np.where(has_hours, high, np.nan),
        "ext_rad_sum": mean * expected,
    }
    return pd.DataFrame(stats, index=index)[list(EXT_RAD_COLUMNS)]


def _hour_grid(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The top of every hour of ``days``, day by day: days x 24 stamps."""
    return days.repeat(HOURS_PER_DAY) + pd.to_timedelta(np.tile(np.arange(HOURS_PER_DAY), len(days)), unit="h")


def _utc_days(days) -> pd.DatetimeIndex:
    """Sorted unique UTC midnights of ``days``; naive input is UTC."""
    index = pd.DatetimeIndex(days)
    index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    return index.floor("D").unique().sort_values()


def _hourly_frame(env_hourly: pd.DataFrame) -> pd.DataFrame:
    index = pd.DatetimeIndex(env_hourly.index)
    index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    if index.has_duplicates:
        raise ValueError("env_daily_statistics needs one row per hour; the index has duplicates")
    if len(index) and not (index == index.floor("h")).all():
        raise ValueError("env_daily_statistics needs whole-hour stamps, as aggregate_env_to_hourly writes them")
    return env_hourly.set_axis(index, axis=0).sort_index()


def _kept_hours(grid: pd.DatetimeIndex, keep_hours, n_days: int) -> np.ndarray:
    if keep_hours is None:
        return np.ones((n_days, HOURS_PER_DAY), dtype=bool)
    return np.asarray(keep_hours(grid), dtype=bool).reshape(n_days, HOURS_PER_DAY)


def _day_hours(values: pd.Series) -> np.ndarray:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return array.reshape(-1, HOURS_PER_DAY)


def _output_columns(continuous: Sequence[str], radiation: Sequence[str]) -> List[str]:
    columns: List[str] = []
    for col in continuous:
        columns += [col, f"{col}_min", f"{col}_max", f"{col}_sum", f"{col}_hours"]
    for col in radiation:
        columns += [col, f"{col}_min", f"{col}_max", f"{col}_sum", f"{col}_hours", f"{col}_toa_coverage"]
    return columns + [EXPECTED_HOURS_COL]


def _extremes(kept_values: np.ndarray, present: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    low = np.where(present, kept_values, np.inf).min(axis=1)
    high = np.where(present, kept_values, -np.inf).max(axis=1)
    return low, high


def _continuous(col, hours, kept, expected, required) -> Dict[str, np.ndarray]:
    values = np.where(kept & np.isfinite(hours), hours, np.nan)
    present = np.isfinite(values)
    count = present.sum(axis=1)
    valid = (count >= required) & (count > 0)
    total = np.where(present, values, 0.0).sum(axis=1)
    mean = np.divide(total, count, out=np.full(len(count), np.nan), where=valid)
    low, high = _extremes(values, present)
    return {
        col: mean,
        f"{col}_min": np.where(valid, low, np.nan),
        f"{col}_max": np.where(valid, high, np.nan),
        f"{col}_sum": mean * expected,
        f"{col}_hours": count,
    }


def _radiation(col, hours, toa, dark, kept, expected, required) -> Dict[str, np.ndarray]:
    measured = np.isfinite(hours)
    filled = np.where(measured, hours, np.where(dark, 0.0, np.nan))
    values = np.where(kept, filled, np.nan)
    present = np.isfinite(values)
    count = present.sum(axis=1)
    total = np.where(present, values, 0.0).sum(axis=1)

    kept_toa = np.where(kept, toa, 0.0)
    energy_day = kept_toa.sum(axis=1)
    energy_present = np.where(present, kept_toa, 0.0).sum(axis=1)
    lit_day = energy_day > 0.0
    coverage = np.divide(energy_present, energy_day, out=np.ones(len(energy_day)), where=lit_day)

    by_hours = (count >= required) & (count > 0)
    scalable = lit_day & (energy_present > 0.0) & (coverage >= MIN_TOA_COVERAGE)
    ratio = np.divide(total, energy_present, out=np.full(len(total), np.nan), where=scalable)
    scaled = ratio * energy_day / np.where(expected > 0, expected, 1)
    plain = np.divide(total, count, out=np.full(len(total), np.nan), where=by_hours)
    mean = np.where(lit_day, np.where(scalable, scaled, np.nan), plain)

    low, high = _extremes(values, present)
    return {
        col: mean,
        f"{col}_min": np.where(by_hours, low, np.nan),
        f"{col}_max": np.where(by_hours, high, np.nan),
        f"{col}_sum": mean * expected,
        f"{col}_hours": count,
        f"{col}_toa_coverage": coverage,
    }
