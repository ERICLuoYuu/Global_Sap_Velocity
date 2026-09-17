"""Measure a site's clock from its own data. Nothing here moves a timestamp -- only the registry does.

Every estimator works on a regular 30-minute grid and reports the sign convention of
:mod:`src.clock_offsets.registry`: positive means the LABEL is LATE, so the true time is ``label - offset``.

``clear_day_lag``   sw_in against computed top-of-atmosphere radiation (TOA) on the clearest quarter of days,
                    where cloud cannot move the peak. Cross-correlation, refined between grid steps.
``edge_shift``      the centre of the observed daylight window against the geometric one, on a 24-hour circle,
                    with the threshold at 1 % of the record's p99. It reads the edges, not the curve's shape,
                    so it fails and agrees differently from the correlation -- which is why the gate asks both.
``pair_lag``        sap against the site's OWN sw_in: the sap clock minus the env clock, plus the
                    physiological delay of sap behind radiation, which the caller removes with a control
                    population (the delay is about 1 h globally and about 2 h at boreal sites, measured over
                    the corpus on 2026-09-16, so the population has to match the site).
``reference_lag``   sw_in against ERA5-Land hourly ``surface_solar_radiation_downwards``, whose accumulation
                    stamp is a constant the control population removes.

Each returns an :class:`Estimate`: a number with the sample it rests on, or the ``reason`` there is none --
a record too short, a dead pyranometer (USA_PAR_FER), a polar day with no edges (FIN_PET in June), a polar
night with no radiation. Readings at night are NOT treated as a fault: that is what a clock error looks like.

Ported from ``probes/radiation/probe_cloud_vs_clock.py`` and ``probe_clock_evidence.py``, validated there
against synthetic cloud regimes and then over the whole corpus (Palma 46756840, 46760679, 46760762).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from src.env_derive.ext_rad import extraterrestrial_radiation

STEP_HOURS = 0.5
MAX_SHIFT_HOURS = 9.0
MIN_PAIRS = 300
MIN_CLEAR_DAYS = 40
MIN_EDGE_DAYS = 30
EDGE_THRESHOLD_FRACTION = 0.01

#: A day's clearness below this is a dead or shaded sensor, not a cloudy site (USA_PAR_FER).
DEAD_SENSOR_CLEARNESS = 0.05

#: The clearest quarter of days carry the clear-day estimate.
CLEAR_QUANTILE = 0.75

#: A day needs this much of its lit hours to judge its clearness.
MIN_DAY_COVERAGE = 0.8

#: Below this the day has no night, so it has no edges to measure (polar day).
MIN_NIGHT_GAP_HOURS = 2.0


@dataclass(frozen=True)
class Estimate:
    """A measured offset in hours, or the reason there is none. Positive = the label is late."""

    hours: Optional[float] = None
    sample: int = 0
    quality: Optional[float] = None
    reason: str = ""

    def __bool__(self) -> bool:
        return self.hours is not None


def on_grid(series: pd.Series, step_hours: float = STEP_HOURS) -> pd.Series:
    """One value per ``step_hours`` on a UTC grid: sorted, de-duplicated, averaged into the step."""
    index = pd.DatetimeIndex(series.index)
    index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    values = pd.Series(pd.to_numeric(pd.Series(series.to_numpy()), errors="coerce").to_numpy(), index=index)
    values = values[values.index.notna()].sort_index()
    values = values[~values.index.duplicated(keep="first")]
    return values.resample(f"{int(round(step_hours * 60))}min").mean()


def toa_for(index, latitude: float, longitude: float) -> pd.Series:
    """Top-of-atmosphere radiation at each stamp (W m-2): the reference every env estimator measures against."""
    stamps = pd.DatetimeIndex(index)
    return pd.Series(np.asarray(extraterrestrial_radiation(stamps, latitude, longitude), dtype=float), index=stamps)


def correlation_lag(values, reference, step_hours: float = STEP_HOURS, min_pairs: int = MIN_PAIRS) -> Estimate:
    """The shift maximising the correlation of ``values`` with ``reference``; positive = ``values`` are late.

    Both arrays lie on one grid. The peak is refined between steps by a parabola through its neighbours, so a
    30-minute grid still resolves a quarter of an hour.
    """
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if values.size != reference.size:
        raise ValueError("a lag is measured between two series on one grid; got different lengths")
    steps = int(round(MAX_SHIFT_HOURS / step_hours))
    scores = np.full(2 * steps + 1, np.nan)
    pairs = np.zeros(2 * steps + 1, dtype=int)
    count = values.size
    for position, shift in enumerate(range(-steps, steps + 1)):
        if shift >= 0:
            left, right = values[shift:], reference[: count - shift]
        else:
            left, right = values[: count + shift], reference[-shift:]
        usable = np.isfinite(left) & np.isfinite(right)
        pairs[position] = int(usable.sum())
        if pairs[position] < min_pairs:
            continue
        if np.std(left[usable]) == 0.0 or np.std(right[usable]) == 0.0:
            continue
        scores[position] = np.corrcoef(left[usable], right[usable])[0, 1]
    if np.all(np.isnan(scores)):
        return Estimate(sample=int(pairs.max()), reason=f"fewer than {min_pairs} usable pairs at every shift")
    best = int(np.nanargmax(scores))
    hours = (best - steps + _between_steps(scores, best)) * step_hours
    return Estimate(hours=round(float(hours), 3), sample=int(pairs[best]), quality=round(float(scores[best]), 4))


def clear_day_lag(
    sw_in: pd.Series,
    latitude: float,
    longitude: float,
    reference_hours: float = 0.0,
    min_days: int = MIN_CLEAR_DAYS,
) -> Estimate:
    """sw_in against TOA over the clearest quarter of days.

    ``reference_hours`` is a rough estimate of the offset, from :func:`edge_shift` or an all-day correlation:
    the day frame is moved half way towards it, so that a badly shifted site still has whole days of daylight
    to select on. It cannot move the answer, only which days are called clear.
    """
    grid = on_grid(sw_in)
    toa = toa_for(grid.index, latitude, longitude)
    unusable = _sensor_reason(grid, toa)
    if unusable is not None:
        return unusable
    clearness = _day_clearness(grid, toa, longitude, reference_hours)
    if clearness.empty:
        return Estimate(reason="no day has both lit hours and 80 % coverage")
    clear = clearness.index[clearness >= clearness.quantile(CLEAR_QUANTILE)]
    if len(clear) < min_days:
        return Estimate(sample=int(len(clear)), reason=f"{len(clear)} clear days, fewer than {min_days}")
    days = _day_key(grid.index, longitude, reference_hours)
    inside = np.isin(days, clear)
    return correlation_lag(np.where(inside, grid.to_numpy(dtype=float), np.nan), toa.to_numpy(dtype=float))


def edge_shift(
    sw_in: pd.Series, latitude: float, longitude: float, min_days: int = MIN_EDGE_DAYS
) -> Estimate:
    """The centre of the observed daylight window against the geometric one, median over the days it can read."""
    grid = on_grid(sw_in)
    toa = toa_for(grid.index, latitude, longitude)
    unusable = _sensor_reason(grid, toa)
    if unusable is not None:
        return unusable
    threshold = EDGE_THRESHOLD_FRACTION * float(np.nanpercentile(grid.to_numpy(dtype=float), 99))
    frame = pd.DataFrame(
        {
            "sw_in": grid.to_numpy(dtype=float),
            "toa": toa.to_numpy(dtype=float),
            "hour": grid.index.hour + grid.index.minute / 60.0,
            "day": grid.index.floor("D"),
        }
    )
    shifts: List[float] = []
    for _day, part in frame.groupby("day", sort=False):
        geometric = _arc(part.loc[part["toa"] > 0.0, "hour"])
        observed = _arc(part.loc[part["sw_in"] > threshold, "hour"])
        if geometric is None or observed is None:
            continue
        if observed[2] < 0.9 * (observed[1] / STEP_HOURS + 1.0):
            continue  # a scattered set of lit steps is not one daylight window
        shifts.append(_circular_difference(observed[0], geometric[0]))
    if len(shifts) < min_days:
        return Estimate(
            sample=len(shifts),
            reason=f"{len(shifts)} days with a readable daylight window, fewer than {min_days}",
        )
    return Estimate(hours=round(float(np.median(shifts)), 3), sample=len(shifts))


def pair_lag(sap: pd.Series, sw_in: pd.Series) -> Estimate:
    """Sap against the site's own sw_in: the sap clock minus the env clock, plus the physiological delay."""
    sap_grid, sw_grid = on_grid(sap), on_grid(sw_in)
    index = sap_grid.index.union(sw_grid.index)
    return correlation_lag(
        sap_grid.reindex(index).to_numpy(dtype=float), sw_grid.reindex(index).to_numpy(dtype=float)
    )


def reference_lag(sw_in: pd.Series, era5_ssrd: pd.Series) -> Estimate:
    """sw_in against ERA5-Land's hourly downward shortwave, interpolated onto the grid."""
    grid = on_grid(sw_in)
    reference = on_grid(era5_ssrd).reindex(grid.index).interpolate(method="time", limit=2)
    return correlation_lag(grid.to_numpy(dtype=float), reference.to_numpy(dtype=float))


def _between_steps(scores: np.ndarray, best: int) -> float:
    """The sub-step position of the correlation peak, by a parabola through its two neighbours."""
    if best == 0 or best == len(scores) - 1:
        return 0.0
    low, high = scores[best - 1], scores[best + 1]
    if not (np.isfinite(low) and np.isfinite(high)):
        return 0.0
    curvature = low - 2.0 * scores[best] + high
    return 0.0 if curvature >= 0 else float(0.5 * (low - high) / curvature)


def _sensor_reason(grid: pd.Series, toa: pd.Series) -> Optional[Estimate]:
    """Why this record cannot be read at all: no radiation to compare against, or a sensor that is not one."""
    values = grid.to_numpy(dtype=float)
    lit = toa.to_numpy(dtype=float) > 0.0
    if not np.isfinite(values).any():
        return Estimate(reason="no sw_in reading")
    if not lit.any():
        return Estimate(reason="the sun never rises over this record (polar night)")
    energy = np.nansum(np.where(np.isfinite(values), values, 0.0))
    if energy <= 0.0:
        return Estimate(reason="sw_in is zero or negative throughout")
    clearness = energy / float(np.nansum(toa.to_numpy(dtype=float)))
    if clearness < DEAD_SENSOR_CLEARNESS:
        return Estimate(
            quality=round(float(clearness), 4),
            reason=f"clearness {clearness:.3f} over the record: a dead or shaded sensor, not a clock",
        )
    return None


def _day_key(index: pd.DatetimeIndex, longitude: float, reference_hours: float) -> np.ndarray:
    """The local day each stamp belongs to, moved half way towards a suspected offset."""
    half = (round(reference_hours * 2.0) / 2.0) / 2.0
    return (index + pd.Timedelta(hours=longitude / 15.0 - half)).floor("D").to_numpy()


def _day_clearness(grid: pd.Series, toa: pd.Series, longitude: float, reference_hours: float) -> pd.Series:
    """Each day's sw_in energy over its TOA energy, for days whose lit hours are 80 % present."""
    values = grid.to_numpy(dtype=float)
    reference = toa.to_numpy(dtype=float)
    frame = pd.DataFrame(
        {"sw_in": values, "toa": reference, "lit": reference > 0.0, "present": np.isfinite(values)},
        index=_day_key(grid.index, longitude, reference_hours),
    )
    by_day = frame.groupby(level=0)
    lit_hours = by_day["lit"].sum()
    covered = by_day.apply(lambda part: float((part["lit"] & part["present"]).sum()))
    toa_sum = by_day.apply(lambda part: float(part.loc[part["present"], "toa"].sum()))
    sw_sum = by_day.apply(lambda part: float(part.loc[part["present"], "sw_in"].sum()))
    usable = (lit_hours > 0) & (toa_sum > 0) & (covered >= MIN_DAY_COVERAGE * lit_hours)
    clearness = (sw_sum / toa_sum.where(toa_sum > 0))[usable]
    return clearness[np.isfinite(clearness) & (clearness > 0) & (clearness < 1.5)]


def _arc(hours) -> Optional[tuple]:
    """(centre, length, count) of the one daylight arc in a set of clock hours, on a 24-hour circle."""
    unique = np.unique(np.mod(np.asarray(hours, dtype=float), 24.0))
    if unique.size < 4:
        return None
    gaps = np.diff(np.concatenate([unique, [unique[0] + 24.0]]))
    widest = int(np.argmax(gaps))
    night = float(gaps[widest])
    if night < MIN_NIGHT_GAP_HOURS:
        return None
    length = 24.0 - night
    centre = float(np.mod(unique[(widest + 1) % unique.size] + length / 2.0, 24.0))
    return centre, length, int(unique.size)


def _circular_difference(first: float, second: float) -> float:
    return (first - second + 12.0) % 24.0 - 12.0
