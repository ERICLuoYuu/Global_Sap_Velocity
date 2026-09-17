"""Apply reviewed clock offsets at the Stage-3 load boundary -- the one place a timestamp moves.

:func:`correct_clock` has exactly two callers, ``EnvironmentalAnalyzer._load_environmental_data`` and
``SapFlowAnalyzer._load_sapflow_data``, each right after a data file and its flags are read. Everything
downstream then sees true UTC -- the QC day/night strata, incomplete-day removal, the merge's hourly join,
every daily statistic -- and nothing downstream takes an offset, so none can be applied twice.

Period membership is judged on the ORIGINAL label, for every entry, before any row moves: two adjoining
periods can never pick up each other's rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.clock_offsets.registry import ClockOffset, offsets_for

#: Columns that hold a reading's time. ``TIMESTAMP_LOCAL`` is not among them: the sap loader derives it
#: from the corrected ``TIMESTAMP``.
DEFAULT_TIME_COLUMNS = ("TIMESTAMP", "solar_TIMESTAMP")

#: The label a period is judged on -- a column, or the index when it carries this name.
LABEL = "TIMESTAMP"

#: Suffixes that follow the full site code in corpus and Stage-3 file names.
DATA_SUFFIXES = (
    "_env_data_outliers_removed.csv",
    "_sapf_data_outliers_removed.csv",
    "_env_data.csv",
    "_sapf_data.csv",
    "_env_flags.csv",
    "_sapf_flags.csv",
)

NANOSECONDS_PER_HOUR = 3_600_000_000_000


def site_code(path) -> str:
    """The full site code of a corpus or Stage-3 file, read from its exact suffix.

    Never by token count or prefix: the loaders' two-token ``location`` is ``USA_ORN`` for both ORN twins,
    and a prefix match would read ``CHE_DAV`` into ``CHE_DAV_SEE``.
    """
    name = Path(path).name
    for suffix in DATA_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    raise ValueError(f"{name!r} is not a site data or flags file; expected one of {DATA_SUFFIXES}")


def correct_clock(
    frame: pd.DataFrame, entries: Iterable[ClockOffset], time_cols: Sequence[str] = DEFAULT_TIME_COLUMNS
) -> pd.DataFrame:
    """Return a copy of ``frame`` whose time columns are moved to true time, entry by entry.

    A row whose label -- the ``TIMESTAMP`` column, or the index when it is named ``TIMESTAMP`` -- falls in an
    entry's ``[start, end)`` has every column of ``time_cols`` it carries, and the index when that is one of
    them, moved by ``-offset_hours``. A naive label is read as UTC. NaT stays NaT and never matches a period.
    Only ``applied`` entries are accepted, and a row two entries claim is refused rather than moved twice.
    ``frame`` itself is never modified.
    """
    entries = tuple(entries)
    for entry in entries:
        if entry.status != "applied":
            raise ValueError(
                f"only applied offsets move timestamps; {entry.site} {entry.component} is {entry.status!r}"
            )
    out = frame.copy()
    if not entries:
        return out
    labels = _labels(frame)
    columns = [col for col in time_cols if col in frame.columns]
    moves_index = frame.index.name in time_cols and frame.index.name not in frame.columns
    for col in columns:
        _require_datetimes(frame[col], col)
    shift = _row_shifts(labels, entries)
    for col in columns:
        out[col] = frame[col] + _as_unit(shift, frame[col].dtype)
    if moves_index:
        out.index = pd.DatetimeIndex(frame.index + _as_unit(shift, frame.index.dtype), name=frame.index.name)
    return out


def _labels(frame: pd.DataFrame) -> pd.DatetimeIndex:
    if LABEL in frame.columns:
        return _require_datetimes(frame[LABEL], LABEL)
    if frame.index.name == LABEL:
        return _require_datetimes(frame.index, LABEL)
    raise ValueError(f"a clock offset is judged on {LABEL!r}, and this frame has no such column or index")


def _require_datetimes(values, name: str) -> pd.DatetimeIndex:
    if not pd.api.types.is_datetime64_any_dtype(values):
        raise TypeError(f"{name} must be parsed to datetimes before a clock offset is applied, got {values.dtype}")
    return pd.DatetimeIndex(values)


def _row_shifts(labels: pd.DatetimeIndex, entries) -> pd.TimedeltaIndex:
    shift = np.zeros(len(labels), dtype="int64")
    claimed = np.zeros(len(labels), dtype=bool)
    for entry in entries:
        inside = _period_mask(labels, entry)
        if (claimed & inside).any():
            raise ValueError(f"{entry.site} {entry.component}: offset periods overlap; validate the registry first")
        claimed |= inside
        shift[inside] = -int(round(float(entry.offset_hours) * NANOSECONDS_PER_HOUR))
    return pd.to_timedelta(shift, unit="ns")


def _period_mask(labels: pd.DatetimeIndex, entry: ClockOffset) -> np.ndarray:
    inside = np.asarray(labels.notna())
    start = _label_time(entry.start, labels)
    end = _label_time(entry.end, labels)
    if start is not None:
        inside &= np.asarray(labels >= start)
    if end is not None:
        inside &= np.asarray(labels < end)
    return inside


def _label_time(value, labels: pd.DatetimeIndex):
    """A registry bound as a Timestamp comparable with ``labels``: aware UTC, or naive UTC for naive labels."""
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp if labels.tz is not None else stamp.tz_localize(None)


def _as_unit(shift: pd.TimedeltaIndex, dtype) -> np.ndarray:
    """The shifts in the resolution of ``dtype``, so a column keeps its own datetime unit."""
    unit = getattr(dtype, "unit", None) or np.datetime_data(dtype)[0]
    return shift.as_unit(unit).to_numpy()


def correct_loaded_frames(path, component: str, data: pd.DataFrame, flags, registry):
    """The loaders' single call: move a file's data and its flags together by the site's applied offsets.

    ``path`` is the file the loader read; its full site code picks the entries, so the ORN twins never share
    one. When the site has no applied entry for ``component`` the very objects passed in come back, and that
    site's Stage-3 output stays byte-identical to what it was before the registry existed.
    """
    site = site_code(path)
    entries = offsets_for(site, component, registry)
    if not entries:
        return data, flags
    moved = "; ".join(f"{entry.offset_hours:+g} h in [{entry.start}, {entry.end})" for entry in entries)
    print(f"Clock offset: {site} {component} labels corrected, true = label - offset: {moved}")
    return correct_clock(data, entries), (None if flags is None else correct_clock(flags, entries))
