"""Does every row of `removal_log.csv` actually do something?

`RemovalLogProcessor._validate_entries` checks the log against *itself* --
missing fields, ``start > end``, unknown type.  It never asks whether a row
resolves to a real file, a real column, or an overlapping time range, and
``get_entries_for_site`` matches on exact equality with no report on a miss.  So
a row that matches nothing simply does not happen, silently.

Thirty-three of 371 entries were in that state when this was first measured, and
seven of them were *reviewed decisions that had stopped applying*.  Nothing in
the pipeline could have told anyone.  This module turns "does this row act?"
into a number that a run can fail on.

Design note: superseded rows are **computed**, never annotated.  An annotation
in the log would be an edit, and a stale one the moment `sensor_grouping`
changes.  The classifier reads `SITE_SENSOR_LAYOUTS` -- the same source of truth
the generator uses -- so it stays correct when the layouts do.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
from pathlib import Path

import pandas as pd


class RowStatus(enum.Enum):
    """What a log row will do when the pipeline reads it."""

    ACTS = "acts"
    #: Names a file that is not in the corpus.
    NO_SUCH_FILE = "no_such_file"
    #: File exists, but the named column does not.
    NO_SUCH_COLUMN = "no_such_column"
    #: Column exists, but the period does not overlap the data's time range.
    NO_OVERLAP = "no_overlap"
    #: Structurally incapable of acting -- blank or unknown removal_type.
    INERT_TYPE = "inert_type"


@dataclasses.dataclass(frozen=True)
class RowVerdict:
    entry_id: str
    location: str
    plant_type: str
    column: str
    status: RowStatus
    detail: str = ""

    @property
    def is_inert(self) -> bool:
        return self.status is not RowStatus.ACTS


VALID_TYPES = frozenset({"site_entire", "column_entire", "period"})


@functools.cache
def _header(path: Path):
    return tuple(pd.read_csv(path, nrows=0).columns)


@functools.cache
def _time_range(path: Path):
    """First and last TIMESTAMP of a data file.

    Reads only the timestamp column -- these files run to hundreds of
    thousands of rows and the validator visits every one named in the log.
    """
    stamps = pd.read_csv(path, usecols=["TIMESTAMP"], parse_dates=["TIMESTAMP"])["TIMESTAMP"]
    if stamps.empty:
        return None, None
    # Some corpus files carry a UTC offset in TIMESTAMP and some do not, while
    # the removal log's dates are always naive wall-clock. Comparing the two
    # raises TypeError, so the tz is dropped rather than assumed: this is a
    # coarse overlap test against a site's full time range, and no removal
    # window in the log is anywhere near tight enough for an hours-scale offset
    # to change the verdict.
    if stamps.dt.tz is not None:
        stamps = stamps.dt.tz_localize(None)
    return stamps.min(), stamps.max()


def validate_row(row, corpus_dir: Path) -> RowVerdict:
    """Classify one log row against the corpus."""
    location = str(row.get("location", "") or "")
    plant_type = str(row.get("plant_type", "") or "")
    column = str(row.get("column", "") or "")
    entry_id = str(row.get("entry_id", "") or "")
    removal_type = str(row.get("removal_type", "") or "").strip().lower()

    def verdict(status, detail=""):
        return RowVerdict(entry_id, location, plant_type, column, status, detail)

    if removal_type not in VALID_TYPES:
        return verdict(RowStatus.INERT_TYPE, f"removal_type={removal_type!r}")

    # location + "_" + plant_type reconstructs the filename stem exactly. The
    # 2-token split is arbitrary but injective: 0 key collisions across all 226
    # files, so the scheme itself is sound.
    data_path = corpus_dir / f"{location}_{plant_type}.csv"
    if not data_path.exists():
        return verdict(RowStatus.NO_SUCH_FILE, str(data_path.name))

    if removal_type == "site_entire":
        return verdict(RowStatus.ACTS)

    if column not in _header(data_path):
        return verdict(RowStatus.NO_SUCH_COLUMN, column)

    if removal_type == "column_entire":
        return verdict(RowStatus.ACTS)

    start = row.get("start_time")
    end = row.get("end_time")
    if pd.isna(start) or pd.isna(end):
        # A period row whose dates failed to parse is inert in the most
        # dangerous way: it looks complete. This is the shape that cost ten
        # CZE_STN rows when the log was reformatted.
        return verdict(RowStatus.NO_OVERLAP, "unparsed dates")

    first, last = _time_range(data_path)
    if first is None or end < first or start > last:
        return verdict(
            RowStatus.NO_OVERLAP,
            f"{start}..{end} vs data {first}..{last}",
        )
    return verdict(RowStatus.ACTS)


def validate_log(log_path, corpus_dir):
    """Classify every row. Returns a list of :class:`RowVerdict`."""
    log_path = Path(log_path)
    corpus_dir = Path(corpus_dir)
    frame = pd.read_csv(log_path, keep_default_na=False)
    for col in ("start_time", "end_time"):
        if col in frame.columns:
            # Per-value, for the reason documented in
            # RemovalLogProcessor._parse_datetimes_per_value: a whole-column
            # parse infers one format and NaTs every other spelling.
            frame[col] = frame[col].apply(lambda v: pd.to_datetime(v, errors="coerce") if str(v).strip() else pd.NaT)
    return [validate_row(row, corpus_dir) for _, row in frame.iterrows()]


def summarise(verdicts):
    """Counts per status, plus the inert rows themselves."""
    counts = {status.value: 0 for status in RowStatus}
    for verdict in verdicts:
        counts[verdict.status.value] += 1
    return {
        "total": len(verdicts),
        "acts": counts[RowStatus.ACTS.value],
        "inert": sum(1 for v in verdicts if v.is_inert),
        "by_status": counts,
        "inert_rows": [dataclasses.asdict(v) | {"status": v.status.value} for v in verdicts if v.is_inert],
    }
