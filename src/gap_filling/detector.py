"""Gap detection — find contiguous NaN runs in time series."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd


class Gap(NamedTuple):
    """A contiguous NaN gap in a time series."""

    start: pd.Timestamp
    end: pd.Timestamp
    size_hours: int


class GapDetector:
    """Identifies contiguous NaN runs and computes gap sizes in hours."""

    @staticmethod
    def detect(series: pd.Series) -> list[Gap]:
        """Find all contiguous NaN gaps in a Series with DatetimeIndex.

        Returns a sorted list of Gap(start, end, size_hours).
        """
        if len(series) == 0:
            return []

        is_nan = series.isna().values.astype(np.int8)
        if not is_nan.any():
            return []

        # Detect gap boundaries via diff on padded array
        padded = np.concatenate([[0], is_nan, [0]])
        diff = np.diff(padded)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1  # inclusive end

        # Infer frequency for size calculation
        freq = pd.infer_freq(series.index)
        freq_offset = pd.tseries.frequencies.to_offset(freq or "h")

        gaps = []
        idx = series.index
        for s, e in zip(starts, ends):
            start_ts = idx[s]
            end_ts = idx[e]
            size_td = end_ts - start_ts + freq_offset
            size_hours = max(1, int(round(size_td.total_seconds() / 3600)))
            gaps.append(Gap(start=start_ts, end=end_ts, size_hours=size_hours))

        return gaps
