import numpy as np
import pandas as pd
import pytest


def _make_series(values, freq="h"):
    """Helper: create a pd.Series with DatetimeIndex."""
    idx = pd.date_range("2020-01-01", periods=len(values), freq=freq)
    return pd.Series(values, index=idx)


def test_no_gaps():
    from src.gap_filling.detector import GapDetector

    s = _make_series([1.0, 2.0, 3.0, 4.0])
    gaps = GapDetector.detect(s)
    assert gaps == []


def test_single_gap():
    from src.gap_filling.detector import GapDetector

    vals = [1.0, np.nan, np.nan, np.nan, 5.0]
    s = _make_series(vals)
    gaps = GapDetector.detect(s)
    assert len(gaps) == 1
    assert gaps[0].size_hours == 3
    assert gaps[0].start == s.index[1]
    assert gaps[0].end == s.index[3]


def test_multiple_gaps():
    from src.gap_filling.detector import GapDetector

    vals = [1.0, np.nan, 3.0, np.nan, np.nan, 6.0]
    s = _make_series(vals)
    gaps = GapDetector.detect(s)
    assert len(gaps) == 2
    assert gaps[0].size_hours == 1
    assert gaps[1].size_hours == 2


def test_gap_at_start():
    from src.gap_filling.detector import GapDetector

    vals = [np.nan, np.nan, 3.0, 4.0]
    s = _make_series(vals)
    gaps = GapDetector.detect(s)
    assert len(gaps) == 1
    assert gaps[0].size_hours == 2


def test_gap_at_end():
    from src.gap_filling.detector import GapDetector

    vals = [1.0, 2.0, np.nan, np.nan]
    s = _make_series(vals)
    gaps = GapDetector.detect(s)
    assert len(gaps) == 1
    assert gaps[0].size_hours == 2


def test_all_nan():
    from src.gap_filling.detector import GapDetector

    vals = [np.nan, np.nan, np.nan]
    s = _make_series(vals)
    gaps = GapDetector.detect(s)
    assert len(gaps) == 1
    assert gaps[0].size_hours == 3


def test_daily_frequency():
    from src.gap_filling.detector import GapDetector

    vals = [1.0, np.nan, np.nan, 4.0]
    s = _make_series(vals, freq="D")
    gaps = GapDetector.detect(s)
    assert len(gaps) == 1
    assert gaps[0].size_hours == 48  # 2 days = 48 hours


def test_empty_series():
    from src.gap_filling.detector import GapDetector

    s = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
    gaps = GapDetector.detect(s)
    assert gaps == []


def test_sorted_by_start():
    from src.gap_filling.detector import GapDetector

    vals = [np.nan, 2.0, np.nan, np.nan, 5.0, np.nan]
    s = _make_series(vals)
    gaps = GapDetector.detect(s)
    assert len(gaps) == 3
    starts = [g.start for g in gaps]
    assert starts == sorted(starts)
