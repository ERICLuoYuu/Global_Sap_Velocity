"""How often a column really logs, and how much of one of its samples each row of a shared grid stands for.

A site file holds every tree on one timestamp grid, and that grid can be finer than a tree's logger: HUN_SIK sits on
10 minutes with values only on the hour, CRI_TAM_TOW on 5 minutes with hourly trees, and MEX_VER_BSJ interleaves a
10-minute and a 15-minute grid. Counting grid rows as expected samples then reads a complete hourly day as 17 %
complete. Stage 3's incomplete-day rule weighs both sides of its ratio in samples at the column's own step instead:
the daytime rows with ``sample_weights``, the valid values with ``valid_sample_weights``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The coarsest logging interval taken as a cadence. A longer most-common gap means a column thinned to a few values
# a day, not a logger, and the column is then judged by rows as before.
MAX_STEP_MINUTES = 120.0


def _gaps_in_minutes(index: pd.DatetimeIndex) -> np.ndarray:
    """Time from each row to the next, rounded to whole minutes (clock jitter of seconds is not a step).

    Measured as timedeltas, so the index's storage unit (ns, us, s) does not matter.
    """
    gaps = (index[1:] - index[:-1]) / pd.Timedelta(minutes=1)
    return np.round(np.asarray(gaps, dtype=float))


def column_step_minutes(index: pd.DatetimeIndex, values: pd.Series) -> float:
    """The most common gap, in whole minutes, between consecutive valid values.

    NaN with fewer than two values, or when that gap exceeds ``MAX_STEP_MINUTES``. The mode, not the median: a logger
    that loses most of its values at random still has its own step as the most common gap, while the median drifts to
    two or three steps and would excuse the missing values. A tie goes to the shorter step, which expects more
    samples. Gaps that round to zero (rows seconds apart) are not steps.
    """
    present = index[pd.notna(np.asarray(values))]
    if len(present) < 2:
        return float("nan")
    gaps = _gaps_in_minutes(present)
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        return float("nan")
    steps, counts = np.unique(gaps, return_counts=True)
    step = float(steps[np.argmax(counts)])
    return step if step <= MAX_STEP_MINUTES else float("nan")


def sample_weights(index: pd.DatetimeIndex, step_minutes: float) -> pd.Series:
    """Each row's share of one sample at ``step_minutes``: time to the next row, capped at the step, over the step.

    On a grid at the column's own step every row weighs exactly 1, so the expected samples are the rows, as before.
    On a finer grid the rows of one sample add up to 1, and a gap in the grid never makes the row before it count for
    more than one sample. The last row takes the gap before it. Without a step (NaN), or with one row, each row
    weighs 1.
    """
    if len(index) < 2 or not np.isfinite(step_minutes) or step_minutes <= 0:
        return pd.Series(1.0, index=index)
    gaps = _gaps_in_minutes(index)
    durations = np.append(gaps, gaps[-1])
    return pd.Series(np.clip(durations, 0.0, step_minutes) / step_minutes, index=index)


def valid_sample_weights(index: pd.DatetimeIndex, values: pd.Series, step_minutes: float) -> pd.Series:
    """Each valid value's share of one sample, the same units as ``sample_weights``; 0 where the value is missing.

    A value weighs the time to the next valid value, capped at the step, over the step. Values at the column's step
    or sparser weigh exactly 1 (so on a regular grid the count of valid values is unchanged), while values logged
    finer than the step, or pairs seconds apart, share one sample instead of counting once each.
    """
    present = pd.notna(np.asarray(values))
    weights = np.zeros(len(index))
    weights[present] = sample_weights(index[present], step_minutes).to_numpy()
    return pd.Series(weights, index=index)
