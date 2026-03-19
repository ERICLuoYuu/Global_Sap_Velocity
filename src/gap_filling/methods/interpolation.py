"""Group A: Classical interpolation methods (no training required)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import interpolate as scipy_interp


def fill_linear(s: pd.Series) -> pd.Series:
    """Linear interpolation between known values."""
    return s.interpolate(method="linear", limit_direction="both").clip(lower=0)


def _find_gap_ranges(s: pd.Series) -> list:
    """Return list of (start_iloc, end_iloc) for each contiguous NaN block."""
    is_nan = s.isna().values.astype(np.int8)
    padded = np.concatenate([[0], is_nan, [0]])
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def fill_cubic(s: pd.Series) -> pd.Series:
    """Cubic spline interpolation scoped to local window around each gap."""
    gaps = _find_gap_ranges(s)
    if not gaps:
        return s.clip(lower=0)
    filled = s.copy()
    for g_start, g_end in gaps:
        gap_len = g_end - g_start
        margin = max(gap_len * 3, 72)
        lo = max(0, g_start - margin)
        hi = min(len(s), g_end + margin)
        window = s.iloc[lo:hi]
        try:
            interped = window.interpolate(method="spline", order=3, limit_direction="both")
            filled.iloc[g_start:g_end] = interped.iloc[g_start - lo : g_end - lo]
        except Exception:
            interped = window.interpolate(method="linear", limit_direction="both")
            filled.iloc[g_start:g_end] = interped.iloc[g_start - lo : g_end - lo]
    if filled.isna().any():
        filled = fill_linear(filled)
    return filled.clip(lower=0)


def fill_akima(s: pd.Series) -> pd.Series:
    """Akima interpolation — monotonicity-preserving, avoids overshooting.

    Extrapolation positions fall back to nearest-neighbour interpolation.
    """
    if s.notna().sum() < 4:
        return fill_linear(s)
    try:
        valid_idx = np.where(s.notna())[0]
        valid_vals = s.values[valid_idx]
        akima = scipy_interp.Akima1DInterpolator(valid_idx, valid_vals)
        filled = s.copy()
        null_pos = np.where(s.isna())[0]
        in_range = (null_pos >= valid_idx[0]) & (null_pos <= valid_idx[-1])
        if in_range.any():
            filled.iloc[null_pos[in_range]] = akima(null_pos[in_range])
        if (~in_range).any():
            filled = filled.interpolate(method="nearest", limit_direction="both")
        return filled.clip(lower=0)
    except Exception:
        return fill_linear(s)


def fill_nearest(s: pd.Series) -> pd.Series:
    """Nearest-neighbour interpolation."""
    return s.interpolate(method="nearest", limit_direction="both").clip(lower=0)
