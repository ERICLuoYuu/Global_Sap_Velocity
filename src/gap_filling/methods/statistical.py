"""Group B: Statistical / climatological methods (no training required)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.gap_filling.methods.interpolation import fill_linear

logger = logging.getLogger(__name__)


def _is_hourly_series(s: pd.Series) -> bool:
    """Return True when the median time step is <= 2 h."""
    if len(s) < 2:
        return True
    median_dt = pd.Series(s.index).diff().dropna().median()
    return bool(median_dt <= pd.Timedelta("2h"))


def fill_mdv(s: pd.Series, window_days: int = 7) -> pd.Series:
    """Mean Diurnal Variation gap-filling (vectorized).

    Each missing timestep is replaced by the mean of observed values at the
    same hour-of-day (hourly) or day-of-week (daily) within +/-window_days.
    Residual NaNs fall back to linear interpolation.
    """
    filled = s.copy()
    null_mask = s.isna()
    if not null_mask.any():
        return filled.clip(lower=0)
    hourly = _is_hourly_series(s)
    period_key = s.index.hour if hourly else s.index.dayofweek
    window_td = pd.Timedelta(days=window_days)
    observed = s[s.notna()]
    obs_period = observed.index.hour if hourly else observed.index.dayofweek
    for period_val in pd.unique(period_key[null_mask]):
        missing_at_period = null_mask & (period_key == period_val)
        if not missing_at_period.any():
            continue
        obs_same_period = observed[obs_period == period_val]
        if obs_same_period.empty:
            continue
        missing_times = s.index[missing_at_period]
        obs_idx = obs_same_period.index
        obs_vals = obs_same_period.values
        for ts in missing_times:
            w_start = ts - window_td
            w_end = ts + window_td
            mask = (obs_idx >= w_start) & (obs_idx <= w_end)
            if mask.any():
                filled[ts] = max(0.0, float(np.mean(obs_vals[mask])))
    if filled.isna().any():
        filled = fill_linear(filled)
    return filled.clip(lower=0)


def fill_rolling_mean(s: pd.Series, window: int = None) -> pd.Series:
    """Rolling window mean gap-filling.

    Default window: 48 timesteps (hourly) or 7 (daily).
    """
    if window is None:
        window = 48 if _is_hourly_series(s) else 7
    rolling = s.rolling(window=window, min_periods=max(1, window // 4), center=True).mean()
    filled = s.fillna(rolling)
    if filled.isna().any():
        filled = fill_linear(filled)
    return filled.clip(lower=0)


def fill_stl(s: pd.Series) -> pd.Series:
    """STL decomposition + residual interpolation.

    Falls back to fill_rolling_mean on error or insufficient data.
    """
    try:
        from statsmodels.tsa.seasonal import STL

        if not s.isna().any():
            return s.clip(lower=0)
        period = 24 if _is_hourly_series(s) else 7
        if len(s) < period * 2:
            return fill_rolling_mean(s)
        s_pre = s.interpolate(method="linear", limit_direction="both")
        if s_pre.isna().any():
            s_pre = s_pre.ffill().bfill()
        if s_pre.isna().any():
            s_pre = s_pre.fillna(s_pre.mean() if not s_pre.isna().all() else 0.0)
        res = STL(s_pre, period=period, robust=True).fit()
        residual = pd.Series(res.resid, index=s.index)
        residual[s.isna()] = np.nan
        residual_filled = residual.interpolate(method="linear", limit_direction="both").fillna(0.0)
        return pd.Series(res.trend + res.seasonal + residual_filled.values, index=s.index).clip(lower=0)
    except Exception:
        logger.debug("STL decomposition failed, falling back to rolling mean", exc_info=True)
        return fill_rolling_mean(s)
