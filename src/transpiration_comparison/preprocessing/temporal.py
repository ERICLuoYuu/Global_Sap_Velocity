from __future__ import annotations

"""Temporal aggregation utilities."""

import logging

import xarray as xr

logger = logging.getLogger(__name__)


def aggregate_to_daily(ds: xr.Dataset, source_freq: str, var: str = "transpiration") -> xr.Dataset:
    """Aggregate sub-daily or multi-day data to daily values.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with time coordinate.
    source_freq : str
        Source temporal frequency: 'hourly', '3-hourly', '8-day', 'daily'.
    var : str
        Variable name to aggregate.

    Returns
    -------
    xr.Dataset
        Daily-resolution dataset.
    """
    if source_freq == "daily":
        logger.info("Already daily, no temporal aggregation needed")
        return ds

    if source_freq in ("hourly", "3-hourly"):
        logger.info(f"Aggregating {source_freq} to daily mean...")
        ds_daily = ds.resample(time="1D").mean()
        logger.info(f"  {len(ds.time)} timesteps -> {len(ds_daily.time)} days")
        return ds_daily

    if source_freq == "8-day":
        logger.info("Interpolating 8-day composites to daily...")
        # Create daily time index spanning the data range
        time_daily = xr.cftime_range(
            start=str(ds.time.values[0])[:10],
            end=str(ds.time.values[-1])[:10],
            freq="D",
        )
        # Use pandas date range for compatibility
        import pandas as pd

        time_daily = pd.date_range(
            start=str(ds.time.values[0])[:10],
            end=str(ds.time.values[-1])[:10],
            freq="D",
        )
        ds_daily = ds.interp(time=time_daily, method="linear")
        logger.info(f"  {len(ds.time)} composites -> {len(ds_daily.time)} days")
        return ds_daily

    raise ValueError(f"Unknown source frequency: {source_freq}")


def compute_monthly_mean(ds: xr.Dataset, var: str = "transpiration") -> xr.Dataset:
    """Compute monthly means from daily data."""
    logger.info("Computing monthly means...")
    return ds.resample(time="1ME").mean()


def compute_annual_mean(ds: xr.Dataset, var: str = "transpiration") -> xr.Dataset:
    """Compute annual means from daily data."""
    logger.info("Computing annual means...")
    return ds.resample(time="1YE").mean()
