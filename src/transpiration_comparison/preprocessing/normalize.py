from __future__ import annotations

"""Z-score normalization and climatology computation.

The core methodology following Bittencourt et al. (2023):
Compare temporal PATTERNS (not absolute values) by normalizing
each pixel's time series to Z-scores. This allows comparing
sap velocity (cm3/cm2/h) with transpiration (mm/day).
"""

import logging

import numpy as np
import xarray as xr

from ..config import MIN_VALID_TIMESTEPS

logger = logging.getLogger(__name__)


def zscore_normalize(
    ds: xr.Dataset,
    var: str,
    dim: str = "time",
    min_valid: int = MIN_VALID_TIMESTEPS,
) -> xr.Dataset:
    """Z-score normalize a variable along the time dimension per pixel.

    z = (x - mean) / std

    Pixels with fewer than min_valid non-NaN values are masked.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    var : str
        Variable name to normalize.
    dim : str
        Dimension to normalize along.
    min_valid : int
        Minimum valid (non-NaN) observations required.

    Returns
    -------
    xr.Dataset
        Dataset with additional '{var}_zscore' variable.
    """
    logger.info(f"Z-score normalizing '{var}' along '{dim}' (min_valid={min_valid})...")

    data = ds[var]
    mean = data.mean(dim=dim)
    std = data.std(dim=dim)

    # Mask pixels with insufficient data or zero variance
    valid_count = data.count(dim=dim)
    mask = (valid_count >= min_valid) & (std > 1e-10)

    zscore = (data - mean) / std
    zscore = zscore.where(mask)

    ds[f"{var}_zscore"] = zscore.astype(np.float32)
    ds[f"{var}_zscore"].attrs = {
        "long_name": f"Z-score normalized {var}",
        "units": "dimensionless",
        "min_valid_timesteps": min_valid,
    }

    n_valid = int(mask.sum().values)
    n_total = int(np.prod(mask.shape))
    logger.info(f"  Z-score: {n_valid}/{n_total} pixels valid ({100 * n_valid / n_total:.1f}%)")

    return ds


def compute_climatology(ds: xr.Dataset, var: str) -> xr.Dataset:
    """Compute monthly climatology (mean seasonal cycle).

    Returns dataset with 'month' dimension (1-12) and '{var}_clim' variable.
    """
    logger.info(f"Computing monthly climatology for '{var}'...")

    monthly_clim = ds[var].groupby("time.month").mean(dim="time")
    monthly_clim = monthly_clim.astype(np.float32)

    ds_clim = monthly_clim.to_dataset(name=f"{var}_clim")
    ds_clim.attrs = ds.attrs.copy()
    ds_clim.attrs["description"] = f"Monthly climatology of {var}"

    logger.info(f"  Climatology shape: {dict(ds_clim.dims)}")
    return ds_clim


def compute_anomaly(ds: xr.Dataset, var: str) -> xr.Dataset:
    """Compute anomaly = value - climatology.

    Removes the mean seasonal cycle to isolate interannual variability.
    """
    logger.info(f"Computing anomaly for '{var}'...")

    climatology = ds[var].groupby("time.month").mean(dim="time")
    anomaly = ds[var].groupby("time.month") - climatology

    ds[f"{var}_anomaly"] = anomaly.astype(np.float32)
    ds[f"{var}_anomaly"].attrs = {
        "long_name": f"Anomaly of {var} (value - climatology)",
        "units": ds[var].attrs.get("units", "unknown"),
    }

    logger.info(f"  Anomaly range: [{float(anomaly.min()):.3f}, {float(anomaly.max()):.3f}]")
    return ds


def normalize_seasonal_amplitude(ds: xr.Dataset, var: str) -> xr.Dataset:
    """Normalize seasonal cycle amplitude to [0, 1] per pixel.

    Useful for comparing seasonal phase agreement independent of amplitude.
    """
    logger.info(f"Normalizing seasonal amplitude for '{var}'...")

    data = ds[var]
    pix_min = data.min(dim="time")
    pix_max = data.max(dim="time")
    pix_range = pix_max - pix_min

    normalized = (data - pix_min) / pix_range.where(pix_range > 1e-10)

    ds[f"{var}_norm"] = normalized.astype(np.float32)
    ds[f"{var}_norm"].attrs = {
        "long_name": f"Amplitude-normalized {var}",
        "units": "dimensionless [0-1]",
    }
    return ds
