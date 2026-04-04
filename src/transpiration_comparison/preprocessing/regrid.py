from __future__ import annotations

"""Regrid datasets to a common 0.1-degree grid using xESMF."""

import logging

import numpy as np
import xarray as xr

from ..config import SpatialDomain, make_common_grid

logger = logging.getLogger(__name__)


def regrid_to_common(
    ds: xr.Dataset,
    target_resolution: float = 0.1,
    method: str | None = None,
    domain: SpatialDomain = SpatialDomain(),
) -> xr.Dataset:
    """Regrid a dataset to the common 0.1-degree grid.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with lat/lon coordinates.
    target_resolution : float
        Target grid resolution in degrees.
    method : str, optional
        Regridding method. Auto-selected if None:
        - 'bilinear' for upsampling (coarser -> finer, e.g. GLDAS 0.25 -> 0.1)
        - 'conservative' for downsampling (finer -> coarser, e.g. PMLv2 0.05 -> 0.1)
    domain : SpatialDomain
        Target spatial domain.

    Returns
    -------
    xr.Dataset
        Regridded dataset on common grid.
    """
    source_res = _estimate_resolution(ds)
    if method is None:
        method = "conservative" if source_res < target_resolution else "bilinear"

    logger.info(f"Regridding from ~{source_res:.3f} deg to {target_resolution} deg using {method} method")

    # Check if already on target grid
    if abs(source_res - target_resolution) < 0.01:
        logger.info("Already on target grid, skipping regrid")
        return _align_to_grid(ds, domain, target_resolution)

    try:
        import xesmf as xe
    except ImportError:
        logger.warning("xESMF not available, falling back to xarray interp")
        return _fallback_interp(ds, domain, target_resolution)

    # Build target grid
    target_lat, target_lon = make_common_grid(domain)
    ds_target = xr.Dataset({"lat": (["lat"], target_lat), "lon": (["lon"], target_lon)})

    # Create regridder (cached internally by xESMF)
    regridder = xe.Regridder(ds, ds_target, method, periodic=True)
    ds_out = regridder(ds)

    logger.info(f"Regridded: {dict(ds.dims)} -> {dict(ds_out.dims)}")
    return ds_out


def _estimate_resolution(ds: xr.Dataset) -> float:
    """Estimate the spatial resolution of a dataset."""
    if "lat" in ds.coords and len(ds.lat) > 1:
        return float(np.abs(np.diff(ds.lat.values[:2])[0]))
    return 0.1  # Default assumption


def _align_to_grid(ds: xr.Dataset, domain: SpatialDomain, resolution: float) -> xr.Dataset:
    """Align an already-correct-resolution dataset to exact grid centers."""
    target_lat, target_lon = make_common_grid(domain)
    return ds.interp(lat=target_lat, lon=target_lon, method="nearest")


def _fallback_interp(ds: xr.Dataset, domain: SpatialDomain, resolution: float) -> xr.Dataset:
    """Fallback interpolation using xarray when xESMF is unavailable."""
    logger.warning("Using xarray interpolation (less accurate than xESMF)")
    target_lat, target_lon = make_common_grid(domain)
    return ds.interp(lat=target_lat, lon=target_lon, method="linear")
