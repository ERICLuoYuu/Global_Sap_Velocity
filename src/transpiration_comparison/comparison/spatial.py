from __future__ import annotations

"""Spatial pattern comparison between products (Phase 3).

Implements:
A. Climatological maps + pixel-wise correlation
B. PFT-level spatial patterns
C. Zonal mean profiles
"""

import logging

import numpy as np
import pandas as pd
import xarray as xr

from ..config import PFT_COLUMNS

logger = logging.getLogger(__name__)


def pixel_temporal_correlation(
    ds_ref: xr.Dataset,
    ds_comp: xr.Dataset,
    var_ref: str = "sap_velocity_zscore",
    var_comp: str = "transpiration_zscore",
    min_overlap: int = 30,
) -> xr.Dataset:
    """Compute pixel-wise temporal Pearson correlation between two products.

    Parameters
    ----------
    ds_ref : xr.Dataset
        Reference product (our sap velocity).
    ds_comp : xr.Dataset
        Comparison product (transpiration).
    var_ref, var_comp : str
        Variable names (should be Z-score normalized).
    min_overlap : int
        Minimum overlapping timesteps required.

    Returns
    -------
    xr.Dataset
        Dataset with 'pearson_r' and 'p_value' per pixel.
    """
    logger.info(f"Computing pixel-wise temporal correlation: {var_ref} vs {var_comp}")

    try:
        import xskillscore as xs

        r = xs.pearson_r(ds_ref[var_ref], ds_comp[var_comp], dim="time")
        p = xs.pearson_r_p_value(ds_ref[var_ref], ds_comp[var_comp], dim="time")
        result = xr.Dataset({"pearson_r": r, "p_value": p})
    except ImportError:
        logger.warning("xskillscore not available, using manual computation")
        result = _manual_correlation(ds_ref[var_ref], ds_comp[var_comp])

    # Mask pixels with insufficient overlap
    overlap = (ds_ref[var_ref].notnull() & ds_comp[var_comp].notnull()).sum(dim="time")
    result = result.where(overlap >= min_overlap)

    n_valid = int((result["pearson_r"].notnull()).sum().values)
    mean_r = float(result["pearson_r"].mean().values)
    logger.info(f"  Valid pixels: {n_valid}, mean r: {mean_r:.3f}")

    return result


def spatial_rmse_map(
    ds_ref: xr.Dataset,
    ds_comp: xr.Dataset,
    var_ref: str = "sap_velocity_zscore",
    var_comp: str = "transpiration_zscore",
) -> xr.DataArray:
    """Pixel-wise RMSE of Z-scored time series."""
    logger.info("Computing pixel-wise RMSE...")
    try:
        import xskillscore as xs

        return xs.rmse(ds_ref[var_ref], ds_comp[var_comp], dim="time")
    except ImportError:
        diff = ds_ref[var_ref] - ds_comp[var_comp]
        return np.sqrt((diff**2).mean(dim="time"))


def pft_stratified_metrics(
    ds_ref: xr.Dataset,
    ds_comp: xr.Dataset,
    pft_map: xr.DataArray,
    var_ref: str = "sap_velocity_zscore",
    var_comp: str = "transpiration_zscore",
) -> pd.DataFrame:
    """Compute comparison metrics stratified by PFT.

    Returns DataFrame with rows=PFTs, columns=metrics.
    """
    logger.info("Computing PFT-stratified metrics...")

    # Pre-compute per-pixel temporal correlation map for correct PFT averaging
    corr_map = pixel_temporal_correlation(ds_ref, ds_comp, var_ref, var_comp)

    records = []
    for pft_code, pft_name in PFT_COLUMNS.items():
        mask = pft_map == pft_code
        n_pixels = int(mask.sum().values)
        if n_pixels < 10:
            logger.info(f"  {pft_code}: skipping ({n_pixels} pixels)")
            continue

        # Per-pixel temporal correlation, averaged within PFT
        pft_corr = corr_map["pearson_r"].where(mask)
        valid_corr = pft_corr.values[np.isfinite(pft_corr.values)]
        if len(valid_corr) < 10:
            continue
        pearson_r = float(np.mean(valid_corr))

        # RMSE, MAE, bias: still use pooled values (these are scale-dependent)
        ref_masked = ds_ref[var_ref].where(mask)
        comp_masked = ds_comp[var_comp].where(mask)
        diff = (comp_masked - ref_masked).values.ravel()
        valid_diff = diff[np.isfinite(diff)]
        if len(valid_diff) < 100:
            continue

        rmse = float(np.sqrt(np.mean(valid_diff ** 2)))
        mae = float(np.mean(np.abs(valid_diff)))
        bias = float(np.mean(valid_diff))

        records.append(
            {
                "pft": pft_code,
                "pft_name": pft_name,
                "pearson_r": pearson_r,
                "rmse": rmse,
                "mae": mae,
                "bias": bias,
                "n_pixels": n_pixels,
            }
        )
        logger.info(f"  {pft_code}: r={pearson_r:.3f}, rmse={rmse:.3f}, n={n_pixels}")

    return pd.DataFrame(records)


def zonal_mean_profile(
    datasets: dict[str, xr.Dataset],
    var_suffix: str = "_zscore",
    lat_bin_size: float = 5.0,
) -> pd.DataFrame:
    """Compute zonal (latitude-band) seasonal amplitude for multiple products.

    Uses the RAW variable (not Z-score, which has std=1 by definition).
    Seasonal amplitude = max(monthly_clim) - min(monthly_clim) per pixel.
    """
    logger.info(f"Computing zonal seasonal amplitude (bin size={lat_bin_size} deg)...")

    records = []
    for product_name, ds in datasets.items():
        # Use raw variable, not Z-score (Z-score std=1 by definition)
        raw_vars = [
            v for v in ds.data_vars
            if not v.endswith("_zscore") and not v.endswith("_anomaly") and not v.endswith("_clim")
            and v != "spatial_ref"
        ]
        if not raw_vars:
            logger.warning(f"  {product_name}: no raw variable found")
            continue
        var = raw_vars[0]

        # Monthly climatology -> seasonal amplitude per pixel
        try:
            monthly_clim = ds[var].groupby("time.month").mean(dim="time").compute()
            seasonal_amp = monthly_clim.max(dim="month") - monthly_clim.min(dim="month")
        except Exception as e:
            logger.warning(f"  {product_name}: climatology failed: {e}")
            continue

        if "lat" not in seasonal_amp.dims:
            logger.warning(f"  {product_name}: no lat dim in seasonal_amp, skipping")
            continue

        lat_bins = np.arange(-60, 80, lat_bin_size)
        for lat_start in lat_bins:
            lat_end = lat_start + lat_bin_size
            band = seasonal_amp.sel(lat=slice(lat_start, lat_end))
            if band.size == 0:
                continue
            mean_val = float(band.mean().values)
            if np.isfinite(mean_val):
                records.append(
                    {
                        "lat_center": lat_start + lat_bin_size / 2,
                        "product": product_name,
                        "seasonal_amplitude": mean_val,
                    }
                )

    return pd.DataFrame(records)


def _manual_correlation(a: xr.DataArray, b: xr.DataArray) -> xr.Dataset:
    """Manual pixel-wise Pearson correlation without xskillscore."""
    a_mean = a.mean(dim="time")
    b_mean = b.mean(dim="time")
    a_anom = a - a_mean
    b_anom = b - b_mean

    cov = (a_anom * b_anom).mean(dim="time")
    a_std = a.std(dim="time")
    b_std = b.std(dim="time")

    r = cov / (a_std * b_std)
    r = r.where(np.isfinite(r))

    return xr.Dataset({"pearson_r": r, "p_value": r * np.nan})  # p_value not computed
