from __future__ import annotations

"""Region-level deep dive analysis (Phase 5)."""

import logging

import numpy as np
import pandas as pd
import xarray as xr

from ..config import PFT_COLUMNS, REGIONS, Region

logger = logging.getLogger(__name__)


def regional_analysis(
    datasets: dict[str, xr.Dataset],
    region: Region,
    pft_map: xr.DataArray,
) -> dict:
    """Deep-dive analysis for a single region.

    Returns dict with:
        'time_series': DataFrame of area-mean normalized time series per product
        'seasonal_overlay': DataFrame of monthly climatology per product
        'pft_breakdown': DataFrame of PFT-level metrics
        'stats': dict of summary statistics
    """
    logger.info(
        f"Regional analysis: {region.name} ({region.lat_min}-{region.lat_max}N, {region.lon_min}-{region.lon_max}E)"
    )

    # Crop all datasets to region
    regional_ds = {}
    for name, ds in datasets.items():
        cropped = ds.sel(
            lat=slice(region.lat_min, region.lat_max),
            lon=slice(region.lon_min, region.lon_max),
        )
        if cropped.sizes.get("lat", 0) == 0 or cropped.sizes.get("lon", 0) == 0:
            logger.warning(f"  {name}: no data in region")
            continue
        regional_ds[name] = cropped

    if not regional_ds:
        return {
            "time_series": pd.DataFrame(),
            "seasonal_overlay": pd.DataFrame(),
            "pft_breakdown": pd.DataFrame(),
            "stats": {},
        }

    # Regional PFT map
    pft_regional = pft_map.sel(
        lat=slice(region.lat_min, region.lat_max),
        lon=slice(region.lon_min, region.lon_max),
    )

    # 1. Area-mean time series (Z-score)
    ts_records = []
    for name, ds in regional_ds.items():
        zscore_var = [v for v in ds.data_vars if v.endswith("_zscore")]
        if not zscore_var:
            continue
        area_mean = ds[zscore_var[0]].mean(dim=["lat", "lon"])
        for t, val in zip(area_mean.time.values, area_mean.values):
            if np.isfinite(val):
                ts_records.append({"time": t, "product": name, "zscore": float(val)})

    # 2. Seasonal overlay (monthly climatology)
    seasonal_records = []
    for name, ds in regional_ds.items():
        zscore_var = [v for v in ds.data_vars if v.endswith("_zscore")]
        if not zscore_var:
            continue
        monthly = ds[zscore_var[0]].groupby("time.month").mean(dim=["time", "lat", "lon"])
        for month, val in zip(monthly.month.values, monthly.values):
            if np.isfinite(val):
                seasonal_records.append({"month": int(month), "product": name, "zscore": float(val)})

    # 3. PFT breakdown within region
    pft_records = []
    for pft_code, pft_name in PFT_COLUMNS.items():
        mask = pft_regional == pft_code
        n_pixels = int(mask.sum().values)
        if n_pixels < 5:
            continue
        pft_records.append({"pft": pft_code, "pft_name": pft_name, "n_pixels": n_pixels})

    # 4. Summary stats
    stats = {
        "region": region.name,
        "n_products": len(regional_ds),
        "lat_range": f"{region.lat_min}-{region.lat_max}",
        "lon_range": f"{region.lon_min}-{region.lon_max}",
    }

    return {
        "time_series": pd.DataFrame(ts_records),
        "seasonal_overlay": pd.DataFrame(seasonal_records),
        "pft_breakdown": pd.DataFrame(pft_records),
        "stats": stats,
    }


def all_regional_analyses(
    datasets: dict[str, xr.Dataset],
    pft_map: xr.DataArray,
) -> dict[str, dict]:
    """Run regional analysis for all predefined regions."""
    results = {}
    for region_key, region in REGIONS.items():
        results[region_key] = regional_analysis(datasets, region, pft_map)
    return results
