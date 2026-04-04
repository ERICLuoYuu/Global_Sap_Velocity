from __future__ import annotations

"""Temporal pattern comparison between products (Phase 4).

Implements:
A. Seasonal cycle comparison (monthly climatology correlation)
B. Interannual variability correlation
C. Trend agreement
"""

import logging

import numpy as np
import pandas as pd
import xarray as xr

from ..config import PFT_COLUMNS

logger = logging.getLogger(__name__)


def seasonal_cycle_comparison(
    datasets: dict[str, xr.Dataset],
    pft_map: xr.DataArray,
    var_suffix: str = "_clim",
) -> dict:
    """Compare monthly climatologies across products, stratified by PFT.

    Parameters
    ----------
    datasets : dict
        product_name -> Dataset with '{var}_clim' variable (12-month climatology).
    pft_map : xr.DataArray
        Dominant PFT per pixel.
    var_suffix : str
        Suffix identifying climatology variables.

    Returns
    -------
    dict with keys:
        'correlations': DataFrame of seasonal correlation per PFT per product pair
        'peak_months': DataFrame of peak month per PFT per product
        'seasonal_profiles': dict of product -> PFT -> 12-month profile array
    """
    logger.info("Computing seasonal cycle comparison...")

    # Find climatology variables per product
    product_clim = {}
    for name, ds in datasets.items():
        clim_vars = [v for v in ds.data_vars if v.endswith(var_suffix)]
        if clim_vars:
            product_clim[name] = ds[clim_vars[0]]

    if len(product_clim) < 2:
        raise ValueError(f"Need ≥2 products with climatology, found {len(product_clim)}")

    # Compute PFT-stratified seasonal profiles
    seasonal_profiles: dict[str, dict[str, np.ndarray]] = {}
    for prod_name, clim_da in product_clim.items():
        seasonal_profiles[prod_name] = {}
        for pft_code in PFT_COLUMNS:
            mask = pft_map == pft_code
            if mask.sum() < 10:
                continue
            # Average climatology over all pixels of this PFT
            pft_mean = clim_da.where(mask).mean(dim=["lat", "lon"])
            profile = pft_mean.values
            # Re-center to mean=0: Z-score offset can arise when
            # different pixels contribute different valid months
            profile = profile - np.nanmean(profile)
            seasonal_profiles[prod_name][pft_code] = profile

    # Compute pairwise seasonal correlations per PFT
    corr_records = []
    products = list(product_clim.keys())
    for i, p1 in enumerate(products):
        for p2 in products[i + 1 :]:
            for pft_code in PFT_COLUMNS:
                if pft_code not in seasonal_profiles.get(p1, {}) or pft_code not in seasonal_profiles.get(p2, {}):
                    continue
                s1 = seasonal_profiles[p1][pft_code]
                s2 = seasonal_profiles[p2][pft_code]
                valid = np.isfinite(s1) & np.isfinite(s2)
                if valid.sum() < 6:
                    continue
                r = float(np.corrcoef(s1[valid], s2[valid])[0, 1])
                corr_records.append({"product_1": p1, "product_2": p2, "pft": pft_code, "seasonal_r": r})

    # Peak month per product per PFT
    peak_records = []
    for prod_name, pft_profiles in seasonal_profiles.items():
        for pft_code, profile in pft_profiles.items():
            if np.all(np.isnan(profile)):
                continue
            peak_month = int(np.nanargmax(profile)) + 1  # 1-indexed
            peak_records.append({"product": prod_name, "pft": pft_code, "peak_month": peak_month})

    return {
        "correlations": pd.DataFrame(corr_records),
        "peak_months": pd.DataFrame(peak_records),
        "seasonal_profiles": seasonal_profiles,
    }


def interannual_correlation(
    ds_ref: xr.Dataset,
    ds_comp: xr.Dataset,
    var_ref: str,
    var_comp: str,
) -> xr.Dataset:
    """Pixel-wise correlation of annual anomalies.

    Aggregates to annual means, removes multi-year mean, then correlates.
    """
    logger.info("Computing interannual correlation...")

    ref_annual = ds_ref[var_ref].resample(time="1YE").mean()
    comp_annual = ds_comp[var_comp].resample(time="1YE").mean()

    # Remove mean to get anomalies
    ref_anom = ref_annual - ref_annual.mean(dim="time")
    comp_anom = comp_annual - comp_annual.mean(dim="time")

    # Correlation (need ≥3 years for meaningful correlation)
    n_years = len(ref_annual.time)
    if n_years < 3:
        logger.warning(f"Only {n_years} years -- interannual correlation unreliable")

    cov = (ref_anom * comp_anom).mean(dim="time")
    r = cov / (ref_anom.std(dim="time") * comp_anom.std(dim="time"))

    return xr.Dataset({"interannual_r": r.where(np.isfinite(r))})


def trend_agreement(
    datasets: dict[str, xr.Dataset],
    var_key: str = "_zscore",
) -> xr.Dataset:
    """Linear trend per pixel per product, and sign agreement map.

    Uses simple linear regression of annual Z-score means vs year.
    Defaults to Z-score variables for unit-consistent cross-product comparison.
    """
    logger.info("Computing trend agreement...")

    trends = {}
    for name, ds in datasets.items():
        # Find the Z-score variable (or fallback to first matching var_key)
        zscore_vars = [v for v in ds.data_vars if v.endswith(var_key)]
        var = zscore_vars[0] if zscore_vars else list(ds.data_vars)[0]
        annual = ds[var].resample(time="1YE").mean()
        n_years = len(annual.time)

        if n_years < 2:
            logger.warning(f"  {name}: only {n_years} years, skipping trend")
            continue

        # Linear regression: slope = cov(x,y) / var(x)
        years = np.arange(n_years, dtype=float)
        year_mean = years.mean()
        year_var = ((years - year_mean) ** 2).sum()

        # Broadcast: annual.values shape = (n_years, lat, lon)
        data = annual.values
        data_mean = np.nanmean(data, axis=0)
        slope = (
            np.nansum(
                (years[:, None, None] - year_mean) * (data - data_mean[None, :, :]),
                axis=0,
            )
            / year_var
        )

        trends[name] = xr.DataArray(
            slope,
            dims=["lat", "lon"],
            coords={"lat": annual.lat, "lon": annual.lon},
        )

    if len(trends) < 2:
        logger.warning("Need ≥2 products for trend agreement")
        return xr.Dataset()

    # Sign agreement: fraction of products with same trend sign
    sign_arrays = [np.sign(t.values) for t in trends.values()]
    sign_stack = np.stack(sign_arrays, axis=0)
    # Most common sign per pixel
    pos_count = (sign_stack > 0).sum(axis=0)
    neg_count = (sign_stack < 0).sum(axis=0)
    n_products = len(sign_arrays)
    agreement = np.maximum(pos_count, neg_count) / n_products

    sample_da = list(trends.values())[0]
    result = xr.Dataset(
        {
            "trend_agreement": xr.DataArray(
                agreement.astype(np.float32),
                dims=["lat", "lon"],
                coords={"lat": sample_da.lat, "lon": sample_da.lon},
            ),
        }
    )

    # Add individual trends
    for name, trend_da in trends.items():
        result[f"trend_{name}"] = trend_da.astype(np.float32)

    logger.info(f"  Mean trend agreement: {float(result['trend_agreement'].mean()):.2f}")
    return result
