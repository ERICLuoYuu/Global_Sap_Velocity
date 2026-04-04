from __future__ import annotations

"""Product agreement analysis (Phase 6)."""

import logging

import numpy as np
import pandas as pd
import xarray as xr

from ..config import PFT_COLUMNS

logger = logging.getLogger(__name__)


def consensus_map(
    correlation_maps: dict[str, xr.Dataset],
) -> xr.Dataset:
    """Identify consensus zones where most products agree with sap velocity.

    Parameters
    ----------
    correlation_maps : dict
        product_name -> Dataset with 'pearson_r' variable (from spatial comparison).

    Returns
    -------
    xr.Dataset with:
        'mean_r': mean correlation across products
        'std_r': std of correlations
        'n_agree': number of products with r > 0.5
        'consensus_zone': categorical (high/medium/low agreement)
    """
    logger.info("Computing consensus map...")

    r_arrays = []
    for name, ds in correlation_maps.items():
        if "pearson_r" in ds.data_vars:
            r_arrays.append(ds["pearson_r"])

    if not r_arrays:
        raise ValueError("No correlation maps provided")

    stacked = xr.concat(r_arrays, dim="product")

    mean_r = stacked.mean(dim="product")
    std_r = stacked.std(dim="product")
    n_agree = (stacked > 0.5).sum(dim="product")

    # Consensus zones
    n_products = len(r_arrays)
    high = n_agree >= (n_products * 0.75)  # >=75% products agree
    medium = (n_agree >= (n_products * 0.5)) & ~high
    low = ~high & ~medium

    consensus = xr.where(high, 3, xr.where(medium, 2, 1))
    consensus = consensus.where(mean_r.notnull())

    result = xr.Dataset(
        {
            "mean_r": mean_r.astype(np.float32),
            "std_r": std_r.astype(np.float32),
            "n_agree": n_agree.astype(np.int8),
            "consensus_zone": consensus.astype(np.int8),
        }
    )

    for level, label in [(3, "high"), (2, "medium"), (1, "low")]:
        n = int((consensus == level).sum().values)
        logger.info(f"  {label} consensus: {n} pixels")

    return result


def product_ranking(
    pft_metrics: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Rank products per PFT based on comparison metrics.

    Parameters
    ----------
    pft_metrics : dict
        product_name -> DataFrame from pft_stratified_metrics().

    Returns
    -------
    DataFrame with product rankings per PFT.
    """
    logger.info("Computing product rankings...")

    records = []
    # Collect all metrics by PFT
    pft_data: dict[str, list] = {}
    for prod_name, df in pft_metrics.items():
        for _, row in df.iterrows():
            pft = row["pft"]
            if pft not in pft_data:
                pft_data[pft] = []
            pft_data[pft].append(
                {
                    "product": prod_name,
                    "pearson_r": row["pearson_r"],
                    "rmse": row["rmse"],
                    "n_pixels": row["n_pixels"],
                }
            )

    for pft, entries in pft_data.items():
        if len(entries) < 2:
            continue
        # Rank by pearson_r (higher = better)
        sorted_by_r = sorted(entries, key=lambda x: x["pearson_r"], reverse=True)
        for rank, entry in enumerate(sorted_by_r, 1):
            records.append(
                {
                    "pft": pft,
                    "product": entry["product"],
                    "rank_by_r": rank,
                    "pearson_r": entry["pearson_r"],
                    "rmse": entry["rmse"],
                }
            )

    return pd.DataFrame(records)


def unique_signal_detection(
    datasets: dict[str, xr.Dataset],
    pft_map: xr.DataArray,
    var_suffix: str = "_zscore",
) -> pd.DataFrame:
    """Identify where sap velocity shows unique patterns not seen in transpiration products.

    Computes mean inter-product correlation vs sap velocity correlation per pixel.
    Pixels where sapv correlation is low but inter-product correlation is high
    indicate unique sap velocity signals.
    """
    logger.info("Detecting unique sap velocity signals...")

    # Separate sap velocity from transpiration products
    sapv_ds = None
    transp_ds = {}
    for name, ds in datasets.items():
        zscore_vars = [v for v in ds.data_vars if v.endswith(var_suffix)]
        if not zscore_vars:
            continue
        if "sap" in name.lower():
            sapv_ds = ds[zscore_vars[0]]
        else:
            transp_ds[name] = ds[zscore_vars[0]]

    if sapv_ds is None or len(transp_ds) < 2:
        logger.warning("Need sap velocity + ≥2 transpiration products")
        return pd.DataFrame()

    records = []
    for pft_code in PFT_COLUMNS:
        mask = pft_map == pft_code
        if mask.sum() < 10:
            continue

        sapv_pft = sapv_ds.where(mask).mean(dim=["lat", "lon"])

        # Mean correlation of each transp product with sapv
        sapv_corrs = []
        for tname, tda in transp_ds.items():
            t_pft = tda.where(mask).mean(dim=["lat", "lon"])
            valid = sapv_pft.notnull() & t_pft.notnull()
            if valid.sum() < 30:
                continue
            r = float(np.corrcoef(sapv_pft.where(valid).values, t_pft.where(valid).values)[0, 1])
            sapv_corrs.append(r)

        # Mean inter-product correlation
        inter_corrs = []
        t_names = list(transp_ds.keys())
        for i in range(len(t_names)):
            for j in range(i + 1, len(t_names)):
                t1 = transp_ds[t_names[i]].where(mask).mean(dim=["lat", "lon"])
                t2 = transp_ds[t_names[j]].where(mask).mean(dim=["lat", "lon"])
                valid = t1.notnull() & t2.notnull()
                if valid.sum() < 30:
                    continue
                r = float(np.corrcoef(t1.where(valid).values, t2.where(valid).values)[0, 1])
                inter_corrs.append(r)

        if sapv_corrs and inter_corrs:
            records.append(
                {
                    "pft": pft_code,
                    "mean_sapv_corr": np.mean(sapv_corrs),
                    "mean_inter_corr": np.mean(inter_corrs),
                    "agreement_excess": np.mean(sapv_corrs) - np.mean(inter_corrs),
                }
            )

    return pd.DataFrame(records)
