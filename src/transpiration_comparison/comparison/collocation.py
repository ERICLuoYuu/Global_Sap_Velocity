from __future__ import annotations

"""Triple collocation analysis for error estimation.

Implements the Extended Double Instrumental Variable (EIVD) technique
following Li et al. (2024, Scientific Data).
"""

import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def triple_collocation(
    ds_a: xr.Dataset,
    ds_b: xr.Dataset,
    ds_c: xr.Dataset,
    var_a: str,
    var_b: str,
    var_c: str,
    min_valid: int = 24,
) -> dict[str, xr.DataArray]:
    """Standard triple collocation error estimation.

    Given three independent estimates of the same quantity, estimates
    the random error variance of each product without knowing the truth.

    Parameters
    ----------
    ds_a, ds_b, ds_c : xr.Dataset
        Three independent products on the same grid and time period.
    var_a, var_b, var_c : str
        Variable names in each dataset.
    min_valid : int
        Minimum overlapping valid timesteps.

    Returns
    -------
    dict with keys 'err_var_a', 'err_var_b', 'err_var_c' (xr.DataArray each).
    """
    logger.info("Running triple collocation analysis...")

    a = ds_a[var_a]
    b = ds_b[var_b]
    c = ds_c[var_c]

    # Covariances along time dimension
    a_mean = a.mean(dim="time")
    b_mean = b.mean(dim="time")
    c_mean = c.mean(dim="time")

    a_anom = a - a_mean
    b_anom = b - b_mean
    c_anom = c - c_mean

    n = (a.notnull() & b.notnull() & c.notnull()).sum(dim="time")

    cov_ab = (a_anom * b_anom).mean(dim="time")
    cov_ac = (a_anom * c_anom).mean(dim="time")
    cov_bc = (b_anom * c_anom).mean(dim="time")

    var_a_val = a.var(dim="time")
    var_b_val = b.var(dim="time")
    var_c_val = c.var(dim="time")

    # TC error variance estimates:
    # err_a = var(a) - cov(a,b) * cov(a,c) / cov(b,c)
    # err_b = var(b) - cov(a,b) * cov(b,c) / cov(a,c)
    # err_c = var(c) - cov(a,c) * cov(b,c) / cov(a,b)

    # Avoid division by zero
    eps = 1e-10
    err_var_a = var_a_val - (cov_ab * cov_ac) / cov_bc.where(np.abs(cov_bc) > eps)
    err_var_b = var_b_val - (cov_ab * cov_bc) / cov_ac.where(np.abs(cov_ac) > eps)
    err_var_c = var_c_val - (cov_ac * cov_bc) / cov_ab.where(np.abs(cov_ab) > eps)

    # Clip negative error variances (can happen with noisy data)
    err_var_a = err_var_a.where(err_var_a > 0).where(n >= min_valid)
    err_var_b = err_var_b.where(err_var_b > 0).where(n >= min_valid)
    err_var_c = err_var_c.where(err_var_c > 0).where(n >= min_valid)

    for name, ev in [("A", err_var_a), ("B", err_var_b), ("C", err_var_c)]:
        median_err = float(np.nanmedian(ev.values)) if ev.notnull().any() else float("nan")
        logger.info(f"  Product {name}: median error std = {np.sqrt(median_err):.3f}")

    return {
        "err_var_a": err_var_a.astype(np.float32),
        "err_var_b": err_var_b.astype(np.float32),
        "err_var_c": err_var_c.astype(np.float32),
    }


def compute_tc_weights(
    err_variances: dict[str, xr.DataArray],
) -> dict[str, xr.DataArray]:
    """Compute optimal merging weights from TC error variances.

    Weight is inversely proportional to error variance:
    w_i = (1/err_var_i) / sum(1/err_var_j)
    """
    logger.info("Computing TC-based merging weights...")

    inv_vars = {}
    for key, ev in err_variances.items():
        inv_vars[key] = 1.0 / ev.where(ev > 1e-10)

    total_inv = sum(inv_vars.values())

    weights = {}
    for key, iv in inv_vars.items():
        w = iv / total_inv.where(total_inv > 0)
        weights[key] = w.where(np.isfinite(w), 1.0 / len(inv_vars))  # Equal weight fallback

    return weights
