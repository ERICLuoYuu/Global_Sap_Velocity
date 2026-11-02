"""One served cell's daily frame from the store's arrays, with the merged daily tree's column names and units.

The frame is what :func:`src.serving.engineer.engineer_cell` takes. ERA5-Land columns come from Stage A,
the soil-moisture anomaly from the cell's climatology (as :func:`normalise_swvl` computes it for a site),
the solar columns and LAI from Stage B, and the statics, PFT and coordinates from ``cells.parquet``.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from src.feature_contract.definitions import SWC_LAYERS, swc_anomaly_col, swc_level_col
from src.feature_contract.swvl_climatology import MIN_VALID_SIGMA
from src.serving.base_columns import DAY_COLUMN, PFT_COLUMN, STATIC_COLUMNS, TIME_COLUMN
from src.serving.era5_daily import solar_timestamp


def swvl_zscore(raw, mean: float, sigma: float) -> np.ndarray:
    """``(raw - mean) / sigma``; all NaN when the cell has no valid climatology (``sigma`` NaN or too small)."""
    values = np.asarray(raw, dtype=float)
    if not (np.isfinite(mean) and np.isfinite(sigma) and sigma >= MIN_VALID_SIGMA):
        return np.full(values.shape, np.nan)
    return (values - mean) / sigma


def cell_frame(days: pd.DatetimeIndex, series: Mapping[str, np.ndarray], cell: Mapping) -> pd.DataFrame:
    """The daily frame of one cell.

    ``days``: UTC midnights. ``series``: base column -> values over ``days`` (ERA5-Land, solar, LAI).
    ``cell``: that cell's ``cells.parquet`` row as a mapping (latitude, longitude, pft, swvl_mean_L,
    swvl_sigma_L, and whichever statics were built).
    """
    n_days = len(days)
    columns = {}
    for name, values in series.items():
        values = np.asarray(values, dtype=float)
        if values.shape != (n_days,):
            raise ValueError(f"{name}: {values.shape} values for {n_days} days")
        columns[name] = values
    for layer in SWC_LAYERS:
        level = swc_level_col(layer)
        if level in columns:
            columns[swc_anomaly_col(layer)] = swvl_zscore(
                columns[level], float(cell[f"swvl_mean_{layer}"]), float(cell[f"swvl_sigma_{layer}"])
            )
    for name in STATIC_COLUMNS:
        if name in cell:
            columns[name] = np.full(n_days, float(cell[name]))
    frame = pd.DataFrame(columns)
    longitude = float(cell["longitude"])
    frame[PFT_COLUMN] = cell[PFT_COLUMN]
    frame["latitude"] = float(cell["latitude"])
    frame["longitude"] = longitude
    frame[DAY_COLUMN] = days
    frame[TIME_COLUMN] = solar_timestamp(days, longitude)
    return frame
