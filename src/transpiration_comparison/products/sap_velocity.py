from __future__ import annotations

"""Loader for our sap velocity predictions (parquet -> xarray)."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..config import PRODUCTS
from .base import ProductBase

logger = logging.getLogger(__name__)


class SapVelocityProduct(ProductBase):
    """Load our XGBoost sap velocity predictions from parquet files."""

    def __init__(self, **kwargs):
        super().__init__(config=PRODUCTS["sap_velocity"], **kwargs)

    def download(self) -> Path:
        """No download needed -- predictions already on disk."""
        parquet_dir = self.paths.predictions_parquet
        if not parquet_dir.exists():
            raise FileNotFoundError(f"Predictions not found: {parquet_dir}")
        n_files = len(list(parquet_dir.glob("*.parquet")))
        logger.info(f"Found {n_files} parquet files in {parquet_dir}")
        return parquet_dir

    def load(self) -> xr.Dataset:
        """Convert monthly parquet files to gridded xarray Dataset.

        Reads only lat, lon, timestamp, sap_velocity_xgb columns.
        Pivots from tabular (row per pixel-day) to gridded (time, lat, lon).
        """
        parquet_dir = self.paths.predictions_parquet
        parquet_files = sorted(parquet_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files in {parquet_dir}")

        logger.info(f"Loading {len(parquet_files)} parquet files...")
        value_col = self.config.variable  # 'sap_velocity_xgb'

        all_months = []
        for pf in parquet_files:
            logger.info(f"  Reading {pf.name}...")
            df = pd.read_parquet(
                pf,
                columns=["latitude", "longitude", "timestamp", value_col],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            # Round coordinates to grid centers (0.1 deg)
            df["lat"] = np.round(df["latitude"], 1)
            df["lon"] = np.round(df["longitude"], 1)
            df = df.drop(columns=["latitude", "longitude"])

            # Aggregate to daily mean per grid cell (handles potential duplicates)
            daily = (
                df.assign(date=df["timestamp"].dt.date).groupby(["date", "lat", "lon"])[value_col].mean().reset_index()
            )
            daily["date"] = pd.to_datetime(daily["date"])
            all_months.append(daily)

        logger.info("Concatenating all months...")
        combined = pd.concat(all_months, ignore_index=True)
        del all_months

        logger.info("Pivoting to xarray grid...")
        combined = combined.set_index(["date", "lat", "lon"])
        ds = combined.to_xarray().rename({"date": "time"})
        ds = ds.rename({value_col: "sap_velocity"})

        # Convert to float32 to save memory
        ds["sap_velocity"] = ds["sap_velocity"].astype(np.float32)

        # Sort coordinates
        ds = ds.sortby(["time", "lat", "lon"])

        ds.attrs["product"] = self.config.name
        ds.attrs["units"] = self.config.units
        ds.attrs["source_type"] = "local"

        logger.info(f"Loaded sap velocity: {ds.dims}, time: {ds.time.values[0]} to {ds.time.values[-1]}")
        return ds

    def load_pft_map(self) -> xr.DataArray:
        """Extract dominant PFT per pixel from the parquet predictions.

        Returns DataArray with dims (lat, lon) and integer PFT labels.
        """
        parquet_dir = self.paths.predictions_parquet
        first_file = sorted(parquet_dir.glob("*.parquet"))[0]

        pft_cols = ["ENF", "EBF", "DNF", "DBF", "MF", "WSA", "SAV", "WET"]
        df = pd.read_parquet(
            first_file,
            columns=["latitude", "longitude"] + pft_cols,
        )
        df["lat"] = np.round(df["latitude"], 1)
        df["lon"] = np.round(df["longitude"], 1)

        # Take first occurrence per grid cell (PFT is static)
        df = df.drop(columns=["latitude", "longitude"]).drop_duplicates(subset=["lat", "lon"], keep="first")

        # Dominant PFT = column with max value
        pft_matrix = df[pft_cols].values
        dominant_idx = np.argmax(pft_matrix, axis=1)
        df["pft"] = [pft_cols[i] for i in dominant_idx]

        # Pivot to grid
        pft_grid = df.set_index(["lat", "lon"])["pft"].to_xarray()
        pft_grid = pft_grid.sortby(["lat", "lon"])
        pft_grid.attrs["description"] = "Dominant PFT per grid cell (IGBP)"
        return pft_grid
