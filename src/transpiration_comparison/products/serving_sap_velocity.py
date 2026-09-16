"""Loader for a servable prediction out of the serving store (NetCDF cube).

The legacy `SapVelocityProduct` reads the stage-2 parquet dump and rounds its
coordinates to one decimal. `src/serving` writes something different: one
NetCDF per year under ``<store>/predictions/<run_id>/``, on the native
ERA5-Land grid (x.0 centres) with latitude DESCENDING, carrying
``served_cell`` / ``unbuilt_cell`` masks, and the PFT label lives in
``<store>/cells.parquet`` rather than in one-hot columns beside the values.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..config import PRODUCTS
from .base import ProductBase

logger = logging.getLogger(__name__)

CELLS_NAME = "cells.parquet"


class ServingSapVelocityProduct(ProductBase):
    """Load ``sap_velocity_<run_id>_<year>.nc`` for every year in the period."""

    def __init__(self, store: Path, run_id: str, **kwargs):
        super().__init__(config=PRODUCTS["sap_velocity"], **kwargs)
        self.store = Path(store)
        self.run_id = run_id

    # -- layout ------------------------------------------------------------

    @property
    def prediction_dir(self) -> Path:
        return self.store / "predictions" / self.run_id

    @property
    def cells_path(self) -> Path:
        return self.store / CELLS_NAME

    @property
    def years(self) -> list:
        return list(range(int(self.period.start[:4]), int(self.period.end[:4]) + 1))

    def year_files(self) -> list:
        """Every year's file, or raise naming the years that are absent.

        A year that has not been predicted yet is the expected failure here --
        it must stop the run, not quietly shorten the period.
        """
        found, missing = [], []
        for year in self.years:
            path = self.prediction_dir / f"sap_velocity_{self.run_id}_{year}.nc"
            if path.exists():
                found.append(path)
            else:
                missing.append(str(year))
        if missing:
            raise FileNotFoundError(
                f"no prediction for {', '.join(missing)} in {self.prediction_dir} "
                f"(period {self.period.start} .. {self.period.end}). "
                "Run grid_predict for those years first."
            )
        return found

    # -- ProductBase -------------------------------------------------------

    def download(self) -> Path:
        """Nothing to fetch; verify every year of the period is on disk."""
        files = self.year_files()
        logger.info(f"Found {len(files)} prediction year(s) in {self.prediction_dir}")
        return self.prediction_dir

    def load(self) -> xr.Dataset:
        """Return (time, lat, lon) sap velocity, ascending lat, masks applied."""
        parts = []
        for path in self.year_files():
            logger.info(f"  Reading {path.name}...")
            year_ds = xr.open_dataset(path, chunks={"time": 365})
            values = year_ds["sap_velocity"]
            # A cell the trainer could not build, or that was never served, is
            # absent -- not zero. Keep it NaN so it cannot enter a mean.
            if "served_cell" in year_ds.variables:
                values = values.where(year_ds["served_cell"] == 1)
            if "unbuilt_cell" in year_ds.variables:
                values = values.where(year_ds["unbuilt_cell"] == 0)
            parts.append(values.to_dataset(name="sap_velocity"))

        ds = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]

        if ds["lat"].values[0] > ds["lat"].values[-1]:
            ds = ds.isel(lat=slice(None, None, -1))

        ds = ds.sel(
            lat=slice(self.domain.lat_min, self.domain.lat_max),
            lon=slice(self.domain.lon_min, self.domain.lon_max),
            time=slice(self.period.start, self.period.end),
        )
        ds = ds.sortby(["time", "lat", "lon"])

        ds["sap_velocity"] = ds["sap_velocity"].astype(np.float32)
        ds.attrs["product"] = self.config.name
        ds.attrs["units"] = self.config.units
        ds.attrs["source_type"] = "serving"
        ds.attrs["run_id"] = self.run_id

        logger.info(
            f"Loaded servable sap velocity: {dict(ds.sizes)}, "
            f"{str(ds.time.values[0])[:10]} .. {str(ds.time.values[-1])[:10]}"
        )
        return ds

    def load_pft_map(self) -> xr.DataArray:
        """Dominant PFT per cell, from the store's cells.parquet."""
        if not self.cells_path.exists():
            raise FileNotFoundError(f"cells table not found: {self.cells_path}")

        cells = pd.read_parquet(self.cells_path, columns=["latitude", "longitude", "pft"])
        cells = cells.rename(columns={"latitude": "lat", "longitude": "lon"})
        cells = cells.drop_duplicates(subset=["lat", "lon"], keep="first")

        pft_grid = cells.set_index(["lat", "lon"])["pft"].to_xarray()
        pft_grid = pft_grid.sortby(["lat", "lon"])
        pft_grid.attrs["description"] = f"Dominant PFT per grid cell ({self.cells_path})"
        logger.info(f"Loaded PFT map: {dict(pft_grid.sizes)}")
        return pft_grid
