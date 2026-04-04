from __future__ import annotations

"""Abstract base class for transpiration product handlers."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import xarray as xr

from ..config import DEFAULT_DOMAIN, DEFAULT_PATHS, DEFAULT_PERIOD, Paths, ProductConfig, SpatialDomain, TimePeriod

logger = logging.getLogger(__name__)


class ProductBase(ABC):
    """Base class for transpiration product download and loading.

    Each product implements download() and load() methods.
    download() is idempotent: skips if files already exist.
    load() returns an xarray.Dataset with standardized coordinates.
    """

    def __init__(
        self,
        config: ProductConfig,
        paths: Paths = DEFAULT_PATHS,
        period: TimePeriod = DEFAULT_PERIOD,
        domain: SpatialDomain = DEFAULT_DOMAIN,
    ):
        self.config = config
        self.paths = paths
        self.period = period
        self.domain = domain

    @property
    def output_dir(self) -> Path:
        d = self.paths.products_dir / self.config.short_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def preprocessed_path(self) -> Path:
        return self.paths.preprocessed_dir / f"{self.config.short_name}_preprocessed.nc"

    @abstractmethod
    def download(self) -> Path:
        """Download raw data. Returns path to downloaded directory/file.

        Must be idempotent: skip download if files already exist and are valid.
        Raises RuntimeError on unrecoverable download failure.
        """
        ...

    @abstractmethod
    def load(self) -> xr.Dataset:
        """Load raw data as xarray.Dataset with standardized coordinates.

        Returns:
            Dataset with dimensions (time, lat, lon) and a single data variable
            named according to self.config.variable.
            Attributes include 'product', 'units', 'source_type'.
        """
        ...

    def _standardize_coords(self, ds: xr.Dataset) -> xr.Dataset:
        """Rename coordinates to standard lat/lon/time names.

        Also wraps longitude from 0..360 to -180..180 if needed,
        and ensures ascending lat order.
        """
        import numpy as np

        rename = {}
        for coord in ds.coords:
            low = coord.lower()
            if low in ("latitude", "lat_0", "y"):
                rename[coord] = "lat"
            elif low in ("longitude", "lon_0", "x"):
                rename[coord] = "lon"
            elif low in ("valid_time", "date", "datetime"):
                rename[coord] = "time"
        if rename:
            ds = ds.rename(rename)
        # Wrap longitude from 0..360 to -180..180
        if float(ds["lon"].max()) > 180.0:
            ds = ds.assign_coords(lon=((ds["lon"] + 180) % 360) - 180)
            ds = ds.sortby("lon")
        # Ensure ascending lat
        if ds["lat"].values[0] > ds["lat"].values[-1]:
            ds = ds.isel(lat=slice(None, None, -1))
        # Add metadata
        ds.attrs["product"] = self.config.name
        ds.attrs["units"] = self.config.units
        ds.attrs["source_type"] = self.config.source_type
        return ds

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.config.name})"
