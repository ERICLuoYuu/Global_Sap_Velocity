
from __future__ import annotations

"""ERA5-Land transpiration downloader via CDS API (monthly means)."""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from ..config import PRODUCTS, UNIT_TO_MM_DAY
from .base import ProductBase

logger = logging.getLogger(__name__)


class ERA5LandProduct(ProductBase):
    """Download and load ERA5-Land vegetation transpiration (monthly means)."""

    def __init__(self, **kwargs):
        super().__init__(config=PRODUCTS["era5land"], **kwargs)

    def download(self) -> Path:
        """Download ERA5-Land monthly mean transpiration via CDS API.

        Uses 'reanalysis-era5-land-monthly-means' which is much faster
        than downloading hourly data. Each file is ~20 MB vs ~18 GB hourly.
        """
        try:
            import cdsapi
        except ImportError:
            raise ImportError("cdsapi required: pip install cdsapi")

        client = cdsapi.Client()
        start_year = int(self.period.start[:4])
        end_year = int(self.period.end[:4])

        for year in range(start_year, end_year + 1):
            output_file = self.output_dir / f"era5land_transp_monthly_{year}.nc"
            if output_file.exists() and output_file.stat().st_size > 100_000:
                logger.info(f"Skipping existing: {output_file.name}")
                continue

            logger.info(f"Requesting ERA5-Land monthly mean transp {year}...")
            client.retrieve(
                "reanalysis-era5-land-monthly-means",
                {
                    "product_type": ["monthly_averaged_reanalysis"],
                    "variable": ["evaporation_from_vegetation_transpiration"],
                    "year": str(year),
                    "month": [f"{m:02d}" for m in range(1, 13)],
                    "time": ["00:00"],
                    "area": [
                        self.domain.lat_max,
                        self.domain.lon_min,
                        self.domain.lat_min,
                        self.domain.lon_max,
                    ],
                    "data_format": "netcdf",
                },
                str(output_file),
            )
            logger.info(f"  Downloaded {output_file.stat().st_size / 1e6:.1f} MB")

        return self.output_dir

    def load(self) -> xr.Dataset:
        """Load ERA5-Land monthly mean transpiration, convert units.

        ERA5-Land transpiration is in meters/day (negative = upward flux).
        Monthly means are already daily averages.
        Conversion: |value| * 1000 = mm/day.
        """
        nc_files = sorted(self.output_dir.glob("era5land_transp_monthly_*.nc"))
        if not nc_files:
            raise FileNotFoundError(f"No ERA5-Land files in {self.output_dir}")

        logger.info(f"Loading {len(nc_files)} ERA5-Land monthly files...")
        ds = xr.open_mfdataset(nc_files, combine="by_coords", chunks={"time": 12})

        # Find the transpiration variable
        transp_var = None
        for var_name in ds.data_vars:
            low = var_name.lower()
            if "transpiration" in low or "evaporation_from_vegetation" in low:
                transp_var = var_name
                break
        if transp_var is None:
            for candidate in ["e_from_vegetation_transpiration", "tevp", "evatc", "evavt"]:
                if candidate in ds.data_vars:
                    transp_var = candidate
                    break
        if transp_var is None:
            raise KeyError(f"Transpiration variable not found. Available: {list(ds.data_vars)}")

        ds = ds[[transp_var]]
        ds = self._standardize_coords(ds)

        # Convert m/day to mm/day, take absolute value (negative = upward)
        conversion = UNIT_TO_MM_DAY[self.config.units]  # 1000
        ds[transp_var] = np.abs(ds[transp_var]) * conversion
        ds = ds.rename({transp_var: "transpiration"})

        ds = ds.sel(
            lat=slice(self.domain.lat_min, self.domain.lat_max),
            lon=slice(self.domain.lon_min, self.domain.lon_max),
            time=slice(self.period.start, self.period.end),
        )

        ds["transpiration"] = ds["transpiration"].astype(np.float32)
        ds.attrs["product"] = self.config.name
        ds.attrs["units"] = "mm/day"
        ds.attrs["temporal_resolution"] = "monthly_mean"

        logger.info(f"Loaded ERA5-Land: {dict(ds.dims)}")
        return ds
