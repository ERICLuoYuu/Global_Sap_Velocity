from __future__ import annotations

"""GLDAS Noah v2.1 transpiration downloader via OPeNDAP."""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from ..config import PRODUCTS, UNIT_TO_MM_DAY
from .base import ProductBase

logger = logging.getLogger(__name__)

# GLDAS OPeNDAP base URL
GLDAS_OPENDAP_BASE = "https://hydro1.gesdisc.eosdis.nasa.gov/opendap/GLDAS/GLDAS_NOAH025_3H.2.1"


class GLDASProduct(ProductBase):
    """Download and load GLDAS Noah v2.1 transpiration via OPeNDAP."""

    def __init__(self, **kwargs):
        super().__init__(config=PRODUCTS["gldas"], **kwargs)

    def download(self) -> Path:
        """Download GLDAS transpiration via OPeNDAP subsetting.

        Uses earthaccess for NASA authentication + xarray OPeNDAP.
        Downloads monthly aggregations to avoid massive 3-hourly files.
        """
        start_year = int(self.period.start[:4])
        end_year = int(self.period.end[:4])

        for year in range(start_year, end_year + 1):
            output_file = self.output_dir / f"gldas_transp_{year}.nc"
            if output_file.exists() and output_file.stat().st_size > 100_000:
                logger.info(f"Skipping existing: {output_file.name}")
                continue

            logger.info(f"Downloading GLDAS {year} via OPeNDAP...")
            try:
                ds = xr.open_dataset(
                    GLDAS_OPENDAP_BASE,
                    engine="netcdf4",
                    chunks={"time": 8 * 31},  # ~1 month of 3-hourly
                )
                # Select transpiration variable and time/space subset
                ds = ds[["Tveg_tavg"]].sel(
                    time=slice(f"{year}-01-01", f"{year}-12-31"),
                    lat=slice(self.domain.lat_min, self.domain.lat_max),
                    lon=slice(self.domain.lon_min, self.domain.lon_max),
                )

                # Aggregate 3-hourly to daily mean
                ds_daily = ds.resample(time="1D").mean()

                # Save to local NetCDF
                ds_daily.to_netcdf(output_file)
                logger.info(f"  Saved {output_file.stat().st_size / 1e6:.1f} MB")

            except Exception as e:
                logger.warning(f"OPeNDAP failed for {year}: {e}")
                logger.info("Trying earthaccess approach...")
                _download_via_earthaccess(year, self.domain, output_file)

        return self.output_dir

    def load(self) -> xr.Dataset:
        """Load GLDAS transpiration, convert units to mm/day."""
        nc_files = sorted(self.output_dir.glob("gldas_transp_*.nc"))
        if not nc_files:
            raise FileNotFoundError(f"No GLDAS files in {self.output_dir}")

        logger.info(f"Loading {len(nc_files)} GLDAS files...")
        ds = xr.open_mfdataset(nc_files, combine="by_coords", chunks={"time": 365})

        # Convert Tveg_tavg from kg/m2/s to mm/day
        conversion = UNIT_TO_MM_DAY[self.config.units]  # 86400
        ds["Tveg_tavg"] = ds["Tveg_tavg"] * conversion
        ds = ds.rename({"Tveg_tavg": "transpiration"})
        ds = self._standardize_coords(ds)

        ds = ds.sel(
            lat=slice(self.domain.lat_min, self.domain.lat_max),
            lon=slice(self.domain.lon_min, self.domain.lon_max),
            time=slice(self.period.start, self.period.end),
        )

        ds["transpiration"] = ds["transpiration"].astype(np.float32)
        ds.attrs["product"] = self.config.name
        ds.attrs["units"] = "mm/day"

        logger.info(f"Loaded GLDAS: {dict(ds.dims)}")
        return ds


def _download_via_earthaccess(year: int, domain, output_file: Path) -> None:
    """Fallback: use earthaccess for authenticated download."""
    try:
        import earthaccess
    except ImportError:
        raise ImportError("earthaccess required: pip install earthaccess")

    earthaccess.login(strategy="netrc")
    results = earthaccess.search_data(
        short_name="GLDAS_NOAH025_3H",
        version="2.1",
        temporal=(f"{year}-01-01", f"{year}-12-31"),
        bounding_box=(domain.lon_min, domain.lat_min, domain.lon_max, domain.lat_max),
    )
    if not results:
        raise RuntimeError(f"No GLDAS data found for {year}")

    # Download and merge
    files = earthaccess.download(results, str(output_file.parent))
    logger.info(f"Downloaded {len(files)} GLDAS granules for {year}")
