from __future__ import annotations

"""GLEAM v4.2a transpiration downloader (SFTP from gleam.eu)."""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from ..config import PRODUCTS
from .base import ProductBase

logger = logging.getLogger(__name__)

# GLEAM SFTP credentials (must be set as env vars after registration)
# GLEAM_USER, GLEAM_PASS
GLEAM_HOST = "hydras.ugent.be"
GLEAM_PORT = 2225
GLEAM_REMOTE_DIR = "/data/v4.2a/daily"


class GLEAMProduct(ProductBase):
    """Download and load GLEAM v4.2a daily transpiration."""

    def __init__(self, **kwargs):
        super().__init__(config=PRODUCTS["gleam"], **kwargs)

    def download(self) -> Path:
        """Download GLEAM Et (transpiration) NetCDF files via SFTP.

        GLEAM provides one NetCDF per year with daily Et at 0.1 deg.
        File pattern: Et_{year}_GLEAM_v4.2a.nc
        """
        import os

        user = os.environ.get("GLEAM_USER")
        password = os.environ.get("GLEAM_PASS")
        if not user or not password:
            raise RuntimeError(
                "GLEAM credentials not set. Register at gleam.eu and set "
                "GLEAM_USER and GLEAM_PASS environment variables."
            )

        start_year = int(self.period.start[:4])
        end_year = int(self.period.end[:4])

        try:
            import paramiko
        except ImportError:
            raise ImportError("paramiko required for GLEAM SFTP: pip install paramiko")

        transport = paramiko.Transport((GLEAM_HOST, GLEAM_PORT))
        transport.connect(username=user, password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)

        try:
            for year in range(start_year, end_year + 1):
                remote_file = f"{GLEAM_REMOTE_DIR}/{year}/Et_{year}_GLEAM_v4.2a.nc"
                local_file = self.output_dir / f"Et_{year}_GLEAM_v4.2a.nc"

                if local_file.exists() and local_file.stat().st_size > 1_000_000_000:
                    logger.info(f"Skipping existing: {local_file.name}")
                    continue

                logger.info(f"Downloading {remote_file} -> {local_file}")
                sftp.get(remote_file, str(local_file))
                logger.info(f"  Downloaded {local_file.stat().st_size / 1e6:.1f} MB")
        finally:
            sftp.close()
            transport.close()

        return self.output_dir

    def load(self) -> xr.Dataset:
        """Load GLEAM Et files and concatenate into single Dataset."""
        nc_files = sorted(self.output_dir.glob("Et_*_GLEAM_v4.2a.nc"))
        if not nc_files:
            raise FileNotFoundError(f"No GLEAM files in {self.output_dir}")

        logger.info(f"Loading {len(nc_files)} GLEAM files...")
        ds = xr.open_mfdataset(nc_files, combine="by_coords", chunks={"time": 365})

        # Select transpiration variable and crop to domain
        ds = ds[["Et"]].rename({"Et": "transpiration"})
        ds = self._standardize_coords(ds)
        ds = ds.sel(
            lat=slice(self.domain.lat_min, self.domain.lat_max),
            lon=slice(self.domain.lon_min, self.domain.lon_max),
            time=slice(self.period.start, self.period.end),
        )

        ds["transpiration"] = ds["transpiration"].astype(np.float32)
        ds.attrs["product"] = self.config.name
        ds.attrs["units"] = "mm/day"

        logger.info(f"Loaded GLEAM: {dict(ds.dims)}")
        return ds
