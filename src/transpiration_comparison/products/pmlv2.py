from __future__ import annotations

"""PMLv2 transpiration accessor via Google Earth Engine (8-day composites)."""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from ..config import PRODUCTS
from .base import ProductBase

logger = logging.getLogger(__name__)

# GEE asset for PMLv2
PMLV2_ASSET = "projects/pml_evapotranspiration/PML/OUTPUT/PML_V2_8day_v016"


class PMLv2Product(ProductBase):
    """Download PMLv2 8-day transpiration composites via GEE export."""

    def __init__(self, **kwargs):
        super().__init__(config=PRODUCTS["pmlv2"], **kwargs)

    def download(self) -> Path:
        """Export individual PMLv2 Ec 8-day composites at 0.1 deg.

        Downloads each 8-day composite in lon tiles (getDownloadURL has a
        pixel limit that prevents full-globe single requests). Tiles are
        merged in load().
        Yields ~138 × n_tiles GeoTIFFs for 2016-2018.
        """
        try:
            import ee
        except ImportError as exc:
            raise ImportError("earthengine-api required: pip install earthengine-api") from exc

        ee.Initialize(project="era5download-447713")

        collection = ee.ImageCollection(PMLV2_ASSET).filterDate(self.period.start, self.period.end).select("Ec")

        # Get image dates from collection metadata
        image_dates = collection.aggregate_array("system:time_start").getInfo()
        logger.info(f"Found {len(image_dates)} PMLv2 8-day composites")

        # Tile by longitude (60° chunks) to stay under GEE download pixel limit
        lon_tiles = list(range(int(self.domain.lon_min), int(self.domain.lon_max), 60))

        for ts_ms in image_dates:
            date_str = _ms_to_date_str(ts_ms)
            merged_file = self.output_dir / f"pmlv2_Ec_8d_{date_str}.tif"
            if merged_file.exists() and merged_file.stat().st_size > 10_000:
                logger.info(f"Skipping existing: {merged_file.name}")
                continue

            img = collection.filter(ee.Filter.eq("system:time_start", ts_ms)).first()
            tile_files = []

            for lon_start in lon_tiles:
                lon_end = min(lon_start + 60, int(self.domain.lon_max))
                tile_file = self.output_dir / f"pmlv2_Ec_8d_{date_str}_tile{lon_start}.tif"

                if not (tile_file.exists() and tile_file.stat().st_size > 5_000):
                    tile_region = ee.Geometry.Rectangle([lon_start, self.domain.lat_min, lon_end, self.domain.lat_max])
                    try:
                        url = img.getDownloadURL(
                            {
                                "crs": "EPSG:4326",
                                "crs_transform": [0.1, 0, lon_start, 0, -0.1, self.domain.lat_max],
                                "region": tile_region,
                                "format": "GEO_TIFF",
                            }
                        )
                        _download_url(url, tile_file)
                    except Exception as e:
                        logger.error(f"  Failed {date_str} tile {lon_start}: {e}")
                        continue

                tile_files.append(tile_file)

            # Merge tiles into single file
            if tile_files:
                _merge_lon_tiles(tile_files, merged_file)
                # Clean up tiles
                for tf in tile_files:
                    tf.unlink(missing_ok=True)
                logger.info(f"  {date_str}: {merged_file.stat().st_size / 1e6:.1f} MB")

        return self.output_dir

    def load(self) -> xr.Dataset:
        """Load PMLv2 8-day GeoTIFFs into xarray Dataset.

        Supports both legacy monthly files (pmlv2_Ec_YYYY_MM.tif)
        and new 8-day files (pmlv2_Ec_8d_YYYY-MM-DD.tif).
        """
        import rioxarray  # noqa: F401

        # Try 8-day files first, fall back to legacy monthly
        tif_files = sorted(self.output_dir.glob("pmlv2_Ec_8d_*.tif"))
        is_8day = True
        if not tif_files:
            tif_files = sorted(self.output_dir.glob("pmlv2_Ec_*.tif"))
            is_8day = False
        if not tif_files:
            raise FileNotFoundError(f"No PMLv2 files in {self.output_dir}")

        logger.info(f"Loading {len(tif_files)} PMLv2 GeoTIFFs ({'8-day' if is_8day else 'monthly'})...")
        datasets = []
        for tf in tif_files:
            da = xr.open_dataarray(tf, engine="rasterio")
            da = da.squeeze(drop=True)

            if is_8day:
                # pmlv2_Ec_8d_YYYY-MM-DD.tif
                date_str = tf.stem.split("8d_")[1]  # "YYYY-MM-DD"
                timestamp = np.datetime64(date_str)
            else:
                # Legacy: pmlv2_Ec_YYYY_MM.tif
                parts = tf.stem.split("_")
                year, month = int(parts[2]), int(parts[3])
                timestamp = np.datetime64(f"{year}-{month:02d}-15")

            da = da.expand_dims(time=[timestamp])
            da.name = "transpiration"
            datasets.append(da)

        ds = xr.concat(datasets, dim="time").to_dataset()
        ds = self._standardize_coords(ds)

        # PMLv2 Ec is stored as integer with scale factor 0.01 (mm/day rate).
        ds["transpiration"] = (ds["transpiration"] * 0.01).astype(np.float32)
        ds.attrs["product"] = self.config.name
        ds.attrs["units"] = "mm/day"
        ds.attrs["temporal_resolution"] = "8-day" if is_8day else "monthly"

        logger.info(f"Loaded PMLv2: {dict(ds.dims)}")
        return ds


def _ms_to_date_str(timestamp_ms: int) -> str:
    """Convert GEE milliseconds-since-epoch to 'YYYY-MM-DD' string."""
    from datetime import datetime, timezone

    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d")


def _merge_lon_tiles(tile_files: list[Path], output_path: Path) -> None:
    """Merge longitude-tiled GeoTIFFs into a single file via xarray."""
    import rioxarray  # noqa: F401

    arrays = []
    for tf in sorted(tile_files):
        da = xr.open_dataarray(tf, engine="rasterio").squeeze(drop=True)
        arrays.append(da)
    merged = xr.concat(arrays, dim="x")
    merged = merged.sortby("x")
    merged.rio.to_raster(str(output_path))


def _download_url(url: str, output_path: Path) -> None:
    """Download a file from URL."""
    import urllib.request

    urllib.request.urlretrieve(url, str(output_path))
