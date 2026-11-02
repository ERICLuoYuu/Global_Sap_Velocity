"""Bootstrap the Fan 2017 root-depth raster at 0.1° global resolution.

Reuses the continent-routing and sign-flip logic from
``extract_root_depth_fan2017.py`` but replaces the per-site point sampling
with an area-mean regrid onto a global 0.1° grid. Writes
``fan2017_root_depth_0p1deg_v1.nc`` per the Section 5 contract of
``.claude/plan/rooted-ingestion-redesign.md``.

Algorithm
---------
For each continent in priority order (EURASIA → NAMERICA → SAMERICA → AFRICA →
AUSTRALIA, matching ``extract_root_depth_fan2017.CONTINENT_BBOXES``):

1. Open the ETDEPTH OPeNDAP dataset.
2. Read the full continent into memory (lat × lon arrays plus mask). One
   continent fits in ~2 GB at 30 arcsec; reading once and regridding locally
   beats per-cell OPeNDAP slicing.
3. For each target 0.1° cell whose centre falls in the continent's bbox AND
   is not yet filled by a higher-priority continent:
   - Slice the native pixels falling in the cell's lat/lon window.
   - Compute the area-mean of mask==1 pixels (cosine-latitude weighted).
   - Record ``n_valid_source_pixels`` (count of unmasked pixels in the window).
   - Apply ``sign = -1`` to flip below-surface z-values to positive magnitude.

Cells with valid_fraction < ``min_valid_fraction`` (default 0.25) are masked.

Output grid
-----------
The target grid spans the union of continent bboxes at exactly 0.1° spacing,
aligned so cell centres fall on multiples of 0.1°. This matches the σ-map
grid produced by ``compute_swvl_sigma.py``, so downstream merging is a simple
xarray ``align``.

CLI
---
python -m src.Extractors.bootstrap_fan2017_raster \\
    --output /scratch/tmp/yluo2/gsv/outputs/derived/fan2017_root_depth_0p1deg_v1.nc \\
    --min-valid-fraction 0.25 \\
    --version v1
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from src.Extractors.extract_root_depth_fan2017 import (
    CONTINENT_BBOXES,
    Continent,
    ContinentBBox,
    opendap_url,
)

LOGGER = logging.getLogger("bootstrap_fan2017_raster")

ETDEPTH_KIND = "ETDEPTH"
ETDEPTH_VAR = "ETDEPTH"
SIGN = -1  # flip below-surface z-values to positive magnitude
TARGET_RES_DEG = 0.1
DEFAULT_MIN_VALID_FRACTION = 0.25
COORD_DECIMALS = 1

CONTINENT_CODES: dict[Continent, int] = {
    "AFRICA": 0,
    "AUSTRALIA": 1,
    "EURASIA": 2,
    "NAMERICA": 3,
    "SAMERICA": 4,
}


@dataclass
class TargetGrid:
    """Regular 0.1° lat/lon target grid spanning all continent bboxes."""

    lats: np.ndarray  # ascending, float32, on 0.1° centres
    lons: np.ndarray  # ascending, float32, on 0.1° centres

    @classmethod
    def union_of_bboxes(cls, bboxes: Sequence[ContinentBBox]) -> TargetGrid:
        lat_min = min(b.lat_min for b in bboxes)
        lat_max = max(b.lat_max for b in bboxes)
        lon_min = min(b.lon_min for b in bboxes)
        lon_max = max(b.lon_max for b in bboxes)

        # Snap to multiples of 0.1° so cell centres are deterministic.
        lat_lo = np.floor(lat_min / TARGET_RES_DEG) * TARGET_RES_DEG
        lat_hi = np.ceil(lat_max / TARGET_RES_DEG) * TARGET_RES_DEG
        lon_lo = np.floor(lon_min / TARGET_RES_DEG) * TARGET_RES_DEG
        lon_hi = np.ceil(lon_max / TARGET_RES_DEG) * TARGET_RES_DEG

        n_lat = int(round((lat_hi - lat_lo) / TARGET_RES_DEG)) + 1
        n_lon = int(round((lon_hi - lon_lo) / TARGET_RES_DEG)) + 1
        lats = np.round(np.linspace(lat_lo, lat_hi, n_lat), COORD_DECIMALS).astype(np.float32)
        lons = np.round(np.linspace(lon_lo, lon_hi, n_lon), COORD_DECIMALS).astype(np.float32)
        return cls(lats=lats, lons=lons)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.lats.size, self.lons.size)


def open_etdepth(continent: Continent) -> xr.Dataset:
    url = opendap_url(continent, ETDEPTH_KIND)
    LOGGER.info("Opening %s", url)
    return xr.open_dataset(url, engine="netcdf4", decode_cf=True)


def regrid_continent(
    continent: Continent,
    bbox: ContinentBBox,
    target: TargetGrid,
    *,
    min_valid_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (root_depth, n_valid) arrays on the target grid for one continent.

    Only cells with centres inside the continent's bbox are filled; the rest
    stay NaN / 0. The caller composites multiple continents in priority order.
    """
    ds = open_etdepth(continent)
    try:
        # Materialize the full continent. ETDEPTH at 30 arcsec is ~1.5-2 GB
        # per continent; reading once beats per-cell OPeNDAP slicing.
        LOGGER.info("Reading %s ETDEPTH + mask into memory", continent)
        et_raw = ds[ETDEPTH_VAR].isel(time=0).values.astype(np.float32)
        mask_raw = np.asarray(ds["mask"].values)
        mask = (mask_raw != 0).astype(np.uint8)
        lat_native = ds["lat"].values.astype(np.float64)
        lon_native = ds["lon"].values.astype(np.float64)
        LOGGER.info(
            "%s: native grid %d × %d, lat=[%g, %g], lon=[%g, %g]",
            continent,
            lat_native.size,
            lon_native.size,
            lat_native.min(),
            lat_native.max(),
            lon_native.min(),
            lon_native.max(),
        )
    finally:
        ds.close()

    # Pre-compute cosine-latitude weights for area-mean.
    cos_lat = np.cos(np.deg2rad(lat_native)).astype(np.float32)

    # Replace masked / non-finite et with 0 (we'll re-mask via the count check).
    et_clean = np.where(mask == 1, et_raw, np.nan)

    root_depth_out = np.full(target.shape, np.nan, dtype=np.float32)
    n_valid_out = np.zeros(target.shape, dtype=np.int32)

    half = TARGET_RES_DEG / 2.0
    # Pre-compute native index spans for each target lat/lon to avoid scanning
    # the full native arrays per cell.
    lat_lo_idx = np.searchsorted(lat_native, target.lats - half, side="left")
    lat_hi_idx = np.searchsorted(lat_native, target.lats + half, side="right")
    lon_lo_idx = np.searchsorted(lon_native, target.lons - half, side="left")
    lon_hi_idx = np.searchsorted(lon_native, target.lons + half, side="right")

    cells_filled = 0
    for i, lat_c in enumerate(target.lats):
        if not (bbox.lat_min <= lat_c <= bbox.lat_max):
            continue
        i_lo, i_hi = int(lat_lo_idx[i]), int(lat_hi_idx[i])
        if i_hi <= i_lo:
            continue
        for j, lon_c in enumerate(target.lons):
            if not (bbox.lon_min <= lon_c <= bbox.lon_max):
                continue
            j_lo, j_hi = int(lon_lo_idx[j]), int(lon_hi_idx[j])
            if j_hi <= j_lo:
                continue

            window = et_clean[i_lo:i_hi, j_lo:j_hi]
            mask_window = mask[i_lo:i_hi, j_lo:j_hi]
            n_total = mask_window.size
            n_valid = int((mask_window == 1).sum())
            if n_total == 0 or n_valid / n_total < min_valid_fraction:
                continue

            # Area-mean: cos(lat) weights, ignore NaNs.
            weights = cos_lat[i_lo:i_hi, None] * np.ones_like(window, dtype=np.float32)
            finite = np.isfinite(window)
            if not finite.any():
                continue
            w = weights * finite
            wsum = w.sum()
            if wsum <= 0:
                continue
            mean_val = float(np.nansum(window * w) / wsum)

            root_depth_out[i, j] = SIGN * mean_val
            n_valid_out[i, j] = n_valid
            cells_filled += 1

    LOGGER.info("%s: filled %d target cells", continent, cells_filled)
    return root_depth_out, n_valid_out


def composite_continents(
    target: TargetGrid,
    *,
    min_valid_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stitch all continents into the global raster, first-match-wins."""
    root_depth = np.full(target.shape, np.nan, dtype=np.float32)
    n_valid = np.zeros(target.shape, dtype=np.int32)
    source_continent = np.full(target.shape, -1, dtype=np.int8)

    for bbox in CONTINENT_BBOXES:  # already in priority order
        rd, nv = regrid_continent(
            bbox.name,
            bbox,
            target,
            min_valid_fraction=min_valid_fraction,
        )
        # First-match-wins: only fill cells still NaN.
        unfilled = np.isnan(root_depth)
        new_fill = unfilled & ~np.isnan(rd)
        root_depth = np.where(new_fill, rd, root_depth)
        n_valid = np.where(new_fill, nv, n_valid)
        source_continent = np.where(new_fill, CONTINENT_CODES[bbox.name], source_continent)

    return root_depth, n_valid, source_continent


def write_netcdf(
    out_path: Path,
    target: TargetGrid,
    root_depth: np.ndarray,
    n_valid: np.ndarray,
    source_continent: np.ndarray,
    *,
    min_valid_fraction: float,
    version: str,
) -> None:
    ds = xr.Dataset(
        data_vars={
            "root_depth": xr.DataArray(
                root_depth,
                dims=("lat", "lon"),
                attrs={"units": "m", "long_name": "effective root depth (positive magnitude)"},
            ),
            "n_valid_source_pixels": xr.DataArray(
                n_valid,
                dims=("lat", "lon"),
                attrs={"long_name": "count of mask==1 source pixels averaged into cell"},
            ),
            "source_continent": xr.DataArray(
                source_continent,
                dims=("lat", "lon"),
                attrs={
                    "long_name": "continent file ID providing the cell value",
                    "flag_values": list(CONTINENT_CODES.values()) + [-1],
                    "flag_meanings": " ".join(sorted(CONTINENT_CODES, key=CONTINENT_CODES.get) + ["outside_all"]),
                },
            ),
        },
        coords={
            "lat": xr.DataArray(target.lats, dims="lat", attrs={"units": "degrees_north"}),
            "lon": xr.DataArray(target.lons, dims="lon", attrs={"units": "degrees_east"}),
        },
        attrs={
            "source": "Fan et al. 2017, USC THREDDS OPeNDAP, ETDEPTH variable",
            "citation": "Fan, Y., Miguez-Macho, G., Jobbágy, E.G., et al. (2017). PNAS 114(40):10572-10577",
            "native_resolution": "30 arcsec (~1 km)",
            "regrid_method": "cosine-latitude-weighted area-mean of mask==1 source pixels per 0.1° cell",
            "min_valid_fraction": float(min_valid_fraction),
            "sign_convention": "positive metres below surface (ETDEPTH sign-flipped from source)",
            "version": version,
        },
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        "root_depth": {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)},
        "n_valid_source_pixels": {"zlib": True, "complevel": 4},
        "source_continent": {"zlib": True, "complevel": 4},
        "lat": {"zlib": True, "complevel": 4},
        "lon": {"zlib": True, "complevel": 4},
    }
    ds.to_netcdf(out_path, encoding=encoding, engine="netcdf4")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-valid-fraction", type=float, default=DEFAULT_MIN_VALID_FRACTION)
    parser.add_argument("--version", default="v1")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    target = TargetGrid.union_of_bboxes(CONTINENT_BBOXES)
    LOGGER.info("Target grid: %d lat × %d lon at 0.1°", target.lats.size, target.lons.size)

    root_depth, n_valid, source_continent = composite_continents(
        target,
        min_valid_fraction=args.min_valid_fraction,
    )

    valid = int(np.isfinite(root_depth).sum())
    LOGGER.info(
        "Global raster: %d/%d cells filled (root_depth range [%g, %g] m)",
        valid,
        root_depth.size,
        float(np.nanmin(root_depth)) if valid else float("nan"),
        float(np.nanmax(root_depth)) if valid else float("nan"),
    )

    write_netcdf(
        args.output,
        target,
        root_depth,
        n_valid,
        source_continent,
        min_valid_fraction=args.min_valid_fraction,
        version=args.version,
    )
    LOGGER.info("Wrote %s", args.output)
    LOGGER.info("SHA-256: %s", sha256_of(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
