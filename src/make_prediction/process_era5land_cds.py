"""
ERA5-Land data processing via Copernicus Climate Data Store (CDS) API.

Replaces Google Earth Engine for ERA5-Land data access. Produces numerically
equivalent output so that predict_sap_velocity_sequantial.py works without
modification.

Data sources
------------
1. ``derived-era5-land-daily-statistics`` — instantaneous variables
   (daily mean / min / max of temperature, dewpoint, wind, soil).
2. ``reanalysis-era5-land`` (hourly, 00:00 only) — accumulated variables
   (precipitation, solar radiation, potential evaporation).  The 00:00
   value contains the *previous* day's 24-h accumulation, so timestamps
   are shifted by -1 day.

Usage
-----
    python process_era5land_cds.py --year 2020 --month 6 --time-scale daily
    python process_era5land_cds.py --year 2020 --month 6 --time-scale hourly
    python process_era5land_cds.py --year 2020 --month 6 --validate
"""

from __future__ import annotations

import argparse
import calendar
import gc
import logging
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr

# ---------------------------------------------------------------------------
# Local config (same module used by predict_sap_velocity_sequantial.py)
# ---------------------------------------------------------------------------
# Ensure the make_prediction directory is on the path so ``import config``
# resolves to the local config.py, not a system package.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import (
    DEFAULT_LAT_MAX,
    DEFAULT_LAT_MIN,
    DEFAULT_LON_MAX,
    DEFAULT_LON_MIN,
    ERA5LAND_TEMP_DIR,
    LAI_DATA_DIR,
    PRECIP_CLIMATE_FILE,
    TEMP_CLIMATE_FILE,
)

# ---------------------------------------------------------------------------
# Physical constants (module-level, matching the GEE script exactly)
# ---------------------------------------------------------------------------
MAGNUS_A: float = 6.1078  # hPa
MAGNUS_B: float = 17.269  # dimensionless
MAGNUS_C: float = 237.3  # degC
PAR_FRACTION: float = 0.45  # fraction of SW that is PAR
PPFD_CONVERSION: float = 4.6  # umol photon / J
SOLAR_CONSTANT: float = 1367.0  # W m-2
KELVIN_OFFSET: float = 273.15

# CDS request area [N, W, S, E]
CDS_AREA: list[float] = [
    DEFAULT_LAT_MAX,  # 78
    DEFAULT_LON_MIN,  # -180
    DEFAULT_LAT_MIN,  # -60
    DEFAULT_LON_MAX,  # 180
]

# CDS variable names for the *daily-statistics* dataset
CDS_INSTANTANEOUS_VARS: list[str] = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "volumetric_soil_water_layer_1",
    "soil_temperature_level_1",
]

# CDS variable names for the *hourly* dataset (accumulated fluxes)
CDS_ACCUMULATED_VARS: list[str] = [
    "total_precipitation",
    "surface_solar_radiation_downwards",
    "potential_evaporation",
]

# Mapping from CDS short names (inside NetCDF files) to our internal names
# CDS may use different short names depending on the dataset version;
# we normalise after loading.
_CDS_SHORT_TO_INTERNAL: dict[str, str] = {
    "t2m": "temperature_2m",
    "d2m": "dewpoint_temperature_2m",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "swvl1": "volumetric_soil_water_layer_1",
    "stl1": "soil_temperature_level_1",
    "tp": "total_precipitation",
    "ssrd": "surface_solar_radiation_downwards",
    "pev": "potential_evaporation",
}

# Retry settings for CDS API
MAX_RETRIES: int = 5
INITIAL_BACKOFF_S: float = 10.0

logger = logging.getLogger(__name__)


# =========================================================================
# Utility helpers
# =========================================================================


def _setup_logging(verbose: bool = False) -> None:
    """Configure structured logging."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")


def _marker_path(output_dir: Path, year: int, month: int, tag: str) -> Path:
    """Return the path to a checkpoint marker file."""
    return output_dir / f".done_{tag}_{year}_{month:02d}"


def _is_done(output_dir: Path, year: int, month: int, tag: str) -> bool:
    return _marker_path(output_dir, year, month, tag).exists()


def _mark_done(output_dir: Path, year: int, month: int, tag: str) -> None:
    marker = _marker_path(output_dir, year, month, tag)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"completed {pd.Timestamp.now(tz='UTC')}")


def _cds_client():
    """Create a ``cdsapi.Client``.

    Credentials come from ``~/.cdsapirc`` or the environment variables
    ``CDSAPI_URL`` / ``CDSAPI_KEY``.  They are **never** hard-coded.
    """
    import cdsapi  # imported here so the rest of the module loads without cdsapi

    return cdsapi.Client()


def _download_with_retry(client, dataset: str, request: dict, target: Path) -> Path:
    """Submit a CDS request with exponential-backoff retry on 429 / transient errors."""
    backoff = INITIAL_BACKOFF_S
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("CDS request attempt %d/%d for %s → %s", attempt, MAX_RETRIES, dataset, target.name)
            client.retrieve(dataset, request, str(target))
            logger.info("Download complete: %s (%.1f MB)", target.name, target.stat().st_size / 1e6)
            return target
        except Exception as exc:
            msg = str(exc).lower()
            retryable = (
                "429" in msg
                or "too many" in msg
                or "timeout" in msg
                or "connection" in msg
                or "server" in msg
                or "503" in msg
            )
            if retryable and attempt < MAX_RETRIES:
                logger.warning("Retryable error (%s). Sleeping %.0fs before retry.", exc, backoff)
                time.sleep(backoff)
                backoff *= 2
            else:
                raise


def _unzip_netcdf(zip_path: Path, extract_dir: Path) -> Path:
    """Extract a single NetCDF from a CDS-delivered ZIP archive."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
        if not nc_names:
            raise FileNotFoundError(f"No .nc file found inside {zip_path}")
        zf.extract(nc_names[0], extract_dir)
        return extract_dir / nc_names[0]


# =========================================================================
# Download functions
# =========================================================================


def download_instantaneous_vars(
    client,
    year: int,
    month: int,
    output_dir: Path,
) -> dict[str, Path]:
    """Download daily mean / min / max of instantaneous ERA5-Land variables.

    Uses the ``derived-era5-land-daily-statistics`` dataset which provides
    pre-computed daily statistics so we do not need to aggregate ourselves.

    Returns a dict mapping statistic tag (``mean``, ``min``, ``max``) to the
    downloaded NetCDF path.
    """
    days_in_month = calendar.monthrange(year, month)[1]
    day_list = [f"{d:02d}" for d in range(1, days_in_month + 1)]

    downloaded: dict[str, Path] = {}

    for stat in ("daily_mean", "daily_minimum", "daily_maximum"):
        tag = stat.replace("daily_", "")  # mean / minimum / maximum
        nc_dir = output_dir / "instantaneous"
        nc_dir.mkdir(parents=True, exist_ok=True)
        zip_target = nc_dir / f"inst_{tag}_{year}_{month:02d}.zip"
        nc_target = nc_dir / f"inst_{tag}_{year}_{month:02d}.nc"

        if nc_target.exists():
            logger.info("Skipping download — already exists: %s", nc_target)
            downloaded[tag] = nc_target
            continue

        request = {
            "variable": CDS_INSTANTANEOUS_VARS,
            "year": str(year),
            "month": f"{month:02d}",
            "day": day_list,
            "daily_statistic": stat,
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            "area": CDS_AREA,
            "data_format": "netcdf_zip",
        }

        _download_with_retry(client, "derived-era5-land-daily-statistics", request, zip_target)

        # Extract NetCDF from ZIP
        extracted = _unzip_netcdf(zip_target, nc_dir)
        extracted.rename(nc_target)
        zip_target.unlink(missing_ok=True)
        downloaded[tag] = nc_target
        logger.info("Extracted %s", nc_target)

    return downloaded


def download_accumulated_vars(
    client,
    year: int,
    month: int,
    output_dir: Path,
    time_scale: str = "daily",
) -> Path:
    """Download accumulated variables from ``reanalysis-era5-land``.

    For **daily** mode we only request ``time=00:00`` (which holds the
    previous day's 24-h accumulation).  We also request day 1 of the
    *next* month at 00:00 so that the last day of the target month is
    covered after the -1 day shift.

    For **hourly** mode we request all 24 time steps.

    Returns the path to the downloaded NetCDF.
    """
    nc_dir = output_dir / "accumulated"
    nc_dir.mkdir(parents=True, exist_ok=True)
    nc_target = nc_dir / f"accum_{time_scale}_{year}_{month:02d}.nc"

    if nc_target.exists():
        logger.info("Skipping download — already exists: %s", nc_target)
        return nc_target

    days_in_month = calendar.monthrange(year, month)[1]

    if time_scale == "daily":
        # Accumulated variables: 00:00 on day D contains the 24h total for day D-1.
        # To get all days of month M, we need timestamps from day 2 of month M
        # through day 1 of month M+1 (inclusive).
        # We download days 1..last of month M (day 1 yields D-1 = prev month's last day,
        # which is filtered out after the -1 day shift) plus day 1 of next month
        # (which yields the last day of month M after the shift).
        day_list = [f"{d:02d}" for d in range(1, days_in_month + 1)]

        # Determine next month
        if month == 12:
            next_year, next_month = year + 1, 1
        else:
            next_year, next_month = year, month + 1

        # Download current month
        zip_target = nc_dir / f"accum_{time_scale}_{year}_{month:02d}.zip"
        request_main = {
            "variable": CDS_ACCUMULATED_VARS,
            "year": str(year),
            "month": f"{month:02d}",
            "day": day_list,
            "time": "00:00",
            "area": CDS_AREA,
            "data_format": "netcdf",
        }
        _download_with_retry(client, "reanalysis-era5-land", request_main, zip_target)
        # reanalysis-era5-land may return plain NC (not zipped)
        if zipfile.is_zipfile(zip_target):
            extracted = _unzip_netcdf(zip_target, nc_dir)
            main_nc = nc_dir / f"accum_main_{year}_{month:02d}.nc"
            extracted.rename(main_nc)
            zip_target.unlink(missing_ok=True)
        else:
            main_nc = zip_target  # already a plain .nc

        # Download day 1 of next month (single day)
        extra_target = nc_dir / f"accum_extra_{next_year}_{next_month:02d}.nc"
        request_extra = {
            "variable": CDS_ACCUMULATED_VARS,
            "year": str(next_year),
            "month": f"{next_month:02d}",
            "day": "01",
            "time": "00:00",
            "area": CDS_AREA,
            "data_format": "netcdf",
        }
        _download_with_retry(client, "reanalysis-era5-land", request_extra, extra_target)

        # Merge the two files and shift time by -1 day
        try:
            ds_main = xr.open_dataset(main_nc)
            ds_extra = xr.open_dataset(extra_target)
            ds_combined = xr.concat([ds_main, ds_extra], dim="time")
            ds_main.close()
            ds_extra.close()

            # CRITICAL: shift timestamps by -1 day
            # 00:00 on day D contains the accumulation for day D-1
            ds_combined = ds_combined.assign_coords(time=ds_combined["time"] - pd.Timedelta(days=1))

            # Keep only the target month after shift
            time_mask = (ds_combined["time"].dt.year == year) & (ds_combined["time"].dt.month == month)
            ds_month = ds_combined.sel(time=time_mask)
            ds_month.to_netcdf(nc_target)
            ds_month.close()
            ds_combined.close()
        finally:
            # Clean up intermediates even on error
            main_nc.unlink(missing_ok=True)
            extra_target.unlink(missing_ok=True)

    else:
        # Hourly mode: download all 24 timesteps
        day_list = [f"{d:02d}" for d in range(1, days_in_month + 1)]
        hours = [f"{h:02d}:00" for h in range(24)]

        request = {
            "variable": CDS_ACCUMULATED_VARS + CDS_INSTANTANEOUS_VARS,
            "year": str(year),
            "month": f"{month:02d}",
            "day": day_list,
            "time": hours,
            "area": CDS_AREA,
            "data_format": "netcdf",
        }
        _download_with_retry(client, "reanalysis-era5-land", request, nc_target)

    logger.info("Accumulated data ready: %s", nc_target)
    return nc_target


# =========================================================================
# Derived variable computation (vectorised, pure numpy/xarray)
# =========================================================================


def _magnus_es(temp_c: np.ndarray) -> np.ndarray:
    """Saturation vapour pressure (hPa) via the Magnus formula."""
    return MAGNUS_A * np.exp((MAGNUS_B * temp_c) / (MAGNUS_C + temp_c))


def compute_vpd_kpa(temp_k: np.ndarray, dewpoint_k: np.ndarray) -> np.ndarray:
    """Compute VPD in **kPa** from temperature and dewpoint in Kelvin.

    Mirrors the patched GEE script (Fix 1.4): result in kPa (÷10 from hPa).
    Physical constraint Td <= T is enforced.
    """
    t_c = temp_k - KELVIN_OFFSET
    td_c = dewpoint_k - KELVIN_OFFSET
    td_c = np.minimum(td_c, t_c)  # physical: Td <= T always

    es = _magnus_es(t_c)
    ea = _magnus_es(td_c)
    return np.maximum(es - ea, 0.0) / 10.0  # hPa → kPa


def compute_wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Wind speed from u and v components (m/s)."""
    return np.sqrt(u**2 + v**2)


def compute_ppfd(sw_in_wm2: np.ndarray) -> np.ndarray:
    """PPFD (umol m-2 s-1) from shortwave radiation in W/m²."""
    return np.clip(sw_in_wm2 * PAR_FRACTION * PPFD_CONVERSION, 0.0, 2500.0)


def compute_ext_rad_daily(
    latitudes: np.ndarray,
    doy: int,
    is_leap_year: bool = False,
) -> np.ndarray:
    """Daily mean extraterrestrial radiation (W/m²) via FAO-56 Eq. 21.

    Parameters
    ----------
    latitudes : 1-D array of latitudes in degrees
    doy : day of year (1-366)
    is_leap_year : whether the year is a leap year

    Returns
    -------
    1-D array of Ra in W/m² (daily mean), same length as *latitudes*.
    """
    days_in_year = 366.0 if is_leap_year else 365.0
    # Earth-Sun distance correction
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / days_in_year)

    # Solar declination (Spencer's formula)
    day_angle = 2.0 * np.pi * (doy - 1) / days_in_year
    decl = (
        0.006918
        - 0.399912 * np.cos(day_angle)
        + 0.070257 * np.sin(day_angle)
        - 0.006758 * np.cos(2.0 * day_angle)
        + 0.000907 * np.sin(2.0 * day_angle)
        - 0.002697 * np.cos(3.0 * day_angle)
        + 0.001480 * np.sin(3.0 * day_angle)
    )

    lat_rad = np.deg2rad(latitudes)

    # Sunset hour angle
    tan_prod = -np.tan(lat_rad) * np.tan(decl)
    # Polar day (sun never sets) → ws = pi; polar night → ws = 0
    ws = np.where(tan_prod < -1, np.pi, np.where(tan_prod > 1, 0.0, np.arccos(np.clip(tan_prod, -1, 1))))

    # FAO-56 Eq. 21  (result in W·min / m² / day)
    ra = (
        (24.0 * 60.0 / np.pi)
        * SOLAR_CONSTANT
        * dr
        * (ws * np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.sin(ws))
    )
    # Convert to daily-mean W/m²
    return ra / (24.0 * 60.0)


def compute_day_length(
    latitudes: np.ndarray,
    doy: int,
    is_leap_year: bool = False,
) -> np.ndarray:
    """Day length (hours) using the CBM model.

    Parameters
    ----------
    latitudes : 1-D array in degrees
    doy : day of year
    is_leap_year : whether the year is a leap year

    Returns
    -------
    1-D array of day lengths in hours.
    """
    days_in_year = 366 if is_leap_year else 365
    lat_rad = np.deg2rad(latitudes)
    decl = 0.4093 * np.sin(2.0 * np.pi * (doy - 81) / days_in_year)
    cos_ha = -np.tan(lat_rad) * np.tan(decl)
    cos_ha = np.clip(cos_ha, -1, 1)
    return 2.0 * np.arccos(cos_ha) * 12.0 / np.pi  # hours


def compute_precip_pet_ratio(
    precip: np.ndarray,
    pet: np.ndarray,
) -> np.ndarray:
    """Precipitation / PET ratio, clipped to [0, 10].

    PET from ERA5 is negative (loss convention); we take ``abs``.
    """
    abs_precip = np.abs(precip)
    abs_pet = np.maximum(np.abs(pet), 1e-10)
    return np.clip(abs_precip / abs_pet, 0.0, 10.0)


def compute_time_features(
    timestamps: pd.DatetimeIndex,
    longitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 'Day sin' and 'Year sin' based on local solar time.

    Matches the GEE script's ``add_time_features`` method.

    Parameters
    ----------
    timestamps : DatetimeIndex (UTC) — one per row
    longitudes : 1-D float array — one per row (same length as timestamps)

    Returns
    -------
    (day_sin, year_sin) arrays, each of shape ``(len(timestamps),)``
    """
    times_utc = timestamps.tz_localize("UTC") if timestamps.tz is None else timestamps
    doy = times_utc.dayofyear.values.astype(float)

    # Equation of time (Spencer approximation)
    B = np.deg2rad((360.0 / 365.0) * (doy - 81))
    eot_min = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)

    # Solar time offset (minutes)
    lon_offset_min = longitudes * 4.0
    total_offset_min = lon_offset_min + eot_min

    # Solar timestamp
    solar_ts = times_utc + pd.to_timedelta(total_offset_min, unit="m")

    solar_seconds = (solar_ts.hour * 3600 + solar_ts.minute * 60 + solar_ts.second).values.astype(float)
    day_sin = np.sin(2.0 * np.pi * solar_seconds / 86400.0)

    solar_doy = solar_ts.dayofyear.values.astype(float)
    year_frac = solar_doy + solar_seconds / 86400.0
    year_sin = np.sin(2.0 * np.pi * year_frac / 365.25)

    return day_sin, year_sin


# =========================================================================
# Static datasets (elevation, PFT, canopy height, LAI, climate normals)
# =========================================================================


def _read_raster_to_era5_grid(
    raster_path: Path,
    era5_lats: np.ndarray,
    era5_lons: np.ndarray,
) -> np.ndarray:
    """Read a GeoTIFF and resample it onto the ERA5-Land grid (nearest)."""
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        transform = src.transform
        nodata = src.nodata

    rows, cols = data.shape
    # Build pixel-centre coordinates from the transform
    src_lons = transform.c + (np.arange(cols) + 0.5) * transform.a
    src_lats = transform.f + (np.arange(rows) + 0.5) * transform.e

    # Nearest-neighbour resampling
    lat_idx = np.searchsorted(-src_lats, -era5_lats)  # descending → negate
    lon_idx = np.searchsorted(src_lons, era5_lons)

    lat_idx = np.clip(lat_idx, 0, rows - 1)
    lon_idx = np.clip(lon_idx, 0, cols - 1)

    out = data[np.ix_(lat_idx, lon_idx)].astype(np.float32)
    if nodata is not None:
        out[out == nodata] = np.nan
    return out


def load_static_datasets(
    era5_lats: np.ndarray,
    era5_lons: np.ndarray,
    base_dir: Path,
    lai_dir: Path,
    year: int,
    month: int,
) -> dict[str, np.ndarray]:
    """Load and resample all static / slowly-varying datasets.

    Returns a dict with keys:
        elevation, canopy_height, pft,
        mean_annual_temp, mean_annual_precip,
        LAI  (shape: nlat x nlon, monthly composite)
    """
    static: dict[str, np.ndarray] = {}

    # --- Elevation (SRTM / MERIT DEM) ---
    elev_candidates = [
        base_dir / "data" / "raw" / "grided" / "spatial_features" / "elevation" / "srtm_elevation.tif",
        base_dir / "data" / "raw" / "grided" / "spatial_features" / "elevation" / "elevation.tif",
    ]
    for p in elev_candidates:
        if p.exists():
            logger.info("Loading elevation from %s", p)
            static["elevation"] = _read_raster_to_era5_grid(p, era5_lats, era5_lons)
            break
    if "elevation" not in static:
        logger.warning("Elevation raster not found — filling with 0")
        static["elevation"] = np.zeros((len(era5_lats), len(era5_lons)), dtype=np.float32)

    # --- Canopy height ---
    ch_candidates = [
        base_dir / "data" / "raw" / "grided" / "spatial_features" / "canopy_height" / "canopy_height.tif",
    ]
    for p in ch_candidates:
        if p.exists():
            logger.info("Loading canopy height from %s", p)
            static["canopy_height"] = _read_raster_to_era5_grid(p, era5_lats, era5_lons)
            break
    if "canopy_height" not in static:
        logger.warning("Canopy height raster not found — filling with 0")
        static["canopy_height"] = np.zeros((len(era5_lats), len(era5_lons)), dtype=np.float32)

    # --- PFT (MODIS land cover) ---
    pft_candidates = [
        base_dir / "data" / "raw" / "grided" / "spatial_features" / "pft" / "pft.tif",
        base_dir / "data" / "raw" / "grided" / "spatial_features" / "pft" / "MODIS_PFT.tif",
    ]
    for p in pft_candidates:
        if p.exists():
            logger.info("Loading PFT from %s", p)
            static["pft"] = _read_raster_to_era5_grid(p, era5_lats, era5_lons)
            break
    if "pft" not in static:
        logger.warning("PFT raster not found — filling with 11 (barren)")
        static["pft"] = np.full((len(era5_lats), len(era5_lons)), 11, dtype=np.float32)

    # --- WorldClim climate normals ---
    temp_clim = Path(TEMP_CLIMATE_FILE)
    precip_clim = Path(PRECIP_CLIMATE_FILE)
    if not temp_clim.is_absolute():
        temp_clim = base_dir / temp_clim
    if not precip_clim.is_absolute():
        precip_clim = base_dir / precip_clim

    if temp_clim.exists():
        logger.info("Loading mean annual temperature from %s", temp_clim)
        static["mean_annual_temp"] = _read_raster_to_era5_grid(temp_clim, era5_lats, era5_lons)
    else:
        logger.warning("Climate temperature file not found — filling with 15")
        static["mean_annual_temp"] = np.full((len(era5_lats), len(era5_lons)), 15.0, dtype=np.float32)

    if precip_clim.exists():
        logger.info("Loading mean annual precipitation from %s", precip_clim)
        static["mean_annual_precip"] = _read_raster_to_era5_grid(precip_clim, era5_lats, era5_lons)
    else:
        logger.warning("Climate precipitation file not found — filling with 800")
        static["mean_annual_precip"] = np.full((len(era5_lats), len(era5_lons)), 800.0, dtype=np.float32)

    # --- LAI (GlobMap, monthly) ---
    lai_dir_path = Path(lai_dir)
    if not lai_dir_path.is_absolute():
        lai_dir_path = base_dir / lai_dir_path
    static["LAI"] = _load_lai_for_month(lai_dir_path, era5_lats, era5_lons, year, month)

    return static


def _load_lai_for_month(
    lai_dir: Path,
    era5_lats: np.ndarray,
    era5_lons: np.ndarray,
    year: int,
    month: int,
) -> np.ndarray:
    """Load GlobMap LAI for a given month, resampled to the ERA5 grid.

    GlobMap files follow the pattern ``GlobMapLAIV3.A{YYYY}{DOY}.*.tif``.
    We find the file whose DOY is closest to the 15th of the target month.
    """
    target_doy = pd.Timestamp(year, month, 15).dayofyear
    best_file: Path | None = None
    best_dist = 9999

    if lai_dir.exists():
        for f in lai_dir.glob("GlobMapLAIV3.A*.tif"):
            try:
                # Parse year and DOY from filename
                stem = f.stem  # e.g. GlobMapLAIV3.A2020015.Global.LAI
                a_idx = stem.index("A") + 1
                file_year = int(stem[a_idx : a_idx + 4])
                file_doy = int(stem[a_idx + 4 : a_idx + 7])
                if file_year == year:
                    dist = abs(file_doy - target_doy)
                    if dist < best_dist:
                        best_dist = dist
                        best_file = f
            except (ValueError, IndexError):
                continue

    if best_file is not None:
        logger.info("Loading LAI from %s (distance %d days)", best_file.name, best_dist)
        lai = _read_raster_to_era5_grid(best_file, era5_lats, era5_lons)
        # GlobMap stores LAI * 100 (scale factor 0.01); convert
        lai = np.where(lai > 0, lai * 0.01, 0.0)
        return lai.astype(np.float32)

    logger.warning("No GlobMap LAI file found for %d-%02d — filling with 0", year, month)
    return np.zeros((len(era5_lats), len(era5_lons)), dtype=np.float32)


# =========================================================================
# Core processing: build daily prediction CSVs
# =========================================================================


def _normalise_ds(ds: xr.Dataset) -> xr.Dataset:
    """Rename CDS short names (t2m, d2m, …) to internal convention.

    Also normalise coordinate names to ``latitude`` / ``longitude``.
    """
    rename_coords: dict[str, str] = {}
    for c in ds.coords:
        cl = c.lower()
        if cl == "lat":
            rename_coords[c] = "latitude"
        elif cl == "lon":
            rename_coords[c] = "longitude"
        elif cl == "valid_time":
            rename_coords[c] = "time"
    if rename_coords:
        ds = ds.rename(rename_coords)

    # Handle expver dimension (CDS includes experiment version for multi-version data)
    if "expver" in ds.dims or "expver" in ds.coords:
        ds = ds.sel(expver=ds["expver"].values[-1], drop=True)
        logger.debug("Dropped expver dimension, selected version %s", ds.attrs.get("expver", "last"))

    # Handle number dimension (ensemble member — deterministic downloads sometimes include it)
    if "number" in ds.dims:
        ds = ds.isel(number=0, drop=True)
        logger.debug("Dropped number dimension (ensemble member)")

    # Handle step dimension (forecast step — can appear in hourly requests)
    if "step" in ds.dims:
        ds = ds.isel(step=0, drop=True)
        logger.debug("Dropped step dimension")

    rename_vars: dict[str, str] = {}
    for v in ds.data_vars:
        if v in _CDS_SHORT_TO_INTERNAL:
            rename_vars[v] = _CDS_SHORT_TO_INTERNAL[v]
    if rename_vars:
        ds = ds.rename(rename_vars)

    return ds


def build_prediction_dataset(
    year: int,
    month: int,
    output_dir: Path,
    time_scale: str = "daily",
    validate: bool = False,
) -> list[Path]:
    """Orchestrate the full pipeline for one month.

    1. Download instantaneous + accumulated variables from CDS.
    2. Compute derived variables (VPD, wind speed, PPFD, ext_rad, …).
    3. Merge with static datasets.
    4. Write one CSV per day matching the GEE script's output contract.

    Returns list of written CSV paths.
    """
    # Month-level checkpoint — skip if already completed
    if _is_done(output_dir, year, month, "full"):
        logger.info("Month %d-%02d already complete (checkpoint found), skipping.", year, month)
        csv_dir = output_dir / "csv"
        return sorted(csv_dir.glob(f"prediction_{year}_{month:02d}_*.csv"))

    dl_dir = output_dir / "downloads" / f"{year}_{month:02d}"
    dl_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    client = _cds_client()

    # ------------------------------------------------------------------
    # Step 1: Download
    # ------------------------------------------------------------------
    if time_scale == "daily":
        logger.info("=== Downloading instantaneous daily-stat variables ===")
        inst_paths = download_instantaneous_vars(client, year, month, dl_dir)

        logger.info("=== Downloading accumulated variables ===")
        accum_path = download_accumulated_vars(client, year, month, dl_dir, time_scale="daily")

        # Load datasets
        ds_mean = _normalise_ds(xr.open_dataset(inst_paths["mean"]))
        ds_min = _normalise_ds(xr.open_dataset(inst_paths["minimum"]))
        ds_max = _normalise_ds(xr.open_dataset(inst_paths["maximum"]))
        ds_accum = _normalise_ds(xr.open_dataset(accum_path))

        lats = ds_mean["latitude"].values
        lons = ds_mean["longitude"].values
        times = pd.DatetimeIndex(ds_mean["time"].values)
    else:
        logger.info("=== Downloading hourly variables ===")
        accum_path = download_accumulated_vars(client, year, month, dl_dir, time_scale="hourly")
        ds_hourly = _normalise_ds(xr.open_dataset(accum_path))
        lats = ds_hourly["latitude"].values
        lons = ds_hourly["longitude"].values
        times = pd.DatetimeIndex(ds_hourly["time"].values)

    logger.info("Grid: %d lats × %d lons, %d timesteps", len(lats), len(lons), len(times))

    # ------------------------------------------------------------------
    # Step 2: Load static data
    # ------------------------------------------------------------------
    base_dir = Path(__file__).resolve().parent.parent.parent
    logger.info("Loading static datasets (base_dir=%s)", base_dir)
    static = load_static_datasets(lats, lons, base_dir, LAI_DATA_DIR, year, month)

    # ------------------------------------------------------------------
    # Step 3: Compute derived variables and write CSVs day by day
    # ------------------------------------------------------------------
    written_paths: list[Path] = []

    if time_scale == "daily":
        written_paths = _process_daily(
            ds_mean,
            ds_min,
            ds_max,
            ds_accum,
            lats,
            lons,
            times,
            static,
            csv_dir,
            year,
            month,
            validate,
        )
        # Close datasets
        for ds in (ds_mean, ds_min, ds_max, ds_accum):
            ds.close()
    else:
        written_paths = _process_hourly(
            ds_hourly,
            lats,
            lons,
            times,
            static,
            csv_dir,
            year,
            month,
            validate,
        )
        ds_hourly.close()

    gc.collect()
    logger.info("=== Month %d-%02d complete: %d files written ===", year, month, len(written_paths))

    # Mark month as fully processed for checkpoint/resume
    _mark_done(output_dir, year, month, "full")
    return written_paths


def _process_daily(
    ds_mean: xr.Dataset,
    ds_min: xr.Dataset,
    ds_max: xr.Dataset,
    ds_accum: xr.Dataset,
    lats: np.ndarray,
    lons: np.ndarray,
    times: pd.DatetimeIndex,
    static: dict[str, np.ndarray],
    csv_dir: Path,
    year: int,
    month: int,
    validate: bool,
) -> list[Path]:
    """Process daily-scale data and write per-day CSVs."""
    written: list[Path] = []
    nlat, nlon = len(lats), len(lons)

    for t_idx, ts in enumerate(times):
        day = ts.day
        doy = ts.dayofyear
        is_leap = calendar.isleap(ts.year)

        csv_path = csv_dir / f"prediction_{year}_{month:02d}_{day:02d}.csv"
        if csv_path.exists():
            logger.info("Skipping existing %s", csv_path.name)
            written.append(csv_path)
            continue

        logger.info("Processing %s (DOY %d) ...", ts.strftime("%Y-%m-%d"), doy)

        # --- Extract slices (Kelvin for temperatures) ---
        ta_mean_k = ds_mean["temperature_2m"].isel(time=t_idx).values  # (lat, lon)
        ta_min_k = ds_min["temperature_2m"].isel(time=t_idx).values
        ta_max_k = ds_max["temperature_2m"].isel(time=t_idx).values

        td_mean_k = ds_mean["dewpoint_temperature_2m"].isel(time=t_idx).values
        td_min_k = ds_min["dewpoint_temperature_2m"].isel(time=t_idx).values
        td_max_k = ds_max["dewpoint_temperature_2m"].isel(time=t_idx).values

        u_mean = ds_mean["10m_u_component_of_wind"].isel(time=t_idx).values
        v_mean = ds_mean["10m_v_component_of_wind"].isel(time=t_idx).values
        u_min = ds_min["10m_u_component_of_wind"].isel(time=t_idx).values
        v_min = ds_min["10m_v_component_of_wind"].isel(time=t_idx).values
        u_max = ds_max["10m_u_component_of_wind"].isel(time=t_idx).values
        v_max = ds_max["10m_v_component_of_wind"].isel(time=t_idx).values

        swvl1 = ds_mean["volumetric_soil_water_layer_1"].isel(time=t_idx).values
        stl1_k = ds_mean["soil_temperature_level_1"].isel(time=t_idx).values

        # Accumulated vars (already shifted by -1 day during download)
        # Check if this day exists in the accumulated dataset
        accum_time_sel = ds_accum["time"].values
        accum_match = np.where(pd.DatetimeIndex(accum_time_sel).normalize() == ts.normalize())[0]

        if len(accum_match) > 0:
            ai = accum_match[0]
            precip_m = ds_accum["total_precipitation"].isel(time=ai).values  # metres
            ssrd_j = ds_accum["surface_solar_radiation_downwards"].isel(time=ai).values  # J/m²
            pev_m = ds_accum["potential_evaporation"].isel(time=ai).values  # metres
        else:
            logger.warning("No accumulated data for %s — filling with 0", ts.strftime("%Y-%m-%d"))
            precip_m = np.zeros((nlat, nlon), dtype=np.float32)
            ssrd_j = np.zeros((nlat, nlon), dtype=np.float32)
            pev_m = np.zeros((nlat, nlon), dtype=np.float32)

        # --- Unit conversions ---
        ta_mean_c = ta_mean_k - KELVIN_OFFSET  # K → °C
        ta_min_c = ta_min_k - KELVIN_OFFSET
        ta_max_c = ta_max_k - KELVIN_OFFSET
        stl1_c = stl1_k - KELVIN_OFFSET  # soil temp K → °C

        # Solar radiation: J/m² accumulated over 24h → daily-mean W/m²
        sw_in = np.maximum(ssrd_j, 0.0) / 86400.0

        # Precipitation: m → mm
        precip_mm = np.abs(precip_m) * 1000.0

        # --- Derived variables ---
        vpd_mean = compute_vpd_kpa(ta_mean_k, td_mean_k)
        vpd_max = compute_vpd_kpa(ta_max_k, td_min_k)  # hottest + driest
        vpd_min = compute_vpd_kpa(ta_min_k, td_max_k)  # coolest + most humid

        ws_mean = compute_wind_speed(u_mean, v_mean)
        # NOTE: ws_min/ws_max are APPROXIMATIONS — component min/max are temporally
        # independent, so sqrt(u_min²+v_min²) ≠ min(sqrt(u²+v²)) over the day.
        # This matches the GEE script's approach for consistency.
        ws_min = compute_wind_speed(u_min, v_min)  # approximate lower bound
        ws_max = compute_wind_speed(u_max, v_max)  # approximate upper bound

        ppfd_in = compute_ppfd(sw_in)
        ext_rad_1d = compute_ext_rad_daily(lats, doy, is_leap_year=is_leap)  # (nlat,)
        ext_rad_2d = np.broadcast_to(
            ext_rad_1d[:, np.newaxis],  # (nlat, nlon)
            (nlat, nlon),
        )

        day_len_1d = compute_day_length(lats, doy, is_leap)
        day_len_2d = np.broadcast_to(day_len_1d[:, np.newaxis], (nlat, nlon))

        precip_pet = compute_precip_pet_ratio(precip_m, pev_m)

        # --- Build flat DataFrame ---
        lon_grid, lat_grid = np.meshgrid(lons, lats)  # each (nlat, nlon)
        n = nlat * nlon
        timestamp_col = np.full(n, ts.strftime("%Y-%m-%d"))

        df = pd.DataFrame(
            {
                "latitude": lat_grid.ravel(),
                "longitude": lon_grid.ravel(),
                "timestamp": timestamp_col,
                "ta": ta_mean_c.ravel(),
                "ta_min": ta_min_c.ravel(),
                "ta_max": ta_max_c.ravel(),
                "vpd": vpd_mean.ravel(),
                "vpd_min": vpd_min.ravel(),
                "vpd_max": vpd_max.ravel(),
                "sw_in": sw_in.ravel(),
                "ppfd_in": ppfd_in.ravel(),
                "ext_rad": ext_rad_2d.ravel(),
                "ws": ws_mean.ravel(),
                "ws_min": ws_min.ravel(),
                "ws_max": ws_max.ravel(),
                "precip": precip_mm.ravel(),
                "volumetric_soil_water_layer_1": swvl1.ravel(),
                "soil_temperature_level_1": stl1_c.ravel(),
                "LAI": static["LAI"].ravel(),
                "prcip/PET": precip_pet.ravel(),
                "day_length": day_len_2d.ravel(),
                "elevation": static["elevation"].ravel(),
                "canopy_height": static["canopy_height"].ravel(),
                "annual_mean_temperature": static["mean_annual_temp"].ravel(),
                "annual_precipitation": static["mean_annual_precip"].ravel(),
            }
        )

        # PFT one-hot encoding (must match training: MF, DNF, ENF, EBF, WSA, WET, DBF, SAV)
        pft_flat = static["pft"].ravel().astype(int)
        # MODIS IGBP PFT classes mapped to the 8 biome types used in training
        # (the exact mapping depends on your training pipeline; replicate it here)
        pft_names = ["MF", "DNF", "ENF", "EBF", "WSA", "WET", "DBF", "SAV"]
        for pft_name in pft_names:
            df[pft_name] = 0
        # Simplified MODIS IGBP → one-hot (same as training extract_pft.py)
        # IGBP→PFT mapping — must match GEE script exactly (codes 6,7 unmapped)
        _IGBP_TO_PFT = {
            1: "ENF",
            2: "EBF",
            3: "DNF",
            4: "DBF",
            5: "MF",
            8: "WSA",
            9: "SAV",
            11: "WET",
        }
        for igbp_code, pft_name in _IGBP_TO_PFT.items():
            mask = pft_flat == igbp_code
            if mask.any():
                df.loc[mask, pft_name] = 1

        # Cyclical time features
        ts_index = pd.DatetimeIndex([ts] * n)
        day_sin, year_sin = compute_time_features(ts_index, df["longitude"].values)
        df["Day sin"] = day_sin
        df["Year sin"] = year_sin

        # Drop rows that are all-NaN (ocean / missing)
        df.dropna(subset=["ta", "sw_in", "volumetric_soil_water_layer_1"], inplace=True)

        # --- Validation (optional) ---
        if validate:
            _validate_daily_row(df, ts)

        # --- Save ---
        df.to_csv(csv_path, index_label="timestamp")
        written.append(csv_path)
        logger.info("Wrote %s (%d rows)", csv_path.name, len(df))

        # Free memory
        del df
        gc.collect()

    return written


def _process_hourly(
    ds: xr.Dataset,
    lats: np.ndarray,
    lons: np.ndarray,
    times: pd.DatetimeIndex,
    static: dict[str, np.ndarray],
    csv_dir: Path,
    year: int,
    month: int,
    validate: bool,
) -> list[Path]:
    """Process hourly data — groups by day and writes per-day CSVs."""
    written: list[Path] = []
    nlat, nlon = len(lats), len(lons)

    # Group timestamps by date
    dates = np.unique(times.date)

    for date in dates:
        day = date.day
        csv_path = csv_dir / f"prediction_{year}_{month:02d}_{day:02d}.csv"
        if csv_path.exists():
            logger.info("Skipping existing %s", csv_path.name)
            written.append(csv_path)
            continue

        day_mask = times.date == date
        day_times = times[day_mask]
        day_indices = np.where(day_mask)[0]

        logger.info("Processing %s (%d hours) ...", date, len(day_times))

        day_frames: list[pd.DataFrame] = []

        for hour_offset, (t_idx, ts) in enumerate(zip(day_indices, day_times)):
            doy = ts.dayofyear
            is_leap = calendar.isleap(ts.year)

            # Extract variables for this hour
            ta_k = ds["temperature_2m"].isel(time=t_idx).values
            td_k = ds["dewpoint_temperature_2m"].isel(time=t_idx).values
            u_val = ds["10m_u_component_of_wind"].isel(time=t_idx).values
            v_val = ds["10m_v_component_of_wind"].isel(time=t_idx).values
            swvl1 = ds["volumetric_soil_water_layer_1"].isel(time=t_idx).values
            stl1_k = ds["soil_temperature_level_1"].isel(time=t_idx).values
            precip_m = ds["total_precipitation"].isel(time=t_idx).values
            ssrd_j = ds["surface_solar_radiation_downwards"].isel(time=t_idx).values
            pev_m = ds["potential_evaporation"].isel(time=t_idx).values

            ta_c = ta_k - KELVIN_OFFSET
            stl1_c = stl1_k - KELVIN_OFFSET
            sw_in = np.maximum(ssrd_j, 0.0) / 3600.0  # J/m² per hour → W/m²
            precip_mm = np.abs(precip_m) * 1000.0

            vpd = compute_vpd_kpa(ta_k, td_k)
            ws = compute_wind_speed(u_val, v_val)
            ppfd_in = compute_ppfd(sw_in)

            ext_rad_1d = compute_ext_rad_daily(lats, doy, is_leap_year=is_leap)
            ext_rad_2d = np.broadcast_to(ext_rad_1d[:, np.newaxis], (nlat, nlon))
            day_len_1d = compute_day_length(lats, doy, is_leap)
            day_len_2d = np.broadcast_to(day_len_1d[:, np.newaxis], (nlat, nlon))
            precip_pet = compute_precip_pet_ratio(precip_m, pev_m)

            lon_grid, lat_grid = np.meshgrid(lons, lats)
            n = nlat * nlon

            frame = pd.DataFrame(
                {
                    "latitude": lat_grid.ravel(),
                    "longitude": lon_grid.ravel(),
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "ta": ta_c.ravel(),
                    "ta_min": ta_c.ravel(),  # hourly: same as mean
                    "ta_max": ta_c.ravel(),
                    "vpd": vpd.ravel(),
                    "vpd_min": vpd.ravel(),
                    "vpd_max": vpd.ravel(),
                    "sw_in": sw_in.ravel(),
                    "ppfd_in": ppfd_in.ravel(),
                    "ext_rad": ext_rad_2d.ravel(),
                    "ws": ws.ravel(),
                    "ws_min": ws.ravel(),
                    "ws_max": ws.ravel(),
                    "precip": precip_mm.ravel(),
                    "volumetric_soil_water_layer_1": swvl1.ravel(),
                    "soil_temperature_level_1": stl1_c.ravel(),
                    "LAI": static["LAI"].ravel(),
                    "prcip/PET": precip_pet.ravel(),
                    "day_length": day_len_2d.ravel(),
                    "elevation": static["elevation"].ravel(),
                    "canopy_height": static["canopy_height"].ravel(),
                    "annual_mean_temperature": static["mean_annual_temp"].ravel(),
                    "annual_precipitation": static["mean_annual_precip"].ravel(),
                }
            )

            # PFT one-hot
            pft_flat = static["pft"].ravel().astype(int)
            pft_names = ["MF", "DNF", "ENF", "EBF", "WSA", "WET", "DBF", "SAV"]
            for pft_name in pft_names:
                frame[pft_name] = 0
            # IGBP→PFT mapping — must match GEE script exactly (codes 6,7 unmapped)
            _IGBP_TO_PFT = {
                1: "ENF",
                2: "EBF",
                3: "DNF",
                4: "DBF",
                5: "MF",
                8: "WSA",
                9: "SAV",
                11: "WET",
            }
            for igbp_code, pft_name in _IGBP_TO_PFT.items():
                mask = pft_flat == igbp_code
                if mask.any():
                    frame.loc[mask, pft_name] = 1

            # Time features
            ts_idx = pd.DatetimeIndex([ts] * n)
            day_sin, year_sin = compute_time_features(ts_idx, frame["longitude"].values)
            frame["Day sin"] = day_sin
            frame["Year sin"] = year_sin

            frame.dropna(subset=["ta", "sw_in", "volumetric_soil_water_layer_1"], inplace=True)
            day_frames.append(frame)

        if day_frames:
            df_day = pd.concat(day_frames, ignore_index=True)
            df_day.to_csv(csv_path, index_label="timestamp")
            written.append(csv_path)
            logger.info("Wrote %s (%d rows)", csv_path.name, len(df_day))
            del df_day, day_frames
            gc.collect()

    return written


# =========================================================================
# Validation helpers
# =========================================================================


def _validate_daily_row(df: pd.DataFrame, ts: pd.Timestamp) -> None:
    """Run sanity checks on a day's output DataFrame."""
    checks_passed = 0
    checks_total = 0

    def _check(name: str, condition: bool) -> None:
        nonlocal checks_passed, checks_total
        checks_total += 1
        if condition:
            checks_passed += 1
        else:
            logger.warning("VALIDATION FAIL [%s] on %s", name, ts.strftime("%Y-%m-%d"))

    _check("ta range", df["ta"].between(-90, 60).all() if "ta" in df.columns else False)
    _check("vpd non-negative", (df["vpd"] >= 0).all() if "vpd" in df.columns else False)
    _check("sw_in non-negative", (df["sw_in"] >= 0).all() if "sw_in" in df.columns else False)
    _check("ws non-negative", (df["ws"] >= 0).all() if "ws" in df.columns else False)
    _check("ext_rad non-negative", (df["ext_rad"] >= 0).all() if "ext_rad" in df.columns else False)
    _check("day_length 0-24", df["day_length"].between(0, 24).all() if "day_length" in df.columns else False)
    _check("LAI non-negative", (df["LAI"] >= 0).all() if "LAI" in df.columns else False)
    _check("precip non-negative", (df["precip"] >= 0).all() if "precip" in df.columns else False)
    _check("prcip/PET 0-10", df["prcip/PET"].between(0, 10).all() if "prcip/PET" in df.columns else False)

    logger.info("Validation: %d/%d checks passed for %s", checks_passed, checks_total, ts.strftime("%Y-%m-%d"))


# =========================================================================
# CLI entry point
# =========================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download and process ERA5-Land data via CDS API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--year", type=int, required=True, help="Year to process")
    p.add_argument("--month", type=int, required=True, help="Month to process (1-12)")
    p.add_argument("--time-scale", choices=["daily", "hourly"], default="daily", help="Temporal resolution")
    p.add_argument("--output-dir", type=str, default=None, help="Output directory (default: config.ERA5LAND_TEMP_DIR)")
    p.add_argument("--validate", action="store_true", help="Run validation checks on each output file")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _setup_logging(verbose=args.verbose)

    if not 1 <= args.month <= 12:
        raise ValueError(f"Month must be 1-12, got {args.month}")

    output_dir = Path(args.output_dir) if args.output_dir else ERA5LAND_TEMP_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Processing ERA5-Land via CDS: year=%d month=%d scale=%s", args.year, args.month, args.time_scale)
    logger.info("Output directory: %s", output_dir)

    written = build_prediction_dataset(
        year=args.year,
        month=args.month,
        output_dir=output_dir,
        time_scale=args.time_scale,
        validate=args.validate,
    )

    logger.info("Done. %d CSV files written to %s/csv/", len(written), output_dir)


if __name__ == "__main__":
    main()
