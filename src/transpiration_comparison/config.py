from __future__ import annotations

"""Configuration for transpiration product comparison pipeline."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SpatialDomain:
    lat_min: float = -60.0
    lat_max: float = 78.0
    lon_min: float = -180.0
    lon_max: float = 180.0
    resolution: float = 0.1


@dataclass(frozen=True)
class TimePeriod:
    start: str = "2016-01-01"
    end: str = "2018-12-31"


@dataclass(frozen=True)
class ProductConfig:
    name: str
    short_name: str
    variable: str
    units: str
    native_resolution: float
    native_temporal: str  # 'hourly', 'daily', '8-day', '3-hourly'
    source_type: str  # 'sftp', 'cds', 'gee', 'opendap', 'local'
    enabled: bool = True


@dataclass(frozen=True)
class Region:
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


# --- Product Definitions ---

PRODUCTS = {
    "sap_velocity": ProductConfig(
        "Sap Velocity (XGBoost)",
        "sapv",
        "sap_velocity_xgb",
        "cm3/cm2/h",
        0.1,
        "daily",
        "local",
    ),
    "gleam": ProductConfig(
        "GLEAM v4.2a",
        "gleam",
        "Et",
        "mm/day",
        0.1,
        "daily",
        "sftp",
    ),
    "era5land": ProductConfig(
        "ERA5-Land",
        "era5",
        "e_from_vegetation_transpiration",
        "m/day",
        0.1,
        "hourly",
        "cds",
    ),
    "pmlv2": ProductConfig(
        "PMLv2",
        "pmlv2",
        "Ec",
        "mm/day",
        0.1,
        "daily",
        "gee",
    ),
    "gldas": ProductConfig(
        "GLDAS Noah v2.1",
        "gldas",
        "Tveg_tavg",
        "kg/m2/s",
        0.25,
        "3-hourly",
        "opendap",
    ),
}

# --- Region Definitions ---

REGIONS = {
    "amazon": Region("Amazon", -15, 5, -75, -45),
    "sahel": Region("Sahel", 5, 20, -15, 40),
    "central_eu": Region("Central Europe", 45, 55, 5, 20),
    "boreal_scan": Region("Boreal Scandinavia", 60, 70, 10, 30),
    "mediterranean": Region("Mediterranean", 35, 45, -5, 20),
    "se_asia": Region("Southeast Asia", -10, 25, 95, 130),
}

# --- PFT Definitions ---

PFT_COLUMNS = {
    "ENF": "Evergreen Needleleaf",
    "EBF": "Evergreen Broadleaf",
    "DNF": "Deciduous Needleleaf",
    "DBF": "Deciduous Broadleaf",
    "MF": "Mixed Forest",
    "WSA": "Woody Savanna",
    "SAV": "Savanna",
    "WET": "Wetland",
}

# --- Path Configuration ---

_SCRATCH = Path(os.environ.get("SCRATCH_DIR", "/scratch/tmp/yluo2"))


@dataclass
class Paths:
    base: Path = _SCRATCH / "gsv"
    worktree: Path = _SCRATCH / "gsv-wt" / "map-viz"
    predictions_parquet: Path = _SCRATCH / "gsv" / "outputs" / "data_for_prediction" / "stage2_no_precip"
    predictions_geotiff: Path = (
        _SCRATCH / "gsv" / "outputs" / "data_for_prediction" / "stage3_maps" / "no_precip_2016_2018"
    )
    products_dir: Path = _SCRATCH / "gsv" / "data" / "transpiration_products"
    preprocessed_dir: Path = _SCRATCH / "gsv" / "data" / "transpiration_preprocessed"
    figures_dir: Path = Path("outputs/figures/transpiration_comparison")
    reports_dir: Path = Path("outputs/transpiration_comparison")

    def ensure_dirs(self):
        for d in [self.products_dir, self.preprocessed_dir, self.figures_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)
        for sub in ["global_maps", "pft_stratified", "temporal", "regional", "summary"]:
            (self.figures_dir / sub).mkdir(parents=True, exist_ok=True)


# --- Common Grid ---


def make_common_grid(domain: SpatialDomain = SpatialDomain()):
    """Create the target lat/lon coordinate arrays for regridding."""
    import numpy as np

    half = domain.resolution / 2
    lat = np.arange(domain.lat_min + half, domain.lat_max, domain.resolution)
    lon = np.arange(domain.lon_min + half, domain.lon_max, domain.resolution)
    return lat, lon


# --- Unit Conversion Factors ---

UNIT_TO_MM_DAY = {
    "mm/day": 1.0,
    "m/day": 1000.0,  # ERA5-Land: meters -> mm
    "kg/m2/s": 86400.0,  # GLDAS: kg/m2/s -> mm/day (water density ~1000 kg/m3)
    "cm3/cm2/h": None,  # Sap velocity: NOT convertible to mm/day (different quantity)
}


# --- Default Instances ---

DEFAULT_DOMAIN = SpatialDomain()
DEFAULT_PERIOD = TimePeriod()
DEFAULT_PATHS = Paths()

# Minimum valid days for Z-score computation
MIN_VALID_TIMESTEPS = 30
