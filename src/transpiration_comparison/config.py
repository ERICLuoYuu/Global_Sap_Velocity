from __future__ import annotations

"""Configuration for transpiration product comparison pipeline."""

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


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
    """Every location the pipeline reads or writes, derived from `base`.

    Each field defaults to None and is filled from `base` in __post_init__.
    Before, only `base` was settable and every other field kept a hardwired
    `/scratch/tmp/yluo2/gsv/...` value, so `--base-dir` moved nothing; and
    figures_dir/reports_dir were RELATIVE, so the output location was decided
    by the caller's cwd. `run_tag`, when set, gives this run its own cache and
    figure directories instead of overwriting the previous run's.
    """

    base: Path = _SCRATCH / "gsv"
    run_tag: str = ""
    worktree: Path = _SCRATCH / "gsv-wt" / "map-viz"
    predictions_parquet: Optional[Path] = None
    serving_store: Optional[Path] = None
    products_dir: Optional[Path] = None
    preprocessed_dir: Optional[Path] = None
    figures_dir: Optional[Path] = None
    reports_dir: Optional[Path] = None

    def __post_init__(self):
        self.base = Path(self.base)
        if self.predictions_parquet is None:
            self.predictions_parquet = self.base / "outputs" / "data_for_prediction" / "stage2_no_precip"
        if self.serving_store is None:
            self.serving_store = self.base / "outputs" / "serving" / "store_v1"
        if self.products_dir is None:
            self.products_dir = self.base / "data" / "transpiration_products"
        if self.preprocessed_dir is None:
            self.preprocessed_dir = self.base / "data" / "transpiration_preprocessed"
        if self.figures_dir is None:
            self.figures_dir = self.base / "outputs" / "figures" / "transpiration_comparison"
        if self.reports_dir is None:
            self.reports_dir = self.base / "outputs" / "transpiration_comparison"
        for name in ("predictions_parquet", "serving_store", "products_dir",
                     "preprocessed_dir", "figures_dir", "reports_dir"):
            setattr(self, name, Path(getattr(self, name)))
        if self.run_tag:
            self.preprocessed_dir = self.preprocessed_dir / self.run_tag
            self.figures_dir = self.figures_dir / self.run_tag
            self.reports_dir = self.reports_dir / self.run_tag

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

# --- Validity thresholds ---
#
# These are applied to whatever frame preprocessing produced, which is MONTHLY
# (pipeline collapses anything longer than 100 steps to monthly means). The old
# hard-coded 30 was a day count: at 36 months it kept 30/36, but a one-year run
# has 12 months and 30 masked every pixel on Earth, silently. The fraction below
# is 30/36 exactly, so a 2016-2018 run reproduces the published mask.
MIN_VALID_FRACTION = 5.0 / 6.0
MIN_VALID_FLOOR = 6

# Retained for callers that still import it; prefer min_valid_timesteps().
MIN_VALID_TIMESTEPS = 30


def min_valid_timesteps(
    n_time: int,
    fraction: float = MIN_VALID_FRACTION,
    floor: int = MIN_VALID_FLOOR,
) -> int:
    """Minimum non-NaN steps a pixel needs, scaled to the frame it is in."""
    if n_time <= 0:
        raise ValueError(f"n_time must be positive, got {n_time}")
    return min(n_time, max(floor, math.ceil(fraction * n_time)))
