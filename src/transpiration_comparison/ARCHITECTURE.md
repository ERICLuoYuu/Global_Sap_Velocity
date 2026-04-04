# Architecture: Transpiration Product Comparison Pipeline

## Overview

Compare our global sap velocity predictions (0.1 deg daily, 2016-2018) against established
transpiration products (GLEAM, ERA5-Land, PMLv2, GLDAS) following evaluation frameworks from
Bittencourt et al. (2023) and Li et al. (2024).

**Key constraint:** Sap velocity (cm3 cm-2 h-1) and transpiration (mm/day) have different
units. All comparisons use Z-score normalized temporal patterns, not absolute values.

## Module Dependency Graph

```
config.py
    |
    v
products/          --> preprocessing/     --> comparison/        --> visualization/
  gleam.py              regrid.py              spatial.py            maps.py
  era5land.py           temporal.py            temporal_comp.py      taylor.py
  pmlv2.py              normalize.py           regional.py           timeseries.py
  gldas.py                                     agreement.py          summary.py
  sap_velocity.py                              collocation.py
    |                       |                       |                    |
    v                       v                       v                    v
[NetCDF on disk]    [Regridded NetCDF]     [Metric DataFrames]    [PNG/PDF figures]
                                                    |
                                                    v
                                              reporting/generate.py --> [Markdown reports]

cli.py  -- entry point, dispatches to individual phases
pipeline.py  -- orchestrates full sequential run
```

Execution order: products -> preprocessing -> comparison -> visualization -> reporting

## Directory Layout

```
src/transpiration_comparison/
|-- __init__.py
|-- config.py                  # All configuration (products, regions, PFTs, paths)
|-- cli.py                     # CLI entry point (argparse)
|-- pipeline.py                # Full pipeline orchestrator
|
|-- products/                  # Phase 2: Data acquisition
|   |-- __init__.py
|   |-- base.py               # ProductBase ABC
|   |-- gleam.py              # GLEAM v4.2 SFTP download
|   |-- era5land.py           # ERA5-Land CDS API download
|   |-- pmlv2.py              # PMLv2 GEE export
|   |-- gldas.py              # GLDAS OPeNDAP download
|   +-- sap_velocity.py       # Our predictions: parquet -> xarray
|
|-- preprocessing/             # Regridding + normalization
|   |-- __init__.py
|   |-- regrid.py             # xESMF regridding to common 0.1 deg grid
|   |-- temporal.py           # Temporal aggregation (hourly->daily, 8-day->daily)
|   +-- normalize.py          # Z-score, anomaly computation
|
|-- comparison/                # Phases 3-6: Analysis
|   |-- __init__.py
|   |-- spatial.py            # Spatial pattern comparison
|   |-- temporal_comp.py      # Temporal pattern comparison
|   |-- regional.py           # Region-level deep dives
|   |-- agreement.py          # Multi-product agreement
|   +-- collocation.py        # Triple collocation analysis
|
|-- visualization/             # Phase 7: Figures
|   |-- __init__.py
|   |-- maps.py               # Global maps (cartopy)
|   |-- taylor.py             # Taylor diagrams (SkillMetrics)
|   |-- timeseries.py         # Time series, seasonal cycles
|   +-- summary.py            # Dashboard/summary figures
|
+-- slurm/                     # SLURM job scripts
    |-- install_deps.sh        # One-time package installation
    |-- download_products.sh   # Phase 2 download jobs
    |-- run_comparison.sh      # Phases 3-6 compute
    +-- generate_figures.sh    # Phase 7 visualization
```

Output layout:
```
outputs/
|-- transpiration_comparison/         # Reports (Phase 8)
|   |-- product_inventory.md
|   |-- spatial_comparison.md
|   |-- temporal_comparison.md
|   |-- regional_deep_dives.md
|   |-- agreement_analysis.md
|   |-- comparison_report.md
|   +-- methodology_notes.md
|
+-- figures/transpiration_comparison/ # Figures (Phase 7)
    |-- global_maps/
    |-- pft_stratified/
    |-- temporal/
    |-- regional/
    +-- summary/
```

## Key Data Structures

### Primary Format: xarray.Dataset

All products are converted to xarray.Dataset with standardized coordinates and variables.

```python
# Standard product dataset after preprocessing
<xarray.Dataset>
Dimensions:              (time: 1096, lat: 1380, lon: 3600)
Coordinates:
  * time                 (time) datetime64[ns]    2016-01-01 ... 2018-12-31
  * lat                  (lat) float64            -60.05 ... 77.95
  * lon                  (lon) float64            -179.95 ... 179.95
Data variables:
    transpiration        (time, lat, lon) float32  ...   # mm/day (or native unit)
    transpiration_zscore (time, lat, lon) float32  ...   # Z-score normalized
Attributes:
    product:     'gleam_v4.2'
    variable:    'Et'
    units:       'mm/day'
    resolution:  0.1
    source_url:  'https://www.gleam.eu'
```

### Sap Velocity Dataset (special case)

```python
# Our predictions -- same structure, different variable name & unit
<xarray.Dataset>
Dimensions:              (time: 1096, lat: 1380, lon: 3600)
Data variables:
    sap_velocity         (time, lat, lon) float32  ...   # cm3 cm-2 h-1
    sap_velocity_zscore  (time, lat, lon) float32  ...   # Z-score normalized
Attributes:
    product:     'sap_velocity_xgb'
    units:       'cm3 cm-2 h-1'
```

### Comparison Results: pandas DataFrame + xarray

```python
# Per-pixel metrics (spatial comparison) -> xarray
<xarray.Dataset>
Dimensions:    (lat: 1380, lon: 3600)
Data variables:
    pearson_r  (lat, lon) float32     # Temporal correlation per pixel
    rmse       (lat, lon) float32     # RMSE of Z-scores
    mae        (lat, lon) float32
    bias       (lat, lon) float32

# PFT-stratified summary -> pandas DataFrame
         | pearson_r | rmse  | mae   | seasonal_r | peak_month_diff | n_pixels
---------|-----------|-------|-------|------------|-----------------|--------
ENF      | 0.72      | 0.45  | 0.33  | 0.85       | 0.2             | 125000
DBF      | 0.68      | 0.51  | 0.38  | 0.91       | 0.1             | 98000
...
```

### Naming Conventions

- Coordinates: `lat`, `lon`, `time` (always)
- Product variable: `transpiration` (for transpiration products), `sap_velocity` (for ours)
- Z-scored: append `_zscore` suffix
- Anomaly: append `_anomaly` suffix
- Climatology: append `_clim` suffix
- File naming: `{product}_{variable}_{resolution}_{start}_{end}.nc`
  - Example: `gleam_v4.2_Et_0.1deg_2016_2018.nc`

## Configuration Schema

```python
# config.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

@dataclass(frozen=True)
class SpatialDomain:
    lat_min: float = -60.0
    lat_max: float = 78.0
    lon_min: float = -180.0
    lon_max: float = 180.0
    resolution: float = 0.1  # degrees

@dataclass(frozen=True)
class TimePeriod:
    start: str = "2016-01-01"
    end: str = "2018-12-31"

@dataclass(frozen=True)
class ProductConfig:
    name: str
    variable: str
    units: str
    native_resolution: float     # degrees
    native_temporal: str         # 'hourly', 'daily', '8-day', '3-hourly'
    source_type: str             # 'sftp', 'cds', 'gee', 'opendap', 'local'
    enabled: bool = True

@dataclass(frozen=True)
class Region:
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

PRODUCTS = {
    "gleam": ProductConfig("GLEAM v4.2", "Et", "mm/day", 0.1, "daily", "sftp"),
    "era5land": ProductConfig("ERA5-Land", "evaporation_from_vegetation_transpiration",
                              "m/day", 0.1, "hourly", "cds"),
    "pmlv2": ProductConfig("PMLv2", "Ec", "mm/day", 0.05, "8-day", "gee"),
    "gldas": ProductConfig("GLDAS Noah v2.1", "Tveg_tavg", "kg/m2/s", 0.25, "3-hourly",
                           "opendap"),
    "sap_velocity": ProductConfig("Sap Velocity (XGBoost)", "sap_velocity_xgb",
                                   "cm3/cm2/h", 0.1, "daily", "local"),
}

REGIONS = {
    "amazon":       Region("Amazon",              -15,   5, -75, -45),
    "sahel":        Region("Sahel",                 5,  20, -15,  40),
    "central_eu":   Region("Central Europe",       45,  55,   5,  20),
    "boreal_scan":  Region("Boreal Scandinavia",   60,  70,  10,  30),
    "mediterranean": Region("Mediterranean",       35,  45,  -5,  20),
    "se_asia":      Region("Southeast Asia",       -10,  25,  95, 130),
}

# PFT mapping: column name -> display name
PFT_COLUMNS = {
    "ENF": "Evergreen Needleleaf",
    "EBF": "Evergreen Broadleaf",
    "DNF": "Deciduous Needleleaf",
    "DBF": "Deciduous Broadleaf",
    "MF":  "Mixed Forest",
    "WSA": "Woody Savanna",
    "SAV": "Savanna",
    "WET": "Wetland",
}

# Paths (HPC)
@dataclass
class Paths:
    base: Path = Path("/scratch/tmp/yluo2/gsv")
    predictions_parquet: Path = Path("/scratch/tmp/yluo2/gsv/outputs/data_for_prediction/stage2_no_precip")
    predictions_geotiff: Path = Path("/scratch/tmp/yluo2/gsv/outputs/data_for_prediction/stage3_maps/no_precip_2016_2018")
    products_dir: Path = Path("/scratch/tmp/yluo2/gsv/data/transpiration_products")
    preprocessed_dir: Path = Path("/scratch/tmp/yluo2/gsv/data/transpiration_preprocessed")
    figures_dir: Path = Path("outputs/figures/transpiration_comparison")
    reports_dir: Path = Path("outputs/transpiration_comparison")
```

## Interface Contracts

### products/base.py -- ProductBase ABC

```python
from abc import ABC, abstractmethod
import xarray as xr

class ProductBase(ABC):
    """Base class for transpiration product downloaders."""

    def __init__(self, config: ProductConfig, paths: Paths, period: TimePeriod,
                 domain: SpatialDomain):
        self.config = config
        self.paths = paths
        self.period = period
        self.domain = domain

    @abstractmethod
    def download(self) -> Path:
        """Download raw data to products_dir. Returns path to downloaded files.
        Side effect: writes files to disk. Idempotent (skips if files exist)."""
        ...

    @abstractmethod
    def load(self) -> xr.Dataset:
        """Load downloaded data as xarray Dataset with standard coordinates.
        Returns Dataset with 'transpiration' variable, dims (time, lat, lon)."""
        ...

    def output_path(self) -> Path:
        return self.paths.products_dir / self.config.name.lower().replace(" ", "_")
```

### products/sap_velocity.py -- Special Loader

```python
class SapVelocityProduct(ProductBase):
    """Load our parquet predictions into xarray format."""

    def download(self) -> Path:
        return self.paths.predictions_parquet  # Already on disk

    def load(self) -> xr.Dataset:
        """Read monthly parquets, pivot to gridded xarray Dataset.
        Uses dask for lazy loading. Returns Dataset with 'sap_velocity' var."""
        ...
```

### preprocessing/regrid.py

```python
def regrid_to_common(ds: xr.Dataset, target_resolution: float = 0.1,
                     method: str = "bilinear") -> xr.Dataset:
    """Regrid dataset to common 0.1 deg grid using xESMF.
    Input: any-resolution Dataset with lat/lon coords.
    Output: Dataset on standard 0.1 deg grid.
    Method: 'bilinear' for upsampling (GLDAS), 'conservative' for downsampling (PMLv2)."""
    ...
```

### preprocessing/temporal.py

```python
def aggregate_to_daily(ds: xr.Dataset, source_freq: str) -> xr.Dataset:
    """Aggregate sub-daily or multi-day data to daily means.
    source_freq: 'hourly', '3-hourly', '8-day'
    For hourly: daily mean. For 8-day: linear interpolation to daily.
    Handles unit conversion (e.g., ERA5 m/day forecast accumulation -> mm/day)."""
    ...
```

### preprocessing/normalize.py

```python
def zscore_normalize(ds: xr.Dataset, var: str, dim: str = "time") -> xr.Dataset:
    """Compute Z-score normalization per pixel along time dimension.
    z = (x - x_mean) / x_std
    Adds '{var}_zscore' variable to dataset.
    NaN handling: pixels with <30 valid days are masked."""
    ...

def compute_climatology(ds: xr.Dataset, var: str) -> xr.Dataset:
    """Compute monthly climatology (12-month seasonal cycle).
    Returns Dataset with 'month' dim (1-12) and '{var}_clim' variable."""
    ...

def compute_anomaly(ds: xr.Dataset, var: str) -> xr.Dataset:
    """Compute anomaly = value - climatology.
    Adds '{var}_anomaly' variable."""
    ...
```

### comparison/spatial.py

```python
def spatial_correlation(ds_ref: xr.Dataset, ds_comp: xr.Dataset,
                        var_ref: str, var_comp: str) -> xr.Dataset:
    """Pixel-wise temporal correlation between two products.
    Uses xskillscore.pearson_r along time dimension.
    Input: two Datasets on same grid, same time period.
    Output: Dataset with pearson_r, p_value per pixel."""
    ...

def pft_stratified_metrics(ds_ref: xr.Dataset, ds_comp: xr.Dataset,
                           pft_map: xr.DataArray) -> pd.DataFrame:
    """Compute comparison metrics stratified by PFT.
    Returns DataFrame: rows=PFTs, cols=metrics (pearson_r, rmse, mae, bias, n_pixels)."""
    ...

def zonal_mean_profile(ds: xr.Dataset, var: str) -> pd.DataFrame:
    """Compute zonal (latitude-band) means.
    Returns DataFrame with lat bands and mean values per product."""
    ...
```

### comparison/temporal_comp.py

```python
def seasonal_cycle_comparison(datasets: dict[str, xr.Dataset],
                              pft_map: xr.DataArray) -> dict:
    """Compare monthly climatologies across products, stratified by PFT.
    Input: dict of product_name -> Dataset (each with '{var}_clim').
    Output: dict with 'correlations', 'peak_months', 'amplitude_ratios' per PFT."""
    ...

def interannual_correlation(ds_ref: xr.Dataset, ds_comp: xr.Dataset,
                            var_ref: str, var_comp: str) -> xr.Dataset:
    """Pixel-wise correlation of annual anomalies.
    Aggregates to annual means, then correlates."""
    ...

def trend_agreement(datasets: dict[str, xr.Dataset]) -> xr.Dataset:
    """Linear trend per pixel per product. Returns sign agreement map."""
    ...
```

### comparison/collocation.py

```python
def triple_collocation(ds_a: xr.Dataset, ds_b: xr.Dataset, ds_c: xr.Dataset,
                       var: str) -> dict[str, xr.DataArray]:
    """Triple collocation error estimation.
    Input: three independent products on same grid/time.
    Output: dict with error variance estimates per product per pixel."""
    ...
```

### comparison/regional.py

```python
def regional_analysis(datasets: dict[str, xr.Dataset], region: Region,
                      pft_map: xr.DataArray) -> dict:
    """Deep-dive analysis for a single region.
    Returns: normalized time series, seasonal overlay, event response metrics."""
    ...
```

### visualization/maps.py

```python
def plot_global_map(data: xr.DataArray, title: str, output_path: Path,
                    cmap: str = "RdYlBu_r", vmin: float = None,
                    vmax: float = None) -> None:
    """Plot a single global map with cartopy. Robinson projection."""
    ...

def plot_correlation_map(corr: xr.DataArray, title: str, output_path: Path) -> None:
    """Specialized map for correlation values (-1 to 1)."""
    ...

def plot_multi_product_maps(datasets: dict[str, xr.Dataset], var: str,
                            output_dir: Path) -> None:
    """Panel plot: one map per product side by side."""
    ...
```

### visualization/taylor.py

```python
def taylor_diagram(reference: xr.DataArray, products: dict[str, xr.DataArray],
                   title: str, output_path: Path) -> None:
    """Taylor diagram using SkillMetrics library.
    One point per product showing correlation, std ratio, centered RMSE."""
    ...

def taylor_diagram_by_pft(reference: xr.DataArray, products: dict[str, xr.DataArray],
                          pft_map: xr.DataArray, output_path: Path) -> None:
    """One Taylor diagram per PFT (subplot grid)."""
    ...
```

### cli.py -- Entry Point

```python
"""CLI entry point for transpiration comparison pipeline.

Usage:
    python -m src.transpiration_comparison.cli download --products gleam era5land pmlv2 gldas
    python -m src.transpiration_comparison.cli preprocess --products all
    python -m src.transpiration_comparison.cli compare --phase spatial
    python -m src.transpiration_comparison.cli compare --phase temporal
    python -m src.transpiration_comparison.cli compare --phase regional --region amazon
    python -m src.transpiration_comparison.cli visualize --all
    python -m src.transpiration_comparison.cli report
    python -m src.transpiration_comparison.cli run --all   # Full pipeline
"""
```

## Error Handling Strategy

1. **Download failures:** Retry 3x with exponential backoff. Log failures. Skip product if
   download fails after retries (pipeline continues with available products).
2. **Missing data / NaN:** Mask pixels with <30 valid days before Z-score computation.
   Report percentage of valid pixels per product.
3. **Regridding edge cases:** Log any pixels that become NaN after regridding.
   Conservative regridding preserves flux totals.
4. **Memory management:** All heavy operations use dask chunking.
   Chunk along time (monthly chunks = ~30 days) for spatial operations.
   Chunk along space for temporal operations.
5. **Intermediate persistence:** Each phase writes results to disk (NetCDF/Parquet).
   Re-running a phase reads inputs from prior phase output, not re-computes.

## Data Flow Summary

```
Phase 2: Download
  GLEAM SFTP -> raw NetCDF -> products_dir/gleam/
  ERA5-Land CDS -> raw GRIB/NetCDF -> products_dir/era5land/
  PMLv2 GEE -> exported GeoTIFF -> products_dir/pmlv2/
  GLDAS OPeNDAP -> raw NetCDF -> products_dir/gldas/
  Our parquets -> (in place, no download needed)

Phase 2.5: Preprocess
  Each product: load -> regrid to 0.1 deg -> aggregate to daily -> save NetCDF
  All products: Z-score normalize -> save with _zscore variable
  Output: preprocessed_dir/{product}_preprocessed.nc

Phase 3: Spatial Comparison
  Load all preprocessed products
  Compute: climatological maps, pixel-wise correlations, PFT metrics, zonal means
  Output: comparison results as NetCDF + DataFrames

Phase 4: Temporal Comparison
  Load preprocessed products + climatologies
  Compute: seasonal correlations, interannual correlations, trends
  Output: temporal metric maps + DataFrames

Phase 5: Regional Deep Dives
  Subset preprocessed products to each region
  Compute: time series, seasonal overlays, event detection
  Output: per-region DataFrames + time series

Phase 6: Product Agreement
  Load all comparison results
  Compute: consensus maps, outlier detection, product rankings
  Output: agreement metrics

Phase 7: Visualization (reads comparison outputs, writes figures)
Phase 8: Reporting (reads all outputs, writes markdown)
```

## Parquet-to-xarray Conversion Strategy

Our predictions are in monthly parquet files (17M rows each, lat/lon/time/features).
Converting to xarray:

```python
def parquet_to_xarray(parquet_dir: Path, value_col: str = "sap_velocity_xgb") -> xr.Dataset:
    """Convert monthly parquet predictions to gridded xarray Dataset.

    Strategy:
    1. Read each monthly parquet with dask (lazy)
    2. Extract lat, lon, timestamp, value_col
    3. Round lat/lon to 0.1 deg grid centers
    4. Pivot: (time, lat, lon) -> value
    5. Concat all months along time dimension

    Memory: ~56 GB parquet -> ~5 GB xarray (float32, just the prediction column)
    """
    ...
```

## SLURM Integration

Each phase gets its own SLURM script for independent submission:

```bash
# Example: download_products.sh
#!/bin/bash
#SBATCH --job-name=transp_download
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=yu.luo@uni-muenster.de
#SBATCH --mail-type=END,FAIL

source /scratch/tmp/yluo2/gsv/.venv/bin/activate
cd /scratch/tmp/yluo2/gsv-wt/map-viz

python -m src.transpiration_comparison.cli download --products gleam era5land gldas
# PMLv2 via GEE needs separate handling (GEE export + download)
```

Dependency chain for full pipeline:
```
download (12h) -> preprocess (4h) -> compare (8h) -> visualize (2h) -> report (0.5h)
```

## Adding a New Product

To add a new transpiration product:

1. Create `products/new_product.py` implementing `ProductBase`
2. Add entry to `PRODUCTS` dict in `config.py`
3. Run: `python -m src.transpiration_comparison.cli download --products new_product`
4. Run: `python -m src.transpiration_comparison.cli preprocess --products new_product`
5. Re-run comparison phases (they auto-discover preprocessed products)

No changes needed to comparison, visualization, or reporting modules.
