# Stage 6: Global Prediction & Post-Analysis Codemap

**Last Updated:** 2026-03-12
**Module:** `src/make_prediction/`
**Entry Points:** `process_era5land_gee_opt_fix.py`, `predict_sap_velocity_sequential.py`, `prediction_visualization.py`, `sap_velocity_by_climatezone_forest.py`

---

## Purpose

Apply best-trained model globally at 0.1° × 0.1° daily resolution, generate global sap velocity maps, and produce regional/biome-level post-analysis summaries.

---

## Global Prediction Pipeline

```
┌────────────────────────────────┐
│ TRAINED MODEL + SCALER         │
│ (from Stage 5)                 │
└────────┬───────────────────────┘
         │
    ┌────▼──────────────────────┐
    │ ERA5-LAND GLOBAL GRIDS    │
    │ (via GEE)                 │
    │ - Hourly to daily agg      │
    │ - 0.1° × 0.1° resolution   │
    └────┬───────────────────────┘
         │
    ┌────▼──────────────────────┐
    │ STATIC GLOBAL GRIDS       │
    │ - MODIS PFT (annual)      │
    │ - Canopy height           │
    │ - ASTER elevation         │
    │ - WorldClim bioclimatic   │
    │ - SoilGrids properties    │
    └────┬───────────────────────┘
         │
    ┌────▼──────────────────────┐
    │ FEATURE ASSEMBLY          │
    │ (temporal windowing,      │
    │  z-score normalization)   │
    └────┬───────────────────────┘
         │
    ┌────▼────────────────────────┐
    │ BATCH PREDICTION            │
    │ (spatial: 1000 cells/batch) │
    │ (temporal: 32 steps/batch)  │
    │ → Parallel via Dask         │
    └────┬────────────────────────┘
         │
    ┌────▼────────────────────────┐
    │ POST-PROCESSING            │
    │ - expm1() inverse transform│
    │ - Mask non-forest          │
    │ - Regional aggregation     │
    │ - Visualization            │
    └────┬────────────────────────┘
         │
    ┌────▼────────────────────────┐
    │ OUTPUT MAPS & SUMMARIES    │
    │ (NetCDF, GeoTIFF, PNG)     │
    └─────────────────────────────┘
```

---

## Key Scripts

### 1. `process_era5land_gee_opt_fix.py`

**Purpose:** Download and aggregate ERA5-Land from GEE to daily grids

**Inputs:**
- GEE ImageCollection("ECMWF/ERA5_LAND/HOURLY")
- Spatial bounds: 60°S–78°N, 180°W–180°E
- Temporal range: User-specified (typically full date range)

**Processing:**
- Hourly → daily aggregation (mean for vars, sum for precip)
- 0.1° × 0.1° grid (GEE native resolution)
- Variables: ta, vpd, sw_in, ws, precip, soil moisture, soil temperature

**Outputs:**
- Intermediate cache: `D:/Temp/era5land_extracted/` (can be deleted after prediction)
- Format: GeoTIFF or NetCDF per variable per date

**Execution:**
```bash
python src/make_prediction/process_era5land_gee_opt_fix.py \
  --start_date 2018-01-01 \
  --end_date 2018-12-31
```

**Notes:**
- Cache disk I/O; can be slow for years of data
- Intermediate files can be deleted after `predict_sap_velocity_sequential.py` completes
- Requires GEE authentication

---

### 2. `predict_sap_velocity_sequential.py` (PRIMARY)

**Purpose:** Apply trained model to global grids; generate daily sap velocity predictions

**Inputs:**
- Trained XGBoost model: `outputs/models/xgb/{run_id}/FINAL_xgb_{run_id}.joblib`
- Feature scaler: `outputs/models/xgb/{run_id}/FINAL_scaler_{run_id}_feature.pkl`
- Daily ERA5 grids (from `process_era5land_gee_opt_fix.py`)
- Static global grids (MODIS PFT, elevation, canopy height, bioclimatic, soil)

**Processing:**

1. **Load static grids:**
   - MODIS PFT (17 classes, annual)
   - ASTER GDEM elevation
   - Meta canopy height
   - WorldClim (20 bioclimatic variables)
   - SoilGrids (bulk density, clay, silt, sand, organic C, pH, CEC)

2. **Create daily features:**
   - ERA5 hourly → daily aggregation (z-score per variable within grid)
   - Temporal lags: `INPUT_WIDTH=2` days
   - Static features broadcast to all dates

3. **Batch prediction:**
   - Spatial batches: 1000 grid cells per batch
   - Temporal batches: 32 days per prediction
   - Dask parallelization (configurable workers, memory limit)

4. **Inverse transformation:**
   - Predictions in log-space: `expm1()` to revert to original scale
   - Output: cm³ cm⁻² h⁻¹

5. **Masking:**
   - Forest detection: MODIS PFT filtered (keep tree types)
   - Water bodies: NA masking
   - Output only grid cells with trees

**Outputs:**
```
outputs/predictions/
├── sap_velocity_global_daily_{run_id}.nc  # NetCDF: lat, lon, time dimensions
├── sap_velocity_global_daily_{run_id}.tif # GeoTIFF (per date if split)
└── prediction_metadata_{run_id}.json      # Spatial extent, grid info, timestamps
```

**Memory & Performance:**
- Dask memory limit: 2GB per chunk (configurable in `config.py`)
- Multi-worker parallel prediction
- Total runtime: 4–12 hours (depends on time period, compute resources)

**Execution:**
```bash
python src/make_prediction/predict_sap_velocity_sequential.py \
  --run_id default_daily_nocoors_swcnor \
  --date_start 2018-01-01 \
  --date_end 2018-12-31 \
  --n_workers 4
```

---

### 3. `prediction_visualization.py`

**Purpose:** Generate maps and time-series plots from global predictions

**Inputs:**
- NetCDF/GeoTIFF global predictions (from `predict_sap_velocity_sequential.py`)
- Mask layers (forest extent, climate zones)

**Outputs:**
```
outputs/figures/predictions/{run_id}/
├── sap_velocity_map_{date}.png            # Single-date spatial map
├── annual_mean_sap_velocity_2018.png      # Annual mean (spatially)
├── sap_velocity_timeseries_global.png     # Global mean time series
├── sap_velocity_by_latitude.png           # Zonal mean (lat-averaged)
└── seasonal_decomposition_{run_id}.png    # Seasonal patterns
```

**Color scales:**
- Low: 0 cm³ cm⁻² h⁻¹ (blue)
- High: Max observed (red)
- Typical range: 0–0.5 cm³ cm⁻² h⁻¹

**Execution:**
```bash
python src/make_prediction/prediction_visualization.py \
  --run_id default_daily_nocoors_swcnor
```

---

### 4. `sap_velocity_by_climatezone_forest.py`

**Purpose:** Regional and biome-level summaries; post-analysis aggregations

**Inputs:**
- Global prediction maps
- Climate zone mask (Köppen-Geiger)
- Forest type mask (MODIS PFT)

**Aggregations:**

| Dimension | Output | Unit |
|-----------|--------|------|
| Global | Mean ± std | cm³ cm⁻² h⁻¹ |
| By Köppen climate | Regional mean | cm³ cm⁻² h⁻¹ |
| By PFT | Biome-level mean | cm³ cm⁻² h⁻¹ |
| By continent | Regional sum (water use) | 10⁹ m³ h⁻¹ |
| Seasonal | Monthly climatology | cm³ cm⁻² h⁻¹ |
| Diurnal | (if hourly output) | cm³ cm⁻² h⁻¹ |

**Outputs:**
```
outputs/predictions/{run_id}/
├── regional_summary_{run_id}.csv          # Climate zone × date table
├── pft_summary_{run_id}.csv               # PFT × date table
├── seasonal_climatology_{run_id}.csv      # Monthly means by region
└── figures/
    ├── regional_heatmap_{run_id}.png      # Climate zone × month
    ├── pft_comparison_{run_id}.png        # Box plot by forest type
    └── water_use_continental_{run_id}.png # Global water use totals
```

**Execution:**
```bash
python src/make_prediction/sap_velocity_by_climatezone_forest.py \
  --run_id default_daily_nocoors_swcnor \
  --input_file outputs/predictions/sap_velocity_global_daily_{run_id}.nc
```

---

## Configuration (src/make_prediction/config.py)

**Key parameters:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `lat_min`, `lat_max` | -60.0, 78.0 | Spatial extent (degrees) |
| `lon_min`, `lon_max` | -180.0, 180.0 | Spatial extent (degrees) |
| `resolution` | 0.1 | Grid size (degrees) |
| `INPUT_WIDTH` | 2 | Days of input history |
| `LABEL_WIDTH` | 1 | Prediction horizon (days) |
| `dask_chunk_memory` | "2GB" | Memory per Dask partition |
| `batch_size_spatial` | 1000 | Cells per prediction batch |
| `batch_size_temporal` | 32 | Days per batch |

---

## Output Format

### NetCDF (Primary)

**File:** `sap_velocity_global_daily_{run_id}.nc`

**Structure:**
```
Dimensions:
  lat: 1800
  lon: 3600
  time: N (number of days)

Variables:
  sap_velocity (time, lat, lon):
    dtype: float32
    units: cm³ cm⁻² h⁻¹
    long_name: Sap velocity (tree water use)
    _FillValue: NaN

Coordinates:
  lat: -60 to 78 (0.1° spacing)
  lon: -180 to 180 (0.1° spacing)
  time: ISO 8601 dates

Attributes:
  model_id: {run_id}
  creation_date: YYYY-MM-DD
  crs: EPSG:4326 (WGS84)
```

### GeoTIFF (Alternative)

- One file per date or concatenated time series
- Geospatial metadata embedded
- Compatible with GIS software (ArcGIS, QGIS)

---

## Parallelization & Performance

### Dask Configuration

```python
from dask.distributed import Client

client = Client(
    n_workers=4,
    threads_per_worker=2,
    memory_limit="2GB"  # per worker
)
```

### Expected Runtime

| Duration | Grid cells | Typical time |
|----------|-----------|--------------|
| 1 month | 6.48M | 2–3 hours |
| 1 year | 6.48M | 12–24 hours |
| Full time period (2000–2020) | 6.48M | 5–10 days |

---

## Error Handling & Restart

**Checkpointing:**
- Predictions saved per batch
- Can resume from last complete batch if interrupted
- Check log files: `outputs/logs/prediction_{run_id}.log`

**Validation:**
- Check for NaN pixels outside ocean/desert (should be minimal)
- Verify spatial/temporal continuity
- Compare global mean with training data range (sanity check)

---

## Execution Order (Stage 6)

```bash
# 1. Download and aggregate ERA5-Land to daily grids
python src/make_prediction/process_era5land_gee_opt_fix.py \
  --start_date 2018-01-01 --end_date 2018-12-31

# 2. Global prediction
python src/make_prediction/predict_sap_velocity_sequential.py \
  --run_id default_daily_nocoors_swcnor \
  --date_start 2018-01-01 --date_end 2018-12-31

# 3. Visualization
python src/make_prediction/prediction_visualization.py \
  --run_id default_daily_nocoors_swcnor

# 4. Regional analysis
python src/make_prediction/sap_velocity_by_climatezone_forest.py \
  --run_id default_daily_nocoors_swcnor

# 5. [OPTIONAL] Clean up intermediate ERA5 cache
rm -rf D:/Temp/era5land_extracted/
```

**Total time:** 20–36 hours (for 1 year of global predictions)

---

## Related Codemaps

- **[training-stage5.md](training-stage5.md)** — Produces trained models
- **[configuration.md](configuration.md)** — ModelConfig for prediction settings
- **[INDEX.md](INDEX.md)** — Overview

---

## Important Notes

1. **Forest masking:** Predictions only for pixels with tree cover (MODIS PFT)
2. **Spatial resolution:** 0.1° × 0.1° (~11 km at equator); aggregation possible but not recommended
3. **Temporal alignment:** Daily midnight UTC; ensure dates match input data
4. **ERA5 dependency:** Large intermediate cache; ensure sufficient disk space
5. **Model reproducibility:** Use same `run_id` to retrieve model artifacts
6. **Output versioning:** Archive `_v1`, `_v2`, etc. if rerunning with different params
7. **Computational requirements:** GPU recommended for large spatial/temporal domains; CPU viable for single year
