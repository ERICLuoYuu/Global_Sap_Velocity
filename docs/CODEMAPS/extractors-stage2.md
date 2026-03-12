# Stage 2: Feature Extraction Codemap

**Last Updated:** 2026-03-12
**Module:** `src/Extractors/`
**Dependency:** Requires GEE authentication (`earthengine authenticate`)

---

## Purpose

Gather spatially and temporally co-located environmental and structural covariates for each sapflow site from satellite, reanalysis, soil, and canopy-height data sources.

---

## Feature Extraction Pipeline

```
┌─────────────────────────────┐
│ Site Info (lat, lon, dates) │
└────────┬────────────────────┘
         │
    ┌────┴────────────────────────┬────────────────────────┬──────────────────┬────────────────┐
    │                             │                        │                  │                │
┌───▼────┐               ┌────────▼─────┐        ┌────────▼────┐    ┌──────▼─────┐   ┌────▼────┐
│ ERA5   │               │ GlobMap LAI  │        │ MODIS PFT   │    │ Canopy Ht  │   │ Soil    │
│ (GEE)  │               │ (local TIF)  │        │ (GEE)       │    │ (GEE+DEM)  │   │ Grid    │
└───┬────┘               └────────┬─────┘        └────────┬────┘    └──────┬─────┘   │ (REST)  │
    │                            │                        │                 │        │         │
    │                            │                        │                 │        └────┬────┘
    │  WorldClim + Köppen ──────┬┤                        │                 │             │
    │  (spatial_features_extr.)  │                        │                 │             │
    │                            │                        │                 │             │
    └────────┬───────────────────┴────────────────────────┴─────────────────┴─────────────┘
             │
        ┌────▼─────────────────────┐
        │ MERGED FEATURE DATASET    │
        │ (hourly + daily, per site)│
        └──────────────────────────┘
```

---

## Extraction Scripts (8)

### 1. `extract_siteinfo.py`

**Purpose:** Extract site metadata from SAPFLUXNET plant files

**Outputs:**
- `site_info.csv` (site_code, latitude, longitude, elevation, biome, PFT, start_date, end_date)

**Execution:**
```bash
python src/Extractors/extract_siteinfo.py
```

---

### 2. `extract_climatedata_gee.py`

**Purpose:** Hourly ERA5-Land for each site via Google Earth Engine

**Input:** GEE ImageCollection("ECMWF/ERA5_LAND/HOURLY")

**Variables extracted:**
- Temperature (2m), dewpoint, wind (u, v), precip, solar radiation, soil water layer 1, soil temperature

**Output:** `era5_extracted_data.csv` (hourly, per site)

**Execution:**
```bash
python src/Extractors/extract_climatedata_gee.py
```

---

### 3. `extract_globmap_lai.py`

**Purpose:** Leaf area index from GlobMap LAI rasters (local pre-downloaded)

**Inputs:** `data/raw/grided/globmap_lai/GlobMapLAIV3.A*.Global.LAI.tif`

**Processing:**
- Nearest-neighbor resampling to site coordinates
- Hourly alignment (forward-fill if needed)

**Output:** `extracted_globmap_lai_hourly.csv`

**Note:** This is the primary LAI source for the merge pipeline. `extract_lai.py` (MODIS+AVHRR via GEE) is standalone and NOT used.

**Execution:**
```bash
python src/Extractors/extract_globmap_lai.py
```

---

### 4. `extract_pft.py`

**Purpose:** Plant functional type from MODIS MCD12Q1 LC_Type1 (IGBP classification)

**Input:** GEE ImageCollection("MODIS/006/MCD12Q1")

**Processing:**
- Annual PFT per site (constant within year)
- Maps to 17 IGBP classes (simplified to 8 major biomes in models)

**Output:** `pft_extracted_data.csv`

**Execution:**
```bash
python src/Extractors/extract_pft.py
```

---

### 5. `extract_canopyheight.py`

**Purpose:** Canopy height from Meta global canopy height + ASTER GDEM

**Input:**
- GEE: Meta Canopy Height (30-m resolution)
- GEE: ASTER GDEM v3 (for elevation, terrain)

**Processing:**
- Nearest-neighbor at site coordinates
- Static (constant across time)

**Output:** `canopy_height.csv`, `terrain_attributes.csv`

**CRITICAL:** Must run BEFORE `spatial_features_extractor.py`

**Execution:**
```bash
python src/Extractors/extract_canopyheight.py
```

---

### 6. `spatial_features_extractor.py`

**Purpose:** WorldClim bioclimatic variables + Köppen-Geiger climate zones

**Inputs:**
- `terrain_attributes.csv` (from `extract_canopyheight.py`)
- Auto-downloads WorldClim and Köppen from web

**Processing:**
- Bilinear interpolation at site coordinates
- Static bioclimatic variables (20 variables)

**Output:** `worldclim_koppen.csv`

**DEPENDENCY:** Must run AFTER `extract_canopyheight.py`

**Execution:**
```bash
python src/Extractors/spatial_features_extractor.py
```

---

### 7. `extract_soilgrid.py`

**Purpose:** Soil properties from SoilGrids via REST API

**Input:** https://rest.isric.org/soilgrids/v2.0/properties/query

**Variables extracted:**
- Bulk density, clay, silt, sand, organic carbon, pH, cation exchange capacity
- At depths: 0–5cm, 5–15cm, 15–30cm

**Processing:**
- REST API queries (may be slow for large site counts)
- Static (constant across time)

**Output:** `soilgrids_extracted.csv`

**Execution:**
```bash
python src/Extractors/extract_soilgrid.py
```

---

### 8. `extract_stand_age.py`

**Purpose:** Forest stand age from BGI/MPI-BGC NetCDF

**Inputs:** `data/raw/grided/stand_age/*.nc` (pre-downloaded local files)

**Processing:**
- Nearest-neighbor at site coordinates
- Annual time series (forest age varies year to year)

**Output:** `stand_age_extracted.csv`

**Execution:**
```bash
python src/Extractors/extract_stand_age.py
```

---

## Execution Order (Stage 2)

```bash
# 1. Site metadata (prerequisite for all)
python src/Extractors/extract_siteinfo.py

# 2. Parallel (in any order):
python src/Extractors/extract_climatedata_gee.py    # ERA5-Land (hourly)
python src/Extractors/extract_pft.py                 # MODIS PFT
python src/Extractors/extract_soilgrid.py            # SoilGrids (REST API)

# 3. Extract canopy height FIRST (prerequisite)
python src/Extractors/extract_canopyheight.py

# 4. Then spatial features (depends on canopy height)
python src/Extractors/spatial_features_extractor.py

# 5. Forest stand age
python src/Extractors/extract_stand_age.py

# 6. GlobMap LAI (for merge pipeline)
python src/Extractors/extract_globmap_lai.py

# 7. [OPTIONAL] MODIS+AVHRR LAI (standalone, not merged)
python src/Extractors/extract_lai.py
```

**Time to completion:** 30 min – 2 hours (depends on GEE and API responsiveness)

---

## Output Directory

```
data/raw/extracted_data/
├── era5land_site_data/sapwood/
│   └── era5_extracted_data.csv
├── globmap_lai_site_data/sapwood/
│   └── extracted_globmap_lai_hourly.csv
├── pft_site_data/sapwood/
│   └── pft_extracted_data.csv
├── canopy_height_site_data/sapwood/
│   ├── canopy_height.csv
│   └── terrain_attributes.csv
├── worldclim_koppen_site_data/sapwood/
│   └── worldclim_koppen.csv
├── soilgrids_site_data/sapwood/
│   └── soilgrids_extracted.csv
└── stand_age_site_data/sapwood/
    └── stand_age_extracted.csv
```

---

## Pre-Download Requirements

| Dataset | Local Path | Source | Notes |
|---------|-----------|--------|-------|
| GlobMap LAI v3 | `data/raw/grided/globmap_lai/GlobMapLAIV3.A*.Global.LAI.tif` | http://globalchange.bnu.edu.cn | ~200 annual GeoTIFFs; download manually |
| BGI Stand Age | `data/raw/grided/stand_age/*.nc` | BGI/MPI-BGC data portal | ~20 global NetCDF files; auto-download script available |

WorldClim and Köppen are auto-downloaded by `spatial_features_extractor.py` at runtime.

---

## GEE Dependency

All GEE-based extractors require:
```bash
earthengine authenticate
earthengine set_project ee-yuluo-2
```

Verify:
```python
import ee
ee.Initialize()
print(ee.Image("ECMWF/ERA5_LAND/HOURLY").first())
```

---

## Related Codemaps

- **[etl-stage1.md](etl-stage1.md)** — Produces Sapflow-internal data for feature merging
- **[analyzers-stage3.md](analyzers-stage3.md)** — Quality control on extracted features
- **[merging-stage4.md](merging-stage4.md)** — Consumes all extracted features
- **[INDEX.md](INDEX.md)** — Overview

---

## Notes

1. **LAI handling:** GlobMap LAI (primary, in merge) vs MODIS+AVHRR LAI (standalone)
2. **Canopy height dependency:** MUST run before `spatial_features_extractor.py`
3. **REST API rate limits:** SoilGrids may throttle requests; retry logic built in
4. **GEE quotas:** Large site counts may exceed daily GEE export quota; batch or request increased quota
5. **Network I/O:** `extract_globmap_lai.py` and `extract_stand_age.py` read local files (fastest); GEE scripts network-dependent
