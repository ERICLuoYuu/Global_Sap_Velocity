# Stage 4: Data Merging Codemap

**Last Updated:** 2026-03-12
**Module:** `notebooks/` (merge scripts)
**Entry Point:** `merge_gap_filled_hourly_orginal.py` (ACTIVE)

---

## Purpose

Combine sapflow observations, environmental variables, and extracted features into a single time-aligned, daily-aggregated dataset ready for model training.

---

## Data Merge Pipeline

```
┌─────────────────────────────────┐
│ SAPFLOW (outlier-removed)       │
│ + ENV (outlier-removed)         │
└────────────┬────────────────────┘
             │
    ┌────────▼────────────┐
    │ HOURLY MERGE        │
    │ (left join on time) │
    └────────┬────────────┘
             │
    ┌────────▼──────────────────┐
    │ ADD EXTRACTED FEATURES:   │
    │ - ERA5 (hourly)          │
    │ - GlobMap LAI (hourly)   │
    │ - PFT (annual)           │
    │ - Canopy height (static) │
    │ - Terrain (static)       │
    │ - WorldClim (static)     │
    │ - SoilGrids (static)     │
    │ - Stand age (annual)     │
    └────────┬──────────────────┘
             │
    ┌────────▼──────────────────┐
    │ HOURLY DATASET           │
    │ (all vars at 1h) │
    └────────┬──────────────────┘
             │
    ┌────────▼──────────────────────┐
    │ DAILY AGGREGATION:           │
    │ - Sap velocity: mean         │
    │ - Temp: min, mean, max       │
    │ - VPD: min, mean, max        │
    │ - Radiation: sum, mean       │
    │ - Precip: sum                │
    │ - LAI, PFT: mean/modal       │
    │ - Static: constant           │
    └────────┬──────────────────────┘
             │
    ┌────────▼──────────────────┐
    │ MERGED DAILY DATASET      │
    │ (model-ready)             │
    └───────────────────────────┘
```

---

## Active Merge Script

**File:** `notebooks/merge_gap_filled_hourly_orginal.py`

**Purpose:** Primary merging script for the full pipeline

**Inputs:**
- Sapflow (hourly, outlier-removed): `src/Analyzers/` output
- Environment (hourly, outlier-removed): `src/Analyzers/` output
- Extracted features (from Stage 2): `data/raw/extracted_data/*/`
- ERA5-Land (hourly): From `extract_climatedata_gee.py`
- GlobMap LAI (hourly): From `extract_globmap_lai.py`

**Key Steps:**

1. **Site iteration:** Loop through all valid sites (SAPFLUXNET + Sapflow-internal)
2. **Hourly merge:**
   - Load outlier-removed sapflow + environment
   - Left-join on TIMESTAMP
   - Match features by site and date
3. **Feature alignment:**
   - ERA5 hourly (reindex if needed)
   - LAI hourly (forward-fill for missing timestamps)
   - PFT annual (broadcast to hourly)
   - Static features (constant across time)
4. **Daily aggregation:**
   - Resample hourly to daily (UTC midnight)
   - Sap velocity: mean daily
   - Temperature: min, mean, max
   - VPD: min, mean, max (diurnal split)
   - Radiation: daily integral (sum W/m² over day)
   - Precipitation: daily total
   - LAI, PFT: daily mean/modal
5. **QC flagging:**
   - Flag days with <75% valid hourly records
   - Flag days with NaN in critical variables

**Output:**
- `outputs/processed_data/sapwood/merged/merged_data.csv` (daily, all sites)
- `outputs/processed_data/sapwood/hourly/merged_data_hourly.csv` (hourly, optional)
- `outputs/processed_data/sapwood/daily/` (per-site daily files, optional)

**Execution:**
```bash
python notebooks/merge_gap_filled_hourly_orginal.py
```

---

## Output Format

**Columns in merged_data.csv:**

| Column | Type | Aggregation | Source |
|--------|------|-------------|--------|
| site_code | string | — | Sapflow metadata |
| TIMESTAMP | datetime | — | Hourly index, midnight |
| sap_velocity | float | Mean (hourly) | Sapflow observations |
| ta | float | Min, mean, max | ERA5 + env data |
| vpd | float | Min, mean, max | ERA5 + env data |
| sw_in | float | Sum (daily integral) | ERA5 + env data |
| ws | float | Mean | ERA5 + env data |
| precip | float | Sum (daily total) | ERA5 + env data |
| ppfd_in | float | Sum (if available) | ERA5 or env data |
| lai | float | Mean (hourly → daily) | GlobMap LAI |
| pft | string/int | Modal (annual) | MODIS MCD12Q1 |
| canopy_height | float | — (static) | Extract canopy height |
| elevation | float | — (static) | ASTER GDEM |
| soil_temperature | float | Mean (or from ERA5) | SoilGrids or ERA5 |
| soil_water | float | Mean (or from ERA5) | SoilGrids or ERA5 |
| worldclim_bio1 | float | — (static) | WorldClim |
| ... | ... | — (static) | WorldClim (19 variables) |
| biome | string | — (static) | Site metadata |

**Shape:**
- ~185 sites × ~8000 days average (site-dependent) = ~1.5M rows
- ~60 columns (merged features)

---

## Alternative Merge Scripts (NOT Active)

### `merge_gap_filled_hourly.py`

**Status:** INACTIVE — GSI-based growing-season variant

**Purpose:** Alternative merging logic using growing-season indicators

**Why not used:** Main pipeline uses original merging; this variant created for comparison studies

**Warning:** Do NOT use unless explicitly requested for GSI-based analysis

---

## Gap-Filling Scripts (Supporting)

**Files:**
- `notebooks/sap_gap_filling.py`
- `notebooks/env_gap_filling.py`

**Purpose:** Optional gap-filling for missing values

**Inputs:** Data in `filtered/` directories (pre-QC)

**Note:** These are NOT part of the main pipeline; used only for exploratory analysis or specific studies

---

## Related Analysis Notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/VPD_mismatch_analysis.py` | Compare site-measured VPD vs ERA5-derived VPD |
| `notebooks/explore_*.py` | Exploratory data analysis (optional) |

---

## Data Quality in Merged Dataset

### Completeness Filtering

Days flagged with <75% valid hourly records are marked but NOT excluded. Filtering applied at:
- **Training:** Feature selection based on data availability
- **Validation:** Optional masking of sparse days

### Missing Value Handling

| Variable | Handling |
|----------|----------|
| Sap velocity | NaN preserved; rows with all-NaN sap dropped |
| ERA5 | Reindex to hourly; gaps left as NaN (rare) |
| LAI | Forward-fill gaps (GlobMap has annual resolution) |
| PFT | Broadcast annual to hourly (constant) |
| Static features | Constant per site; no time variation |

---

## Execution Order (Stage 4)

```bash
# Prerequisites: Stages 1, 2, 3 completed

# Main merge
python notebooks/merge_gap_filled_hourly_orginal.py

# [OPTIONAL] Per-site daily files (for inspection)
python notebooks/split_merged_to_per_site.py

# [OPTIONAL] VPD mismatch check
python notebooks/VPD_mismatch_analysis.py
```

**Time to completion:** 30–60 minutes (depends on site count and feature merge complexity)

---

## Configuration

**Input paths:** `path_config.py` (PathConfig class)
```python
paths = PathConfig(scale='sapwood')
print(paths.sap_outliers_removed_dir)  # Input sapflow
print(paths.env_outliers_removed_dir)  # Input environment
print(paths.merged_data_dir)           # Output
```

**Aggregation rules:** Hardcoded in `merge_gap_filled_hourly_orginal.py`
- Daily start: UTC midnight
- Sap velocity: Arithmetic mean
- Temperature/VPD: Min, mean, max
- Radiation: Daily integral (W m⁻²·s → J m⁻² → kWh m⁻²)
- Precip: Daily total

---

## Related Codemaps

- **[etl-stage1.md](etl-stage1.md)** — Produces Sapflow-internal data
- **[extractors-stage2.md](extractors-stage2.md)** — Produces extracted features
- **[analyzers-stage3.md](analyzers-stage3.md)** — Produces outlier-removed data
- **[training-stage5.md](training-stage5.md)** — Consumes merged data
- **[INDEX.md](INDEX.md)** — Overview

---

## Notes

1. **Active script is clear:** Use `merge_gap_filled_hourly_orginal.py` only; ignore other variants
2. **No automatic gap-filling:** NaN values preserved; filling done only if explicitly requested
3. **Daily aggregation is irreversible:** Hourly variability lost; keep hourly merged for analysis if needed
4. **Static features broadcast:** PFT and canopy height constant across time per site
5. **ERA5 as ground truth:** When site-measured env vars unavailable, ERA5 is fallback
