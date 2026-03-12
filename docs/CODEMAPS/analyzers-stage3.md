# Stage 3: Quality Control & Analysis Codemap

**Last Updated:** 2026-03-12
**Module:** `src/Analyzers/`
**Entry Point:** `SapFlowAnalyzer`, `EnvironmentalAnalyzer`, `RelationshipExplorer`

---

## Purpose

Quality control, outlier detection, and exploratory data analysis for sapflow and environmental datasets. Prepares validated data for downstream merging and modeling.

---

## Architecture Overview

```
┌──────────────────────────────────────┐
│   RAW SAPFLOW DATA                   │
│  (hourly, per site)                  │
└─────────────┬────────────────────────┘
              │
    ┌─────────▼─────────┐
    │ FLAG FILTERING    │
    │ (quality flags)   │
    └─────────┬─────────┘
              │
    ┌─────────▼─────────────────┐
    │ MANUAL REMOVAL            │
    │ (removal_log.csv applied) │
    └─────────┬─────────────────┘
              │
    ┌─────────▼──────────────────────┐
    │ AUTOMATED QC SEQUENCE:         │
    │ 1. Reverse-measurement detect  │
    │ 2. Rolling z-score removal     │
    │    (day: 3σ, night: 5σ, 30-d) │
    │ 3. Variability filtering       │
    │    (2-day window)              │
    │ 4. Incomplete-day removal      │
    └─────────┬──────────────────────┘
              │
    ┌─────────▼──────────┐
    │ OUTLIER REMOVED    │
    │ (flagged records)  │
    └─────────┬──────────┘
              │
    ┌─────────▼──────────────────────┐
    │ RELATIONSHIP EXPLORATION       │
    │ (scatter + LOWESS analysis)    │
    └────────────────────────────────┘
```

---

## Key Modules

### 1. `sap_analyzer.py` — `SapFlowAnalyzer`

**Purpose:** Apply 6-step QC pipeline to sapflow measurements

**Key Methods:**

| Method | Description |
|--------|-------------|
| `run_analysis_in_batches(batch_size, switch)` | Execute full QC pipeline on all sites (batch processing) |
| `apply_flags()` | Filter by quality flags (columns: `TSPD_F_QC`, etc.) |
| `apply_manual_removal()` | Apply skip/remove operations from `removal_log.csv` |
| `detect_reverse_measurement()` | Flag suspect reversed heat-pulse probes |
| `remove_outliers_rolling_zscore()` | Rolling z-score on 30-day windows; day/night split (3σ day, 5σ night) |
| `filter_variability()` | Remove timestamps with excessive 2-day window variance |
| `remove_incomplete_days()` | Flag days with <75% valid hourly records |

**Inputs:**
- `{scale}/raw_csv_dir/{site}_sapf_data.csv` (hourly sapflow)
- `removal_log.csv` (site skip list, column removals, period exclusions)

**Outputs:**
- `{scale}/sap_outliers_removed_dir/{site}_sapf_data_outliers_removed.csv`
- Log files in `outputs/logs/qc/`

**Batch Processing:**
```python
from src.Analyzers.sap_analyzer import SapFlowAnalyzer
analyzer = SapFlowAnalyzer(scale='sapwood')
analyzer.run_analysis_in_batches(batch_size=10, switch='both')
```

---

### 2. `env_analyzer.py` — `EnvironmentalAnalyzer`

**Purpose:** Validate and standardize environmental variable measurements

**Key Methods:**

| Method | Description |
|--------|-------------|
| `run_analysis_in_batches(batch_size)` | Execute QC on all environment files |
| `apply_flags()` | Filter by env QC flags |
| `remove_outliers_rolling_zscore()` | Diurnal-split z-score (radiation vars handled separately) |
| `standardise_env_data()` | Min-max scaling (0–1) per site |
| `resample_to_daily()` | Daily aggregation: min/max for temp, mean for radiation/vpd |

**Inputs:**
- `{scale}/raw_csv_dir/{site}_env_data.csv` (hourly environmental)

**Outputs:**
- `{scale}/env_outliers_removed_dir/{site}_env_data_outliers_removed.csv`

**Batch Processing:**
```python
from src.Analyzers.env_analyzer import EnvironmentalAnalyzer
ea = EnvironmentalAnalyzer()
ea.run_analysis_in_batches(batch_size=10)
```

**Variables Processed:**
- Core: `ta`, `vpd`, `sw_in`, `ws`, `precip`
- Optional: `ppfd_in`, `rh`, `ts`, `swc` (if present)

---

### 3. `manual_removal_processor.py` — `RemovalLogProcessor`

**Purpose:** Parse and apply manual removal instructions from `removal_log.csv`

**Key Methods:**

| Method | Description |
|--------|-------------|
| `process_removal_log(removal_csv_path)` | Parse removal log; return operations dict |
| `apply_removals(site_code, df, removal_ops)` | Skip site, remove columns, or date ranges |

**removal_log.csv Format:**
```
site_code,operation,column,start_date,end_date
ES-Abr,skip_site,,,
IT-CP2_sapwood,remove_column,TSPD_3_QC,,
AT_Mmg,remove_period,TSPD_1,,2015-01-01,2015-06-30
```

**Operations:**
- `skip_site` — Exclude entire site from analysis
- `remove_column` — Drop specific measurement column
- `remove_period` — Remove data within date range

---

### 4. `explore_relationship_observations.py` — `RelationshipExplorer` (NEW)

**Purpose:** Generate scatter plots with LOWESS fitted lines to explore correlations between tree properties and environmental variables

**Key Attributes:**

| Attribute | Description |
|-----------|-------------|
| `scale` | Data scale: `sapwood`, `plant`, or `site` |
| `output_dir` | Figures output directory (auto-created) |
| `max_scatter_points` | Subsample for scatter performance (default: 50,000) |
| `use_raw` | Use raw data or outlier-removed (default: False) |

**Key Methods:**

| Method | Description |
|--------|-------------|
| `load_tree_metadata()` | Load plant-level DBH, height, sapwood area from `*_plant_md.csv` |
| `load_sapflow_env_data()` | Load site-mean sap flow density + env vars from `*_sapf_data.csv` and `*_env_data.csv` |
| `plot_tree_size_vs_sapwood_area()` | DBH/height vs sapwood area (overall, by biome, by PFT) |
| `plot_sapflow_vs_env()` | Sap flow density vs 6 env variables (vpd, ws, sw_in, ta, ppfd_in, ext_rad) |
| `_scatter_with_lowess()` | Render scatter + LOWESS line with Spearman ρ and n annotation |
| `_make_faceted_plot()` | Grid of subplots (one per group: biome, PFT) |
| `_make_overall_plot()` | Single overall scatter + LOWESS with optional hue coloring |
| `run()` | Execute all plots and save to output directory |

**Inputs:**
- SAPFLUXNET metadata: `*_site_md.csv`, `*_plant_md.csv`
- Sapflow data: `*_sapf_data.csv` or `*_sapf_data_outliers_removed.csv`
- Environmental data: `*_env_data.csv` or `*_env_data_outliers_removed.csv`

**Outputs:**
- PNG figures in `outputs/figures/relationship_exploration/{raw|outlier_removed}/`:
  - `dbh_vs_sapwood_area_overall.png`, `_by_biome.png`, `_by_pft.png`
  - `height_vs_sapwood_area_overall.png`, `_by_biome.png`, `_by_pft.png`
  - `sapflow_vs_env_overall.png` (2×3 grid, all env vars)
  - `sapflow_vs_{vpd|ws|sw_in|ta|ppfd_in|ext_rad}_by_biome.png`
  - `sapflow_vs_{vpd|ws|sw_in|ta|ppfd_in|ext_rad}_by_pft.png`

**Color Schemes:**
- **Biome colors:** Dictionary mapping biome names to hex codes (8 colors)
- **PFT colors:** Dictionary for 12 plant functional types
- **Default:** Blue scatter, red LOWESS trend line

**CLI Usage:**

```bash
# Default: sapwood scale, raw data, default output dir
python src/Analyzers/explore_relationship_observations.py

# Use outlier-removed data
python src/Analyzers/explore_relationship_observations.py --use-raw False

# Sapwood scale with custom output
python src/Analyzers/explore_relationship_observations.py --scale sapwood --output-dir ./my_plots

# Subsample to 20k points for speed
python src/Analyzers/explore_relationship_observations.py --max-points 20000
```

**Statistics Computed:**
- **Spearman rank correlation (ρ)** with significance stars: `*` (p<0.05), `**` (p<0.01), `***` (p<0.001)
- **Sample size (n)** annotated on each subplot
- **LOWESS smooth** computed on subsampled data (max 10,000 points) for efficiency

**Error Handling:**
- Gracefully handles missing files, empty DataFrames, and NaN values
- Warns if site metadata files are unreadable
- Skips faceted groups with fewer than 10 valid observations

---

### 5. `sap_statistics_analyzer.py` — `SapFlowStatisticsAnalyzer` (Supporting)

**Purpose:** Compute descriptive statistics and variability metrics for sapflow data

**Outputs:**
- Summary statistics CSV (n, mean, std, min, max per site)
- Hourly/daily coefficient of variation
- Diurnal amplitude analysis

---

## Data Dependencies

### Input Files

```
{scale}/raw_csv_dir/
├── {site}_sapf_data.csv          ← sap_analyzer, explore_relationship_observations
├── {site}_env_data.csv           ← env_analyzer, explore_relationship_observations
├── {site}_site_md.csv            ← explore_relationship_observations (biome, PFT)
└── {site}_plant_md.csv           ← explore_relationship_observations (DBH, height, sapwood area)

removal_log.csv                    ← mannual_removal_processor (skip sites, remove periods)
```

### Output Directories

```
{scale}/
├── sap_outliers_removed_dir/
│   └── {site}_sapf_data_outliers_removed.csv
├── env_outliers_removed_dir/
│   └── {site}_env_data_outliers_removed.csv

outputs/
├── figures/relationship_exploration/
│   ├── raw/                     ← if use_raw=True
│   │   ├── dbh_vs_sapwood_area_*.png
│   │   ├── height_vs_sapwood_area_*.png
│   │   └── sapflow_vs_*.png
│   └── outlier_removed/         ← default (use_raw=False)
│       └── [same structure]
└── logs/qc/
    ├── sap_analyzer_{date}.log
    └── env_analyzer_{date}.log
```

---

## Execution Order (Stage 3 within Full Pipeline)

```bash
# 1. Apply manual removal rules BEFORE automated QC
python -c "
from src.Analyzers.mannual_removal_processor import RemovalLogProcessor
processor = RemovalLogProcessor()
processor.process_removal_log('removal_log.csv')
"

# 2. QC sapflow data (6-step pipeline)
python -c "
from src.Analyzers.sap_analyzer import SapFlowAnalyzer
analyzer = SapFlowAnalyzer(scale='sapwood')
analyzer.run_analysis_in_batches(batch_size=10, switch='both')
"

# 3. QC environmental data
python -c "
from src.Analyzers.env_analyzer import EnvironmentalAnalyzer
ea = EnvironmentalAnalyzer()
ea.run_analysis_in_batches(batch_size=10)
"

# 4. [OPTIONAL] Exploratory relationship analysis (NEW)
python src/Analyzers/explore_relationship_observations.py --scale sapwood
```

---

## Related Codemaps

- **[extractors-stage2.md](extractors-stage2.md)** — Produces feature data that may be included in relationship exploration
- **[merging-stage4.md](merging-stage4.md)** — Consumes outlier-removed data
- **[INDEX.md](INDEX.md)** — Overview of all stages

---

## Key Implementation Details

### Spearman Correlation

Used instead of Pearson because relationships may be non-linear; LOWESS visualizes the trend.

### LOWESS (Locally Weighted Scatterplot Smoothing)

- **Requires:** `statsmodels` package (`pip install statsmodels`)
- **Fraction:** Adaptive based on sample size: `frac = max(0.3, 30 / n)`
- **Max points for LOWESS:** 10,000 (cap for performance)
- **Full data used for correlation:** All valid pairs, not subsampled

### Diurnal Splitting (env_analyzer only)

Radiation variables (`sw_in`, `ppfd_in`, `ext_rad`) are split into day (6–18 h) and night (18–6 h) before outlier detection to account for natural diurnal patterns.

### Biome/PFT Metadata Caching

RelationshipExplorer caches all site metadata on first call to `_load_site_metadata_cache()` for efficiency.

---

## Notes

1. **Outlier-removed data is optional:** If not present, scripts fall back to raw data
2. **RemovalLogProcessor is standalone:** Can be invoked independently for manual intervention
3. **explore_relationship_observations is exploratory:** Does not feed into main pipeline; useful for understanding data quality and variable relationships
4. **LOWESS requires statsmodels:** Graceful warning if not installed; scatter plots still render
