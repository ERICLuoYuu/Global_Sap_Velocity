# Global Sap Velocity — Architecture Codemaps

**Last Updated:** 2026-03-12
**Project:** Machine Learning pipeline for global tree sap velocity prediction
**Author:** Documentation auto-generated from codebase

This directory contains architectural codemaps describing the structure, data flow, and key responsibilities of modules across the global-sap-velocity project.

---

## Overview

The project is organized into **6 major processing stages**:

1. **Sapflow-internal ETL** — Standardize and extract environmental data from 20 internal European sites
2. **Site-level Feature Extraction** — Gather covariates from satellite, reanalysis, and soil databases
3. **Quality Control & Analysis** — Outlier removal, environmental validation, relationship exploration
4. **Data Merging** — Combine sapflow observations with environmental and feature datasets
5. **Model Training** — Hyperparameter optimization and training (XGBoost, deep learning)
6. **Global Prediction** — Apply best model globally via Google Earth Engine, post-analysis

---

## Codemaps

### Core Execution Pipeline

| Codemap | Coverage | Purpose |
|---------|----------|---------|
| **[etl-stage1.md](etl-stage1.md)** | `Sapflow-internal/` | ETL for internal sapflow sites: reorganization, unit conversion, ICOS extraction |
| **[extractors-stage2.md](extractors-stage2.md)** | `src/Extractors/` | Feature extraction from satellite, reanalysis, and soil databases |
| **[analyzers-stage3.md](analyzers-stage3.md)** | `src/Analyzers/` | Quality control, filtering, and exploratory relationship analysis |
| **[merging-stage4.md](merging-stage4.md)** | `notebooks/` (merge scripts) | Integration of sapflow, environment, and feature data |
| **[training-stage5.md](training-stage5.md)** | `src/hyperparameter_optimization/` | Model training pipelines (XGBoost, DL) and cross-validation |
| **[prediction-stage6.md](prediction-stage6.md)** | `src/make_prediction/` | Global prediction, ERA5 retrieval, and post-analysis visualization |

### Supporting Infrastructure

| Codemap | Coverage | Purpose |
|---------|----------|---------|
| **[cross-validation.md](cross-validation.md)** | `src/cross_validation/` | Temporal and spatial CV strategies to prevent data leakage |
| **[configuration.md](configuration.md)** | `path_config.py`, `src/make_prediction/config.py` | Centralized path and hyperparameter configuration |

---

## Data Flow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ RAW INPUT: SAPFLUXNET (165 sites) + Sapflow-internal (20 sites) │
└──────────────────┬──────────────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
   ┌────▼────┐          ┌────▼────┐
   │SAPFLUXNET       │SI ETL    │
   │(ready)          │(reorganize,
   │                 │ convert units,
   │                 │ extract ICOS)
   └────┬────┘       └────┬────┘
        │                  │
        └──────────┬───────┘
                   │
        ┌──────────▼──────────┐
        │  STANDARDIZED SAP   │
        │  + ENVIRONMENT      │
        │  (hourly)           │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  FEATURE EXTRACTION │
        │  (ERA5, LAI, PFT,   │
        │   elevation, soil)  │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  QUALITY CONTROL    │
        │  (outliers, flags,  │
        │   validity checks)  │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  MERGED DAILY DATA  │
        │  (features ready)   │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  MODEL TRAINING     │
        │  (stratified CV,    │
        │   XGBoost + DL)     │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  GLOBAL PREDICTION  │
        │  (0.1° × 0.1°,      │
        │   daily maps)       │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  POST-ANALYSIS      │
        │  (SHAP, regional,   │
        │   climatic summary) │
        └──────────┘──────────┘
```

---

## Module Responsibilities at a Glance

| Module | Primary Classes/Functions | Key Responsibility |
|--------|---------------------------|-------------------|
| **Sapflow-internal/** | `sap_reorganizer.py`, `sap_unit_converter.py`, `icos_data_extractor.py` | Standardize 20 internal European sites to SAPFLUXNET format |
| **src/Extractors/** | `extract_*.py` (8 scripts) | Pull satellite, reanalysis, soil covariates for each site |
| **src/Analyzers/** | `SapFlowAnalyzer`, `EnvironmentalAnalyzer`, `RelationshipExplorer` | QC outliers, flags, and explore variable relationships |
| **notebooks/merge_\*.py** | Active: `merge_gap_filled_hourly_orginal.py` | Merge sapflow, environment, features into daily training data |
| **src/hyperparameter_optimization/** | `BaseOptimizer`, `GridSearchOptimizer`, `RandomSearchOptimizer` | Grid/random search over XGBoost and DL hyperparameters |
| **src/cross_validation/** | `TimesSeriesSplit`, `GroupedTimeSeriesSplit`, `BlockCV` | Prevent temporal/spatial data leakage during training |
| **src/make_prediction/** | `ModelConfig`, `process_era5land_gee_opt_fix.py` | Apply trained model globally, generate predictions |

---

## Key Configuration & Paths

**Central path registry:** `/path_config.py`
- All input/output paths defined via `PathConfig` class
- Scales: `sapwood`, `plant`, `site`

**Model configuration:** `src/make_prediction/config.py`
- Prediction spatial extent (60°S–78°N, 180°W–180°E)
- Temporal window, feature scaling, batch sizes

**Ruff linter:** `.ruff.toml` (Python 3.10, scientific code conventions)

---

## Entry Points

| Use Case | Command | Output |
|----------|---------|--------|
| **Full pipeline** | See CLAUDE.md "Full Pipeline (run in order)" | Global sap velocity maps + SHAP analysis |
| **Explore relationships** | `python src/Analyzers/explore_relationship_observations.py --scale sapwood` | Scatter + LOWESS plots in `outputs/figures/relationship_exploration/` |
| **Run QC analysis** | `SapFlowAnalyzer().run_analysis_in_batches(batch_size=10)` | Outlier-flagged data in `sap/outliers_removed/` |
| **Hyperparameter tuning** | `python src/hyperparameter_optimization/test_hyperparameter_tuning_ML_spatial_stratified.py --model xgb --run_id my_run` | Models + SHAP in `outputs/models/xgb/{run_id}/` |

---

## Caveats & Important Notes

1. **Active merge script:** `notebooks/merge_gap_filled_hourly_orginal.py` — the GSI-based variant (`merge_gap_filled_hourly.py`) is NOT part of the main pipeline
2. **LAI data:** `extract_globmap_lai.py` provides GlobMap LAI (used in merge); `extract_lai.py` provides MODIS+AVHRR via GEE but is standalone and NOT consumed
3. **Gap-filling scripts:** Inputs to `sap_gap_filling.py`, `env_gap_filling.py` are in `filtered/` dirs and are independent of main pipeline
4. **ERA5 VPD:** Derived from 2-m dewpoint, may differ from site-measured VPD; see `notebooks/VPD_mismatch_analysis.py`
5. **Canopy height dependency:** `extract_canopyheight.py` MUST run before `spatial_features_extractor.py`
6. **Claude infrastructure:** `everything-claude-code/` is unrelated Claude Code plugin code; ignore for science pipeline

---

## Quick Links

- **Pipeline instructions:** [CLAUDE.md](../../CLAUDE.md)
- **Technical deep-dive:** [docs/technical_documentation.md](../technical_documentation.md)
- **Data flow diagram:** [docs/data_flow.md](../data_flow.md)
- **Methods review:** [docs/sapflow_methods_review.md](../sapflow_methods_review.md)

---

## How to Update This Index

When adding new modules or changing architecture:

1. Update the relevant stage codemap (e.g., `analyzers-stage3.md`)
2. Update the "Module Responsibilities" table above
3. Update the data flow diagram if pipeline flow changes
4. Update entry points if new CLI scripts are added
5. Refresh "Last Updated" timestamp

Keep all codemaps under 500 lines for readability. Link between codemaps using relative paths.
