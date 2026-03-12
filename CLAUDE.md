# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A machine learning pipeline for predicting global tree sap velocity (cm³ cm⁻² h⁻¹) at 0.1° × 0.1° daily resolution. Trains on 165 SAPFLUXNET sites + 20 internal European sites (1995–2018, 8 biomes, 33 countries), merges with ERA5-Land, GlobMap LAI, MODIS PFT, WorldClim, SoilGrids, and canopy height features, then applies the best model globally via Google Earth Engine.

**GEE project:** `ee-yuluo-2/era5download-447713`
**Central path config:** `path_config.py`

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
earthengine authenticate
earthengine set_project ee-yuluo-2
```

Key versions: Python 3.10.11, XGBoost 2.1.3, scikit-learn 1.5.2, TensorFlow 2.18.0, Keras 3.6.0, earthengine-api 1.5.1, pvlib 0.13.1, shap 0.49.1.

## Tests

```bash
pytest Sapflow-internal/tests/
pytest src/cross_validation/test_cossvalidation.py -v
```

## Full Pipeline (run in order)

### Stage 1 — Sapflow-internal ETL (internal sites only)

```bash
python Sapflow-internal/sap_reorganizer.py
python Sapflow-internal/sap_unit_converter.py
python Sapflow-internal/plant_to_sapwood_converter.py
python Sapflow-internal/icos_env_extractor.py
```

Output: `Sapflow_SAPFLUXNET_format_unitcon/sapwood/` + `env_icos/`

### Stage 2 — Feature Extraction (requires GEE auth)

```bash
python src/Extractors/extract_siteinfo.py
python src/Extractors/extract_climatedata_gee.py
python src/Extractors/extract_pft.py
python src/Extractors/extract_canopyheight.py          # Must run BEFORE spatial_features_extractor
python src/Extractors/spatial_features_extractor.py   # Auto-downloads WorldClim + Köppen
python src/Extractors/extract_globmap_lai.py           # Requires local GeoTIFFs (see pre-downloads)
python src/Extractors/extract_soilgrid.py              # SoilGrids REST API
python src/Extractors/extract_stand_age.py             # Requires local NetCDF (see pre-downloads)
```

### Stage 3 — Quality Control

```python
from src.Analyzers.sap_analyzer import SapFlowAnalyzer
analyzer = SapFlowAnalyzer(scale='sapwood')
analyzer.run_analysis_in_batches(batch_size=10, switch='both')

from src.Analyzers.env_analyzer import EnvironmentalAnalyzer
ea = EnvironmentalAnalyzer()
ea.run_analysis_in_batches(batch_size=10)
```

### Stage 4 — Data Merging

```bash
# Active script — NOT merge_gap_filled_hourly.py (that variant uses GSI growing-season logic)
python notebooks/merge_gap_filled_hourly_orginal.py
```

Output: `outputs/processed_data/sapwood/merged/merged_data.csv` + `hourly/` + `daily/`

### Stage 5 — Model Training

```bash
python src/hyperparameter_optimization/test_hyperparameter_tuning_ML_spatial_stratified.py \
  --model xgb --RANDOM_SEED 42 --n_groups 10 \
  --SPLIT_TYPE spatial_stratified --TIME_SCALE daily \
  --IS_TRANSFORM True --TRANSFORM_METHOD log1p \
  --IS_STRATIFIED True --IS_CV True \
  --SHAP_SAMPLE_SIZE 50000 \
  --run_id default_daily_nocoors_swcnor

# Deep learning (CNN-LSTM, Transformer):
python src/hyperparameter_optimization/test_hyperparameter_tuning_DL_spatial_stratified.py
```

### Stage 6 — Global Prediction

```bash
python src/make_prediction/process_era5land_gee_opt_fix.py   # ERA5 retrieval → D:/Temp/era5land_extracted/
python src/make_prediction/predict_sap_velocity_sequantial.py
python src/make_prediction/prediction_visualization.py
python src/make_prediction/sap_velocity_by_climatezone_forest.py
```

## Architecture

### Module Responsibilities

| Module                                           | Key Class / Script                           | Role                                                                                                                                                                                                        |
| ------------------------------------------------ | -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/Analyzers/sap_analyzer.py`                  | `SapFlowAnalyzer`                            | 6-step QC: flag filtering → manual QC → reverse-measurement detection → rolling z-score outlier removal (day: 3σ, night: 5σ, 30-day window) → variability filtering (2-day window) → incomplete-day removal |
| `src/Analyzers/env_analyzer.py`                  | `EnvironmentalAnalyzer`                      | Env QC: flag filtering → rolling z-score (diurnal-split for radiation vars) → standardisation → daily resampling                                                                                            |
| `src/Analyzers/mannual_removal_processor.py`     | `RemovalLogProcessor`                        | Applies `removal_log.csv` (skip_site / remove_column / remove_period) _before_ automated QC                                                                                                                 |
| `src/hyperparameter_optimization/hyper_tuner.py` | `BaseOptimizer`                              | Abstract base → `GridSearchOptimizer` / `RandomSearchOptimizer`                                                                                                                                             |
| `src/make_prediction/config.py`                  | `ModelConfig`                                | Prediction window, lat/lon bounds, Dask memory limit                                                                                                                                                        |
| `path_config.py`                                 | —                                            | All data and output paths                                                                                                                                                                                   |
| `src/cross_validation/`                          | `TimesSeriesSplit`, `GroupedTimeSeriesSplit` | Prevent temporal / site data leakage                                                                                                                                                                        |

### Model Input / Output

- **Target:** `sap_velocity` (cm³ cm⁻² h⁻¹); `log1p`-transformed for training, `expm1` at inference
- **Temporal window:** `INPUT_WIDTH = 2`, `LABEL_WIDTH = 1`, `SHIFT = 1`; feature names `{feat}_t-{lag}` (static features not windowed)
- **Dynamic features (daily):** ta, ta_min, ta_max, vpd, vpd_min, vpd_max, sw_in, ppfd_in, ext_rad, ws, precip, volumetric_soil_water_layer_1 (z-score per site), soil_temperature_level_1, LAI, precip/PET, day_length, Year sin
- **Static features:** canopy_height, elevation, PFT one-hot (MF, DNF, ENF, EBF, WSA, WET, DBF, SAV)
- **Spatial domain:** 60°S–78°N, 180°W–180°E (1000 cells/batch, 32-step time chunks)

### Primary Model (XGBoost)

Best params: `n_estimators=1000`, `learning_rate=0.05`, `max_depth=10`, `min_child_weight=5`, `subsample=0.67`, `colsample_bytree=0.8`, `gamma=0.2`. 10-fold stratified spatial GroupKFold, `RANDOM_SEED=42`, SHAP sample 50,000.

Model artifacts: `outputs/models/xgb/{run_id}/FINAL_xgb_{run_id}.joblib`, `FINAL_scaler_{run_id}_feature.pkl`, `model_config_{run_id}.json`. Feature order: `outputs/plots/hyperparameter_optimization/xgb/{run_id}/feature_units_{run_id}.json`.

### Cross-Validation

- **Primary:** Stratified spatial GroupKFold — sites binned to 10 spatial groups (0.05° grid), biome as stratification variable
- **Temporal:** `TimeSeriesSegmenter` leave-one-segment-out
- **Block CV:** `BlockCV` in `src/cross_validation/blockcv.py` — block size from variogram analysis

### Pre-Download Requirements

| Dataset              | Local Path                                                   | Source                  |
| -------------------- | ------------------------------------------------------------ | ----------------------- |
| GlobMap LAI GeoTIFFs | `data/raw/grided/globmap_lai/GlobMapLAIV3.A*.Global.LAI.tif` | globalchange.bnu.edu.cn |
| BGI Stand Age NetCDF | `data/raw/grided/stand_age/*.nc`                             | BGI/MPI-BGC data portal |

WorldClim and Köppen are auto-downloaded at runtime by `spatial_features_extractor.py`.

## Important Caveats

- **Active merge script** is `notebooks/merge_gap_filled_hourly_orginal.py` — `merge_gap_filled_hourly.py` is a GSI-based variant and is NOT the main pipeline
- `src/Extractors/extract_lai.py` (MODIS+AVHRR via GEE) is a standalone tool — its output is NOT consumed by the merge pipeline
- Gap-filling scripts (`notebooks/sap_gap_filling.py`, `notebooks/env_gap_filling.py`) read from `filtered/` dirs and are NOT part of the main pipeline
- ERA5 data bypasses `env_analyzer.py` and joins the merge directly as quality-assured reanalysis
- ERA5 VPD (derived from dewpoint) can differ from site-measured VPD — see `notebooks/VPD_mismatch_analysis.py`
- `extract_canopyheight.py` must run before `spatial_features_extractor.py` (hard dependency)
- `everything-claude-code/` is unrelated Claude Code plugin infrastructure — ignore for science pipeline work
- ERA5 intermediate cache at `D:/Temp/era5land_extracted/` can be deleted after prediction

## Work Log

After completing any full task loop (including tests passing and code review),
always use the work-log-reporter skill to write a report.
Notion Work Log database: https://www.notion.so/ec1a5066f7644e3b805be43778dbad1d?v=2843abcbe2ba4fd5bfd93a96fced2b56
