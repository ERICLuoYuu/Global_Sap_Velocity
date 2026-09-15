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
# Builds a SERVABLE tree by default (see Feature contract below)
python notebooks/merge_gap_filled_hourly_orginal.py

# The pre-2026-09-16 behaviour, if an analysis needs gauge precip and per-site soil moisture
python notebooks/merge_gap_filled_hourly_orginal.py --feature-contract site
```

Output: `outputs/processed_data/sapwood/merged/merged_data.csv` + `hourly/` + `daily/`

#### Feature contract (`--feature-contract`, default `servable`)

The merge emits servable features directly, so a tree is ready to train and to serve with no pass
afterwards. Two columns mean something different under each contract:

| | `site` (legacy) | `servable` (default) |
|---|---|---|
| `precip` (daily) | gauge daily MEAN of hourly mm | ERA5-Land daily TOTAL, mm/day (~24× larger) |
| `volumetric_soil_water_layer_L` | `raw / σ` of the site's own record | `raw / σ_clim` of the 0.1° cell |
| `..._zscore` | z-score over the site's hours | `(raw − μ_clim) / σ_clim` |
| gauge totals | under `precip*` | kept under `precip_gauge*` |

`--swvl-cube` selects the climatology (default `swvl_sigma_climatology_v2_1995_2019.nc`). A servable
tree writes `feature_contract.json` + `feature_contract_cells.csv` beside `daily/`, recording the
cube, its sha256 and the cell every site was normalised against. `assert_dataset_contract` reads
either that sidecar or the `feature_contract` block in `RUN_PROVENANCE.json`, and **refuses to read a
servable tree under `site`** — the `_zscore` columns are climatological anomalies, not per-site
z-scores, and nothing else would notice the difference.

This replaces two passes that used to follow the merge: `src/feature_contract/rebuild_daily.py`
(still the only route for trees built before this) and the `_servable_precip` override in
`feature_engineering.py` (still the producer for the prediction grid, and it recomputes the identical
value). `rebuild_daily` refuses a tree that already carries the sidecar, because normalising an
already-normalised tree would square σ and nothing downstream could detect it.

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
python src/make_prediction/predict_sap_velocity_sequential.py
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
| `src/qc_flags/`                                  | `build_mask` / `Mechanism` / `RunReport`     | Quality masking for all three flag mechanisms; both analyzers call it. Also `removal_log_validator`, which fails the run on any log row that cannot act                                                     |
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
- **`SapFlowAnalyzer.run_analysis_in_batches` does not discover files.** Its body replaces the glob with a hardcoded one-element list (`sites_with_negatives = ["ESP_MAJ_NOR_LM1"]`), so the Stage-3 sap command above processes **one site, not the corpus** — long-standing, since commit `4f09776`. `EnvironmentalAnalyzer`'s equivalent globs correctly. To process a real set, pass the files explicitly to `_load_sapflow_data(files_to_load=...)`
- **Quality flags come in three forms** and `src/qc_flags` handles all of them: 202 SAPFLUXNET sites ship a companion `*_flags.csv`, 19 internal ICOS-derived sites carry in-file `*_qc` columns (mask at `qc >= 1`), and 4 carry neither. `SWE_NOR` has no env variables at all and is skipped with a reason
- **`NOR_HUR` is a known, accepted exclusion.** Its raw env delivery is 100% complete on all 15 variables, but 97.5% of records carry `qc = 2`, so the `qc >= 1` mask leaves `ta`/`vpd`/`rh` on 279 of 14,614 rows (all in the last six days) and the site reaches training with **zero** rows. Not a merge defect — env timestamps are an exact subset of the sap days. Unlike the sites the threshold was calibrated on (level 2 there is 1.4–7.4% of rows and usually all-NaN), NOR_HUR's level-2 values are non-NaN and full-resolution. Relaxing `qc_threshold` is **not** a surgical fix: 1→3 would admit 593,356 values across all 19 in-file-qc sites (FRA_HES +135k, CHE_DAV +118k, NOR_HUR +85k). Decision 2026-08-25: leave `qc_threshold = 1` and accept the loss of this one site
- `OUT_WARN` is **never** acted on, by design — this pipeline runs its own outlier detector and stacking the archive's on top would let the archive's definition win silently. `RANGE_WARN` is the only actionable token
- Masking `sw_in` propagates to `ppfd_in` (and `ta`/`vpd` to `rh`) **only at the ICOS sites**, where `icos_data_extractor` synthesises them. At the external sites both are independent measurements — 19 of 176 carry `ppfd_in` on rows where `sw_in` is absent
- `removal_log.csv` and `removal_log.pending_review.csv` are **tracked in git** despite the blanket `*.csv` ignore (negated by name). They hold ~300 reviewed decisions and were absent from Palma until 2026-08-16, so every Stage-3 run there had applied zero manual removals
- **`zip(..., strict=)` is a `TypeError` on Palma, and ruff actively re-adds it.** `.ruff.toml` declares `target-version = "py310"`, which switches on `B905` (*zip-without-explicit-strict*); its unsafe-fix rewrites `zip(a, b)` into `zip(a, b, strict=False)` — 3.10+ syntax that Palma's Python 3.9.25 rejects. It had landed in 7 live call sites, two of them the `site_to_group` line in the DA and prediction trainers, so both crashed immediately after spatial grouping, before fitting anything. Every bare `zip()` must carry `# noqa: B905`; verified against the repo's own config that the marker survives `ruff check --fix --unsafe-fixes` while an unmarked line is rewritten. Gate: `src/hyperparameter_optimization/tests/test_python39_compat.py`. The upstream fix would be `target-version = "py39"`, but `.ruff.toml` is hook-protected
- **`FINAL_config_*.json` records how the run was configured**, via `build_run_provenance()` in `training_utils.py`: `selected_features`, `selected_features_path`, and a `split_config` block (`spatial_split_method`, `n_groups`, `grid_size`, `balanced`, `is_stratified`, `is_shuffle`, `is_only_day`, `split_type`, `time_scale`, `data_dir`). Before this, the config held only `feature_names` — the *post*-engineering list — so two runs differing only in `--spatial_split_method` (one group per site vs a 0.05° grid, worth +0.145 pooled R²) wrote configs identical apart from their scores. All six writers call it; the gate is `tests/test_run_provenance.py`. Fields use `getattr(args, ..., None)` because five of the six build their own parser — this runs *after* the model is fitted, so it must never raise
- `src/pcmci_analysis/` is **entirely untracked** — 14 `.py` files, 0 in git

## Work Log

After completing any full task loop (including tests passing and code review),
always use the work-log-reporter skill to write a report.
Notion Work Log database: https://www.notion.so/ec1a5066f7644e3b805be43778dbad1d?v=2843abcbe2ba4fd5bfd93a96fced2b56
