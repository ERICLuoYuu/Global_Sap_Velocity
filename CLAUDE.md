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

**Static site features come from `src/static_features`** (since 2026-11-03), not from the static
extractors below. Each of 16 products is read on a 1 km grid, anything finer aggregated to 1 km first,
and `assemble` joins them into `data/raw/extracted_data/static_1km/<scale>/`, the root the merge reads.
It supersedes the static outputs of `extract_canopyheight.py`, `spatial_features_extractor.py`,
`extract_soilgrid.py`, `extract_stand_age.py` and the Fan root-depth extractor, whose reader it reuses.
On Palma, one SLURM job per product through `src/static_features/jobs/job_module.sh`, with the Earth
Engine products (terrain, canopy, soil, tree_cover, phenology, evi) one at a time on `normal`; the
others read files (worldclim, stand_age, fan, han, yang, lusk, liang, cci, gami, lesiv):

```bash
python -m src.static_features.run --product <name> --scale sapwood_combined   # --fetch-only: download and verify inputs, write no part
python -m src.static_features.assemble --scale sapwood_combined                # refuses a missing or stale part
python -m src.static_features.checks.reproduce --scale sapwood_combined        # exit 1 on a change from the old files not named in EXPECTED_DIFFERENCES
python -m src.static_features.checks.ranges --scale sapwood_combined           # also checks.compare and checks.agreement
python -m src.static_features.checks.merge_parity --scale sapwood_combined --before <tree> --after <tree>
```

`static_quality.csv` gives every site and column a status: `ok`, `neighbour` (taken from a nearby
cell) or `no_data`. Three sites are extracted at corrected coordinates because their published point
is water (`COORDINATE_CORRECTIONS` in `src/static_features/sites.py`: CHE_DAV_SEE, GBR_DEV_CON,
GBR_DEV_DRO); the merge still places their daily climate at the published point.

Gaps are filled from the nearest cell or pixel holding a value within 25 km (`neighbours.py`; status
`neighbour`, each borrowed value listed with its distance in `<product>_neighbours.csv`): Lesiv's
dominant class, SoilGrids columns, and BGI stand age, which goes in its own column `stand_age_neighbour`
that the merge reads only after the ground `st_age`. GAMI, Han, Liang and Yang, and the MCD12Q2 dates,
are extracted but kept out of `new_static_features.csv` (too many sites lack them); seasonality reaches
the merge as the EVI harmonic (`evi_season_amplitude`, `evi_season_r2`, `evi_peak_cos_w`,
`evi_peak_sin_w`) plus `phen_cycle_share` and `phen_num_cycles`.

The remaining extractors still supply site info, ERA5, PFT and LAI:

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

**The merge needs the static root built first.** It always reads `static_1km/<scale>/` (there is no
flag) and, before reading any table, stops if a site of the sap set is missing from or repeated in one
of the five static tables, or if `provenance.json` was built for a different site list. A scale other
than `sapwood_combined` therefore cannot merge until Stage 2 has built its root. When the merge writes
`MANIFEST.json`, it records the root's site-list hash and commit under `static_provenance`.

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

Two records can name a contract and `RUN_PROVENANCE.json` outranks the sidecar, so the merge
**refuses to write into a directory whose provenance names the other contract**. Clearing the badge
is not enough on its own: a block beside it still speaks for whatever is written there next. Merge
into a different `--output-dir`, or delete the stale `feature_contract` block. A provenance file
carrying no such block — what `job_merge_sweep5.sh` writes after every merge — claims nothing and is
accepted.

Under `servable` there is no `precip_sum`; the gauge's daily total is `precip_gauge_sum`. A consumer
that wants the gauge should read whichever of the two names the tree carries, because both hold the
same quantity — the total over the gauge's OWN hours. `src/Analyzers/per_site_sm_shap.py` shows the
pattern: resolve the name on load, keep one internal name so outputs never depend on the contract.

**Every consumer defaults to `servable` as well**, so a freshly merged tree needs no flag anywhere:
`run_selection.py --feature_contract`, the trainer's `--FEATURE_CONTRACT`, and `FEATURE_CONTRACT` in
`sweep5_config.sh`. Pass `site` for a tree merged that way, or for any tree built before 2026-09-16.
A mismatch stops the run at the dataset gate; it never trains on the other contract's features.
(`DEFAULT_CONTRACT` in `definitions.py` stays `site`: it is also the default of
`apply_feature_engineering`, and a library function must not change behaviour for callers who never
asked.)

Under `servable`, **`total_precipitation_hourly_sum` is not a feature**. It is still loaded — feature
engineering's `cwd` group reads it — but the FFS loader keeps it out of the design matrix, because
`precip` is that same ERA5-Land daily total: the merge writes precip as the daily mean × 24 × 1000,
and a complete day's sum is exactly 24 × its mean, so `precip == 1000 × total_precipitation_hourly_sum`
to the bit. Offering both would spend two selection slots on one driver and split its SHAP credit
across two names, and nothing would report a problem: both columns are present and numeric. Under
`site` they are the gauge and the reanalysis — different measurements — and both stay selectable.

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
| `src/clock_offsets/`                             | `REGISTRY` / `correct_clock` / `gate`        | Reviewed site clock offsets, applied once at the Stage-3 **load boundary** in both analyzers and nowhere else. `python -m src.clock_offsets.gate` re-measures every site from its own radiation against two independent estimators and fails on a correction that did not land                                            |
| `src/env_daily/`                                 | `env_daily_statistics` / `check_support`     | Daily env statistics from env's **own** hours under a support rule (continuous columns ≥ 20 h; radiation dark-filled and clearness-scaled, valid at ≥ 0.8 TOA coverage), `ext_rad` from geometry. The merge calls it; `check_support` raises before every daily write; `python -m src.env_daily.gate` refuses an old tree |
| `src/static_features/`                           | `run_product` / `assemble` / `checks`        | Static site features at 1 km for the merge: 16 products, one part each with a per-site status and the commit that built it; `assemble` writes `static_1km/<scale>/` and its `provenance.json`; `checks.reproduce` and `checks.merge_parity` exit 1 on any change not named with a reason |
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
- **Daily env values come from env's OWN hours** (since 2026-09-17), not from the sap∩env join the rest of the daily frame uses. The archive ships `sw_in = NA` at night at 8 sites while `ext_rad = 0` is present, and a skip-NaN mean made a daylight mean that was divided by a 24-hour `ext_rad` — the trained `sw_in` feature was 20–97 % too high there. `src/env_daily/support.py` holds the rule and `env_daily_rule.version` in the tree's provenance block records which rule a tree was built under
- **`ext_rad` is always recomputed from geometry** and never read from the archive column, which is displaced 2–11 h at 15 sites (`fill_ext_rad` never overwrote a shipped value). Training and serving therefore share one definition, `src/serving/era5_daily.ext_rad_daily`
- **Site clocks are corrected at load**, from the reviewed registry in `src/clock_offsets/registry.py`: 7 `applied` entries (ISR_YAT_YAT in two periods, USA_HUY_LIN_NON, USA_ORN_ST2_AMB env and sap, FIN_PET env, SWE_SVA_MIX_NON sap) and 17 `acknowledged` ones that are measured and deliberately left alone. Positive offset = the label is LATE; true time = label − offset. Nothing downstream of Stage 3 knows about offsets
- ERA5 VPD (derived from dewpoint) can differ from site-measured VPD — see `notebooks/VPD_mismatch_analysis.py`
- `extract_canopyheight.py` must run before `spatial_features_extractor.py` (hard dependency; old extractors only, the merge no longer reads their static output)
- **Static features differ between training and serving.** Training reads the 1 km values from `src/static_features`; `src/serving/statics.py` still serves its own elevation, canopy height and soil. Its `weighted_soil_properties` changed with the merge on 2026-11-03 (a soil property with no value at any depth is now empty, not 0), but no built serving store has been rebuilt since
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
