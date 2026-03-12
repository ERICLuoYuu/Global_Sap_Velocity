# Codemaps Updates — 2026-03-12

**Purpose:** Document changes to architecture codemaps reflecting recent codebase updates

---

## Summary of Changes

This update introduces a comprehensive codemap system documenting the 6-stage ML pipeline and supporting infrastructure. The codemaps are generated from the current codebase state and reflect three recent additions/changes:

1. **New analyzer:** `explore_relationship_observations.py` — Scatter + LOWESS plots for tree-size vs sapwood area and sap-flow vs environmental relationships
2. **Unified ICOS extraction:** `icos_data_extractor.py` — Strategy pattern implementation replacing two separate scripts
3. **Code quality:** `.ruff.toml` — Linter configuration for Python 3.10 scientific code

---

## Files Created

### 1. INDEX.md

**Purpose:** Central navigation hub for all codemaps

**Contents:**
- Overview of 6-stage pipeline (Stages 1–6)
- 8 focused codemaps (one per stage + supporting modules)
- Module responsibilities table
- Key configuration & paths
- Entry points (commands to run each stage)
- Important caveats (merge script variants, LAI handling, etc.)

**Lines:** ~200 | **Status:** Navigational index

---

### 2. etl-stage1.md

**Purpose:** Sapflow-internal ETL (standardization & ICOS extraction)

**Key Content:**
- 4 sequential scripts: reorganizer, unit converter, plant-to-sapwood converter, **unified ICOS extractor**
- **NEW:** Detailed architecture of `icos_data_extractor.py`
  - Strategy pattern (L2 → Fluxnet Product fallback)
  - 17 registered site pairs (ES-LM1, ES-LM2, IT-CP2, etc.)
  - Variable mapping (TA_F → ta, VPD_F → vpd, PPFD_IN → ppfd, etc.)
  - Unified QC flag handling
- Unit conversion formulas (cm h⁻¹ → cm³ cm⁻² h⁻¹, mm h⁻¹, g h⁻¹, etc.)
- Sapwood area weighting logic
- Outputs: SAPFLUXNET-format CSVs ready for merging

**Lines:** ~400 | **Status:** Complete with new ICOS extractor details

---

### 3. analyzers-stage3.md

**Purpose:** Quality control & exploratory analysis

**Key Content:**
- 5 analyzer modules: `SapFlowAnalyzer`, `EnvironmentalAnalyzer`, `RemovalLogProcessor`, **`RelationshipExplorer` (NEW)**
- **NEW:** `RelationshipExplorer` class documentation
  - Loads tree metadata (DBH, height, sapwood area) from `*_plant_md.csv`
  - Loads sapflow + environment from SAPFLUXNET CSVs
  - Scatter plots with LOWESS trend lines
  - Spearman rank correlation (ρ) with significance stars
  - 3 plot sets:
    1. Tree size vs sapwood area (overall, by biome, by PFT)
    2. Sap flow density vs 6 env variables (vpd, ws, sw_in, ta, ppfd_in, ext_rad)
    3. Overall grid (2×3) + per-variable by-biome and by-PFT faceted plots
  - Color schemes: BIOME_COLORS (8 colors), PFT_COLORS (12 colors)
  - Subsampling & performance optimization
  - CLI: `--scale`, `--output-dir`, `--max-points`, `--use-raw` flags
  - Output: PNG figures in `outputs/figures/relationship_exploration/{raw|outlier_removed}/`
- Complete 6-step SapFlowAnalyzer pipeline (flag filter → manual removal → reverse detection → z-score outliers → variability → incomplete day removal)
- EnvironmentalAnalyzer with diurnal-split z-score for radiation variables
- RemovalLogProcessor for manual removal_log.csv handling

**Lines:** ~450 | **Status:** Complete with extensive RelationshipExplorer details

---

### 4. configuration.md

**Purpose:** Centralized path & model configuration

**Key Content:**
- `PathConfig` class (single source of truth for all paths)
  - Scales: sapwood, plant, site
  - All I/O directories documented
- `ModelConfig` (hyperparameters, spatial/temporal bounds, batch sizes)
- **NEW:** `.ruff.toml` documentation
  - Python 3.10 target, 120-char lines
  - Enabled rule sets: E, W, F, I, N, UP, B, SIM
  - Disabled rules: E501, N803, N806, B008, SIM108 (science conventions)
  - Per-file exceptions: tests (F401, S101), Sapflow-internal (B007)
  - Auto-format: double quotes, 4-space indent, preserve trailing commas
- Directory structure diagram
- Optional environment variables

**Lines:** ~280 | **Status:** Complete with ruff linter config

---

### 5. cross-validation.md

**Purpose:** Temporal & spatial CV strategies to prevent data leakage

**Key Content:**
- 3 CV approaches: `TimesSeriesSplit`, `GroupedTimeSeriesSplit`, `BlockCV`
- **Primary strategy:** Stratified Spatial GroupKFold
  - 10 spatial groups (0.05° grid)
  - Biome stratification (ensures all biomes in each fold)
  - No temporal leakage (respects time ordering)
  - 10-fold cross-validation
- Alternative strategies: temporal segmentation, leave-one-year-out, block CV
- Cross-validation metrics: R², RMSE, MAE, Spearman ρ
- Leakage prevention table (temporal, spatial, feature scaling)
- Usage in training pipeline
- Per-fold reporting
- Implementation details with code examples

**Lines:** ~220 | **Status:** Complete

---

### 6. extractors-stage2.md

**Purpose:** Feature extraction from satellite, reanalysis, soil databases

**Key Content:**
- 8 extraction scripts documented
  1. `extract_siteinfo.py` — Site metadata
  2. `extract_climatedata_gee.py` — ERA5-Land hourly
  3. `extract_globmap_lai.py` — GlobMap LAI (primary, used in merge)
  4. `extract_pft.py` — MODIS PFT
  5. `extract_canopyheight.py` — Meta canopy height + ASTER GDEM (hard prerequisite)
  6. `spatial_features_extractor.py` — WorldClim + Köppen (depends on canopy height)
  7. `extract_soilgrid.py` — SoilGrids REST API
  8. `extract_stand_age.py` — BGI stand age from NetCDF
- GEE project: `ee-yuluo-2/era5download-447713`
- Execution order with dependencies clearly marked
- Pre-download requirements table
- Output directory structure
- GEE authentication setup

**Lines:** ~340 | **Status:** Complete

---

### 7. merging-stage4.md

**Purpose:** Combine sapflow, environment, and features into daily training data

**Key Content:**
- Main merge script: `merge_gap_filled_hourly_orginal.py` (ACTIVE, primary)
- Pipeline diagram (hourly merge → feature joining → daily aggregation)
- Key steps: site iteration, hourly merge, feature alignment, daily aggregation, QC flagging
- Output format: columns, shape (~185 sites × ~8000 days = ~1.5M rows, ~60 columns)
- Daily aggregation rules (min/max/mean for temps, sum for precip & radiation, modal for categorical)
- Inactive variants: `merge_gap_filled_hourly.py` (GSI variant, NOT used)
- Gap-filling scripts as supporting (NOT in main pipeline)
- Data quality handling (missing values, completeness filtering)
- Configuration (PathConfig, aggregation rules)

**Lines:** ~310 | **Status:** Complete with clear warnings about active script

---

### 8. training-stage5.md

**Purpose:** Hyperparameter optimization & model training

**Key Content:**
- XGBoost pipeline (primary)
  - Hyperparameter grid (n_estimators, lr, depth, subsample, colsample, gamma, min_child_weight)
  - Best known params documented
  - Features: ~40–50 (dynamic + static, temporal lags)
  - Target: `sap_velocity` (cm³ cm⁻² h⁻¹), log1p-transformed for training
  - Cross-validation: 10-fold stratified spatial GroupKFold
  - Output artifacts: joblib model, scaler pickle, feature order JSON
- DL pipeline (alternative)
  - CNN-LSTM, Transformer architectures
  - Hyperparameter grids for both
- BaseOptimizer abstract class & subclasses (GridSearchOptimizer, RandomSearchOptimizer)
- SHAP interpretability analysis (TreeExplainer, summary plots, dependence plots)
- Feature engineering: z-score normalization per site, temporal lagging, static broadcasting
- Reproducibility: `random_state=42` throughout
- Model selection criteria and timing estimates

**Lines:** ~380 | **Status:** Complete

---

### 9. prediction-stage6.md

**Purpose:** Global sap velocity prediction & post-analysis

**Key Content:**
- 4 execution scripts:
  1. `process_era5land_gee_opt_fix.py` — ERA5 download & daily aggregation
  2. `predict_sap_velocity_sequential.py` — Global inference (PRIMARY)
  3. `prediction_visualization.py` — Maps & time-series plots
  4. `sap_velocity_by_climatezone_forest.py` — Regional & biome summaries
- Global extent: 60°S–78°N, 180°W–180°E, 0.1° × 0.1° daily
- Batch prediction: 1000 spatial cells, 32 temporal steps, Dask parallelization
- Forest masking (MODIS PFT), water body NA handling
- Output formats: NetCDF (primary), GeoTIFF (alternative)
- Performance estimates: 2–3 hours for 1 month, 12–24 hours for 1 year
- Regional aggregations: Köppen zones, PFT, continent, seasonal
- Checkpointing & error handling
- Memory & parallelization config

**Lines:** ~360 | **Status:** Complete

---

## Codemap Statistics

| Codemap | Lines | Purpose |
|---------|-------|---------|
| INDEX.md | ~200 | Navigation hub |
| etl-stage1.md | ~400 | Sapflow-internal ETL (unified ICOS) |
| extractors-stage2.md | ~340 | Feature extraction (8 scripts) |
| analyzers-stage3.md | ~450 | QC & exploration (new RelationshipExplorer) |
| merging-stage4.md | ~310 | Data merging (active script emphasized) |
| training-stage5.md | ~380 | Model training (XGBoost + DL) |
| prediction-stage6.md | ~360 | Global prediction & post-analysis |
| cross-validation.md | ~220 | CV strategies |
| configuration.md | ~280 | Paths, config, ruff linting |
| **TOTAL** | **~3,140** | **Complete architecture documentation** |

All codemaps are under 500 lines (except INDEX, which is a hub) for readability and focused content.

---

## Key Updates Reflected

### 1. New File: `src/Analyzers/explore_relationship_observations.py`

**Codemap:** `analyzers-stage3.md`

**Coverage:**
- RelationshipExplorer class (lines 103–879 in source)
- Loaded tree metadata (DBH, height, sapwood area)
- Sap flow vs environment relationships
- LOWESS trend fitting & Spearman correlation
- Scatter plots with faceting by biome & PFT
- Output directory structure & PNG naming
- CLI arguments (--scale, --output-dir, --max-points, --use-raw)

**Integration note:** Exploratory tool (does NOT feed into main pipeline); useful for understanding data quality before QC and training.

### 2. Unified File: `Sapflow-internal/icos_data_extractor.py`

**Codemap:** `etl-stage1.md`

**Coverage:**
- Replaces 2 separate scripts: `icos_env_extractor.py` (L2 products) + `icos_fluxnet_extractor.py` (Fluxnet fallback)
- Strategy pattern implementation (abstract EnvironmentDataStrategy, concrete L2ProductStrategy & FluxnetProductStrategy)
- ICOSSiteConfig dataclass (sapflow_name ↔ icos_id mapping)
- SITE_REGISTRY with 17 sites
- VARIABLE_MAPPING & QC_MAPPING (unified across both strategies)
- Execution flow: Try L2 → fallback to Fluxnet Product if needed
- Error handling & site-by-site logging
- Test file: `Sapflow-internal/tests/test_icos_data_extractor.py`

**Integration note:** Replaces two scripts with one unified, cleaner implementation; maintains backward compatibility with existing site mappings.

### 3. New File: `.ruff.toml`

**Codemap:** `configuration.md`

**Coverage:**
- Target: Python 3.10.11
- Line length: 120 characters
- Rule sets enabled: E, W, F, I, N, UP, B, SIM
- Rules disabled for scientific conventions (N803, N806, B008, SIM108)
- Per-file exceptions (tests, Sapflow-internal)
- Auto-format config (double quotes, 4-space indent)

**Integration note:** Code quality standards for the entire project; enforced via `ruff check` and `ruff format`.

---

## Navigation & Cross-References

**All codemaps include:**
- Related codemaps links (markdown relative paths)
- Data flow connections to upstream/downstream stages
- Cross-reference tables (modules, outputs, inputs)

**Entry points from INDEX.md:**
1. Pipeline overview (6 stages with execution order)
2. Module responsibilities table
3. Data flow diagram (ASCII art)
4. Configuration & paths section
5. Entry points (CLI commands)
6. Important caveats section

**Reading order recommended:**
1. Start: `INDEX.md` (overview)
2. For pipeline execution: Follow stage numbers (1 → 6)
3. For specific module: Jump to relevant stage codemap
4. For configuration: See `configuration.md`
5. For CV details: See `cross-validation.md`

---

## Usage Instructions

### For Developers

1. **Understanding the pipeline:** Read `INDEX.md` data flow section
2. **Adding new modules:** Update relevant stage codemap, then INDEX.md
3. **Running the pipeline:** Follow execution order in each stage codemap
4. **Configuration:** All paths/params in `configuration.md`

### For Documentation Maintenance

1. **Updating a codemap:**
   - Edit relevant `*.md` file in `docs/CODEMAPS/`
   - Update "Last Updated" timestamp
   - Regenerate examples if code changed
   - Test cross-references (markdown links)

2. **Adding a new stage or module:**
   - Create new stage codemap (keep under 500 lines)
   - Add entry to INDEX.md table
   - Add cross-references in related codemaps

3. **Validation checklist:**
   - [ ] All file paths verified to exist (relative paths from repo root)
   - [ ] All code examples compile/run (if shown)
   - [ ] Internal cross-links work (relative markdown paths)
   - [ ] ASCII diagrams align and render correctly
   - [ ] "Last Updated" timestamp is current
   - [ ] No obsolete references to removed scripts

---

## Quality Standards

All codemaps meet the following criteria:

- **Source of truth:** Generated from actual codebase, not aspirational
- **Freshness:** Last updated 2026-03-12; refresh when pipeline changes
- **Brevity:** Under 500 lines each (exceptions: INDEX as hub)
- **Actionable:** Include setup commands, execution order, expected outputs
- **Cross-reference:** Link related documentation and codemaps
- **Caveats:** Document pitfalls, dependencies, and non-obvious behaviors

---

## Files in docs/CODEMAPS/

```
docs/CODEMAPS/
├── INDEX.md                    # Navigation hub (200 lines)
├── etl-stage1.md               # Sapflow-internal ETL (400 lines)
├── extractors-stage2.md        # Feature extraction (340 lines)
├── analyzers-stage3.md         # QC & exploration (450 lines)
├── merging-stage4.md           # Data merging (310 lines)
├── training-stage5.md          # Model training (380 lines)
├── prediction-stage6.md        # Global prediction (360 lines)
├── cross-validation.md         # CV strategies (220 lines)
├── configuration.md            # Config & linting (280 lines)
└── UPDATES.md                  # This file (release notes)
```

---

## Next Steps

1. **Commit codemaps** to git with message:
   ```
   docs: add comprehensive codemaps for 6-stage ML pipeline

   - New: explore_relationship_observations analyzer
   - New: unified icos_data_extractor with strategy pattern
   - New: ruff linting configuration (.ruff.toml)
   - Docs: 9 focused codemaps covering all stages
   - Docs: architecture diagrams, config, CV strategies
   ```

2. **Update README.md** to reference `docs/CODEMAPS/INDEX.md` as architecture guide

3. **Maintain freshness:** Refresh timestamps when pipeline changes, especially:
   - Stage 1: New sites added to ICOS registry
   - Stage 2: New feature extractors added
   - Stage 3: New analyzers or QC steps
   - Stage 5: New model architectures or hyperparameters
   - Stage 6: New post-analysis aggregations

4. **Consider automation:** Could generate portions from code docstrings + AST analysis (future enhancement)

---

**Documentation completed:** 2026-03-12
**Codemaps verified:** All 9 files generated and cross-linked
**Status:** Ready for production use
