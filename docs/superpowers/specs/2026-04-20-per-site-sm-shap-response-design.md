# Per-Site Soil Moisture → Sap Flow Response: XGBoost + SHAP Design

**Date:** 2026-04-20
**Status:** Design approved; pending implementation plan
**Owner:** yu.luo110266@gmail.com
**Related prior work:** `src/Analyzers/explore_relationship_observations.py` (LOWESS scatter plots for the same drivers)

## 1. Scientific Motivation

Characterise how tree sap flow density (`sap_velocity`, cm³ cm⁻² h⁻¹) responds to soil moisture at each of the 185 SAPFLUXNET + internal European sites individually, after accounting for confounding atmospheric demand (VPD, radiation) and other drivers. Global pooled models smear regional soil-moisture regimes and dilute site-specific non-linear thresholds; per-site modelling preserves them.

Two complementary lenses on soil moisture are produced:

1. **Physical lens** — raw `volumetric_soil_water_layer_1_raw` (m³ m⁻³). Reveals absolute thresholds (wilting-point, field-capacity range).
2. **Anomaly lens** — per-site z-scored `volumetric_soil_water_layer_1_zscore`. Reveals relative stress ("sap flow drops when SM falls >1σ below site mean").

Each lens is a separate run of the same pipeline.

## 2. Data

**Source:** `outputs/processed_data/sapwood/merged/daytime_only/growing_season/daily/{SITE_CODE}_daily.csv`
(canonical daily, growing-season, daytime-only subset; see `CLAUDE.md` Stage 4).

**Features (6, no lags, no aggregations beyond what is in the CSV):**

| Column used | Semantics | Units |
|---|---|---|
| `vpd` | daily mean vapour pressure deficit | kPa |
| `ta` | daily mean air temperature | °C |
| `ws` | daily mean wind speed | m s⁻¹ |
| `sw_in` | daily mean shortwave radiation | W m⁻² |
| `precip_sum` | daily total precipitation (ERA5-Land) | mm |
| `volumetric_soil_water_layer_1_raw` **or** `..._zscore` | daily mean soil moisture layer 1 (0–7 cm, ERA5-Land) | m³ m⁻³ (raw) or z-score |

Feature column renamed to generic `sm` inside the pipeline so plot labels and output schemas stay variant-agnostic.

**Site metadata (PFT, biome)** are joined from `outputs/processed_data/sapwood/merged/site_biome_mapping.csv` for figure annotations and pool-level faceting.

**Target:** `sap_velocity` (daily mean, cm³ cm⁻² h⁻¹). No log-transform at this stage (per-site ranges are narrower than the pooled pipeline's).

**Row filters (applied per site, in order):**
1. Drop rows with NaN in any selected column.
2. Drop rows where `sap_velocity ≤ 0` (non-physical, QC residue).
3. If remaining `n_rows < 100`, skip the site and record reason `TOO_FEW_ROWS`.

## 3. Modelling Pipeline (per site)

```
load_site_data(site_code, sm_variant)
        ↓
build_feature_matrix()        →  X (n × 6), y (n,)
        ↓
tune_site_hp(X, y)            →  best_params, cv_r2_mean, cv_r2_std
        ↓
fit_final_model(X, y, best_params)   →  model, in_sample_r2
        ↓
compute_shap(model, X)        →  shap_values (n × 6), shap_interaction (n × 6 × 6)
        ↓
plot_dependence_pair(...)     →  {SITE}_SM_dependence.png
        ↓
save artifacts + append summary row
```

### 3.1 Hyperparameter search

- `sklearn.model_selection.RandomizedSearchCV`
- `n_iter = 30`, `cv = KFold(5, shuffle=True, random_state=42)`
- `scoring = "neg_root_mean_squared_error"`
- Inner workers `n_jobs = 1` (outer parallelism across sites; avoid nested joblib)

**Search space:**

```python
PARAM_DIST = {
    "max_depth":        [3, 4, 5],
    "min_child_weight": [1, 3, 5, 10],
    "n_estimators":     [200, 400, 600],
    "subsample":        [0.8, 1.0],
    "gamma":            [0.0, 0.1],
    # fixed:
    "learning_rate":    [0.05],
    "colsample_bytree": [1.0],
    "reg_alpha":        [0.0],
    "tree_method":      ["hist"],
    "random_state":     [42],
    "n_jobs":           [1],
}
```

- Early stopping not used inside the search (incompatible with sklearn CV). `n_estimators` is a searched parameter.
- Each site produces its own `best_params`; they are not shared across sites.
- `cv_r2_mean`, `cv_r2_std`, and `in_sample_r2` are recorded per site as overfitting diagnostics.

### 3.2 Final refit

`XGBRegressor(**best_params).fit(X, y)` on full site data. SHAP computed on this refit.

### 3.3 SHAP computation

```python
explainer        = shap.TreeExplainer(model)
shap_values      = explainer.shap_values(X)              # (n, 6)
shap_interaction = explainer.shap_interaction_values(X)  # (n, 6, 6)
main_effect_sm   = shap_interaction[:, sm_idx, sm_idx]   # pure main effect, interactions removed
```

**Sanity check** (asserted in tests): `shap_values[:, i] ≈ shap_interaction[:, i, :].sum(axis=1)` per row, tolerance 1e-4.

## 4. Output

### 4.1 Per-site 2-panel figure

- **Left panel** — standard SHAP dependence: x = SM, y = SHAP_SM, colour = VPD (viridis). Vertical scatter visualises interaction-driven modulation.
- **Right panel** — pure main-effect dependence: x = SM, y = `main_effect_sm`. LOWESS overlay (`frac=0.3`, `statsmodels.nonparametric.smoothers_lowess.lowess`).
- Figtitle: `{SITE_CODE}  PFT={PFT}  biome={BIOME}  SM={raw|zscore}`
- Figtext below: `n={n_rows}   CV-R²={mean:.2f}±{std:.2f}   in-sample R²={r2:.2f}   best_params=...`
- Axes: x-axis `SM (m³/m³)` for raw, `SM z-score` for zscore; y-axis `SHAP value (Δ sap_velocity, cm³ cm⁻² h⁻¹)`; horizontal dashed grey line at y=0.
- Size 12 × 5 in, dpi 150, PNG.

### 4.2 Pool-level figure

One per SM variant, written after all sites finish. Small-multiples grid of the right-panel curve only, faceted by biome or PFT, shared axes. For manuscript-level cross-site comparison.

### 4.3 Directory layout

```
outputs/analysis/per_site_sm_shap/
├── raw/
│   ├── plots/{SITE}_SM_dependence.png
│   ├── pool/pool_by_biome.png
│   ├── pool/pool_by_pft.png
│   ├── shap_values/{SITE}_shap.parquet
│   │     columns: [TIMESTAMP, sm, shap_sm, shap_vpd, shap_ta,
│   │              shap_ws, shap_sw_in, shap_precip_sum, main_effect_sm]
│   ├── models/{SITE}.joblib
│   └── summary.csv
├── zscore/
│   └── ... (same layout)
└── run_log.txt
```

### 4.4 `summary.csv` schema

| Column | Type | Notes |
|---|---|---|
| site_code | str | |
| sm_variant | str | `raw` or `zscore` |
| status | str | `OK` / `TOO_FEW_ROWS` / `MISSING_FILE` / `MISSING_SM_VARIANT` / `CV_FAILED` / `SHAP_FAILED` |
| n_rows | int | after NaN drop |
| PFT | str | from site metadata |
| biome | str | from site metadata |
| cv_r2_mean | float | |
| cv_r2_std | float | |
| in_sample_r2 | float | |
| best_params | str | JSON-encoded dict |
| sm_shap_mean_abs | float | mean |SHAP_SM|, a site-level SM importance |
| sm_main_effect_range | float | max(main_effect_sm) − min(main_effect_sm), captures SM non-linearity |
| runtime_sec | float | |

## 5. Error Handling

All errors are logged as rows in `summary.csv` with a populated `status` column and a human-readable line appended to `run_log.txt` (UTC ISO timestamp + site_code + reason). Skipped sites do not halt the run.

| Condition | `status` | Action |
|---|---|---|
| Site CSV missing | `MISSING_FILE` | skip |
| `n_rows < 100` | `TOO_FEW_ROWS` | skip |
| Any feature column all-NaN | `MISSING_FEATURE_{name}` | skip |
| SM variant column missing | `MISSING_SM_VARIANT` | skip (handles sites predating commit `9c7a149`) |
| `RandomizedSearchCV` raises | `CV_FAILED` | skip, log message |
| `shap_interaction_values` raises | `SHAP_FAILED` | save `shap_values` + left panel only, log, status `OK_NO_INTERACTION` |
| Site succeeds | `OK` | save all artifacts |

No `try/except` swallows exceptions silently: all caught exceptions are re-logged with traceback to `run_log.txt`.

## 6. Testing

`src/Analyzers/tests/test_per_site_sm_shap.py` (pytest):

- `test_build_feature_matrix_drops_nans` — synthetic DataFrame; verifies NaN rows removed and `sap_velocity ≤ 0` rows removed.
- `test_skip_when_n_below_threshold` — n=50 returns `None` with reason `TOO_FEW_ROWS`.
- `test_sm_variant_switch_selects_correct_column` — both `raw` and `zscore` variants resolve and rename to `sm`.
- `test_hp_search_returns_valid_params` — synthetic (n=150, 6 features); `best_params` contains all searched keys.
- `test_shap_interaction_identity` — per row, `|shap_values[i, j] − shap_interaction[i, j, :].sum()| < 1e-4`.
- `test_main_effect_has_same_length_as_X` — shape assertion.
- `test_end_to_end_one_site` (@pytest.mark.slow) — one real site end-to-end; verifies PNG exists + summary row appended + parquet readable.

## 7. Execution

### 7.1 HPC SLURM job

`.claude/plan/job_per_site_sm_shap.sh`:

```bash
#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --job-name=per_site_sm_shap
#SBATCH --output=logs/per_site_sm_shap_%j.out

cd /scratch/tmp/yluo2/gsv
source .venv/bin/activate

python src/Analyzers/per_site_sm_shap.py --sm-variant raw    --n-jobs 20
python src/Analyzers/per_site_sm_shap.py --sm-variant zscore --n-jobs 20
```

### 7.2 CLI contract

```
python src/Analyzers/per_site_sm_shap.py \
    --sm-variant {raw,zscore}       (required)
    --n-jobs INT                    (default: os.cpu_count())
    --min-rows INT                  (default: 100)
    --sites SITE [SITE ...]         (optional filter for debugging)
    --output-dir PATH               (default: outputs/analysis/per_site_sm_shap)
    --random-seed INT               (default: 42)
```

Exit code 0 if at least one site succeeded; 1 only on catastrophic failure (e.g., output dir unwritable).

**Re-run semantics:** the script overwrites `summary.csv`, plots, models, and parquet artifacts on each invocation — no append-and-merge. This keeps the output directory consistent with the most recent HP search and SHAP computation.

### 7.3 Resource budget

- 185 sites × 30 HP trials × 5 folds × 2 SM variants ≈ 56 000 XGBoost fits.
- Fit time ≈ 10 ms each (hist method, n ≤ 400, 6 features).
- 20-worker joblib across sites → ~30 min per SM variant ≈ 1 h total wall time.
- 40 GB memory is generous: ~50 MB peak per worker × 20 = ~1 GB actual.

## 8. Non-Goals

- No hourly analysis. Daily is the natural timestep for slow-varying SM.
- No time-lag features. Current-day SM already encodes prior precipitation; lags would dilute SM SHAP via collinearity.
- No deeper soil layers (layer 2/3/4) in this design. Layer 1 only. Deeper-layer extension is a follow-up design.
- Not replacing `src/Analyzers/explore_relationship_observations.py` — that script stays for model-free LOWESS exploration.
- No training-pipeline modifications. This is a downstream analysis script, not a new model variant.
- No new features introduced to `path_config.py`.

## 9. Open Follow-ups (out of scope for this spec)

1. Extend to soil layers 2/3/4 with a `--sm-layer` flag.
2. Dry-down episode identification (rain-free windows) as a SHAP stratification variable — flagged in prior session notes.
3. Cross-site pooling of main-effect curves weighted by PFT area — for biome-level SM response functions.
