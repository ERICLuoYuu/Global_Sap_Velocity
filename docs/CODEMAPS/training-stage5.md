# Stage 5: Model Training & Tuning Codemap

**Last Updated:** 2026-03-12
**Module:** `src/hyperparameter_optimization/`
**Entry Points:** `test_hyperparameter_tuning_ML_spatial_stratified.py`, `test_hyperparameter_tuning_DL_spatial_stratified.py`

---

## Purpose

Optimize hyperparameters and train tree-ensemble (XGBoost) and deep-learning models (CNN-LSTM, Transformer) on merged sapflow data with stratified spatial cross-validation.

---

## Training Pipeline

```
┌──────────────────────────┐
│ MERGED DAILY DATA        │
│ (185 sites, ~1.5M rows)  │
└────────┬─────────────────┘
         │
    ┌────▼──────────────────┐
    │ TEMPORAL WINDOWING    │
    │ INPUT_WIDTH=2, SHIFT=1│
    │ → {feat}_t-0, _t-1    │
    └────┬──────────────────┘
         │
    ┌────▼──────────────────────────┐
    │ SPATIAL STRATIFIED CV         │
    │ 10 spatial groups × 10 folds  │
    │ Biome stratification          │
    └────┬──────────────────────────┘
         │
    ┌────▼──────────────────┐
    │ HYPERPARAMETER GRID   │
    │ (XGBoost, DL)         │
    └────┬──────────────────┘
         │
    ┌────▼──────────────────┐
    │ TRAIN-VALIDATE LOOP   │
    │ (per fold, per config)│
    └────┬──────────────────┘
         │
    ┌────▼──────────────────┐
    │ MODEL SELECTION       │
    │ Best CV score         │
    └────┬──────────────────┘
         │
    ┌────▼──────────────────┐
    │ FINAL MODEL           │
    │ (retrain on all data) │
    └─────────────────────┘
```

---

## ML Pipeline (XGBoost)

**File:** `test_hyperparameter_tuning_ML_spatial_stratified.py`

**Execution:**

```bash
python src/hyperparameter_optimization/test_hyperparameter_tuning_ML_spatial_stratified.py \
  --model xgb \
  --RANDOM_SEED 42 \
  --n_groups 10 \
  --SPLIT_TYPE spatial_stratified \
  --TIME_SCALE daily \
  --IS_TRANSFORM True \
  --TRANSFORM_METHOD log1p \
  --IS_STRATIFIED True \
  --IS_CV True \
  --SHAP_SAMPLE_SIZE 50000 \
  --run_id default_daily_nocoors_swcnor
```

### Hyperparameter Grid (XGBoost)

| Parameter | Grid | Rationale |
|-----------|------|-----------|
| n_estimators | [500, 1000, 1500] | Tree count; 1000 typically optimal |
| learning_rate | [0.01, 0.05, 0.1] | Gradient boost shrinkage |
| max_depth | [5, 8, 10, 12] | Tree complexity; 10 common for tabular |
| min_child_weight | [1, 5, 10] | Minimum sample per leaf |
| subsample | [0.6, 0.67, 0.8] | Row sampling for each tree |
| colsample_bytree | [0.6, 0.8, 1.0] | Feature sampling per tree |
| gamma | [0, 0.1, 0.2] | Minimum loss reduction to split |

**Best known params:** `n_estimators=1000, lr=0.05, depth=10, min_child_weight=5, subsample=0.67, colsample=0.8, gamma=0.2`

### Feature Input

**Dynamic (time-varying):**
- `ta_t-0`, `ta_t-1` (air temperature, 2-day lag)
- `vpd_t-0`, `vpd_t-1` (vapor pressure deficit)
- `sw_in_t-0`, `sw_in_t-1` (shortwave radiation)
- `ws_t-0`, `ws_t-1` (wind speed)
- `precip_t-0`, `precip_t-1` (precipitation)
- `lai_t-0`, `lai_t-1` (leaf area index)
- `ppfd_in_t-0`, `ppfd_in_t-1` (PPFD if available)
- Derived: `precip_pet_t-0`, `day_length_t-0`, `year_sin_t-0`

**Static (time-invariant per site):**
- Canopy height
- Elevation
- PFT one-hot (MF, DNF, ENF, EBF, WSA, WET, DBF, SAV)
- SoilGrids properties (bulk density, clay, silt, sand, organic C, pH)
- WorldClim bioclimatic variables (20 variables)

**Total:** ~40–50 features per observation

### Target & Transformation

- **Target:** `sap_velocity` (cm³ cm⁻² h⁻¹)
- **Transform (for training):** `log1p` (log(1 + x))
  - Stabilizes variance
  - Reduces impact of outliers
  - Predictions back-transformed via `expm1`

### Cross-Validation

- **Method:** 10-fold Stratified Spatial GroupKFold
- **Grouping:** 10 spatial groups (0.05° grid)
- **Stratification:** Biome labels (ensures all biomes in each fold)
- **Temporal:** Respect time ordering (no future leakage)

### Outputs

```
outputs/models/xgb/{run_id}/
├── FINAL_xgb_{run_id}.joblib              # Trained model (pickled)
├── FINAL_scaler_{run_id}_feature.pkl      # Feature scaler (StandardScaler)
├── model_config_{run_id}.json             # Hyperparameters + metadata
├── feature_units_{run_id}.json            # Feature order and units
├── cv_results.csv                         # Per-fold metrics
└── feature_importance.csv                 # SHAP-based feature importance (top 20)

outputs/plots/hyperparameter_optimization/xgb/{run_id}/
├── cv_fold_performance.png                # Box plots of R², RMSE per fold
├── feature_importance_shap.png            # SHAP summary bar chart
├── shap_dependence_*.png                  # Top 6 SHAP dependence plots
└── hyperparameter_search_heatmap.png      # Grid search results (if applicable)
```

---

## DL Pipeline (CNN-LSTM, Transformer)

**File:** `test_hyperparameter_tuning_DL_spatial_stratified.py`

**Status:** Optional alternative to XGBoost

**Architecture Options:**
- **CNN-LSTM:** 1D convolution → LSTM → dense layers
- **Transformer:** Multi-head self-attention → feed-forward → dense

**Hyperparameter Grid:**
- Units (LSTM/attention heads): [32, 64, 128]
- Dropout: [0.1, 0.2, 0.3]
- Learning rate: [0.0001, 0.0005, 0.001]
- Epochs: [50, 100, 200]

**Outputs:** Similar structure to XGBoost (models, SHAP plots, metrics)

---

## Key Classes

### `BaseOptimizer` (Abstract Base)

**Purpose:** Template for all optimization strategies

**Subclasses:**
- `GridSearchOptimizer` — Exhaustive grid search
- `RandomSearchOptimizer` — Random sampling from grid

**Methods:**
- `fit(X_train, y_train, X_val, y_val)` — Single fold training
- `tune()` — Grid/random search across hyperparameters
- `get_best_params()` — Return best config from search

---

## Feature Engineering (In-Pipeline)

### Z-Score Normalization (per site)

All dynamic variables normalized within each site to:
```
z = (x - μ_site) / σ_site
```

**Rationale:** Account for site-level differences in measurement scales

### Temporal Lagging

Features automatically lagged:
- `INPUT_WIDTH=2`: Use 2 days of history
- `LABEL_WIDTH=1`: Predict 1 day ahead
- `SHIFT=1`: No gap between input and label

### Static Feature Broadcasting

Constant per site; broadcast to all timestamps:
- Canopy height, elevation, PFT, soil, terrain, climate

---

## SHAP Interpretability Analysis

**Method:** TreeExplainer (XGBoost), GradientExplainer (DL)

**Outputs:**
- **SHAP value matrix:** Impact of each feature on prediction
- **Summary bar chart:** Average |SHAP| per feature
- **Dependence plots:** Feature value vs SHAP value (top 6 features)
- **Force plots:** Individual prediction explanations (sample)

**Sample size:** Configurable (default: 50,000 observations)

---

## Execution Order (Stage 5)

```bash
# 1. Hyperparameter tuning (XGBoost, spatial-stratified CV)
python src/hyperparameter_optimization/test_hyperparameter_tuning_ML_spatial_stratified.py \
  --model xgb --run_id my_run --IS_CV True

# 2. [OPTIONAL] Deep learning tuning
python src/hyperparameter_optimization/test_hyperparameter_tuning_DL_spatial_stratified.py \
  --run_id my_run_dl

# 3. [OPTIONAL] Compare model performance
python src/analysis/compare_models.py
```

**Time to completion:**
- XGBoost grid search (10 folds × 20 configs): 2–4 hours
- DL tuning (10 folds × 10 configs): 4–8 hours

---

## Configuration

**File:** `src/hyperparameter_optimization/config.py` (if exists)

**Key settings:**
- `RANDOM_SEED=42` — For reproducibility
- `N_JOBS=-1` — Parallel jobs (XGBoost, sklearn)
- `BATCH_SIZE=32` — For DL training

---

## Model Selection Criteria

**Primary metric:** Cross-validation R² score (coefficient of determination)

**Secondary metrics:**
- RMSE (root mean squared error)
- MAE (mean absolute error)
- Spearman rank correlation (ρ)

**Selection rule:** Highest mean CV R² across 10 folds

---

## Reproducibility

All experiments use:
- `random_state=42` in all sklearn/XGBoost calls
- `np.random.seed(42)` for NumPy operations
- `tf.random.set_seed(42)` for TensorFlow (DL only)
- Fixed data splits via stratified CV

---

## Related Codemaps

- **[cross-validation.md](cross-validation.md)** — CV strategies for training
- **[configuration.md](configuration.md)** — Path and model config
- **[merging-stage4.md](merging-stage4.md)** — Input data format
- **[prediction-stage6.md](prediction-stage6.md)** — Uses trained model
- **[INDEX.md](INDEX.md)** — Overview

---

## Notes

1. **Model artifact lifecycle:** After training, models stored in `outputs/models/` and referenced by `run_id`
2. **Feature scaler critical:** Must be applied to validation/test data in exact same way as training
3. **Class imbalance handling:** Not applicable (continuous target); focus on outliers via transformation
4. **Interpretability priority:** SHAP analysis mandatory for publication; explains feature contributions
5. **GPU acceleration:** XGBoost supports GPU (tree_method='gpu_hist'); DL benefits from GPU for TensorFlow
