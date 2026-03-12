# Cross-Validation Strategies Codemap

**Last Updated:** 2026-03-12
**Module:** `src/cross_validation/`
**Used by:** Stage 5 (hyperparameter optimization)

---

## Purpose

Prevent temporal and spatial data leakage during model training/validation. Ensure that training and test sets respect site boundaries and temporal ordering.

---

## Key Classes

### 1. `TimesSeriesSplit`

**Purpose:** Leave-one-segment-out temporal CV

**Behavior:**
- Divides time series into consecutive non-overlapping segments
- Each fold: Train on all segments except one; validate on held-out segment
- Respects temporal ordering (no future data leakage)

**Use Case:** Validate temporal generalization

---

### 2. `GroupedTimeSeriesSplit`

**Purpose:** Time-series CV with site grouping

**Behavior:**
- Groups data by site
- Each fold: Hold out all records from N sites; train on remaining sites
- Respects temporal ordering within each site

**Use Case:** Validate spatial generalization (site-level cross-validation)

---

### 3. `BlockCV`

**Purpose:** Block cross-validation for spatially auto-correlated data

**Features:**
- Defines rectangular spatial blocks (~0.05° grid)
- Respects spatial auto-correlation decay from variogram analysis
- Each fold: Train on blocks outside region; validate on held-out block

**Parameters:**
- Block size derived from empirical variogram
- Minimum buffer distance between train/test blocks

**Use Case:** Account for spatial auto-correlation in prediction

---

## Primary Strategy: Stratified Spatial GroupKFold

**File:** `src/hyperparameter_optimization/test_hyperparameter_tuning_ML_spatial_stratified.py`

**Approach:**

1. **Spatial grouping:** Cluster sites into 10 groups using 0.05° grid
2. **Stratification variable:** Biome (ensures each fold has representation from all biomes)
3. **Fold structure:** 10-fold GroupKFold
   - Each fold: Hold out one spatial group; train on 9 groups
   - All sites in a group are kept together (no leakage)

**Code:**

```python
from sklearn.model_selection import GroupKFold

spatial_groups = assign_spatial_groups(sites, grid_size=0.05)  # 10 groups
biome_strata = extract_biome_labels(sites)

splitter = GroupKFold(n_splits=10)
for train_idx, val_idx in splitter.split(X, groups=spatial_groups):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    # Biome representation preserved in each fold
```

**Characteristics:**
- **Spatial separation:** No site appears in both train and validation
- **Temporal ordering:** All data within fold respects time ordering
- **Stratification:** Biome distribution balanced across folds
- **Robustness:** Tests model ability to generalize to unseen sites

---

## Alternative Strategies (Available)

### Temporal Segmentation

**When:** Testing temporal generalization over years

**Approach:**
- Divide data into annual or multi-year segments
- Leave-one-segment-out CV

### Leave-One-Year-Out (LOYO)

**When:** Evaluating inter-annual variability

**Approach:**
- Train on all years except one
- Validate on held-out year across all sites

### Block Cross-Validation

**When:** Accounting for spatial auto-correlation

**Approach:**
- Define rectangular blocks from variogram
- Train on non-adjacent blocks; validate on block
- Ensures spatial independence

---

## Cross-Validation Metrics

**Reported for each fold:**
- **R² score** (coefficient of determination)
- **RMSE** (root mean squared error, cm³ cm⁻² h⁻¹)
- **MAE** (mean absolute error)
- **Spearman ρ** (rank correlation)

**Aggregation:**
- Mean CV score ± std across all folds
- Feature importance stable across folds?
- Fold-wise performance consistency

---

## Implementation Details

### Data Leakage Prevention

| Leakage Type | Prevented By | Mechanism |
|--------------|--------------|-----------|
| Temporal | TimesSeriesSplit, GroupedTimeSeriesSplit | Respect time ordering; no future data in training |
| Spatial | GroupKFold with spatial grouping | Sites kept together; no spatial interpolation across groups |
| Feature leakage | Feature scaling per fold | Fit scaler on train; apply to validation |

### Stratification in GroupKFold

```python
# Biome-stratified split: ensure each fold has all biomes
splitter = StratifiedGroupKFold(
    n_splits=10,
    shuffle=False,  # Preserve temporal order
)

for train_idx, val_idx in splitter.split(
    X, y=biome_labels, groups=spatial_groups
):
    # Each fold preserves biome distribution
```

---

## Usage in Training Pipeline

**File:** `src/hyperparameter_optimization/test_hyperparameter_tuning_ML_spatial_stratified.py`

**Example:**

```bash
python src/hyperparameter_optimization/test_hyperparameter_tuning_ML_spatial_stratified.py \
  --model xgb \
  --RANDOM_SEED 42 \
  --n_groups 10 \
  --SPLIT_TYPE spatial_stratified \
  --TIME_SCALE daily \
  --IS_CV True
```

**Parameters:**
- `--SPLIT_TYPE`: `spatial_stratified` (primary), `temporal`, `block`
- `--n_groups`: Number of spatial/temporal groups (default: 10)
- `--IS_CV`: Enable cross-validation (default: True)

---

## Output & Reporting

**Files generated:**
- `outputs/models/xgb/{run_id}/cv_results.csv` — Per-fold metrics
- `outputs/plots/hyperparameter_optimization/xgb/{run_id}/cv_fold_performance.png` — Fold distributions

**Metrics reported:**
```
Fold    Train_R²    Val_R²    Train_RMSE    Val_RMSE    Biomes_in_Fold
1       0.82        0.74      0.15          0.22        8/8
2       0.81        0.73      0.15          0.23        8/8
...
Mean    0.82±0.01   0.73±0.02 0.15±0.01     0.23±0.02
```

---

## Related Codemaps

- **[training-stage5.md](training-stage5.md)** — Uses these CV strategies during model training
- **[INDEX.md](INDEX.md)** — Overview

---

## Notes

1. **GroupKFold is primary:** Prevents site leakage; most robust for spatial ML
2. **Stratification critical:** Ensures biome representation in each fold
3. **No shuffling:** Time ordering preserved to prevent temporal leakage
4. **Block CV optional:** Use if spatial auto-correlation is concern; more conservative but slower
5. **Reproducibility:** Set `random_state=42` in splitters and model for consistent results
