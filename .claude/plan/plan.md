# Implementation Plan: Area of Applicability (AOA)

**Spec:** `docs/superpowers/specs/2026-03-22-aoa-integration-design.md`
**Paper:** Meyer & Pebesma (2021), DOI: 10.1111/2041-210X.13650
**Branch:** `aoa` based on `refactor/prediction-pipeline-cleanup`
**Worktree:** `/scratch/tmp/yluo2/gsv-wt/aoa/` on Palma2 HPC

---

## 1. Problem Space

**What changes:**
- **NEW:** `src/aoa/` package — 8 modules (~1800 lines production code): `__init__.py`, `config.py`, `core.py`, `prepare.py`, `apply.py`, `aggregate.py`, `visualize.py`, `backfill.py`
- **NEW:** `src/aoa/tests/` — 9 test files + conftest + fixtures (~2500 lines test code)
- **MODIFY:** Training script — ~20 lines to save fold labels, X_train backup, and call `build_aoa_reference()`
- **NEW:** 4 SLURM job scripts for HPC execution

**Untouched:**
- `predict_sap_velocity_sequential.py` — AOA decoupled from prediction
- `process_era5land_gee_opt_fix.py` — ERA5 processing unchanged
- All existing tests

**Out of scope:** DI-to-RMSE relationship (spec §18.1), multipurpose CV (§18.2), k-NN variants (§18.3)

### Codebase Facts (verified on HPC)

| Fact | Value |
|------|-------|
| Active run_id | `default_daily_without_coordinates` |
| Model features | 21: `prcip/PET, sw_in, volumetric_soil_water_layer_1, ppfd_in, soil_temperature_level_1, LAI, canopy_height, ws, vpd, ext_rad, ta, elevation, Year sin, MF, DNF, ENF, EBF, WSA, WET, DBF, SAV` |
| IS_WINDOWING | False |
| n_samples | 49,227 |
| ERA5 file pattern | `prediction_{YYYY}_{MM}_daily.csv` in `data/dataset_for_prediction/{YYYY}_daily/` |
| ERA5 columns | 29 cols — model features + extras (`annual_mean_temperature`, `annual_precipitation`, `Day sin`, `solar_timestamp`, `name`, duplicate `timestamp`) |
| SHAP CSV | **Does not exist** for current model |
| aoa_X_train.npy | **Does not exist** — training script doesn't save these yet |
| Merged training data | Per-site CSVs at `outputs/processed_data/sapwood/merged/daily/` |
| Model artifacts | `FINAL_config_*.json`, `FINAL_xgb_*.joblib`, `FINAL_scaler_*.pkl` |
| rasterio | 1.4.3 installed |
| scipy | 1.13.1 installed |
| R | Not available on HPC |

---

## 2. Approach Analysis

### Approach A: Monolithic (rejected)
Single `aoa.py` file. (-) Hard to test, 1000+ lines, mixes I/O with math.

### Approach B: Modular with pure core (chosen)
Separate `core.py` (pure math), `prepare.py` (train-time I/O), `apply.py` (predict-time I/O), `aggregate.py`, `visualize.py`.
(+) Pure functions trivially unit-testable, different HPC execution schedules, separation of concerns.
(-) More files, inter-module API surface.
**Why chosen:** Scientific method implementation demands step-by-step verifiability.

### Approach C: Integrated into prediction script (rejected)
(-) 2200-line script already too large, different lifecycle, couples AOA to prediction.

### Backfill Strategy: Retrain (recommended) vs. Re-derive

Since no SHAP CSV or X_train arrays exist for the current model:
- **Path A (recommended):** After M7 training integration, retrain with same params/seed → deterministic, all artifacts saved.
- **Path B (fallback):** `backfill.py` reloads merged CSVs, replicates feature engineering, computes SHAP via TreeExplainer on saved model, reconstructs folds.

Plan supports both. M3 (Backfill) implements Path B. M7 (Training Integration) enables Path A.

---

## 3. Failure Modes

| Failure | Mitigation | Milestone |
|---------|-----------|-----------|
| Feature name mismatch between reference and ERA5 | Validate against ModelConfig at prepare and apply | M2, M4 |
| NaN in X_train | Assert no NaN before KD-tree build | M2 |
| NaN in prediction data | Drop rows, log count, continue | M4 |
| d_bar = 0 (all points identical in weighted space) | Guard → ValueError | M1 |
| All points in one CV fold | Guard → ValueError | M1 |
| All SHAP weights = 0 | d_bar = 0 → ValueError (caught by d_bar guard) | M1 |
| Memory OOM on hourly (1.3B rows) | Batch processing, configurable batch_size | M4 |
| Feature order mismatch | Reorder columns to match reference feature_names | M4 |
| ERA5 duplicate `timestamp` column | Drop duplicate before processing | M4 |
| Empty input file | Skip with warning | M4 |
| Corrupted reference NPZ | Validate expected keys on load | M2 |

---

## 4. Dependency Chain

```
M0 (Scaffolding)          → no deps
M1 (Core Algorithm)       → M0
M2 (Prepare)              → M1
M3 (Backfill)             → M2
M4 (Apply)                → M1, M2
M5 (Aggregation)          → M4 (output format)
M6 (Visualization)        → M5 (output format)
M7 (Training Integration) → M2
M8 (R CAST Reference)     → M1 (can parallel with M2-M7)
M9 (HPC Validation)       → all above
```

**Critical path:** M0 → M1 → M2 → M4 → M5 → M6 → M9

---

## 5. Test Strategy

TDD: write test file first (RED), create stubs, implement (GREEN), then refine.

| Test file | Focus | Marker | Runtime |
|-----------|-------|--------|---------|
| `test_core.py` | Mathematical correctness, known-answer | (none) | <5s |
| `test_core_properties.py` | Invariants with random data | (none) | <10s |
| `test_edge_cases.py` | Boundary conditions, errors | (none) | <5s |
| `test_cast_reference.py` | R CAST gold standard | `@slow` | <30s |
| `test_prepare.py` | prepare roundtrip, validation | (none) | <10s |
| `test_backfill.py` | backfill from saved/merged data | `@slow` | <30s |
| `test_apply.py` | apply pipeline, file I/O | (none) | <30s |
| `test_aggregate.py` | Temporal aggregation | (none) | <10s |
| `test_visualize.py` | GeoTIFF/NetCDF validity | `@requires_rasterio` | <10s |
| `test_performance.py` | HPC scalability | `@performance` | 10min+ |

```bash
# Fast CI (<2 min)
pytest src/aoa/tests/ -m "not slow and not performance" -v
# Full (<5 min)
pytest src/aoa/tests/ -m "not performance" -v
# HPC only
pytest src/aoa/tests/test_performance.py -v --timeout=600
# Coverage
pytest src/aoa/tests/ -m "not performance" --cov=src/aoa --cov-report=term-missing
```

---

## 6. Acceptance Criteria

- [x] AC-1: All `core.py` functions produce correct results verified by hand-computed known-answer tests
- [x] AC-2: Mathematical invariants hold: DI≥0, threshold≤max(training_DI), importance scale invariance, feature permutation invariance
- [x] AC-3: Edge cases raise appropriate errors: NaN→ValueError, d_bar=0→ValueError, single fold→ValueError
- [x] AC-4: `prepare.py` builds reference NPZ with all 10 fields: `reference_cloud_weighted`, `feature_means`, `feature_stds`, `feature_weights`, `d_bar`, `threshold`, `fold_assignments`, `feature_names`, `training_di`, `d_bar_method`
- [x] AC-5: `prepare.py` validates no NaN, correct shapes, ≥2 unique folds
- [x] AC-6: `apply.py` computes DI for prediction grid, saves per-timestamp parquet (lat,lon,timestamp,DI,aoa_mask) + monthly summary (lat,lon,median_DI,mean_DI,std_DI,frac_inside_aoa,n_timestamps)
- [x] AC-7: `apply.py` validates features against ModelConfig.feature_names, errors on mismatch
- [x] AC-8: `apply.py` handles CSV and parquet input via extension detection
- [x] AC-9: `apply.py` supports windowed (IS_WINDOWING=True) and non-windowed models — (15 new tests)
- [x] AC-10: `aggregate.py` produces yearly (30), climatological (12), overall (1) summaries
- [x] AC-11: `visualize.py` produces valid GeoTIFF: EPSG:4326, 0.1°, 2 bands (DI float32, AOA mask float32)
- [x] AC-12: `backfill.py` creates reference NPZ from merged training data + saved model — (99K samples, d_bar=0.3201, threshold=0.5091)
- [x] AC-13: Training script calls `build_aoa_reference()`, saves reference NPZ + X_train parquet + cv_folds NPZ — (code done, requires retrain for exact artifact match)
- [x] AC-14: R CAST comparison: DI, d_bar match within 1e-10; threshold uses whisker formula (CAST impl differs from paper) — (6 tests)
- [x] AC-15: `apply.py` processes 1 month daily on zen4 in 286s (< 1800s target) — parallel KDTree (workers=36) + parquet input. See Deviation 13.
- [x] AC-16: `aoa_meta.json` written with all spec §6 metadata fields
- [x] AC-17: `--generate-slurm` flag writes working SLURM array job script
- [x] AC-18: Test coverage ≥ 80% for `src/aoa/` — (90% achieved, 230 tests)

---

## Milestones

### M0: Project Scaffolding

**Goal:** Package structure, config dataclass, shared fixtures, pytest markers.

#### Step 0.1: Create package directories

**Files:** `src/aoa/__init__.py` (create), `src/aoa/tests/__init__.py` (create), `src/aoa/tests/fixtures/` (dir)
**Deps:** None
**Spec:** §3

```python
# src/aoa/__init__.py
"""Area of Applicability (AOA) — Meyer & Pebesma (2021)."""
__version__ = "0.1.0"
```

**AC:** `python -c "import src.aoa"` succeeds.

#### Step 0.2: Create AOAConfig dataclass

**Files:** `src/aoa/config.py` (create)
**Deps:** 0.1
**Spec:** §7

```python
from dataclasses import dataclass, field
from pathlib import Path

@dataclass(frozen=True)
class AOAConfig:
    model_type: str
    run_id: str
    time_scale: str                          # "daily" or "hourly"
    aoa_reference_path: Path
    input_dir: Path                          # ERA5 preprocessed dir
    model_config_path: Path                  # FINAL_config JSON
    output_dir: Path
    iqr_multiplier: float = 1.5
    batch_size: int = 500_000
    n_jobs: int = -1
    save_per_timestamp: bool | None = None   # None → auto (True daily, False hourly)
    map_format: str = "geotiff"              # "geotiff" or "netcdf"
    years: tuple[int, ...] = ()              # immutable
    months: tuple[int, ...] = tuple(range(1, 13))

    def __post_init__(self):
        if self.time_scale not in ("daily", "hourly"):
            raise ValueError(f"time_scale must be 'daily' or 'hourly', got '{self.time_scale}'")
        if self.iqr_multiplier <= 0:
            raise ValueError(f"iqr_multiplier must be > 0, got {self.iqr_multiplier}")
        if self.map_format not in ("geotiff", "netcdf"):
            raise ValueError(f"map_format must be 'geotiff' or 'netcdf', got '{self.map_format}'")
        # Auto-resolve save_per_timestamp
        if self.save_per_timestamp is None:
            object.__setattr__(self, "save_per_timestamp", self.time_scale == "daily")
```

**Edge cases:** Invalid time_scale → ValueError; iqr_multiplier ≤ 0 → ValueError; mutation → FrozenInstanceError. Using `tuple` instead of `list` for immutable years/months.

**AC:** Instantiates with valid args, rejects invalid, is immutable, auto-resolves save_per_timestamp.

#### Step 0.3: Create test conftest.py

**Files:** `src/aoa/tests/conftest.py` (create)
**Deps:** 0.2
**Spec:** §16.9

```python
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# --- Constants ---
N_TRAIN = 50
N_NEW = 20
N_FEATURES_CONT = 5
N_FEATURES_PFT = 3
N_FEATURES = N_FEATURES_CONT + N_FEATURES_PFT  # 8
N_FOLDS = 3
SEED = 42
FEATURE_NAMES_CONT = ["ta", "vpd", "sw_in", "elevation", "LAI"]
FEATURE_NAMES_PFT = ["ENF", "EBF", "DBF"]
FEATURE_NAMES = FEATURE_NAMES_CONT + FEATURE_NAMES_PFT

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (> 10 seconds)")
    config.addinivalue_line("markers", "performance: HPC-only scalability tests")
    config.addinivalue_line("markers", "requires_rasterio: requires rasterio installed")

@pytest.fixture
def rng():
    return np.random.default_rng(SEED)

@pytest.fixture
def synthetic_X_train(rng):
    """50×8 training matrix: 5 continuous + 3 one-hot PFT."""
    X_cont = rng.standard_normal((N_TRAIN, N_FEATURES_CONT))
    X_pft = np.zeros((N_TRAIN, N_FEATURES_PFT))
    for i in range(N_TRAIN):
        X_pft[i, rng.integers(0, N_FEATURES_PFT)] = 1.0
    return np.hstack([X_cont, X_pft])

@pytest.fixture
def synthetic_fold_labels(rng):
    """3-fold assignments for 50 points (balanced)."""
    labels = np.tile(np.arange(N_FOLDS), N_TRAIN // N_FOLDS + 1)[:N_TRAIN]
    rng.shuffle(labels)
    return labels

@pytest.fixture
def synthetic_shap_weights(rng):
    """Raw |SHAP| importance for 8 features."""
    return np.abs(rng.standard_normal(N_FEATURES)) + 0.01  # ensure no zeros

@pytest.fixture
def synthetic_X_new(rng):
    """20×8 prediction matrix."""
    X_cont = rng.standard_normal((N_NEW, N_FEATURES_CONT))
    X_pft = np.zeros((N_NEW, N_FEATURES_PFT))
    for i in range(N_NEW):
        X_pft[i, rng.integers(0, N_FEATURES_PFT)] = 1.0
    return np.hstack([X_cont, X_pft])

@pytest.fixture
def era5_like_df(rng):
    """Mock ERA5 CSV DataFrame with metadata + features + extras."""
    n = 100
    df = pd.DataFrame({
        "timestamp": pd.date_range("2015-01-01", periods=n, freq="D"),
        "latitude": rng.choice([50.0, 50.1, 50.2], n),
        "longitude": rng.choice([10.0, 10.1], n),
        "name": "test_site",
        "solar_timestamp": pd.date_range("2015-01-01", periods=n, freq="D"),
    })
    for feat in FEATURE_NAMES_CONT:
        df[feat] = rng.standard_normal(n)
    for feat in FEATURE_NAMES_PFT:
        df[feat] = 0.0
    # One-hot PFT assignment
    pft_idx = rng.integers(0, N_FEATURES_PFT, n)
    for i in range(n):
        df.loc[i, FEATURE_NAMES_PFT[pft_idx[i]]] = 1.0
    df["annual_mean_temperature"] = 15.0  # extra col (not in model features)
    df["Year sin"] = np.sin(2 * np.pi * np.arange(n) / 365.25)  # extra for non-model
    return df
```

**AC:** All fixtures instantiate; shapes correct (50×8, 20×8, etc.); era5_like_df has required + extra columns.

#### Step 0.4: Create fixture generator script

**Files:** `src/aoa/tests/fixtures/generate_synthetic.py` (create)
**Deps:** 0.3

**What:** Generates deterministic test data files committed to repo. Saves:
- `synthetic_train.csv`, `synthetic_new.csv`, `synthetic_weights.csv`, `synthetic_folds.npy`
- CAST-comparison data (20 training, 7 features, 3 folds, 10 new): `cast_train.csv`, `cast_new.csv`, `cast_weights.csv`, `cast_folds.csv`

**AC:** Running twice produces identical files.

#### Tests for M0

```bash
pytest src/aoa/tests/ -k "test_config or test_import" -v  # (write minimal test_config tests in conftest or separate)
```

| Test | Expected |
|------|----------|
| `import src.aoa` | No error |
| `AOAConfig(valid_args)` | Instantiates |
| `AOAConfig(time_scale="weekly")` | ValueError |
| `AOAConfig(iqr_multiplier=-1)` | ValueError |
| `config.batch_size = 100` | FrozenInstanceError |
| `AOAConfig(save_per_timestamp=None, time_scale="daily")` → `save_per_timestamp == True` | Auto-resolved |
| `AOAConfig(save_per_timestamp=None, time_scale="hourly")` → `save_per_timestamp == False` | Auto-resolved |

**Gate:** All pass.

---

### M1: Core Algorithm

**Goal:** All pure mathematical functions in `core.py`. Foundation for everything else.
**Spec:** §2 (algorithm), §9 (core.py), §16.1-16.4

#### Step 1.1: Write test_core.py (RED)

**Files:** `src/aoa/tests/test_core.py` (create)
**Deps:** M0

**What:** Known-answer unit tests for every `core.py` function.

**Key test cases — worked example for `compute_training_di`:**

```python
# 4 points in 1D weighted space, 2 folds
X_w = np.array([[0.0], [10.0], [3.0], [12.0]])
folds = np.array([0, 0, 1, 1])
# d_bar = mean(|0-10|, |0-3|, |0-12|, |10-3|, |10-12|, |3-12|)
#       = mean(10, 3, 12, 7, 2, 9) = 43/6 ≈ 7.16667
# DI[0] (fold 0): nearest in fold 1 → point 2 (d=3). DI = 3/7.16667 ≈ 0.41860
# DI[1] (fold 0): nearest in fold 1 → point 3 (d=2). DI = 2/7.16667 ≈ 0.27907
# DI[2] (fold 1): nearest in fold 0 → point 0 (d=3). DI = 3/7.16667 ≈ 0.41860
# DI[3] (fold 1): nearest in fold 0 → point 1 (d=2). DI = 2/7.16667 ≈ 0.27907
```

**Worked example for `compute_prediction_di`:**
```python
# Same training cloud, new point at x=20
# Nearest training point: point 3 (x=12, d=8). DI = 8/7.16667 ≈ 1.11628
# New point at x=3 (identical to training point 2): DI = 0/7.16667 = 0.0
```

**Worked example for `compute_threshold`:**
```python
# training_di = [0.1, 0.2, 0.3, 0.4, 2.0]
# Q25=0.2, Q75=0.4, IQR=0.2, upper_whisker=0.4+1.5*0.2=0.7
# Values <= 0.7: [0.1, 0.2, 0.3, 0.4]. Threshold = 0.4
```

**Full test list per function** (see spec §16.1 for all cases):

| Function | Tests | Count |
|----------|-------|-------|
| `standardize_features` | known output, zero-variance col→0, new data uses training stats, single row | 4 |
| `apply_importance_weights` | weights [2,0,1], zero weight→invisible, equal weights | 3 |
| `compute_d_bar_full` | equilateral triangle, 2 points, identical→ValueError, row shuffle invariance | 4 |
| `compute_d_bar_sample` | sample=N→same as full, large sample fallback | 2 |
| `compute_training_di` | 4-point worked example, fold constraint verified, same d_bar used, single fold→ValueError | 4 |
| `compute_threshold` | no outliers→max, with outlier→0.4, all identical→value, single value | 4 |
| `compute_prediction_di` | identical to training→0, far point→>1, worked example | 3 |
| `build_kdtree` | correct nearest neighbor, exact point→distance 0 | 2 |
| **Total** | | **26** |

**AC:** Tests run and fail with ImportError/NotImplementedError (RED).

#### Step 1.2: Create core.py with stubs

**Files:** `src/aoa/core.py` (create)
**Deps:** 1.1

```python
"""Pure AOA algorithm functions — no I/O, no side effects.

All functions are stateless: arrays in, arrays out.
Reference: Meyer & Pebesma (2021), Section 2.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

def standardize_features(X: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Z-score standardize features. Zero-variance columns → 0."""
    raise NotImplementedError

def apply_importance_weights(X_standardized: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Multiply standardized features by importance weights."""
    raise NotImplementedError

def compute_d_bar_full(X_weighted: np.ndarray) -> float:
    """Mean pairwise Euclidean distance in weighted space. Raises ValueError if 0."""
    raise NotImplementedError

def compute_d_bar_sample(X_weighted: np.ndarray, sample_size: int, seed: int = 42) -> float:
    """Approximate d_bar via random subsample. Falls back to full if sample_size >= N."""
    raise NotImplementedError

def compute_training_di(X_weighted: np.ndarray, fold_labels: np.ndarray, d_bar: float) -> np.ndarray:
    """Training DI: nearest neighbor in DIFFERENT fold, normalized by d_bar."""
    raise NotImplementedError

def compute_threshold(training_di: np.ndarray, iqr_multiplier: float = 1.5) -> float:
    """Outlier-removed max of training DI: max(DI[DI <= Q75 + iqr * IQR])."""
    raise NotImplementedError

def compute_prediction_di(X_new_weighted: np.ndarray, tree: cKDTree, d_bar: float) -> np.ndarray:
    """Prediction DI: nearest training neighbor (no fold constraint), normalized by d_bar."""
    raise NotImplementedError

def build_kdtree(X_weighted: np.ndarray) -> cKDTree:
    """Build cKDTree on weighted training data."""
    raise NotImplementedError
```

**AC:** Tests run and fail with NotImplementedError (RED confirmed).

#### Step 1.3: Implement core.py (GREEN)

**Files:** `src/aoa/core.py` (modify)
**Deps:** 1.2

**Implementation per function:**

```python
def standardize_features(X, means, stds):
    safe_stds = np.where(stds == 0, 1.0, stds)
    result = (X - means) / safe_stds
    result[:, stds == 0] = 0.0
    return result

def apply_importance_weights(X_standardized, weights):
    return X_standardized * weights[np.newaxis, :]

def compute_d_bar_full(X_weighted):
    if X_weighted.shape[0] < 2:
        raise ValueError("Need >= 2 training points for d_bar")
    distances = pdist(X_weighted, metric="euclidean")
    d_bar = float(np.mean(distances))
    if d_bar == 0.0:
        raise ValueError("d_bar is zero — all training points identical in weighted space")
    return d_bar

def compute_d_bar_sample(X_weighted, sample_size, seed=42):
    n = X_weighted.shape[0]
    if sample_size >= n:
        return compute_d_bar_full(X_weighted)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=sample_size, replace=False)
    return compute_d_bar_full(X_weighted[idx])

def compute_training_di(X_weighted, fold_labels, d_bar):
    unique_folds = np.unique(fold_labels)
    if len(unique_folds) < 2:
        raise ValueError("Need >= 2 CV folds for training DI")
    di = np.empty(X_weighted.shape[0])
    for fold_k in unique_folds:
        in_fold = fold_labels == fold_k
        not_in_fold = ~in_fold
        if not np.any(not_in_fold):
            raise ValueError(f"Fold {fold_k} has no cross-fold neighbors")
        tree = cKDTree(X_weighted[not_in_fold])
        distances, _ = tree.query(X_weighted[in_fold], k=1)
        di[in_fold] = distances / d_bar
    return di

def compute_threshold(training_di, iqr_multiplier=1.5):
    q25, q75 = np.percentile(training_di, [25, 75])
    iqr = q75 - q25
    upper_whisker = q75 + iqr_multiplier * iqr
    below_whisker = training_di[training_di <= upper_whisker]
    return float(np.max(below_whisker))

def compute_prediction_di(X_new_weighted, tree, d_bar):
    distances, _ = tree.query(X_new_weighted, k=1)
    return distances / d_bar

def build_kdtree(X_weighted):
    return cKDTree(X_weighted)
```

**AC:** All 26 test_core.py tests pass.

#### Step 1.4: Write test_core_properties.py

**Files:** `src/aoa/tests/test_core_properties.py` (create)
**Deps:** 1.3
**Spec:** §16.2

Test 10 properties with 5 random seeds each (see spec §16.2 for full list):
1. DI ≥ 0 always
2. DI = 0 iff identical to training point
3. threshold ≥ 0
4. threshold ≤ max(training_DI)
5. Feature permutation invariance (shuffle cols consistently → same DI)
6. Importance scale invariance (multiply all weights by c>0 → same DI)
7. Monotonicity (move point away → DI increases)
8. Row order invariance of d_bar
9. Adding duplicate training point doesn't increase DI for new points
10. DI = 1.0 when distance = d_bar

**AC:** All 50 property tests pass.

#### Step 1.5: Write test_edge_cases.py

**Files:** `src/aoa/tests/test_edge_cases.py` (create)
**Deps:** 1.3
**Spec:** §16.4

| Edge case | Expected |
|-----------|----------|
| Single training point | ValueError from d_bar |
| Two points same fold | ValueError from training_di |
| All weights zero | d_bar=0 → ValueError |
| One feature only | 1D distance, hand-verifiable |
| 100+ features | Correct (slower, but works) |
| All training DI identical | IQR=0, threshold = that value |
| Prediction at exact training location | DI = 0 |

**AC:** All edge case tests pass.

#### Tests for M1

```bash
pytest src/aoa/tests/test_core.py src/aoa/tests/test_core_properties.py src/aoa/tests/test_edge_cases.py -v
```

**Gate:** ALL pass (~80 tests). **Must pass before M2.**

---

### M2: Prepare Step (Reference Builder)

**Goal:** Build AOA reference artifact from training data + SHAP importance + CV folds.
**Spec:** §5, §9 (prepare.py)

#### Step 2.1: Write prepare tests (RED)

**Files:** `src/aoa/tests/test_prepare.py` (create)
**Deps:** M1

| Test | Input | Expected |
|------|-------|----------|
| `test_build_reference_roundtrip` | 50-pt synthetic | NPZ loads, all shapes correct |
| `test_reference_npz_all_keys` | valid inputs | All 10 keys present (incl. d_bar_method) |
| `test_reference_values_consistent` | valid inputs | d_bar>0, 0≤threshold≤max(training_di) |
| `test_build_reference_rejects_nan` | NaN in X_train | ValueError |
| `test_build_reference_rejects_shape_mismatch` | wrong fold_labels length | ValueError |
| `test_feature_names_preserved` | known names | loaded names match input |
| `test_d_bar_method_sample` | d_bar_method="sample:20" | Runs without error, d_bar>0 |
| `test_load_validates_missing_keys` | corrupted NPZ | ValueError |

#### Step 2.2: Implement prepare.py (GREEN)

**Files:** `src/aoa/prepare.py` (create)
**Deps:** 2.1, M1

```python
"""Build AOA reference artifact from training outputs."""
import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from src.aoa import core

logger = logging.getLogger(__name__)

REQUIRED_NPZ_KEYS = frozenset({
    "reference_cloud_weighted", "feature_means", "feature_stds",
    "feature_weights", "d_bar", "threshold", "fold_assignments",
    "feature_names", "training_di", "d_bar_method",
})

def build_aoa_reference(
    X_train: np.ndarray,
    fold_labels: np.ndarray,
    shap_importances: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
    run_id: str,
    d_bar_method: str = "full",
) -> Path:
    """Build and save AOA reference NPZ.

    Args:
        X_train: Raw training features (N, P) — NOT pre-scaled
        fold_labels: CV fold index per sample (N,)
        shap_importances: Mean |SHAP| per feature (P,) — raw, unnormalized
        feature_names: Feature names matching X_train columns
        output_dir: Directory for output (usually models/{type}/{run_id}/)
        run_id: Model run identifier
        d_bar_method: "full" or "sample:N"

    Returns:
        Path to saved NPZ
    """
    # Validate
    if X_train.ndim != 2:
        raise ValueError(f"X_train must be 2D, got {X_train.ndim}D")
    n, p = X_train.shape
    if fold_labels.shape != (n,):
        raise ValueError(f"fold_labels length {len(fold_labels)} != X_train rows {n}")
    if shap_importances.shape != (p,):
        raise ValueError(f"shap_importances length {len(shap_importances)} != features {p}")
    if len(feature_names) != p:
        raise ValueError(f"feature_names length {len(feature_names)} != features {p}")
    if np.any(np.isnan(X_train)):
        raise ValueError("X_train contains NaN")
    if np.any(np.isnan(shap_importances)):
        raise ValueError("shap_importances contains NaN")

    # Algorithm
    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0, ddof=1)  # sample std (ddof=1) to match R CAST's sd()
    X_s = core.standardize_features(X_train, means, stds)
    X_sw = core.apply_importance_weights(X_s, shap_importances)

    if d_bar_method == "full":
        d_bar = core.compute_d_bar_full(X_sw)
    elif d_bar_method.startswith("sample:"):
        sample_size = int(d_bar_method.split(":")[1])
        d_bar = core.compute_d_bar_sample(X_sw, sample_size)
    else:
        raise ValueError(f"Unknown d_bar_method: {d_bar_method}")

    training_di = core.compute_training_di(X_sw, fold_labels, d_bar)
    threshold = core.compute_threshold(training_di)

    logger.info(f"AOA reference: d_bar={d_bar:.4f}, threshold={threshold:.4f}, "
                f"n_train={n}, n_features={p}")

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"FINAL_aoa_reference_{run_id}.npz"
    np.savez(
        output_path,
        reference_cloud_weighted=X_sw,
        feature_means=means,
        feature_stds=stds,
        feature_weights=shap_importances,
        d_bar=np.float64(d_bar),
        threshold=np.float64(threshold),
        fold_assignments=fold_labels,
        feature_names=np.array(feature_names, dtype=str),
        training_di=training_di,
        d_bar_method=np.array(d_bar_method, dtype=str),
    )
    return output_path


def load_aoa_reference(path: Path) -> dict:
    """Load and validate reference NPZ."""
    data = dict(np.load(path, allow_pickle=True))
    missing = REQUIRED_NPZ_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Reference NPZ missing keys: {missing}")
    # Unbox scalars
    data["d_bar"] = float(data["d_bar"])
    data["threshold"] = float(data["threshold"])
    data["feature_names"] = list(data["feature_names"])
    data["d_bar_method"] = str(data["d_bar_method"])
    return data
```

**CLI** (`if __name__ == "__main__"` block) — matches spec §8:
- `--training-data`: .npy or .parquet path (required)
- `--shap-csv`: CSV with `feature`, `importance` columns (required)
- `--cv-folds`: .npy or .npz with fold_labels (required)
- `--output`: full path for output NPZ file (required) — e.g., `models/xgb/{run_id}/FINAL_aoa_reference_{run_id}.npz`
- `--d-bar-method`: "full" (default) or "sample:N"
- `--feature-names-json`: ModelConfig JSON for feature_names validation (required)

Load inputs → align SHAP to feature order via `shap_df.set_index("feature")["importance"].reindex(feature_names).values` → extract `run_id` from output filename → call `build_aoa_reference`.

**AC:** All prepare tests pass. NPZ has correct data.

#### Tests for M2

```bash
pytest src/aoa/tests/test_prepare.py -v
```
**Gate:** ALL pass.

---

### M3: Backfill (Existing Model Support)

**Goal:** Generate AOA reference for existing `default_daily_without_coordinates` model.
**Spec:** §5, §14

#### Critical context

The current model lacks SHAP CSV and X_train arrays. Backfill must:
1. Re-derive X_train from merged per-site CSVs (same pipeline as training)
2. Reconstruct CV fold labels from spatial groups
3. Compute SHAP via TreeExplainer on saved model
4. Call `build_aoa_reference()`

This is the most complex I/O module — it replicates the training data pipeline.

#### Step 3.1: Write backfill tests (RED)

**Files:** `src/aoa/tests/test_backfill.py` (create)
**Deps:** M2

| Test | Input | Expected |
|------|-------|----------|
| `test_backfill_with_saved_arrays` | mock aoa_*.npy files + SHAP CSV | Valid reference NPZ |
| `test_backfill_fold_reconstruction` | known groups + pfts + seed=42 | Fold labels deterministic |
| `test_backfill_missing_model` | wrong path | FileNotFoundError |
| `test_backfill_shap_from_model` | saved XGB model + X_train | SHAP importance computed |

#### Step 3.2: Implement backfill.py

**Files:** `src/aoa/backfill.py` (create)
**Deps:** 3.1, M2

**Two modes:**

**Mode A — saved arrays exist** (future models after M7):
```python
def backfill_from_saved_arrays(model_dir, run_id, shap_csv_path, d_bar_method="full"):
    """Use pre-saved aoa_X_train, aoa_groups, aoa_pfts_encoded."""
    X_train = np.load(model_dir / f"aoa_X_train_{run_id}.npy")
    groups = np.load(model_dir / f"aoa_groups_{run_id}.npy")
    pfts = np.load(model_dir / f"aoa_pfts_encoded_{run_id}.npy")
    # ... reconstruct folds, load SHAP, call build_aoa_reference
```

**Mode B — re-derive from merged data** (current model):
```python
def backfill_from_merged_data(
    models_dir: Path, model_type: str, run_id: str,
    data_dir: Path, d_bar_method: str = "full",
    validate_against_logs: Path | None = None,
) -> Path:
    """Re-derive training data and compute SHAP for models without saved arrays.

    Replicates the training pipeline's data loading + feature engineering.
    """
    model_dir = models_dir / model_type / run_id
    merged_data_dir = data_dir / "outputs" / "processed_data" / "sapwood" / "merged" / "daily"

    # 1. Load config for feature_names, n_samples, PFT info
    config_path = model_dir / f"FINAL_config_{run_id}.json"
    with open(config_path) as f:
        config = json.load(f)
    feature_names = config["feature_names"]
    expected_n = config["data_info"]["n_samples"]

    # 2. Load per-site CSVs from merged_data_dir
    #    Pattern: {site_id}_daily.csv
    site_csvs = sorted(merged_data_dir.glob("*_daily.csv"))

    # 3. For each site CSV:
    #    a. Read CSV
    #    b. Select base features: ta, vpd, sw_in, ppfd_in, ext_rad, ws, precip,
    #       LAI, prcip/PET, volumetric_soil_water_layer_1, soil_temperature_level_1,
    #       canopy_height, elevation, day_length, sap_velocity
    #    c. Add time features: Year sin = sin(2*pi*day_of_year/365.25)
    #    d. One-hot encode PFT using EXACT category order from training:
    #       all_possible_pft_types = ['MF', 'DNF', 'ENF', 'EBF', 'WSA', 'WET', 'DBF', 'SAV']
    #       pd.get_dummies(pd.Categorical(df['pft'], categories=all_possible_pft_types))
    #    e. Drop 'pft' and 'sap_velocity' columns
    #    f. Track site_id → group mapping via 0.05° lat/lon grid

    # 4. Concatenate all sites → X_all (N, P)
    #    Validate: X_all.shape[0] == expected_n
    #    Validate: columns match feature_names (same set, same order)

    # 5. Create spatial groups: same logic as training
    #    from sklearn.cluster import KMeans
    #    Grid sites by 0.05° resolution, assign to n_groups=10 spatial groups
    #    (replicate create_spatial_groups() from training script)

    # 6. Reconstruct fold labels:
    #    pfts_encoded = pd.factorize(pfts_all)[0]
    #    cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    #    fold_labels = np.full(N, -1, dtype=int)
    #    for fold_idx, (_, test_idx) in enumerate(cv.split(X_all, pfts_encoded, groups)):
    #        fold_labels[test_idx] = fold_idx

    # 7. Compute SHAP:
    #    model = joblib.load(model_dir / f"FINAL_{model_type}_{run_id}.joblib")
    #    scaler = joblib.load(model_dir / f"FINAL_scaler_{run_id}_feature.pkl")
    #    X_scaled = scaler.transform(X_all)  # only continuous features
    #    explainer = shap.TreeExplainer(model)
    #    shap_values = explainer.shap_values(X_scaled[:50000])
    #    shap_importances = np.abs(shap_values).mean(axis=0)  # per feature

    # 8. Optional: validate against training logs
    #    Check n_samples, per-fold site IDs, PFT encoding order

    # 9. Call build_aoa_reference()
    return build_aoa_reference(
        X_train=X_all, fold_labels=fold_labels,
        shap_importances=shap_importances, feature_names=feature_names,
        output_dir=model_dir, run_id=run_id, d_bar_method=d_bar_method,
    )
```

**For the current model (`default_daily_without_coordinates`):**
The simpler approach is to **retrain with AOA integration** (after M7). Both options exposed via CLI per spec §8:
```bash
# CLI matches spec §8 for backfill.py:
python -m src.aoa.backfill \
    --models-dir models/ \
    --model-type xgb \
    --run-id default_daily_without_coordinates \
    --data-dir data/ \
    --validate-against-logs path/to/training.log  # optional

# Recommended: retrain model instead (after M7 integration)
```

**Edge cases:**
- Config n_samples doesn't match derived data → warning + continue (with validation log)
- SHAP computation on 50K samples takes ~5-10 min

**AC:** Backfill produces valid reference NPZ.

#### Tests for M3

```bash
pytest src/aoa/tests/test_backfill.py -v
```
**Gate:** ALL pass.

---

### M4: Apply Step (DI Computation)

**Goal:** Compute DI for prediction grids. Largest and most complex module.
**Spec:** §8 (apply CLI), §9 (apply section), §11 (feature alignment)

#### Step 4.1: Write apply tests (RED)

**Files:** `src/aoa/tests/test_apply.py` (create)
**Deps:** M2

| Test | Input | Expected | Type |
|------|-------|----------|------|
| `test_prepare_apply_roundtrip` | 50 train + 20 new | DI matches manual computation | happy |
| `test_training_point_as_prediction` | exact training point | DI ≈ 0 | happy |
| `test_csv_input` | ERA5-format CSV | Correct DI output | happy |
| `test_parquet_input` | parquet | Same DI as CSV | happy |
| `test_csv_parquet_equivalence` | same data both formats | Identical DI | happy |
| `test_feature_order_robustness` | shuffled columns | Same DI (reordered internally) | edge |
| `test_missing_feature_error` | column removed | ValueError with feature list | error |
| `test_nan_rows_dropped` | NaN in features | Dropped, count logged | edge |
| `test_empty_file` | no valid rows | Skip gracefully | edge |
| `test_duplicate_timestamp_col` | ERA5 duplicate `timestamp` | Handled (deduplicate) | edge |
| `test_multiple_months` | 3 monthly files | 3 monthly summaries produced | happy |
| `test_monthly_summary_stats` | known DI values for 30 timestamps | Exact median, mean, std, frac_inside_aoa | happy |
| `test_aoa_meta_json` | valid run | JSON has all spec §6 fields | happy |
| `test_per_timestamp_parquet_schema` | output | lat/lon float32, timestamp, DI float32, aoa_mask bool | happy |
| `test_monthly_summary_schema` | output | All 7 columns, correct dtypes | happy |

#### Step 4.2: Implement apply.py core functions

**Files:** `src/aoa/apply.py` (create)
**Deps:** 4.1, M1, M2

**Key functions:**

```python
def load_input_file(path: Path) -> pd.DataFrame:
    """Load CSV or parquet; deduplicate column names."""
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")
    # Handle duplicate column names (ERA5 has duplicate 'timestamp')
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    return df

def select_and_validate_features(
    df: pd.DataFrame,
    feature_names: list[str],
    reference_feature_names: list[str],
) -> pd.DataFrame:
    """Select features in reference order, validate completeness."""
    if list(feature_names) != list(reference_feature_names):
        raise ValueError(
            f"Feature mismatch: config has {len(feature_names)} features, "
            f"reference has {len(reference_feature_names)}. "
            f"Missing from config: {set(reference_feature_names) - set(feature_names)}, "
            f"Extra in config: {set(feature_names) - set(reference_feature_names)}"
        )
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        # Case-insensitive fallback (matches prediction script pattern)
        col_lower = {c.lower(): c for c in df.columns}
        still_missing = []
        for f in missing:
            if f.lower() in col_lower:
                df = df.rename(columns={col_lower[f.lower()]: f})
            else:
                still_missing.append(f)
        if still_missing:
            raise ValueError(f"Missing features in input: {still_missing}")
    return df[feature_names]

def compute_di_for_dataframe(
    df: pd.DataFrame,
    reference: dict,
    tree: cKDTree,
    feature_names: list[str],
    batch_size: int = 500_000,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Compute DI and AOA mask for a DataFrame.

    Returns: (di_values, aoa_mask, valid_mask) — di/aoa aligned to valid rows only.
             valid_mask is a boolean Series aligned to original df index.
    """
    X_df = select_and_validate_features(df, feature_names, reference["feature_names"])
    valid_mask = ~X_df.isna().any(axis=1)
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        logger.warning(f"Dropped {n_dropped} rows with NaN features")
    X = X_df[valid_mask].values
    if len(X) == 0:
        return np.array([]), np.array([]), valid_mask

    X_s = core.standardize_features(X, reference["feature_means"], reference["feature_stds"])
    X_sw = core.apply_importance_weights(X_s, reference["feature_weights"])

    # Batch KD-tree query
    di = np.empty(len(X_sw))
    for start in range(0, len(X_sw), batch_size):
        end = min(start + batch_size, len(X_sw))
        di[start:end] = core.compute_prediction_di(X_sw[start:end], tree, reference["d_bar"])

    aoa_mask = di <= reference["threshold"]
    return di, aoa_mask, valid_mask
```

**File discovery for ERA5 data:**
```python
def discover_era5_files(
    input_dir: Path, time_scale: str, years: tuple[int, ...], months: tuple[int, ...]
) -> list[Path]:
    """Find ERA5 preprocessed files matching year/month filters.

    Pattern: {input_dir}/{YYYY}_{time_scale}/prediction_{YYYY}_{MM}_{time_scale}.csv
    """
    files = []
    for year in years:
        year_dir = input_dir / f"{year}_{time_scale}"
        if not year_dir.exists():
            logger.warning(f"Year dir not found: {year_dir}")
            continue
        for month in months:
            pattern = f"prediction_{year}_{month:02d}_{time_scale}.*"
            matches = list(year_dir.glob(pattern))
            files.extend(matches)
    return sorted(files)
```

**Monthly summary computation:**
```python
def compute_monthly_summary(per_ts_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-timestamp DI to monthly per-pixel summary."""
    summary = per_ts_df.groupby(["latitude", "longitude"]).agg(
        median_DI=("DI", "median"),
        mean_DI=("DI", "mean"),
        std_DI=("DI", "std"),
        frac_inside_aoa=("aoa_mask", "mean"),
        n_timestamps=("DI", "count"),
    ).reset_index()
    return summary.astype({
        "latitude": np.float32, "longitude": np.float32,
        "median_DI": np.float32, "mean_DI": np.float32,
        "std_DI": np.float32, "frac_inside_aoa": np.float32,
        "n_timestamps": np.int32,
    })
```

#### Step 4.3: Implement apply.py main loop + CLI + aoa_meta.json

**Files:** `src/aoa/apply.py` (add main + meta)
**Deps:** 4.2
**Spec:** §8 (apply CLI), §6 (aoa_meta.json, output structure)

**Main orchestration loop** — the central processing function:
```python
def process_files(config: AOAConfig, reference: dict, model_config) -> None:
    """Main processing loop: iterate files, compute DI, write outputs."""
    feature_names = model_config.feature_names
    tree = core.build_kdtree(reference["reference_cloud_weighted"])

    # Output directories: {output_dir}/{time_scale}/per_timestamp/ and /monthly/
    ts_dir = config.output_dir / config.time_scale / "per_timestamp"
    monthly_dir = config.output_dir / config.time_scale / "monthly"
    ts_dir.mkdir(parents=True, exist_ok=True)
    monthly_dir.mkdir(parents=True, exist_ok=True)

    files = discover_era5_files(config.input_dir, config.time_scale, config.years, config.months)
    if not files:
        logger.warning("No input files found")
        return

    for file_path in files:
        try:
            logger.info(f"Processing {file_path.name}")
            df = load_input_file(file_path)
            meta_cols = ["latitude", "longitude", "timestamp"]
            meta = df[meta_cols].copy()

        di_values, aoa_mask, valid_mask = compute_di_for_dataframe(
            df, reference, tree, feature_names, config.batch_size
        )
        if len(di_values) == 0:
            logger.warning(f"No valid rows in {file_path.name}, skipping")
            continue

        # Build per-timestamp DataFrame — use valid_mask returned by compute_di
        meta_valid = meta[valid_mask].reset_index(drop=True)
        per_ts_df = pd.DataFrame({
            "latitude": meta_valid["latitude"].values.astype(np.float32),
            "longitude": meta_valid["longitude"].values.astype(np.float32),
            "timestamp": meta_valid["timestamp"].values,
            "DI": di_values.astype(np.float32),
            "aoa_mask": aoa_mask,
        })

        # Extract year/month from filename: prediction_{YYYY}_{MM}_{time_scale}.*
        stem = file_path.stem  # e.g. "prediction_2015_01_daily"
        parts = stem.split("_")
        year, month = parts[1], parts[2]

        # Save per-timestamp (if enabled)
        if config.save_per_timestamp:
            ts_path = ts_dir / f"di_{year}_{month}_{config.time_scale}.parquet"
            per_ts_df.to_parquet(ts_path, index=False, compression="gzip")

        # Save monthly summary
        summary = compute_monthly_summary(per_ts_df)
        summary_path = monthly_dir / f"di_monthly_{year}_{month}.parquet"
        summary.to_parquet(summary_path, index=False, compression="gzip")

        except Exception as e:
            logger.error(f"Failed processing {file_path.name}: {e}")
            continue  # skip corrupt/malformed files, process the rest

    # Write metadata JSON to {output_dir}/{time_scale}/aoa_meta.json
    meta_dir = config.output_dir / config.time_scale
    write_aoa_meta(config, reference, meta_dir)
```

**CLI flags** per spec §8: `--aoa-reference`, `--input-dir`, `--model-config`, `--output-dir`, `--time-scale`, `--save-per-timestamp`, `--years`, `--months`, `--batch-size`, `--n-jobs`, `--generate-slurm`.
`model_type` and `run_id` inferred from `--model-config` JSON.

**`--input-dir` value:** Must point to the ERA5 **parent** directory (e.g., `data/dataset_for_prediction/`), NOT a year subdirectory. The `discover_era5_files()` function appends `{YYYY}_{time_scale}/` internally.

**ModelConfig import:** Import from prediction script: `from src.make_prediction.predict_sap_velocity_sequential import ModelConfig`. This creates a dependency on the prediction module but only for the dataclass (no heavy imports at module level). Alternative: parse the JSON manually and extract `feature_names`, `model_type`, `run_id`, `is_windowing`, `input_width`, `shift` as a simple dict — avoids the import entirely. **Decision: use manual JSON parsing** to keep `src/aoa/` self-contained:
```python
def load_model_config(path: Path) -> dict:
    """Load model config JSON and extract fields needed by AOA."""
    with open(path) as f:
        raw = json.load(f)
    return {
        "model_type": raw["model_type"],
        "run_id": raw["run_id"],
        "feature_names": raw["feature_names"],
        "is_windowing": raw.get("data_info", {}).get("IS_WINDOWING", False),
        "input_width": raw.get("data_info", {}).get("input_width"),
        "shift": raw.get("data_info", {}).get("shift"),
    }
```

**`parse_range()` utility** — defined in apply.py:
```python
def parse_range(s: str) -> list[int]:
    """Parse '1995-2018' → [1995,...,2018] or '2015,2016' → [2015,2016]."""
    if "-" in s and "," not in s:
        start, end = s.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(x.strip()) for x in s.split(",")]
```

**`main()` function:**
```python
def main():
    args = parse_args()
    model_config = load_model_config(args.model_config)
    reference = load_aoa_reference(args.aoa_reference)
    config = AOAConfig(
        model_type=model_config["model_type"],
        run_id=model_config["run_id"],
        time_scale=args.time_scale,
        aoa_reference_path=args.aoa_reference,
        input_dir=args.input_dir,
        model_config_path=args.model_config,
        output_dir=args.output_dir,
        save_per_timestamp=args.save_per_timestamp,
        years=tuple(parse_range(args.years)),
        months=tuple(parse_range(args.months)),
        batch_size=args.batch_size,
        n_jobs=args.n_jobs,
    )
    if args.generate_slurm:
        generate_slurm_script(config, args)
    else:
        process_files(config, reference, model_config)
```

**`write_aoa_meta`** writes to `{output_dir}/{time_scale}/aoa_meta.json`:
```python
def write_aoa_meta(config, reference, output_dir):
    """Write metadata JSON. output_dir should be {base}/{time_scale}/."""
    meta = {
        "run_id": config.run_id,
        "model_type": config.model_type,
        "time_scale": config.time_scale,
        "threshold": reference["threshold"],
        "d_bar": reference["d_bar"],
        "d_bar_method": reference["d_bar_method"],
        "n_training_samples": reference["reference_cloud_weighted"].shape[0],
        "n_features": reference["reference_cloud_weighted"].shape[1],
        "feature_names": reference["feature_names"],
        "weighting_method": "mean_abs_shap",
        "iqr_multiplier": config.iqr_multiplier,
        "cv_strategy": "StratifiedGroupKFold",
        "cv_n_splits": len(np.unique(reference["fold_assignments"])),
        "random_seed": 42,
        "prediction_years": list(config.years),
        "save_per_timestamp": config.save_per_timestamp,
        "aoa_reference_path": str(config.aoa_reference_path),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(output_dir / "aoa_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
```

**SLURM generation** (`--generate-slurm` flag):
```python
def generate_slurm_script(config, args):
    """Write SLURM array job script based on spec §10 template."""
    n_months = len(config.years) * len(config.months)
    script = f"""#!/bin/bash
#SBATCH --partition=zen4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=36
#SBATCH --mem=200G --time=01:00:00
#SBATCH --array=0-{n_months - 1}
#SBATCH --job-name=aoa_di_{config.run_id}

YEARS=({' '.join(str(y) for y in config.years)})
MONTHS=({' '.join(str(m) for m in config.months)})
N_MONTHS={len(config.months)}
YEAR=${{YEARS[$((SLURM_ARRAY_TASK_ID / N_MONTHS))]}}
MONTH=${{MONTHS[$((SLURM_ARRAY_TASK_ID % N_MONTHS))]}}

python -m src.aoa.apply \\
    --aoa-reference {config.aoa_reference_path} \\
    --input-dir {config.input_dir} \\
    --model-config {config.model_config_path} \\
    --output-dir {config.output_dir} \\
    --time-scale {config.time_scale} \\
    {'--save-per-timestamp' if config.save_per_timestamp else ''} \\
    --years $YEAR --months $MONTH \\
    --n-jobs 36
"""
    script_path = Path(f"job_aoa_apply_{config.run_id}.sh")
    script_path.write_text(script)
    logger.info(f"SLURM script written to {script_path}")
```

**Windowed model support:**
When `ModelConfig.is_windowing == True`:
1. Import `create_prediction_windows_improved` from prediction script
2. Apply windowing to ERA5 data before DI computation — insert between file loading and `compute_di_for_dataframe`
3. Feature names already include lag suffixes (`ta_t-0`, `ta_t-1`) in both reference and ModelConfig

When `is_windowing == False` (the default XGB case): direct column selection, no windowing needed.

**AC:** All apply tests pass. Per-timestamp + monthly parquets in correct paths and format.

#### Tests for M4

Add these to the M4 Step 4.1 test table (in addition to tests already listed):

| Test | Input | Expected | Type |
|------|-------|----------|------|
| `test_windowed_model` | IS_WINDOWING=True, windowed features | DI computed correctly in windowed space | happy |
| `test_non_windowed_model` | IS_WINDOWING=False, flat features | Correct DI | happy |
| `test_generate_slurm_flag` | `--generate-slurm` | SLURM .sh file written with correct array size and paths | happy |
| `test_output_dir_structure` | process 2 months | `{time_scale}/per_timestamp/` and `{time_scale}/monthly/` dirs created with correct files | happy |

```bash
pytest src/aoa/tests/test_apply.py -v
```
**Gate:** ALL pass.

---

### M5: Aggregation

**Goal:** Yearly, climatological, overall summaries from monthly data.
**Spec:** §9 (aggregate), §16.6

#### Step 5.1: Write test_aggregate.py (RED)

**Files:** `src/aoa/tests/test_aggregate.py` (create)
**Deps:** M4 (output format)

| Test | Input | Expected |
|------|-------|----------|
| `test_yearly_from_12_months` | 12 monthly summaries, known medians | yearly median = median of monthly medians |
| `test_climatological_all_januaries` | 3 years × 12 months | "all Jans" = median of 3 January medians |
| `test_overall` | 36 monthly summaries | Single overall summary |
| `test_frac_inside_aoa_boundaries` | all inside, all outside, half | 1.0, 0.0, 0.5 |
| `test_n_timestamps_sum` | yearly | sum of 12 monthly n_timestamps |
| `test_empty_month` | 0 files for one month | Skip gracefully |
| `test_single_month` | 1 monthly summary | yearly/overall = same values |

#### Step 5.2: Implement aggregate.py (GREEN)

**Files:** `src/aoa/aggregate.py` (create)
**Deps:** 5.1

**Function signatures and I/O:**
```python
def _aggregate_group(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate multiple monthly summary DataFrames into one summary per pixel.
    Median of monthly medians, mean of monthly means, mean of stds, mean of fracs, sum of n."""
    combined = pd.concat(dfs)
    return combined.groupby(["latitude", "longitude"]).agg(
        median_DI=("median_DI", "median"),
        mean_DI=("mean_DI", "mean"),
        std_DI=("std_DI", "mean"),
        frac_inside_aoa=("frac_inside_aoa", "mean"),
        n_timestamps=("n_timestamps", "sum"),
    ).reset_index()

def aggregate_yearly(monthly_dir: Path, output_dir: Path, years: list[int]) -> list[Path]:
    """Read monthly summaries → per-year summary. Returns list of written paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for year in years:
        files = sorted(monthly_dir.glob(f"di_monthly_{year}_*.parquet"))
        if not files:
            continue
        dfs = [pd.read_parquet(f) for f in files]
        result = _aggregate_group(dfs)
        out = output_dir / f"di_yearly_{year}.parquet"
        result.to_parquet(out, index=False, compression="gzip")
        paths.append(out)
    return paths

def aggregate_climatological(monthly_dir: Path, output_dir: Path) -> list[Path]:
    """Read all monthly summaries → 12 climatological summaries (all Jans, all Febs, etc.)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for month in range(1, 13):
        files = sorted(monthly_dir.glob(f"di_monthly_*_{month:02d}.parquet"))
        if not files:
            continue
        dfs = [pd.read_parquet(f) for f in files]
        result = _aggregate_group(dfs)
        out = output_dir / f"di_clim_{month:02d}.parquet"
        result.to_parquet(out, index=False, compression="gzip")
        paths.append(out)
    return paths

def aggregate_overall(monthly_dir: Path, output_path: Path) -> Path:
    """Read ALL monthly summaries → single overall summary."""
    files = sorted(monthly_dir.glob("di_monthly_*.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    result = _aggregate_group(dfs)
    result.to_parquet(output_path, index=False, compression="gzip")
    return output_path

def main():
    args = parse_args()  # --input-dir, --output-dir, --time-scale
    base = Path(args.input_dir) / args.time_scale
    monthly_dir = base / "monthly"
    # Infer years from monthly files
    years = sorted({int(f.stem.split("_")[2]) for f in monthly_dir.glob("di_monthly_*.parquet")})
    aggregate_yearly(monthly_dir, base / "yearly", years)
    aggregate_climatological(monthly_dir, base / "climatological")
    aggregate_overall(monthly_dir, base / "di_overall.parquet")
```

Aggregation method: **median of monthly medians** per pixel (approximate — spec §9 note). `std_DI` = mean of monthly stds (approximation). `frac_inside_aoa` = mean of monthly fractions. `n_timestamps` = sum.

CLI: `--input-dir`, `--output-dir`, `--time-scale`. Time_scale path appended internally by reading `{input_dir}/{time_scale}/monthly/` and writing to `{input_dir}/{time_scale}/yearly/`, `climatological/`, `di_overall.parquet`.

**AC:** All aggregate tests pass. Correct file counts (30 yearly, 12 clim, 1 overall).

#### Tests for M5

```bash
pytest src/aoa/tests/test_aggregate.py -v
```
**Gate:** ALL pass.

---

### M6: Visualization

**Goal:** GeoTIFF/NetCDF maps from DI parquets.
**Spec:** §9 (visualize), §16.7

#### Step 6.1: Write test_visualize.py (RED)

**Files:** `src/aoa/tests/test_visualize.py` (create)
**Deps:** M5

| Test | Input | Expected |
|------|-------|----------|
| `test_geotiff_crs` | DI parquet → GeoTIFF | CRS = EPSG:4326 |
| `test_geotiff_resolution` | parquet with known lat/lon | Pixel = 0.1° × 0.1° |
| `test_geotiff_bands` | output | 2 bands |
| `test_geotiff_dtype` | output | Both float32 |
| `test_aoa_mask_values` | output | Band 2 only 0.0, 1.0, NaN |
| `test_di_non_negative` | output | Band 1 ≥ 0 or NaN |
| `test_geotiff_bounds` | known lat/lon extent | Covers expected bounds |
| `test_nodata_handling` | parquet with gaps | Ocean/missing pixels → NaN |
| `test_roundtrip` | write → read | Values match source parquet |
| `test_netcdf_validity` | NetCDF output | Valid NetCDF4, correct dims/vars/attrs (if map_format=netcdf) |
| `test_all_levels_produce_output` | run with `--levels all` | All map files exist |

Mark all with `@pytest.mark.requires_rasterio`.

#### Step 6.2: Implement visualize.py (GREEN)

**Files:** `src/aoa/visualize.py` (create)
**Deps:** 6.1

Key function — vectorized grid construction (NOT iterrows):
```python
def parquet_to_geotiff(df, output_path, di_col="median_DI", threshold=None, resolution=0.1):
    import rasterio
    from rasterio.transform import from_bounds

    lats = np.sort(df["latitude"].unique())[::-1]  # N→S
    lons = np.sort(df["longitude"].unique())
    nrows, ncols = len(lats), len(lons)

    lat_to_row = {lat: i for i, lat in enumerate(lats)}
    lon_to_col = {lon: j for j, lon in enumerate(lons)}

    di_grid = np.full((nrows, ncols), np.nan, dtype=np.float32)
    mask_grid = np.full((nrows, ncols), np.nan, dtype=np.float32)

    rows = df["latitude"].map(lat_to_row).values
    cols = df["longitude"].map(lon_to_col).values
    di_grid[rows, cols] = df[di_col].values.astype(np.float32)

    if "aoa_mask" in df.columns:
        mask_grid[rows, cols] = df["aoa_mask"].astype(np.float32)
    elif "frac_inside_aoa" in df.columns and threshold is not None:
        mask_grid[rows, cols] = (df["frac_inside_aoa"] >= 0.5).astype(np.float32)

    transform = from_bounds(
        lons.min() - resolution/2, lats.min() - resolution/2,
        lons.max() + resolution/2, lats.max() + resolution/2,
        ncols, nrows,
    )
    with rasterio.open(
        output_path, "w", driver="GTiff",
        height=nrows, width=ncols, count=2,
        dtype="float32", crs="EPSG:4326",
        transform=transform, nodata=np.nan,
    ) as dst:
        dst.write(di_grid, 1)
        dst.write(mask_grid, 2)
```

**Level-to-directory dispatch:**
```python
LEVEL_CONFIG = {
    # level_name: (input_subdir, input_pattern, output_subdir, di_column)
    "per_timestamp": ("per_timestamp", "di_*_*.parquet", "per_timestamp", "DI"),
    "monthly":       ("monthly", "di_monthly_*.parquet", "monthly", "median_DI"),
    "climatological":("climatological", "di_clim_*.parquet", "climatological", "median_DI"),
    "yearly":        ("yearly", "di_yearly_*.parquet", "yearly", "median_DI"),
    "overall":       (None, None, None, "median_DI"),  # single file: di_overall.parquet
}

def main():
    args = parse_args()
    base = Path(args.input_dir) / args.time_scale
    maps_base = Path(args.output_dir) / args.time_scale / "maps"
    levels = args.levels.split(",")  # e.g. ["per_timestamp", "monthly", ...]
    threshold = None  # loaded from aoa_meta.json for summary mask

    # Load threshold from meta for summary-level AOA mask derivation
    meta_path = base / "aoa_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            threshold = json.load(f).get("threshold")

    for level in levels:
        if level == "overall":
            src = base / "di_overall.parquet"
            if src.exists():
                df = pd.read_parquet(src)
                parquet_to_geotiff(df, maps_base / "di_overall.tif",
                                   di_col="median_DI", threshold=threshold)
        else:
            cfg = LEVEL_CONFIG[level]
            src_dir = base / cfg[0]
            out_dir = maps_base / cfg[2]
            out_dir.mkdir(parents=True, exist_ok=True)
            for pq in sorted(src_dir.glob(cfg[1])):
                df = pd.read_parquet(pq)
                tif_name = pq.stem + ".tif"
                parquet_to_geotiff(df, out_dir / tif_name,
                                   di_col=cfg[3], threshold=threshold)
```

**Design decision: AOA mask for summary-level GeoTIFFs:**
Summary parquets have `frac_inside_aoa` (continuous 0-1) but not `aoa_mask` (binary). For GeoTIFF band 2, we threshold `frac_inside_aoa >= 0.5` to produce binary mask (1.0=majority inside, 0.0=majority outside). Per-timestamp parquets have the actual `aoa_mask` column and use it directly.

CLI: `--input-dir`, `--output-dir`, `--time-scale`, `--map-format`, `--levels per_timestamp,monthly,climatological,yearly,overall`

**AC:** All visualize tests pass. GeoTIFFs valid.

#### Tests for M6

```bash
pytest src/aoa/tests/test_visualize.py -v
```
**Gate:** ALL pass.

---

### M7: Training Script Integration

**Goal:** Training script saves AOA reference alongside model.
**Spec:** §5 (integration)

#### Step 7.1: Add AOA integration to training script

**Files:** `src/hyperparameter_optimization/test_hyperparameter_tuning_ML_spatial_stratified.py` (modify)
**Deps:** M2
**Where:** AFTER SHAP computation + CSV save (after line ~4717), before the training function returns.

**Code to add (~20 lines):**
```python
# === AOA Reference Generation ===
try:
    from src.aoa.prepare import build_aoa_reference

    # 1. Derive fold labels from CV splitter
    fold_labels = np.full(len(X_all_records), -1, dtype=int)
    for fold_idx, (_, test_idx) in enumerate(
        outer_cv.split(X_all_records, y_all_stratified, groups_all_records)
    ):
        fold_labels[test_idx] = fold_idx
    assert (fold_labels >= 0).all(), "Some samples not assigned to any fold"

    # 2. Load SHAP importance (already saved above)
    shap_csv_path = plot_dir / "shap_feature_importance.csv"
    shap_df = pd.read_csv(shap_csv_path)
    shap_importances = shap_df.set_index("feature")["importance"].reindex(
        final_feature_names
    ).values
    assert not np.any(np.isnan(shap_importances)), \
        f"SHAP features don't match final_feature_names: {final_feature_names}"

    # 3. Build AOA reference
    aoa_ref_path = build_aoa_reference(
        X_train=X_all_records,
        fold_labels=fold_labels,
        shap_importances=shap_importances,
        feature_names=final_feature_names,
        output_dir=model_dir,
        run_id=run_id,
        d_bar_method="full",
    )
    logging.info(f"AOA reference saved to {aoa_ref_path}")

    # 4. Save backups (spec §5 — fold_labels, spatial_groups, site_ids, pfts)
    np.savez(
        model_dir / f"FINAL_cv_folds_{run_id}.npz",
        fold_labels=fold_labels,
        spatial_groups=groups_all_records,
        site_ids=site_ids_all_records,  # verify var name from training script
        pfts=pfts_all_records,
    )
    pd.DataFrame(X_all_records, columns=final_feature_names).to_parquet(
        model_dir / f"FINAL_X_train_{run_id}.parquet", index=False,
    )
    logging.info("AOA backups (cv_folds, X_train) saved")
except Exception as e:
    logging.error(f"AOA reference generation failed: {e}")
    logging.info("Model training completed successfully — AOA step failed separately")
```

**Important ordering:** This code goes AFTER the SHAP CSV is written (line 4717) and BEFORE any cleanup/return. The `try/except` ensures training doesn't fail if AOA has a bug.

#### Step 7.2: Verify by retraining

After M7 is committed, retrain with same parameters → verify:
- `FINAL_aoa_reference_{run_id}.npz` exists with correct keys
- `FINAL_X_train_{run_id}.parquet` has n_samples rows
- `FINAL_cv_folds_{run_id}.npz` has fold_labels, spatial_groups, pfts

**AC:** Training script produces all AOA artifacts.

#### Tests for M7

Add to `src/aoa/tests/test_training_integration.py` (create):

| Test | Input | Expected |
|------|-------|----------|
| `test_training_integration_produces_npz` | Run training on tiny subset (10 sites, 100 records) | `FINAL_aoa_reference_{run_id}.npz` exists with all 10 keys |
| `test_training_integration_x_train_backup` | Same | `FINAL_X_train_{run_id}.parquet` has correct shape |
| `test_training_integration_cv_folds_backup` | Same | `FINAL_cv_folds_{run_id}.npz` has fold_labels, spatial_groups, site_ids, pfts |
| `test_training_vs_backfill_deterministic` | Both produce reference for same model | `np.allclose` for all numeric arrays |

Mark with `@pytest.mark.slow` (requires loading model + data).

Additionally verify by HPC retraining smoke test.

**Gate:** Automated tests pass + NPZ produced by training on HPC.

---

### M8: R CAST Reference Validation

**Goal:** Gold-standard comparison against official R implementation.
**Spec:** §16.3

#### Step 8.1: Create synthetic data + R script

**Files:** `src/aoa/tests/fixtures/generate_synthetic.py` (modify), `src/aoa/tests/fixtures/run_cast_aoa.R` (create)
**Deps:** M1

Generate: 20 training, 5 continuous + 2 binary features, 3 folds, 10 new points.

**CAST weight limitation:** `CAST::aoa()` does NOT accept pre-computed importance weights — it derives them internally from the trained model. To ensure comparability:
1. Train identical Random Forest models in both R (caret) and Python (sklearn) on the synthetic data with same hyperparameters and seed
2. Extract R's permutation importance from the trained model
3. Use those SAME importance values as Python SHAP weights
4. Compare DI values (which should match since the standardization, weighting, and distance computations are deterministic given the same weights)

Alternatively, use CAST's lower-level functions if available, or extract CAST's internal importance values and feed them to Python.

R script saves: DI values, threshold, d_bar, AOA mask, training DI, AND the variable importance weights used internally by CAST (so Python can replicate them exactly).

**R not available on HPC:** Generate fixtures on local machine with R, commit `cast_outputs/` to repo. Tests load pre-computed fixtures — no R needed at test time.

**Fallback:** If R is not available anywhere, create hand-verified reference for a minimal 3-point example with manually computed DI values.

#### Step 8.2: Write test_cast_reference.py

**Files:** `src/aoa/tests/test_cast_reference.py` (create)
**Deps:** 8.1

| Test | Tolerance |
|------|-----------|
| d_bar matches R | < 1e-10 relative |
| threshold matches R | < 1e-10 relative |
| All prediction DI match R | < 1e-10 per value |
| All training DI match R | < 1e-10 per value |
| AOA mask matches R | Exact boolean |

Variant tests: continuous only, with dummies, 3/5/10 folds, single feature, equal weights.

All marked `@pytest.mark.slow`.

**Known deviation:** CAST uses model's permutation importance; we use mean|SHAP|. Since DI is scale-invariant (d/d_bar), we pass identical pre-computed weights to both Python and R to ensure comparability.

**AC:** Python DI/threshold/d_bar match R CAST within 1e-10.

#### Tests for M8

```bash
pytest src/aoa/tests/test_cast_reference.py -v
```
**Gate:** ALL pass.

---

### M9: HPC Validation

**Goal:** Full-scale execution on Palma2 HPC.
**Spec:** §10 (HPC), §16.8 (performance)

#### Step 9.1: Create SLURM scripts

5 scripts:
1. `job_aoa_backfill.sh` — single job, backfill for existing model
2. `job_aoa_apply_daily.sh` — array=0-47 (4 years × 12 months for 2015-2018), zen4, 36 cpus, 200G, time=01:00:00
3. `job_aoa_apply_hourly.sh` — same array, zen4, 192 cpus, 745G, time=06:00:00 (per spec §10 hourly estimates)
4. `job_aoa_aggregate.sh` — single job, depends on apply
5. `job_aoa_visualize.sh` — single job, depends on aggregate

#### Step 9.2: Smoke test (1 month)

Run apply on January 2015 only. Verify:
- Per-timestamp parquet produced: `di_2015_01_daily.parquet`
- Monthly summary: `di_monthly_2015_01.parquet`
- DI values ≥ 0, aoa_mask is boolean
- File size reasonable

#### Step 9.3: Full 4-year run (2015-2018)

Submit array job → aggregate → visualize. Verify output structure matches spec §6.

#### Step 9.4: Write test_performance.py

**Files:** `src/aoa/tests/test_performance.py` (create)
**Spec:** §16.8

| Test | Criterion |
|------|-----------|
| d_bar on 50K points | < 10 min |
| KD-tree build on 50K | < 5 sec |
| 1M queries on 50K tree | < 30 sec |
| apply.py 1 month daily | < 30 min on zen4 |
| Memory (daily) | < 50 GB peak |
| Memory (hourly) | < 200 GB peak |
| Parquet size (daily, 1 month) | < 5 GB |
| Monthly summary size | < 50 MB per file |
| GeoTIFF output size | < 100 MB per file |
| d_bar_sample(10K) on 1M points | < 5 sec |

**AC:** All performance criteria met.

#### Step 9.5: Coverage verification gate

```bash
pytest src/aoa/tests/ -m "not performance" --cov=src/aoa --cov-report=term-missing -v
```

**Gate:** Coverage ≥ 80% for `src/aoa/`. If below 80%, add targeted tests for uncovered functions/branches before proceeding.

---

## Deviations from Spec

| # | Deviation | Justification |
|---|-----------|---------------|
| 1 | Run ID is `default_daily_without_coordinates`, not `default_daily_nocoors_swcnor` | Verified on HPC — spec used placeholder name |
| 2 | No saved SHAP CSV or aoa_X_train.npy for current model | Model predates SHAP CSV saving. Backfill must re-derive or retrain |
| 3 | ERA5 file pattern is `prediction_{YYYY}_{MM}_daily.csv` in `{YYYY}_daily/` subdirs | Discovered on HPC — spec didn't specify exact pattern |
| 4 | ERA5 CSVs have duplicate `timestamp` column | Apply.py must deduplicate (added to Step 4.2) |
| 5 | R not available on HPC | CAST fixtures generated locally, committed to repo |
| 6 | AOAConfig uses `tuple` instead of `list` for years/months | Immutable collection for frozen dataclass |
| 7 | Added `d_bar_method` field to reference NPZ | Not in original NPZ spec — needed for aoa_meta.json |
| 8 | Backfill has two modes (saved-arrays vs merged-data) | Current model lacks saved arrays; future models will have them |
| 9 | Summary-level GeoTIFF AOA mask uses `frac_inside_aoa >= 0.5` threshold | Spec defines binary mask but doesn't address derivation from summary stats. Majority-vote threshold is reasonable |
| 10 | AC-4 says "10 fields" (added `d_bar_method` to spec's original 9) | Needed for traceability in aoa_meta.json |
| 11 | Spec's `test_integration.py` split into `test_prepare.py`, `test_backfill.py`, `test_apply.py`, `test_training_integration.py` | Per coding-style rule: many small files > few large files. Each file stays <400 lines. Same total test coverage. |
| 12 | `ddof=1` (sample std) instead of `ddof=0` (population std) in prepare.py | R CAST uses `sd()` which is `ddof=1`. Required to match AC-14 (1e-10 tolerance). sklearn's StandardScaler uses `ddof=0` but AOA does NOT reuse the model scaler — it has its own z-score standardization. |
| 13 | apply.py uses manual JSON parsing instead of importing ModelConfig from prediction script | Keeps `src/aoa/` self-contained with no dependency on the 2200-line prediction script. `load_model_config()` extracts only the 6 fields AOA needs. |

---

## Open Questions (resolve during implementation)

1. **ERA5 hourly file pattern:** Verify `prediction_{YYYY}_{MM}_hourly.csv` in `{YYYY}_hourly/`. Currently only daily files confirmed.
2. **SHAP computation order in training script:** Verify SHAP CSV is written BEFORE the AOA integration code insertion point. Need exact line-by-line review.
3. **Training data column order:** Verify `X_all_records` column order matches `final_feature_names` exactly (not just same set, but same order).
4. **Windowed ERA5 data:** When IS_WINDOWING=True, verify `create_prediction_windows_improved()` can be imported standalone from prediction script.
5. **Population vs sample std:** Resolved — AOA uses `ddof=1` to match R CAST. This deliberately differs from sklearn's StandardScaler (`ddof=0`) because AOA has its own independent z-score standardization.
6. **rasterio in requirements.txt:** Verify rasterio is listed in `requirements.txt` on the branch. If not, add it. (Confirmed installed in venv — 1.4.3 — but may need to be in requirements.txt for reproducibility.)
7. **`prcip/PET` column spelling:** Verify ERA5 CSV column headers use exact spelling `prcip/PET` (with the "typo"). The ERA5 processing script's VARIABLE_RENAME maps `precip_pet_ratio` → `prcip/PET`, so this should match. Confirm during M4 implementation.
8. **`precip` feature vs `prcip/PET`:** The model uses `prcip/PET` (a derived ratio), NOT raw `precip`. ERA5 CSVs should have `prcip/PET` pre-computed. Verify this column exists in the actual CSV during M4 smoke test.
9. **pytest-cov dependency:** Needed for M9 step 9.5 coverage gate. Verify installed in HPC venv or add to requirements.
