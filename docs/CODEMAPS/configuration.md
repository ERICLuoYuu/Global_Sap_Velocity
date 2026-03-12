# Configuration & Paths Codemap

**Last Updated:** 2026-03-12
**Files:** `path_config.py`, `src/make_prediction/config.py`, `.ruff.toml`

---

## Central Path Configuration

**File:** `path_config.py`

**Purpose:** Single source of truth for all input/output directory paths

**Class:** `PathConfig(scale: str = "sapwood")`

**Scales:** `sapwood`, `plant`, `site`

**Key Attributes:**

| Attribute | Example Path | Purpose |
|-----------|--------------|---------|
| `raw_csv_dir` | `data/raw/0.1.5/..../csv/{scale}/` | SAPFLUXNET input CSV files |
| `sap_outliers_removed_dir` | `outputs/processed_data/{scale}/sap/outliers_removed/` | Outlier-removed sapflow data |
| `env_outliers_removed_dir` | `outputs/processed_data/{scale}/env/outliers_removed/` | Outlier-removed environment data |
| `merged_data_dir` | `outputs/processed_data/{scale}/merged/` | Merged sapflow + features (daily) |
| `figures_root` | `outputs/figures/` | All generated plots |
| `models_root` | `outputs/models/` | Trained model artifacts |
| `feature_importance_dir` | `outputs/plots/hyperparameter_optimization/{model_type}/{run_id}/` | SHAP analysis plots |

**Usage:**

```python
from path_config import PathConfig

paths = PathConfig(scale="sapwood")
print(paths.raw_csv_dir)
print(paths.merged_data_dir)
print(paths.models_root)
```

---

## Model Configuration

**File:** `src/make_prediction/config.py`

**Purpose:** Hyperparameters, boundaries, and batch settings for global prediction

**Class:** `ModelConfig`

**Key Parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lat_min` | -60.0 | Southern boundary for prediction |
| `lat_max` | 78.0 | Northern boundary for prediction |
| `lon_min` | -180.0 | Western boundary for prediction |
| `lon_max` | 180.0 | Eastern boundary for prediction |
| `resolution` | 0.1 | Grid resolution (degrees) |
| `INPUT_WIDTH` | 2 | Lookback window (days) |
| `LABEL_WIDTH` | 1 | Prediction window (days) |
| `SHIFT` | 1 | Step size between windows |
| `dask_chunk_memory` | "2GB" | Per-chunk memory limit for parallelization |
| `batch_size_spatial` | 1000 | Grid cells per prediction batch |
| `batch_size_temporal` | 32 | Time steps per batch |
| `model_path` | `outputs/models/xgb/{run_id}/FINAL_xgb_{run_id}.joblib` | Trained model artifact location |
| `scaler_path` | `outputs/models/xgb/{run_id}/FINAL_scaler_{run_id}_feature.pkl` | Feature scaler artifact |

**Usage:**

```python
from src.make_prediction.config import ModelConfig

config = ModelConfig()
print(f"Spatial extent: {config.lat_min}°–{config.lat_max}°, {config.lon_min}°–{config.lon_max}°")
print(f"Temporal window: INPUT {config.INPUT_WIDTH}d, LABEL {config.LABEL_WIDTH}d, SHIFT {config.SHIFT}d")
```

---

## Ruff Linting Configuration

**File:** `.ruff.toml`

**Purpose:** Code style and quality standards for Python 3.10

**Target Version:** Python 3.10.11

**Line Length:** 120 characters

**Enabled Rules:**

| Rule Set | Purpose |
|----------|---------|
| E, W | pycodestyle errors & warnings |
| F | pyflakes (unused imports, undefined names) |
| I | isort (import ordering) |
| N | pep8-naming conventions |
| UP | pyupgrade (modernize syntax) |
| B | flake8-bugbear (common bugs) |
| SIM | flake8-simplify (code simplification) |

**Disabled Rules:**

| Rule | Reason |
|------|--------|
| E501 | Line too long (handled by formatter) |
| N803, N806 | Allow scientific variable names (X, Y) |
| B008 | Allow function calls in defaults (edge case) |
| SIM108 | Allow ternary in some scientific contexts |

**Per-File Exceptions:**

```toml
[lint.per-file-ignores]
"**/tests/**/*.py" = ["F401", "S101"]           # Allow unused imports, assert in tests
"Sapflow-internal/*.py" = ["B007"]               # Allow broad exception handling in ETL
```

**Auto-Format Style:**

- **Quote style:** Double quotes
- **Indent:** 4 spaces
- **Magic trailing comma:** Preserve (don't remove)

**Usage:**

```bash
# Check code style
ruff check src/
ruff check Sapflow-internal/

# Auto-format
ruff format src/
```

---

## Environment Variables (Optional)

**Recommended .env file** (not tracked in git):

```
# Earth Engine
GOOGLE_CLOUD_PROJECT=ee-yuluo-2
EE_PROJECT=ee-yuluo-2

# Logging
LOG_LEVEL=INFO

# Model paths (override defaults)
MODEL_RUN_ID=default_daily_nocoors_swcnor
```

---

## Directory Structure Diagram

```
global-sap-velocity/
├── path_config.py                       ← Central path registry
├── .ruff.toml                           ← Linting config
├── .claude/                             ← Claude Code plugin files
├── Sapflow-internal/                    ← Stage 1 ETL (20 sites)
├── src/
│   ├── Extractors/                      ← Stage 2 (feature extraction)
│   ├── Analyzers/                       ← Stage 3 (QC, exploration)
│   ├── cross_validation/                ← CV strategies
│   ├── hyperparameter_optimization/     ← Stage 5 (training)
│   └── make_prediction/                 ← Stage 6 (global prediction)
│       └── config.py                    ← Model hyperparameters
├── notebooks/                           ← Stage 4 (merging, notebooks)
├── data/                                ← Input data (gitignored)
│   └── raw/
│       ├── 0.1.5/.../csv/sapwood/      ← SAPFLUXNET (from download)
│       └── grided/                      ← Pre-downloaded rasters
├── Sapflow_SAPFLUXNET_format_unitcon/   ← Stage 1 output
├── outputs/                             ← All pipeline outputs
│   ├── processed_data/sapwood/
│   │   ├── sap/outliers_removed/       ← Stage 3 output
│   │   ├── env/outliers_removed/       ← Stage 3 output
│   │   └── merged/                     ← Stage 4 output
│   ├── models/xgb/{run_id}/            ← Stage 5 output
│   ├── figures/                        ← Stage 3, 6 plots
│   ├── predictions/                    ← Stage 6 output
│   └── plots/                          ← All analysis plots
└── docs/
    ├── CODEMAPS/                        ← This directory
    ├── technical_documentation.md
    ├── data_flow.md
    └── sapflow_methods_review.md
```

---

## Related Codemaps

- **[INDEX.md](INDEX.md)** — Lists all stages and modules
- **[etl-stage1.md](etl-stage1.md)** — Uses PathConfig for Stage 1 outputs
- **[extractors-stage2.md](extractors-stage2.md)** — Uses PathConfig for Stage 2 inputs/outputs
- **[analyzers-stage3.md](analyzers-stage3.md)** — Uses PathConfig for QC data paths
- **[prediction-stage6.md](prediction-stage6.md)** — Uses ModelConfig for global prediction

---

## Notes

1. **PathConfig is immutable:** Paths derived at instantiation; use for reads, writes, and consistency
2. **Scale parameter:** All stages operate on one scale at a time; re-run pipeline for other scales if needed
3. **No hardcoded paths:** All I/O should reference PathConfig or ModelConfig; never hardcode `/tmp/`, `/home/`, etc.
4. **Ruff version:** Ensure ruff >= 0.1.0; check with `ruff --version`
5. **GEE authentication:** Requires `earthengine authenticate` and `earthengine set_project ee-yuluo-2` before Stage 2
