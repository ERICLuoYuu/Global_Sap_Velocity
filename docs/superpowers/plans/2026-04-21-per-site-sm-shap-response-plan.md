# Per-Site SM→Sap-Flow XGBoost+SHAP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-file CLI script `src/Analyzers/per_site_sm_shap.py` that, per site, tunes an XGBoost regressor (`sap_velocity ~ vpd + ta + ws + sw_in + precip_sum + sm`), refits on full site data, computes SHAP values + SHAP interaction values, and writes a two-panel SM-dependence PNG plus per-site artifacts and a global `summary.csv`. Runs in two passes for `--sm-variant {raw, zscore}`.

**Architecture:** Single Python script with clearly-sectioned functions (mirrors the style of `src/Analyzers/explore_relationship_observations.py`). Per-site work is a pure function `process_one_site(site_code, cfg) → SiteResult`. The script dispatches sites via `joblib.Parallel` and aggregates results into `summary.csv`. Tests in `src/Analyzers/tests/test_per_site_sm_shap.py` use synthetic DataFrames for unit tests and one real site (`ARG_MAZ`) for an end-to-end slow test.

**Tech Stack:** Python 3.10.11, `xgboost==2.1.3`, `scikit-learn==1.5.2`, `shap==0.49.1`, `joblib`, `statsmodels` (for LOWESS), `matplotlib`, `pandas`, `pyarrow` (parquet), `pytest`.

**Reference spec:** `docs/superpowers/specs/2026-04-20-per-site-sm-shap-response-design.md`

---

## File Structure

| Path | Role | Status |
|---|---|---|
| `src/Analyzers/per_site_sm_shap.py` | main script + CLI + all functions | CREATE |
| `src/Analyzers/tests/__init__.py` | package marker | CREATE (if missing) |
| `src/Analyzers/tests/test_per_site_sm_shap.py` | unit + integration tests | CREATE |
| `src/Analyzers/tests/fixtures/fake_site_daily.csv` | synthetic daily CSV for unit tests | CREATE |
| `.claude/plan/job_per_site_sm_shap.sh` | SLURM batch job | CREATE |
| `path_config.py` | | UNCHANGED |
| `src/Analyzers/explore_relationship_observations.py` | | UNCHANGED |

All other files untouched.

---

## Task 1: Scaffolding + CLI skeleton

**Files:**
- Create: `src/Analyzers/per_site_sm_shap.py`
- Create: `src/Analyzers/tests/__init__.py`
- Create: `src/Analyzers/tests/test_per_site_sm_shap.py`

- [ ] **Step 1: Create `src/Analyzers/tests/__init__.py` (empty)**

```python
```

- [ ] **Step 2: Write the failing test** — `src/Analyzers/tests/test_per_site_sm_shap.py`

```python
"""Tests for per_site_sm_shap."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "src" / "Analyzers" / "per_site_sm_shap.py"


def test_cli_help_runs_without_error() -> None:
    """--help must exit 0 and print the --sm-variant flag."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--sm-variant" in result.stdout
    assert "--n-jobs" in result.stdout
    assert "--min-rows" in result.stdout
```

- [ ] **Step 3: Run test — verify it fails**

```bash
cd /scratch/tmp/yluo2/gsv && pytest src/Analyzers/tests/test_per_site_sm_shap.py::test_cli_help_runs_without_error -v
```
Expected: FAIL (script doesn't exist yet).

- [ ] **Step 4: Create scaffold** — `src/Analyzers/per_site_sm_shap.py`

```python
"""Per-Site Soil Moisture → Sap Flow XGBoost + SHAP analysis.

Fits one XGBoost regressor per site with per-site hyperparameter search, then
computes SHAP values and SHAP interaction values. Produces a two-panel
dependence plot per site plus aggregated summary artifacts.

See docs/superpowers/specs/2026-04-20-per-site-sm-shap-response-design.md
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("per_site_sm_shap")


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Per-site SM → sap_velocity XGBoost + SHAP analysis."
    )
    parser.add_argument(
        "--sm-variant",
        choices=["raw", "zscore"],
        required=True,
        help="Which ERA5-Land layer-1 SM column to use.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of parallel site workers.",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=100,
        help="Skip sites with fewer rows after NaN + non-physical filtering.",
    )
    parser.add_argument(
        "--sites",
        nargs="+",
        default=None,
        help="Optional list of site codes to process (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "analysis" / "per_site_sm_shap",
        help="Root output directory.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for KFold and XGBoost.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _args = build_parser().parse_args(argv)
    logger.info("Scaffold only — implementation arrives in later tasks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run test — verify it passes**

```bash
pytest src/Analyzers/tests/test_per_site_sm_shap.py::test_cli_help_runs_without_error -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/Analyzers/per_site_sm_shap.py src/Analyzers/tests/__init__.py src/Analyzers/tests/test_per_site_sm_shap.py
git commit -m "feat(per_site_sm_shap): scaffold script + CLI skeleton"
```

---

## Task 2: Constants + type aliases

**Files:**
- Modify: `src/Analyzers/per_site_sm_shap.py`

- [ ] **Step 1: Append constants block after the logger block**

```python
# ── Constants ────────────────────────────────────────────────────────────────

FEATURE_COLS_BASE: tuple[str, ...] = ("vpd", "ta", "ws", "sw_in", "precip_sum")
TARGET_COL: str = "sap_velocity"
SM_COL_NAME: str = "sm"                                   # generic internal name
FEATURE_COLS: tuple[str, ...] = FEATURE_COLS_BASE + (SM_COL_NAME,)  # final 6 features
SM_IDX: int = FEATURE_COLS.index(SM_COL_NAME)

SM_VARIANT_TO_COL: dict[str, str] = {
    "raw":    "volumetric_soil_water_layer_1_raw",
    "zscore": "volumetric_soil_water_layer_1_zscore",
}

DATA_DIR_REL = Path("outputs/processed_data/sapwood/merged/daytime_only/growing_season/daily")
SITE_META_REL = Path("outputs/processed_data/sapwood/merged/site_biome_mapping.csv")

PARAM_DIST: dict[str, list] = {
    "max_depth":        [3, 4, 5],
    "min_child_weight": [1, 3, 5, 10],
    "n_estimators":     [200, 400, 600],
    "subsample":        [0.8, 1.0],
    "gamma":            [0.0, 0.1],
}

FIXED_XGB_PARAMS: dict[str, object] = {
    "learning_rate":    0.05,
    "colsample_bytree": 1.0,
    "reg_alpha":        0.0,
    "tree_method":      "hist",
    "n_jobs":           1,           # outer parallelism is across sites
    "objective":        "reg:squarederror",
}

N_HP_TRIALS: int = 30
CV_FOLDS: int = 5
```

- [ ] **Step 2: Smoke-check by running CLI help**

```bash
python src/Analyzers/per_site_sm_shap.py --help
```
Expected: still exits 0 with the same help text.

- [ ] **Step 3: Commit**

```bash
git add src/Analyzers/per_site_sm_shap.py
git commit -m "feat(per_site_sm_shap): add constants + HP search space"
```

---

## Task 3: `load_site_data()` — TDD

**Files:**
- Create: `src/Analyzers/tests/fixtures/fake_site_daily.csv`
- Modify: `src/Analyzers/tests/test_per_site_sm_shap.py`
- Modify: `src/Analyzers/per_site_sm_shap.py`

- [ ] **Step 1: Create fixture CSV** — `src/Analyzers/tests/fixtures/fake_site_daily.csv`

Content (headers only shown; paste as-is, it's a valid minimal 12-row CSV):

```csv
TIMESTAMP,sap_velocity,vpd,ta,ws,sw_in,precip_sum,volumetric_soil_water_layer_1_raw,volumetric_soil_water_layer_1_zscore
2020-06-01,1.2,0.8,18.2,1.5,200,0.0,0.28,-0.4
2020-06-02,1.5,1.1,19.5,1.6,220,0.2,0.27,-0.5
2020-06-03,0.0,1.3,20.0,1.2,230,0.0,0.26,-0.6
2020-06-04,-0.1,1.4,20.2,1.1,240,0.1,0.26,-0.6
2020-06-05,1.8,1.6,21.1,1.8,250,0.0,0.25,-0.7
2020-06-06,,1.8,22.0,1.9,260,0.0,0.24,-0.8
2020-06-07,1.7,,21.5,1.7,255,0.0,0.25,-0.7
2020-06-08,1.9,2.0,22.8,1.9,270,0.0,0.23,-0.9
2020-06-09,2.0,2.1,23.0,2.0,275,0.0,0.22,-1.0
2020-06-10,2.1,2.2,23.4,2.1,280,0.0,0.21,-1.1
2020-06-11,2.2,2.3,23.8,2.2,285,0.0,0.21,-1.1
2020-06-12,2.3,2.4,24.1,2.3,290,0.0,0.20,-1.2
```

Row 3 has `sap_velocity=0`, row 4 has `sap_velocity=-0.1` (both should be dropped). Row 6 has NaN `sap_velocity`, row 7 has NaN `vpd` (both should be dropped). Net 8 rows remain.

- [ ] **Step 2: Write failing tests** — append to `test_per_site_sm_shap.py`

```python
import pandas as pd

from src.Analyzers.per_site_sm_shap import load_site_data, SKIP_STATUSES

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_load_site_data_drops_nans_and_nonphysical(tmp_path: Path) -> None:
    """Rows with NaN in any selected col or sap_velocity<=0 are removed."""
    # copy fixture to a temp directory simulating the canonical layout
    site_csv = FIXTURE_DIR / "fake_site_daily.csv"
    assert site_csv.exists(), "Fixture missing"

    # load with raw variant
    df, status = load_site_data(
        site_csv=site_csv,
        sm_variant="raw",
        min_rows=1,
    )
    assert status == "OK"
    assert len(df) == 8  # 12 − 4 dropped rows
    assert df[["sap_velocity", "vpd"]].notna().all().all()
    assert (df["sap_velocity"] > 0).all()
    assert "sm" in df.columns
    # raw values preserved:
    assert df["sm"].between(0.20, 0.30).all()


def test_load_site_data_selects_zscore_variant() -> None:
    df, status = load_site_data(
        site_csv=FIXTURE_DIR / "fake_site_daily.csv",
        sm_variant="zscore",
        min_rows=1,
    )
    assert status == "OK"
    assert df["sm"].between(-1.5, 0.0).all()  # zscore range


def test_load_site_data_skips_when_too_few_rows() -> None:
    df, status = load_site_data(
        site_csv=FIXTURE_DIR / "fake_site_daily.csv",
        sm_variant="raw",
        min_rows=100,
    )
    assert df is None
    assert status == "TOO_FEW_ROWS"
    assert status in SKIP_STATUSES


def test_load_site_data_missing_file(tmp_path: Path) -> None:
    df, status = load_site_data(
        site_csv=tmp_path / "nope.csv",
        sm_variant="raw",
        min_rows=1,
    )
    assert df is None
    assert status == "MISSING_FILE"


def test_load_site_data_missing_sm_variant(tmp_path: Path) -> None:
    # write a CSV that lacks the zscore column
    df_in = pd.read_csv(FIXTURE_DIR / "fake_site_daily.csv")
    df_in.drop(columns=["volumetric_soil_water_layer_1_zscore"], inplace=True)
    path = tmp_path / "no_zscore.csv"
    df_in.to_csv(path, index=False)

    df, status = load_site_data(site_csv=path, sm_variant="zscore", min_rows=1)
    assert df is None
    assert status == "MISSING_SM_VARIANT"
```

- [ ] **Step 3: Run tests — verify they fail**

```bash
pytest src/Analyzers/tests/test_per_site_sm_shap.py -v -k load_site_data
```
Expected: all fail with `ImportError: cannot import name 'load_site_data'`.

- [ ] **Step 4: Implement** — append to `src/Analyzers/per_site_sm_shap.py`

```python
# ── Status codes ─────────────────────────────────────────────────────────────

STATUS_OK = "OK"
STATUS_OK_NO_INTERACTION = "OK_NO_INTERACTION"
SKIP_STATUSES: frozenset[str] = frozenset({
    "MISSING_FILE",
    "MISSING_SM_VARIANT",
    "MISSING_FEATURE",
    "TOO_FEW_ROWS",
    "CV_FAILED",
})


# ── Data loading ─────────────────────────────────────────────────────────────

def load_site_data(
    site_csv: Path,
    sm_variant: str,
    min_rows: int,
) -> tuple["pd.DataFrame | None", str]:
    """Load one site's daily CSV and return (frame, status).

    On success returns (DataFrame with generic `sm` column, "OK").
    On failure returns (None, one of SKIP_STATUSES).
    """
    import pandas as pd

    if not site_csv.exists():
        return None, "MISSING_FILE"

    sm_source_col = SM_VARIANT_TO_COL[sm_variant]
    required_cols = [TARGET_COL, *FEATURE_COLS_BASE, sm_source_col]

    df = pd.read_csv(site_csv, usecols=lambda c: c in {"TIMESTAMP", *required_cols})

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        if sm_source_col in missing:
            return None, "MISSING_SM_VARIANT"
        return None, f"MISSING_FEATURE:{missing[0]}"

    df = df.rename(columns={sm_source_col: SM_COL_NAME})
    df = df.dropna(subset=[TARGET_COL, *FEATURE_COLS])
    df = df[df[TARGET_COL] > 0].reset_index(drop=True)

    if len(df) < min_rows:
        return None, "TOO_FEW_ROWS"

    return df, STATUS_OK
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
pytest src/Analyzers/tests/test_per_site_sm_shap.py -v -k load_site_data
```
Expected: 5 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/Analyzers/per_site_sm_shap.py src/Analyzers/tests/test_per_site_sm_shap.py src/Analyzers/tests/fixtures/fake_site_daily.csv
git commit -m "feat(per_site_sm_shap): load_site_data + tests"
```

---

## Task 4: `build_feature_matrix()` — TDD

**Files:**
- Modify: `src/Analyzers/tests/test_per_site_sm_shap.py`
- Modify: `src/Analyzers/per_site_sm_shap.py`

- [ ] **Step 1: Write failing tests**

```python
import numpy as np

from src.Analyzers.per_site_sm_shap import (
    FEATURE_COLS,
    SM_COL_NAME,
    TARGET_COL,
    build_feature_matrix,
)


def _make_clean_df(n: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "TIMESTAMP": pd.date_range("2020-06-01", periods=n, freq="D"),
        TARGET_COL:   rng.uniform(0.5, 3.0, n),
        "vpd":        rng.uniform(0.5, 2.5, n),
        "ta":         rng.uniform(10, 30, n),
        "ws":         rng.uniform(0.5, 3.0, n),
        "sw_in":      rng.uniform(100, 300, n),
        "precip_sum": rng.uniform(0, 5, n),
        SM_COL_NAME:  rng.uniform(0.15, 0.35, n),
    })


def test_build_feature_matrix_shape():
    df = _make_clean_df(n=30)
    X, y = build_feature_matrix(df)
    assert X.shape == (30, 6)
    assert y.shape == (30,)


def test_build_feature_matrix_column_order():
    df = _make_clean_df(n=10)
    X, _ = build_feature_matrix(df)
    assert list(X.columns) == list(FEATURE_COLS)


def test_build_feature_matrix_preserves_row_order():
    df = _make_clean_df(n=10)
    X, y = build_feature_matrix(df)
    np.testing.assert_array_equal(y.values, df[TARGET_COL].values)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest src/Analyzers/tests/test_per_site_sm_shap.py -v -k build_feature_matrix
```
Expected: 3 FAIL — `build_feature_matrix` not defined.

- [ ] **Step 3: Implement**

```python
# ── Feature matrix ───────────────────────────────────────────────────────────

def build_feature_matrix(df):
    """Select feature columns in canonical order and return (X, y).

    Assumes `df` has already been filtered by `load_site_data`.
    """
    X = df[list(FEATURE_COLS)].copy()
    y = df[TARGET_COL].copy()
    return X, y
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest src/Analyzers/tests/test_per_site_sm_shap.py -v -k build_feature_matrix
```
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/Analyzers/per_site_sm_shap.py src/Analyzers/tests/test_per_site_sm_shap.py
git commit -m "feat(per_site_sm_shap): build_feature_matrix + tests"
```

---

## Task 5: Site metadata join — TDD

**Files:**
- Modify: `src/Analyzers/tests/test_per_site_sm_shap.py`
- Modify: `src/Analyzers/per_site_sm_shap.py`

- [ ] **Step 1: Write failing tests**

```python
from src.Analyzers.per_site_sm_shap import load_site_metadata, lookup_site_meta


def test_load_site_metadata_returns_frame_with_site_code_column(tmp_path):
    csv = tmp_path / "meta.csv"
    csv.write_text("site_code,PFT,biome\nARG_MAZ,ENF,Temperate forest\n")
    meta = load_site_metadata(csv)
    assert "site_code" in meta.columns
    assert meta.loc[meta.site_code == "ARG_MAZ", "PFT"].iloc[0] == "ENF"


def test_lookup_site_meta_unknown_site_returns_sentinels(tmp_path):
    csv = tmp_path / "meta.csv"
    csv.write_text("site_code,PFT,biome\nARG_MAZ,ENF,Temperate forest\n")
    meta = load_site_metadata(csv)
    info = lookup_site_meta(meta, "NOT_IN_FILE")
    assert info == {"PFT": "unknown", "biome": "unknown"}


def test_lookup_site_meta_known_site():
    import io
    meta = pd.read_csv(io.StringIO("site_code,PFT,biome\nFIN_HYY,ENF,Boreal forest\n"))
    info = lookup_site_meta(meta, "FIN_HYY")
    assert info == {"PFT": "ENF", "biome": "Boreal forest"}
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest src/Analyzers/tests/test_per_site_sm_shap.py -v -k site_meta
```
Expected: FAIL (symbols not defined).

- [ ] **Step 3: Implement**

```python
# ── Site metadata ────────────────────────────────────────────────────────────

def load_site_metadata(path: Path):
    """Load site_biome_mapping.csv, normalising expected columns."""
    import pandas as pd
    meta = pd.read_csv(path)
    # file may use any of these header conventions; normalise to canonical set
    renames = {"Site": "site_code", "PFT_MODIS": "PFT", "Biome": "biome"}
    meta = meta.rename(columns={k: v for k, v in renames.items() if k in meta.columns})
    required = {"site_code", "PFT", "biome"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"site metadata missing columns: {missing}")
    return meta[list(required)]


def lookup_site_meta(meta, site_code: str) -> dict[str, str]:
    """Return {'PFT': ..., 'biome': ...}; 'unknown' sentinels if not found."""
    row = meta.loc[meta["site_code"] == site_code]
    if row.empty:
        return {"PFT": "unknown", "biome": "unknown"}
    return {"PFT": str(row["PFT"].iloc[0]), "biome": str(row["biome"].iloc[0])}
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest src/Analyzers/tests/test_per_site_sm_shap.py -v -k site_meta
```
Expected: 3 PASS.

- [ ] **Step 5: Spot-check the real metadata file header** (once, on HPC):

```bash
head -2 /scratch/tmp/yluo2/gsv/outputs/processed_data/sapwood/merged/site_biome_mapping.csv
```

If the header uses different column names than `{site_code, PFT, biome}`, extend the `renames` dict in `load_site_metadata` — do not change the internal contract. (If the header is already canonical, nothing to do.)

- [ ] **Step 6: Commit**

```bash
git add src/Analyzers/per_site_sm_shap.py src/Analyzers/tests/test_per_site_sm_shap.py
git commit -m "feat(per_site_sm_shap): site metadata loader + lookup"
```

---

## Task 6: Hyperparameter search — TDD

**Files:**
- Modify: `src/Analyzers/tests/test_per_site_sm_shap.py`
- Modify: `src/Analyzers/per_site_sm_shap.py`

- [ ] **Step 1: Write failing tests**

```python
from src.Analyzers.per_site_sm_shap import tune_site_hp, PARAM_DIST


def _make_synthetic_xy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "vpd":        rng.uniform(0.5, 2.5, n),
        "ta":         rng.uniform(10, 30, n),
        "ws":         rng.uniform(0.5, 3.0, n),
        "sw_in":      rng.uniform(100, 300, n),
        "precip_sum": rng.uniform(0, 5, n),
        SM_COL_NAME:  rng.uniform(0.15, 0.35, n),
    })
    # y is roughly linear in sw_in and sm → XGBoost should learn something
    y = (
        0.002 * X["sw_in"]
        + 5.0 * X["sm"]
        + 0.05 * X["vpd"]
        + rng.normal(0, 0.05, n)
    )
    return X, pd.Series(y)


def test_tune_site_hp_returns_expected_keys():
    X, y = _make_synthetic_xy(n=200)
    result = tune_site_hp(X, y, random_state=42)
    assert set(PARAM_DIST.keys()).issubset(result.best_params.keys())


def test_tune_site_hp_cv_r2_is_positive_on_learnable_data():
    X, y = _make_synthetic_xy(n=200)
    result = tune_site_hp(X, y, random_state=42)
    assert result.cv_r2_mean > 0.5, f"Expected learnable synthetic data; got R²={result.cv_r2_mean}"
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest src/Analyzers/tests/test_per_site_sm_shap.py -v -k tune_site_hp
```
Expected: FAIL.

- [ ] **Step 3: Implement — add a `HPResult` dataclass + function**

```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class HPResult:
    best_params: dict[str, object]
    cv_r2_mean: float
    cv_r2_std: float
    cv_rmse_mean: float


def tune_site_hp(X, y, random_state: int = 42) -> HPResult:
    """30-trial RandomizedSearchCV with 5-fold KFold. Returns HPResult."""
    from sklearn.model_selection import KFold, RandomizedSearchCV
    from xgboost import XGBRegressor

    search = RandomizedSearchCV(
        estimator=XGBRegressor(**FIXED_XGB_PARAMS, random_state=random_state),
        param_distributions=PARAM_DIST,
        n_iter=N_HP_TRIALS,
        cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state),
        scoring=["neg_root_mean_squared_error", "r2"],
        refit="neg_root_mean_squared_error",
        n_jobs=1,
        random_state=random_state,
        return_train_score=False,
    )
    search.fit(X, y)

    best_idx = int(search.best_index_)
    cv_r2_mean = float(search.cv_results_["mean_test_r2"][best_idx])
    cv_r2_std = float(search.cv_results_["std_test_r2"][best_idx])
    cv_rmse_mean = float(-search.cv_results_["mean_test_neg_root_mean_squared_error"][best_idx])

    best_params_clean = {k: search.best_params_[k] for k in PARAM_DIST.keys()}
    return HPResult(
        best_params=best_params_clean,
        cv_r2_mean=cv_r2_mean,
        cv_r2_std=cv_r2_std,
        cv_rmse_mean=cv_rmse_mean,
    )
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest src/Analyzers/tests/test_per_site_sm_shap.py -v -k tune_site_hp
```
Expected: 2 PASS (may take 30–90 s).

- [ ] **Step 5: Commit**

```bash
git add src/Analyzers/per_site_sm_shap.py src/Analyzers/tests/test_per_site_sm_shap.py
git commit -m "feat(per_site_sm_shap): HP random search + CV reporting"
```

---

## Task 7: Final model fit + SHAP — TDD

**Files:**
- Modify: `src/Analyzers/tests/test_per_site_sm_shap.py`
- Modify: `src/Analyzers/per_site_sm_shap.py`

- [ ] **Step 1: Write failing tests**

```python
from src.Analyzers.per_site_sm_shap import (
    FinalFit,
    compute_shap,
    fit_final_model,
)


def test_fit_final_model_predicts_reasonably():
    X, y = _make_synthetic_xy(n=200)
    best_params = {
        "max_depth": 4, "min_child_weight": 3, "n_estimators": 300,
        "subsample": 1.0, "gamma": 0.0,
    }
    fit = fit_final_model(X, y, best_params, random_state=42)
    assert fit.in_sample_r2 > 0.6


def test_compute_shap_shapes_and_identity():
    X, y = _make_synthetic_xy(n=120)
    best_params = {
        "max_depth": 3, "min_child_weight": 3, "n_estimators": 200,
        "subsample": 1.0, "gamma": 0.0,
    }
    fit = fit_final_model(X, y, best_params, random_state=42)
    shap_res = compute_shap(fit.model, X)

    n, p = X.shape
    assert shap_res.shap_values.shape == (n, p)
    assert shap_res.shap_interaction is not None
    assert shap_res.shap_interaction.shape == (n, p, p)
    # identity: sum of interaction row ≈ marginal shap
    recovered = shap_res.shap_interaction.sum(axis=2)
    np.testing.assert_allclose(recovered, shap_res.shap_values, atol=1e-3)


def test_compute_shap_main_effect_length_matches_X():
    X, y = _make_synthetic_xy(n=150)
    best_params = {
        "max_depth": 3, "min_child_weight": 3, "n_estimators": 200,
        "subsample": 1.0, "gamma": 0.0,
    }
    fit = fit_final_model(X, y, best_params, random_state=42)
    shap_res = compute_shap(fit.model, X)
    assert shap_res.main_effect_sm.shape == (150,)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest src/Analyzers/tests/test_per_site_sm_shap.py -v -k "final_model or compute_shap"
```
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class FinalFit:
    model: object            # xgboost.XGBRegressor
    in_sample_r2: float


@dataclass(frozen=True)
class ShapResult:
    shap_values: np.ndarray       # (n, p)
    shap_interaction: "np.ndarray | None"  # (n, p, p) or None if computation failed
    main_effect_sm: np.ndarray    # (n,)


def fit_final_model(X, y, best_params: dict, random_state: int = 42) -> FinalFit:
    """Refit XGBoost on full site data with already-tuned HPs."""
    from sklearn.metrics import r2_score
    from xgboost import XGBRegressor

    model = XGBRegressor(
        **FIXED_XGB_PARAMS,
        **best_params,
        random_state=random_state,
    )
    model.fit(X, y)
    in_sample_r2 = float(r2_score(y, model.predict(X)))
    return FinalFit(model=model, in_sample_r2=in_sample_r2)


def compute_shap(model, X) -> ShapResult:
    """Compute SHAP values + interaction values + SM main-effect vector."""
    import numpy as np
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = np.asarray(explainer.shap_values(X))

    try:
        shap_interaction = np.asarray(explainer.shap_interaction_values(X))
        main_effect_sm = shap_interaction[:, SM_IDX, SM_IDX]
    except Exception as exc:  # pragma: no cover - rare numerical failures
        logger.warning("shap_interaction_values failed: %s", exc)
        shap_interaction = None
        main_effect_sm = shap_values[:, SM_IDX]  # fallback: marginal SHAP

    return ShapResult(
        shap_values=shap_values,
        shap_interaction=shap_interaction,
        main_effect_sm=main_effect_sm,
    )
```

Add `import numpy as np` at the top of the file (next to other imports).

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest src/Analyzers/tests/test_per_site_sm_shap.py -v -k "final_model or compute_shap"
```
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/Analyzers/per_site_sm_shap.py src/Analyzers/tests/test_per_site_sm_shap.py
git commit -m "feat(per_site_sm_shap): fit_final_model + compute_shap + tests"
```

---

## Task 8: Plotting — two-panel dependence plot

**Files:**
- Modify: `src/Analyzers/tests/test_per_site_sm_shap.py`
- Modify: `src/Analyzers/per_site_sm_shap.py`

- [ ] **Step 1: Write failing test**

```python
from src.Analyzers.per_site_sm_shap import plot_dependence_pair


def test_plot_dependence_pair_writes_png(tmp_path):
    X, y = _make_synthetic_xy(n=120)
    best_params = {
        "max_depth": 3, "min_child_weight": 3, "n_estimators": 200,
        "subsample": 1.0, "gamma": 0.0,
    }
    from src.Analyzers.per_site_sm_shap import fit_final_model, compute_shap

    fit = fit_final_model(X, y, best_params, random_state=42)
    shap_res = compute_shap(fit.model, X)

    out_png = tmp_path / "test_plot.png"
    plot_dependence_pair(
        X=X,
        shap_result=shap_res,
        sm_variant="raw",
        site_meta={
            "site_code": "FAKE_SITE",
            "PFT": "ENF",
            "biome": "Temperate forest",
            "n_rows": len(X),
            "cv_r2_mean": 0.78, "cv_r2_std": 0.04,
            "in_sample_r2": 0.91,
            "best_params": best_params,
        },
        output_path=out_png,
    )
    assert out_png.exists()
    assert out_png.stat().st_size > 10_000  # non-trivial PNG
```

- [ ] **Step 2: Run test — verify it fails**

```bash
pytest src/Analyzers/tests/test_per_site_sm_shap.py -v -k plot_dependence_pair
```
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_dependence_pair(
    *,
    X,
    shap_result: ShapResult,
    sm_variant: str,
    site_meta: dict,
    output_path: Path,
) -> None:
    """Render the per-site 2-panel SM dependence figure as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except ImportError:
        lowess = None

    sm_vals = X[SM_COL_NAME].to_numpy()
    shap_sm_marginal = shap_result.shap_values[:, SM_IDX]
    main_effect_sm = shap_result.main_effect_sm
    vpd_vals = X["vpd"].to_numpy()

    x_unit = "m³/m³" if sm_variant == "raw" else "z-score"

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(12, 5), dpi=150, sharey=False
    )

    # Left — standard dependence, coloured by VPD
    sc = ax_left.scatter(sm_vals, shap_sm_marginal, c=vpd_vals, cmap="viridis", s=20, alpha=0.8)
    ax_left.axhline(0.0, color="grey", linestyle="--", linewidth=0.8)
    ax_left.set_xlabel(f"SM ({x_unit})")
    ax_left.set_ylabel("SHAP value  (Δ sap_velocity, cm³·cm⁻²·h⁻¹)")
    ax_left.set_title("Standard SHAP dependence")
    cbar = fig.colorbar(sc, ax=ax_left)
    cbar.set_label("VPD (kPa)")

    # Right — pure main effect
    ax_right.scatter(sm_vals, main_effect_sm, color="steelblue", s=20, alpha=0.8)
    if lowess is not None and len(sm_vals) >= 10:
        smoothed = lowess(main_effect_sm, sm_vals, frac=0.3, return_sorted=True)
        ax_right.plot(smoothed[:, 0], smoothed[:, 1], color="firebrick", linewidth=2)
    ax_right.axhline(0.0, color="grey", linestyle="--", linewidth=0.8)
    ax_right.set_xlabel(f"SM ({x_unit})")
    ax_right.set_ylabel("Main-effect SHAP value")
    ax_right.set_title("Pure main effect (interactions removed)")

    suptitle = (
        f"{site_meta['site_code']}  "
        f"PFT={site_meta['PFT']}  biome={site_meta['biome']}  "
        f"SM={sm_variant}"
    )
    fig.suptitle(suptitle, fontsize=12)

    bp = site_meta["best_params"]
    footer = (
        f"n={site_meta['n_rows']}   "
        f"CV-R²={site_meta['cv_r2_mean']:.2f}±{site_meta['cv_r2_std']:.2f}   "
        f"in-sample R²={site_meta['in_sample_r2']:.2f}   "
        f"max_depth={bp['max_depth']}, n_est={bp['n_estimators']}, "
        f"min_child_wt={bp['min_child_weight']}, subsample={bp['subsample']}, "
        f"gamma={bp['gamma']}"
    )
    fig.text(0.5, 0.01, footer, ha="center", fontsize=9)

    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 4: Run test — verify it passes**

```bash
pytest src/Analyzers/tests/test_per_site_sm_shap.py -v -k plot_dependence_pair
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/Analyzers/per_site_sm_shap.py src/Analyzers/tests/test_per_site_sm_shap.py
git commit -m "feat(per_site_sm_shap): 2-panel dependence plot"
```

---

## Task 9: Artifact writers — parquet + model

**Files:**
- Modify: `src/Analyzers/per_site_sm_shap.py`
- Modify: `src/Analyzers/tests/test_per_site_sm_shap.py`

- [ ] **Step 1: Write failing tests**

```python
from src.Analyzers.per_site_sm_shap import save_model, save_shap_parquet


def test_save_model_roundtrip(tmp_path):
    import joblib
    X, y = _make_synthetic_xy(n=120)
    best_params = {
        "max_depth": 3, "min_child_weight": 3, "n_estimators": 200,
        "subsample": 1.0, "gamma": 0.0,
    }
    from src.Analyzers.per_site_sm_shap import fit_final_model

    fit = fit_final_model(X, y, best_params, random_state=42)
    path = tmp_path / "m.joblib"
    save_model(fit.model, path)
    loaded = joblib.load(path)
    np.testing.assert_allclose(loaded.predict(X), fit.model.predict(X))


def test_save_shap_parquet_has_expected_columns(tmp_path):
    X, y = _make_synthetic_xy(n=50)
    best_params = {
        "max_depth": 3, "min_child_weight": 3, "n_estimators": 200,
        "subsample": 1.0, "gamma": 0.0,
    }
    from src.Analyzers.per_site_sm_shap import fit_final_model, compute_shap

    fit = fit_final_model(X, y, best_params, random_state=42)
    shap_res = compute_shap(fit.model, X)
    timestamps = pd.date_range("2020-06-01", periods=50, freq="D")
    path = tmp_path / "s.parquet"
    save_shap_parquet(
        shap_result=shap_res,
        X=X,
        timestamps=timestamps,
        output_path=path,
    )
    loaded = pd.read_parquet(path)
    assert {"TIMESTAMP", "sm", "shap_sm", "main_effect_sm",
            "shap_vpd", "shap_ta", "shap_ws", "shap_sw_in",
            "shap_precip_sum"}.issubset(loaded.columns)
    assert len(loaded) == 50
```

- [ ] **Step 2: Run — verify fail**

```bash
pytest src/Analyzers/tests/test_per_site_sm_shap.py -v -k "save_model or save_shap"
```
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# ── Artifact writers ─────────────────────────────────────────────────────────

def save_model(model, path: Path) -> None:
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def save_shap_parquet(
    *,
    shap_result: ShapResult,
    X,
    timestamps,
    output_path: Path,
) -> None:
    """Write a flat parquet: one row per observation, SHAP columns per feature."""
    import pandas as pd

    shap_df = pd.DataFrame(
        shap_result.shap_values,
        columns=[f"shap_{c}" for c in FEATURE_COLS],
    )
    out = pd.concat(
        [
            pd.Series(timestamps, name="TIMESTAMP").reset_index(drop=True),
            X.reset_index(drop=True),
            shap_df.reset_index(drop=True),
        ],
        axis=1,
    )
    out["main_effect_sm"] = shap_result.main_effect_sm
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
```

- [ ] **Step 4: Run — verify pass**

```bash
pytest src/Analyzers/tests/test_per_site_sm_shap.py -v -k "save_model or save_shap"
```
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/Analyzers/per_site_sm_shap.py src/Analyzers/tests/test_per_site_sm_shap.py
git commit -m "feat(per_site_sm_shap): model + SHAP parquet writers"
```

---

## Task 10: Per-site orchestrator + summary row

**Files:**
- Modify: `src/Analyzers/per_site_sm_shap.py`
- Modify: `src/Analyzers/tests/test_per_site_sm_shap.py`

- [ ] **Step 1: Write failing tests**

```python
from src.Analyzers.per_site_sm_shap import SiteConfig, process_one_site


def test_process_one_site_missing_file(tmp_path):
    cfg = SiteConfig(
        site_code="NO_SUCH_SITE",
        site_csv=tmp_path / "missing.csv",
        sm_variant="raw",
        min_rows=100,
        output_root=tmp_path / "out",
        site_meta={"PFT": "unknown", "biome": "unknown"},
        random_state=42,
    )
    row = process_one_site(cfg)
    assert row["status"] == "MISSING_FILE"
    assert row["site_code"] == "NO_SUCH_SITE"


def test_process_one_site_too_few_rows(tmp_path):
    site_csv = FIXTURE_DIR / "fake_site_daily.csv"
    cfg = SiteConfig(
        site_code="FAKE_SITE",
        site_csv=site_csv,
        sm_variant="raw",
        min_rows=1_000,
        output_root=tmp_path / "out",
        site_meta={"PFT": "ENF", "biome": "Temperate forest"},
        random_state=42,
    )
    row = process_one_site(cfg)
    assert row["status"] == "TOO_FEW_ROWS"


@pytest.mark.slow
def test_process_one_site_end_to_end(tmp_path):
    """Runs the real pipeline on the fixture. ~15–30 s."""
    site_csv = FIXTURE_DIR / "fake_site_daily.csv"
    cfg = SiteConfig(
        site_code="FAKE_SITE",
        site_csv=site_csv,
        sm_variant="raw",
        min_rows=5,                          # fixture has 8 usable rows
        output_root=tmp_path / "out",
        site_meta={"PFT": "ENF", "biome": "Temperate forest"},
        random_state=42,
    )
    row = process_one_site(cfg)
    assert row["status"] == "OK"
    assert row["n_rows"] == 8
    assert (tmp_path / "out" / "plots" / "FAKE_SITE_SM_dependence.png").exists()
    assert (tmp_path / "out" / "shap_values" / "FAKE_SITE_shap.parquet").exists()
    assert (tmp_path / "out" / "models" / "FAKE_SITE.joblib").exists()
```

Add `@pytest.mark.slow` marker — register in `pytest.ini` or `pyproject.toml` if not already present; otherwise acceptable as a custom mark.

- [ ] **Step 2: Run — verify fail**

```bash
pytest src/Analyzers/tests/test_per_site_sm_shap.py -v -k process_one_site
```
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# ── Per-site orchestrator ────────────────────────────────────────────────────

import time
import traceback


@dataclass(frozen=True)
class SiteConfig:
    site_code: str
    site_csv: Path
    sm_variant: str
    min_rows: int
    output_root: Path
    site_meta: dict
    random_state: int


def process_one_site(cfg: SiteConfig) -> dict:
    """Run the full pipeline for one site. Always returns a summary row dict.

    On skip / failure the row's `status` field records the reason; on success
    all artifacts are persisted under cfg.output_root.
    """
    t0 = time.time()
    row = {
        "site_code": cfg.site_code,
        "sm_variant": cfg.sm_variant,
        "status": "",
        "n_rows": 0,
        "PFT": cfg.site_meta.get("PFT", "unknown"),
        "biome": cfg.site_meta.get("biome", "unknown"),
        "cv_r2_mean": float("nan"),
        "cv_r2_std": float("nan"),
        "cv_rmse_mean": float("nan"),
        "in_sample_r2": float("nan"),
        "best_params": "",
        "sm_shap_mean_abs": float("nan"),
        "sm_main_effect_range": float("nan"),
        "runtime_sec": float("nan"),
    }

    try:
        df, status = load_site_data(cfg.site_csv, cfg.sm_variant, cfg.min_rows)
        if df is None:
            row["status"] = status
            return row
        row["n_rows"] = int(len(df))

        X, y = build_feature_matrix(df)

        hp = tune_site_hp(X, y, random_state=cfg.random_state)
        row.update({
            "cv_r2_mean": hp.cv_r2_mean,
            "cv_r2_std": hp.cv_r2_std,
            "cv_rmse_mean": hp.cv_rmse_mean,
            "best_params": _dumps(hp.best_params),
        })

        fit = fit_final_model(X, y, hp.best_params, random_state=cfg.random_state)
        row["in_sample_r2"] = fit.in_sample_r2

        shap_res = compute_shap(fit.model, X)
        row["sm_shap_mean_abs"] = float(np.mean(np.abs(shap_res.shap_values[:, SM_IDX])))
        row["sm_main_effect_range"] = float(
            shap_res.main_effect_sm.max() - shap_res.main_effect_sm.min()
        )

        plot_dir = cfg.output_root / "plots"
        parquet_dir = cfg.output_root / "shap_values"
        model_dir = cfg.output_root / "models"

        full_meta = {**cfg.site_meta, **hp.__dict__, "in_sample_r2": fit.in_sample_r2,
                     "site_code": cfg.site_code, "n_rows": row["n_rows"]}
        plot_dependence_pair(
            X=X,
            shap_result=shap_res,
            sm_variant=cfg.sm_variant,
            site_meta=full_meta,
            output_path=plot_dir / f"{cfg.site_code}_SM_dependence.png",
        )
        save_shap_parquet(
            shap_result=shap_res,
            X=X,
            timestamps=df["TIMESTAMP"].to_numpy(),
            output_path=parquet_dir / f"{cfg.site_code}_shap.parquet",
        )
        save_model(fit.model, model_dir / f"{cfg.site_code}.joblib")

        row["status"] = STATUS_OK if shap_res.shap_interaction is not None else STATUS_OK_NO_INTERACTION

    except Exception as exc:
        logger.error("Site %s failed:\n%s", cfg.site_code, traceback.format_exc())
        row["status"] = f"ERROR:{type(exc).__name__}:{exc}"[:200]
    finally:
        row["runtime_sec"] = round(time.time() - t0, 2)

    return row


def _dumps(params: dict) -> str:
    import json
    return json.dumps(params, sort_keys=True)
```

- [ ] **Step 4: Run — verify pass (slow test optional, normal tests must pass)**

```bash
pytest src/Analyzers/tests/test_per_site_sm_shap.py -v -k "process_one_site and not end_to_end"
pytest src/Analyzers/tests/test_per_site_sm_shap.py -v -k process_one_site_end_to_end
```
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/Analyzers/per_site_sm_shap.py src/Analyzers/tests/test_per_site_sm_shap.py
git commit -m "feat(per_site_sm_shap): per-site orchestrator + summary row"
```

---

## Task 11: Main loop, CLI wiring, summary.csv writer

**Files:**
- Modify: `src/Analyzers/per_site_sm_shap.py`
- Modify: `src/Analyzers/tests/test_per_site_sm_shap.py`

- [ ] **Step 1: Write failing test**

```python
def test_main_writes_summary_and_runs_on_fixture_only(tmp_path, monkeypatch):
    """End-to-end CLI using a private data dir containing the fixture only."""
    # Arrange: build a data dir with one site using the fixture CSV
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    import shutil
    shutil.copy(
        FIXTURE_DIR / "fake_site_daily.csv",
        data_dir / "FAKE_SITE_daily.csv",
    )
    meta_csv = tmp_path / "meta.csv"
    meta_csv.write_text("site_code,PFT,biome\nFAKE_SITE,ENF,Temperate forest\n")
    out_root = tmp_path / "out"

    # Act: call main() directly
    from src.Analyzers import per_site_sm_shap as m

    rc = m.main([
        "--sm-variant", "raw",
        "--n-jobs", "1",
        "--min-rows", "5",
        "--sites", "FAKE_SITE",
        "--output-dir", str(out_root),
        "--data-dir", str(data_dir),
        "--site-meta-csv", str(meta_csv),
    ])
    assert rc == 0
    summary_csv = out_root / "raw" / "summary.csv"
    assert summary_csv.exists()
    summary = pd.read_csv(summary_csv)
    assert "FAKE_SITE" in summary["site_code"].values
    assert summary.loc[summary.site_code == "FAKE_SITE", "status"].iloc[0] == "OK"
```

- [ ] **Step 2: Run — verify fail**

```bash
pytest src/Analyzers/tests/test_per_site_sm_shap.py -v -k test_main_writes_summary
```
Expected: FAIL (new CLI flags missing + main is a stub).

- [ ] **Step 3: Implement — rewrite `build_parser`, add `main()` body, add `discover_sites` + `append_to_run_log`**

Update `build_parser` to add these flags (add them alongside existing ones):

```python
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / DATA_DIR_REL,
        help="Directory containing {SITE}_daily.csv files.",
    )
    parser.add_argument(
        "--site-meta-csv",
        type=Path,
        default=REPO_ROOT / SITE_META_REL,
        help="CSV file with columns site_code, PFT, biome.",
    )
```

Replace `main()` with:

```python
def discover_sites(data_dir: Path) -> list[str]:
    """Return all site codes present in data_dir."""
    return sorted(p.stem.replace("_daily", "") for p in data_dir.glob("*_daily.csv"))


def append_to_run_log(log_path: Path, rows: list[dict]) -> None:
    """Append one line per site result to run_log.txt."""
    from datetime import datetime, timezone
    log_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with log_path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{now}\t{row['site_code']}\t{row['status']}\t{row['runtime_sec']}s\n")


def main(argv: list[str] | None = None) -> int:
    from joblib import Parallel, delayed
    import pandas as pd

    args = build_parser().parse_args(argv)

    output_root = args.output_dir / args.sm_variant
    output_root.mkdir(parents=True, exist_ok=True)

    sites = args.sites or discover_sites(args.data_dir)
    if not sites:
        logger.error("No sites found in %s", args.data_dir)
        return 1

    logger.info("Processing %d sites, variant=%s, n_jobs=%d",
                len(sites), args.sm_variant, args.n_jobs)

    meta = load_site_metadata(args.site_meta_csv)

    configs = [
        SiteConfig(
            site_code=sc,
            site_csv=args.data_dir / f"{sc}_daily.csv",
            sm_variant=args.sm_variant,
            min_rows=args.min_rows,
            output_root=output_root,
            site_meta=lookup_site_meta(meta, sc),
            random_state=args.random_seed,
        )
        for sc in sites
    ]

    rows = Parallel(n_jobs=args.n_jobs, verbose=10, backend="loky")(
        delayed(process_one_site)(cfg) for cfg in configs
    )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_root / "summary.csv", index=False)
    append_to_run_log(args.output_dir / "run_log.txt", rows)

    ok_count = int((summary_df["status"] == STATUS_OK).sum())
    logger.info("Finished. OK=%d of %d.", ok_count, len(rows))
    return 0 if ok_count > 0 else 1
```

- [ ] **Step 4: Run — verify pass**

```bash
pytest src/Analyzers/tests/test_per_site_sm_shap.py -v -k test_main_writes_summary
```
Expected: PASS.

- [ ] **Step 5: Run the full test suite (fast + slow)**

```bash
pytest src/Analyzers/tests/test_per_site_sm_shap.py -v
pytest src/Analyzers/tests/test_per_site_sm_shap.py -v -m slow
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/Analyzers/per_site_sm_shap.py src/Analyzers/tests/test_per_site_sm_shap.py
git commit -m "feat(per_site_sm_shap): main loop + joblib parallel + summary.csv"
```

---

## Task 12: Pool-level small-multiples figure

**Files:**
- Modify: `src/Analyzers/per_site_sm_shap.py`
- Modify: `src/Analyzers/tests/test_per_site_sm_shap.py`

- [ ] **Step 1: Write failing test**

```python
def test_make_pool_figure_writes_png(tmp_path):
    from src.Analyzers.per_site_sm_shap import make_pool_figure

    # Build a dummy output dir with 3 fake site parquets
    shap_dir = tmp_path / "shap_values"
    shap_dir.mkdir()
    rng = np.random.default_rng(0)
    for site in ["A", "B", "C"]:
        n = 50
        df = pd.DataFrame({
            "TIMESTAMP":     pd.date_range("2020-06-01", periods=n),
            "sm":            rng.uniform(0.1, 0.4, n),
            "main_effect_sm": rng.normal(0, 0.05, n),
        })
        df.to_parquet(shap_dir / f"{site}_shap.parquet")

    summary = pd.DataFrame({
        "site_code": ["A", "B", "C"],
        "status":    ["OK"] * 3,
        "biome":     ["Boreal forest"] * 3,
        "PFT":       ["ENF"] * 3,
        "n_rows":    [50, 50, 50],
    })
    summary.to_csv(tmp_path / "summary.csv", index=False)

    out_png = tmp_path / "pool_by_biome.png"
    make_pool_figure(
        output_root=tmp_path,
        facet_by="biome",
        output_path=out_png,
        sm_variant="raw",
    )
    assert out_png.exists()
```

- [ ] **Step 2: Run — verify fail**

```bash
pytest src/Analyzers/tests/test_per_site_sm_shap.py -v -k pool_figure
```
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
def make_pool_figure(
    *,
    output_root: Path,
    facet_by: str,
    output_path: Path,
    sm_variant: str,
) -> None:
    """Small-multiples: one subplot per site, x=SM, y=main_effect_sm, facet column=facet_by."""
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    summary = pd.read_csv(output_root / "summary.csv")
    summary = summary[summary["status"].isin([STATUS_OK, STATUS_OK_NO_INTERACTION])]
    if summary.empty:
        logger.warning("No successful sites; skipping pool figure.")
        return

    n_sites = len(summary)
    ncols = min(6, max(1, int(math.ceil(math.sqrt(n_sites)))))
    nrows = int(math.ceil(n_sites / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.2 * ncols, 1.8 * nrows),
        dpi=150, sharex=True, sharey=True,
    )
    axes = np.atleast_2d(axes)

    x_unit = "m³/m³" if sm_variant == "raw" else "z-score"

    for idx, (_, site_row) in enumerate(summary.iterrows()):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        parquet = output_root / "shap_values" / f"{site_row['site_code']}_shap.parquet"
        if not parquet.exists():
            ax.axis("off")
            continue
        df = pd.read_parquet(parquet, columns=["sm", "main_effect_sm"])
        ax.scatter(df["sm"], df["main_effect_sm"], s=4, alpha=0.6)
        ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.5)
        ax.set_title(
            f"{site_row['site_code']}\n{site_row.get(facet_by, '')}",
            fontsize=7,
        )
        ax.tick_params(labelsize=6)

    # turn off any unused axes
    for idx in range(n_sites, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    fig.suptitle(
        f"Per-site SM main-effect ({sm_variant})  —  faceted by {facet_by}",
        fontsize=12,
    )
    fig.supxlabel(f"SM ({x_unit})")
    fig.supylabel("Main-effect SHAP")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 4: Call pool figure at the end of `main()` — modify `main()`**

Just before `return 0 if ok_count > 0 else 1`, insert:

```python
    pool_dir = output_root / "pool"
    try:
        make_pool_figure(output_root=output_root, facet_by="biome",
                         output_path=pool_dir / "pool_by_biome.png",
                         sm_variant=args.sm_variant)
        make_pool_figure(output_root=output_root, facet_by="PFT",
                         output_path=pool_dir / "pool_by_pft.png",
                         sm_variant=args.sm_variant)
    except Exception:
        logger.error("Pool figure failed:\n%s", traceback.format_exc())
```

- [ ] **Step 5: Run — verify pass**

```bash
pytest src/Analyzers/tests/test_per_site_sm_shap.py -v -k pool_figure
pytest src/Analyzers/tests/test_per_site_sm_shap.py -v -k test_main_writes_summary
```
Expected: PASS (the summary test already exercises `main` and will now also create pool figures).

- [ ] **Step 6: Commit**

```bash
git add src/Analyzers/per_site_sm_shap.py src/Analyzers/tests/test_per_site_sm_shap.py
git commit -m "feat(per_site_sm_shap): pool small-multiples figure"
```

---

## Task 13: SLURM job script

**Files:**
- Create: `.claude/plan/job_per_site_sm_shap.sh`

- [ ] **Step 1: Create the job file**

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
#SBATCH --error=logs/per_site_sm_shap_%j.err

set -euo pipefail

cd /scratch/tmp/yluo2/gsv
mkdir -p logs

source .venv/bin/activate
export PYTHONUNBUFFERED=1
# keep XGBoost / sklearn / BLAS serial; joblib handles outer parallelism
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== raw variant ==="
python src/Analyzers/per_site_sm_shap.py --sm-variant raw    --n-jobs 20

echo "=== zscore variant ==="
python src/Analyzers/per_site_sm_shap.py --sm-variant zscore --n-jobs 20

echo "Done."
```

- [ ] **Step 2: Commit**

```bash
git add .claude/plan/job_per_site_sm_shap.sh
git commit -m "chore(per_site_sm_shap): SLURM job script"
```

---

## Task 14: HPC dry-run on 3 sites

**Files:** (no code changes expected; may reveal bugs that trigger fixes)

- [ ] **Step 1: Push to HPC** (SCP — per project rule, commits happen on HPC; local commits from Tasks 1–13 should have already been synced via worktree or SCP as you worked)

If you've been developing locally, run the SCP loop now (adjust paths as needed):

```bash
/c/Windows/System32/OpenSSH/scp.exe -i "$HOME/.ssh/id_ecdsa_palma" \
  src/Analyzers/per_site_sm_shap.py \
  yluo2@palma-login.uni-muenster.de:/scratch/tmp/yluo2/gsv/src/Analyzers/
/c/Windows/System32/OpenSSH/scp.exe -i "$HOME/.ssh/id_ecdsa_palma" \
  src/Analyzers/tests/test_per_site_sm_shap.py \
  yluo2@palma-login.uni-muenster.de:/scratch/tmp/yluo2/gsv/src/Analyzers/tests/
/c/Windows/System32/OpenSSH/scp.exe -i "$HOME/.ssh/id_ecdsa_palma" -r \
  src/Analyzers/tests/fixtures \
  yluo2@palma-login.uni-muenster.de:/scratch/tmp/yluo2/gsv/src/Analyzers/tests/
/c/Windows/System32/OpenSSH/scp.exe -i "$HOME/.ssh/id_ecdsa_palma" \
  .claude/plan/job_per_site_sm_shap.sh \
  yluo2@palma-login.uni-muenster.de:/scratch/tmp/yluo2/gsv/.claude/plan/
```

(If you worked directly on HPC, skip this step — just commit.)

- [ ] **Step 2: Commit on HPC**

```bash
/c/Windows/System32/OpenSSH/ssh.exe -i "$HOME/.ssh/id_ecdsa_palma" \
  yluo2@palma-login.uni-muenster.de \
  "cd /scratch/tmp/yluo2/gsv && git add -A && git commit -m 'feat(per_site_sm_shap): implementation'"
```

- [ ] **Step 3: Run pytest on HPC (login-node OK for tests; only full run needs sbatch)**

```bash
/c/Windows/System32/OpenSSH/ssh.exe -i "$HOME/.ssh/id_ecdsa_palma" \
  yluo2@palma-login.uni-muenster.de \
  "cd /scratch/tmp/yluo2/gsv && source .venv/bin/activate && \
   pytest src/Analyzers/tests/test_per_site_sm_shap.py -v"
```
Expected: all tests pass.

- [ ] **Step 4: Submit a tiny dry-run via `srun` (NOT on login node — use `srun` with a short allocation)**

```bash
/c/Windows/System32/OpenSSH/ssh.exe -i "$HOME/.ssh/id_ecdsa_palma" \
  yluo2@palma-login.uni-muenster.de \
  "cd /scratch/tmp/yluo2/gsv && \
   srun --partition=normal --time=00:30:00 --cpus-per-task=3 --mem=8G --pty \
     bash -lc 'source .venv/bin/activate && \
       python src/Analyzers/per_site_sm_shap.py \
         --sm-variant raw --n-jobs 3 \
         --sites ARG_MAZ FIN_HYY AUS_KAR \
         --output-dir outputs/analysis/per_site_sm_shap_dryrun'"
```

- [ ] **Step 5: Inspect dry-run outputs**

```bash
/c/Windows/System32/OpenSSH/ssh.exe -i "$HOME/.ssh/id_ecdsa_palma" \
  yluo2@palma-login.uni-muenster.de \
  "cd /scratch/tmp/yluo2/gsv && \
    ls outputs/analysis/per_site_sm_shap_dryrun/raw/plots && \
    cat outputs/analysis/per_site_sm_shap_dryrun/raw/summary.csv"
```
Expected: three PNGs present, summary.csv has 3 rows with `status=OK`.

- [ ] **Step 6: If any issues surface, fix in code + rerun tests + re-scp + commit — repeat until dry-run is clean.**

- [ ] **Step 7: Remove dry-run output to keep the outputs dir tidy**

```bash
/c/Windows/System32/OpenSSH/ssh.exe -i "$HOME/.ssh/id_ecdsa_palma" \
  yluo2@palma-login.uni-muenster.de \
  "rm -rf /scratch/tmp/yluo2/gsv/outputs/analysis/per_site_sm_shap_dryrun"
```

---

## Task 15: Full production run on HPC

- [ ] **Step 1: Submit the SLURM job**

```bash
/c/Windows/System32/OpenSSH/ssh.exe -i "$HOME/.ssh/id_ecdsa_palma" \
  yluo2@palma-login.uni-muenster.de \
  "cd /scratch/tmp/yluo2/gsv && sbatch .claude/plan/job_per_site_sm_shap.sh"
```

Capture the job ID printed.

- [ ] **Step 2: Monitor**

```bash
/c/Windows/System32/OpenSSH/ssh.exe -i "$HOME/.ssh/id_ecdsa_palma" \
  yluo2@palma-login.uni-muenster.de \
  "squeue -u yluo2 -j <JOB_ID>"
# and, once running:
/c/Windows/System32/OpenSSH/ssh.exe -i "$HOME/.ssh/id_ecdsa_palma" \
  yluo2@palma-login.uni-muenster.de \
  "tail -F /scratch/tmp/yluo2/gsv/logs/per_site_sm_shap_<JOB_ID>.out"
```

- [ ] **Step 3: Verify outputs after job completes**

```bash
/c/Windows/System32/OpenSSH/ssh.exe -i "$HOME/.ssh/id_ecdsa_palma" \
  yluo2@palma-login.uni-muenster.de \
  "cd /scratch/tmp/yluo2/gsv && \
    ls outputs/analysis/per_site_sm_shap/raw/plots | wc -l && \
    ls outputs/analysis/per_site_sm_shap/zscore/plots | wc -l && \
    awk -F, 'NR>1 {print \$3}' outputs/analysis/per_site_sm_shap/raw/summary.csv | sort | uniq -c && \
    awk -F, 'NR>1 {print \$3}' outputs/analysis/per_site_sm_shap/zscore/summary.csv | sort | uniq -c && \
    ls outputs/analysis/per_site_sm_shap/raw/pool/"
```
Expected: per-SM-variant directories each have ~150–185 plots, `summary.csv` statuses dominated by `OK`, pool figures exist.

- [ ] **Step 4: Work-log report** per project rule — use the `work-log-reporter` skill once the run is validated.

---

## Self-Review (filled)

**1. Spec coverage**
- §2 data — Tasks 3, 5 ✓
- §3.1 HP search — Task 6 ✓
- §3.2 final refit — Task 7 ✓
- §3.3 SHAP (+ identity sanity) — Task 7 ✓
- §4.1 per-site 2-panel figure — Task 8 ✓
- §4.2 pool figure — Task 12 ✓
- §4.3 directory layout — Tasks 9, 11, 12 ✓
- §4.4 summary.csv schema — Task 10 ✓
- §5 error handling — Task 10 (all status branches) ✓
- §6 testing — Tasks 3–12 inline ✓
- §7.1 SLURM job — Task 13 ✓
- §7.2 CLI contract — Task 11 (+ `--data-dir`, `--site-meta-csv` added for testability; spec permits since contract is additive) ✓
- §7.3 resource budget — Task 13 + Task 15 validation ✓

**2. Placeholder scan:** grep turned up no TBD / TODO / "similar to X" — every code block is complete and self-contained.

**3. Type consistency:** `HPResult`, `FinalFit`, `ShapResult`, `SiteConfig` dataclasses have stable field names across tasks; `FEATURE_COLS`, `SM_COL_NAME`, `SM_IDX` constants referenced identically wherever they appear.
