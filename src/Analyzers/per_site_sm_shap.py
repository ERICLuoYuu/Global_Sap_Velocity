"""Per-Site Soil Moisture -> Sap Flow XGBoost + SHAP analysis.

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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("per_site_sm_shap")


# ── Constants ────────────────────────────────────────────────────────────────

FEATURE_COLS_BASE: tuple[str, ...] = ("vpd", "ta", "ws", "sw_in", "precip_sum")
TARGET_COL: str = "sap_velocity"
SM_COL_NAME: str = "sm"  # generic internal name
FEATURE_COLS: tuple[str, ...] = FEATURE_COLS_BASE + (SM_COL_NAME,)  # final 6 features
SM_IDX: int = FEATURE_COLS.index(SM_COL_NAME)

SM_VARIANT_TO_COL: dict[str, str] = {
    "raw": "volumetric_soil_water_layer_1_raw",
    "zscore": "volumetric_soil_water_layer_1_zscore",
}

DATA_DIR_REL = Path("outputs/processed_data/sapwood/merged/daytime_only/growing_season/daily")
SITE_META_REL = Path("outputs/processed_data/sapwood/merged/site_biome_mapping.csv")

PARAM_DIST: dict[str, list] = {
    "max_depth": [3, 4, 5],
    "min_child_weight": [1, 3, 5, 10],
    "n_estimators": [200, 400, 600],
    "subsample": [0.8, 1.0],
    "gamma": [0.0, 0.1],
}

FIXED_XGB_PARAMS: dict[str, object] = {
    "learning_rate": 0.05,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "tree_method": "hist",
    "n_jobs": 1,  # outer parallelism is across sites
    "objective": "reg:squarederror",
}

N_HP_TRIALS: int = 30
CV_FOLDS: int = 5


# ── Status codes ─────────────────────────────────────────────────────────────

STATUS_OK = "OK"
STATUS_OK_NO_INTERACTION = "OK_NO_INTERACTION"
SKIP_STATUSES: frozenset[str] = frozenset(
    {
        "MISSING_FILE",
        "MISSING_SM_VARIANT",
        "MISSING_FEATURE",
        "TOO_FEW_ROWS",
        "CV_FAILED",
    }
)


# ── Data loading ─────────────────────────────────────────────────────────────


def load_site_data(
    site_csv: Path,
    sm_variant: str,
    min_rows: int,
) -> tuple[pd.DataFrame | None, str]:
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
        logger.warning("Site %s missing feature columns: %s", site_csv.stem, missing)
        return None, "MISSING_FEATURE"

    df = df.rename(columns={sm_source_col: SM_COL_NAME})
    df = df.dropna(subset=[TARGET_COL, *FEATURE_COLS])
    df = df[df[TARGET_COL] > 0].reset_index(drop=True)

    if len(df) < min_rows:
        return None, "TOO_FEW_ROWS"

    return df, STATUS_OK


# ── Feature matrix ───────────────────────────────────────────────────────────


def build_feature_matrix(df):
    """Select feature columns in canonical order and return (X, y).

    Assumes `df` has already been filtered by `load_site_data`.
    """
    X = df[list(FEATURE_COLS)].copy()
    y = df[TARGET_COL].copy()
    return X, y


# ── Site metadata ────────────────────────────────────────────────────────────


def load_site_metadata(path: Path):
    """Load site_biome_mapping.csv, normalising expected columns.

    The canonical file ships with `site,biome` only (no PFT). We rename
    `site -> site_code` and fill PFT="unknown" when absent so the returned
    frame always exposes the contract {site_code, PFT, biome}. Per-row PFT
    from individual daily CSVs is preferred downstream via
    `process_one_site` (see Task 10) but this loader keeps working when only
    biome is available.
    """
    import pandas as pd

    meta = pd.read_csv(path)
    renames = {
        "Site": "site_code",
        "site": "site_code",
        "PFT_MODIS": "PFT",
        "pft": "PFT",
        "Biome": "biome",
    }
    meta = meta.rename(columns={k: v for k, v in renames.items() if k in meta.columns})
    if "site_code" not in meta.columns:
        raise ValueError("site metadata missing site identifier column")
    if "biome" not in meta.columns:
        raise ValueError("site metadata missing biome column")
    if "PFT" not in meta.columns:
        meta["PFT"] = "unknown"
    return meta[["site_code", "PFT", "biome"]]


def lookup_site_meta(meta, site_code: str) -> dict[str, str]:
    """Return {'PFT': ..., 'biome': ...}; 'unknown' sentinels if not found."""
    row = meta.loc[meta["site_code"] == site_code]
    if row.empty:
        return {"PFT": "unknown", "biome": "unknown"}
    return {"PFT": str(row["PFT"].iloc[0]), "biome": str(row["biome"].iloc[0])}


# ── Hyperparameter tuning ────────────────────────────────────────────────────


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

    best_params_clean = {k: search.best_params_[k] for k in PARAM_DIST}
    return HPResult(
        best_params=best_params_clean,
        cv_r2_mean=cv_r2_mean,
        cv_r2_std=cv_r2_std,
        cv_rmse_mean=cv_rmse_mean,
    )


# ── Final fit + SHAP ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FinalFit:
    model: object  # xgboost.XGBRegressor
    in_sample_r2: float


@dataclass(frozen=True)
class ShapResult:
    shap_values: np.ndarray  # (n, p)
    shap_interaction: np.ndarray | None  # (n, p, p) or None
    main_effect_sm: np.ndarray | None  # (n,) or None if interaction failed


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
    """Compute SHAP values + interaction values + SM main-effect vector.

    When ``shap_interaction_values`` fails (rare numerical edge cases),
    ``shap_interaction`` and ``main_effect_sm`` are set to None — callers MUST
    check for None rather than receiving a silent fallback to marginal SHAP
    (which would visually duplicate the left-panel curve).
    """
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = np.asarray(explainer.shap_values(X))

    try:
        shap_interaction = np.asarray(explainer.shap_interaction_values(X))
        main_effect_sm = shap_interaction[:, SM_IDX, SM_IDX]
    except Exception as exc:  # pragma: no cover - rare numerical failures
        logger.warning("shap_interaction_values failed: %s", exc)
        shap_interaction = None
        main_effect_sm = None

    return ShapResult(
        shap_values=shap_values,
        shap_interaction=shap_interaction,
        main_effect_sm=main_effect_sm,
    )


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Per-site SM -> sap_velocity XGBoost + SHAP analysis.")
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
