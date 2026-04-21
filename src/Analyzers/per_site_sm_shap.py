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
from pathlib import Path

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
