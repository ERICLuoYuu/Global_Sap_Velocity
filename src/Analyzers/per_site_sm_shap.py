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


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_dependence_pair(
    *,
    X,
    shap_result: ShapResult,
    sm_variant: str,
    site_meta: dict,
    output_path: Path,
) -> None:
    """Render the per-site SM dependence figure as PNG.

    Two-panel layout when ``shap_result.main_effect_sm`` is available; falls
    back to a single-panel figure with a visible red banner when the
    interaction computation failed (``main_effect_sm is None``). The fallback
    is visually distinct so viewers don't mistake it for the two-panel plot.
    """
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

    x_unit = "m^3/m^3" if sm_variant == "raw" else "z-score"
    two_panel = main_effect_sm is not None

    if two_panel:
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5), dpi=150, sharey=False)
    else:
        fig, ax_left = plt.subplots(1, 1, figsize=(7, 5), dpi=150)
        ax_right = None

    sc = ax_left.scatter(sm_vals, shap_sm_marginal, c=vpd_vals, cmap="viridis", s=20, alpha=0.8)
    ax_left.axhline(0.0, color="grey", linestyle="--", linewidth=0.8)
    ax_left.set_xlabel(f"SM ({x_unit})")
    ax_left.set_ylabel("SHAP value  (delta sap_velocity, cm3/cm2/h)")
    ax_left.set_title("Standard SHAP dependence")
    cbar = fig.colorbar(sc, ax=ax_left)
    cbar.set_label("VPD (kPa)")

    if two_panel:
        ax_right.scatter(sm_vals, main_effect_sm, color="steelblue", s=20, alpha=0.8)
        if lowess is not None and len(sm_vals) >= 10:
            smoothed = lowess(main_effect_sm, sm_vals, frac=0.3, return_sorted=True)
            ax_right.plot(smoothed[:, 0], smoothed[:, 1], color="firebrick", linewidth=2)
        ax_right.axhline(0.0, color="grey", linestyle="--", linewidth=0.8)
        ax_right.set_xlabel(f"SM ({x_unit})")
        ax_right.set_ylabel("Main-effect SHAP value")
        ax_right.set_title("Pure main effect (interactions removed)")
    else:
        ax_left.text(
            0.98,
            0.02,
            "main-effect computation unavailable\n(shap_interaction_values failed)",
            transform=ax_left.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="firebrick",
            bbox=dict(facecolor="white", edgecolor="firebrick", alpha=0.85),
        )

    suptitle = f"{site_meta['site_code']}  PFT={site_meta['PFT']}  biome={site_meta['biome']}  SM={sm_variant}"
    fig.suptitle(suptitle, fontsize=12)

    bp = site_meta["best_params"]
    footer = (
        f"n={site_meta['n_rows']}   "
        f"CV-R2={site_meta['cv_r2_mean']:.2f}+/-{site_meta['cv_r2_std']:.2f}   "
        f"in-sample R2={site_meta['in_sample_r2']:.2f}   "
        f"max_depth={bp['max_depth']}, n_est={bp['n_estimators']}, "
        f"min_child_wt={bp['min_child_weight']}, subsample={bp['subsample']}, "
        f"gamma={bp['gamma']}"
    )
    fig.text(0.5, 0.01, footer, ha="center", fontsize=9)

    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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


# ── Per-site orchestrator ────────────────────────────────────────────────────


@dataclass(frozen=True)
class SiteConfig:
    site_code: str
    site_csv: Path
    sm_variant: str
    min_rows: int
    output_root: Path
    site_meta: dict
    random_state: int


def _dumps(params: dict) -> str:
    import json

    return json.dumps(params, sort_keys=True)


def process_one_site(cfg: SiteConfig) -> dict:
    """Run the full pipeline for one site. Always returns a summary row dict.

    On skip / failure the row's `status` field records the reason; on success
    all artifacts are persisted under cfg.output_root.
    """
    import time
    import traceback
    from dataclasses import asdict

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

        try:
            hp = tune_site_hp(X, y, random_state=cfg.random_state)
        except Exception as exc:
            logger.error("Site %s CV_FAILED:\n%s", cfg.site_code, traceback.format_exc())
            row["status"] = f"CV_FAILED:{type(exc).__name__}"[:200]
            return row

        row.update(
            {
                "cv_r2_mean": hp.cv_r2_mean,
                "cv_r2_std": hp.cv_r2_std,
                "cv_rmse_mean": hp.cv_rmse_mean,
                "best_params": _dumps(hp.best_params),
            }
        )

        try:
            fit = fit_final_model(X, y, hp.best_params, random_state=cfg.random_state)
        except Exception as exc:
            logger.error("Site %s FIT_FAILED:\n%s", cfg.site_code, traceback.format_exc())
            row["status"] = f"FIT_FAILED:{type(exc).__name__}"[:200]
            return row
        row["in_sample_r2"] = fit.in_sample_r2

        try:
            shap_res = compute_shap(fit.model, X)
        except Exception as exc:
            logger.error("Site %s SHAP_FAILED:\n%s", cfg.site_code, traceback.format_exc())
            row["status"] = f"SHAP_FAILED:{type(exc).__name__}"[:200]
            return row

        row["sm_shap_mean_abs"] = float(np.mean(np.abs(shap_res.shap_values[:, SM_IDX])))
        if shap_res.main_effect_sm is not None:
            row["sm_main_effect_range"] = float(shap_res.main_effect_sm.max() - shap_res.main_effect_sm.min())

        plot_dir = cfg.output_root / "plots"
        parquet_dir = cfg.output_root / "shap_values"
        model_dir = cfg.output_root / "models"

        full_meta = {
            **cfg.site_meta,
            **asdict(hp),
            "in_sample_r2": fit.in_sample_r2,
            "site_code": cfg.site_code,
            "n_rows": row["n_rows"],
        }

        try:
            plot_dependence_pair(
                X=X,
                shap_result=shap_res,
                sm_variant=cfg.sm_variant,
                site_meta=full_meta,
                output_path=plot_dir / f"{cfg.site_code}_SM_dependence.png",
            )
        except Exception as exc:
            logger.error("Site %s PLOT_FAILED:\n%s", cfg.site_code, traceback.format_exc())
            row["status"] = f"PLOT_FAILED:{type(exc).__name__}"[:200]

        if "TIMESTAMP" in df.columns:
            ts = df["TIMESTAMP"].to_numpy()
        else:
            logger.warning("Site %s has no TIMESTAMP column; using row index", cfg.site_code)
            ts = np.arange(len(df))
        save_shap_parquet(
            shap_result=shap_res,
            X=X,
            timestamps=ts,
            output_path=parquet_dir / f"{cfg.site_code}_shap.parquet",
        )
        save_model(fit.model, model_dir / f"{cfg.site_code}.joblib")

        if not row["status"]:
            row["status"] = STATUS_OK if shap_res.shap_interaction is not None else STATUS_OK_NO_INTERACTION

    except Exception as exc:
        logger.error("Site %s unexpected failure:\n%s", cfg.site_code, traceback.format_exc())
        row["status"] = f"ERROR:{type(exc).__name__}:{exc}"[:200]
    finally:
        row["runtime_sec"] = round(time.time() - t0, 2)

    return row


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
    return parser


def discover_sites(data_dir: Path) -> list[str]:
    """Return all site codes present in data_dir. Trailing ``_daily`` only."""
    import re

    return sorted(re.sub(r"_daily$", "", p.stem) for p in data_dir.glob("*_daily.csv"))


def make_pool_figure(
    *,
    output_root: Path,
    facet_by: str,
    output_path: Path,
    sm_variant: str,
) -> None:
    """Small-multiples: one subplot per site, x=SM, y=main_effect_sm."""
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = pd.read_csv(output_root / "summary.csv")
    summary = summary[summary["status"].isin([STATUS_OK, STATUS_OK_NO_INTERACTION])]
    if summary.empty:
        logger.warning("No successful sites; skipping pool figure.")
        return

    n_sites = len(summary)
    ncols = min(6, max(1, int(math.ceil(math.sqrt(n_sites)))))
    nrows = int(math.ceil(n_sites / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.2 * ncols, 1.8 * nrows),
        dpi=150,
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_2d(axes)

    x_unit = "m^3/m^3" if sm_variant == "raw" else "z-score"

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

    for idx in range(n_sites, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    fig.suptitle(
        f"Per-site SM main-effect ({sm_variant}) - faceted by {facet_by}",
        fontsize=12,
    )
    fig.supxlabel(f"SM ({x_unit})")
    fig.supylabel("Main-effect SHAP")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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

    args = build_parser().parse_args(argv)

    output_root = args.output_dir / args.sm_variant
    output_root.mkdir(parents=True, exist_ok=True)

    sites = args.sites or discover_sites(args.data_dir)
    if not sites:
        logger.error("No sites found in %s", args.data_dir)
        return 1

    logger.info(
        "Processing %d sites, variant=%s, n_jobs=%d",
        len(sites),
        args.sm_variant,
        args.n_jobs,
    )

    try:
        meta = load_site_metadata(args.site_meta_csv)
    except (FileNotFoundError, ValueError) as exc:
        logger.warning(
            "Could not load site metadata (%s): %s - continuing with 'unknown' PFT/biome annotations.",
            args.site_meta_csv,
            exc,
        )
        meta = pd.DataFrame(columns=["site_code", "PFT", "biome"])

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

    rows = Parallel(n_jobs=args.n_jobs, verbose=5, backend="loky")(delayed(process_one_site)(cfg) for cfg in configs)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_root / "summary.csv", index=False)
    append_to_run_log(args.output_dir / "run_log.txt", rows)

    pool_dir = output_root / "pool"
    try:
        make_pool_figure(
            output_root=output_root,
            facet_by="biome",
            output_path=pool_dir / "pool_by_biome.png",
            sm_variant=args.sm_variant,
        )
        make_pool_figure(
            output_root=output_root,
            facet_by="PFT",
            output_path=pool_dir / "pool_by_pft.png",
            sm_variant=args.sm_variant,
        )
    except Exception:
        import traceback

        logger.error("Pool figure failed:\n%s", traceback.format_exc())

    ok_count = int((summary_df["status"] == STATUS_OK).sum())
    logger.info("Finished. OK=%d of %d.", ok_count, len(rows))
    return 0 if ok_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
