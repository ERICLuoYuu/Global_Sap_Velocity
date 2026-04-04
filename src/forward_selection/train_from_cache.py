"""Train XGBoost using FFS cache data directly.

Loads data from the FFS feature_cache.npz, selects only the FFS-chosen
features, and runs the same 10-fold StratifiedGroupKFold CV as the main
training script.  Guarantees exact data match with FFS evaluation.

Usage:
    python -m src.forward_selection.train_from_cache \
        --cache_path outputs/forward_selection/feature_cache.npz \
        --selected_features outputs/forward_selection/selected_features_neg_rmse.json \
        --run_id ffs_neg_rmse_cache \
        --output_dir outputs/models/xgb/ffs_neg_rmse_cache
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGBoost from FFS cache")
    p.add_argument("--cache_path", type=str, required=True, help="Path to feature_cache.npz")
    p.add_argument("--selected_features", type=str, required=True, help="Path to selected features JSON")
    p.add_argument("--run_id", type=str, default="ffs_cache", help="Run identifier")
    p.add_argument("--output_dir", type=str, default=None, help="Output directory for model + results")
    p.add_argument("--hyperparameters", type=str, default=None, help="Path to hyperparameters JSON")
    p.add_argument("--n_splits", type=int, default=10, help="Number of CV folds")
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--shap_sample_size", type=int, default=50000)
    p.add_argument("--n_jobs", type=int, default=48, help="XGBoost n_jobs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    # --- Load FFS cache ---
    logger.info("Loading FFS cache from %s", args.cache_path)
    cache = np.load(args.cache_path, allow_pickle=True)
    X_all = cache["X"]
    y_all = cache["y"]
    groups = cache["groups"]
    pfts_encoded = cache["pfts_encoded"]
    feature_names = list(cache["feature_names"])
    pft_categories = list(cache["pft_categories"])
    logger.info("Cache: %d samples, %d features, %d groups", X_all.shape[0], X_all.shape[1], len(np.unique(groups)))

    # --- Select FFS features ---
    with open(args.selected_features) as f:
        sf_data = json.load(f)
    sel_list = sf_data.get("selected_features", sf_data) if isinstance(sf_data, dict) else sf_data
    sel_set = set(sel_list)

    sel_idx = [i for i, fn in enumerate(feature_names) if fn in sel_set]
    sel_names = [feature_names[i] for i in sel_idx]
    missing = sel_set - set(sel_names)
    if missing:
        logger.warning("Features in JSON but not in cache: %s", sorted(missing))

    X = X_all[:, sel_idx].astype(np.float32)
    y = y_all.astype(np.float32)
    logger.info("Selected %d / %d features", X.shape[1], X_all.shape[1])
    del X_all  # free memory

    # --- Hyperparameters ---
    if args.hyperparameters:
        with open(args.hyperparameters) as f:
            hparams = json.load(f)
    else:
        hparams = {
            "n_estimators": 1000,
            "learning_rate": 0.01,
            "max_depth": 10,
            "min_child_weight": 7,
            "subsample": 0.67,
            "colsample_bytree": 0.7,
            "gamma": 0.3,
            "reg_alpha": 1.0,
            "reg_lambda": 10.0,
        }
    logger.info("Hyperparameters: %s", hparams)

    # --- Target transform ---
    y_log = np.log1p(y)

    # --- Output directory ---
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path("outputs/models/xgb") / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- CV ---
    cv = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_seed)

    oof_pred_log = np.full(len(y), np.nan)
    oof_pred_orig = np.full(len(y), np.nan)
    fold_r2s = []
    fold_rmses = []
    best_model = None
    best_scaler = None
    best_fold_r2 = -np.inf

    pft_cols_set = {"MF", "DNF", "ENF", "EBF", "WSA", "WET", "DBF", "SAV"}
    numeric_idx = np.array([i for i, c in enumerate(sel_names) if c not in pft_cols_set])

    logger.info("Starting %d-fold CV...", args.n_splits)
    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, pfts_encoded, groups)):
        fold_start = time.time()

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_log[train_idx], y_log[test_idx]

        # Scale only numeric features
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[:, numeric_idx] = scaler.fit_transform(X_train[:, numeric_idx])
        X_test_scaled[:, numeric_idx] = scaler.transform(X_test[:, numeric_idx])

        model = XGBRegressor(
            **hparams,
            n_jobs=args.n_jobs,
            random_state=args.random_seed,
            tree_method="hist",
        )
        model.fit(X_train_scaled, y_train)
        pred_log = model.predict(X_test_scaled)
        pred_orig = np.expm1(pred_log)
        y_orig_test = y[test_idx]

        fold_r2 = r2_score(y_orig_test, pred_orig)
        fold_rmse = np.sqrt(mean_squared_error(y_orig_test, pred_orig))
        fold_r2s.append(fold_r2)
        fold_rmses.append(fold_rmse)

        oof_pred_log[test_idx] = pred_log
        oof_pred_orig[test_idx] = pred_orig

        elapsed = time.time() - fold_start
        logger.info(
            "  Fold %d/%d: R2=%.4f, RMSE=%.4f (n_train=%d, n_test=%d, %.1fs)",
            fold_i + 1,
            args.n_splits,
            fold_r2,
            fold_rmse,
            len(train_idx),
            len(test_idx),
            elapsed,
        )

        # Save best model
        if fold_r2 > best_fold_r2:
            best_fold_r2 = fold_r2
            best_model = model
            best_scaler = scaler

        # Save per-fold model
        model.save_model(str(out_dir / f"spatial_xgb_fold_{fold_i + 1}_{args.run_id}.json"))

    # --- Aggregate metrics ---
    pooled_r2 = r2_score(y, oof_pred_orig)
    pooled_rmse = np.sqrt(mean_squared_error(y, oof_pred_orig))
    mean_r2 = np.mean(fold_r2s)
    std_r2 = np.std(fold_r2s)
    mean_rmse = np.mean(fold_rmses)
    std_rmse = np.std(fold_rmses)

    logger.info("=" * 60)
    logger.info("RESULTS (%d-fold CV on FFS cache)", args.n_splits)
    logger.info("  Pooled R2:  %.4f", pooled_r2)
    logger.info("  Mean R2:    %.4f +/- %.4f", mean_r2, std_r2)
    logger.info("  Pooled RMSE: %.4f", pooled_rmse)
    logger.info("  Mean RMSE:  %.4f +/- %.4f", mean_rmse, std_rmse)
    logger.info("  n_samples:  %d", len(y))
    logger.info("  n_features: %d", X.shape[1])
    logger.info("=" * 60)

    # --- Save final model + config ---
    joblib.dump(best_model, str(out_dir / f"FINAL_xgb_{args.run_id}.joblib"))
    joblib.dump(best_scaler, str(out_dir / f"FINAL_scaler_{args.run_id}_feature.pkl"))

    config = {
        "model_type": "xgb",
        "run_id": args.run_id,
        "best_params": hparams,
        "cv_results": {
            "mean_r2": float(mean_r2),
            "pooled_r2": float(pooled_r2),
            "std_r2": float(std_r2),
            "mean_rmse": float(mean_rmse),
            "pooled_rmse": float(pooled_rmse),
            "std_rmse": float(std_rmse),
            "n_folds": args.n_splits,
            "fold_r2s": [float(r) for r in fold_r2s],
            "fold_rmses": [float(r) for r in fold_rmses],
        },
        "data_info": {
            "n_samples": int(len(y)),
            "n_features": int(X.shape[1]),
            "source": "ffs_cache",
            "cache_path": str(args.cache_path),
        },
        "feature_names": sel_names,
        "random_seed": args.random_seed,
        "split_type": "spatial_stratified",
    }
    with open(out_dir / f"FINAL_config_{args.run_id}.json", "w") as f:
        json.dump(config, f, indent=2)

    # --- SHAP analysis ---
    try:
        import shap

        logger.info("Running SHAP analysis (sample_size=%d)...", args.shap_sample_size)
        sample_size = min(args.shap_sample_size, len(X))
        rng = np.random.RandomState(args.random_seed)
        shap_idx = rng.choice(len(X), sample_size, replace=False)
        X_shap = X[shap_idx].copy()
        X_shap[:, numeric_idx] = best_scaler.transform(X_shap[:, numeric_idx])

        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_shap)
        np.save(str(out_dir / f"shap_values_{args.run_id}.npy"), shap_values)

        # Feature importance summary
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance = sorted(zip(sel_names, mean_abs_shap), key=lambda x: -x[1])
        logger.info("Top 15 SHAP features:")
        for name, val in importance[:15]:
            logger.info("  %-30s %.4f", name, val)

        with open(out_dir / f"shap_importance_{args.run_id}.json", "w") as f:
            json.dump([{"feature": n, "mean_abs_shap": float(v)} for n, v in importance], f, indent=2)
    except ImportError:
        logger.warning("shap not available, skipping SHAP analysis")

    elapsed_total = time.time() - t0
    logger.info("Total time: %.1f seconds", elapsed_total)
    logger.info("All outputs saved to %s", out_dir)


if __name__ == "__main__":
    main()
