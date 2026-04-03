"""mlxtend SFS wrapper for forward feature selection.

Orchestrates: load cache -> build estimator pipeline -> run SFS -> save results.
After SFS completes, true pooled R2/RMSE are computed via cross_val_predict
for every step (concatenate all OOF predictions, then score).

Heavy dependencies (xgboost, sklearn.model_selection, mlxtend) are imported
inside the functions that need them so that _extract_results / _save_results
remain importable without those packages (enables local testing).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.forward_selection.config import HyperparamConfig, SelectionConfig
from src.forward_selection.feature_registry import (
    build_feature_groups,
    get_candidate_group_names,
)
from src.forward_selection.scorers import get_scorer

logger = logging.getLogger(__name__)


def run_forward_selection(
    config: SelectionConfig,
    hyper: HyperparamConfig,
    cache_path: Path,
) -> dict[str, Any]:
    """Run mlxtend sequential feature selection on cached feature data.

    Parameters
    ----------
    config : SelectionConfig
        Selection run configuration (scoring mode, CV, parallelism).
    hyper : HyperparamConfig
        Fixed XGBoost hyperparameters.
    cache_path : Path
        Path to the .npz feature cache produced by ``data_loader``.

    Returns
    -------
    dict
        Results including selected features, mean-fold scores, true pooled
        R2/RMSE at each step, and timing information.
    """
    # Heavy imports — deferred so _extract_results/_save_results stay light
    from mlxtend.feature_selection import SequentialFeatureSelector
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    from src.forward_selection.data_loader import load_cache

    # 1. Load cached data
    data = load_cache(cache_path)
    x_all = data["X"]
    y = data["y"]
    groups = data["groups"]
    pfts_encoded = data["pfts_encoded"]
    feature_names = data["feature_names"]

    logger.info("Data loaded: X=%s, scoring=%s", x_all.shape, config.scoring.value)

    # 2. Apply log1p transform to target
    y = np.log1p(y)

    # 3. Build feature groups (column indices)
    mandatory_idx, candidate_groups = build_feature_groups(feature_names)
    candidate_names = get_candidate_group_names(feature_names)

    logger.info(
        "Mandatory features (%d): indices %s",
        len(mandatory_idx),
        mandatory_idx,
    )
    logger.info(
        "Candidate groups (%d): %s",
        len(candidate_groups),
        candidate_names[:10],
    )

    # 4. Build estimator pipeline
    xgb_params = hyper.to_xgb_params(n_jobs=config.n_jobs_xgb, random_state=config.random_seed)
    estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("xgb", XGBRegressor(**xgb_params)),
        ]
    )

    # 5. Precompute CV splits — stratify by PFT, group by spatial cluster.
    #    mlxtend passes the regression target y to cv.split(), but
    #    StratifiedGroupKFold needs PFT categories for stratification.
    #    Precomputing materialises the folds once; sklearn's cross_val_score
    #    accepts list[(train_idx, test_idx)] and reuses them every call.
    cv_splitter = StratifiedGroupKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_seed,
    )
    cv_splits = list(cv_splitter.split(x_all, pfts_encoded, groups))
    logger.info("Precomputed %d CV splits (stratified by PFT, grouped by spatial)", len(cv_splits))

    # 6. Get scorer
    scorer = get_scorer(config.scoring)

    # 7. Run mlxtend SFS
    total_features = len(mandatory_idx) + len(candidate_groups)
    k_feat = config.k_features
    if k_feat == "best":
        k_feat = (len(mandatory_idx) + 1, total_features)

    sfs = SequentialFeatureSelector(
        estimator,
        k_features=k_feat,
        forward=config.forward,
        floating=config.floating,
        scoring=scorer,
        cv=cv_splits,
        n_jobs=config.n_jobs_sfs,
        feature_groups=candidate_groups,
        fixed_features=list(mandatory_idx),
        verbose=2,
    )

    logger.info("Starting SFS (k_features=%s) ...", config.k_features)
    t0 = time.time()
    sfs.fit(x_all, y)
    sfs_elapsed = time.time() - t0
    logger.info("SFS completed in %.1f minutes", sfs_elapsed / 60)

    # 8. Compute true pooled R2/RMSE for every step
    logger.info("Computing true pooled metrics for each step ...")
    t1 = time.time()
    pooled_metrics = _compute_pooled_metrics(
        sfs,
        x_all,
        y,
        cv_splits,
        estimator,
        config.n_jobs_sfs,
    )
    pooled_elapsed = time.time() - t1
    logger.info("Pooled metrics completed in %.1f minutes", pooled_elapsed / 60)

    # 9. Extract results
    total_elapsed = sfs_elapsed + pooled_elapsed
    results = _extract_results(sfs, feature_names, config, total_elapsed, pooled_metrics)

    # 10. Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_results(results, output_dir, config.scoring.value)

    return results


def _compute_pooled_metrics(
    sfs: Any,
    x_all: np.ndarray,
    y: np.ndarray,
    cv_splits: list,
    estimator: Any,
    n_jobs: int,
) -> dict[int, dict[str, float]]:
    """Compute true pooled R2 and RMSE for each SFS step.

    Uses cross_val_predict to collect OOF predictions across all folds,
    then computes metrics on the concatenated result — matching the
    training script's pooled evaluation.
    """
    from sklearn.base import clone
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import cross_val_predict

    pooled: dict[int, dict[str, float]] = {}

    for step, info in sfs.subsets_.items():
        feature_idx = list(info["feature_idx"])
        x_subset = x_all[:, feature_idx]

        y_pred_oof = cross_val_predict(
            clone(estimator),
            x_subset,
            y,
            cv=cv_splits,
            n_jobs=n_jobs,
        )

        pr2 = float(r2_score(y, y_pred_oof))
        prmse = float(np.sqrt(mean_squared_error(y, y_pred_oof)))
        pooled[step] = {"pooled_r2": pr2, "pooled_rmse": prmse}

        logger.info(
            "  Step %d (%d features): pooled_r2=%.4f, pooled_rmse=%.4f",
            step,
            len(feature_idx),
            pr2,
            prmse,
        )

    return pooled


def _extract_results(
    sfs: Any,
    feature_names: list[str],
    config: SelectionConfig,
    elapsed_seconds: float,
    pooled_metrics: dict[int, dict[str, float]],
) -> dict[str, Any]:
    """Extract structured results from a fitted SFS object."""
    selected_idx = list(sfs.k_feature_idx_)
    selected_names = [feature_names[i] for i in selected_idx]

    step_scores: dict[int, float] = {}
    step_features: dict[int, list[str]] = {}
    step_pooled_r2: dict[int, float] = {}
    step_pooled_rmse: dict[int, float] = {}

    for step, info in sfs.subsets_.items():
        step_scores[step] = info["avg_score"]
        step_features[step] = [feature_names[i] for i in info["feature_idx"]]
        if step in pooled_metrics:
            step_pooled_r2[step] = pooled_metrics[step]["pooled_r2"]
            step_pooled_rmse[step] = pooled_metrics[step]["pooled_rmse"]

    return {
        "scoring": config.scoring.value,
        "best_score": float(sfs.k_score_),
        "n_selected": len(selected_names),
        "selected_features": selected_names,
        "selected_indices": selected_idx,
        "step_scores": step_scores,
        "step_pooled_r2": step_pooled_r2,
        "step_pooled_rmse": step_pooled_rmse,
        "step_features": step_features,
        "elapsed_minutes": elapsed_seconds / 60,
        "config": {
            "n_splits": config.n_splits,
            "random_seed": config.random_seed,
            "k_features": config.k_features,
            "forward": config.forward,
            "floating": config.floating,
        },
    }


def _save_results(
    results: dict[str, Any],
    output_dir: Path,
    scoring_name: str,
) -> None:
    """Save selection results to JSON and numpy files."""
    json_path = output_dir / f"ffs_{scoring_name}_results.json"

    serializable = dict(results)
    for key in ("step_scores", "step_pooled_r2", "step_pooled_rmse", "step_features"):
        serializable[key] = {str(k): v for k, v in results[key].items()}

    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Results saved to %s", json_path)

    sorted_steps = sorted(results["step_scores"])
    np.savez(
        output_dir / f"ffs_{scoring_name}_arrays.npz",
        selected_indices=np.array(results["selected_indices"]),
        step_scores=np.array([results["step_scores"][k] for k in sorted_steps]),
        step_pooled_r2=np.array([results["step_pooled_r2"][k] for k in sorted_steps]),
        step_pooled_rmse=np.array([results["step_pooled_rmse"][k] for k in sorted_steps]),
    )
