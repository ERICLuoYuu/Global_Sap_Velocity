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

    # 2. Apply log1p transform to target (keep original for metrics)
    y_orig = y.copy()
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

    # 3b. Build complete feature_groups for mlxtend (must cover ALL indices).
    #     Mandatory features become individual groups; fixed_features locks them.
    mandatory_groups = [[idx] for idx in mandatory_idx]
    all_groups = mandatory_groups + candidate_groups

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
    total_groups = len(all_groups)
    k_feat = config.k_features
    if k_feat == "best":
        # min = mandatory only + 1 candidate; max = all groups
        k_feat = (len(mandatory_idx) + 1, total_groups)

    sfs = SequentialFeatureSelector(
        estimator,
        k_features=k_feat,
        forward=config.forward,
        floating=config.floating,
        scoring=scorer,
        cv=cv_splits,
        n_jobs=config.n_jobs_sfs,
        feature_groups=all_groups,
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
        y_orig=y_orig,
        # avg_score is only an R2 to compare against when SFS scored with r2;
        # under neg_rmse it is a negative RMSE and the check would misfire.
        check_avg_score=config.scoring.value == "mean_r2",
    )
    pooled_elapsed = time.time() - t1
    logger.info("Pooled metrics completed in %.1f minutes", pooled_elapsed / 60)

    # 9. Extract results
    total_elapsed = sfs_elapsed + pooled_elapsed
    results = _extract_results(
        sfs,
        feature_names,
        config,
        total_elapsed,
        pooled_metrics,
        pooled_scale="original" if y_orig is not None else "log1p",
    )

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
    y_orig: np.ndarray | None = None,
    check_avg_score: bool = False,
) -> dict[int, dict[str, float]]:
    """Compute pooled AND mean-fold R2/RMSE, on BOTH scales, for each SFS step.

    One ``cross_val_predict`` yields the out-of-fold vector; every metric below
    is arithmetic on top of it, so reporting four numbers costs no more compute
    than reporting one.

    Reporting all four is the point. mlxtend fits and scores on ``log1p(y)``, so
    ``avg_score`` is R2 on the log scale, while the pooled metrics here are
    back-transformed with ``expm1`` and scored on the original scale to match the
    training script. Those two used to be the only numbers stored, under adjacent
    names, and comparing them mixed a scale change with an aggregation change: on
    the real corpus 0.109 of a 0.164 apparent gap was scale, two thirds of it,
    while reading as a cross-validation effect.

    Parameters
    ----------
    y_orig : array, optional
        Original-scale target (before log1p). When given, ``pooled_r2`` and
        ``mean_fold_r2_orig`` are on that scale. When omitted there is nothing
        to back-transform to, so they mirror their ``_log`` counterparts.
    check_avg_score : bool
        Warn when the reconstructed ``mean_fold_r2_log`` disagrees with the
        SFS ``avg_score``. Only meaningful when SFS scored with r2.
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

        entry = {
            "pooled_r2_log": float(r2_score(y, y_pred_oof)),
            "pooled_rmse_log": float(np.sqrt(mean_squared_error(y, y_pred_oof))),
            "mean_fold_r2_log": _mean_fold_r2(y, y_pred_oof, cv_splits),
        }
        if y_orig is not None:
            # expm1 of a log-scale prediction is a biased estimate of the
            # conditional mean, so these are NOT a rescaling of the _log values.
            y_pred_orig = np.expm1(y_pred_oof)
            entry["pooled_r2"] = float(r2_score(y_orig, y_pred_orig))
            entry["pooled_rmse"] = float(np.sqrt(mean_squared_error(y_orig, y_pred_orig)))
            entry["mean_fold_r2_orig"] = _mean_fold_r2(y_orig, y_pred_orig, cv_splits)
        else:
            entry["pooled_r2"] = entry["pooled_r2_log"]
            entry["pooled_rmse"] = entry["pooled_rmse_log"]
            entry["mean_fold_r2_orig"] = entry["mean_fold_r2_log"]
        pooled[step] = entry

        # mean_fold_r2_log is rebuilt from the OOF vector over the same folds
        # mlxtend scored, so it has to reproduce avg_score. If it does not, the
        # folds here are not the folds the selection ran on and every pooled
        # number describes a different experiment. Warn rather than raise: this
        # runs at the very end of a multi-hour job that is otherwise complete.
        avg = info.get("avg_score") if check_avg_score else None
        if avg is not None and abs(entry["mean_fold_r2_log"] - float(avg)) > 1e-4:
            logger.warning(
                "  Step %d: mean_fold_r2_log %.6f != SFS avg_score %.6f — fold "
                "reconstruction disagrees with the selection run",
                step,
                entry["mean_fold_r2_log"],
                float(avg),
            )

        logger.info(
            "  Step %d (%d features): pooled_r2=%.4f [orig] pooled_r2_log=%.4f "
            "mean_fold_r2_orig=%.4f mean_fold_r2_log=%.4f pooled_rmse=%.4f [orig]",
            step,
            len(feature_idx),
            entry["pooled_r2"],
            entry["pooled_r2_log"],
            entry["mean_fold_r2_orig"],
            entry["mean_fold_r2_log"],
            entry["pooled_rmse"],
        )

    return pooled


def _mean_fold_r2(y_true: np.ndarray, y_pred: np.ndarray, cv_splits: list) -> float:
    """Unweighted mean of per-fold R2 — the aggregation sklearn's scorer uses.

    Each fold is scored against ITS OWN mean and the folds are averaged with
    equal weight whatever their size. Pooled R2 instead scores one concatenated
    vector against the global mean, which both weights each fold by its share of
    the total sum of squares and adds the between-fold term
    ``sum_k n_k (ybar_k - ybar)^2`` to the denominator. Stratified CV shrinks
    that term, so here the weighting dominates and pooled lands BELOW the
    per-fold mean rather than above it.
    """
    from sklearn.metrics import r2_score

    return float(np.mean([r2_score(y_true[test], y_pred[test]) for _, test in cv_splits]))


def _extract_results(
    sfs: Any,
    feature_names: list[str],
    config: SelectionConfig,
    elapsed_seconds: float,
    pooled_metrics: dict[int, dict[str, float]],
    pooled_scale: str = "original",
) -> dict[str, Any]:
    """Extract structured results from a fitted SFS object.

    ``metric_scales`` records, per metric key, which target it was computed on,
    so nothing downstream has to infer it from the name. Without it the file
    invites comparing ``step_scores`` (log1p) with ``step_pooled_r2``
    (original) as though they were two aggregations of one quantity.
    """
    selected_idx = list(sfs.k_feature_idx_)
    selected_names = [feature_names[i] for i in selected_idx]

    step_scores: dict[int, float] = {}
    step_features: dict[int, list[str]] = {}
    per_metric: dict[str, dict[int, float]] = {
        "pooled_r2": {},
        "pooled_rmse": {},
        "pooled_r2_log": {},
        "pooled_rmse_log": {},
        "mean_fold_r2_log": {},
        "mean_fold_r2_orig": {},
    }

    for step, info in sfs.subsets_.items():
        step_scores[step] = info["avg_score"]
        step_features[step] = [feature_names[i] for i in info["feature_idx"]]
        # Key by key rather than wholesale: results written before the
        # both-scales change carry only pooled_r2/pooled_rmse, and those files
        # must still extract instead of raising KeyError.
        metrics = pooled_metrics.get(step) or {}
        for name, target in per_metric.items():
            if name in metrics:
                target[step] = metrics[name]

    return {
        "scoring": config.scoring.value,
        "best_score": float(sfs.k_score_),
        "n_selected": len(selected_names),
        "selected_features": selected_names,
        "selected_indices": selected_idx,
        "step_scores": step_scores,
        "step_pooled_r2": per_metric["pooled_r2"],
        "step_pooled_rmse": per_metric["pooled_rmse"],
        "step_pooled_r2_log": per_metric["pooled_r2_log"],
        "step_pooled_rmse_log": per_metric["pooled_rmse_log"],
        "step_mean_fold_r2_log": per_metric["mean_fold_r2_log"],
        "step_mean_fold_r2_orig": per_metric["mean_fold_r2_orig"],
        "step_features": step_features,
        "metric_scales": {
            # SFS fits on log1p(y), so its own score is on that scale.
            "step_scores": "log1p",
            "step_pooled_r2": pooled_scale,
            "step_pooled_rmse": pooled_scale,
            "step_pooled_r2_log": "log1p",
            "step_pooled_rmse_log": "log1p",
            "step_mean_fold_r2_log": "log1p",
            "step_mean_fold_r2_orig": "original",
        },
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
    # .get, not [], so a results dict assembled before the both-scales change
    # (or trimmed by a caller) still serialises instead of raising.
    for key in (
        "step_scores",
        "step_pooled_r2",
        "step_pooled_rmse",
        "step_pooled_r2_log",
        "step_pooled_rmse_log",
        "step_mean_fold_r2_log",
        "step_mean_fold_r2_orig",
        "step_features",
    ):
        stepwise = results.get(key)
        if stepwise is not None:
            serializable[key] = {str(k): v for k, v in stepwise.items()}

    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Results saved to %s", json_path)

    sorted_steps = sorted(results["step_scores"])

    def _column(key: str) -> np.ndarray:
        """Stepwise metric as an array, or empty if it is not present for every step.

        The npz is positional, so a partially populated metric would silently
        misalign against step_scores. Emit nothing rather than something wrong.
        """
        stepwise = results.get(key) or {}
        if not all(step in stepwise for step in sorted_steps):
            return np.array([])
        return np.array([stepwise[step] for step in sorted_steps])

    np.savez(
        output_dir / f"ffs_{scoring_name}_arrays.npz",
        selected_indices=np.array(results["selected_indices"]),
        step_scores=np.array([results["step_scores"][k] for k in sorted_steps]),
        step_pooled_r2=_column("step_pooled_r2"),
        step_pooled_rmse=_column("step_pooled_rmse"),
        step_pooled_r2_log=_column("step_pooled_r2_log"),
        step_pooled_rmse_log=_column("step_pooled_rmse_log"),
        step_mean_fold_r2_log=_column("step_mean_fold_r2_log"),
        step_mean_fold_r2_orig=_column("step_mean_fold_r2_orig"),
    )
