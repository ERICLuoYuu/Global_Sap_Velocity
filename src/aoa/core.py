"""Pure AOA algorithm functions — no I/O, no side effects.

All functions are stateless: arrays in, arrays out.
Reference: Meyer & Pebesma (2021), Section 2.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist


def standardize_features(X: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Z-score standardize features. Zero-variance columns -> 0."""
    safe_stds = np.where(stds == 0, 1.0, stds)
    result = (X - means) / safe_stds
    result[:, stds == 0] = 0.0
    return result


def apply_importance_weights(X_standardized: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Multiply standardized features by importance weights."""
    return X_standardized * weights[np.newaxis, :]


def compute_d_bar_full(X_weighted: np.ndarray) -> float:
    """Mean pairwise Euclidean distance in weighted space.

    Raises ValueError if < 2 points or d_bar == 0.
    """
    if X_weighted.shape[0] < 2:
        raise ValueError("Need >= 2 training points for d_bar")
    distances = pdist(X_weighted, metric="euclidean")
    d_bar = float(np.mean(distances))
    if d_bar == 0.0:
        raise ValueError("d_bar is zero — all training points identical in weighted space")
    return d_bar


def compute_d_bar_sample(X_weighted: np.ndarray, sample_size: int, seed: int = 42) -> float:
    """Approximate d_bar via random subsample.

    Falls back to full if sample_size >= N.
    """
    n = X_weighted.shape[0]
    if sample_size >= n:
        return compute_d_bar_full(X_weighted)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=sample_size, replace=False)
    return compute_d_bar_full(X_weighted[idx])


def compute_training_di(X_weighted: np.ndarray, fold_labels: np.ndarray, d_bar: float) -> np.ndarray:
    """Training DI: nearest neighbor in DIFFERENT fold, normalized by d_bar."""
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


def compute_threshold(training_di: np.ndarray, iqr_multiplier: float = 1.5) -> float:
    """Outlier-removed max of training DI.

    Formula: max(DI[DI <= Q75 + iqr_multiplier * IQR])
    """
    q25, q75 = np.percentile(training_di, [25, 75])
    iqr = q75 - q25
    upper_whisker = q75 + iqr_multiplier * iqr
    below_whisker = training_di[training_di <= upper_whisker]
    return float(np.max(below_whisker))


def compute_prediction_di(X_new_weighted: np.ndarray, tree: cKDTree, d_bar: float) -> np.ndarray:
    """Prediction DI: nearest training neighbor, normalized by d_bar."""
    distances, _ = tree.query(X_new_weighted, k=1)
    return distances / d_bar


def build_kdtree(X_weighted: np.ndarray) -> cKDTree:
    """Build cKDTree on weighted training data."""
    return cKDTree(X_weighted)
