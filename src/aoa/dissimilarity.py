"""Steps 1-4 of Meyer & Pebesma (2021) AOA: standardize, weight, compute DI.

1. Standardize predictors using training-only statistics
2. Weight by variable importance
3. Compute mean pairwise training distance (d_bar)
4. Compute Dissimilarity Index for new points
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist


@dataclass(frozen=True)
class StandardizationStats:
    """Immutable record of training standardization statistics."""

    mean: np.ndarray
    std: np.ndarray


def standardize_predictors(
    train_X: np.ndarray,
    new_X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardizationStats]:
    """Standardize both arrays using training-only mean and std.

    Zero-variance predictors are set to 0 in both arrays.

    Parameters
    ----------
    train_X : (n_train, n_features) training feature matrix
    new_X : (n_new, n_features) new/prediction feature matrix

    Returns
    -------
    train_scaled, new_scaled, stats
    """
    if np.any(np.isnan(train_X)):
        raise ValueError("train_X contains NaN values. Impute or drop before computing AOA.")
    if np.any(np.isnan(new_X)):
        raise ValueError("new_X contains NaN values. Impute or drop before computing AOA.")

    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0, ddof=1)

    # Handle zero-variance columns: use np.where to avoid in-place mutation
    zero_var = std == 0.0
    safe_std = np.where(zero_var, 1.0, std)

    train_scaled = np.where(zero_var, 0.0, (train_X - mean) / safe_std)
    new_scaled = np.where(zero_var, 0.0, (new_X - mean) / safe_std)

    stats = StandardizationStats(mean=mean, std=std)
    return train_scaled, new_scaled, stats


def weight_by_importance(
    X: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Multiply each feature column by its importance weight.

    Parameters
    ----------
    X : (n_samples, n_features)
    weights : (n_features,) importance weights

    Returns
    -------
    Weighted copy of X.

    Raises
    ------
    ValueError if weights length != number of features.
    """
    if weights.shape[0] != X.shape[1]:
        msg = f"Weight vector length ({weights.shape[0]}) must match number of features ({X.shape[1]})"
        raise ValueError(msg)
    return X * weights


_MAX_PDIST_SAMPLES = 5_000


def compute_mean_training_distance(train_weighted: np.ndarray, *, seed: int = 0) -> float:
    """Compute mean pairwise Euclidean distance among training points (d_bar).

    For large training sets (> 5000 points), uses a random subsample to
    avoid O(n²) memory from scipy.spatial.distance.pdist.

    Parameters
    ----------
    train_weighted : (n_train, n_features) weighted training matrix
    seed : random seed for subsampling (deterministic by default)

    Returns
    -------
    d_bar : mean pairwise distance (scalar)
    """
    n = train_weighted.shape[0]
    if n > _MAX_PDIST_SAMPLES:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=_MAX_PDIST_SAMPLES, replace=False)
        sample = train_weighted[idx]
    else:
        sample = train_weighted
    distances = pdist(sample, metric="euclidean")
    return float(np.mean(distances)) if len(distances) > 0 else 0.0


def compute_di(
    new_weighted: np.ndarray,
    train_weighted: np.ndarray,
    d_bar: float,
) -> np.ndarray:
    """Compute Dissimilarity Index for new points.

    DI_k = min_i(dist(new_k, train_i)) / d_bar

    Uses a KD-tree for efficient nearest-neighbor lookup.

    Parameters
    ----------
    new_weighted : (n_new, n_features)
    train_weighted : (n_train, n_features)
    d_bar : mean pairwise training distance

    Returns
    -------
    di : (n_new,) dissimilarity index per new point
    """
    if new_weighted.ndim != 2:
        raise ValueError(f"new_weighted must be 2-D (n_samples, n_features), got {new_weighted.ndim}-D")

    tree = KDTree(train_weighted)
    min_dists, _ = tree.query(new_weighted, k=1)

    if d_bar == 0.0:
        return np.where(min_dists == 0.0, 0.0, np.inf)

    return min_dists / d_bar
