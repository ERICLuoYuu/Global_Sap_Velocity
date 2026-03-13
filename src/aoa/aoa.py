"""Step 7: AOA orchestrator — ties together standardization, weighting, DI, and threshold.

Supports chunked processing for large prediction grids.
"""

from typing import Any

import numpy as np

from src.aoa.dissimilarity import (
    compute_di,
    compute_mean_training_distance,
    standardize_predictors,
    weight_by_importance,
)
from src.aoa.threshold import compute_cv_training_di, compute_threshold


def compute_aoa(
    train_X: np.ndarray,
    new_X: np.ndarray,
    weights: np.ndarray,
    fold_indices: list[np.ndarray],
    *,
    chunk_size: int = 10_000,
) -> dict[str, Any]:
    """Compute Area of Applicability following Meyer & Pebesma (2021).

    Parameters
    ----------
    train_X : (n_train, n_features) raw training features
    new_X : (n_new, n_features) raw prediction features
    weights : (n_features,) variable importance weights
    fold_indices : list of index arrays, one per CV fold
    chunk_size : max points per batch for new_X (0 = no chunking)

    Returns
    -------
    dict with keys:
        di : (n_new,) Dissimilarity Index
        aoa : (n_new,) boolean mask (True = inside AOA)
        threshold : float
        d_bar : float
        di_train : (n_train,) training DI values
    """
    # Step 1: Standardize
    train_scaled, new_scaled, _stats = standardize_predictors(train_X, new_X)

    # Step 2: Weight by importance
    train_weighted = weight_by_importance(train_scaled, weights)
    new_weighted = weight_by_importance(new_scaled, weights)

    # Step 3: Mean pairwise training distance
    d_bar = compute_mean_training_distance(train_weighted)

    # Steps 5-6: Training DI and threshold (computed before new-data DI
    # since threshold is independent of prediction points)
    di_train = compute_cv_training_di(train_weighted, fold_indices, d_bar)
    threshold = compute_threshold(di_train)

    # Step 4: DI for new points (optionally chunked)
    if chunk_size <= 0 or new_weighted.shape[0] <= chunk_size:
        di = compute_di(new_weighted, train_weighted, d_bar)
    else:
        n_new = new_weighted.shape[0]
        di = np.empty(n_new)
        for start in range(0, n_new, chunk_size):
            end = min(start + chunk_size, n_new)
            di[start:end] = compute_di(new_weighted[start:end], train_weighted, d_bar)

    # Step 7: Apply threshold
    aoa_mask = di <= threshold

    return {
        "di": di,
        "aoa": aoa_mask,
        "threshold": threshold,
        "d_bar": d_bar,
        "di_train": di_train,
    }
