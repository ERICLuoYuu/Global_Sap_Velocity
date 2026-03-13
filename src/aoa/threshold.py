"""Steps 5-6 of Meyer & Pebesma (2021) AOA: CV-based training DI and threshold.

5. Compute training DI using cross-validation (out-of-fold neighbors only)
6. Derive AOA threshold from the training DI distribution
"""

import numpy as np
from scipy.spatial import KDTree


def compute_cv_training_di(
    train_weighted: np.ndarray,
    fold_indices: list[np.ndarray],
    d_bar: float,
) -> np.ndarray:
    """Compute DI for each training point using only out-of-fold neighbors.

    For each fold, points in that fold compute their nearest-neighbor
    distance to all points NOT in that fold, then divide by d_bar.

    Parameters
    ----------
    train_weighted : (n_train, n_features) weighted training matrix
    fold_indices : list of arrays, each containing indices for one CV fold
    d_bar : mean pairwise training distance

    Returns
    -------
    di_train : (n_train,) DI for each training point

    Raises
    ------
    ValueError if fewer than 2 folds provided.
    """
    if len(fold_indices) < 2:
        raise ValueError("At least 2 CV folds required for training DI computation.")

    n_train = train_weighted.shape[0]

    # Collect (index_array, di_array) pairs per fold — no mutation
    fold_results: list[tuple[np.ndarray, np.ndarray]] = []

    for fold_idx in fold_indices:
        oof_mask = np.ones(n_train, dtype=bool)
        oof_mask[fold_idx] = False
        oof_points = train_weighted[oof_mask]

        if oof_points.shape[0] == 0:
            raise ValueError("A fold contains all training points; out-of-fold set is empty.")

        tree = KDTree(oof_points)
        fold_points = train_weighted[fold_idx]
        min_dists, _ = tree.query(fold_points, k=1)

        if d_bar == 0.0:
            fold_di = np.where(min_dists == 0.0, 0.0, np.inf)
        else:
            fold_di = min_dists / d_bar

        fold_results.append((fold_idx, fold_di))

    # Assemble results into a single array (single write, no incremental mutation)
    di_train = np.full(n_train, np.nan)
    for fold_idx, fold_di in fold_results:
        di_train[fold_idx] = fold_di

    if np.any(np.isnan(di_train)):
        raise ValueError(
            "fold_indices do not cover all training points; some positions were never assigned a DI value."
        )
    return di_train


def compute_threshold(di_train: np.ndarray) -> float:
    """Compute AOA threshold: Q75 + 1.5 * IQR of training DI values.

    Parameters
    ----------
    di_train : (n_train,) DI values for training points

    Returns
    -------
    threshold : scalar

    Raises
    ------
    ValueError if di_train is empty.
    """
    if len(di_train) == 0:
        raise ValueError("Cannot compute threshold from empty training DI array.")

    q25 = np.percentile(di_train, 25)
    q75 = np.percentile(di_train, 75)
    iqr = q75 - q25
    return float(q75 + 1.5 * iqr)
