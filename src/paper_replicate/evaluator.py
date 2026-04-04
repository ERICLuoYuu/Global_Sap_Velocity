"""Evaluation metrics for Loritz et al. (2024) replication.

Ports their Cell 19-21: KGE/NSE/MAE computation with hydroeval,
per-genus aggregation, and overall metrics.
"""

import numpy as np
import pandas as pd

try:
    import hydroeval as he
except ImportError:
    he = None

try:
    import sklearn.metrics as metrics
except ImportError:
    metrics = None


def compute_kge(pred: np.ndarray, obs: np.ndarray) -> tuple[float, float, float, float]:
    """Compute KGE using hydroeval (matching their Cell 21).

    Returns:
        kge: Kling-Gupta Efficiency
        r: Pearson correlation
        alpha: variability ratio (std_pred / std_obs)
        beta: bias ratio (mean_pred / mean_obs)
    """
    if he is None:
        raise ImportError("hydroeval is required: pip install hydroeval")
    if len(pred) == 0 or len(obs) == 0:
        return np.nan, np.nan, np.nan, np.nan

    result = he.kge(pred, obs)
    # hydroeval returns [[kge], [r], [alpha], [beta]]
    kge_val = float(result[0][0])
    r_val = float(result[1][0])
    alpha_val = float(result[2][0])
    beta_val = float(result[3][0])
    return kge_val, r_val, alpha_val, beta_val


def compute_nse(pred: np.ndarray, obs: np.ndarray) -> float:
    """Compute NSE using hydroeval."""
    if he is None:
        raise ImportError("hydroeval is required: pip install hydroeval")
    if len(pred) == 0 or len(obs) == 0:
        return np.nan
    result = he.nse(pred, obs)
    return float(result[0]) if hasattr(result, "__len__") else float(result)


def compute_mae(pred: np.ndarray, obs: np.ndarray) -> float:
    """Compute MAE using sklearn."""
    if metrics is None:
        raise ImportError("scikit-learn is required: pip install scikit-learn")
    if len(pred) == 0 or len(obs) == 0:
        return np.nan
    return float(metrics.mean_absolute_error(pred, obs))


def evaluate_per_genus(
    pred_dict: dict[str, list[np.ndarray]],
    obs_dict: dict[str, list[np.ndarray]],
    plant_list: list[str],
) -> pd.DataFrame:
    """Evaluate per-genus metrics (matching their Cell 21).

    Args:
        pred_dict: genus_name -> list of prediction arrays
        obs_dict: genus_name -> list of observation arrays
        plant_list: list of genus names to evaluate

    Returns:
        DataFrame with columns: genus, n_trees, kge, r, alpha, beta, nse, mae
    """
    rows = []
    for genus in plant_list:
        if genus not in pred_dict:
            continue

        selected_pred = np.concatenate(pred_dict[genus])
        selected_obs = np.concatenate(obs_dict[genus])

        kge_val, r_val, alpha_val, beta_val = compute_kge(selected_pred, selected_obs)
        nse_val = compute_nse(selected_pred, selected_obs)
        mae_val = compute_mae(selected_pred, selected_obs)

        rows.append(
            {
                "genus": genus,
                "n_samples": len(selected_pred),
                "kge": kge_val,
                "r": r_val,
                "alpha": alpha_val,
                "beta": beta_val,
                "nse": nse_val,
                "mae": mae_val,
            }
        )

    return pd.DataFrame(rows)


def evaluate_overall(
    pred_dict: dict[str, list[np.ndarray]],
    obs_dict: dict[str, list[np.ndarray]],
    plant_list: list[str],
) -> dict[str, float]:
    """Evaluate overall metrics across all genera (their Cell 21 end).

    Returns dict with kge, r, alpha, beta, nse, mae.
    """
    all_pred = np.array([])
    all_obs = np.array([])

    for genus in plant_list:
        if genus in pred_dict:
            all_pred = np.concatenate([all_pred, np.concatenate(pred_dict[genus])])
            all_obs = np.concatenate([all_obs, np.concatenate(obs_dict[genus])])

    if len(all_pred) == 0:
        return {"kge": np.nan, "r": np.nan, "alpha": np.nan, "beta": np.nan, "nse": np.nan, "mae": np.nan}

    kge_val, r_val, alpha_val, beta_val = compute_kge(all_pred, all_obs)
    nse_val = compute_nse(all_pred, all_obs)
    mae_val = compute_mae(all_pred, all_obs)

    return {
        "kge": kge_val,
        "r": r_val,
        "alpha": alpha_val,
        "beta": beta_val,
        "nse": nse_val,
        "mae": mae_val,
        "n_samples": len(all_pred),
    }
