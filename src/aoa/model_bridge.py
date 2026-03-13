"""Bridge between saved model artifacts and AOA computation.

Loads model config, SHAP feature importances, and reconstructs
CV fold indices needed by compute_aoa().
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

_REQUIRED_CONFIG_KEYS = {"feature_names", "random_seed", "split_type", "cv_results"}


def load_model_config(config_path: Path) -> dict:
    """Load and validate model config JSON.

    Parameters
    ----------
    config_path : path to FINAL_config_{run_id}.json

    Returns
    -------
    config dict

    Raises
    ------
    FileNotFoundError if path doesn't exist.
    ValueError if required keys are missing.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    missing = _REQUIRED_CONFIG_KEYS - set(config.keys())
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    return config


def load_shap_weights(
    csv_path: Path,
    feature_names: list[str],
) -> np.ndarray:
    """Load SHAP feature importances and align to feature_names order.

    Normalizes weights to sum to 1.

    Parameters
    ----------
    csv_path : path to shap_feature_importance.csv
    feature_names : ordered list of feature names from model config

    Returns
    -------
    weights : (n_features,) normalized importance weights

    Raises
    ------
    ValueError if CSV is missing features present in feature_names.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    available = set(df["feature"].values)
    required = set(feature_names)
    missing = required - available
    if missing:
        raise ValueError(f"SHAP CSV missing features: {missing}")

    importance_map = dict(zip(df["feature"], df["importance"]))
    weights = np.array([importance_map[f] for f in feature_names], dtype=np.float64)

    total = weights.sum()
    if total > 0:
        weights = weights / total

    return weights


def reconstruct_fold_indices(
    n_samples: int,
    groups: np.ndarray,
    pfts_encoded: np.ndarray,
    n_folds: int,
    random_seed: int,
    split_type: str,
) -> list[np.ndarray]:
    """Reconstruct CV fold test indices from spatial groups and PFTs.

    Uses the same sklearn splitter and seed as training to reproduce
    identical fold assignments.

    Parameters
    ----------
    n_samples : total number of training records
    groups : (n_samples,) spatial group ID per record
    pfts_encoded : (n_samples,) encoded PFT label per record
    n_folds : number of CV folds
    random_seed : random seed used during training
    split_type : 'spatial_stratified' or other

    Returns
    -------
    fold_indices : list of n_folds arrays, each containing test indices
    """
    X_dummy = np.zeros((n_samples, 1))

    if split_type == "spatial_stratified":
        cv = StratifiedGroupKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_seed,
        )
        splits = cv.split(X_dummy, pfts_encoded, groups)
    else:
        cv = GroupKFold(n_splits=n_folds)
        splits = cv.split(X_dummy, groups=groups)

    return [test_idx for _train_idx, test_idx in splits]


def save_aoa_arrays(
    model_dir: Path,
    run_id: str,
    X_all_records: np.ndarray,
    groups_all_records: np.ndarray,
    pfts_encoded: np.ndarray,
) -> None:
    """Save arrays needed for AOA computation alongside model artifacts.

    Call this from the training script after final model is saved.

    Parameters
    ----------
    model_dir : directory containing FINAL_*.joblib etc.
    run_id : model run identifier
    X_all_records : (n_samples, n_features) raw training features
    groups_all_records : (n_samples,) spatial group IDs
    pfts_encoded : (n_samples,) integer-encoded PFT labels
    """
    model_dir = Path(model_dir)
    np.save(model_dir / f"aoa_X_train_{run_id}.npy", X_all_records)
    np.save(model_dir / f"aoa_groups_{run_id}.npy", groups_all_records)
    np.save(model_dir / f"aoa_pfts_encoded_{run_id}.npy", pfts_encoded)


def load_aoa_arrays(
    model_dir: Path,
    run_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load pre-saved AOA arrays from model directory.

    Parameters
    ----------
    model_dir : directory containing aoa_*.npy files
    run_id : model run identifier

    Returns
    -------
    X_train, groups, pfts_encoded
    """
    model_dir = Path(model_dir)
    X_train = np.load(model_dir / f"aoa_X_train_{run_id}.npy", allow_pickle=False)
    groups = np.load(model_dir / f"aoa_groups_{run_id}.npy", allow_pickle=False)
    pfts_encoded = np.load(model_dir / f"aoa_pfts_encoded_{run_id}.npy", allow_pickle=False)
    return X_train, groups, pfts_encoded
