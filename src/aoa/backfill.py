"""Backfill AOA reference for models without saved training artifacts."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.aoa.prepare import build_aoa_reference

logger = logging.getLogger(__name__)

ALL_PFT_TYPES = ["MF", "DNF", "ENF", "EBF", "WSA", "WET", "DBF", "SAV"]


def _reconstruct_fold_labels(
    groups: np.ndarray,
    pfts: np.ndarray,
    n_splits: int = 10,
    random_state: int = 42,
) -> np.ndarray:
    """Reconstruct CV fold labels from spatial groups and PFT labels.

    Uses StratifiedGroupKFold with the same params as the training script.
    """
    pfts_encoded, _ = pd.factorize(pfts)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_labels = np.full(len(groups), -1, dtype=int)
    X_dummy = np.zeros((len(groups), 1))
    for fold_idx, (_, test_idx) in enumerate(cv.split(X_dummy, pfts_encoded, groups)):
        fold_labels[test_idx] = fold_idx
    assert (fold_labels >= 0).all(), "Some samples not assigned to any fold"
    return fold_labels


def _compute_shap_importances(model, X: np.ndarray, sample_size: int = 50_000, seed: int = 42) -> np.ndarray:
    """Compute mean |SHAP| importance per feature via TreeExplainer."""
    import shap

    if X.shape[0] > sample_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=sample_size, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return np.abs(shap_values).mean(axis=0)


def _create_spatial_groups(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    lat_grid_size: float = 0.05,
    lon_grid_size: float = 0.05,
) -> np.ndarray:
    """Replicate the training script's grid-based spatial grouping."""
    lat_bins = np.arange(
        np.floor(latitudes.min()),
        np.ceil(latitudes.max()) + lat_grid_size,
        lat_grid_size,
    )
    lon_bins = np.arange(
        np.floor(longitudes.min()),
        np.ceil(longitudes.max()) + lon_grid_size,
        lon_grid_size,
    )
    lat_indices = np.digitize(latitudes, lat_bins) - 1
    lon_indices = np.digitize(longitudes, lon_bins) - 1
    n_lon_bins = len(lon_bins)
    grid_cell_ids = lat_indices * n_lon_bins + lon_indices
    unique_groups = np.unique(grid_cell_ids)
    group_map = {g: n for n, g in enumerate(unique_groups)}
    return np.array([group_map[g] for g in grid_cell_ids])


def backfill_from_saved_arrays(
    model_dir: Path,
    run_id: str,
    shap_csv_path: Path,
    feature_names: list[str],
    d_bar_method: str = "full",
) -> Path:
    """Build AOA reference using pre-saved training arrays.

    Expects files from M7 training integration:
      - FINAL_X_train_{run_id}.parquet
      - FINAL_cv_folds_{run_id}.npz (fold_labels, spatial_groups, pfts)
    """
    x_path = model_dir / f"FINAL_X_train_{run_id}.parquet"
    folds_path = model_dir / f"FINAL_cv_folds_{run_id}.npz"

    if not x_path.exists():
        raise FileNotFoundError(f"Training data not found: {x_path}")
    if not folds_path.exists():
        raise FileNotFoundError(f"CV folds not found: {folds_path}")

    X_train = pd.read_parquet(x_path)[feature_names].values
    folds_data = np.load(folds_path)
    fold_labels = folds_data["fold_labels"]

    shap_df = pd.read_csv(shap_csv_path)
    shap_importances = shap_df.set_index("feature")["importance"].reindex(feature_names).values

    return build_aoa_reference(
        X_train=X_train,
        fold_labels=fold_labels,
        shap_importances=shap_importances,
        feature_names=feature_names,
        output_dir=model_dir,
        run_id=run_id,
        d_bar_method=d_bar_method,
    )


def backfill_from_merged_data(
    models_dir: Path,
    model_type: str,
    run_id: str,
    merged_data_dir: Path,
    d_bar_method: str = "full",
    shap_sample_size: int = 50_000,
    output_dir: Path | None = None,
) -> Path:
    """Re-derive training data from merged CSVs and compute SHAP.

    For models trained before M7 integration (no saved arrays).

    Args:
        output_dir: Where to save the reference NPZ. Defaults to model_dir.
    """
    model_dir = models_dir / model_type / run_id

    config_path = model_dir / f"FINAL_config_{run_id}.json"
    with open(config_path) as f:
        config = json.load(f)
    feature_names = config["feature_names"]
    expected_n = config["data_info"]["n_samples"]

    site_csvs = sorted(merged_data_dir.glob("*_daily.csv"))
    if not site_csvs:
        raise FileNotFoundError(f"No merged CSVs in {merged_data_dir}")

    logger.info(f"Loading {len(site_csvs)} site CSVs from {merged_data_dir}")

    list_X = []
    list_groups = []
    list_pfts = []
    site_lats = []
    site_lons = []

    for csv_path in site_csvs:
        df = pd.read_csv(csv_path)
        if df.empty or "sap_velocity" not in df.columns:
            continue

        df = df.dropna(subset=["sap_velocity"])
        if df.empty:
            continue

        # Extract site metadata
        lat = df["latitude_x"].iloc[0] if "latitude_x" in df.columns else None
        lon = df["longitude_x"].iloc[0] if "longitude_x" in df.columns else None
        pft = df["pft"].iloc[0] if "pft" in df.columns else "Unknown"

        if lat is None or lon is None:
            logger.warning(f"Skipping {csv_path.name}: no lat/lon")
            continue

        # Add time feature: Year sin
        if "TIMESTAMP" in df.columns:
            ts = pd.to_datetime(df["TIMESTAMP"])
            df["Year sin"] = np.sin(2 * np.pi * ts.dt.dayofyear / 365.25)

        # One-hot encode PFT
        pft_cat = pd.Categorical([pft], categories=ALL_PFT_TYPES)
        pft_dummies = pd.get_dummies(pft_cat, dtype=float)
        for col in ALL_PFT_TYPES:
            df[col] = float(pft_dummies[col].iloc[0])

        # Select model features
        missing = [f for f in feature_names if f not in df.columns]
        if missing:
            logger.warning(f"Skipping {csv_path.name}: missing features {missing}")
            continue

        X_site = df[feature_names].dropna().values
        n_site = len(X_site)
        if n_site == 0:
            continue

        list_X.append(X_site)
        list_groups.append(np.full(n_site, len(site_lats)))
        list_pfts.append(np.full(n_site, pft, dtype=object))
        site_lats.append(lat)
        site_lons.append(lon)

    if not list_X:
        raise ValueError("No valid site data loaded")

    X_all = np.vstack(list_X)
    pfts_all = np.concatenate(list_pfts)

    # Create spatial groups at site level, then expand to record level
    site_groups = _create_spatial_groups(np.array(site_lats), np.array(site_lons))
    groups_all = np.concatenate([np.full(len(x), site_groups[i]) for i, x in enumerate(list_X)])

    logger.info(f"Loaded {X_all.shape[0]} records from {len(site_lats)} sites (expected {expected_n})")
    if X_all.shape[0] != expected_n:
        logger.warning(f"Sample count mismatch: got {X_all.shape[0]}, expected {expected_n}. Proceeding anyway.")

    # Reconstruct fold labels
    fold_labels = _reconstruct_fold_labels(groups_all, pfts_all)

    # Compute SHAP from saved model
    model_path = model_dir / f"FINAL_{model_type}_{run_id}.joblib"
    scaler_path = model_dir / f"FINAL_scaler_{run_id}_feature.pkl"

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Scale only numeric features (not PFT one-hots)
    pft_cols_idx = [i for i, f in enumerate(feature_names) if f in ALL_PFT_TYPES]
    numeric_idx = [i for i in range(len(feature_names)) if i not in pft_cols_idx]
    X_scaled = X_all.copy()
    X_scaled[:, numeric_idx] = scaler.transform(X_all[:, numeric_idx])

    logger.info(f"Computing SHAP on {min(shap_sample_size, len(X_scaled))} samples...")
    shap_importances = _compute_shap_importances(model, X_scaled, sample_size=shap_sample_size)

    save_dir = output_dir if output_dir is not None else model_dir
    return build_aoa_reference(
        X_train=X_all,
        fold_labels=fold_labels,
        shap_importances=shap_importances,
        feature_names=feature_names,
        output_dir=save_dir,
        run_id=run_id,
        d_bar_method=d_bar_method,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill AOA reference for existing models")
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--model-type", default="xgb")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--merged-data-dir",
        type=Path,
        help="Path to merged daily CSVs (for re-derivation mode)",
    )
    parser.add_argument(
        "--shap-csv",
        type=Path,
        help="Pre-computed SHAP CSV (for saved-arrays mode)",
    )
    parser.add_argument("--d-bar-method", default="full")
    parser.add_argument("--shap-sample-size", type=int, default=50_000)
    args = parser.parse_args()

    model_dir = args.models_dir / args.model_type / args.run_id
    config_path = model_dir / f"FINAL_config_{args.run_id}.json"
    with open(config_path) as f:
        feature_names = json.load(f)["feature_names"]

    # Try saved-arrays mode first
    x_path = model_dir / f"FINAL_X_train_{args.run_id}.parquet"
    if x_path.exists() and args.shap_csv:
        logger.info("Using saved-arrays mode")
        path = backfill_from_saved_arrays(
            model_dir,
            args.run_id,
            args.shap_csv,
            feature_names,
            args.d_bar_method,
        )
    elif args.merged_data_dir:
        logger.info("Using merged-data re-derivation mode")
        path = backfill_from_merged_data(
            args.models_dir,
            args.model_type,
            args.run_id,
            args.merged_data_dir,
            args.d_bar_method,
            args.shap_sample_size,
        )
    else:
        parser.error("Provide --merged-data-dir (re-derivation) or --shap-csv with saved arrays")

    print(f"AOA reference saved to {path}")


if __name__ == "__main__":
    main()
