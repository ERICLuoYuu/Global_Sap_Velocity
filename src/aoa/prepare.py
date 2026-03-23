"""Build AOA reference artifact from training outputs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.aoa import core

logger = logging.getLogger(__name__)

REQUIRED_NPZ_KEYS = frozenset(
    {
        "reference_cloud_weighted",
        "feature_means",
        "feature_stds",
        "feature_weights",
        "d_bar",
        "threshold",
        "fold_assignments",
        "feature_names",
        "training_di",
        "d_bar_method",
    }
)


def build_aoa_reference(
    X_train: np.ndarray,
    fold_labels: np.ndarray,
    shap_importances: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
    run_id: str,
    d_bar_method: str = "full",
) -> Path:
    """Build and save AOA reference NPZ.

    Args:
        X_train: Raw training features (N, P) — NOT pre-scaled.
        fold_labels: CV fold index per sample (N,).
        shap_importances: Mean |SHAP| per feature (P,) — raw, unnormalized.
        feature_names: Feature names matching X_train columns.
        output_dir: Directory for output (usually models/{type}/{run_id}/).
        run_id: Model run identifier.
        d_bar_method: "full" or "sample:N".

    Returns:
        Path to saved NPZ.
    """
    if X_train.ndim != 2:
        raise ValueError(f"X_train must be 2D, got {X_train.ndim}D")
    n, p = X_train.shape
    if fold_labels.shape != (n,):
        raise ValueError(f"fold_labels length {len(fold_labels)} != X_train rows {n}")
    if shap_importances.shape != (p,):
        raise ValueError(f"shap_importances length {len(shap_importances)} != features {p}")
    if len(feature_names) != p:
        raise ValueError(f"feature_names length {len(feature_names)} != features {p}")
    if np.any(np.isnan(X_train)):
        raise ValueError("X_train contains NaN")
    if np.any(np.isnan(shap_importances)):
        raise ValueError("shap_importances contains NaN")

    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0, ddof=1)  # ddof=1 to match R CAST sd()
    X_s = core.standardize_features(X_train, means, stds)
    X_sw = core.apply_importance_weights(X_s, shap_importances)

    if d_bar_method == "full":
        d_bar = core.compute_d_bar_full(X_sw)
    elif d_bar_method.startswith("sample:"):
        sample_size = int(d_bar_method.split(":")[1])
        d_bar = core.compute_d_bar_sample(X_sw, sample_size)
    else:
        raise ValueError(f"Unknown d_bar_method: {d_bar_method}")

    training_di = core.compute_training_di(X_sw, fold_labels, d_bar)
    threshold = core.compute_threshold(training_di)

    logger.info(f"AOA reference: d_bar={d_bar:.4f}, threshold={threshold:.4f}, n_train={n}, n_features={p}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"FINAL_aoa_reference_{run_id}.npz"
    np.savez(
        output_path,
        reference_cloud_weighted=X_sw,
        feature_means=means,
        feature_stds=stds,
        feature_weights=shap_importances,
        d_bar=np.float64(d_bar),
        threshold=np.float64(threshold),
        fold_assignments=fold_labels,
        feature_names=np.array(feature_names, dtype=str),
        training_di=training_di,
        d_bar_method=np.array(d_bar_method, dtype=str),
    )
    return output_path


def load_aoa_reference(path: Path) -> dict:
    """Load and validate reference NPZ."""
    data = dict(np.load(path, allow_pickle=True))
    missing = REQUIRED_NPZ_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Reference NPZ missing keys: {missing}")
    data["d_bar"] = float(data["d_bar"])
    data["threshold"] = float(data["threshold"])
    data["feature_names"] = list(data["feature_names"])
    data["d_bar_method"] = str(data["d_bar_method"])
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AOA reference NPZ")
    parser.add_argument(
        "--training-data",
        type=Path,
        required=True,
        help=".npy or .parquet path with training features",
    )
    parser.add_argument(
        "--shap-csv",
        type=Path,
        required=True,
        help="CSV with 'feature' and 'importance' columns",
    )
    parser.add_argument(
        "--cv-folds",
        type=Path,
        required=True,
        help=".npy or .npz with fold_labels array",
    )
    parser.add_argument(
        "--feature-names-json",
        type=Path,
        required=True,
        help="ModelConfig JSON for feature_names validation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Full path for output NPZ file",
    )
    parser.add_argument(
        "--d-bar-method",
        default="full",
        help="'full' (default) or 'sample:N'",
    )
    args = parser.parse_args()

    with open(args.feature_names_json) as f:
        config = json.load(f)
    feature_names = config["feature_names"]

    if args.training_data.suffix == ".npy":
        X_train = np.load(args.training_data)
    elif args.training_data.suffix == ".parquet":
        df = pd.read_parquet(args.training_data)
        X_train = df[feature_names].values
    else:
        raise ValueError(f"Unsupported format: {args.training_data.suffix}")

    shap_df = pd.read_csv(args.shap_csv)
    shap_importances = shap_df.set_index("feature")["importance"].reindex(feature_names).values

    if args.cv_folds.suffix == ".npy":
        fold_labels = np.load(args.cv_folds)
    elif args.cv_folds.suffix == ".npz":
        fold_labels = np.load(args.cv_folds)["fold_labels"]
    else:
        raise ValueError(f"Unsupported format: {args.cv_folds.suffix}")

    run_id = args.output.stem.replace("FINAL_aoa_reference_", "")
    output_dir = args.output.parent

    path = build_aoa_reference(
        X_train=X_train,
        fold_labels=fold_labels,
        shap_importances=shap_importances,
        feature_names=feature_names,
        output_dir=output_dir,
        run_id=run_id,
        d_bar_method=args.d_bar_method,
    )
    print(f"AOA reference saved to {path}")


if __name__ == "__main__":
    main()
