"""End-to-end AOA runner using saved model artifacts.

Usage:
    python -m src.aoa.run_aoa \
        --model-dir models/xgb/default_daily_nocoors_swcnor \
        --shap-csv outputs/plots/hyperparameter_optimization/xgb/default_daily_nocoors_swcnor/shap_feature_importance.csv \
        --run-id default_daily_nocoors_swcnor \
        [--output-dir outputs/aoa] \
        [--chunk-size 10000]

Requires pre-saved AOA arrays (aoa_X_train_*.npy, aoa_groups_*.npy,
aoa_pfts_encoded_*.npy) in model_dir. Generate them by running training
with the save_aoa_arrays() call in the training script.

For a quick demo without real prediction data, use --demo to generate
synthetic new-X from training distribution.
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.aoa.aoa import compute_aoa
from src.aoa.model_bridge import (
    load_aoa_arrays,
    load_model_config,
    load_shap_weights,
    reconstruct_fold_indices,
)
from src.aoa.plotting import plot_di_histogram

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_aoa(
    model_dir: Path,
    shap_csv: Path,
    run_id: str,
    output_dir: Path,
    chunk_size: int = 10_000,
    new_X: np.ndarray | None = None,
) -> dict:
    """Run AOA pipeline end-to-end.

    Parameters
    ----------
    model_dir : directory with model artifacts and aoa_*.npy files
    shap_csv : path to shap_feature_importance.csv
    run_id : model run identifier
    output_dir : directory to write AOA outputs
    chunk_size : batch size for DI computation
    new_X : (n_new, n_features) prediction points; if None, uses demo data

    Returns
    -------
    result dict from compute_aoa
    """
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load config
    config_path = model_dir / f"FINAL_config_{run_id}.json"
    config = load_model_config(config_path)
    feature_names = config["feature_names"]
    n_folds = config["cv_results"]["n_folds"]
    random_seed = config["random_seed"]
    split_type = config["split_type"]
    logger.info(
        "Config loaded: %d features, %d folds, split=%s, seed=%d",
        len(feature_names),
        n_folds,
        split_type,
        random_seed,
    )

    # 2. Load SHAP weights
    weights = load_shap_weights(shap_csv, feature_names)
    logger.info("SHAP weights loaded and normalized (sum=%.4f)", weights.sum())

    # 3. Load training arrays
    X_train, groups, pfts_encoded = load_aoa_arrays(model_dir, run_id)
    logger.info("Training arrays loaded: X_train=%s, groups=%s", X_train.shape, groups.shape)

    # 4. Reconstruct fold indices
    fold_indices = reconstruct_fold_indices(
        n_samples=X_train.shape[0],
        groups=groups,
        pfts_encoded=pfts_encoded,
        n_folds=n_folds,
        random_seed=random_seed,
        split_type=split_type,
    )
    logger.info("Reconstructed %d CV folds", len(fold_indices))

    # 5. Prepare new_X (demo mode if not provided)
    if new_X is None:
        logger.info("No new_X provided — generating demo data from training distribution")
        rng = np.random.default_rng(123)
        n_demo = min(5000, X_train.shape[0])
        # Mix of in-distribution and out-of-distribution points
        in_dist = X_train[rng.choice(X_train.shape[0], n_demo // 2, replace=False)]
        # Out-of-distribution: shift training mean by 3 std
        train_mean = X_train.mean(axis=0)
        train_std = X_train.std(axis=0)
        out_dist = rng.normal(
            loc=train_mean + 3 * train_std,
            scale=train_std,
            size=(n_demo // 2, X_train.shape[1]),
        )
        new_X = np.vstack([in_dist, out_dist])
        logger.info("Demo new_X: %s (%d in-dist + %d out-dist)", new_X.shape, n_demo // 2, n_demo // 2)

    # 6. Compute AOA
    logger.info("Computing AOA (chunk_size=%d)...", chunk_size)
    result = compute_aoa(
        train_X=X_train,
        new_X=new_X,
        weights=weights,
        fold_indices=fold_indices,
        chunk_size=chunk_size,
    )

    n_inside = result["aoa"].sum()
    n_total = len(result["aoa"])
    logger.info(
        "AOA complete: threshold=%.4f, d_bar=%.4f, %d/%d (%.1f%%) inside AOA",
        result["threshold"],
        result["d_bar"],
        n_inside,
        n_total,
        100.0 * n_inside / n_total,
    )

    # 7. Save results
    np.save(output_dir / f"di_{run_id}.npy", result["di"])
    np.save(output_dir / f"aoa_mask_{run_id}.npy", result["aoa"])
    np.save(output_dir / f"di_train_{run_id}.npy", result["di_train"])

    metadata = {
        "threshold": float(result["threshold"]),
        "d_bar": float(result["d_bar"]),
        "n_inside_aoa": int(n_inside),
        "n_total": n_total,
        "pct_inside_aoa": round(100.0 * n_inside / n_total, 2),
    }

    with open(output_dir / f"aoa_metadata_{run_id}.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Results saved to %s", output_dir)

    # 8. Plot DI histogram
    fig, ax = plot_di_histogram(
        result["di"],
        result["threshold"],
        di_train=result["di_train"],
    )
    fig.savefig(output_dir / f"di_histogram_{run_id}.png", dpi=150, bbox_inches="tight")

    plt.close(fig)
    logger.info("DI histogram saved")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run AOA analysis on model artifacts")
    parser.add_argument("--model-dir", required=True, help="Path to model directory")
    parser.add_argument("--shap-csv", required=True, help="Path to SHAP feature importance CSV")
    parser.add_argument("--run-id", required=True, help="Model run ID")
    parser.add_argument("--output-dir", default="outputs/aoa", help="Output directory")
    parser.add_argument("--chunk-size", type=int, default=10_000, help="Chunk size for DI")
    args = parser.parse_args()

    run_aoa(
        model_dir=args.model_dir,
        shap_csv=args.shap_csv,
        run_id=args.run_id,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
