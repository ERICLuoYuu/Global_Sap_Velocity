"""Generate training CLI commands from FFS results.

Reads FFS results JSON and produces a training command using
--selected_features. The training script computes all features via
apply_all_feature_engineering() and filters to the selected set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def generate_training_command(
    selected_features_json: str,
    run_id: str = "ffs_best",
    time_scale: str = "daily",
    hyperparams_json: str = "src/hyperparameter_optimization/JSON/XGB_hyperparameters_fixed.json",
) -> str:
    """Generate a training script CLI command.

    Parameters
    ----------
    selected_features_json : str
        Path to JSON file with selected feature names.
    run_id : str
        Identifier for the training run.
    time_scale : str
        "daily" or "hourly".
    hyperparams_json : str
        Path to hyperparameter JSON file.
    """
    cmd_parts = [
        "python src/hyperparameter_optimization/test_hyperparameter_tuning_ML_spatial_stratified_prediction.py",
        "--model xgb --RANDOM_SEED 42 --n_groups 10",
        f"--SPLIT_TYPE spatial_stratified --TIME_SCALE {time_scale}",
        "--IS_TRANSFORM True --TRANSFORM_METHOD log1p",
        "--IS_STRATIFIED True --IS_CV True",
        "--SHAP_SAMPLE_SIZE 50000",
        f"--hyperparameters {hyperparams_json}",
        f"--run_id {run_id}",
        f"--selected_features {selected_features_json}",
    ]

    return " \\\n  ".join(cmd_parts)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python ffs_to_training_args.py <ffs_results.json> [run_id] [time_scale]")
        sys.exit(1)

    results_path = Path(sys.argv[1])
    run_id = sys.argv[2] if len(sys.argv) > 2 else "ffs_best"
    time_scale = sys.argv[3] if len(sys.argv) > 3 else "daily"

    with open(results_path) as f:
        results = json.load(f)

    selected = results["selected_features"]
    scoring = results["scoring"]
    best_score = results["best_score"]
    n_selected = results["n_selected"]

    print(f"FFS scoring: {scoring}")
    print(f"Best score: {best_score:.4f}")
    print(f"Selected features ({n_selected}): {selected}")
    print()

    # Save selected features JSON for --selected_features arg
    sf_json_file = results_path.parent / f"selected_features_{scoring}.json"
    with open(sf_json_file, "w") as f:
        json.dump(
            {"selected_features": selected, "n_selected": n_selected, "scoring": scoring},
            f,
            indent=2,
        )
    print(f"Selected features JSON saved to {sf_json_file}")

    cmd = generate_training_command(
        selected_features_json=str(sf_json_file),
        run_id=run_id,
        time_scale=time_scale,
    )
    print("\nTraining command:")
    print(cmd)

    # Save command to file
    cmd_file = results_path.parent / f"training_cmd_{scoring}.sh"
    with open(cmd_file, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Auto-generated from FFS {scoring} results\n")
        f.write(f"# Best score: {best_score:.4f}, {n_selected} features\n\n")
        f.write(cmd + "\n")
    print(f"Saved to {cmd_file}")


if __name__ == "__main__":
    main()
