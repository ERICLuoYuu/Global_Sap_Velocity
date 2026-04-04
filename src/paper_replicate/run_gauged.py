"""Run gauged-continental LSTM for Loritz et al. (2024) replication.

10 Monte Carlo runs (seeds 1-10), each:
1. Load data, compute stats
2. Split 50/10/40
3. Build dataloaders
4. Train LSTM for 20 epochs
5. Predict on test set
6. Compute KGE per genus and overall

Target: KGE 0.77 +/- 0.04
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import PLANT_LIST, NetworkParams
from .data_loader import (
    compute_standardization_stats,
    get_all_columns,
    load_all_data,
    split_gauged,
)
from .evaluator import evaluate_overall, evaluate_per_genus
from .model import RNN_network
from .trainer import build_dataloader, predict, train_model


def run_one_seed(
    all_data: dict,
    igbp_class: dict,
    all_columns: list,
    features: list,
    all_mean,
    all_std,
    sapf_mean: float,
    sapf_std: float,
    min_obs: float,
    seed: int,
    params: NetworkParams,
    device: torch.device,
    save_dir: Path,
) -> dict:
    """Run one gauged Monte Carlo iteration."""
    # Set seeds (Cell 3)
    torch.manual_seed(params.torch_seed if hasattr(params, "torch_seed") else 1)
    torch.cuda.manual_seed(params.cuda_seed if hasattr(params, "cuda_seed") else 1)

    # Split data (Cell 9)
    all_plant_ids = sorted(all_data.keys())
    train_plants, val_plants, test_plants = split_gauged(all_plant_ids, seed)

    print(f"Train: {len(train_plants)}, Val: {len(val_plants)}, Test: {len(test_plants)}")

    # Verify training data has all features
    train_cols = set()
    for pid in train_plants:
        train_cols.update(all_data[pid].columns.tolist())
    train_cols_sorted = sorted(train_cols)
    if train_cols_sorted != all_columns:
        print("WARNING: Training data does not contain all features!")

    # Build dataloaders (Cell 13-14)
    train_loader = build_dataloader(
        all_data,
        train_plants,
        all_columns,
        "sapf",
        features,
        params,
        standardize=True,
        all_mean=all_mean,
        all_std=all_std,
        sapf_mean=sapf_mean,
        sapf_std=sapf_std,
        shuffle=True,
    )
    val_loader = build_dataloader(
        all_data,
        val_plants,
        all_columns,
        "sapf",
        features,
        params,
        standardize=True,
        all_mean=all_mean,
        all_std=all_std,
        sapf_mean=sapf_mean,
        sapf_std=sapf_std,
        shuffle=False,
    )

    # Create model (Cell 12)
    input_size = len(features)
    network = RNN_network(
        input_size=input_size,
        hidden_size=params.hidden_size,
        num_layers=params.no_of_layers,
        dropout=params.drop_out,
    ).to(device)

    # Train (Cell 15-18)
    model_save_dir = save_dir / f"seed_{seed}"
    history = train_model(
        network,
        train_loader,
        val_loader,
        device,
        params,
        min_obs,
        save_dir=model_save_dir,
    )

    # Predict (Cell 20)
    pred_dict, obs_dict, test_counters = predict(
        network,
        all_data,
        test_plants,
        all_columns,
        features,
        params,
        device,
        all_mean,
        all_std,
        sapf_mean,
        sapf_std,
        PLANT_LIST,
    )

    # Metrics (Cell 21)
    per_genus_df = evaluate_per_genus(pred_dict, obs_dict, PLANT_LIST)
    overall = evaluate_overall(pred_dict, obs_dict, PLANT_LIST)

    print(f"\nSeed {seed} results:")
    print(f"Overall KGE: {overall['kge']:.4f}")
    print(per_genus_df.to_string())

    return {
        "seed": seed,
        "overall": overall,
        "per_genus": per_genus_df.to_dict("records"),
        "test_counters": test_counters,
        "history": {k: [float(v) for v in vals] for k, vals in history.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Run gauged-continental LSTM")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=1)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) / "gauged"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    params = NetworkParams()

    # Load data (Cell 6)
    print("Loading data...")
    all_data, igbp_class = load_all_data(data_dir)
    print(f"Loaded {len(all_data)} plant-year time series")

    # Compute stats (Cell 7) - over ALL data (matching their code)
    all_columns, features = get_all_columns(all_data)
    all_mean, all_std, sapf_mean, sapf_std, min_obs = compute_standardization_stats(all_data)
    print(f"Features: {len(features)}, sapf_mean: {sapf_mean:.4f}, sapf_std: {sapf_std:.4f}")
    print(f"min_obs threshold: {min_obs:.4f}")

    # Monte Carlo runs
    all_results = []
    for seed in range(args.seed_start, args.seed_start + args.n_seeds):
        print(f"\n{'=' * 60}")
        print(f"Gauged LSTM - Seed {seed}")
        print(f"{'=' * 60}")

        result = run_one_seed(
            all_data,
            igbp_class,
            all_columns,
            features,
            all_mean,
            all_std,
            sapf_mean,
            sapf_std,
            min_obs,
            seed,
            params,
            device,
            output_dir,
        )
        all_results.append(result)

    # Summary
    kge_values = [r["overall"]["kge"] for r in all_results]
    print(f"\n{'=' * 60}")
    print(f"Gauged LSTM KGE: {np.mean(kge_values):.4f} +/- {np.std(kge_values):.4f}")
    print("Paper target: 0.77 +/- 0.04")
    print(f"{'=' * 60}")

    # Save
    out_path = output_dir / "gauged_results.json"
    serializable = json.loads(
        json.dumps(all_results, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    )
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
