"""Run gauged and ungauged baselines for Loritz et al. (2024) replication.

Baseline = monthly-averaged hourly diurnal sap flow cycle.
- Gauged: per stand+genus diurnal cycle (test key = stand_genus)
- Ungauged: per genus diurnal cycle (test key = genus)

Ports their baseline notebook Cells 5-9.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import (
    load_baseline_data,
    split_baseline_gauged,
    split_baseline_ungauged,
)
from .evaluator import compute_kge, compute_mae, compute_nse

GENERA = ["Quercus", "Pseudotsuga", "Pinus", "Picea", "Fagus", "Larix"]


def custom_mean(series: pd.Series) -> float:
    """Mean of non-zero values, or 0 if all zero (their Cell 5)."""
    if (series == 0).all():
        return 0.0
    return series[series != 0].mean()


def build_diurnal_lookup(
    all_data: dict[str, pd.DataFrame],
    train_plants: list[str],
    gauged: bool,
) -> dict[str, pd.DataFrame]:
    """Build monthly-hourly diurnal cycle lookup from training data (Cell 7).

    Args:
        all_data: plant_year_id -> DataFrame with 'sapf' column
        train_plants: list of training plant IDs
        gauged: if True, key = stand_genus; if False, key = genus

    Returns:
        dict mapping key -> DataFrame with columns ['DateHour', 'pred_sapf']
    """
    merged = defaultdict(pd.DataFrame)

    for plant_id in train_plants:
        if gauged:
            # Gauged key: stand_genus (everything except last 2 parts: idx_year)
            key = plant_id.rsplit("_", 2)[0]
        else:
            # Ungauged key: genus only (3rd from last part in stand_genus_idx_year)
            key = plant_id.rsplit("_", 3)[1]

        df = all_data[plant_id].copy()
        merged[key] = pd.concat([merged[key], df], axis=0)

    lookup = {}
    for key, df in merged.items():
        df["DateHour"] = df.index.strftime("%m %H")
        result = df.groupby("DateHour")["sapf"].agg(custom_mean).reset_index()
        result.rename(columns={"sapf": "pred_sapf"}, inplace=True)
        lookup[key] = result

    return lookup


def run_baseline_predictions(
    all_data: dict[str, pd.DataFrame],
    test_plants: list[str],
    lookup: dict[str, pd.DataFrame],
    gauged: bool,
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]], dict[str, int]]:
    """Run baseline predictions on test data (Cell 8).

    Returns:
        pred_dict, obs_dict (keyed by genus name), test_counters
    """
    pred_dict: dict[str, list[np.ndarray]] = {}
    obs_dict: dict[str, list[np.ndarray]] = {}
    test_counters = {name: 0 for name in GENERA}

    for plant_id in test_plants:
        if gauged:
            test_code = plant_id.rsplit("_", 2)[0]
            plant_name = plant_id.rsplit("_", 3)[1]
        else:
            test_code = plant_id.rsplit("_", 3)[1]
            plant_name = test_code

        df = all_data[plant_id].copy()
        df["DateHour"] = df.index.strftime("%m %H")

        if test_code not in lookup:
            print(f"Plant with ID: {plant_id} not in training sets")
            continue

        merged = df.merge(lookup[test_code], on="DateHour", how="left")
        merged.fillna(0, inplace=True)
        merged = merged[merged["sapf"] > 0]

        if len(merged) == 0:
            continue

        if plant_name not in pred_dict:
            pred_dict[plant_name] = []
            obs_dict[plant_name] = []

        pred_dict[plant_name].append(merged["pred_sapf"].values)
        obs_dict[plant_name].append(merged["sapf"].values)

        test_counters[plant_name] += 1

    return pred_dict, obs_dict, test_counters


def run_one_seed(
    all_data: dict[str, pd.DataFrame],
    igbp_class: dict[str, list[str]],
    seed: int,
    gauged: bool,
) -> dict:
    """Run baseline for one seed."""
    all_plant_ids = sorted(all_data.keys())

    if gauged:
        train_plants, test_plants = split_baseline_gauged(all_plant_ids, seed)
    else:
        train_plants, test_plants = split_baseline_ungauged(all_plant_ids, igbp_class, seed)

    lookup = build_diurnal_lookup(all_data, train_plants, gauged)
    pred_dict, obs_dict, test_counters = run_baseline_predictions(all_data, test_plants, lookup, gauged)

    # Compute per-genus metrics
    results = {"seed": seed, "gauged": gauged, "per_genus": {}}

    all_pred = np.array([])
    all_obs = np.array([])

    for genus in GENERA:
        if genus in pred_dict:
            preds = np.concatenate(pred_dict[genus])
            obs = np.concatenate(obs_dict[genus])
            all_pred = np.concatenate([all_pred, preds])
            all_obs = np.concatenate([all_obs, obs])

            kge, r, alpha, beta = compute_kge(preds, obs)
            results["per_genus"][genus] = {
                "kge": kge,
                "n_trees": test_counters[genus],
                "n_samples": len(preds),
            }

    # Overall
    if len(all_pred) > 0:
        kge, r, alpha, beta = compute_kge(all_pred, all_obs)
        nse = compute_nse(all_pred, all_obs)
        mae = compute_mae(all_pred, all_obs)
        results["overall"] = {
            "kge": kge,
            "nse": nse,
            "mae": mae,
            "n_samples": len(all_pred),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Run baseline models")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--mode", choices=["gauged", "ungauged", "both"], default="both")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading baseline data...")
    all_data, igbp_class = load_baseline_data(data_dir)
    print(f"Loaded {len(all_data)} plant-year time series")

    modes = []
    if args.mode in ("gauged", "both"):
        modes.append(True)
    if args.mode in ("ungauged", "both"):
        modes.append(False)

    for gauged in modes:
        mode_name = "gauged" if gauged else "ungauged"
        print(f"\n{'=' * 60}")
        print(f"Running {mode_name} baseline")
        print(f"{'=' * 60}")

        all_results = []
        for seed in range(1, args.n_seeds + 1):
            print(f"\n--- Seed {seed} ---")
            result = run_one_seed(all_data, igbp_class, seed, gauged)
            all_results.append(result)

            if "overall" in result:
                print(f"Overall KGE: {result['overall']['kge']:.4f}")

        # Summary statistics
        kge_values = [r["overall"]["kge"] for r in all_results if "overall" in r]
        if kge_values:
            print(f"\n{mode_name} baseline KGE: {np.mean(kge_values):.4f} +/- {np.std(kge_values):.4f}")

        # Save results
        out_path = output_dir / f"baseline_{mode_name}_results.json"
        # Convert numpy types for JSON serialization
        serializable = json.loads(
            json.dumps(all_results, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
        )
        with open(out_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
