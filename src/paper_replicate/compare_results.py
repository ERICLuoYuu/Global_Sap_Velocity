"""Compare replication results against Loritz et al. (2024) paper values.

Loads all results JSONs and produces a comparison table.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Paper-reported values (Table from Section 3)
PAPER_VALUES = {
    "gauged_lstm": {"kge_mean": 0.77, "kge_std": 0.04},
    "gauged_baseline": {"kge_mean": 0.64, "kge_std": 0.05},
    "ungauged_lstm": {"kge_mean": 0.52, "kge_std": 0.16},
    "ungauged_baseline": {"kge_mean": -0.11, "kge_std": 0.15},
}


def load_results(output_dir: Path) -> dict:
    """Load all result files."""
    results = {}

    for name, filename in [
        ("gauged_lstm", "gauged/gauged_results.json"),
        ("ungauged_lstm", "ungauged/ungauged_results.json"),
        ("gauged_baseline", "baseline_gauged_results.json"),
        ("ungauged_baseline", "baseline_ungauged_results.json"),
    ]:
        path = output_dir / filename
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)

    return results


def summarize(results: dict) -> pd.DataFrame:
    """Create comparison table."""
    rows = []

    for model_name, paper in PAPER_VALUES.items():
        row = {
            "model": model_name,
            "paper_kge": f"{paper['kge_mean']:.2f} +/- {paper['kge_std']:.2f}",
        }

        if model_name in results:
            kge_values = [r["overall"]["kge"] for r in results[model_name] if "overall" in r]
            if kge_values:
                our_mean = np.mean(kge_values)
                our_std = np.std(kge_values)
                row["our_kge"] = f"{our_mean:.2f} +/- {our_std:.2f}"
                row["delta"] = f"{our_mean - paper['kge_mean']:+.2f}"

                # Check if within acceptable range
                if model_name.endswith("lstm"):
                    threshold = 0.05 if "gauged" in model_name else 0.10
                else:
                    threshold = 0.10
                match = abs(our_mean - paper["kge_mean"]) <= threshold
                row["match"] = "PASS" if match else "FAIL"
            else:
                row["our_kge"] = "no data"
                row["delta"] = "-"
                row["match"] = "-"
        else:
            row["our_kge"] = "not run"
            row["delta"] = "-"
            row["match"] = "-"

        rows.append(row)

    return pd.DataFrame(rows)


def per_genus_summary(results: dict) -> pd.DataFrame:
    """Summarize per-genus KGE across seeds for LSTM models."""
    rows = []

    for model_name in ["gauged_lstm", "ungauged_lstm"]:
        if model_name not in results:
            continue

        # Collect per-genus KGE across seeds
        genus_kges = {}
        for run in results[model_name]:
            for genus_data in run.get("per_genus", []):
                genus = genus_data["genus"]
                kge = genus_data["kge"]
                if genus not in genus_kges:
                    genus_kges[genus] = []
                genus_kges[genus].append(kge)

        for genus, kges in sorted(genus_kges.items()):
            rows.append(
                {
                    "model": model_name,
                    "genus": genus,
                    "kge_mean": f"{np.mean(kges):.2f}",
                    "kge_std": f"{np.std(kges):.2f}",
                    "n_seeds": len(kges),
                }
            )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Compare replication results")
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results = load_results(output_dir)

    print("=" * 70)
    print("Loritz et al. (2024) Replication Results")
    print("=" * 70)

    comparison = summarize(results)
    print("\n## Overall KGE Comparison\n")
    print(comparison.to_string(index=False))

    genus_df = per_genus_summary(results)
    if not genus_df.empty:
        print("\n## Per-Genus KGE\n")
        print(genus_df.to_string(index=False))

    # Save markdown report
    report_path = output_dir / "results_comparison.md"
    with open(report_path, "w") as f:
        f.write("# Loritz et al. (2024) Replication Results\n\n")
        f.write("## Overall KGE Comparison\n\n")
        f.write(comparison.to_markdown(index=False))
        f.write("\n\n")
        if not genus_df.empty:
            f.write("## Per-Genus KGE\n\n")
            f.write(genus_df.to_markdown(index=False))
            f.write("\n")

    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
