"""Diagnostics: performance curves, feature ranking, scorer comparison."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def load_results(json_path: Path) -> dict[str, Any]:
    """Load a forward selection results JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    # Convert step keys back to int
    for key in ("step_scores", "step_features", "step_pooled_r2", "step_pooled_rmse"):
        if key in data:
            data[key] = {int(k): v for k, v in data[key].items()}
    return data


def plot_performance_curve(
    results: dict[str, Any],
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot the performance curve across selection steps.

    Parameters
    ----------
    results : dict
        Output from ``selector.run_forward_selection()``.
    output_path : Path, optional
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    step_scores = results["step_scores"]
    steps = sorted(step_scores.keys())
    scores = [step_scores[s] for s in steps]
    scoring_name = results["scoring"]

    # Overlay matching pooled metric (R2 with R2, RMSE with RMSE)
    step_pooled_r2 = results.get("step_pooled_r2", {})
    step_pooled_rmse = results.get("step_pooled_rmse", {})
    if scoring_name == "mean_r2":
        pooled_data, pooled_label = step_pooled_r2, "Pooled R\u00b2"
    else:
        pooled_data, pooled_label = step_pooled_rmse, "Pooled RMSE (neg)"
    has_pooled = bool(pooled_data)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, scores, "o-", linewidth=2, markersize=6, label=f"Mean-fold {scoring_name}")

    if has_pooled:
        if scoring_name == "mean_r2":
            pooled_scores = [pooled_data[s] for s in steps]
        else:
            # Negate RMSE to match sklearn neg convention for visual comparison
            pooled_scores = [-pooled_data[s] for s in steps]
        ax.plot(
            steps,
            pooled_scores,
            "s--",
            linewidth=2,
            markersize=5,
            color="darkorange",
            alpha=0.8,
            label=pooled_label,
        )

    # Mark the best step (by mean-fold score)
    best_step = steps[int(np.argmax(scores))]
    best_score = max(scores)
    ax.axvline(best_step, color="red", linestyle="--", alpha=0.7, label=f"Best: step {best_step}")
    ax.axhline(best_score, color="red", linestyle=":", alpha=0.5)

    ax.set_xlabel("Number of features (mandatory + selected)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Forward Feature Selection \u2014 {scoring_name}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Performance curve saved to %s", output_path)

    return fig


def plot_feature_ranking(
    results: dict[str, Any],
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot the order in which features were selected.

    Parameters
    ----------
    results : dict
        Output from ``selector.run_forward_selection()``.
    output_path : Path, optional
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    step_features = results["step_features"]
    steps = sorted(step_features.keys())

    # Determine the feature added at each step
    added_features: list[str] = []
    prev_set: set[str] = set()
    for step in steps:
        current_set = set(step_features[step])
        new_feats = current_set - prev_set
        if new_feats:
            added_features.append(", ".join(sorted(new_feats)))
        else:
            added_features.append("(fixed)")
        prev_set = current_set

    step_scores = results["step_scores"]
    scores = [step_scores[s] for s in steps]

    fig, ax = plt.subplots(figsize=(14, max(8, len(added_features) * 0.35)))
    y_pos = np.arange(len(added_features))

    bars = ax.barh(y_pos, scores, align="center", color="steelblue", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(added_features, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(results["scoring"], fontsize=12)
    ax.set_title("Feature Selection Order (top = first added)", fontsize=14)

    # Annotate bars with score values
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Feature ranking saved to %s", output_path)

    return fig


def compare_scorers(results_dir: Path) -> dict[str, Any]:
    """Compare results across different scoring modes.

    Parameters
    ----------
    results_dir : Path
        Directory containing ``ffs_*_results.json`` files.

    Returns
    -------
    dict
        Comparison summary with per-scorer best scores and selected features.
    """
    results_dir = Path(results_dir)
    json_files = sorted(results_dir.glob("ffs_*_results.json"))

    if not json_files:
        logger.warning("No result files found in %s", results_dir)
        return {}

    comparison: dict[str, Any] = {}
    all_selected: dict[str, set[str]] = {}

    for jf in json_files:
        data = load_results(jf)
        scoring = data["scoring"]
        comparison[scoring] = {
            "best_score": data["best_score"],
            "n_selected": data["n_selected"],
            "selected_features": data["selected_features"],
            "elapsed_minutes": data.get("elapsed_minutes"),
        }
        all_selected[scoring] = set(data["selected_features"])

    # Find features selected by all scorers (consensus)
    if all_selected:
        consensus = set.intersection(*all_selected.values())
        comparison["_consensus"] = sorted(consensus)
        comparison["_n_scorers"] = len(all_selected)

    # Print summary
    logger.info("=== Scorer Comparison ===")
    for scoring, info in comparison.items():
        if scoring.startswith("_"):
            continue
        logger.info(
            "  %s: best=%.4f, n_features=%d, time=%.1f min",
            scoring,
            info["best_score"],
            info["n_selected"],
            info.get("elapsed_minutes", 0),
        )
    if "_consensus" in comparison:
        logger.info(
            "  Consensus features (%d): %s",
            len(comparison["_consensus"]),
            comparison["_consensus"],
        )

    # Save comparison
    comp_path = results_dir / "ffs_comparison.json"
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    logger.info("Comparison saved to %s", comp_path)

    return comparison


def export_selected_features(
    results: dict[str, Any],
    output_path: Path,
) -> None:
    """Export selected features to a JSON config usable by the training script.

    Parameters
    ----------
    results : dict
        Output from ``selector.run_forward_selection()``.
    output_path : Path
        Path to save the JSON config.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "scoring": results["scoring"],
        "best_score": results["best_score"],
        "selected_features": results["selected_features"],
        "n_selected": results["n_selected"],
    }

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Selected features exported to %s", output_path)
