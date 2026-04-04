from __future__ import annotations

"""Time series and seasonal cycle plots."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Color scheme for products (consistent across all plots)
PRODUCT_COLORS = {
    "sap_velocity": "#e41a1c",  # Red
    "gleam": "#377eb8",  # Blue
    "era5land": "#4daf4a",  # Green
    "pmlv2": "#984ea3",  # Purple
    "gldas": "#ff7f00",  # Orange
}


def plot_seasonal_overlay(
    seasonal_profiles: dict[str, dict[str, np.ndarray]],
    pft: str,
    output_path: Path,
) -> None:
    """Plot seasonal cycle overlay for a specific PFT across all products."""
    fig, ax = plt.subplots(figsize=(10, 6))

    months = np.arange(1, 13)
    month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

    for product_name, pft_profiles in seasonal_profiles.items():
        if pft not in pft_profiles:
            continue
        profile = pft_profiles[pft]
        color = PRODUCT_COLORS.get(product_name, "gray")
        ax.plot(months, profile, "-o", color=color, label=product_name, linewidth=2, markersize=4)

    ax.set_xlabel("Month")
    ax.set_ylabel("Z-score (normalized)")
    ax.set_title(f"Seasonal Cycle: {pft}", fontsize=13, fontweight="bold")
    ax.set_xticks(months)
    ax.set_xticklabels(month_labels)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_regional_timeseries(
    ts_df: pd.DataFrame,
    region_name: str,
    output_path: Path,
) -> None:
    """Plot area-mean Z-score time series for a region."""
    if ts_df.empty:
        logger.warning(f"No data for region {region_name}")
        return

    fig, ax = plt.subplots(figsize=(14, 5))

    for product in ts_df["product"].unique():
        subset = ts_df[ts_df["product"] == product].sort_values("time")
        color = PRODUCT_COLORS.get(product, "gray")
        # Smooth with 30-day rolling mean for clarity
        if len(subset) > 60:
            subset = subset.set_index("time")
            smoothed = subset["zscore"].rolling(30, center=True, min_periods=10).mean()
            ax.plot(smoothed.index, smoothed.values, color=color, label=product, linewidth=1.5, alpha=0.8)
        else:
            ax.plot(subset["time"], subset["zscore"], color=color, label=product, linewidth=1.5)

    ax.set_xlabel("Time")
    ax.set_ylabel("Z-score")
    ax.set_title(f"Regional Time Series: {region_name}", fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_zonal_mean(
    zonal_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot zonal mean profiles (latitude vs Z-score per product)."""
    if zonal_df.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 10))

    # Support both old (mean_zscore) and new (seasonal_amplitude) column names
    value_col = "seasonal_amplitude" if "seasonal_amplitude" in zonal_df.columns else "mean_zscore"

    for product in zonal_df["product"].unique():
        subset = zonal_df[zonal_df["product"] == product].sort_values("lat_center")
        color = PRODUCT_COLORS.get(product, "gray")
        ax.plot(subset[value_col], subset["lat_center"], color=color, label=product, linewidth=2)

    ax.set_xlabel("Seasonal Amplitude (max − min monthly mean)")
    ax.set_ylabel("Latitude")
    ax.set_title("Zonal Signal Strength Profiles", fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_pft_heatmap(
    pft_metrics_all: dict[str, pd.DataFrame],
    metric: str,
    output_path: Path,
) -> None:
    """Heatmap of a metric (e.g., pearson_r) per PFT per product."""
    # Build pivot table: rows=PFTs, cols=products
    records = []
    for prod_name, df in pft_metrics_all.items():
        for _, row in df.iterrows():
            records.append(
                {
                    "pft": row["pft"],
                    "product": prod_name,
                    metric: row[metric],
                }
            )

    if not records:
        return

    combined = pd.DataFrame(records)
    pivot = combined.pivot(index="pft", columns="product", values=metric)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    plt.colorbar(im, ax=ax, label=metric, shrink=0.8)
    ax.set_title(f"{metric} by PFT and Product", fontsize=13, fontweight="bold")

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")
