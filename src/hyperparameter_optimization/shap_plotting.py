"""
SHAP plotting and aggregation functions.

Extracted from test_hyperparameter_tuning_ML_spatial_stratified.py. Contains
all PFT grouping, SHAP aggregation, and SHAP visualization functions.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.hyperparameter_optimization.shap_constants import (
    PFT_COLORS,
    PFT_COLUMNS,
    PFT_FULL_NAMES,
    SHAP_UNITS,
    get_feature_unit,
    get_shap_label,
)


def group_pft_for_summary_plots(shap_values, X_df, feature_names, pft_cols):
    """
    Combines one-hot encoded PFT columns into a single 'PFT' feature for SHAP plotting.

    Returns:
        new_shap_values (np.array): SHAP matrix with PFT columns summed
        new_X_df (pd.DataFrame): Feature matrix with PFT columns converted to categorical codes
        new_feature_names (list): Updated list of feature names
    """
    # 1. Identify indices
    pft_indices = [i for i, f in enumerate(feature_names) if f in pft_cols]

    if not pft_indices:
        print("No PFT columns found to group.")
        return shap_values, X_df, feature_names

    print(f"Grouping {len(pft_indices)} PFT features into one 'PFT' feature...")

    # 2. Aggregate SHAP values (Sum across PFT columns)
    # shape: (n_samples, 1)
    pft_shap_sum = shap_values[:, pft_indices].sum(axis=1).reshape(-1, 1)

    # 3. Create Feature Value (Integer encoding of the active PFT)
    # Extract just the PFT columns from X_df
    X_pft = X_df[pft_cols]
    # Find which column is '1' (or max) -> returns series of strings ('ENF', 'MF', etc.)
    pft_labels = X_pft.idxmax(axis=1)
    # Convert to integer codes (0, 1, 2...) so SHAP plots can color-code them
    pft_codes = pft_labels.astype("category").cat.codes.values.reshape(-1, 1)

    # 4. Remove old columns and append new one
    # Create mask for non-PFT columns
    keep_mask = np.ones(shap_values.shape[1], dtype=bool)
    keep_mask[pft_indices] = False

    # Filter SHAP
    shap_non_pft = shap_values[:, keep_mask]
    new_shap_values = np.hstack([shap_non_pft, pft_shap_sum])

    # Filter X
    # Get non-PFT feature names
    non_pft_names = [f for i, f in enumerate(feature_names) if i not in pft_indices]
    X_non_pft = X_df[non_pft_names].values
    new_X_values = np.hstack([X_non_pft, pft_codes])

    # Update Feature Names
    new_feature_names = non_pft_names + ["PFT"]

    # Create new DataFrame for X
    new_X_df = pd.DataFrame(new_X_values, columns=new_feature_names)

    return new_shap_values, new_X_df, new_feature_names


def aggregate_pft_shap_values(
    shap_values: np.ndarray, feature_names: list[str], pft_columns: list[str] = None, aggregation: str = "sum"
) -> tuple[np.ndarray, list[str]]:
    """
    Aggregate SHAP values for PFT one-hot encoded columns into a single 'PFT' feature.

    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values array of shape (n_samples, n_features)
    feature_names : list
        List of feature names matching shap_values columns
    pft_columns : list, optional
        List of PFT column names to aggregate. Default uses PFT_COLUMNS.
    aggregation : str
        How to aggregate: 'sum', 'mean', or 'abs_sum'

    Returns:
    --------
    Tuple[np.ndarray, List[str]]
        (aggregated_shap_values, aggregated_feature_names)
    """
    if pft_columns is None:
        pft_columns = PFT_COLUMNS

    # Find which PFT columns exist in the feature names
    pft_indices = []
    existing_pft_cols = []
    for pft in pft_columns:
        if pft in feature_names:
            pft_indices.append(feature_names.index(pft))
            existing_pft_cols.append(pft)

    if not pft_indices:
        logging.warning("No PFT columns found in feature names. Returning original data.")
        return shap_values, feature_names

    logging.info(f"Aggregating {len(existing_pft_cols)} PFT columns: {existing_pft_cols}")

    # Get non-PFT feature indices and names
    non_pft_indices = [i for i, name in enumerate(feature_names) if name not in pft_columns]
    non_pft_names = [feature_names[i] for i in non_pft_indices]

    # Extract SHAP values
    shap_non_pft = shap_values[:, non_pft_indices]
    shap_pft = shap_values[:, pft_indices]

    # Aggregate PFT SHAP values
    if aggregation == "sum":
        shap_pft_agg = shap_pft.sum(axis=1, keepdims=True)
    elif aggregation == "mean":
        shap_pft_agg = shap_pft.mean(axis=1, keepdims=True)
    elif aggregation == "abs_sum":
        shap_pft_agg = np.abs(shap_pft).sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    # Combine: non-PFT features + aggregated PFT
    shap_aggregated = np.hstack([shap_non_pft, shap_pft_agg])
    feature_names_aggregated = non_pft_names + ["PFT"]

    logging.info(f"Aggregated SHAP shape: {shap_aggregated.shape}")
    logging.info(f"New feature count: {len(feature_names_aggregated)} (was {len(feature_names)})")

    return shap_aggregated, feature_names_aggregated


def get_sample_pft_labels(
    X_original: np.ndarray, feature_names: list[str], pft_columns: list[str] = None
) -> np.ndarray:
    """
    Get PFT label for each sample based on one-hot encoded columns.

    Parameters:
    -----------
    X_original : np.ndarray
        Original feature values
    feature_names : list
        List of feature names
    pft_columns : list, optional
        List of PFT column names

    Returns:
    --------
    np.ndarray
        Array of PFT labels (strings) for each sample
    """
    if pft_columns is None:
        pft_columns = PFT_COLUMNS

    # Find PFT column indices
    pft_indices = {}
    for pft in pft_columns:
        if pft in feature_names:
            pft_indices[pft] = feature_names.index(pft)

    if not pft_indices:
        logging.warning("No PFT columns found. Returning 'Unknown' for all samples.")
        return np.array(["Unknown"] * len(X_original))

    # For each sample, find which PFT has value 1 (or highest value)
    pft_labels = []
    for i in range(len(X_original)):
        max_val = -np.inf
        max_pft = "Unknown"
        for pft, idx in pft_indices.items():
            if X_original[i, idx] > max_val:
                max_val = X_original[i, idx]
                max_pft = pft
        pft_labels.append(max_pft)

    return np.array(pft_labels)


def plot_shap_by_pft_boxplot(
    shap_values: np.ndarray,
    feature_names: list[str],
    pft_labels: np.ndarray,
    top_n: int = 10,
    output_dir: Path | None = None,
) -> plt.Figure:
    """
    Create boxplots showing SHAP value distributions for each feature, stratified by PFT.

    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values array (should NOT include PFT columns, or use aggregated version)
    feature_names : list
        List of feature names
    pft_labels : np.ndarray
        PFT label for each sample
    top_n : int
        Number of top features to plot
    output_dir : Path, optional
        Directory to save the plot

    Returns:
    --------
    matplotlib.figure.Figure
    """
    logging.info(f"Generating SHAP by PFT boxplots for top {top_n} features...")

    # Get top features by mean absolute SHAP
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]
    top_features = [feature_names[i] for i in top_indices]

    # Get unique PFTs (excluding 'Unknown' if present)
    unique_pfts = [p for p in np.unique(pft_labels) if p != "Unknown"]
    n_pfts = len(unique_pfts)

    if n_pfts == 0:
        logging.warning("No valid PFT labels found. Cannot create PFT-stratified plot.")
        return None

    logging.info(f"  Found {n_pfts} PFT types: {unique_pfts}")

    # Create subplots
    n_cols = 2
    n_rows = (top_n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten()

    for i, feature in enumerate(top_features):
        ax = axes[i]
        feat_idx = feature_names.index(feature)

        # Prepare data for boxplot
        data_by_pft = []
        labels = []
        colors = []

        for pft in unique_pfts:
            mask = pft_labels == pft
            if mask.sum() > 0:
                data_by_pft.append(shap_values[mask, feat_idx])
                labels.append(f"{pft}\n(n={mask.sum()})")
                colors.append(PFT_COLORS.get(pft, "gray"))

        # Create boxplot
        bp = ax.boxplot(data_by_pft, labels=labels, patch_artist=True)

        # Color the boxes
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)

        # Add units to title
        feat_unit = get_feature_unit(feature)
        if feat_unit:
            title = f"{feature} ({feat_unit})"
        else:
            title = feature
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(get_shap_label(), fontsize=10)
        # ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis="x", rotation=45)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        f"SHAP Value Distribution by Plant Functional Type\n(Top {top_n} Features)", fontsize=14, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_dir:
        save_path = output_dir / "shap_by_pft_boxplot.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"    Saved: {save_path}")

    plt.close()
    return fig


def plot_shap_by_pft_violin(
    shap_values: np.ndarray,
    feature_names: list[str],
    pft_labels: np.ndarray,
    top_n: int = 8,
    output_dir: Path | None = None,
) -> plt.Figure:
    """
    Create violin plots showing SHAP value distributions for each feature, stratified by PFT.

    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values array
    feature_names : list
        List of feature names
    pft_labels : np.ndarray
        PFT label for each sample
    top_n : int
        Number of top features to plot
    output_dir : Path, optional
        Directory to save the plot

    Returns:
    --------
    matplotlib.figure.Figure
    """
    logging.info(f"Generating SHAP by PFT violin plots for top {top_n} features...")

    # Get top features by mean absolute SHAP
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]
    top_features = [feature_names[i] for i in top_indices]

    # Get unique PFTs
    unique_pfts = [p for p in np.unique(pft_labels) if p != "Unknown"]
    n_pfts = len(unique_pfts)

    if n_pfts == 0:
        logging.warning("No valid PFT labels found.")
        return None

    # Create DataFrame for seaborn
    plot_data = []
    for i, feature in enumerate(top_features):
        feat_idx = feature_names.index(feature)
        for pft in unique_pfts:
            mask = pft_labels == pft
            for val in shap_values[mask, feat_idx]:
                plot_data.append({"Feature": feature, "PFT": pft, "SHAP Value": val})

    df_plot = pd.DataFrame(plot_data)

    # Create subplots
    n_cols = 2
    n_rows = (top_n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    palette = {pft: PFT_COLORS.get(pft, "gray") for pft in unique_pfts}

    for i, feature in enumerate(top_features):
        ax = axes[i]

        feature_data = df_plot[df_plot["Feature"] == feature]

        sns.violinplot(
            data=feature_data,
            x="PFT",
            y="SHAP Value",
            hue="PFT",
            palette=palette,
            ax=ax,
            inner="box",
            cut=0,
            legend=False,
        )

        ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)

        feat_unit = get_feature_unit(feature)
        if feat_unit:
            title = f"{feature} ({feat_unit})"
        else:
            title = feature
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(get_shap_label(), fontsize=10)
        ax.set_xlabel("")
        # ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis="x", rotation=45)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        f"SHAP Value Distribution by Plant Functional Type\n(Violin Plots - Top {top_n} Features)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_dir:
        save_path = output_dir / "shap_by_pft_violin.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"    Saved: {save_path}")

    plt.close()
    return fig


def plot_feature_importance_by_pft(
    shap_values: np.ndarray,
    feature_names: list[str],
    pft_labels: np.ndarray,
    top_n: int = 10,
    output_dir: Path | None = None,
) -> plt.Figure:
    """
    Create a heatmap showing feature importance (mean |SHAP|) for each PFT.

    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values array
    feature_names : list
        List of feature names
    pft_labels : np.ndarray
        PFT label for each sample
    top_n : int
        Number of top features to show
    output_dir : Path, optional
        Directory to save the plot

    Returns:
    --------
    matplotlib.figure.Figure
    """
    logging.info("Generating feature importance heatmap by PFT...")

    # Get unique PFTs
    unique_pfts = [p for p in np.unique(pft_labels) if p != "Unknown"]
    n_pfts = len(unique_pfts)

    if n_pfts == 0:
        logging.warning("No valid PFT labels found.")
        return None

    # Calculate mean absolute SHAP for each feature and PFT
    importance_matrix = np.zeros((len(feature_names), n_pfts))

    for j, pft in enumerate(unique_pfts):
        mask = pft_labels == pft
        if mask.sum() > 0:
            importance_matrix[:, j] = np.abs(shap_values[mask]).mean(axis=0)

    # Get top features by overall importance
    overall_importance = importance_matrix.mean(axis=1)
    top_indices = np.argsort(overall_importance)[::-1][:top_n]

    # Subset to top features
    importance_subset = importance_matrix[top_indices]
    top_feature_names = [feature_names[i] for i in top_indices]

    # Add units to feature names
    top_feature_labels = []
    for feat in top_feature_names:
        unit = get_feature_unit(feat)
        if unit:
            top_feature_labels.append(f"{feat} ({unit})")
        else:
            top_feature_labels.append(feat)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))

    # Use PFT full names for columns
    pft_display_names = [PFT_FULL_NAMES.get(p, p) for p in unique_pfts]

    im = ax.imshow(importance_subset, aspect="auto", cmap="YlOrRd")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"Mean |SHAP Value| ({SHAP_UNITS})", fontsize=11)

    # Set ticks
    ax.set_xticks(range(n_pfts))
    ax.set_xticklabels(pft_display_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_feature_labels, fontsize=10)

    # Add value annotations
    for i in range(top_n):
        for j in range(n_pfts):
            val = importance_subset[i, j]
            text_color = "white" if val > importance_subset.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color=text_color, fontweight="bold")

    ax.set_xlabel("Plant Functional Type", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title(
        f"Feature Importance by Plant Functional Type\n(Mean |SHAP Value| - Top {top_n} Features)",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    if output_dir:
        save_path = output_dir / "shap_importance_heatmap_by_pft.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"    Saved: {save_path}")

    plt.close()
    return fig


def plot_top_features_per_pft(
    shap_values: np.ndarray,
    feature_names: list[str],
    pft_labels: np.ndarray,
    top_n: int = 5,
    output_dir: Path | None = None,
) -> plt.Figure:
    """
    Create bar plots showing top features for each PFT separately.

    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values array
    feature_names : list
        List of feature names
    pft_labels : np.ndarray
        PFT label for each sample
    top_n : int
        Number of top features to show per PFT
    output_dir : Path, optional
        Directory to save the plot

    Returns:
    --------
    matplotlib.figure.Figure
    """
    logging.info(f"Generating top {top_n} features plot for each PFT...")

    # Get unique PFTs
    unique_pfts = [p for p in np.unique(pft_labels) if p != "Unknown"]
    n_pfts = len(unique_pfts)

    if n_pfts == 0:
        logging.warning("No valid PFT labels found.")
        return None

    # Create subplots
    n_cols = min(3, n_pfts)
    n_rows = (n_pfts + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

    if n_pfts == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, pft in enumerate(unique_pfts):
        ax = axes[i]
        mask = pft_labels == pft
        n_samples = mask.sum()

        if n_samples == 0:
            ax.text(0.5, 0.5, f"No data for {pft}", ha="center", va="center")
            continue

        # Calculate mean absolute SHAP for this PFT
        mean_abs_shap = np.abs(shap_values[mask]).mean(axis=0)

        # Get top features
        top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]
        top_feat_names = [feature_names[idx] for idx in top_indices]
        top_values = mean_abs_shap[top_indices]

        # Add units to feature names
        top_feat_labels = []
        for feat in top_feat_names:
            unit = get_feature_unit(feat)
            if unit:
                top_feat_labels.append(f"{feat}\n({unit})")
            else:
                top_feat_labels.append(feat)

        # Create horizontal bar plot
        colors = [PFT_COLORS.get(pft, "steelblue")] * top_n
        y_pos = np.arange(top_n)

        ax.barh(y_pos, top_values, color=colors, alpha=0.8, edgecolor="black")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_feat_labels, fontsize=9)
        ax.invert_yaxis()  # Top feature at top
        ax.set_xlabel(f"Mean |SHAP Value| ({SHAP_UNITS})", fontsize=10)

        pft_full_name = PFT_FULL_NAMES.get(pft, pft)
        ax.set_title(f"{pft} - {pft_full_name}\n(n={n_samples})", fontsize=11, fontweight="bold")
        # ax.grid(True, alpha=0.3, axis='x')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Top {top_n} Most Important Features by Plant Functional Type", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_dir:
        save_path = output_dir / "shap_top_features_per_pft.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"    Saved: {save_path}")

    plt.close()
    return fig


def plot_pft_shap_summary(
    shap_values: np.ndarray,
    X_original: np.ndarray,
    feature_names: list[str],
    pft_labels: np.ndarray,
    top_n: int = 15,
    output_dir: Path | None = None,
) -> None:
    """
    Create SHAP summary (beeswarm) plots for each PFT separately.

    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values array
    X_original : np.ndarray
        Original feature values for coloring
    feature_names : list
        List of feature names
    pft_labels : np.ndarray
        PFT label for each sample
    top_n : int
        Number of features to display
    output_dir : Path, optional
        Directory to save the plots
    """
    logging.info("Generating SHAP summary plots for each PFT...")

    # Get unique PFTs
    unique_pfts = [p for p in np.unique(pft_labels) if p != "Unknown"]

    for pft in unique_pfts:
        mask = pft_labels == pft
        n_samples = mask.sum()

        if n_samples < 10:
            logging.warning(f"  Skipping {pft} - only {n_samples} samples")
            continue

        logging.info(f"  Generating summary plot for {pft} (n={n_samples})...")

        # Subset data for this PFT
        shap_pft = shap_values[mask]
        X_pft = X_original[mask]

        # Create DataFrame for plotting
        df_X_pft = pd.DataFrame(X_pft, columns=feature_names)

        import shap as _shap  # lazy import — heavy dependency

        plt.figure(figsize=(12, 10))
        _shap.summary_plot(shap_pft, df_X_pft, show=False, max_display=top_n)

        pft_full_name = PFT_FULL_NAMES.get(pft, pft)
        plt.title(
            f"Feature Contributions - {pft} ({pft_full_name})\nn={n_samples} samples | SHAP units: {SHAP_UNITS}",
            fontsize=13,
            fontweight="bold",
        )
        plt.xlabel(get_shap_label(), fontsize=11)
        plt.tight_layout()

        if output_dir:
            save_path = output_dir / f"shap_summary_{pft}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logging.info(f"    Saved: {save_path}")

        plt.close()


def plot_pft_contribution_comparison(
    shap_values: np.ndarray, feature_names: list[str], pft_labels: np.ndarray, output_dir: Path | None = None
) -> plt.Figure:
    """
    Create a grouped bar chart comparing mean SHAP values across PFTs for top features.
    Shows both positive and negative contributions.

    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values array
    feature_names : list
        List of feature names
    pft_labels : np.ndarray
        PFT label for each sample
    output_dir : Path, optional
        Directory to save the plot

    Returns:
    --------
    matplotlib.figure.Figure
    """
    logging.info("Generating PFT contribution comparison plot...")

    # Get unique PFTs
    unique_pfts = [p for p in np.unique(pft_labels) if p != "Unknown"]
    n_pfts = len(unique_pfts)

    if n_pfts == 0:
        logging.warning("No valid PFT labels found.")
        return None

    # Get top 8 features by overall mean absolute SHAP
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:8]
    top_features = [feature_names[i] for i in top_indices]

    # Calculate mean SHAP (not absolute) for each feature and PFT
    mean_shap_by_pft = {}
    for pft in unique_pfts:
        mask = pft_labels == pft
        if mask.sum() > 0:
            mean_shap_by_pft[pft] = shap_values[mask].mean(axis=0)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(top_features))
    width = 0.8 / n_pfts

    for i, pft in enumerate(unique_pfts):
        values = [mean_shap_by_pft[pft][feature_names.index(f)] for f in top_features]
        offset = (i - n_pfts / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, values, width, label=pft, color=PFT_COLORS.get(pft, "gray"), alpha=0.8, edgecolor="black"
        )

    ax.axhline(0, color="black", linestyle="-", linewidth=0.8)

    # X-axis labels with units
    x_labels = []
    for feat in top_features:
        unit = get_feature_unit(feat)
        if unit:
            x_labels.append(f"{feat}\n({unit})")
        else:
            x_labels.append(feat)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel(f"Mean SHAP Value ({SHAP_UNITS})", fontsize=12)
    ax.set_xlabel("Feature", fontsize=12)
    ax.set_title(
        "Mean Feature Contribution by Plant Functional Type\n"
        "(Positive = increases prediction, Negative = decreases prediction)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(title="PFT", loc="upper right", bbox_to_anchor=(1.15, 1))
    # ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_dir:
        save_path = output_dir / "shap_pft_contribution_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"    Saved: {save_path}")

    plt.close()
    return fig


def plot_pft_radar_chart(
    shap_values: np.ndarray,
    feature_names: list[str],
    pft_labels: np.ndarray,
    top_n: int = 8,
    output_dir: Path | None = None,
) -> plt.Figure:
    """
    Create radar charts comparing feature importance across PFTs.

    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values array
    feature_names : list
        List of feature names
    pft_labels : np.ndarray
        PFT label for each sample
    top_n : int
        Number of features to include in radar
    output_dir : Path, optional
        Directory to save the plot

    Returns:
    --------
    matplotlib.figure.Figure
    """
    logging.info("Generating PFT radar chart...")

    # Get unique PFTs
    unique_pfts = [p for p in np.unique(pft_labels) if p != "Unknown"]
    n_pfts = len(unique_pfts)

    if n_pfts == 0:
        logging.warning("No valid PFT labels found.")
        return None

    # Get top features by overall importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]
    top_features = [feature_names[i] for i in top_indices]

    # Calculate normalized importance for each PFT
    importance_by_pft = {}
    for pft in unique_pfts:
        mask = pft_labels == pft
        if mask.sum() > 0:
            imp = np.abs(shap_values[mask]).mean(axis=0)[top_indices]
            # Normalize to 0-1 for radar chart
            importance_by_pft[pft] = imp / imp.max() if imp.max() > 0 else imp

    # Create radar chart
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, polar=True)

    # Compute angle for each feature
    angles = np.linspace(0, 2 * np.pi, top_n, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    for pft in unique_pfts:
        values = importance_by_pft[pft].tolist()
        values += values[:1]  # Close the polygon

        ax.plot(
            angles,
            values,
            "o-",
            linewidth=2,
            label=f"{pft} ({PFT_FULL_NAMES.get(pft, pft)})",
            color=PFT_COLORS.get(pft, "gray"),
        )
        ax.fill(angles, values, alpha=0.15, color=PFT_COLORS.get(pft, "gray"))

    # Set feature labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(top_features, fontsize=10)

    ax.set_title(
        "Feature Importance Profile by Plant Functional Type\n(Normalized Mean |SHAP|)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()

    if output_dir:
        save_path = output_dir / "shap_pft_radar_chart.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"    Saved: {save_path}")

    plt.close()
    return fig


def generate_pft_shap_report(
    shap_values: np.ndarray, feature_names: list[str], pft_labels: np.ndarray, output_dir: Path
) -> pd.DataFrame:
    """
    Generate a comprehensive CSV report of SHAP statistics by PFT.

    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values array
    feature_names : list
        List of feature names
    pft_labels : np.ndarray
        PFT label for each sample
    output_dir : Path
        Directory to save the report

    Returns:
    --------
    pd.DataFrame
        Report dataframe
    """
    logging.info("Generating PFT SHAP statistics report...")

    unique_pfts = [p for p in np.unique(pft_labels) if p != "Unknown"]

    report_data = []

    for pft in unique_pfts:
        mask = pft_labels == pft
        n_samples = mask.sum()

        if n_samples == 0:
            continue

        shap_pft = shap_values[mask]

        for i, feature in enumerate(feature_names):
            feat_shap = shap_pft[:, i]

            report_data.append(
                {
                    "PFT": pft,
                    "PFT_Full_Name": PFT_FULL_NAMES.get(pft, pft),
                    "N_Samples": n_samples,
                    "Feature": feature,
                    "Feature_Unit": get_feature_unit(feature),
                    "Mean_SHAP": feat_shap.mean(),
                    "Std_SHAP": feat_shap.std(),
                    "Mean_Abs_SHAP": np.abs(feat_shap).mean(),
                    "Median_SHAP": np.median(feat_shap),
                    "Min_SHAP": feat_shap.min(),
                    "Max_SHAP": feat_shap.max(),
                    "Pct_Positive": (feat_shap > 0).mean() * 100,
                    "Pct_Negative": (feat_shap < 0).mean() * 100,
                }
            )

    report_df = pd.DataFrame(report_data)

    # Save report
    report_path = output_dir / "shap_statistics_by_pft.csv"
    report_df.to_csv(report_path, index=False)
    logging.info(f"    Saved: {report_path}")

    # Also create a summary pivot table
    pivot_df = report_df.pivot_table(index="Feature", columns="PFT", values="Mean_Abs_SHAP", aggfunc="mean")
    pivot_path = output_dir / "shap_importance_pivot_by_pft.csv"
    pivot_df.to_csv(pivot_path)
    logging.info(f"    Saved: {pivot_path}")

    return report_df


# =============================================================================
# STATIC FEATURE AGGREGATION FUNCTIONS
# =============================================================================


def aggregate_static_feature_shap(
    shap_values: np.ndarray,
    windowed_feature_names: list[str],
    base_feature_names: list[str],
    input_width: int,
    static_features: list[str] = None,
    aggregation: str = "sum",
) -> tuple[np.ndarray, list[str]]:
    """
    Aggregate SHAP values for static features across time steps.

    For static features (that don't change within a window), it makes no sense
    to have separate SHAP values for each time step. This function combines them.

    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values array of shape (n_samples, n_windowed_features)
    windowed_feature_names : list
        Feature names with time suffixes (e.g., 'temperature_t-2', 'elevation_t-0')
    base_feature_names : list
        Original feature names (without time suffixes)
    input_width : int
        Number of time steps in the window
    static_features : list, optional
        List of feature names that are static (don't change over time).
        If None, will use a default list of common static features.
    aggregation : str
        How to aggregate static features: 'sum' or 'mean'

    Returns:
    --------
    Tuple[np.ndarray, List[str]]
        (aggregated_shap_values, aggregated_feature_names)
    """
    if static_features is None:
        static_features = [
            "canopy_height",
            "elevation",
            "LAI",
            "latitude",
            "longitude",
            "MF",
            "DNF",
            "ENF",
            "EBF",
            "WSA",
            "WET",
            "DBF",
            "SAV",
            "prcip/PET",
        ]

    n_samples = shap_values.shape[0]
    n_base_features = len(base_feature_names)

    # Identify which base features are static vs dynamic
    static_base_indices = [i for i, f in enumerate(base_feature_names) if f in static_features]
    dynamic_base_indices = [i for i, f in enumerate(base_feature_names) if f not in static_features]

    static_feature_list = [base_feature_names[i] for i in static_base_indices]
    dynamic_feature_list = [base_feature_names[i] for i in dynamic_base_indices]

    logging.info(f"Static features ({len(static_base_indices)}): {static_feature_list}")
    logging.info(f"Dynamic features ({len(dynamic_base_indices)}): {dynamic_feature_list}")

    # Build new feature list and aggregate SHAP values
    new_shap_columns = []
    new_feature_names = []

    # 1. Handle DYNAMIC features - keep separate by time step
    for t in range(input_width):
        time_offset = input_width - 1 - t
        suffix = f"_t-{time_offset}" if time_offset > 0 else "_t-0"

        for base_idx in dynamic_base_indices:
            windowed_idx = t * n_base_features + base_idx
            new_shap_columns.append(shap_values[:, windowed_idx])
            new_feature_names.append(f"{base_feature_names[base_idx]}{suffix}")

    # 2. Handle STATIC features - aggregate across time steps
    for base_idx in static_base_indices:
        feature_name = base_feature_names[base_idx]

        # Collect SHAP values for this feature across all time steps
        static_shap_across_time = []
        for t in range(input_width):
            windowed_idx = t * n_base_features + base_idx
            static_shap_across_time.append(shap_values[:, windowed_idx])

        # Stack and aggregate
        stacked = np.stack(static_shap_across_time, axis=1)  # (n_samples, input_width)

        if aggregation == "sum":
            aggregated = stacked.sum(axis=1)
        elif aggregation == "mean":
            aggregated = stacked.mean(axis=1)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        new_shap_columns.append(aggregated)
        new_feature_names.append(feature_name)  # No time suffix for static features

    # Combine into new array
    aggregated_shap = np.column_stack(new_shap_columns)

    logging.info(f"Aggregated SHAP shape: {aggregated_shap.shape}")
    logging.info(f"New feature count: {len(new_feature_names)} (was {len(windowed_feature_names)})")

    return aggregated_shap, new_feature_names


def aggregate_static_feature_values(
    X_original: np.ndarray, base_feature_names: list[str], input_width: int, static_features: list[str] = None
) -> tuple[np.ndarray, list[str]]:
    """
    Aggregate feature values for static features (for plotting axes).

    Takes the value from the most recent time step (t-0) for static features,
    since they should be the same across all time steps anyway.

    Parameters:
    -----------
    X_original : np.ndarray
        Original feature values (windowed), shape (n_samples, n_windowed_features)
    base_feature_names : list
        Original feature names (without time suffixes)
    input_width : int
        Number of time steps in the window
    static_features : list, optional
        List of feature names that are static

    Returns:
    --------
    Tuple[np.ndarray, List[str]]
        (aggregated_X, aggregated_feature_names)
    """
    if static_features is None:
        static_features = [
            "canopy_height",
            "elevation",
            "LAI",
            "latitude",
            "longitude",
            "MF",
            "DNF",
            "ENF",
            "EBF",
            "WSA",
            "WET",
            "DBF",
            "SAV",
            "prcip/PET",
        ]

    n_base_features = len(base_feature_names)

    static_base_indices = [i for i, f in enumerate(base_feature_names) if f in static_features]
    dynamic_base_indices = [i for i, f in enumerate(base_feature_names) if f not in static_features]

    new_X_columns = []
    new_feature_names = []

    # 1. Dynamic features - keep separate by time step
    for t in range(input_width):
        time_offset = input_width - 1 - t
        suffix = f"_t-{time_offset}" if time_offset > 0 else "_t-0"

        for base_idx in dynamic_base_indices:
            windowed_idx = t * n_base_features + base_idx
            new_X_columns.append(X_original[:, windowed_idx])
            new_feature_names.append(f"{base_feature_names[base_idx]}{suffix}")

    # 2. Static features - use value from t-0 (most recent)
    last_time_step = input_width - 1
    for base_idx in static_base_indices:
        feature_name = base_feature_names[base_idx]
        windowed_idx = last_time_step * n_base_features + base_idx  # t-0 position
        new_X_columns.append(X_original[:, windowed_idx])
        new_feature_names.append(feature_name)

    aggregated_X = np.column_stack(new_X_columns)

    return aggregated_X, new_feature_names


def get_dynamic_features_only(
    shap_values: np.ndarray,
    feature_names: list[str],
    base_feature_names: list[str],
    input_width: int,
    static_features: list[str] = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Extract only dynamic features from SHAP values (for temporal analysis).

    Parameters:
    -----------
    shap_values : np.ndarray
        Aggregated SHAP values
    feature_names : list
        Aggregated feature names
    base_feature_names : list
        Original base feature names
    input_width : int
        Number of time steps
    static_features : list, optional
        List of static feature names

    Returns:
    --------
    Tuple[np.ndarray, List[str]]
        (dynamic_shap_values, dynamic_feature_names)
    """
    if static_features is None:
        static_features = [
            "canopy_height",
            "elevation",
            "LAI",
            "latitude",
            "longitude",
            "MF",
            "DNF",
            "ENF",
            "EBF",
            "WSA",
            "WET",
            "DBF",
            "SAV",
            "prcip/PET",
        ]

    # Find indices of dynamic features (those with time suffixes)
    dynamic_indices = []
    dynamic_names = []

    for i, name in enumerate(feature_names):
        # Check if this is a dynamic feature (has time suffix)
        is_static = name in static_features
        if not is_static:
            dynamic_indices.append(i)
            dynamic_names.append(name)

    dynamic_shap = shap_values[:, dynamic_indices]

    return dynamic_shap, dynamic_names


def generate_windowed_feature_names(base_feature_names: list[str], input_width: int) -> list[str]:
    """
    Generate feature names for windowed data.

    Parameters:
    -----------
    base_feature_names : list
        Original feature names (for a single time step)
    input_width : int
        Number of time steps in the window

    Returns:
    --------
    list
        Feature names with time step suffixes (e.g., 'temperature_t-2', 'temperature_t-1', 'temperature_t-0')
    """
    windowed_names = []
    for t in range(input_width):
        time_offset = input_width - 1 - t  # t-2, t-1, t-0 for input_width=3
        suffix = f"_t-{time_offset}" if time_offset > 0 else "_t-0"
        windowed_names.extend([f"{feat}{suffix}" for feat in base_feature_names])
    return windowed_names


def plot_seasonal_drivers_by_hemisphere(
    shap_values: np.ndarray,
    feature_names: list[str],
    timestamps: np.ndarray,
    latitudes: np.ndarray,
    top_n: int = 5,
    output_dir: Path | None = None,
) -> plt.Figure:
    """
    Seasonal Driver Analysis with hemisphere separation (Replicates Figure 7).

    Separates SHAP values into Positive (Pushing flow UP) and Negative (Pushing flow DOWN)
    contributions and stacks them on a Day-of-Year axis, with separate plots for
    Northern and Southern hemispheres.

    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values array of shape (n_samples, n_features)
    feature_names : list
        List of feature names matching shap_values columns
    timestamps : np.ndarray
        Timestamps corresponding to each SHAP value row (MUST be same length as shap_values)
    latitudes : np.ndarray
        Latitude for each sample (MUST be same length as shap_values)
    top_n : int
        Number of top features to display individually
    output_dir : Path, optional
        Directory to save the plot

    Returns:
    --------
    matplotlib.figure.Figure
    """
    # Validate inputs
    if len(shap_values) != len(timestamps):
        raise ValueError(f"Length mismatch: shap_values ({len(shap_values)}) != timestamps ({len(timestamps)})")
    if len(shap_values) != len(latitudes):
        raise ValueError(f"Length mismatch: shap_values ({len(shap_values)}) != latitudes ({len(latitudes)})")

    logging.info(f"Generating seasonal drivers plot with {len(shap_values)} samples...")

    # 1. Create DataFrame
    df = pd.DataFrame(shap_values, columns=feature_names)

    # Extract Day of Year (DOY)
    ts = pd.to_datetime(timestamps)
    if hasattr(ts, "dayofyear"):
        df["DOY"] = ts.dayofyear
    else:
        df["DOY"] = ts.dt.dayofyear

    # Add hemisphere indicator
    df["is_southern"] = np.array(latitudes) < 0

    # 2. Identify Top Features globally (by total absolute impact)
    feature_cols = [c for c in df.columns if c not in ["DOY", "is_southern"]]
    total_impact = df[feature_cols].abs().sum().sort_values(ascending=False)
    top_features = total_impact.head(top_n).index.tolist()

    logging.info(f"Top {top_n} features by absolute SHAP impact: {top_features}")

    # 3. Define colors
    colors = sns.color_palette("husl", len(top_features) + 1)  # +1 for "Rest"

    # 4. Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

    hemisphere_data = {"Northern Hemisphere": df[~df["is_southern"]], "Southern Hemisphere": df[df["is_southern"]]}

    for ax, (hemi_name, hemi_df) in zip(axes, hemisphere_data.items()):
        n_samples = len(hemi_df)
        logging.info(f"  {hemi_name}: {n_samples} samples")

        if n_samples == 0:
            ax.text(0.5, 0.5, f"No data for {hemi_name}", ha="center", va="center", transform=ax.transAxes, fontsize=14)
            ax.set_title(hemi_name, fontsize=13, fontweight="bold")
            continue

        # Calculate Mean SHAP per DOY for this hemisphere
        daily_avg = hemi_df.groupby("DOY")[feature_cols].mean()

        # Prepare positive/negative stacks
        pos_data = pd.DataFrame(index=daily_avg.index)
        neg_data = pd.DataFrame(index=daily_avg.index)

        plot_features = top_features.copy()

        for feat in top_features:
            if feat in daily_avg.columns:
                pos_data[feat] = daily_avg[feat].clip(lower=0)
                neg_data[feat] = daily_avg[feat].clip(upper=0)
            else:
                logging.warning(f"Feature '{feat}' not found in daily_avg columns")

        # "Rest" category (sum of all other features)
        other_cols = [c for c in feature_cols if c not in top_features]
        if other_cols:
            pos_data["Rest"] = daily_avg[other_cols].clip(lower=0).sum(axis=1)
            neg_data["Rest"] = daily_avg[other_cols].clip(upper=0).sum(axis=1)
            plot_features.append("Rest")

        color_map = dict(zip(plot_features, colors[: len(plot_features)]))

        # Net prediction line (sum of all SHAP values)
        net_prediction = daily_avg[feature_cols].sum(axis=1)

        # Plot stacked bars - Positive
        if not pos_data.empty:
            pos_data.plot(
                kind="bar",
                stacked=True,
                ax=ax,
                width=1.0,
                color=[color_map.get(c, "gray") for c in pos_data.columns],
                legend=False,
                edgecolor="none",
            )

        # Plot stacked bars - Negative
        if not neg_data.empty:
            neg_data.plot(
                kind="bar",
                stacked=True,
                ax=ax,
                width=1.0,
                color=[color_map.get(c, "gray") for c in neg_data.columns],
                legend=False,
                edgecolor="none",
            )

        # Plot net prediction line
        ax.plot(
            range(len(net_prediction)),
            net_prediction.values,
            color="black",
            linewidth=2,
            label="Modeled Mean",
            zorder=10,
        )

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylabel("SHAP Value (Contribution to Prediction)", fontsize=11)
        ax.set_title(f"{hemi_name} (n={n_samples:,})", fontsize=13, fontweight="bold")

        # Add season background shading
        if "Northern" in hemi_name:
            # Northern Hemisphere seasons
            ax.axvspan(0, 59, alpha=0.08, color="blue")  # Winter (Jan-Feb)
            ax.axvspan(59, 151, alpha=0.08, color="green")  # Spring (Mar-May)
            ax.axvspan(151, 243, alpha=0.08, color="red")  # Summer (Jun-Aug)
            ax.axvspan(243, 334, alpha=0.08, color="orange")  # Fall (Sep-Nov)
            ax.axvspan(334, 366, alpha=0.08, color="blue")  # Winter (Dec)
        else:
            # Southern Hemisphere seasons (flipped)
            ax.axvspan(0, 59, alpha=0.08, color="red")  # Summer (Jan-Feb)
            ax.axvspan(59, 151, alpha=0.08, color="orange")  # Fall (Mar-May)
            ax.axvspan(151, 243, alpha=0.08, color="blue")  # Winter (Jun-Aug)
            ax.axvspan(243, 334, alpha=0.08, color="green")  # Spring (Sep-Nov)
            ax.axvspan(334, 366, alpha=0.08, color="red")  # Summer (Dec)

        # ax.grid(True, alpha=0.3, axis='y')

    # X-axis formatting (on bottom plot only)
    ticks = np.arange(0, 366, 30)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(ticks, rotation=0)
    axes[-1].set_xlabel("Day of Year", fontsize=12)

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map.get(f, "gray")) for f in plot_features]
    handles.append(plt.Line2D([0], [0], color="black", lw=2))
    labels = plot_features + ["Modeled Mean"]

    # Add season legend
    season_handles = [
        plt.Rectangle((0, 0), 1, 1, alpha=0.3, color="blue"),
        plt.Rectangle((0, 0), 1, 1, alpha=0.3, color="green"),
        plt.Rectangle((0, 0), 1, 1, alpha=0.3, color="red"),
        plt.Rectangle((0, 0), 1, 1, alpha=0.3, color="orange"),
    ]
    season_labels = ["Winter", "Spring", "Summer", "Fall"]

    # Create two legends
    leg1 = fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.98), title="Drivers", fontsize=9)
    fig.legend(
        season_handles, season_labels, loc="upper right", bbox_to_anchor=(0.99, 0.60), title="Seasons (NH)", fontsize=9
    )
    fig.add_artist(leg1)  # Re-add first legend

    plt.suptitle("Seasonal Driver Analysis by Hemisphere", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.85, 0.96])

    if output_dir:
        save_path = output_dir / "fig7_seasonal_drivers_by_hemisphere.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Saved seasonal drivers plot to: {save_path}")

    plt.close()
    return fig


def plot_diurnal_drivers(
    shap_values: np.ndarray,
    feature_names: list[str],
    timestamps: np.ndarray,
    observed_values: np.ndarray = None,  # Observed values (original scale)
    base_value: float = 0.0,  # SHAP base value (transformed scale if IS_TRANSFORM)
    observed_mean: float = None,  # Mean of observed values (original scale) for centering
    top_n: int = 5,
    output_dir: Path | None = None,
) -> plt.Figure:
    """
    Diurnal Driver Analysis with Observed Data comparison.

    Both SHAP values and observed values are plotted as deviations from their means,
    allowing direct comparison even when they're on different scales.

    - SHAP values: sum of SHAP = deviation from model's expected value (base_value)
    - Observed values: plotted as deviation from observed mean
    """
    # 1. Create DataFrame
    df = pd.DataFrame(shap_values, columns=feature_names)

    # Extract Hour
    ts = pd.to_datetime(timestamps)
    if hasattr(ts, "hour"):
        df["Hour"] = ts.hour
    else:
        df["Hour"] = ts.dt.hour

    # Add Observed data if provided
    if observed_values is not None:
        df["Observed"] = observed_values
        # Use provided mean or calculate from data
        if observed_mean is None:
            observed_mean = np.mean(observed_values)

    # 2. Identify Features (skip Hour and Observed)
    feature_cols = [c for c in df.columns if c not in ["Hour", "Observed"]]

    # If top_n equals total features, don't use "Rest"
    if top_n >= len(feature_cols):
        top_features = feature_cols  # Use all
    else:
        total_impact = df[feature_cols].abs().sum().sort_values(ascending=False)
        top_features = total_impact.head(top_n).index.tolist()

    # 3. Colors
    colors = sns.color_palette("husl", len(top_features) + 1)

    # 4. Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Calculate Hourly Means
    hourly_avg = df.groupby("Hour").mean()
    hourly_avg = hourly_avg.reindex(range(24), fill_value=0)

    # Prepare stacks
    pos_data = pd.DataFrame(index=hourly_avg.index)
    neg_data = pd.DataFrame(index=hourly_avg.index)

    plot_features = top_features.copy()

    # Fill feature data
    for feat in top_features:
        if feat in hourly_avg.columns:
            pos_data[feat] = hourly_avg[feat].clip(lower=0)
            neg_data[feat] = hourly_avg[feat].clip(upper=0)

    # Handle "Rest"
    other_cols = [c for c in feature_cols if c not in top_features]
    if other_cols:
        pos_data["Rest"] = hourly_avg[other_cols].clip(lower=0).sum(axis=1)
        neg_data["Rest"] = hourly_avg[other_cols].clip(upper=0).sum(axis=1)
        plot_features.append("Rest")

    color_map = dict(zip(plot_features, colors[: len(plot_features)]))

    # --- PLOT BARS ---
    bar_width = 0.8
    bottom_pos = np.zeros(24)
    bottom_neg = np.zeros(24)

    for feat in plot_features:
        if feat in pos_data.columns:
            ax.bar(
                pos_data.index,
                pos_data[feat],
                bottom=bottom_pos,
                width=bar_width,
                color=color_map.get(feat, "gray"),
                label=feat,
                edgecolor="none",
            )
            bottom_pos += pos_data[feat].values

            ax.bar(
                neg_data.index,
                neg_data[feat],
                bottom=bottom_neg,
                width=bar_width,
                color=color_map.get(feat, "gray"),
                edgecolor="none",
            )
            bottom_neg += neg_data[feat].values

    # --- PLOT LINES ---

    # 1. Modeled Net Effect (Sum of SHAP)
    # This represents (Predicted - BaseValue) on transformed scale
    net_prediction = hourly_avg[feature_cols].sum(axis=1)
    ax.plot(
        hourly_avg.index,
        net_prediction.values,
        color="black",
        linewidth=3,
        linestyle="-",
        label="Modeled (Net SHAP)",
        zorder=10,
    )

    # 2. Observed Mean (Relative to Observed Mean)
    # Both lines are now "deviation from mean" style, making them comparable
    if observed_values is not None:
        # Center observed around its own mean (not base_value which may be on different scale)
        obs_relative = hourly_avg["Observed"] - observed_mean
        ax.plot(
            hourly_avg.index,
            obs_relative.values,
            color="red",
            linewidth=2.5,
            linestyle="--",
            marker="d",
            markersize=6,
            label="Observed (Centered)",
            zorder=11,
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel(f"Deviation from Mean ({SHAP_UNITS})", fontsize=12)

    # Day/Night shading
    ax.axvspan(-0.5, 6, alpha=0.1, color="navy")
    ax.axvspan(18, 23.5, alpha=0.1, color="navy")
    ax.axvspan(6, 18, alpha=0.1, color="gold")

    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, ha="right")
    ax.set_xlabel("Hour of Day (Local Solar Time)", fontsize=12)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    # Ensure lines are at the top of legend
    ax.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.18, 1.0), title="Drivers & Targets", fontsize=9)

    title_text = f"Diurnal Driver Analysis (n={len(df):,})\n(Grouped PFTs)"
    if observed_values is not None:
        title_text += f"\nBase Value (Model Average): {base_value:.2f}"

    plt.title(title_text, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.85, 1.0])

    if output_dir:
        save_path = output_dir / "fig_diurnal_drivers.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Saved diurnal drivers plot to: {save_path}")

    plt.close()
    return fig


def plot_diurnal_drivers_heatmap(
    shap_values: np.ndarray,
    feature_names: list[str],
    timestamps: np.ndarray,
    top_n: int = 10,
    output_dir: Path | None = None,
) -> plt.Figure:
    """
    Diurnal Driver Analysis as a heatmap.

    Shows mean SHAP values for top features across hours of the day.
    Rows = features, Columns = hours, Color = SHAP value.

    Uses local solar time, so no hemisphere separation needed.

    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values array of shape (n_samples, n_features)
    feature_names : list
        List of feature names matching shap_values columns
    timestamps : np.ndarray
        Timestamps corresponding to each SHAP value row (local solar time)
    top_n : int
        Number of top features to display
    output_dir : Path, optional
        Directory to save the plot

    Returns:
    --------
    matplotlib.figure.Figure
    """
    # Validate inputs
    if len(shap_values) != len(timestamps):
        raise ValueError(f"Length mismatch: shap_values ({len(shap_values)}) != timestamps ({len(timestamps)})")

    logging.info(f"Generating diurnal heatmap with {len(shap_values)} samples...")

    # 1. Create DataFrame
    df = pd.DataFrame(shap_values, columns=feature_names)

    # Extract Hour of Day from local solar time
    ts = pd.to_datetime(timestamps)
    if hasattr(ts, "hour"):
        df["Hour"] = ts.hour
    else:
        df["Hour"] = ts.dt.hour

    # 2. Identify Top Features (by total absolute impact)
    feature_cols = [c for c in df.columns if c != "Hour"]
    total_impact = df[feature_cols].abs().sum().sort_values(ascending=False)
    top_features = total_impact.head(top_n).index.tolist()

    # 3. Calculate hourly means for top features
    hourly_avg = df.groupby("Hour")[top_features].mean()
    hourly_avg = hourly_avg.reindex(range(24), fill_value=0)

    # 4. Create heatmap
    fig, ax = plt.subplots(figsize=(16, 8))

    # Transpose so features are rows and hours are columns
    heatmap_data = hourly_avg[top_features].T

    # Use diverging colormap centered at 0
    vmax = max(abs(heatmap_data.values.min()), abs(heatmap_data.values.max()))

    im = ax.imshow(heatmap_data.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(f"Mean {get_shap_label()}", fontsize=12)

    # Set ticks
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)

    # Labels
    ax.set_xlabel("Hour of Day (Local Solar Time)", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title(
        f"Diurnal Pattern of Feature Contributions (n={len(df):,})\n(Mean SHAP Values in {SHAP_UNITS})",
        fontsize=14,
        fontweight="bold",
    )

    # Add grid lines between hours
    ax.set_xticks(np.arange(-0.5, 24, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(top_features), 1), minor=True)
    # ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)

    # Add vertical lines for dawn/dusk (approximate)
    ax.axvline(x=5.5, color="orange", linestyle="--", linewidth=2, alpha=0.7, label="Dawn (~6:00)")
    ax.axvline(x=17.5, color="purple", linestyle="--", linewidth=2, alpha=0.7, label="Dusk (~18:00)")

    # Add text annotations for significant values
    threshold = vmax * 0.5  # Only annotate values > 50% of max
    for i in range(len(top_features)):
        for j in range(24):
            val = heatmap_data.values[i, j]
            if abs(val) > threshold:
                text_color = "white" if abs(val) > vmax * 0.7 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=text_color, fontweight="bold")

    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.0))

    plt.tight_layout()

    if output_dir:
        save_path = output_dir / "fig_diurnal_drivers_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Saved diurnal heatmap to: {save_path}")

    plt.close()
    return fig


def plot_diurnal_feature_lines(
    shap_values: np.ndarray,
    feature_names: list[str],
    timestamps: np.ndarray,
    top_n: int = 6,
    output_dir: Path | None = None,
) -> plt.Figure:
    """
    Diurnal Driver Analysis as line plots with confidence intervals.

    Shows mean SHAP values ± 95% CI for each feature across hours of the day.
    Separate subplots for each top feature.

    Uses local solar time, so no hemisphere separation needed.

    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values array of shape (n_samples, n_features)
    feature_names : list
        List of feature names matching shap_values columns
    timestamps : np.ndarray
        Timestamps corresponding to each SHAP value row (local solar time)
    top_n : int
        Number of top features to display (max 9 recommended)
    output_dir : Path, optional
        Directory to save the plot

    Returns:
    --------
    matplotlib.figure.Figure
    """
    # Validate inputs
    if len(shap_values) != len(timestamps):
        raise ValueError(f"Length mismatch: shap_values ({len(shap_values)}) != timestamps ({len(timestamps)})")

    logging.info(f"Generating diurnal line plots with {len(shap_values)} samples...")

    # 1. Create DataFrame
    df = pd.DataFrame(shap_values, columns=feature_names)

    # Extract Hour of Day from local solar time
    ts = pd.to_datetime(timestamps)
    if hasattr(ts, "hour"):
        df["Hour"] = ts.hour
    else:
        df["Hour"] = ts.dt.hour

    # 2. Identify Top Features
    feature_cols = [c for c in df.columns if c != "Hour"]
    total_impact = df[feature_cols].abs().sum().sort_values(ascending=False)
    top_features = total_impact.head(top_n).index.tolist()

    # 3. Create subplots
    n_cols = 3
    n_rows = (top_n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharex=True)
    axes = axes.flatten()

    n_samples = len(df)

    for i, feature in enumerate(top_features):
        ax = axes[i]

        # Calculate mean and std per hour
        hourly_stats = df.groupby("Hour")[feature].agg(["mean", "std", "count"])
        hourly_stats = hourly_stats.reindex(range(24))

        # Calculate 95% CI
        hourly_stats["se"] = hourly_stats["std"] / np.sqrt(hourly_stats["count"])
        hourly_stats["ci95"] = 1.96 * hourly_stats["se"]

        hours = hourly_stats.index
        means = hourly_stats["mean"].values
        ci = hourly_stats["ci95"].fillna(0).values

        # Plot line with confidence interval
        ax.plot(hours, means, color="steelblue", linewidth=2, marker="o", markersize=4)
        ax.fill_between(hours, means - ci, means + ci, color="steelblue", alpha=0.2, label="95% CI")

        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_title(feature, fontsize=11, fontweight="bold")
        # ax.grid(True, alpha=0.3)

        # Add day/night shading
        ax.axvspan(-0.5, 6, alpha=0.08, color="navy")
        ax.axvspan(18, 23.5, alpha=0.08, color="navy")
        ax.axvspan(6, 18, alpha=0.08, color="gold")

        if i % n_cols == 0:
            ax.set_ylabel(get_shap_label(), fontsize=10)
        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Hour of Day", fontsize=10)

        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Set x-axis ticks on all visible plots
    for ax in axes[:top_n]:
        ax.set_xticks(range(0, 24, 3))
        ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 3)])
        ax.set_xlim(-0.5, 23.5)

    plt.suptitle(
        f"Diurnal Pattern of Feature Contributions (n={n_samples:,})\n(Mean SHAP ± 95% CI in {SHAP_UNITS})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_dir:
        save_path = output_dir / "fig_diurnal_drivers_lines.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Saved diurnal line plots to: {save_path}")

    plt.close()
    return fig


def plot_interaction_dependencies(
    shap_values: np.ndarray,
    X_original: np.ndarray,
    feature_names: list[str],
    interaction_pairs: list[tuple[str, str]] | None = None,
    output_dir: Path | None = None,
    cmap: str = "coolwarm",
) -> plt.Figure:
    """
    Dependence plots with interaction coloring (Replicates Figure 8).

    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values array
    X_original : np.ndarray
        Original feature values (for human-readable axes)
    feature_names : list
        List of feature names
    interaction_pairs : list of tuples, optional
        Explicit pairs like [('T_air', 'Daylength'), ('VPD', 'SWC')]
        If None, automatically selects top features and finds best interaction
    output_dir : Path, optional
        Directory to save the plot
    cmap : str, optional
        Colormap for interaction coloring (default: 'coolwarm')
        Options: 'coolwarm', 'RdBu_r', 'viridis', 'plasma', 'Spectral_r', 'seismic'

    Returns:
    --------
    matplotlib.figure.Figure
    """
    logging.info("Generating interaction dependence plots...")

    # Validate inputs
    if len(shap_values) != len(X_original):
        raise ValueError(f"Length mismatch: shap_values ({len(shap_values)}) != X_original ({len(X_original)})")
    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature count mismatch: shap_values has {shap_values.shape[1]} features, "
            f"but {len(feature_names)} names provided"
        )

    # Create DataFrames
    df_X = pd.DataFrame(X_original, columns=feature_names)
    df_shap = pd.DataFrame(shap_values, columns=feature_names)

    # Determine which features/pairs to plot
    if interaction_pairs is not None:
        # Use explicit pairs
        features_to_plot = [(pair[0], pair[1]) for pair in interaction_pairs]
        n_plots = len(features_to_plot)
    else:
        # Auto-select top 3 features by mean absolute SHAP
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:3]
        top_features = [feature_names[i] for i in top_indices]

        # Find best interaction partner for each
        features_to_plot = []
        for feature in top_features:
            # Find feature with highest correlation to SHAP value residual
            candidates = [f for f in feature_names if f != feature]
            if "sw_in" in candidates:
                inter_feat = "sw_in"
            elif "vpd" in candidates:
                inter_feat = "vpd"
            elif "ta" in candidates:
                inter_feat = "ta"
            else:
                inter_feat = candidates[0] if candidates else feature
            features_to_plot.append((feature, inter_feat))
        n_plots = len(features_to_plot)

    logging.info(f"Plotting {n_plots} interaction pairs: {features_to_plot}")

    # Create subplots
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

    # Handle single plot case
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (feature, inter_feat) in enumerate(features_to_plot):
        ax = axes[i]

        # Check if features exist
        if feature not in df_X.columns:
            logging.warning(f"Feature '{feature}' not found in data. Skipping.")
            ax.text(
                0.5,
                0.5,
                f"Feature '{feature}' not found",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_frame_on(False)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            continue
        if inter_feat not in df_X.columns:
            logging.warning(f"Interaction feature '{inter_feat}' not found. Using feature values for color.")
            inter_feat = feature

        # Get data
        x_data = df_X[feature].values
        y_data = df_shap[feature].values
        c_data = df_X[inter_feat].values

        # Remove NaN/inf for plotting
        valid_mask = np.isfinite(x_data) & np.isfinite(y_data) & np.isfinite(c_data)
        x_plot = x_data[valid_mask]
        y_plot = y_data[valid_mask]
        c_plot = c_data[valid_mask]

        if len(x_plot) == 0:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_frame_on(False)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            continue

        # Scatter plot with colormap
        scatter = ax.scatter(x_plot, y_plot, c=c_plot, cmap=cmap, s=10, alpha=0.6, edgecolor="none")

        # Add colorbar with units
        inter_unit = get_feature_unit(inter_feat)
        if inter_unit:
            cbar_label = f"{inter_feat} ({inter_unit})"
        else:
            cbar_label = inter_feat
        cbar = plt.colorbar(scatter, ax=ax, location="bottom", pad=0.15, aspect=30)
        cbar.set_label(cbar_label, fontsize=10)

        # Formatting with units
        feat_unit = get_feature_unit(feature)
        if feat_unit:
            xlabel = f"{feature} ({feat_unit})"
        else:
            xlabel = feature

        ax.axhline(0, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
        ax.set_ylabel(get_shap_label() if i % n_cols == 0 else "", fontsize=11)
        ax.set_title(f"Driver: {feature}\nInteraction: {inter_feat}", fontsize=11)
    # ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Partial Dependence with Interactions\n(SHAP Values in {SHAP_UNITS})", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_dir:
        save_path = output_dir / "fig8_dependence_interaction.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Saved interaction dependence plot to: {save_path}")

    plt.close()
    return fig


# NOTE: Dead run_shap_analysis() function removed (was never called from main()).
# SHAP analysis is performed inline in main() and will be extracted to
# run_shap_analysis_ml.py as part of the separation work.
