"""
Standalone SHAP analysis script for ML models.

Loads saved model artifacts (model, scaler, transformer, config, context bundle)
and runs the full 13-step SHAP analysis independently of training.

Usage:
    python -m src.hyperparameter_optimization.run_shap_analysis_ml \\
        --model_dir outputs/models/xgb/my_run_id \\
        --run_id my_run_id \\
        --model_type xgb \
        --shap_sample_size 50000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from path_config import PathConfig
from src.hyperparameter_optimization.shap_constants import (
    FEATURE_UNITS,
    PFT_COLORS,
    PFT_COLUMNS,
    PFT_FULL_NAMES,
    SHAP_UNITS,
    get_feature_unit,
    get_shap_label,
)
from src.hyperparameter_optimization.shap_plotting import (
    aggregate_pft_shap_values,
    aggregate_static_feature_shap,
    aggregate_static_feature_values,
    generate_pft_shap_report,
    get_sample_pft_labels,
    group_pft_for_summary_plots,
    plot_diurnal_drivers,
    plot_diurnal_drivers_heatmap,
    plot_diurnal_feature_lines,
    plot_feature_importance_by_pft,
    plot_interaction_dependencies,
    plot_pft_contribution_comparison,
    plot_pft_radar_chart,
    plot_pft_shap_summary,
    plot_seasonal_drivers_by_hemisphere,
    plot_shap_by_pft_boxplot,
    plot_shap_by_pft_violin,
    plot_top_features_per_pft,
)
from src.hyperparameter_optimization.target_transformer import TargetTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone SHAP analysis for ML models")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to model directory")
    parser.add_argument("--run_id", type=str, required=True, help="Run ID for artifact loading")
    parser.add_argument("--model_type", type=str, default="xgb", help="Model type (xgb, rf, etc.)")
    parser.add_argument("--shap_sample_size", type=int, default=50000, help="Number of samples for SHAP")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory for plots")
    return parser.parse_args()


def load_artifacts(model_dir: Path, run_id: str, model_type: str):
    """Load all saved artifacts needed for SHAP analysis."""
    model_dir = Path(model_dir)

    # Load model
    model_path = model_dir / f"FINAL_{model_type}_{run_id}.joblib"
    logging.info(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    # Load config
    config_path = model_dir / f"FINAL_config_{run_id}.json"
    logging.info(f"Loading config from: {config_path}")
    with open(config_path) as f:
        config = json.load(f)

    # Load SHAP context bundle
    context_path = model_dir / f"SHAP_context_{run_id}.npz"
    logging.info(f"Loading SHAP context from: {context_path}")
    context = np.load(context_path, allow_pickle=True)

    # Load site info
    site_info_path = model_dir / f"site_info_{run_id}.json"
    site_info_dict = {}
    if site_info_path.exists():
        logging.info(f"Loading site info from: {site_info_path}")
        with open(site_info_path) as f:
            site_info_dict = json.load(f)

    # Load scaler
    scaler_path = model_dir / f"FINAL_scaler_{run_id}_feature.pkl"
    scaler = None
    if scaler_path.exists():
        import pickle

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    # Load target transformer (config nests under "preprocessing")
    transformer = None
    preprocessing = config.get("preprocessing", {})
    transform_method = preprocessing.get("target_transform")
    if transform_method and transform_method != "none":
        transformer_params = preprocessing.get("target_transform_params")
        if transformer_params:
            transformer = TargetTransformer.from_params(transformer_params)

    return model, config, context, site_info_dict, scaler, transformer


def run_shap_analysis(
    model,
    config: dict,
    context: np.lib.npyio.NpzFile,
    site_info_dict: dict,
    transformer: TargetTransformer | None,
    run_id: str,
    plot_dir: Path,
    shap_sample_size: int = 50000,
):
    """Run the full 13-step SHAP analysis from saved artifacts."""
    # Extract data from context bundle
    X_all_scaled = context["X_all_scaled"]
    X_all_records = context["X_all_records"]
    y_all_records = context["y_all_records"]
    site_ids_all_records = context["site_ids"]
    timestamps_all = context["timestamps"]
    pfts_all_records = context["pfts"]

    # Extract config values (nested under data_info in training config)
    data_info = config.get("data_info", {})
    IS_WINDOWING = data_info.get("IS_WINDOWING", True)
    IS_TRANSFORM = config.get("preprocessing", {}).get("target_transform") not in (None, "none")
    INPUT_WIDTH = data_info.get("input_width", 2)
    TIME_SCALE = config.get("time_scale", "daily")
    final_feature_names = config.get("feature_names", [])
    shap_feature_names = config.get("shap_feature_names", final_feature_names)

    SHAP_SAMPLE_SIZE = shap_sample_size

    # Define static features
    STATIC_FEATURES = [
        "canopy_height",
        "elevation",
        "prcip/PET",
        "MF",
        "DNF",
        "ENF",
        "EBF",
        "WSA",
        "WET",
        "DBF",
        "SAV",
    ]

    # Prepare data
    X_for_shap_calculation = X_all_scaled
    X_for_plotting_axes = X_all_records

    # Extract coordinates
    lat_all = np.array([site_info_dict[s]["latitude"] for s in site_ids_all_records])
    lon_all = np.array([site_info_dict[s]["longitude"] for s in site_ids_all_records])

    os.makedirs(str(plot_dir), exist_ok=True)

    try:
        # =================================================================
        # STEP 1: Calculate raw SHAP values
        # =================================================================
        logging.info("Step 1: Calculating raw SHAP values...")

        n_total = len(X_for_shap_calculation)
        if n_total > SHAP_SAMPLE_SIZE:
            logging.info(f"Sampling {SHAP_SAMPLE_SIZE} from {n_total} total samples...")
            np.random.seed(42)
            sampled_indices = np.random.choice(n_total, SHAP_SAMPLE_SIZE, replace=False)
            sampled_indices = np.sort(sampled_indices)
        else:
            logging.info(f"Using all {n_total} samples (below sample_size threshold)")
            sampled_indices = np.arange(n_total)

        X_shap = X_for_shap_calculation[sampled_indices]
        X_original_sampled = X_for_plotting_axes[sampled_indices]
        lat_sampled = lat_all[sampled_indices]
        lon_sampled = lon_all[sampled_indices]
        site_ids_sampled = site_ids_all_records[sampled_indices]
        timestamps_sampled = timestamps_all[sampled_indices]
        y_sampled = y_all_records[sampled_indices]

        logging.info(f"SHAP calculation on {len(X_shap)} samples")

        logging.info("Creating SHAP TreeExplainer...")
        explainer = shap.TreeExplainer(model)

        logging.info("Calculating SHAP values (this may take a while)...")
        shap_values_raw = explainer.shap_values(X_shap)

        if isinstance(shap_values_raw, list):
            logging.info("Model returned list of SHAP values, using first element")
            shap_values_raw = shap_values_raw[0]

        logging.info(f"Raw SHAP values shape: {shap_values_raw.shape}")

        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[0] if len(base_value) > 0 else base_value
        logging.info(f"Base value (expected_value): {base_value}")

        # =================================================================
        # STEP 2: Aggregate static features (if windowing)
        # =================================================================
        if IS_WINDOWING:
            logging.info("\nStep 2: Aggregating SHAP values for static features...")

            shap_values_agg, feature_names_agg = aggregate_static_feature_shap(
                shap_values=shap_values_raw,
                windowed_feature_names=shap_feature_names,
                base_feature_names=final_feature_names,
                input_width=INPUT_WIDTH,
                static_features=STATIC_FEATURES,
                aggregation="sum",
            )

            X_original_agg, _ = aggregate_static_feature_values(
                X_original=X_original_sampled,
                base_feature_names=final_feature_names,
                input_width=INPUT_WIDTH,
                static_features=STATIC_FEATURES,
            )

            shap_values = shap_values_agg
            shap_feature_names_final = feature_names_agg
            X_for_plots = X_original_agg

            shap_values_windowed = shap_values_raw
            shap_feature_names_windowed = shap_feature_names
            X_for_plots_windowed = X_original_sampled

            logging.info(f"Aggregated SHAP values shape: {shap_values.shape}")
            logging.info(f"Aggregated feature count: {len(shap_feature_names_final)}")
        else:
            shap_values = shap_values_raw
            shap_feature_names_final = shap_feature_names
            X_for_plots = X_original_sampled
            shap_values_windowed = None
            shap_feature_names_windowed = None
            X_for_plots_windowed = None

        df_shap = pd.DataFrame(shap_values, columns=shap_feature_names_final)
        df_X = pd.DataFrame(X_for_plots, columns=shap_feature_names_final)

        # =================================================================
        # STEP 3 & 4: Global Importance Plots
        # =================================================================
        logging.info("\nStep 3 & 4: Generating Grouped PFT Plots...")

        pft_cols_to_group = PFT_COLUMNS

        shap_values_grouped, df_X_grouped, feature_names_grouped = group_pft_for_summary_plots(
            shap_values=shap_values,
            X_df=df_X,
            feature_names=shap_feature_names_final,
            pft_cols=pft_cols_to_group,
        )

        logging.info(f"Grouped Data Shape: {shap_values_grouped.shape}")

        # Plot 3a: Beeswarm
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values_grouped, df_X_grouped, show=False, max_display=20)
        plt.xlabel(get_shap_label(), fontsize=12)
        plt.title("Feature Contribution (Land Cover Grouped as 'PFT')\n", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(plot_dir / "shap_summary_beeswarm_grouped.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Plot 3b: Bar
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values_grouped, df_X_grouped, plot_type="bar", show=False, max_display=20)
        plt.xlabel(f"Mean |SHAP Value| ({SHAP_UNITS})", fontsize=12)
        plt.title("Feature Importance (Land Cover Grouped as 'PFT')\n", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(plot_dir / "shap_global_importance_bar_grouped.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Plot 4: Partial Dependence
        logging.info("  Generating Hybrid Partial Dependence Plots...")
        mean_abs_shap = np.abs(shap_values_grouped).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:9]
        top_features = [feature_names_grouped[i] for i in top_indices]

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.flatten()

        for i, feature in enumerate(top_features):
            ax = axes[i]
            x_val = df_X_grouped[feature].values
            y_val = shap_values_grouped[:, feature_names_grouped.index(feature)]

            if feature == "PFT":
                temp_X_pft = df_X[pft_cols_to_group]
                pft_labels_series = temp_X_pft.idxmax(axis=1)
                plot_df = pd.DataFrame({"PFT": pft_labels_series.values, "SHAP": y_val})
                order = plot_df.groupby("PFT")["SHAP"].median().sort_values().index
                sns.boxplot(data=plot_df, x="PFT", y="SHAP", ax=ax, palette="Set2", order=order)
                ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
                ax.set_title(f"Effect of {feature}", fontsize=12, fontweight="bold")
                ax.set_ylabel(get_shap_label(), fontsize=10)
                ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=45)
            else:
                valid_mask = np.isfinite(x_val) & np.isfinite(y_val)
                x_valid = x_val[valid_mask]
                y_valid = y_val[valid_mask]
                if len(x_valid) == 0:
                    ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
                    continue
                ax.scatter(x_valid, y_valid, alpha=0.3, color="steelblue", s=10)
                try:
                    sns.regplot(
                        x=x_valid, y=y_valid, scatter=False, lowess=True, ax=ax, color="red", line_kws={"linewidth": 2}
                    )
                except Exception:
                    pass
                ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
                feat_unit = get_feature_unit(feature)
                xlabel = f"{feature} ({feat_unit})" if feat_unit else feature
                ax.set_title(f"Effect of {feature}", fontsize=12, fontweight="bold")
                ax.set_xlabel(xlabel, fontsize=10)
                ax.set_ylabel(get_shap_label(), fontsize=10)

        plt.suptitle("Partial Dependence Plots (Top 9 Features)\n(PFT Grouped)", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(plot_dir / "shap_partial_dependence_grouped.png", dpi=300, bbox_inches="tight")
        plt.close()

        # =================================================================
        # STEP 5: Spatial SHAP Maps
        # =================================================================
        logging.info("\nStep 5: Generating Spatial SHAP Maps...")
        top_4_features = top_features[:4]
        fig = plt.figure(figsize=(20, 12))

        for i, feature in enumerate(top_4_features):
            ax = fig.add_subplot(2, 2, i + 1, projection=ccrs.PlateCarree())
            ax.add_feature(cfeature.BORDERS, linestyle=":")
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)

            if feature == "PFT":
                pft_index = feature_names_grouped.index("PFT")
                vals = shap_values_grouped[:, pft_index]
            else:
                vals = df_shap[feature].values

            valid_mask = np.isfinite(vals) & np.isfinite(lon_sampled) & np.isfinite(lat_sampled)
            if valid_mask.sum() == 0:
                ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
                continue

            vals_valid = vals[valid_mask]
            lon_valid = lon_sampled[valid_mask]
            lat_valid = lat_sampled[valid_mask]

            vmax = max(abs(vals_valid.min()), abs(vals_valid.max()))
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

            scatter = ax.scatter(
                lon_valid,
                lat_valid,
                c=vals_valid,
                s=30,
                cmap="RdBu_r",
                norm=norm,
                transform=ccrs.PlateCarree(),
                edgecolor="k",
                linewidth=0.3,
                alpha=0.7,
            )
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
            cbar.set_label(f"SHAP Value ({SHAP_UNITS})", fontsize=10)

            if feature == "PFT":
                title = "Spatial Contribution: Land Cover (PFT)"
            else:
                feat_unit = get_feature_unit(feature)
                title = (
                    f"Spatial Contribution: {feature}\n({feat_unit})"
                    if feat_unit
                    else f"Spatial Contribution: {feature}"
                )
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_extent(
                [lon_valid.min() - 5, lon_valid.max() + 5, lat_valid.min() - 5, lat_valid.max() + 5],
                crs=ccrs.PlateCarree(),
            )

        plt.suptitle(
            f"Spatial Distribution of Feature Contributions\n(SHAP Values in {SHAP_UNITS})",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(plot_dir / "shap_spatial_maps.png", dpi=300, bbox_inches="tight")
        plt.close()
        logging.info("    Saved: shap_spatial_maps.png")

        # =================================================================
        # STEP 6: Waterfall Plots
        # =================================================================
        logging.info("\nStep 6: Generating Local Waterfall Plots...")
        pred_vals = model.predict(X_shap)
        high_idx = np.argmax(pred_vals)
        low_idx = np.argmin(pred_vals)

        for idx, name in [(high_idx, "High_Flow"), (low_idx, "Low_Flow")]:
            plt.figure(figsize=(12, 9))
            row_explainer = shap.Explanation(
                values=shap_values[idx],
                base_values=float(base_value),
                data=X_for_plots[idx],
                feature_names=shap_feature_names_final,
            )
            shap.plots.waterfall(row_explainer, show=False, max_display=12)

            if IS_TRANSFORM and transformer is not None:
                pred_original = transformer.inverse_transform(np.array([pred_vals[idx]]))[0]
            else:
                pred_original = pred_vals[idx]

            plt.title(
                f"Why did the model predict {name}?\n"
                f"Site: {site_ids_sampled[idx]} | Predicted: {pred_original:.2f} {SHAP_UNITS}",
                fontsize=12,
                fontweight="bold",
            )
            plt.xlabel(f"SHAP Value ({SHAP_UNITS})", fontsize=11)
            plt.tight_layout()
            plt.savefig(plot_dir / f"shap_waterfall_{name}.png", dpi=300, bbox_inches="tight")
            plt.close()
            logging.info(f"    Saved: shap_waterfall_{name}.png")

        # =================================================================
        # STEP 7: Seasonal Drivers
        # =================================================================
        logging.info("\nStep 7: Generating Seasonal Driver Analysis...")
        try:
            plot_seasonal_drivers_by_hemisphere(
                shap_values=shap_values,
                feature_names=shap_feature_names_final,
                timestamps=timestamps_sampled,
                latitudes=lat_sampled,
                top_n=5,
                output_dir=plot_dir,
            )
        except Exception as e:
            logging.warning(f"    Could not generate seasonal plots: {e}")

        # =================================================================
        # STEP 8: Diurnal Drivers (hourly only)
        # =================================================================
        if TIME_SCALE == "hourly":
            logging.info("\nStep 8: Generating Diurnal Driver Analysis...")
            try:
                shap_values_diurnal, feature_names_diurnal = aggregate_pft_shap_values(
                    shap_values=shap_values,
                    feature_names=shap_feature_names_final,
                    pft_columns=PFT_COLUMNS,
                    aggregation="sum",
                )
                n_all_features = len(feature_names_diurnal)

                plot_diurnal_drivers(
                    shap_values=shap_values_diurnal,
                    feature_names=feature_names_diurnal,
                    timestamps=timestamps_sampled,
                    observed_values=y_sampled,
                    base_value=float(base_value),
                    observed_mean=float(np.mean(y_sampled)),
                    top_n=n_all_features,
                    output_dir=plot_dir,
                )
                plot_diurnal_drivers_heatmap(
                    shap_values=shap_values_diurnal,
                    feature_names=feature_names_diurnal,
                    timestamps=timestamps_sampled,
                    top_n=n_all_features,
                    output_dir=plot_dir,
                )
                plot_diurnal_feature_lines(
                    shap_values=shap_values_diurnal,
                    feature_names=feature_names_diurnal,
                    timestamps=timestamps_sampled,
                    top_n=12,
                    output_dir=plot_dir,
                )
            except Exception as e:
                logging.warning(f"    Could not generate diurnal plots: {e}")
        else:
            logging.info(f"\nStep 8: Skipping diurnal plots (TIME_SCALE={TIME_SCALE})")

        # =================================================================
        # STEP 9: Interaction Dependencies
        # =================================================================
        logging.info("\nStep 9: Generating Interaction Dependence Plots...")
        try:
            if IS_WINDOWING:
                potential_pairs = [
                    ("sw_in_t-0", "vpd_t-0"),
                    ("vpd_t-0", "ta_t-0"),
                    ("ta_t-0", "volumetric_soil_water_layer_1_t-0"),
                    ("sw_in_t-0", "LAI"),
                    ("vpd_t-0", "canopy_height"),
                ]
            else:
                potential_pairs = [
                    ("vpd", "volumetric_soil_water_layer_1"),
                    ("sw_in", "vpd"),
                    ("canopy_height", "LAI"),
                    ("vpd", "PFT"),
                ]

            valid_pairs = [
                (f1, f2)
                for f1, f2 in potential_pairs
                if f1 in shap_feature_names_final and f2 in shap_feature_names_final
            ]
            if valid_pairs:
                plot_interaction_dependencies(
                    shap_values=shap_values,
                    X_original=X_for_plots,
                    feature_names=shap_feature_names_final,
                    interaction_pairs=valid_pairs[:3],
                    output_dir=plot_dir,
                )
        except Exception as e:
            logging.warning(f"    Could not generate interaction plots: {e}")

        # =================================================================
        # STEP 10: Time-Step Comparison (windowed only)
        # =================================================================
        if IS_WINDOWING and shap_values_windowed is not None:
            logging.info("\nStep 10: Generating Time-Step Importance Comparison...")
            try:
                n_base_features = len(final_feature_names)
                dynamic_features = [f for f in final_feature_names if f not in STATIC_FEATURES]

                time_step_importance = {}
                for t in range(INPUT_WIDTH):
                    time_offset = INPUT_WIDTH - 1 - t
                    time_label = f"t-{time_offset}" if time_offset > 0 else "t-0"
                    time_step_importance[time_label] = {}
                    for feat_idx, feat_name in enumerate(final_feature_names):
                        if feat_name in dynamic_features:
                            windowed_idx = t * n_base_features + feat_idx
                            mean_abs = np.abs(shap_values_windowed[:, windowed_idx]).mean()
                            time_step_importance[time_label][feat_name] = mean_abs

                fig, ax = plt.subplots(figsize=(14, 8))
                x = np.arange(len(dynamic_features))
                width = 0.8 / INPUT_WIDTH
                colors_bar = plt.cm.viridis(np.linspace(0.2, 0.8, INPUT_WIDTH))

                for t_idx, (time_label, feat_importance) in enumerate(time_step_importance.items()):
                    values = [feat_importance.get(f, 0) for f in dynamic_features]
                    offset = (t_idx - INPUT_WIDTH / 2 + 0.5) * width
                    ax.bar(x + offset, values, width, label=time_label, color=colors_bar[t_idx])

                ax.set_xlabel("Feature", fontsize=12)
                ax.set_ylabel(f"Mean |SHAP Value| ({SHAP_UNITS})", fontsize=12)
                ax.set_title("Feature Importance by Time Step\n(Dynamic Features Only)", fontsize=14, fontweight="bold")
                ax.set_xticks(x)
                ax.set_xticklabels(dynamic_features, rotation=45, ha="right")
                ax.legend(title="Time Step", loc="upper right")
                plt.tight_layout()
                plt.savefig(plot_dir / "shap_time_step_comparison.png", dpi=300, bbox_inches="tight")
                plt.close()
            except Exception as e:
                logging.warning(f"    Could not generate time-step comparison: {e}")

        # =================================================================
        # STEP 11: Static vs Dynamic Feature Importance
        # =================================================================
        logging.info("\nStep 11: Generating Static vs Dynamic Feature Comparison...")
        try:
            mean_abs_importance = np.abs(shap_values).mean(axis=0)
            feature_importance_df = pd.DataFrame(
                {"feature": shap_feature_names_final, "importance": mean_abs_importance}
            )

            def categorize_feature(name):
                if name in STATIC_FEATURES:
                    return "Static"
                if "_t-" in name or name.endswith("_t-0"):
                    return "Dynamic"
                return "Static"

            feature_importance_df["category"] = feature_importance_df["feature"].apply(categorize_feature)
            feature_importance_df["unit"] = feature_importance_df["feature"].apply(get_feature_unit)
            feature_importance_df = feature_importance_df.sort_values("importance", ascending=True)

            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            colors_plot = feature_importance_df["category"].map({"Static": "coral", "Dynamic": "steelblue"})
            axes[0].barh(range(len(feature_importance_df)), feature_importance_df["importance"], color=colors_plot)
            axes[0].set_yticks(range(len(feature_importance_df)))
            labels_with_units = [
                f"{row['feature']} ({row['unit']})" if row["unit"] else row["feature"]
                for _, row in feature_importance_df.iterrows()
            ]
            axes[0].set_yticklabels(labels_with_units, fontsize=8)
            axes[0].set_xlabel(f"Mean |SHAP Value| ({SHAP_UNITS})", fontsize=12)
            axes[0].set_title("Feature Importance by Category", fontsize=14, fontweight="bold")
            legend_elements = [Patch(facecolor="coral", label="Static"), Patch(facecolor="steelblue", label="Dynamic")]
            axes[0].legend(handles=legend_elements, loc="lower right")

            category_totals = feature_importance_df.groupby("category")["importance"].sum()
            axes[1].pie(
                category_totals,
                labels=category_totals.index,
                autopct="%1.1f%%",
                colors=["steelblue", "coral"],
                startangle=90,
                explode=[0.02] * len(category_totals),
            )
            axes[1].set_title("Total SHAP Importance\nby Feature Category", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(plot_dir / "shap_static_vs_dynamic.png", dpi=300, bbox_inches="tight")
            plt.close()

            feature_importance_df.to_csv(plot_dir / "shap_feature_importance.csv", index=False)
            logging.info("    Saved: shap_static_vs_dynamic.png + shap_feature_importance.csv")
        except Exception as e:
            logging.warning(f"    Could not generate static vs dynamic comparison: {e}")

        # =================================================================
        # STEP 12: PFT-Stratified SHAP Analysis
        # =================================================================
        logging.info("\nStep 12: Generating PFT-Stratified SHAP Analysis...")
        try:
            pft_labels_sampled = get_sample_pft_labels(
                X_original=X_original_sampled,
                feature_names=shap_feature_names if not IS_WINDOWING else shap_feature_names_windowed,
                pft_columns=PFT_COLUMNS,
            )

            shap_values_pft_agg, feature_names_pft_agg = aggregate_pft_shap_values(
                shap_values=shap_values,
                feature_names=shap_feature_names_final,
                pft_columns=PFT_COLUMNS,
                aggregation="sum",
            )

            non_pft_mask = [name not in PFT_COLUMNS for name in shap_feature_names_final]
            X_for_plots_no_pft = X_for_plots[:, non_pft_mask]
            X_for_pft_plots = np.hstack([X_for_plots_no_pft, np.zeros((len(X_for_plots), 1))])

            plot_feature_importance_by_pft(
                shap_values=shap_values_pft_agg,
                feature_names=feature_names_pft_agg,
                pft_labels=pft_labels_sampled,
                top_n=12,
                output_dir=plot_dir,
            )
            plot_top_features_per_pft(
                shap_values=shap_values_pft_agg,
                feature_names=feature_names_pft_agg,
                pft_labels=pft_labels_sampled,
                top_n=6,
                output_dir=plot_dir,
            )
            plot_shap_by_pft_boxplot(
                shap_values=shap_values_pft_agg,
                feature_names=feature_names_pft_agg,
                pft_labels=pft_labels_sampled,
                top_n=8,
                output_dir=plot_dir,
            )
            plot_shap_by_pft_violin(
                shap_values=shap_values_pft_agg,
                feature_names=feature_names_pft_agg,
                pft_labels=pft_labels_sampled,
                top_n=8,
                output_dir=plot_dir,
            )
            plot_pft_contribution_comparison(
                shap_values=shap_values_pft_agg,
                feature_names=feature_names_pft_agg,
                pft_labels=pft_labels_sampled,
                output_dir=plot_dir,
            )
            plot_pft_radar_chart(
                shap_values=shap_values_pft_agg,
                feature_names=feature_names_pft_agg,
                pft_labels=pft_labels_sampled,
                top_n=8,
                output_dir=plot_dir,
            )
            plot_pft_shap_summary(
                shap_values=shap_values_pft_agg,
                X_original=X_for_pft_plots,
                feature_names=feature_names_pft_agg,
                pft_labels=pft_labels_sampled,
                top_n=15,
                output_dir=plot_dir,
            )
            generate_pft_shap_report(
                shap_values=shap_values_pft_agg,
                feature_names=feature_names_pft_agg,
                pft_labels=pft_labels_sampled,
                output_dir=plot_dir,
            )

            logging.info("  PFT-Stratified SHAP Analysis complete!")
        except Exception as e:
            logging.warning(f"  Could not complete PFT-stratified analysis: {e}")
            import traceback

            traceback.print_exc()

        # =================================================================
        # STEP 13: Save SHAP values and metadata
        # =================================================================
        logging.info("\nStep 13: Saving SHAP values...")

        shap_output_path = plot_dir / f"shap_values_{run_id}.npz"
        np.savez_compressed(
            shap_output_path,
            shap_values=shap_values,
            shap_values_pft_aggregated=shap_values_pft_agg,
            shap_values_raw=shap_values_raw if IS_WINDOWING else shap_values,
            sampled_indices=sampled_indices,
            feature_names=shap_feature_names_final,
            feature_names_pft_aggregated=feature_names_pft_agg,
            feature_names_windowed=shap_feature_names_windowed if IS_WINDOWING else shap_feature_names_final,
            pft_labels=pft_labels_sampled,
            timestamps=timestamps_sampled,
            latitudes=lat_sampled,
            longitudes=lon_sampled,
            base_value=base_value,
            static_features=STATIC_FEATURES,
            pft_columns=PFT_COLUMNS,
            shap_units=SHAP_UNITS,
        )
        logging.info(f"    SHAP values saved to: {shap_output_path}")

        feature_units_used = {f: get_feature_unit(f) for f in shap_feature_names_final}
        units_path = plot_dir / f"feature_units_{run_id}.json"
        with open(units_path, "w") as f:
            json.dump(
                {
                    "shap_units": SHAP_UNITS,
                    "feature_units": feature_units_used,
                    "all_feature_units": FEATURE_UNITS,
                    "pft_full_names": PFT_FULL_NAMES,
                    "pft_colors": PFT_COLORS,
                },
                f,
                indent=2,
            )
        logging.info(f"    Feature units saved to: {units_path}")

        logging.info("\n" + "=" * 60)
        logging.info("=== SHAP ANALYSIS COMPLETE ===")
        logging.info("=" * 60)
        logging.info(f"Total plots generated in: {plot_dir}")

    except Exception as e:
        logging.error(f"SHAP Analysis failed: {e}")
        import traceback

        traceback.print_exc()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    model_dir = Path(args.model_dir)
    run_id = args.run_id
    model_type = args.model_type

    # Determine plot directory
    if args.output_dir:
        plot_dir = Path(args.output_dir)
    else:
        paths = PathConfig()
        plot_dir = paths.hyper_tuning_plots_dir / model_type / run_id

    logging.info(f"Loading artifacts from: {model_dir}")
    logging.info(f"Run ID: {run_id}")
    logging.info(f"Output directory: {plot_dir}")

    model, config, context, site_info_dict, scaler, transformer = load_artifacts(model_dir, run_id, model_type)

    run_shap_analysis(
        model=model,
        config=config,
        context=context,
        site_info_dict=site_info_dict,
        transformer=transformer,
        run_id=run_id,
        plot_dir=plot_dir,
        shap_sample_size=args.shap_sample_size,
    )


if __name__ == "__main__":
    main()
