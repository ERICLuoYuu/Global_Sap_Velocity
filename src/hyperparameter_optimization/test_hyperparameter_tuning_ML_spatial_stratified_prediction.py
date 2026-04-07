"""
Machine Learning model with a spatial cross-validation approach for site-based prediction.
Implements group-based spatial cross-validation with proper time windowing.
FIXED VERSION: Corrected SHAP analysis with hemisphere separation and proper index tracking.
"""

import argparse
import contextlib
import logging
import os
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import joblib
from sklearn.preprocessing import StandardScaler

# Set environment variables for determinism BEFORE importing TensorFlow/other libraries
os.environ["PYTHONHASHSEED"] = "42"

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the randomization control module first
from src.hyperparameter_optimization.plot_fold_distribution import FoldDistributionAnalyzer  # noqa: E402

# Import the time series windowing modules
from src.hyperparameter_optimization.timeseries_processor1 import TimeSeriesSegmenter  # noqa: E402
from src.utils.random_control import deterministic, set_seed  # noqa: E402

# Set the master seed at the very beginning
set_seed(42)

# Now import all other dependencies
import json  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402

# Import StratifiedGroupKFold and GroupKFold for spatial stratified cross-validation
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold  # noqa: E402

from path_config import PathConfig  # noqa: E402

# Apply additional determinism settings
tf.random.set_seed(42)
np.random.seed(42)

# Import the hyperparameter optimizer
import seaborn as sns  # noqa: E402
import shap  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from src.hyperparameter_optimization.feature_engineering import (  # noqa: E402
    add_sap_flow_features,
    apply_feature_engineering,
)
from src.hyperparameter_optimization.hyper_tuner import MLOptimizer  # noqa: E402

# Constants and helpers imported from shap_constants.py
from src.hyperparameter_optimization.shap_constants import (  # noqa: E402
    FEATURE_UNITS,
    PFT_COLORS,
    PFT_COLUMNS,
    PFT_FULL_NAMES,
    SHAP_UNITS,
    get_feature_unit,
    get_shap_label,
)

# SHAP plotting functions imported from shap_plotting.py
from src.hyperparameter_optimization.shap_plotting import (  # noqa: E402
    aggregate_pft_shap_values,
    aggregate_static_feature_shap,
    aggregate_static_feature_values,
    generate_pft_shap_report,
    generate_windowed_feature_names,
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
from src.hyperparameter_optimization.target_transformer import TargetTransformer  # noqa: E402
from src.hyperparameter_optimization.training_utils import (  # noqa: E402
    add_time_features,
    convert_windows_to_numpy,
    create_spatial_groups,
    create_windows_from_segments,
    setup_logging,
)


def parse_args():
    """Create and return the argument parser for this script."""
    parser = argparse.ArgumentParser(description="Machine learning models with spatial cross-validation")
    parser.add_argument("--RANDOM_SEED", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--model", type=str, default="xgb", help="Model type to train")
    parser.add_argument("--run_id", type=str, default="default_daily_nocoors_swcnor", help="Run identifier for logging")
    parser.add_argument("--n_groups", type=int, default=10, help="Number of spatial groups for cross-validation")
    parser.add_argument("--INPUT_WIDTH", type=int, default=2, help="Input width for time series windows")
    parser.add_argument("--LABEL_WIDTH", type=int, default=1, help="Label width for time series windows")
    parser.add_argument("--SHIFT", type=int, default=1, help="Shift for time series windows")
    parser.add_argument("--TARGET_COL", type=str, default="sap_velocity", help="Target column name")
    parser.add_argument("--EXCLUDE_LABELS", type=bool, default=True, help="Exclude labels from input features")
    parser.add_argument("--EXCLUDE_TARGETS", type=bool, default=True, help="Exclude targets from input features")
    parser.add_argument("--IS_WINDOWING", type=bool, default=False, help="Enable time windowing for data processing")
    parser.add_argument(
        "--spatial_split_method", type=str, default="default", help="Method for spatial splitting of data"
    )
    parser.add_argument("--hyperparameters", type=str, help="Path to the JSON file of hyperparameters")
    parser.add_argument("--IS_SHUFFLE", type=bool, default=True, help="Whether to enable shuffling of data")
    parser.add_argument("--N_ITERATIONS", type=int, default=None, help="Number of iterations for random search")
    parser.add_argument("--IS_CV", type=bool, default=True, help="Whether to enable cross-validation for inner loop")
    parser.add_argument("--IS_STRATIFIED", type=bool, default=True, help="Whether to use stratified sampling")
    parser.add_argument("--BALANCED", type=bool, default=False, help="Whether to balance the spatial groups")
    parser.add_argument("--SPLIT_TYPE", type=str, default="spatial_stratified", help="Type of data splitting strategy")
    parser.add_argument("--IS_ONLY_DAY", type=bool, default=False, help="Whether to use only day data")
    parser.add_argument(
        "--selected_features",
        type=str,
        required=True,
        help="Path to JSON file with feature names. Training uses exactly these features.",
    )
    parser.add_argument("--TIME_SCALE", type=str, default="daily", help="Time scale of the data: hourly or daily")
    parser.add_argument("--SHAP_SAMPLE_SIZE", type=int, default=50000, help="Sample size for SHAP analysis")
    parser.add_argument("--IS_TRANSFORM", type=bool, default=True, help="Whether to apply target transformation")
    parser.add_argument(
        "--TRANSFORM_METHOD",
        type=str,
        default="log1p",
        choices=["log1p", "sqrt", "box-cox", "yeo-johnson", "none"],
        help="Target transformation method: log1p, sqrt, box-cox, yeo-johnson, or none",
    )
    parser.add_argument(
        "--grid_size", type=float, default=0.05, help="Grid cell size in degrees for spatial grouping (default: 0.05)"
    )
    parser.add_argument(
        "--r2_method",
        type=str,
        default="mean",
        choices=["mean", "pooled", "both"],
        help="R2 reporting: mean (per-fold average), pooled (concatenated OOF), or both",
    )
    parser.add_argument(
        "--feature_groups",
        nargs="*",
        default=[],
        help="Feature engineering groups: interactions lags_1d rolling_3d rolling_7d rolling_14d physics precip_memory indicators static_enrich root_zone_swc rew et0 psi_soil cwd",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override data directory path (e.g. for growing-season-filtered data)",
    )
    return parser.parse_args()


@deterministic
def main(run_id="default"):
    """
    Main function implementing the leave-one-group-out spatial CV for an ML model.
    """
    # Unpack arguments
    args = parse_args()
    RANDOM_SEED = args.RANDOM_SEED
    INPUT_WIDTH = args.INPUT_WIDTH
    LABEL_WIDTH = args.LABEL_WIDTH
    SHIFT = args.SHIFT
    TARGET_COL = args.TARGET_COL
    EXCLUDE_LABELS = args.EXCLUDE_LABELS
    EXCLUDE_TARGETS = args.EXCLUDE_TARGETS
    IS_WINDOWING = args.IS_WINDOWING
    MODEL_TYPE = args.model
    IS_SHUFFLE = args.IS_SHUFFLE
    IS_STRATIFIED = args.IS_STRATIFIED
    N_ITERATIONS = args.N_ITERATIONS
    IS_CV = args.IS_CV
    IS_ONLY_DAY = args.IS_ONLY_DAY
    n_groups = args.n_groups
    spatial_split_method = args.spatial_split_method
    SPLIT_TYPE = args.SPLIT_TYPE
    BALANCED = args.BALANCED
    feature_groups = args.feature_groups
    TIME_SCALE = args.TIME_SCALE
    SHAP_SAMPLE_SIZE = args.SHAP_SAMPLE_SIZE
    IS_TRANSFORM = args.IS_TRANSFORM
    TRANSFORM_METHOD = args.TRANSFORM_METHOD if IS_TRANSFORM else "none"
    GRID_SIZE = args.grid_size
    R2_METHOD = args.r2_method

    # Initialize target transformer
    target_transformer = TargetTransformer(method=TRANSFORM_METHOD)

    with open(args.hyperparameters) as f:
        hyperparameters = json.load(f)

    run_id = args.run_id
    scale = "sapwood"
    paths = PathConfig(scale=scale)
    setup_logging(logger_name=MODEL_TYPE, log_dir=paths.optimization_logs_dir)

    logging.info(f"Starting {MODEL_TYPE} model training with {SPLIT_TYPE} CV, seed {RANDOM_SEED})")
    logging.info(
        f"IS_CV={IS_CV}, n_groups={n_groups}, input_width={INPUT_WIDTH}, "
        f"label_width={LABEL_WIDTH}, shift={SHIFT}, exclude_labels={EXCLUDE_LABELS}, "
        f"exclude_targets={EXCLUDE_TARGETS}, is_windowing={IS_WINDOWING}, "
        f"spatial_split_method={spatial_split_method}, is_shuffle={IS_SHUFFLE}, "
        f"is_stratified={IS_STRATIFIED}, is_only_day={IS_ONLY_DAY}, time_scale={TIME_SCALE}, "
        f"is_transform={IS_TRANSFORM}, transform_method={TRANSFORM_METHOD}"
    )

    # --- File Paths ---
    import os

    plot_dir = paths.hyper_tuning_plots_dir / MODEL_TYPE / run_id
    os.makedirs(str(plot_dir), exist_ok=True)
    model_dir = paths.models_root / MODEL_TYPE / run_id
    os.makedirs(str(model_dir), exist_ok=True)
    # data_dir = paths.merged_data_root / TIME_SCALE
    if args.data_dir is not None:
        data_dir = Path(args.data_dir) / TIME_SCALE
        logging.info(f"Using custom data_dir: {data_dir}")
    else:
        data_dir = paths.merged_daytime_only_dir / TIME_SCALE

    # --- Data Loading and Processing ---
    data_list = sorted(list(data_dir.glob(f"*{TIME_SCALE}.csv")))
    logging.debug("Found %d data files: %s", len(data_list), data_list)
    data_list = [f for f in data_list if "all_biomes_merged" not in f.name]

    if not data_list:
        logging.critical(f"ERROR: No CSV files found in {data_dir}. Exiting.")
        return [], []

    site_data_dict = {}
    site_info_dict = {}

    # Load selected features from JSON — single source of truth
    with open(args.selected_features) as _sf:
        _sf_data = json.load(_sf)
    _sf_list = _sf_data.get("selected_features", _sf_data) if isinstance(_sf_data, dict) else _sf_data
    if not isinstance(_sf_list, list):
        raise ValueError(
            f"selected_features JSON must contain a list or a dict with 'selected_features' key. "
            f"Got: {type(_sf_list).__name__}"
        )
    if "pft" in _sf_list:
        raise ValueError(
            f"selected_features JSON contains raw 'pft' column. Use individual PFT names instead: {PFT_COLUMNS}"
        )
    used_cols = sorted(_sf_list)
    if TARGET_COL not in used_cols:
        used_cols = [TARGET_COL] + used_cols
    logging.info(f"Loaded {len(used_cols)} features from {args.selected_features}: {used_cols}")

    all_possible_pft_types = PFT_COLUMNS
    max_sap = 0
    max_site = None
    final_feature_names = None

    for data_file in data_list:
        try:
            df = pd.read_csv(data_file, parse_dates=["TIMESTAMP"])
            logging.info(f"Loaded data for site {data_file.name}, shape: {df.shape}")

            if IS_ONLY_DAY:
                if "sw_in" not in df.columns:
                    logging.warning(f"  Warning: 'sw_in' column not found in {data_file.name}. Skipping.")
                    continue
                df = df[df["sw_in"] > 10]
                if df.empty:
                    logging.warning(f"  Warning: No data after filtering for day time in {data_file.name}. Skipping.")
                    continue

            site_id = df["site_name"].iloc[0]
            if df[TARGET_COL].max() > max_sap:
                max_sap = df[TARGET_COL].max()
                max_site = data_file.name

            site_max = df[TARGET_COL].max()
            if site_max > 100:
                logging.info(f"!!! HIGH VELOCITY ALERT: Site {site_id} has max value: {site_max:.2f}")

            lat_col = next((col for col in df.columns if col.lower() in ["lat", "latitude_x"]), None)
            lon_col = next((col for col in df.columns if col.lower() in ["lon", "longitude_x"]), None)
            pft_col = next(
                (col for col in df.columns if col.lower() in ["pft", "plant_functional_type", "biome"]), None
            )

            logging.info(f"PFT column found: {pft_col}")

            if not (lat_col and lon_col and pft_col):
                logging.warning(f"  Warning: Lat/Lon/PFT not found for site {site_id}. Skipping.")
                continue

            latitude = df[lat_col].median()
            longitude = df[lon_col].median()

            if site_id.startswith("CZE"):
                logging.info(f"  Skipping site {site_id} as per rules.")
                continue

            df["latitude"] = latitude
            df["longitude"] = longitude
            # Add engineered features
            df = add_sap_flow_features(df, verbose=False)

            pft_value = df[pft_col].mode()[0]
            logging.debug(f"PFT value: {pft_value}")

            df.set_index("solar_TIMESTAMP", inplace=True)
            df.sort_index(inplace=True)  # Ensure chronological order
            df = add_time_features(df, datetime_column=None)

            # Apply feature engineering groups (if any requested)
            if feature_groups:
                df, _ = apply_feature_engineering(df, feature_groups, TIME_SCALE, verbose=True)

            # Create PFT one-hot columns if requested in selected_features
            requested_pft = [c for c in used_cols if c in PFT_COLUMNS]
            if requested_pft and pft_col and pft_col in df.columns:
                pft_cat = pd.Categorical(df[pft_col], categories=PFT_COLUMNS)
                pft_dummies = pd.get_dummies(pft_cat).astype(int)
                pft_dummies.index = df.index
                for col in pft_dummies.columns:
                    df[col] = pft_dummies[col]

            missing_cols = [col for col in used_cols if col not in df.columns]
            if missing_cols:
                logging.warning(f"Warning: Missing columns: {missing_cols} in {data_file.name}: skipping.")
                continue

            df = df[used_cols].copy()

            final_feature_names = [col for col in df.columns.tolist() if col != TARGET_COL]
            logging.info(f"Captured the final feature order: {final_feature_names}")
            logging.info(f"Total number of features before windowing: {len(final_feature_names)}")

            df = df.astype(np.float32)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)

            site_info = {"latitude": latitude, "longitude": longitude, "pft": pft_value}

            if IS_WINDOWING:
                min_segment_length = INPUT_WIDTH + SHIFT
                if len(df) < min_segment_length:
                    logging.warning(f"  Warning: Not enough data after cleaning for site {site_id}. Skipping.")
                    continue
                segments = TimeSeriesSegmenter.segment_time_series(
                    df,
                    gap_threshold=2,
                    unit="hours" if TIME_SCALE == "hourly" else "days",
                    min_segment_length=min_segment_length,
                )
                data_count = sum(len(s) for s in segments)
                if segments:
                    site_data_dict[site_id] = segments
                    site_info["data_count"] = data_count
                    site_info_dict[site_id] = site_info
                    logging.info(
                        f"  {site_id}: Successfully processed {len(segments)} segments with {data_count} total records"
                    )
                else:
                    logging.warning(f"  {site_id}: No valid segments after time series segmentation. Skipping.")
            else:
                site_data_dict[site_id] = df
                minimum_required_length = 0
                if len(df) < minimum_required_length:
                    logging.warning(f"  Warning: Not enough data after cleaning for site {site_id}. Skipping.")
                    site_data_dict.pop(site_id)
                    continue
                site_info["data_count"] = len(df)
                site_info_dict[site_id] = site_info

        except Exception as e:
            logging.error(f"Error processing {data_file.name}: {e}")

    logging.info(f"\nSite with max {TARGET_COL}: {max_site} with value {max_sap}")

    if not site_data_dict:
        logging.critical("ERROR: No valid site data could be processed.")
        return [], []

    # --- Prepare Data for CV Split ---
    site_ids = np.array(list(site_info_dict.keys()))
    latitudes = [site_info_dict[site]["latitude"] for site in site_ids]
    longitudes = [site_info_dict[site]["longitude"] for site in site_ids]
    data_counts = [site_info_dict[site]["data_count"] for site in site_ids]

    logging.info("\nCreating spatial groups...")

    logging.info(f"Grid size: {GRID_SIZE} degrees, R2 method: {R2_METHOD}")

    if IS_STRATIFIED:
        spatial_groups, group_stats = create_spatial_groups(
            lat=latitudes,
            lon=longitudes,
            method=spatial_split_method,
            lat_grid_size=GRID_SIZE,
            lon_grid_size=GRID_SIZE,
        )
    else:
        spatial_groups, group_stats = create_spatial_groups(
            lat=latitudes,
            lon=longitudes,
            data_counts=data_counts,
            n_groups=n_groups,
            method=spatial_split_method,
            balanced=BALANCED,
            lat_grid_size=GRID_SIZE,
            lon_grid_size=GRID_SIZE,
        )

    logging.info(f"Spatial groups assigned: {np.unique(spatial_groups)}")

    site_to_group = {site_id: group for site_id, group in zip(site_ids, spatial_groups, strict=False)}
    site_to_pft = {site_id: site_info_dict[site_id]["pft"] for site_id in site_ids}

    # Plot spatial grouping
    fig = plt.figure(figsize=(20, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="gray")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.3)

    colors = plt.cm.tab20.colors[:n_groups]
    color_map = ListedColormap(colors)

    scatter = ax.scatter(
        longitudes, latitudes, c=spatial_groups, s=100, edgecolor="k", cmap=color_map, transform=ccrs.PlateCarree()
    )

    cbar = plt.colorbar(scatter, ax=ax, label="Spatial Group", shrink=0.8)

    ax.set_extent(
        [min(longitudes) - 5, max(longitudes) + 5, min(latitudes) - 5, max(latitudes) + 5], crs=ccrs.PlateCarree()
    )

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Spatial Grouping of Sites")

    plt.savefig(plot_dir / f"spatial_groups_{run_id}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Process all data into unified, record-level numpy arrays
    logging.info("Preparing final RECORD-LEVEL data structures for all sites...")

    list_X, list_y, list_groups, list_pfts = [], [], [], []
    list_timestamps = []
    list_site_ids_str = []

    for site_id, raw_data in site_data_dict.items():
        if IS_WINDOWING:
            windows = create_windows_from_segments(
                segments=raw_data,
                input_width=INPUT_WIDTH,
                label_width=LABEL_WIDTH,
                shift=SHIFT,
                label_columns=[TARGET_COL],
                exclude_targets_from_features=EXCLUDE_TARGETS,
                exclude_labels_from_inputs=EXCLUDE_LABELS,
            )
            if not windows:
                continue
            X_site, y_site = convert_windows_to_numpy(windows)

            site_timestamps = []
            min_len = INPUT_WIDTH + SHIFT

            for segment in raw_data:
                if len(segment) < min_len:
                    continue
                start_index = min_len - 1
                # Use the DataFrame index (which contains TIMESTAMP) for timestamps
                valid_dates = segment.index.values[start_index:]
                site_timestamps.append(valid_dates)

            if not site_timestamps:
                continue

            timestamps_site_final = np.concatenate(site_timestamps)

            if len(timestamps_site_final) != len(y_site):
                logging.warning(f"Timestamp mismatch for {site_id}! y={len(y_site)}, t={len(timestamps_site_final)}")
                min_l = min(len(y_site), len(timestamps_site_final))
                timestamps_site_final = timestamps_site_final[:min_l]
        else:
            if raw_data.empty:
                continue
            # Extract timestamps from the index before modifying the DataFrame
            timestamps_site_final = raw_data.index.values
            y_site = raw_data.pop(TARGET_COL).values
            X_site = raw_data.values

        num_records = len(y_site)
        if num_records == 0:
            continue

        list_X.append(X_site)
        list_y.append(y_site)
        list_groups.append(np.full(num_records, site_to_group[site_id]))
        list_pfts.append(np.full(num_records, site_to_pft[site_id]))
        list_site_ids_str.append(np.full(num_records, site_id))
        list_timestamps.append(timestamps_site_final)

    del site_data_dict

    if not list_X:
        logging.critical("ERROR: No data records were generated after processing. Exiting.")
        return [], []

    X_all_records = np.vstack(list_X)
    y_all_records = np.concatenate(list_y)
    groups_all_records = np.concatenate(list_groups)
    pfts_all_records = np.concatenate(list_pfts)
    site_ids_all_records = np.concatenate(list_site_ids_str)
    timestamps_all = np.concatenate(list_timestamps)

    logging.info(f"Total records: {len(y_all_records)}")
    logging.info(f"Total timestamps: {len(timestamps_all)}")

    assert len(y_all_records) == len(timestamps_all), "Error: Timestamps do not align with Data!"
    logging.info(f"Total records processed and ready for CV: {len(y_all_records)}")

    pfts_encoded, pft_categories = pd.factorize(pfts_all_records)
    logging.info(f"Encoded PFTs into {len(pft_categories)} integer classes. All PFTs: {list(pft_categories)}")

    # --- Stratified Group K-Fold Cross-Validation ---
    logging.info("\nInitializing K-Fold Cross-Validation at RECORD LEVEL...")

    if IS_STRATIFIED:
        outer_cv = StratifiedGroupKFold(n_splits=n_groups, shuffle=True, random_state=RANDOM_SEED)
        y_all_stratified = pfts_encoded
        logging.info(f"Using StratifiedGroupKFold with {n_groups} splits, stratifying by PFTs.")
    else:
        outer_cv = GroupKFold(n_splits=n_groups)
        y_all_stratified = y_all_records
        logging.info(f"Using GroupKFold with {n_groups} splits, grouping by spatial groups.")

    # --- Prepare indices for selective scaling ---
    pft_cols = all_possible_pft_types

    base_numeric_indices = [i for i, col in enumerate(final_feature_names) if col not in pft_cols]

    if IS_WINDOWING:
        full_numeric_indices = []
        n_features = len(final_feature_names)
        for step in range(INPUT_WIDTH):
            offset = step * n_features
            step_indices = [idx + offset for idx in base_numeric_indices]
            full_numeric_indices.extend(step_indices)
        numeric_indices = np.array(full_numeric_indices)
    else:
        numeric_indices = np.array(base_numeric_indices)

    logging.info(
        f"Selective scaling: Scaling {len(numeric_indices)} numeric features, "
        f"preserving {X_all_records.shape[1] - len(numeric_indices)} categorical features."
    )

    # Diagnostic: Target Distribution by Fold
    logging.info("\n=== DIAGNOSTIC: Target Distribution by Fold ===")
    temp_split = outer_cv.split(X_all_records, y_all_stratified, groups_all_records)
    for fold_idx, (train_idx, test_idx) in enumerate(temp_split):
        y_train_temp = y_all_records[train_idx]
        y_test_temp = y_all_records[test_idx]
        logging.info(
            f"Fold {fold_idx + 1}: "
            f"Train max={y_train_temp.max():.1f}, Test max={y_test_temp.max():.1f}, "
            f"Train 95th={np.percentile(y_train_temp, 95):.1f}, Test 95th={np.percentile(y_test_temp, 95):.1f}"
        )

    # Reset the generator for actual training
    split_generator = outer_cv.split(X_all_records, y_all_stratified, groups_all_records)

    all_test_r2_scores, all_test_rmse_scores = [], []
    all_predictions, all_actuals = [], []
    all_test_pfts = []
    all_test_site_ids = []  # Track site IDs for each test prediction
    dist_analyzer = FoldDistributionAnalyzer(output_dir=str(plot_dir / "distributions"))

    for fold_idx, (train_val_indices, test_indices) in enumerate(split_generator):
        logging.info(f"\n=== Fold {fold_idx + 1}/{n_groups} ===")

        test_groups = np.unique(groups_all_records[test_indices])
        test_site_ids = [site_ids[np.where(spatial_groups == g)[0][0]] for g in test_groups]

        logging.info(f"Fold {fold_idx + 1} TEST sites: {test_site_ids}")
        for sid in test_site_ids:
            site_mask = groups_all_records[test_indices] == site_to_group[sid]
            site_y = y_all_records[test_indices][site_mask]
            logging.info(
                f"  {sid}: n={len(site_y)}, median={np.median(site_y):.1f}, "
                f"max={site_y.max():.1f}, PFT={site_to_pft[sid]}"
            )

        X_train_val = X_all_records[train_val_indices]
        y_train_val = y_all_records[train_val_indices]
        pfts_train_val = pfts_all_records[train_val_indices]
        groups_train_val = groups_all_records[train_val_indices]

        X_test = X_all_records[test_indices]
        y_test = y_all_records[test_indices]

        # Apply target transformation using the TargetTransformer
        if IS_TRANSFORM:
            # Create a new transformer for this fold (fit on training data only)
            fold_transformer = TargetTransformer(method=TRANSFORM_METHOD)
            y_train_val = fold_transformer.fit_transform(y_train_val)
            logging.info(f"  Applied {TRANSFORM_METHOD} transformation to training target")

        # Scaling X (numeric columns only)
        # Get the names of the numeric features being scaled
        # (Assuming you created 'numeric_indices' earlier based on 'final_feature_names')
        # If IS_WINDOWING is True,generate the windowed feature names first

        if IS_WINDOWING:
            # Generate the full list of windowed feature names matching the columns in X_all_records
            all_windowed_names = generate_windowed_feature_names(final_feature_names, INPUT_WIDTH)
            numeric_feature_names = [all_windowed_names[i] for i in numeric_indices]
        else:
            numeric_feature_names = [final_feature_names[i] for i in numeric_indices]

        # Scaling X (numeric columns only)
        scaler = StandardScaler()

        # 1. Create a DataFrame for the fit so the scaler learns the names
        X_train_numeric_df = pd.DataFrame(X_train_val[:, numeric_indices], columns=numeric_feature_names)

        # 2. Fit on the DataFrame
        scaler.fit(X_train_numeric_df)

        # 3. Transform (You can pass numpy array here, but passing DF is safer)
        # Note: We assign back to the numpy array, which is fine
        X_train_val[:, numeric_indices] = scaler.transform(X_train_numeric_df)

        # Transform test set
        X_test_numeric_df = pd.DataFrame(X_test[:, numeric_indices], columns=numeric_feature_names)
        X_test[:, numeric_indices] = scaler.transform(X_test_numeric_df)

        try:
            sw_in_base_index = final_feature_names.index("sw_in")
        except ValueError as e:
            raise ValueError("'sw_in' not found in the final feature list!") from e

        num_features = len(final_feature_names)
        SW_IN_INDEX = ((INPUT_WIDTH - 1) * num_features) + sw_in_base_index
        logging.info(f"SW_IN_INDEX for nighttime masking: {SW_IN_INDEX}")

        dist_analyzer.capture_fold_data(
            fold_idx=fold_idx, X_train=X_train_val, y_train=y_train_val, X_test=X_test, y_test=y_test
        )

        if IS_SHUFFLE:
            shuffled_indices = np.random.permutation(len(X_train_val))
            X_train_val, y_train_val = X_train_val[shuffled_indices], y_train_val[shuffled_indices]
            pfts_train_val, groups_train_val = pfts_train_val[shuffled_indices], groups_train_val[shuffled_indices]

        logging.info(f"  Train/Val records: {len(y_train_val)}, Test records: {len(y_test)}")

        # --- Hyperparameter Optimization and Training ---
        optimizer = MLOptimizer(
            param_grid=hyperparameters,
            scoring="neg_mean_squared_error",
            model_type=MODEL_TYPE,
            task="regression",
            random_state=RANDOM_SEED,
            n_splits=5,
        )

        optimizer.fit(
            X=X_train_val,
            y=y_train_val,
            is_cv=IS_CV,
            y_stratify=pfts_train_val,
            groups=groups_train_val,
            is_refit=True,
            split_type=SPLIT_TYPE,
            n_iterations=N_ITERATIONS,
        )
        best_model = optimizer.get_best_model()

        # --- Evaluation ---
        test_predictions = best_model.predict(X_test)
        if IS_TRANSFORM:
            # Use the fold transformer to inverse transform predictions
            test_predictions = fold_transformer.inverse_transform(test_predictions)

        y_test = np.asarray(y_test)
        test_predictions = np.asarray(test_predictions)
        pfts_test = pfts_all_records[test_indices]
        site_ids_test = site_ids_all_records[test_indices]  # Get site IDs for test set

        finite_mask = np.isfinite(y_test) & np.isfinite(test_predictions)
        n_total = y_test.shape[0]
        n_valid = finite_mask.sum()

        if n_valid == 0:
            logging.error(f"Fold {fold_idx + 1}: No finite prediction/actual pairs. Skipping fold.")
            continue

        if n_valid < n_total:
            logging.warning(f"Fold {fold_idx + 1}: Dropping {n_total - n_valid} rows with NaN/inf.")

        y_test_clean = y_test[finite_mask]
        preds_clean = test_predictions[finite_mask]
        pfts_test_clean = pfts_test[finite_mask]
        site_ids_test_clean = site_ids_test[finite_mask]  # Track site IDs
        all_test_pfts.extend(pfts_test_clean)
        all_test_site_ids.extend(site_ids_test_clean)  # Collect site IDs

        fold_test_r2 = r2_score(y_test_clean, preds_clean)
        fold_test_rmse = np.sqrt(mean_squared_error(y_test_clean, preds_clean))

        all_test_r2_scores.append(fold_test_r2)
        all_test_rmse_scores.append(fold_test_rmse)
        all_predictions.extend(test_predictions)
        all_actuals.extend(y_test)

        logging.info(f"  Fold {fold_idx + 1} Results -> Test R²: {fold_test_r2:.4f}, Test RMSE: {fold_test_rmse:.4f}")

        best_model.save_model(str(model_dir / f"spatial_{MODEL_TYPE}_fold_{fold_idx + 1}_{run_id}.json"))

        # --- Generate Individual Fold Plots ---
        logging.info(f"  Generating plots for Fold {fold_idx + 1}...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].scatter(
            y_test_clean, preds_clean, alpha=0.5, s=20, color="steelblue", edgecolors="black", linewidth=0.5
        )
        axes[0].set_xlabel("Observed Sap Velocity (cm³ cm⁻² h⁻¹)", fontsize=13)
        axes[0].set_ylabel("Predicted Sap Velocity (cm³ cm⁻² h⁻¹)", fontsize=13)
        axes[0].set_title(
            f"Fold {fold_idx + 1}: Predictions vs Actuals\n$R^2 = {fold_test_r2:.3f}$, RMSE = ${fold_test_rmse:.3f}$",
            fontsize=13,
            fontweight="bold",
        )

        min_val = min(y_test_clean.min(), preds_clean.min())
        max_val = max(y_test_clean.max(), preds_clean.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], "r-", alpha=0.8, linewidth=2, label="Perfect Prediction")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, alpha=0.3)
        axes[0].axis("equal")
        axes[0].axis("square")

        residuals = y_test_clean - preds_clean
        axes[1].scatter(preds_clean, residuals, alpha=0.5, s=20, color="coral", edgecolors="black", linewidth=0.5)
        axes[1].axhline(y=0, color="red", linestyle="--", linewidth=2, label="Zero Residual")
        axes[1].set_xlabel("Predicted Sap Velocity", fontsize=13)
        axes[1].set_ylabel("Residuals (Observed - Predicted)", fontsize=13)
        axes[1].set_title(
            f"Fold {fold_idx + 1}: Residual Plot\nMAE = ${mean_absolute_error(y_test_clean, preds_clean):.3f}$",
            fontsize=13,
            fontweight="bold",
        )
        axes[1].legend(loc="upper left")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fold_plot_path = plot_dir / f"fold_{fold_idx + 1}_performance_{run_id}_{spatial_split_method}.png"
        plt.savefig(fold_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(f"  Fold {fold_idx + 1} plot saved to: {fold_plot_path}")

        # Distribution comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(y_test_clean, bins=50, alpha=0.6, label="Observed", color="blue", edgecolor="black")
        axes[0].hist(preds_clean, bins=50, alpha=0.6, label="Predicted", color="orange", edgecolor="black")
        axes[0].set_xlabel("Sap Velocity", fontsize=12)
        axes[0].set_ylabel("Frequency", fontsize=12)
        axes[0].set_title(f"Fold {fold_idx + 1}: Distribution Comparison", fontsize=13, fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis="y")

        from scipy import stats

        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title(f"Fold {fold_idx + 1}: Q-Q Plot of Residuals", fontsize=13, fontweight="bold")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        dist_plot_path = plot_dir / f"fold_{fold_idx + 1}_distributions_{run_id}_{spatial_split_method}.png"
        plt.savefig(dist_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(f"  Fold {fold_idx + 1} distribution plot saved to: {dist_plot_path}")

        fold_results_df = pd.DataFrame(
            {
                "Observed": y_test_clean,
                "Predicted": preds_clean,
                "Residual": residuals,
                "Absolute_Error": np.abs(residuals),
            }
        )
        fold_csv_path = plot_dir / f"fold_{fold_idx + 1}_predictions_{run_id}_{spatial_split_method}.csv"
        fold_results_df.to_csv(fold_csv_path, index=False)
        logging.info(f"  Fold {fold_idx + 1} predictions saved to: {fold_csv_path}")

    dist_analyzer.generate_all_plots()

    # --- Final Results ---
    if all_test_r2_scores:
        mean_r2 = np.mean(all_test_r2_scores)
        std_r2 = np.std(all_test_r2_scores)
        mean_rmse = np.mean(all_test_rmse_scores)
        std_rmse = np.std(all_test_rmse_scores)

        # Pooled R2: single R2 from all concatenated OOF predictions
        arr_act = np.array(all_actuals)
        arr_pred = np.array(all_predictions)
        finite = np.isfinite(arr_act) & np.isfinite(arr_pred)
        pooled_r2 = r2_score(arr_act[finite], arr_pred[finite])
        pooled_rmse = np.sqrt(mean_squared_error(arr_act[finite], arr_pred[finite]))

        logging.info("\n=== OVERALL SPATIAL CROSS-VALIDATION RESULTS ===")
        logging.info(f"R2 method: {R2_METHOD}")
        logging.info(f"Mean-fold R²: {mean_r2:.4f} ± {std_r2:.4f}")
        logging.info(f"Pooled R²:    {pooled_r2:.4f}")
        logging.info(f"Mean-fold RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
        logging.info(f"Pooled RMSE:    {pooled_rmse:.4f}")

        plt.figure(figsize=(8, 8))
        plt.scatter(all_actuals, all_predictions, alpha=0.5, s=20)
        plt.xlabel("Observed Sap Velocity (cm³ cm⁻² h⁻¹)", fontsize=14)
        plt.ylabel("Predicted Sap Velocity", fontsize=14)
        plt.title(
            f"Spatial CV Results\n$R^2 = {mean_r2:.3f} \\pm {std_r2:.3f}$, "
            f"RMSE = ${mean_rmse:.3f} \\pm {std_rmse:.3f}$",
            fontsize=12,
        )

        min_val = min(min(all_actuals), min(all_predictions))
        max_val = max(max(all_actuals), max(all_predictions))
        plt.plot([min_val, max_val], [min_val, max_val], "r-", alpha=0.8, linewidth=2)

        plt.grid(True, alpha=0.3)
        plt.axis("equal")
        plt.axis("square")
        plt.tight_layout()
        # Ensure plot directory exists (OneDrive sync can sometimes cause issues)
        import os

        save_path = plot_dir / f"spatial_cv_predictions_vs_actual_{run_id}_{spatial_split_method}.png"
        parent_dir = str(save_path.parent)
        logging.info(f"Creating directory: {parent_dir}")
        os.makedirs(parent_dir, exist_ok=True)
        # Verify directory was created
        if not os.path.exists(parent_dir):
            logging.error(f"Failed to create directory: {parent_dir}")
            raise FileNotFoundError(f"Could not create directory: {parent_dir}")
        logging.info(f"Saving plot to: {save_path}")
        plt.savefig(str(save_path), dpi=300)
        plt.close()

        logging.info(f"\nResults plot saved to: {save_path}")

        # PFT-stratified scatter plots
        logging.info("Generating PFT-stratified performance plots...")

        arr_actuals = np.array(all_actuals)
        arr_predictions = np.array(all_predictions)
        arr_pfts = np.array(all_test_pfts)

        unique_pfts = np.unique(arr_pfts)
        n_pfts = len(unique_pfts)

        if n_pfts > 0:
            n_cols = 3
            n_rows = (n_pfts + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
            axes = axes.flatten()

            g_min = min(arr_actuals.min(), arr_predictions.min())
            g_max = max(arr_actuals.max(), arr_predictions.max())

            for i, pft in enumerate(unique_pfts):
                ax = axes[i]
                mask = arr_pfts == pft
                pft_actuals = arr_actuals[mask]
                pft_preds = arr_predictions[mask]

                if len(pft_actuals) < 10:
                    ax.text(0.5, 0.5, f"{pft}\nInsufficient Data (n={len(pft_actuals)})", ha="center", va="center")
                    continue

                pft_r2 = r2_score(pft_actuals, pft_preds)
                pft_rmse = np.sqrt(mean_squared_error(pft_actuals, pft_preds))

                ax.scatter(pft_actuals, pft_preds, alpha=0.4, s=15, color="teal", edgecolor="k", linewidth=0.3)
                ax.plot([g_min, g_max], [g_min, g_max], "r--", alpha=0.7)
                ax.set_title(
                    f"PFT: {pft} (n={len(pft_actuals)})\nR²={pft_r2:.3f}, RMSE={pft_rmse:.3f}", fontweight="bold"
                )
                ax.set_xlabel("Observed")
                ax.set_ylabel("Predicted")
                ax.grid(True, alpha=0.3)
                ax.set_xlim(g_min, g_max)
                ax.set_ylim(g_min, g_max)
                ax.set_aspect("equal")

            for j in range(i + 1, len(axes)):
                axes[j].axis("off")

            plt.tight_layout()
            pft_plot_path = plot_dir / f"pft_performance_{run_id}.png"
            plt.savefig(pft_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            logging.info(f"PFT performance plot saved to: {pft_plot_path}")

            pft_metrics = []
            for pft in unique_pfts:
                mask = arr_pfts == pft
                if np.sum(mask) > 0:
                    y_p = arr_actuals[mask]
                    y_hat_p = arr_predictions[mask]
                    pft_metrics.append(
                        {
                            "PFT": pft,
                            "N_Samples": len(y_p),
                            "R2": r2_score(y_p, y_hat_p),
                            "RMSE": np.sqrt(mean_squared_error(y_p, y_hat_p)),
                            "MAE": mean_absolute_error(y_p, y_hat_p),
                        }
                    )
            pd.DataFrame(pft_metrics).to_csv(plot_dir / f"pft_metrics_{run_id}.csv", index=False)

        # =========================================================================
        # SITE-LEVEL PERFORMANCE ANALYSIS
        # =========================================================================
        logging.info("\n" + "=" * 60)
        logging.info("=== SITE-LEVEL PERFORMANCE ANALYSIS ===")
        logging.info("=" * 60)

        arr_site_ids = np.array(all_test_site_ids)
        unique_sites = np.unique(arr_site_ids)

        site_metrics = []
        for site in unique_sites:
            mask = arr_site_ids == site
            if np.sum(mask) > 1:  # Need at least 2 samples for R2
                y_site = arr_actuals[mask]
                y_hat_site = arr_predictions[mask]

                # Calculate metrics
                site_r2 = r2_score(y_site, y_hat_site)
                site_rmse = np.sqrt(mean_squared_error(y_site, y_hat_site))
                site_mae = mean_absolute_error(y_site, y_hat_site)

                # Calculate bias (mean residual)
                residuals = y_site - y_hat_site
                site_bias = np.mean(residuals)

                # Get site PFT
                site_pft = arr_pfts[mask][0]

                site_metrics.append(
                    {
                        "site_name": site,
                        "PFT": site_pft,
                        "N_Samples": len(y_site),
                        "R2": site_r2,
                        "RMSE": site_rmse,
                        "MAE": site_mae,
                        "Bias": site_bias,
                        "Mean_Observed": np.mean(y_site),
                        "Mean_Predicted": np.mean(y_hat_site),
                        "Std_Observed": np.std(y_site),
                        "Std_Predicted": np.std(y_hat_site),
                    }
                )

        site_metrics_df = pd.DataFrame(site_metrics)
        site_metrics_df = site_metrics_df.sort_values("R2", ascending=True)
        site_metrics_df.to_csv(plot_dir / f"site_metrics_{run_id}.csv", index=False)
        logging.info(f"Site-level metrics saved to: {plot_dir / f'site_metrics_{run_id}.csv'}")

        # Log worst performing sites
        logging.info("\n=== WORST PERFORMING SITES (Bottom 10 by R²) ===")
        worst_sites = site_metrics_df.head(10)
        for _, row in worst_sites.iterrows():
            logging.info(
                f"  {row['site_name']}: R²={row['R2']:.3f}, RMSE={row['RMSE']:.3f}, "
                f"Bias={row['Bias']:.3f}, N={row['N_Samples']}, PFT={row['PFT']}"
            )

        # Log best performing sites
        logging.info("\n=== BEST PERFORMING SITES (Top 10 by R²) ===")
        best_sites = site_metrics_df.tail(10).iloc[::-1]  # Reverse to show best first
        for _, row in best_sites.iterrows():
            logging.info(
                f"  {row['site_name']}: R²={row['R2']:.3f}, RMSE={row['RMSE']:.3f}, "
                f"Bias={row['Bias']:.3f}, N={row['N_Samples']}, PFT={row['PFT']}"
            )

        # Sites with negative R² (predictions worse than mean)
        negative_r2_sites = site_metrics_df[site_metrics_df["R2"] < 0]
        if len(negative_r2_sites) > 0:
            logging.info(f"\n=== SITES WITH NEGATIVE R² ({len(negative_r2_sites)} sites) ===")
            for _, row in negative_r2_sites.iterrows():
                logging.info(
                    f"  {row['site_name']}: R²={row['R2']:.3f}, RMSE={row['RMSE']:.3f}, "
                    f"Bias={row['Bias']:.3f}, N={row['N_Samples']}, PFT={row['PFT']}"
                )

        # Create site performance visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. R² distribution
        axes[0, 0].hist(site_metrics_df["R2"], bins=30, edgecolor="black", alpha=0.7)
        axes[0, 0].axvline(x=0, color="red", linestyle="--", linewidth=2, label="R²=0")
        axes[0, 0].axvline(
            x=site_metrics_df["R2"].median(),
            color="green",
            linestyle="-",
            linewidth=2,
            label=f"Median={site_metrics_df['R2'].median():.3f}",
        )
        axes[0, 0].set_xlabel("R² Score", fontsize=12)
        axes[0, 0].set_ylabel("Number of Sites", fontsize=12)
        axes[0, 0].set_title("Distribution of Site-Level R² Scores", fontsize=13, fontweight="bold")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. R² vs Sample Size
        scatter = axes[0, 1].scatter(
            site_metrics_df["N_Samples"],
            site_metrics_df["R2"],
            c=site_metrics_df["RMSE"],
            cmap="viridis",
            alpha=0.7,
            s=50,
        )
        axes[0, 1].axhline(y=0, color="red", linestyle="--", linewidth=1.5)
        axes[0, 1].set_xlabel("Number of Samples", fontsize=12)
        axes[0, 1].set_ylabel("R² Score", fontsize=12)
        axes[0, 1].set_title("R² vs Sample Size (colored by RMSE)", fontsize=13, fontweight="bold")
        plt.colorbar(scatter, ax=axes[0, 1], label="RMSE")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Bias distribution
        axes[1, 0].hist(site_metrics_df["Bias"], bins=30, edgecolor="black", alpha=0.7, color="coral")
        axes[1, 0].axvline(x=0, color="red", linestyle="--", linewidth=2)
        axes[1, 0].set_xlabel("Bias (Observed - Predicted)", fontsize=12)
        axes[1, 0].set_ylabel("Number of Sites", fontsize=12)
        axes[1, 0].set_title("Distribution of Site-Level Bias", fontsize=13, fontweight="bold")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. R² by PFT (boxplot)
        pft_order = site_metrics_df.groupby("PFT")["R2"].median().sort_values().index
        site_metrics_df["PFT_ordered"] = pd.Categorical(site_metrics_df["PFT"], categories=pft_order, ordered=True)
        site_metrics_df_sorted = site_metrics_df.sort_values("PFT_ordered")

        unique_pfts_for_box = site_metrics_df_sorted["PFT"].unique()
        box_data = [
            site_metrics_df_sorted[site_metrics_df_sorted["PFT"] == pft]["R2"].values for pft in unique_pfts_for_box
        ]

        bp = axes[1, 1].boxplot(box_data, labels=unique_pfts_for_box, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
        axes[1, 1].axhline(y=0, color="red", linestyle="--", linewidth=1.5)
        axes[1, 1].set_xlabel("Plant Functional Type", fontsize=12)
        axes[1, 1].set_ylabel("R² Score", fontsize=12)
        axes[1, 1].set_title("R² Distribution by PFT", fontsize=13, fontweight="bold")
        axes[1, 1].tick_params(axis="x", rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        site_plot_path = plot_dir / f"site_performance_{run_id}.png"
        plt.savefig(str(site_plot_path), dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Site performance plot saved to: {site_plot_path}")

    # =========================================================================
    # TRAINING FINAL MODEL ON ALL DATA
    # =========================================================================
    logging.info("\n" + "=" * 60)
    logging.info("=== TRAINING FINAL MODEL ON ALL DATA ===")
    logging.info("=" * 60)

    logging.info("Applying preprocessing to all data (Numeric Only)...")

    # Generate feature names for the columns corresponding to numeric_indices
    if IS_WINDOWING:
        all_windowed_names = generate_windowed_feature_names(final_feature_names, INPUT_WIDTH)
        numeric_feature_names = [all_windowed_names[i] for i in numeric_indices]
    else:
        numeric_feature_names = [final_feature_names[i] for i in numeric_indices]

    scaler_final = StandardScaler()
    X_all_scaled = X_all_records.copy()

    # Create DataFrame for fitting
    X_all_numeric_df = pd.DataFrame(X_all_scaled[:, numeric_indices], columns=numeric_feature_names)

    # Fit on DataFrame -> Scaler stores feature names
    scaler_final.fit(X_all_numeric_df)

    # Transform
    X_all_scaled[:, numeric_indices] = scaler_final.transform(X_all_numeric_df)

    # Apply target transformation using the global transformer
    if IS_TRANSFORM:
        y_all_transformed = target_transformer.fit_transform(y_all_records)
        logging.info(f"Applied {TRANSFORM_METHOD} transformation to all target data")
    else:
        y_all_transformed = y_all_records

    logging.info(f"Data shapes: X={X_all_scaled.shape}, y={y_all_transformed.shape}")

    logging.info("Running hyperparameter optimization on all data...")

    final_optimizer = MLOptimizer(
        param_grid=hyperparameters,
        scoring="neg_mean_squared_error",
        model_type=MODEL_TYPE,
        task="regression",
        random_state=RANDOM_SEED,
        n_splits=5,
    )

    final_optimizer.fit(
        X=X_all_scaled,
        y=y_all_transformed,
        is_cv=True,
        y_stratify=pfts_encoded,
        groups=groups_all_records,
        is_refit=True,
        split_type=SPLIT_TYPE,
        n_iterations=N_ITERATIONS,
    )

    final_model = final_optimizer.get_best_model()
    best_params = final_optimizer.best_params_
    best_cv_score = final_optimizer.best_score_

    logging.info(f"Best hyperparameters: {best_params}")
    logging.info(f"Best CV score (neg_mse): {best_cv_score:.4f}")

    final_model_path = model_dir / f"FINAL_{MODEL_TYPE}_{run_id}.joblib"
    joblib.dump(final_model, final_model_path)
    logging.info(f"Final model saved to: {final_model_path}")

    final_scaler_path = model_dir / f"FINAL_scaler_{run_id}_feature.pkl"
    joblib.dump(scaler_final, final_scaler_path)
    logging.info(f"Scaler saved to: {final_scaler_path}")

    # Save target transformer if transformation was applied
    if IS_TRANSFORM:
        transformer_path = model_dir / f"FINAL_target_transformer_{run_id}.pkl"
        joblib.dump(target_transformer, transformer_path)
        logging.info(f"Target transformer saved to: {transformer_path}")

    # =========================================================================
    # GENERATE FEATURE NAMES FOR SHAP (HANDLES WINDOWING)
    # =========================================================================
    if IS_WINDOWING:
        shap_feature_names = generate_windowed_feature_names(final_feature_names, INPUT_WIDTH)
        logging.info(f"Generated {len(shap_feature_names)} windowed feature names for SHAP")
    else:
        shap_feature_names = final_feature_names
        logging.info(f"Using {len(shap_feature_names)} feature names for SHAP")

        # Save config/metadata
    final_config = {
        "model_type": MODEL_TYPE,
        "run_id": run_id,
        "best_params": best_params,
        "best_cv_score": float(best_cv_score),
        "cv_results": {
            "mean_r2": float(mean_r2),
            "pooled_r2": float(pooled_r2),
            "std_r2": float(std_r2),
            "mean_rmse": float(mean_rmse),
            "pooled_rmse": float(pooled_rmse),
            "std_rmse": float(std_rmse),
            "n_folds": n_groups,
            "grid_size": GRID_SIZE,
            "r2_method": R2_METHOD,
        },
        "preprocessing": {
            "target_transform": TRANSFORM_METHOD if IS_TRANSFORM else None,
            "target_transform_params": target_transformer.get_params() if IS_TRANSFORM else None,
            "feature_scaling": "StandardScaler",
        },
        "data_info": {
            "n_samples": len(y_all_records),
            "n_features": X_all_records.shape[1],
            "IS_WINDOWING": IS_WINDOWING,
            "input_width": INPUT_WIDTH if IS_WINDOWING else None,
            "label_width": LABEL_WIDTH if IS_WINDOWING else None,
            "shift": SHIFT if IS_WINDOWING else None,
        },
        "feature_names": final_feature_names,
        "shap_feature_names": shap_feature_names,
        "random_seed": RANDOM_SEED,
        "split_type": SPLIT_TYPE,
        "time_scale": TIME_SCALE,
    }

    config_path = model_dir / f"FINAL_config_{run_id}.json"
    with open(config_path, "w") as f:
        json.dump(final_config, f, indent=2, default=str)
    logging.info(f"Config saved to: {config_path}")

    # Save SHAP context bundle (for standalone SHAP analysis)
    shap_context_path = model_dir / f"SHAP_context_{run_id}.npz"
    np.savez_compressed(
        shap_context_path,
        X_all_scaled=X_all_scaled,
        X_all_records=X_all_records,
        y_all_records=y_all_records,
        site_ids=site_ids_all_records,
        timestamps=timestamps_all,
        groups=groups_all_records,
        pfts=pfts_all_records,
    )
    logging.info(f"SHAP context bundle saved to: {shap_context_path}")

    # Save site_info_dict for standalone SHAP
    site_info_path = model_dir / f"site_info_{run_id}.json"
    with open(site_info_path, "w") as f:
        json.dump(
            {
                k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()}
                for k, v in site_info_dict.items()
            },
            f,
            indent=2,
            default=str,
        )
    logging.info(f"Site info saved to: {site_info_path}")

    # Sanity check
    logging.info("Running sanity check on training data...")
    y_pred_all_transformed = final_model.predict(X_all_scaled)
    if IS_TRANSFORM:
        y_pred_all = target_transformer.inverse_transform(y_pred_all_transformed)
    else:
        y_pred_all = y_pred_all_transformed

    sanity_r2 = r2_score(y_all_records, y_pred_all)
    sanity_rmse = np.sqrt(mean_squared_error(y_all_records, y_pred_all))
    sanity_mae = mean_absolute_error(y_all_records, y_pred_all)

    logging.info("Sanity check (on training data):")
    logging.info(f"  R²   = {sanity_r2:.4f}")
    logging.info(f"  RMSE = {sanity_rmse:.4f}")
    logging.info(f"  MAE  = {sanity_mae:.4f}")
    logging.info("(Note: These metrics are on training data - expect better than CV)")

    # Plot sanity check
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(y_all_records, y_pred_all, alpha=0.3, s=10, color="steelblue")
    axes[0].set_xlabel("Observed Sap Velocity (cm³ cm⁻² h⁻¹)", fontsize=13)
    axes[0].set_ylabel("Predicted Sap Velocity (cm³ cm⁻² h⁻¹)", fontsize=13)
    axes[0].set_title(
        f"Final Model: Training Data Fit\n"
        f"$R^2 = {sanity_r2:.3f}$, RMSE = ${sanity_rmse:.3f}$\n"
        f"(CV estimate: $R^2 = {mean_r2:.3f} \\pm {std_r2:.3f}$)",
        fontsize=12,
        fontweight="bold",
    )

    min_val = min(y_all_records.min(), y_pred_all.min())
    max_val = max(y_all_records.max(), y_pred_all.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], "r-", linewidth=2, label="1:1 Line")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect("equal", "box")

    residuals_all = y_all_records - y_pred_all
    axes[1].scatter(y_pred_all, residuals_all, alpha=0.3, s=10, color="coral")
    axes[1].axhline(y=0, color="red", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Predicted Sap Velocity", fontsize=13)
    axes[1].set_ylabel("Residuals (Observed - Predicted)", fontsize=13)
    axes[1].set_title(f"Final Model: Residual Plot\nMAE = ${sanity_mae:.3f}$", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_dir / f"FINAL_model_sanity_check_{run_id}.png", dpi=300, bbox_inches="tight")
    plt.close()

    logging.info("\n" + "=" * 60)
    logging.info("=== FINAL MODEL TRAINING COMPLETE ===")
    logging.info("=" * 60)
    logging.info(f"Model:   {final_model_path}")
    logging.info(f"Scaler:  {final_scaler_path}")
    logging.info(f"Config:  {config_path}")
    logging.info("")
    logging.info("Expected generalization performance (from CV):")
    logging.info(f"  R²   = {mean_r2:.4f} ± {std_r2:.4f}")
    logging.info(f"  RMSE = {mean_rmse:.4f} ± {std_rmse:.4f}")

    # =========================================================================
    # RUN SHAP ANALYSIS (WITH STATIC FEATURE AGGREGATION)
    # =========================================================================
    logging.info("\n" + "=" * 60)
    logging.info("=== RUNNING SHAP ANALYSIS ===")
    logging.info("=" * 60)

    # Define static features for your dataset
    # These are features that don't change within a time window
    STATIC_FEATURES = [
        "canopy_height",
        "elevation",
        "prcip/PET",
        # PFT one-hot encoded columns
        "MF",
        "DNF",
        "ENF",
        "EBF",
        "WSA",
        "WET",
        "DBF",
        "SAV",
    ]

    # Prepare data for SHAP
    X_for_shap_calculation = X_all_scaled
    X_for_plotting_axes = X_all_records

    # Re-extract coordinates and site IDs for the full dataset
    lat_all = np.array([site_info_dict[s]["latitude"] for s in site_ids_all_records])
    lon_all = np.array([site_info_dict[s]["longitude"] for s in site_ids_all_records])

    try:
        # =================================================================
        # STEP 1: Calculate raw SHAP values
        # =================================================================
        logging.info("Step 1: Calculating raw SHAP values...")

        # Sample for SHAP (computation is expensive)
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
        y_sampled = y_all_records[sampled_indices]  # Observed values (original scale)

        logging.info(f"SHAP calculation on {len(X_shap)} samples")

        # Create SHAP explainer and calculate values
        logging.info("Creating SHAP TreeExplainer...")
        explainer = shap.TreeExplainer(final_model)

        logging.info("Calculating SHAP values (this may take a while)...")
        shap_values_raw = explainer.shap_values(X_shap)

        # Handle multi-output models
        if isinstance(shap_values_raw, list):
            logging.info("Model returned list of SHAP values, using first element")
            shap_values_raw = shap_values_raw[0]

        logging.info(f"Raw SHAP values shape: {shap_values_raw.shape}")

        # Get base value
        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[0] if len(base_value) > 0 else base_value
        logging.info(f"Base value (expected_value): {base_value}")

        # =================================================================
        # STEP 2: Aggregate static features (if windowing)
        # =================================================================
        if IS_WINDOWING:
            logging.info("\nStep 2: Aggregating SHAP values for static features...")

            # Aggregate SHAP values
            shap_values_agg, feature_names_agg = aggregate_static_feature_shap(
                shap_values=shap_values_raw,
                windowed_feature_names=shap_feature_names,
                base_feature_names=final_feature_names,
                input_width=INPUT_WIDTH,
                static_features=STATIC_FEATURES,
                aggregation="sum",
            )

            # Aggregate X values for plotting
            X_original_agg, _ = aggregate_static_feature_values(
                X_original=X_original_sampled,
                base_feature_names=final_feature_names,
                input_width=INPUT_WIDTH,
                static_features=STATIC_FEATURES,
            )

            # Use aggregated values for main plots
            shap_values = shap_values_agg
            shap_feature_names_final = feature_names_agg
            X_for_plots = X_original_agg

            # Also keep raw values for time-step specific analysis
            shap_values_windowed = shap_values_raw
            shap_feature_names_windowed = shap_feature_names

            logging.info(f"Aggregated SHAP values shape: {shap_values.shape}")
            logging.info(f"Aggregated feature count: {len(shap_feature_names_final)}")
        else:
            shap_values = shap_values_raw
            shap_feature_names_final = shap_feature_names
            X_for_plots = X_original_sampled

            shap_values_windowed = None
            shap_feature_names_windowed = None

        # Create DataFrames for easier handling
        df_shap = pd.DataFrame(shap_values, columns=shap_feature_names_final)
        df_X = pd.DataFrame(X_for_plots, columns=shap_feature_names_final)

        # =================================================================
        # STEP 3: Generate Global Importance Plots
        # =================================================================
        logging.info("\nStep 3 & 4: Generating Grouped PFT Plots...")

        # 1. DEFINE PFT COLUMNS
        pft_cols_to_group = ["MF", "DNF", "ENF", "EBF", "WSA", "WET", "DBF", "SAV"]

        # 2. PERFORM GROUPING
        # We perform this ONCE so all plots use the exact same data structure
        shap_values_grouped, df_X_grouped, feature_names_grouped = group_pft_for_summary_plots(
            shap_values=shap_values, X_df=df_X, feature_names=shap_feature_names_final, pft_cols=pft_cols_to_group
        )

        logging.info(f"Grouped Data Shape: {shap_values_grouped.shape}")

        # ---------------------------------------------------------
        # PLOT 3a: Beeswarm Summary (Grouped)
        # ---------------------------------------------------------
        logging.info("  Generating Beeswarm plot (PFT grouped)...")
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values_grouped, df_X_grouped, show=False, max_display=20)
        plt.xlabel(get_shap_label(), fontsize=12)
        plt.title("Feature Contribution (Land Cover Grouped as 'PFT')\n", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(plot_dir / "shap_summary_beeswarm_grouped.png", dpi=300, bbox_inches="tight")
        plt.close()

        # ---------------------------------------------------------
        # PLOT 3b: Bar Plot (Grouped)
        # ---------------------------------------------------------
        logging.info("  Generating Bar plot (PFT grouped)...")
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values_grouped, df_X_grouped, plot_type="bar", show=False, max_display=20)
        plt.xlabel(f"Mean |SHAP Value| ({SHAP_UNITS})", fontsize=12)
        plt.title("Feature Importance (Land Cover Grouped as 'PFT')\n", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(plot_dir / "shap_global_importance_bar_grouped.png", dpi=300, bbox_inches="tight")
        plt.close()

        # ---------------------------------------------------------
        # PLOT 4: Smart Partial Dependence Plots (Grouped)
        # ---------------------------------------------------------
        logging.info("  Generating Hybrid Partial Dependence Plots (Scatter for Numeric, Box for PFT)...")

        # Calculate importance using the GROUPED values
        mean_abs_shap = np.abs(shap_values_grouped).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:9]  # Top 9
        top_features = [feature_names_grouped[i] for i in top_indices]

        logging.info(f"  Top 9 features (after grouping): {top_features}")

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.flatten()

        for i, feature in enumerate(top_features):
            ax = axes[i]

            # Get data for this feature
            x_val = df_X_grouped[feature].values
            y_val = shap_values_grouped[:, feature_names_grouped.index(feature)]

            # --- SPECIAL HANDLING FOR PFT (Categorical) ---
            if feature == "PFT":
                # Convert numeric codes back to string labels for the plot
                # We do this by looking at which PFT column was active in the original df_X

                # 1. Reconstruct labels from original data for mapping
                temp_X_pft = df_X[pft_cols_to_group]
                # Series of strings ('ENF', 'MF', etc.)
                pft_labels_series = temp_X_pft.idxmax(axis=1)

                # Prepare DataFrame for Seaborn
                plot_df = pd.DataFrame({"PFT": pft_labels_series.values, "SHAP": y_val})

                # Sort order by median SHAP value for cleaner look
                order = plot_df.groupby("PFT")["SHAP"].median().sort_values().index

                # Draw Boxplot
                sns.boxplot(data=plot_df, x="PFT", y="SHAP", ax=ax, palette="Set2", order=order)
                ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
                ax.set_title(f"Effect of {feature}", fontsize=12, fontweight="bold")
                ax.set_ylabel(get_shap_label(), fontsize=10)
                ax.set_xlabel("")  # Labels are on ticks
                ax.tick_params(axis="x", rotation=45)
                # ax.grid(True, alpha=0.3, axis='y')

            # --- HANDLING FOR NUMERIC FEATURES ---
            else:
                # Filter valid values
                valid_mask = np.isfinite(x_val) & np.isfinite(y_val)
                x_valid = x_val[valid_mask]
                y_valid = y_val[valid_mask]

                if len(x_valid) == 0:
                    ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
                    continue

                # Scatter plot
                ax.scatter(x_valid, y_valid, alpha=0.3, color="steelblue", s=10)

                # Add smooth curve (LOWESS) — skip if it fails (e.g. constant data)
                with contextlib.suppress(Exception):
                    sns.regplot(
                        x=x_valid, y=y_valid, scatter=False, lowess=True, ax=ax, color="red", line_kws={"linewidth": 2}
                    )

                ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

                # Add units
                feat_unit = get_feature_unit(feature)
                xlabel = f"{feature} ({feat_unit})" if feat_unit else feature

                ax.set_title(f"Effect of {feature}", fontsize=12, fontweight="bold")
                ax.set_xlabel(xlabel, fontsize=10)
                ax.set_ylabel(get_shap_label(), fontsize=10)
                # ax.grid(True, alpha=0.3)

        plt.suptitle("Partial Dependence Plots (Top 9 Features)\n(PFT Grouped)", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(plot_dir / "shap_partial_dependence_grouped.png", dpi=300, bbox_inches="tight")
        plt.close()

        # =================================================================
        # STEP 5: Spatial SHAP Maps
        # =================================================================
        logging.info("\nStep 5: Generating Spatial SHAP Maps...")

        # Use top features calculated in Step 4 (which might include 'PFT')
        top_4_features = top_features[:4]

        fig = plt.figure(figsize=(20, 12))

        for i, feature in enumerate(top_4_features):
            ax = fig.add_subplot(2, 2, i + 1, projection=ccrs.PlateCarree())

            # Add map features
            ax.add_feature(cfeature.BORDERS, linestyle=":")
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)

            # --- FIX START: Handle 'PFT' vs Standard Features ---
            if feature == "PFT":
                # If feature is PFT, get values from the grouped array
                # (feature_names_grouped and shap_values_grouped come from Step 3)
                pft_index = feature_names_grouped.index("PFT")
                vals = shap_values_grouped[:, pft_index]
            else:
                # Otherwise get from the original dataframe
                vals = df_shap[feature].values
            # --- FIX END ---

            # Filter valid values
            valid_mask = np.isfinite(vals) & np.isfinite(lon_sampled) & np.isfinite(lat_sampled)

            if valid_mask.sum() == 0:
                ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes, fontsize=12)
                continue

            vals_valid = vals[valid_mask]
            lon_valid = lon_sampled[valid_mask]
            lat_valid = lat_sampled[valid_mask]

            # Diverging colormap centered at 0
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

            # Colorbar with units
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
            cbar.set_label(f"SHAP Value ({SHAP_UNITS})", fontsize=10)

            # Title with feature unit
            if feature == "PFT":
                title = "Spatial Contribution: Land Cover (PFT)"
            else:
                feat_unit = get_feature_unit(feature)
                if feat_unit:
                    title = f"Spatial Contribution: {feature}\n({feat_unit})"
                else:
                    title = f"Spatial Contribution: {feature}"

            ax.set_title(title, fontsize=13, fontweight="bold")

            # Set extent based on data
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
        # STEP 6: Waterfall Plots (High/Low Predictions)
        # =================================================================
        logging.info("\nStep 6: Generating Local Waterfall Plots...")

        pred_vals = final_model.predict(X_shap)
        high_idx = np.argmax(pred_vals)
        low_idx = np.argmin(pred_vals)

        for idx, name in [(high_idx, "High_Flow"), (low_idx, "Low_Flow")]:
            plt.figure(figsize=(12, 9))

            # Create Explanation object
            row_explainer = shap.Explanation(
                values=shap_values[idx],
                base_values=float(base_value),
                data=X_for_plots[idx],
                feature_names=shap_feature_names_final,
            )

            shap.plots.waterfall(row_explainer, show=False, max_display=12)

            # Get predicted value for title
            if IS_TRANSFORM:
                pred_original = target_transformer.inverse_transform(np.array([pred_vals[idx]]))[0]
            else:
                pred_original = pred_vals[idx]

            plt.title(
                f"Why did the model predict {name}?\n"
                f"Site: {site_ids_sampled[idx]} | "
                f"Predicted: {pred_original:.2f} {SHAP_UNITS}",
                fontsize=12,
                fontweight="bold",
            )
            plt.xlabel(f"SHAP Value ({SHAP_UNITS})", fontsize=11)
            plt.tight_layout()
            plt.savefig(plot_dir / f"shap_waterfall_{name}.png", dpi=300, bbox_inches="tight")
            plt.close()
            logging.info(f"    Saved: shap_waterfall_{name}.png")

        # =================================================================
        # STEP 7: Seasonal Drivers by Hemisphere (Figure 7)
        # =================================================================
        logging.info("\nStep 7: Generating Seasonal Driver Analysis (Figure 7)...")

        try:
            plot_seasonal_drivers_by_hemisphere(
                shap_values=shap_values,
                feature_names=shap_feature_names_final,
                timestamps=timestamps_sampled,
                latitudes=lat_sampled,
                top_n=5,
                output_dir=plot_dir,
            )
            logging.info("    Generated Figure 7 (Seasonal Drivers by Hemisphere)")
        except Exception as e:
            logging.warning(f"    Could not generate Fig 7 (Seasonal): {e}")
            import traceback

            traceback.print_exc()

        # =================================================================
        # STEP 8: Diurnal Drivers (Hourly Data Only)
        # =================================================================
        if TIME_SCALE == "hourly":
            logging.info("\nStep 8: Generating Diurnal Driver Analysis (Grouped PFT, All Variables)...")

            try:
                # 1. Aggregate PFTs into a single feature for the diurnal data
                # This removes 'MF', 'ENF', etc. and creates a single 'PFT' column
                shap_values_diurnal, feature_names_diurnal = aggregate_pft_shap_values(
                    shap_values=shap_values,
                    feature_names=shap_feature_names_final,
                    pft_columns=PFT_COLUMNS,
                    aggregation="sum",
                )

                # 2. Set top_n to the total number of features
                # This ensures the "Rest" category is never created and ALL variables are shown
                n_all_features = len(feature_names_diurnal)
                logging.info(f"    Plotting all {n_all_features} features (no 'Rest' category)")

                # Plot 8a: Stacked bar chart
                # Note: observed_values should be on original scale (not transformed)
                # base_value is the expected/mean prediction from SHAP explainer
                # observed_mean is the mean of observed values for centering
                plot_diurnal_drivers(
                    shap_values=shap_values_diurnal,
                    feature_names=feature_names_diurnal,
                    timestamps=timestamps_sampled,
                    observed_values=y_sampled,  # Original scale observed values
                    base_value=float(base_value),  # SHAP base value (model average prediction)
                    observed_mean=float(np.mean(y_sampled)),  # Mean of observed for centering
                    top_n=n_all_features,  # Show ALL features
                    output_dir=plot_dir,
                )
                logging.info("    Generated Diurnal Drivers (Stacked Bar with Observed)")

                # Plot 8b: Heatmap
                plot_diurnal_drivers_heatmap(
                    shap_values=shap_values_diurnal,
                    feature_names=feature_names_diurnal,
                    timestamps=timestamps_sampled,
                    top_n=n_all_features,  # Show ALL features
                    output_dir=plot_dir,
                )
                logging.info("    Generated Diurnal Drivers (Heatmap)")

                # Plot 8c: Line plots with CI
                # Note: We limit this to top 12 to prevent generating 20+ tiny subplots,
                # but it will include PFT as a single line if it's important.
                # If you truly want ALL lines, change 12 to n_all_features.
                plot_diurnal_feature_lines(
                    shap_values=shap_values_diurnal,
                    feature_names=feature_names_diurnal,
                    timestamps=timestamps_sampled,
                    top_n=12,
                    output_dir=plot_dir,
                )
                logging.info("    Generated Diurnal Drivers (Line Plots)")

            except Exception as e:
                logging.warning(f"    Could not generate diurnal plots: {e}")
                import traceback

                traceback.print_exc()
        else:
            logging.info(f"\nStep 8: Skipping diurnal plots (TIME_SCALE={TIME_SCALE}, requires 'hourly')")

        # =================================================================
        # STEP 9: Interaction Dependencies (Figure 8)
        # =================================================================
        logging.info("\nStep 9: Generating Interaction Dependence Plots (Figure 8)...")

        try:
            # Define meaningful interaction pairs
            # Use feature names that exist in the aggregated feature list
            potential_pairs = [
                ("sw_in_t-0", "vpd_t-0"),
                ("vpd_t-0", "ta_t-0"),
                ("ta_t-0", "volumetric_soil_water_layer_1_t-0"),
                ("sw_in_t-0", "LAI"),  # Static feature
                ("vpd_t-0", "canopy_height"),  # Static feature
            ]

            # For non-windowed data
            if not IS_WINDOWING:
                potential_pairs = [
                    ("vpd", "volumetric_soil_water_layer_1"),  # Supply vs Demand
                    ("sw_in", "vpd"),  # Energy vs Demand
                    ("canopy_height", "LAI"),  # Structural interaction
                    ("vpd", "PFT"),  # Strategy difference (Categorical interaction)
                ]

            # Filter to only include pairs where both features exist
            valid_pairs = []
            for f1, f2 in potential_pairs:
                if f1 in shap_feature_names_final and f2 in shap_feature_names_final:
                    valid_pairs.append((f1, f2))
                else:
                    missing = []
                    if f1 not in shap_feature_names_final:
                        missing.append(f1)
                    if f2 not in shap_feature_names_final:
                        missing.append(f2)
                    logging.debug(f"    Skipping pair ({f1}, {f2}) - missing: {missing}")

            if valid_pairs:
                logging.info(f"    Using interaction pairs: {valid_pairs}")
                plot_interaction_dependencies(
                    shap_values=shap_values,
                    X_original=X_for_plots,
                    feature_names=shap_feature_names_final,
                    interaction_pairs=valid_pairs[:3],  # Use top 3 valid pairs
                    output_dir=plot_dir,
                )
                logging.info("    Generated Figure 8 (Interaction Dependencies)")
            else:
                logging.warning("    No valid interaction pairs found - skipping Figure 8")

        except Exception as e:
            logging.warning(f"    Could not generate Fig 8 (Interaction): {e}")
            import traceback

            traceback.print_exc()

        # =================================================================
        # STEP 10: Time-Step Comparison (for windowed data)
        # =================================================================
        if IS_WINDOWING and shap_values_windowed is not None:
            logging.info("\nStep 10: Generating Time-Step Importance Comparison...")

            try:
                # Calculate mean absolute SHAP by time step for each base feature
                n_base_features = len(final_feature_names)

                # Get dynamic features only
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

                # Create comparison plot
                fig, ax = plt.subplots(figsize=(14, 8))

                x = np.arange(len(dynamic_features))
                width = 0.8 / INPUT_WIDTH

                colors = plt.cm.viridis(np.linspace(0.2, 0.8, INPUT_WIDTH))

                for t_idx, (time_label, feat_importance) in enumerate(time_step_importance.items()):
                    values = [feat_importance.get(f, 0) for f in dynamic_features]
                    offset = (t_idx - INPUT_WIDTH / 2 + 0.5) * width
                    ax.bar(x + offset, values, width, label=time_label, color=colors[t_idx])

                ax.set_xlabel("Feature", fontsize=12)
                ax.set_ylabel(f"Mean |SHAP Value| ({SHAP_UNITS})", fontsize=12)
                ax.set_title("Feature Importance by Time Step\n(Dynamic Features Only)", fontsize=14, fontweight="bold")
                ax.set_xticks(x)
                ax.set_xticklabels(dynamic_features, rotation=45, ha="right")
                ax.legend(title="Time Step", loc="upper right")
                # ax.grid(True, alpha=0.3, axis='y')

                plt.tight_layout()
                plt.savefig(plot_dir / "shap_time_step_comparison.png", dpi=300, bbox_inches="tight")
                plt.close()
                logging.info("    Saved: shap_time_step_comparison.png")

            except Exception as e:
                logging.warning(f"    Could not generate time-step comparison: {e}")
                import traceback

                traceback.print_exc()

        # =================================================================
        # STEP 11: Static vs Dynamic Feature Importance
        # =================================================================
        logging.info("\nStep 11: Generating Static vs Dynamic Feature Comparison...")

        try:
            # Calculate mean absolute SHAP for each feature
            mean_abs_importance = np.abs(shap_values).mean(axis=0)
            feature_importance_df = pd.DataFrame(
                {"feature": shap_feature_names_final, "importance": mean_abs_importance}
            )

            # Categorize as static or dynamic
            def categorize_feature(name):
                # Check if it's a static feature (no time suffix)
                if name in STATIC_FEATURES:
                    return "Static"
                # Check for time suffix patterns
                if "_t-" in name or name.endswith("_t-0"):
                    return "Dynamic"
                return "Static"  # Default to static if no time suffix

            feature_importance_df["category"] = feature_importance_df["feature"].apply(categorize_feature)

            # Add units column
            feature_importance_df["unit"] = feature_importance_df["feature"].apply(get_feature_unit)

            # Sort by importance
            feature_importance_df = feature_importance_df.sort_values("importance", ascending=True)

            # Create plot
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            # Left: Bar plot colored by category
            colors_plot = feature_importance_df["category"].map({"Static": "coral", "Dynamic": "steelblue"})
            axes[0].barh(range(len(feature_importance_df)), feature_importance_df["importance"], color=colors_plot)
            axes[0].set_yticks(range(len(feature_importance_df)))

            # Create labels with units
            labels_with_units = []
            for _, row in feature_importance_df.iterrows():
                if row["unit"]:
                    labels_with_units.append(f"{row['feature']} ({row['unit']})")
                else:
                    labels_with_units.append(row["feature"])

            axes[0].set_yticklabels(labels_with_units, fontsize=8)
            axes[0].set_xlabel(f"Mean |SHAP Value| ({SHAP_UNITS})", fontsize=12)
            axes[0].set_title("Feature Importance by Category", fontsize=14, fontweight="bold")

            # Add legend
            from matplotlib.patches import Patch

            legend_elements = [Patch(facecolor="coral", label="Static"), Patch(facecolor="steelblue", label="Dynamic")]
            axes[0].legend(handles=legend_elements, loc="lower right")
            # axes[0].grid(True, alpha=0.3, axis='x')

            # Right: Pie chart of total importance by category
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
            logging.info("    Saved: shap_static_vs_dynamic.png")

            # Save feature importance to CSV (with units)
            feature_importance_df.to_csv(plot_dir / "shap_feature_importance.csv", index=False)
            logging.info("    Saved: shap_feature_importance.csv")

        except Exception as e:
            logging.warning(f"    Could not generate static vs dynamic comparison: {e}")
            import traceback

            traceback.print_exc()
        # =================================================================
        # STEP 12: PFT-Stratified SHAP Analysis
        # =================================================================
        logging.info("\nStep 12: Generating PFT-Stratified SHAP Analysis...")

        try:
            # First, get PFT labels for each sample
            pft_labels_sampled = get_sample_pft_labels(
                X_original=X_original_sampled,
                feature_names=shap_feature_names if not IS_WINDOWING else shap_feature_names_windowed,
                pft_columns=PFT_COLUMNS,
            )

            logging.info("  PFT distribution in SHAP samples:")
            for pft in np.unique(pft_labels_sampled):
                count = (pft_labels_sampled == pft).sum()
                logging.info(f"    {pft}: {count} samples ({100 * count / len(pft_labels_sampled):.1f}%)")

            # Create a version of SHAP values and feature names with PFT aggregated
            # (for plots that don't need individual PFT columns)
            shap_values_pft_agg, feature_names_pft_agg = aggregate_pft_shap_values(
                shap_values=shap_values,
                feature_names=shap_feature_names_final,
                pft_columns=PFT_COLUMNS,
                aggregation="sum",
            )

            # Also get corresponding X values (for beeswarm plots)
            # For X values, we just need to remove PFT columns
            non_pft_mask = [name not in PFT_COLUMNS for name in shap_feature_names_final]
            X_for_plots_no_pft = X_for_plots[:, non_pft_mask]

            # Add a dummy PFT column for X (just use 0s, won't affect beeswarm coloring much)
            X_for_pft_plots = np.hstack([X_for_plots_no_pft, np.zeros((len(X_for_plots), 1))])

            # 12a: Feature importance heatmap by PFT
            logging.info("  Generating feature importance heatmap by PFT...")
            plot_feature_importance_by_pft(
                shap_values=shap_values_pft_agg,
                feature_names=feature_names_pft_agg,
                pft_labels=pft_labels_sampled,
                top_n=12,
                output_dir=plot_dir,
            )

            # 12b: Top features per PFT
            logging.info("  Generating top features per PFT plot...")
            plot_top_features_per_pft(
                shap_values=shap_values_pft_agg,
                feature_names=feature_names_pft_agg,
                pft_labels=pft_labels_sampled,
                top_n=6,
                output_dir=plot_dir,
            )

            # 12c: Boxplots by PFT
            logging.info("  Generating SHAP boxplots by PFT...")
            plot_shap_by_pft_boxplot(
                shap_values=shap_values_pft_agg,
                feature_names=feature_names_pft_agg,
                pft_labels=pft_labels_sampled,
                top_n=8,
                output_dir=plot_dir,
            )

            # 12d: Violin plots by PFT
            logging.info("  Generating SHAP violin plots by PFT...")
            plot_shap_by_pft_violin(
                shap_values=shap_values_pft_agg,
                feature_names=feature_names_pft_agg,
                pft_labels=pft_labels_sampled,
                top_n=8,
                output_dir=plot_dir,
            )

            # 12e: PFT contribution comparison
            logging.info("  Generating PFT contribution comparison...")
            plot_pft_contribution_comparison(
                shap_values=shap_values_pft_agg,
                feature_names=feature_names_pft_agg,
                pft_labels=pft_labels_sampled,
                output_dir=plot_dir,
            )

            # 12f: Radar chart
            logging.info("  Generating PFT radar chart...")
            plot_pft_radar_chart(
                shap_values=shap_values_pft_agg,
                feature_names=feature_names_pft_agg,
                pft_labels=pft_labels_sampled,
                top_n=8,
                output_dir=plot_dir,
            )

            # 12g: Individual SHAP summary plots per PFT
            logging.info("  Generating individual SHAP summary plots per PFT...")
            plot_pft_shap_summary(
                shap_values=shap_values_pft_agg,
                X_original=X_for_pft_plots,
                feature_names=feature_names_pft_agg,
                pft_labels=pft_labels_sampled,
                top_n=15,
                output_dir=plot_dir,
            )

            # 12h: Generate comprehensive CSV report
            logging.info("  Generating PFT SHAP statistics report...")
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
        # STEP 13: Save SHAP values and metadata (was Step 12)
        # =================================================================
        logging.info("\nStep 13: Saving SHAP values...")

        # Save aggregated SHAP values
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

        # Save feature units mapping
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
        logging.info(f"SHAP value units: {SHAP_UNITS}")
        logging.info(f"PFT types analyzed: {PFT_COLUMNS}")

    except Exception as e:
        logging.error(f"SHAP Analysis failed: {e}")
        import traceback

        traceback.print_exc()

    # === AOA Reference Generation (M7) ===
    try:
        from src.aoa.prepare import build_aoa_reference

        # 1. Derive fold labels from CV splitter
        fold_labels = np.full(len(X_all_records), -1, dtype=int)
        for fold_idx, (_, test_idx) in enumerate(outer_cv.split(X_all_records, y_all_stratified, groups_all_records)):
            fold_labels[test_idx] = fold_idx
        assert (fold_labels >= 0).all(), "Some samples not assigned to any fold"

        # 2. Load SHAP importance (already saved above)
        shap_csv_path = plot_dir / "shap_feature_importance.csv"
        shap_df = pd.read_csv(shap_csv_path)
        shap_importances = shap_df.set_index("feature")["importance"].reindex(final_feature_names).values
        assert not np.any(np.isnan(shap_importances)), (
            f"SHAP features do not match final_feature_names: {final_feature_names}"
        )

        # 3. Build AOA reference
        aoa_ref_path = build_aoa_reference(
            X_train=X_all_records,
            fold_labels=fold_labels,
            shap_importances=shap_importances,
            feature_names=final_feature_names,
            output_dir=model_dir,
            run_id=run_id,
            d_bar_method="full",
        )
        logging.info(f"AOA reference saved to {aoa_ref_path}")

        # 4. Save backups for future backfill
        np.savez(
            model_dir / f"FINAL_cv_folds_{run_id}.npz",
            fold_labels=fold_labels,
            spatial_groups=groups_all_records,
            site_ids=site_ids_all_records,
            pfts=pfts_all_records,
        )
        pd.DataFrame(X_all_records, columns=final_feature_names).to_parquet(
            model_dir / f"FINAL_X_train_{run_id}.parquet",
            index=False,
        )
        logging.info("AOA backups (cv_folds, X_train) saved")
    except Exception as e:
        logging.error(f"AOA reference generation failed: {e}")
        logging.info("Model training completed successfully - AOA step failed separately")

    logging.info("\n" + "=" * 60)
    logging.info("=== ALL PROCESSING COMPLETE ===")
    logging.info("=" * 60)

    return all_test_r2_scores, all_test_rmse_scores


if __name__ == "__main__":
    main()
