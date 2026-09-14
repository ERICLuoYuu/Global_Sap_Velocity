"""
Training utility functions for ML spatial stratified pipeline.

Extracted from test_hyperparameter_tuning_ML_spatial_stratified.py for reuse
across training, prediction, and standalone SHAP analysis scripts.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path


def _str_to_bool(v: str) -> bool:
    """Parse boolean CLI arguments correctly (type=bool treats any string as True)."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


import numpy as np
import pandas as pd

try:
    from src.utils.random_control import deterministic
except ImportError:
    # Fallback when TensorFlow is not available (e.g., test environments)
    def deterministic(func):
        return func


try:
    import tensorflow as tf

    from src.hyperparameter_optimization.timeseries_processor1 import WindowGenerator
except ImportError:
    tf = None
    WindowGenerator = None


def setup_logging(logger_name, log_dir=None):
    """Set up basic logging configuration"""
    _logger = logging.getLogger(logger_name)  # noqa: F841 — configures named logger

    if log_dir is None:
        log_dir = Path("./outputs/logs")

    log_dir = Path(log_dir)
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{logger_name}_optimizer.log"

    handlers = [logging.FileHandler(log_file, mode="a", encoding="utf-8"), logging.StreamHandler()]

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=handlers
    )

    warnings.filterwarnings("ignore")

    return logging.getLogger()


@deterministic
def add_time_features(df, datetime_column="solar_TIMESTAMP"):
    """Create cyclical time features from a datetime column or index."""
    df = df.copy()

    if datetime_column is not None:
        if datetime_column not in df.columns:
            raise ValueError(f"Column '{datetime_column}' not found in DataFrame")

        if df[datetime_column].isna().any():
            n_nulls = df[datetime_column].isna().sum()
            logging.warning(f"Dropping {n_nulls} rows with NaN in {datetime_column}")
            df = df.dropna(subset=[datetime_column])

        date_time = pd.to_datetime(df[datetime_column])
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                date_time = pd.to_datetime(df.index)
            except (ValueError, TypeError) as e:
                raise ValueError("DataFrame index cannot be converted to datetime") from e
        else:
            date_time = df.index

    # Use vectorized conversion to avoid pandas version issues with tz-aware datetimes
    # Convert to nanoseconds since epoch, then to seconds
    timestamp_s = date_time.astype("int64") / 1e9
    day = 24 * 60 * 60
    year = 365.2425 * day
    week = 7 * day
    month = 30.44 * day

    df["Day sin"] = np.sin(timestamp_s * (2 * np.pi / day))
    df["Day cos"] = np.cos(timestamp_s * (2 * np.pi / day))
    df["Year sin"] = np.sin(timestamp_s * (2 * np.pi / year))
    df["Year cos"] = np.cos(timestamp_s * (2 * np.pi / year))
    df["Week sin"] = np.sin(timestamp_s * (2 * np.pi / week))
    df["Week cos"] = np.cos(timestamp_s * (2 * np.pi / week))
    df["Month sin"] = np.sin(timestamp_s * (2 * np.pi / month))
    df["Month cos"] = np.cos(timestamp_s * (2 * np.pi / month))

    return df


@deterministic
def create_windows_from_segments(
    segments: list[pd.DataFrame],
    input_width: int,
    label_width: int,
    shift: int,
    label_columns: list[str] = None,
    exclude_targets_from_features: bool = True,
    exclude_labels_from_inputs: bool = True,
) -> list[tuple[tf.Tensor, tf.Tensor]]:
    """Create windows from multiple segments without splitting them."""
    all_windows = []

    for segment in segments:
        min_segment_length = input_width + shift
        if len(segment) < min_segment_length:
            continue

        window_gen = WindowGenerator(
            input_width=input_width,
            label_width=label_width,
            shift=shift,
            data_df=segment,
            label_columns=label_columns,
            batch_size=1,
            shuffle=False,
            exclude_targets_from_features=exclude_targets_from_features,
            exclude_labels_from_inputs=exclude_labels_from_inputs,
        )

        segment_ds = window_gen.dataset
        for inputs, labels in segment_ds:
            all_windows.append((inputs, labels))

    return all_windows


@deterministic
def get_predictions(model, test_windows, X_test, y_test, is_windowing=True):
    """Get predictions from the model using windows data."""
    if is_windowing:
        X_test, y_test = convert_windows_to_numpy(test_windows)

    y_pred = model.predict(X_test)

    return y_pred.flatten(), y_test.flatten()


@deterministic
def convert_windows_to_numpy(windows: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Convert a list of window tuples to numpy arrays."""
    if not windows:
        return np.array([]), np.array([])

    features = [w[0] for w in windows]
    labels = [w[1] for w in windows]

    X = np.array(features)
    y = np.array(labels)

    X_reshaped = X.reshape(X.shape[0], -1)
    y_flattened = y.flatten()

    return X_reshaped, y_flattened


@deterministic
def create_spatial_groups(
    lat: np.ndarray | list,
    lon: np.ndarray | list,
    data_counts: np.ndarray | list = None,
    n_groups: int = None,
    balanced: bool = True,
    method: str = "grid",
    lat_grid_size: float = 0.05,
    lon_grid_size: float = 0.05,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Group geographical data points using various spatial grouping methods."""
    lat = np.asarray(lat)
    lon = np.asarray(lon)

    if method == "grid":
        lat_bins = np.arange(np.floor(lat.min()), np.ceil(lat.max()) + lat_grid_size, lat_grid_size)
        lon_bins = np.arange(np.floor(lon.min()), np.ceil(lon.max()) + lon_grid_size, lon_grid_size)

        lat_indices = np.digitize(lat, lat_bins) - 1
        lon_indices = np.digitize(lon, lon_bins) - 1

        n_lon_bins = len(lon_bins)
        grid_cell_ids = lat_indices * n_lon_bins + lon_indices

        if data_counts is None or n_groups is None or not balanced:
            unique_groups = np.unique(grid_cell_ids)
            group_map = {g: n for n, g in enumerate(unique_groups)}
            groups = np.array([group_map[g] for g in grid_cell_ids])
        else:
            unique_cells, cell_indices = np.unique(grid_cell_ids, return_inverse=True)
            cell_data_counts = np.bincount(cell_indices, weights=data_counts)

            # stable sort: quicksort reorders TIED counts if the site list changes
            sorted_cell_indices = np.argsort(cell_data_counts, kind="stable")[::-1]

            fold_data_counts = np.zeros(n_groups)
            cell_to_fold = np.zeros(len(unique_cells), dtype=int)
            for cell_idx in sorted_cell_indices:
                target_fold = np.argmin(fold_data_counts)
                cell_to_fold[cell_idx] = target_fold
                fold_data_counts[target_fold] += cell_data_counts[cell_idx]

            groups = cell_to_fold[cell_indices]

    elif method == "default":
        if not len(lat) == len(lon):
            raise ValueError("Latitude and longitude arrays must have the same length.")
        if data_counts is None or n_groups is None or not balanced:
            groups = np.arange(len(lat))
        else:
            # stable sort: quicksort reorders TIED counts if the site list changes
            sorted_indices = np.argsort(data_counts, kind="stable")[::-1]
            groups = np.zeros(len(lat), dtype=int) - 1
            fold_data_counts = np.zeros(n_groups)

            for idx in sorted_indices:
                target_group = np.argmin(fold_data_counts)
                groups[idx] = target_group
                fold_data_counts[target_group] += data_counts[idx]
    else:
        raise ValueError(f"Unknown method: {method}. Supported methods are 'grid' or 'default'.")

    # Calculate group statistics
    stats = []
    for group in np.unique(groups):
        if group == -1:
            continue

        mask = groups == group
        stats.append(
            {
                "group": group,
                "size": np.sum(mask),
                "mean_lat": np.mean(lat[mask]),
                "mean_lon": np.mean(lon[mask]),
                "std_lat": np.std(lat[mask]),
                "std_lon": np.std(lon[mask]),
                "min_lat": np.min(lat[mask]),
                "max_lat": np.max(lat[mask]),
                "min_lon": np.min(lon[mask]),
                "max_lon": np.max(lon[mask]),
            }
        )

    stats_df = pd.DataFrame(stats)

    if len(stats_df) > 0:
        logging.info("Spatial Group Statistics:")
        logging.info(f"Total groups: {len(stats_df)}")
        logging.info(f"Group details:\n{stats_df[['group', 'size', 'mean_lat', 'mean_lon']].to_string(index=False)}")

    return groups, stats_df


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
    parser.add_argument("--EXCLUDE_LABELS", type=_str_to_bool, default=True, help="Exclude labels from input features")
    parser.add_argument(
        "--EXCLUDE_TARGETS", type=_str_to_bool, default=True, help="Exclude targets from input features"
    )
    parser.add_argument(
        "--IS_WINDOWING", type=_str_to_bool, default=False, help="Enable time windowing for data processing"
    )
    parser.add_argument(
        "--spatial_split_method", type=str, default="default", help="Method for spatial splitting of data"
    )
    parser.add_argument(
        "--hyperparameters",
        type=str,
        default="src/hyperparameter_optimization/JSON/XGB_hyperparameters_fixed.json",
        help="Path to the JSON file of hyperparameters",
    )
    parser.add_argument("--IS_SHUFFLE", type=_str_to_bool, default=True, help="Whether to enable shuffling of data")
    parser.add_argument("--N_ITERATIONS", type=int, default=None, help="Number of iterations for random search")
    parser.add_argument(
        "--IS_CV", type=_str_to_bool, default=True, help="Whether to enable cross-validation for inner loop"
    )
    parser.add_argument("--IS_STRATIFIED", type=_str_to_bool, default=True, help="Whether to use stratified sampling")
    parser.add_argument("--BALANCED", type=_str_to_bool, default=False, help="Whether to balance the spatial groups")
    parser.add_argument("--SPLIT_TYPE", type=str, default="spatial_stratified", help="Type of data splitting strategy")
    parser.add_argument("--IS_ONLY_DAY", type=_str_to_bool, default=False, help="Whether to use only day data")
    parser.add_argument(
        "--selected_features",
        type=str,
        required=True,
        help="Path to JSON file with feature names. Training uses exactly these features.",
    )
    parser.add_argument("--TIME_SCALE", type=str, default="daily", help="Time scale of the data: hourly or daily")
    parser.add_argument("--SHAP_SAMPLE_SIZE", type=int, default=50000, help="Sample size for SHAP analysis")
    parser.add_argument(
        "--IS_TRANSFORM", type=_str_to_bool, default=True, help="Whether to apply target transformation"
    )
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
        "--data_dir",
        type=str,
        default=None,
        help="Override data directory path (e.g. for growing-season-filtered data)",
    )
    parser.add_argument(
        "--vpd_filter",
        type=float,
        default=0.0,
        help="Min VPD (kPa). Rows below are dropped. Default 0 (no filter) — V target "
        "trains on full data. For canopy_conductance target pass 0.3 explicitly "
        "(G_c undefined at low VPD; 0.6 per Ewers & Oren 2000 for sensitivity).",
    )
    from src.feature_contract.definitions import DEFAULT_CONTRACT, DEFAULT_SWC_SCALE, FEATURE_CONTRACTS, SWC_SCALES

    parser.add_argument(
        "--FEATURE_CONTRACT",
        type=str,
        default=DEFAULT_CONTRACT,
        choices=list(FEATURE_CONTRACTS),
        help="site: legacy per-site soil-moisture statistics and gauge precip memory. "
        "servable: climatology-normalised soil moisture and ERA5-Land precip memory, "
        "reproducible at prediction time; needs a tree built by src.feature_contract.rebuild_daily.",
    )
    parser.add_argument(
        "--SWC_SCALE",
        type=str,
        default=DEFAULT_SWC_SCALE,
        choices=list(SWC_SCALES),
        help="Scale of every soil-moisture feature, servable contract only. ratio: level columns "
        "raw / sigma_clim and swc_layer*_norm the climatology z-score. zscore: all from "
        "(raw - mu_clim) / sigma_clim. raw: all from raw m3/m3.",
    )
    return parser.parse_args()


def build_pft_onehot(
    df: "pd.DataFrame",
    pft_col: str,
    site_id: str,
    pft_columns: list,
) -> "pd.DataFrame":
    """One-hot encode the PFT column, dropping rows that never had a PFT.

    Two different situations produce an all-zero one-hot row and only one of
    them is a defect:

    * **The value is present but unrecognised** -- e.g. the long IGBP name
      ``'Mixed Forest'`` where the short code ``'MF'`` is expected.
      ``pd.Categorical`` coerces it to NaN in silence, XGBoost then never splits
      on PFT, and SHAP reports exactly 0 for every PFT column. This stays fatal.

    * **The value was already missing in the source.** Since the 2026-08-24
      re-merge, observation-free calendar rows carry a blank ``pft``; they hold
      neither sap flow nor environmental data, so they carry no target either.
      Measured across the corpus: 27 sites have such rows and ZERO of those rows
      hold a target value.

    Conflating the two cost 27 whole sites -- FRA_HES and the four largest CZE
    stands among them -- because a single blank row rejected the entire site.

    Returns the frame with the one-hot columns joined on, minus rows that had no
    PFT to begin with.
    """
    missing = df[pft_col].isna()
    if missing.any():
        logging.info(
            "  %s: dropping %d of %d rows with no %s (they carry no target)",
            site_id,
            int(missing.sum()),
            len(df),
            pft_col,
        )
        df = df[~missing].copy()
    if df.empty:
        raise ValueError(f"Site {site_id}: no row carries a '{pft_col}' value")

    dummies = pd.get_dummies(pd.Categorical(df[pft_col], categories=pft_columns)).astype(int)
    dummies.index = df.index

    all_zero = dummies.sum(axis=1) == 0
    if all_zero.any():
        unrecognised = sorted({str(v) for v in df.loc[all_zero, pft_col].unique()})
        raise ValueError(
            f"Site {site_id}: PFT one-hot produced all-zero rows from column "
            f"'{pft_col}'. Unrecognised value(s): {unrecognised}. Expected values "
            f"in PFT_COLUMNS = {pft_columns}. Check that merge pipeline emits "
            f"short IGBP codes."
        )

    for col in dummies.columns:
        df[col] = dummies[col]
    return df


def _first_set(args, *names):
    """First of *names* that ``args`` actually defines, else ``None``.

    ``getattr(args, n, None) or ...`` would be wrong: ``n_groups=0`` is falsy
    but set, and would fall through to the next name.
    """
    for name in names:
        value = getattr(args, name, None)
        if value is not None:
            return value
    return None


def _as_str(value):
    """Stringify Paths so the result survives a plain ``json.dump``.

    Five of the six callers pass ``default=str``; ``train_from_cache.py`` does
    not, and a bare ``Path`` there raises ``TypeError`` at write time.
    """
    return None if value is None else str(value)


def build_run_provenance(args, selected_features=None, data_dir=None):
    """Describe how a run was configured, for ``FINAL_config_<run_id>.json``.

    Every trainer recorded ``feature_names`` -- the FINAL, post-engineering,
    post-one-hot list -- but never the *inputs* that produced it: which
    selected-features JSON was passed, and how sites were grouped into folds.
    Two runs differing only in ``--spatial_split_method`` (one group per SITE
    vs a 0.05 deg grid, worth +0.145 pooled R2 through co-location leakage)
    wrote configs identical apart from their scores, so a finished run could
    not be told apart from another by its own config.

    Fields are read with ``getattr(..., None)`` because the six callers do not
    share a parser: five build their own ``ArgumentParser`` and only the main
    ML trainer imports :func:`parse_args`. The DL and autoML parsers define no
    ``--grid_size``; ``train_from_cache`` takes its groups from the FFS cache
    and defines none of the spatial flags. A flag that is absent must record
    ``None``, never raise -- this runs *after* the model is fitted, so an
    ``AttributeError`` here would discard the entire run.

    Args:
        args: The parsed ``argparse.Namespace``.
        selected_features: The resolved feature list the run actually used
            (the trainers' ``used_cols``), including the target column if the
            trainer prepended it. ``None`` if the caller has no such list.
        data_dir: The directory the run actually read, after the
            ``--data_dir``-or-``PathConfig`` fallback has been resolved.

    Returns:
        A dict with ``selected_features``, ``selected_features_path`` and a
        ``split_config`` block, ready to splat into a ``final_config`` literal.
    """
    return {
        "selected_features": None if selected_features is None else list(selected_features),
        "selected_features_path": _as_str(getattr(args, "selected_features", None)),
        "feature_contract": _first_set(args, "FEATURE_CONTRACT"),
        "swc_scale": _first_set(args, "SWC_SCALE"),
        "split_config": {
            "spatial_split_method": _first_set(args, "spatial_split_method"),
            "n_groups": _first_set(args, "n_groups", "n_splits"),
            "grid_size": _first_set(args, "grid_size"),
            "balanced": _first_set(args, "BALANCED"),
            "is_stratified": _first_set(args, "IS_STRATIFIED"),
            "is_shuffle": _first_set(args, "IS_SHUFFLE"),
            "is_only_day": _first_set(args, "IS_ONLY_DAY"),
            "split_type": _first_set(args, "SPLIT_TYPE"),
            "time_scale": _first_set(args, "TIME_SCALE"),
            "data_dir": _as_str(data_dir),
        },
    }
