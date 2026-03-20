#!/usr/bin/env python
"""
Improved Sap Velocity Prediction Script

This script works with preprocessed ERA5-Land data to predict sap velocity,
using an improved index mapping approach for more reliable time series predictions.

IMPROVEMENTS OVER ORIGINAL:
1. Conditionally creates time windows based on model config (IS_WINDOWING)
2. Dynamically loads features from model config (feature_names)
3. Applies data transformations based on config (preprocessing settings)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


# Set up logging
def _get_log_path() -> Path:
    """Return log file path, preferring SAP_LOG_DIR env var over package directory."""
    log_dir = os.environ.get("SAP_LOG_DIR", "")
    if log_dir and Path(log_dir).is_dir():
        return Path(log_dir) / "sap_velocity_prediction.log"
    # Fallback: current working directory, then package directory
    return Path("sap_velocity_prediction.log")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(_get_log_path()), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sap_prediction")


def _get_tf():
    """Lazy-import TensorFlow to avoid startup cost when using non-DL models."""
    import tensorflow as tf

    return tf


# Default parameters if config is not available
DEFAULT_PARAMS = {"MODEL_TYPES": ["xgb", "rf", "cnn_lstm"], "INPUT_WIDTH": 8, "LABEL_WIDTH": 1, "SHIFT": 1}

# Maximum expected sap velocity (cm3 cm-2 h-1). Literature values for
# high-conducting tropical species can reach 200-300+.
MAX_EXPECTED_SAP_VELOCITY = 500

try:
    from config import (
        BASE_DIR,
        MODEL_DIR,
        OUTPUT_DIR,
    )

    MODELS_DIR = MODEL_DIR  # Alias for backward compat within this script
except ImportError:
    logger.warning("Config module not found. Using default parameters.")
    BASE_DIR = Path(".")
    MODELS_DIR = BASE_DIR / "models"
    OUTPUT_DIR = BASE_DIR / "outputs"


# =============================================================================
# Security: path validation and safe deserialization helpers
# =============================================================================

# Regex for safe path components (model_type, run_id): alphanumeric, dash, underscore, dot
_SAFE_PATH_COMPONENT = re.compile(r"^[a-zA-Z0-9_\-\.]+$")


def validate_path_component(value: str, name: str) -> str:
    """Validate that a path component contains no traversal characters.

    Parameters
    ----------
    value : str
        The path component to validate (e.g., model_type or run_id).
    name : str
        Human-readable name for error messages.

    Returns
    -------
    str
        The validated value.

    Raises
    ------
    ValueError
        If the value contains unsafe characters.
    """
    if not _SAFE_PATH_COMPONENT.match(value):
        raise ValueError(
            f"Invalid characters in {name}: {value!r}. Only alphanumeric, dash, underscore, and dot are allowed."
        )
    return value


def resolve_and_contain(user_path: str, allowed_root: Path, label: str) -> Path:
    """Resolve a user-supplied path and verify it stays inside allowed_root.

    Parameters
    ----------
    user_path : str
        Path string from CLI argument.
    allowed_root : Path
        The directory that must contain the resolved path.
    label : str
        Human-readable label for error messages.

    Returns
    -------
    Path
        The resolved, validated path.

    Raises
    ------
    ValueError
        If the resolved path escapes the allowed root.
    """
    resolved = Path(user_path).resolve()
    root_resolved = allowed_root.resolve()
    if not str(resolved).startswith(str(root_resolved)):
        raise ValueError(f"{label} path escapes allowed root: {resolved} is not inside {root_resolved}")
    return resolved


def safe_joblib_load(path: Path, allowed_root: Path) -> Any:
    """Load a joblib/pickle file after verifying path containment.

    Parameters
    ----------
    path : Path
        File to load.
    allowed_root : Path
        The directory that must contain the file.

    Returns
    -------
    Any
        The deserialized object.

    Raises
    ------
    ValueError
        If the path escapes the allowed root.
    """
    resolved = path.resolve()
    root_resolved = allowed_root.resolve()
    if not str(resolved).startswith(str(root_resolved)):
        raise ValueError(f"Refusing to load file outside allowed root: {resolved} is not inside {root_resolved}")
    return joblib.load(resolved)


# Maximum input CSV file size (bytes) to prevent OOM on shared HPC nodes
MAX_CSV_BYTES = 5 * 1024**3  # 5 GB


# =============================================================================
# NEW: Model Configuration Classes
# =============================================================================


@dataclass
class ModelConfig:
    """
    Data class to hold model configuration loaded from JSON.

    This enables config-driven behavior for:
    - Windowing (IS_WINDOWING)
    - Feature selection (feature_names)
    - Data transformations (preprocessing)
    """

    model_type: str
    run_id: str = "default"
    best_params: dict[str, Any] = field(default_factory=dict)

    # Preprocessing settings
    target_transform: str | None = None  # e.g., "log1p", "sqrt", "log"
    feature_scaling: str | None = None  # e.g., "StandardScaler", "MinMaxScaler"

    # Data info / windowing settings
    is_windowing: bool = False
    input_width: int | None = None
    label_width: int | None = None
    shift: int | None = None
    n_samples: int | None = None
    n_features: int | None = None

    # Feature names from training
    feature_names: list[str] = field(default_factory=list)

    # Other settings
    random_seed: int = 42
    split_type: str = "random"
    cv_results: dict[str, Any] = field(default_factory=dict)

    # Allowed values for validated fields
    _ALLOWED_TRANSFORMS = frozenset({"log1p", "log", "sqrt", None})
    _ALLOWED_SCALERS = frozenset(
        {
            "StandardScaler",
            "MinMaxScaler",
            "RobustScaler",
            "standard",
            "minmax",
            "robust",
            None,
        }
    )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> ModelConfig:
        """Create ModelConfig from a dictionary (loaded from JSON)."""
        # Extract preprocessing settings
        preprocessing = config_dict.get("preprocessing", {})
        target_transform = preprocessing.get("target_transform")
        feature_scaling = preprocessing.get("feature_scaling")

        # Validate against allowlists
        if target_transform not in cls._ALLOWED_TRANSFORMS:
            raise ValueError(f"Invalid target_transform: {target_transform!r}")
        if feature_scaling not in cls._ALLOWED_SCALERS:
            raise ValueError(f"Invalid feature_scaling: {feature_scaling!r}")

        # Extract data_info settings
        data_info = config_dict.get("data_info", {})
        is_windowing = data_info.get("IS_WINDOWING", False)
        input_width = data_info.get("input_width")
        label_width = data_info.get("label_width")
        shift = data_info.get("shift")
        n_samples = data_info.get("n_samples")
        n_features = data_info.get("n_features")

        return cls(
            model_type=config_dict.get("model_type", "unknown"),
            run_id=config_dict.get("run_id", "default"),
            best_params=config_dict.get("best_params", {}),
            target_transform=target_transform,
            feature_scaling=feature_scaling,
            is_windowing=is_windowing,
            input_width=input_width,
            label_width=label_width,
            shift=shift,
            n_samples=n_samples,
            n_features=n_features,
            feature_names=config_dict.get("feature_names", []),
            random_seed=config_dict.get("random_seed", 42),
            split_type=config_dict.get("split_type", "random"),
            cv_results=config_dict.get("cv_results", {}),
        )

    @classmethod
    def from_json(cls, json_path: Path) -> ModelConfig:
        """Load ModelConfig from a JSON file."""
        with open(json_path) as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert ModelConfig to dictionary."""
        return {
            "model_type": self.model_type,
            "run_id": self.run_id,
            "best_params": self.best_params,
            "preprocessing": {"target_transform": self.target_transform, "feature_scaling": self.feature_scaling},
            "data_info": {
                "IS_WINDOWING": self.is_windowing,
                "input_width": self.input_width,
                "label_width": self.label_width,
                "shift": self.shift,
                "n_samples": self.n_samples,
                "n_features": self.n_features,
            },
            "feature_names": self.feature_names,
            "random_seed": self.random_seed,
            "split_type": self.split_type,
            "cv_results": self.cv_results,
        }


class DataTransformer:
    """
    Handles data transformations based on model config.

    Supports:
    - Target transforms: log1p, log, sqrt, none
    - Feature scaling: StandardScaler, MinMaxScaler, RobustScaler
    """

    def __init__(self, config: ModelConfig | None = None):
        self.config = config
        self.target_transform = config.target_transform if config else None
        self.feature_scaling = config.feature_scaling if config else None

    def get_scaler(self) -> Any | None:
        """Get the appropriate scaler based on config."""
        if self.feature_scaling is None:
            return None

        scaler_map = {
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler(),
            "RobustScaler": RobustScaler(),
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }

        return scaler_map.get(self.feature_scaling)

    def transform_target(self, y: np.ndarray, inverse: bool = False) -> np.ndarray:
        """
        Apply or inverse target transformation based on config.

        Parameters:
        -----------
        y : np.ndarray
            Target values
        inverse : bool
            If True, apply inverse transformation

        Returns:
        --------
        np.ndarray
            Transformed values
        """
        if self.target_transform is None:
            return y

        y = np.asarray(y)

        if self.target_transform == "log1p":
            if inverse:
                result = np.expm1(np.clip(y, -20, 20))
                return self._guard_non_finite(result, "expm1")
            else:
                return np.log1p(np.maximum(y, 0))
        elif self.target_transform == "log":
            if inverse:
                result = np.exp(np.clip(y, -20, 20))
                return self._guard_non_finite(result, "exp")
            else:
                return np.log(np.maximum(y, 1e-8))
        elif self.target_transform == "sqrt":
            if inverse:
                return np.square(y)
            else:
                return np.sqrt(np.maximum(y, 0))
        else:
            logger.warning("Unknown target transform: %s. Returning unchanged.", self.target_transform)
            return y

    @staticmethod
    def _guard_non_finite(arr: np.ndarray, op_name: str) -> np.ndarray:
        """Replace non-finite values (inf, -inf) with NaN and log a warning."""
        n_bad = int(np.sum(~np.isfinite(arr)))
        if n_bad > 0:
            logger.warning("%s produced %d non-finite values; replacing with NaN", op_name, n_bad)
            arr = np.where(np.isfinite(arr), arr, np.nan)
        return arr

    def should_transform_target(self) -> bool:
        """Check if target transformation should be applied."""
        return self.target_transform is not None

    def should_scale_features(self) -> bool:
        """Check if feature scaling should be applied."""
        return self.feature_scaling is not None


def get_model_dir(models_dir: Path, model_type: str, run_id: str) -> Path:
    """
    Get the model directory path based on new structure.

    New structure: ./models/{model_type}/{run_id}/

    Parameters:
    -----------
    models_dir : Path
        Base models directory
    model_type : str
        Type of model (e.g., 'xgb', 'rf', 'cnn_lstm')
    run_id : str
        Run identifier

    Returns:
    --------
    Path
        Path to the model directory
    """
    return models_dir / model_type / run_id


def get_model_path(models_dir: Path, model_type: str, run_id: str) -> Path:
    """
    Get the model file path based on new naming convention.

    Model file: FINAL_{model_type}_{run_id}.joblib

    Parameters:
    -----------
    models_dir : Path
        Base models directory
    model_type : str
        Type of model
    run_id : str
        Run identifier

    Returns:
    --------
    Path
        Path to the model file
    """
    model_dir = get_model_dir(models_dir, model_type, run_id)
    return model_dir / f"FINAL_{model_type}_{run_id}.joblib"


def get_config_path(models_dir: Path, model_type: str, run_id: str) -> Path:
    """
    Get the config file path based on new naming convention.

    Config file: FINAL_config_{run_id}.json

    Parameters:
    -----------
    models_dir : Path
        Base models directory
    model_type : str
        Type of model
    run_id : str
        Run identifier

    Returns:
    --------
    Path
        Path to the config file
    """
    model_dir = get_model_dir(models_dir, model_type, run_id)
    return model_dir / f"FINAL_config_{run_id}.json"


def get_scaler_paths(models_dir: Path, model_type: str, run_id: str) -> tuple[Path, Path]:
    """
    Get the scaler file paths based on new naming convention.

    Feature scaler: FINAL_scaler_{run_id}_feature.pkl
    Label scaler: FINAL_scaler_{run_id}_label.pkl

    Parameters:
    -----------
    models_dir : Path
        Base models directory
    model_type : str
        Type of model
    run_id : str
        Run identifier

    Returns:
    --------
    Tuple[Path, Path]
        (feature_scaler_path, label_scaler_path)
    """
    model_dir = get_model_dir(models_dir, model_type, run_id)
    feature_scaler_path = model_dir / f"FINAL_scaler_{run_id}_feature.pkl"
    label_scaler_path = model_dir / f"FINAL_scaler_{run_id}_label.pkl"
    return feature_scaler_path, label_scaler_path


def load_model_config(config_path: Path) -> ModelConfig | None:
    """
    Load model configuration from JSON file.

    Parameters:
    -----------
    config_path : Path
        Path to the model config JSON file

    Returns:
    --------
    ModelConfig or None
        Loaded configuration or None if loading fails
    """
    try:
        if not config_path.exists():
            logger.warning("Config file not found: %s", config_path)
            return None

        config = ModelConfig.from_json(config_path)
        logger.info("Loaded model config for %s (run_id: %s)", config.model_type, config.run_id)
        logger.info("  - IS_WINDOWING: %s", config.is_windowing)
        if config.is_windowing:
            logger.info(
                f"  - input_width: {config.input_width}, label_width: {config.label_width}, shift: {config.shift}"
            )
        logger.info("  - target_transform: %s", config.target_transform)
        logger.info("  - feature_scaling: %s", config.feature_scaling)
        logger.info("  - Number of features in config: %s", len(config.feature_names))

        return config

    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in config file %s: %s", config_path, e)
        return None
    except Exception as e:
        logger.error("Error loading config from %s: %s", config_path, e)
        logger.error(traceback.format_exc())
        return None


def find_available_run_ids(models_dir: Path, model_type: str) -> list[str]:
    """
    Find all available run_ids for a given model type.

    Parameters:
    -----------
    models_dir : Path
        Base models directory
    model_type : str
        Type of model

    Returns:
    --------
    List[str]
        List of available run_ids
    """
    type_dir = models_dir / model_type

    if not type_dir.exists():
        logger.warning("Model type directory not found: %s", type_dir)
        return []

    run_ids = []
    for subdir in type_dir.iterdir():
        if subdir.is_dir():
            # Check if this directory contains a valid model file
            model_file = subdir / f"FINAL_{model_type}_{subdir.name}.joblib"
            if model_file.exists():
                run_ids.append(subdir.name)

    return sorted(run_ids)


# =============================================================================
# Model Loading (Updated for new directory structure)
# =============================================================================


def load_model(models_dir: Path, model_type: str, run_id: str) -> Any | None:
    """
    Load a single model based on new directory structure.

    Structure: ./models/{model_type}/{run_id}/FINAL_{model_type}_{run_id}.joblib

    Parameters:
    -----------
    models_dir : Path
        Base models directory
    model_type : str
        Type of model (e.g., 'xgb', 'rf', 'cnn_lstm')
    run_id : str
        Run identifier

    Returns:
    --------
    model object or None
        Loaded model or None if loading fails
    """
    model_path = get_model_path(models_dir, model_type, run_id)

    if not model_path.exists():
        # Try to find .keras file for deep learning models
        model_dir = get_model_dir(models_dir, model_type, run_id)
        keras_path = model_dir / f"FINAL_{model_type}_{run_id}.keras"

        if keras_path.exists():
            model_path = keras_path
        else:
            logger.error("Model file not found: %s", model_path)
            return None

    try:
        if model_path.suffix == ".joblib":
            logger.info("Loading %s model from %s", model_type, model_path)
            model = safe_joblib_load(model_path, models_dir)
        elif model_path.suffix == ".keras":
            logger.info("Loading %s Keras model from %s", model_type, model_path)
            model = _get_tf().keras.models.load_model(model_path)
        else:
            logger.error("Unknown model format: %s", model_path.suffix)
            return None

        return model

    except Exception as e:
        logger.error("Error loading %s model from %s: %s", model_type, model_path, e)
        logger.debug("Full traceback:", exc_info=True)
        return None


def load_models(models_dir: Path, model_type: str, run_id: str) -> dict[str, Any]:
    """
    Load model for specified model_type and run_id.

    Parameters:
    -----------
    models_dir : Path
        Base models directory (e.g., './models')
    model_type : str
        Type of model (e.g., 'xgb', 'rf', 'cnn_lstm')
    run_id : str
        Run identifier

    Returns:
    --------
    dict
        Dictionary mapping model type to loaded model
    """
    models_dir = Path(models_dir)
    loaded_models = {}

    model = load_model(models_dir, model_type, run_id)
    if model is not None:
        loaded_models[model_type] = model
        logger.info("Successfully loaded model: %s/%s", model_type, run_id)
    else:
        logger.error("Failed to load model: %s/%s", model_type, run_id)

    return loaded_models


def load_models_multiple(models_dir: Path, model_specs: list[tuple[str, str]]) -> dict[str, Any]:
    """
    Load multiple models with different model_types and run_ids.

    Parameters:
    -----------
    models_dir : Path
        Base models directory
    model_specs : List[Tuple[str, str]]
        List of (model_type, run_id) tuples

    Returns:
    --------
    dict
        Dictionary mapping model_type to loaded model
    """
    models_dir = Path(models_dir)
    loaded_models = {}

    for model_type, run_id in model_specs:
        model = load_model(models_dir, model_type, run_id)
        if model is not None:
            loaded_models[model_type] = model
            logger.info("Successfully loaded model: %s/%s", model_type, run_id)

    if not loaded_models:
        logger.error("Failed to load any models")
    else:
        logger.info("Loaded %s models: %s", len(loaded_models), list(loaded_models.keys()))

    return loaded_models


def load_model_config_for_run(models_dir: Path, model_type: str, run_id: str) -> ModelConfig | None:
    """
    Load model configuration for a specific model_type and run_id.

    Config file: ./models/{model_type}/{run_id}/FINAL_config_{run_id}.json

    Parameters:
    -----------
    models_dir : Path
        Base models directory
    model_type : str
        Type of model
    run_id : str
        Run identifier

    Returns:
    --------
    ModelConfig or None
    """
    config_path = get_config_path(models_dir, model_type, run_id)
    return load_model_config(config_path)


def load_model_configs(models_dir: Path, model_type: str, run_id: str) -> dict[str, ModelConfig]:
    """
    Load model configuration for specified model_type and run_id.

    Parameters:
    -----------
    models_dir : Path
        Base models directory
    model_type : str
        Type of model
    run_id : str
        Run identifier

    Returns:
    --------
    dict
        Dictionary mapping model type to ModelConfig
    """
    model_configs = {}

    config = load_model_config_for_run(models_dir, model_type, run_id)
    if config:
        model_configs[model_type] = config
        logger.info("Loaded config for %s/%s", model_type, run_id)
    else:
        logger.warning("No config found for %s/%s", model_type, run_id)

    return model_configs


def load_scalers(models_dir: Path, model_type: str, run_id: str) -> tuple[Any | None, Any | None]:
    """
    Load feature and label scalers from model directory.

    Scaler files:
    - FINAL_scaler_{run_id}_feature.pkl
    - FINAL_scaler_{run_id}_label.pkl

    If scalers are not found, returns None (scaling will be skipped).

    Parameters:
    -----------
    models_dir : Path
        Base models directory
    model_type : str
        Type of model
    run_id : str
        Run identifier

    Returns:
    --------
    tuple
        (feature_scaler, label_scaler) - either can be None if not found
    """
    models_dir = Path(models_dir)
    feature_scaler_path, label_scaler_path = get_scaler_paths(models_dir, model_type, run_id)

    feature_scaler = None
    label_scaler = None

    # Load feature scaler (optional)
    if feature_scaler_path.exists():
        try:
            feature_scaler = safe_joblib_load(feature_scaler_path, models_dir)
            logger.info("Loaded feature scaler from %s", feature_scaler_path)
        except Exception as e:
            logger.warning("Error loading feature scaler: %s", e)
    else:
        logger.info("No feature scaler found at %s. Feature scaling will be skipped.", feature_scaler_path)

    # Load label scaler (optional)
    if label_scaler_path.exists():
        try:
            label_scaler = safe_joblib_load(label_scaler_path, models_dir)
            logger.info("Loaded label scaler from %s", label_scaler_path)
        except Exception as e:
            logger.warning("Error loading label scaler: %s", e)
    else:
        logger.info("No label scaler found at %s. Label scaling will be skipped.", label_scaler_path)

    return feature_scaler, label_scaler


# Canonical timestamp column candidates (tried in order)
_TIMESTAMP_CANDIDATES = [
    "timestamp",
    "timestamp.1",
    "date",
    "datetime",
    "time",
    "Date",
    "Timestamp",
    "TIMESTAMP",
]


def _resolve_timestamp_column(
    columns: Iterable[str],
    preferred: str = "timestamp",
) -> str | None:
    """Find the best timestamp column from available column names.

    Parameters
    ----------
    columns : iterable of str
        Available column names.
    preferred : str
        Column name to try first.

    Returns
    -------
    str or None
        Resolved column name, or *None* if nothing matches.
    """
    cols_list = list(columns)
    cols_set = set(cols_list)
    if preferred in cols_set:
        return preferred
    for candidate in _TIMESTAMP_CANDIDATES:
        if candidate in cols_set:
            return candidate
    # Fallback: first column containing 'time' or 'date'
    for col in cols_list:
        if "time" in col.lower() or "date" in col.lower():
            return col
    return None


def load_preprocessed_data(data_file):
    """
    Load preprocessed ERA5-Land data from CSV.

    Parameters:
    -----------
    data_file : str or Path
        Path to preprocessed CSV file

    Returns:
    --------
    pandas.DataFrame or None
        Loaded data, or None if loading fails
    """
    try:
        data_file_path = Path(data_file)
        logger.info("Loading preprocessed data from %s", data_file_path)

        if not data_file_path.exists():
            logger.error("Input data file not found: %s", data_file_path)
            return None

        # Guard against OOM from oversized files on shared HPC nodes
        file_size = data_file_path.stat().st_size
        if file_size > MAX_CSV_BYTES:
            logger.error(
                "Input file exceeds size limit (%d bytes > %d): %s",
                file_size,
                MAX_CSV_BYTES,
                data_file_path,
            )
            return None

        # Load the data using pandas
        df = pd.read_csv(data_file_path, low_memory=False)
        # Drop unnamed index column if present (e.g. from to_csv with index=True)
        unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
            logger.info("Dropped unnamed index columns: %s", unnamed_cols)

        # Detect and normalise the timestamp column to 'timestamp'
        ts_col = _resolve_timestamp_column(df.columns)
        if ts_col and ts_col != "timestamp":
            df = df.rename(columns={ts_col: "timestamp"})
            logger.info("Renamed timestamp column '%s' -> 'timestamp'", ts_col)
            ts_col = "timestamp"
        if ts_col:
            df = df.sort_values(by=ts_col)

        logger.info("Loaded data with %s rows and %s columns", len(df), len(df.columns))

        if df.empty:
            logger.error("Loaded DataFrame is empty.")
            return None

        # Check for important columns (warning only)
        required_features = [
            "ta",
            "vpd",
            "sw_in",
            "ppfd_in",
            "ext_rad",
            "ws",
            "annual_mean_temperature",
            "annual_precipitation",
        ]
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            logger.warning("Some important features are missing: %s", missing_features)

        return df

    except Exception as e:
        logger.error("Error loading preprocessed data from %s: %s", data_file, e)
        logger.error(traceback.format_exc())
        return None


# =============================================================================
# Windowing Functions (CONDITIONAL based on config)
# =============================================================================


def create_prediction_windows_improved(df, feature_columns, input_width=8, shift=1, batch_size=32):
    """
    Create time series windows for prediction with explicit metadata tracking.

    NOTE: This function is only called when IS_WINDOWING=True in the model config.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with time series data
    feature_columns : list
        List of feature column names to include in the windows
    input_width : int
        Window size for input sequence
    shift : int
        Number of steps to shift for prediction (default: 1)
    batch_size : int
        Batch size for the TensorFlow dataset

    Returns:
    --------
    tuple: (tf.data.Dataset, list, dict, int)
        - Dataset with windowed features ready for prediction
        - List of metadata dictionaries for each window
        - Dictionary mapping location identifier to sorted indices
        - Total number of windows created
    """
    logger.info(
        "Creating prediction windows with input_width=%s, shift=%s, batch_size=%s", input_width, shift, batch_size
    )
    logger.info("Using features: %s", feature_columns)

    df_processed = df.copy()
    if not df_processed.index.is_unique:
        logger.warning("DataFrame index is not unique. Resetting index for reliable mapping.")
        df_processed = df_processed.reset_index(drop=True)

    window_data = []
    window_metadata = []
    location_indices_map = {}

    # Identify time column for sorting
    time_col = _resolve_timestamp_column(df_processed.columns)
    if time_col:
        logger.info("Using '%s' column for time reference.", time_col)
    else:
        time_cols = [col for col in df_processed.columns if "time" in col.lower() or "date" in col.lower()]
        if time_cols:
            time_col = time_cols[0]
            logger.info("Using '%s' column for time reference.", time_col)
        else:
            logger.warning("No time column found. Sorting will use index only.")

    # Identify locations for grouping
    if "name" in df_processed.columns:
        logger.info("Grouping data by 'name' column.")
        location_groups = [
            (name, df_processed[df_processed["name"] == name].copy()) for name in sorted(df_processed["name"].unique())
        ]
        location_id_col = "name"
    elif "latitude" in df_processed.columns and "longitude" in df_processed.columns:
        logger.info("Grouping data by 'latitude' and 'longitude' columns.")
        latlon_pairs = sorted(df_processed[["latitude", "longitude"]].drop_duplicates().values.tolist())
        location_groups = [
            ((lat, lon), df_processed[(df_processed["latitude"] == lat) & (df_processed["longitude"] == lon)].copy())
            for lat, lon in latlon_pairs
        ]
        location_id_col = ("latitude", "longitude")
    else:
        logger.warning("No location columns found. Treating data as a single sequence.")
        location_groups = [("single_sequence", df_processed.copy())]
        location_id_col = None

    total_windows = 0

    for location_id, group_df in location_groups:
        if time_col and time_col in group_df.columns:
            try:
                group_df = group_df.sort_values(time_col)
                logger.debug("Sorted location %s data by %s", location_id, time_col)
            except Exception as e:
                logger.warning("Error sorting by %s: %s. Using index order.", time_col, e)
                group_df = group_df.sort_index()
        else:
            group_df = group_df.sort_index()

        location_indices_map[location_id] = group_df.index.tolist()

        missing_cols = [col for col in feature_columns if col not in group_df.columns]
        if missing_cols:
            logger.error("Missing required feature columns for location %s: %s", location_id, missing_cols)
            continue

        feature_data = group_df[feature_columns].values

        if len(feature_data) < input_width:
            logger.warning(
                "Location %s has insufficient data points: %s < %s", location_id, len(feature_data), input_width
            )
            continue

        for i in range(len(feature_data) - input_width - shift + 1):
            window = feature_data[i : i + input_width]
            window_data.append(window)

            target_idx = None
            if i + input_width - 1 + shift < len(group_df):
                target_idx = group_df.index[i + input_width - 1 + shift]

            metadata = {
                "location_id": location_id,
                "window_start_idx": group_df.index[i],
                "window_end_idx": group_df.index[i + input_width - 1],
                "prediction_target_idx": target_idx,
                "window_position": i,
                "global_window_index": total_windows,
                "shift": shift,
            }
            window_metadata.append(metadata)
            total_windows += 1

    if not window_data:
        logger.error("No windows were created. Cannot proceed with prediction.")
        return None, [], {}, 0

    try:
        windows_array = np.array(window_data)
        logger.info("Created a total of %s windows with shape %s", total_windows, windows_array.shape)
    except ValueError as ve:
        logger.error("Could not convert windows to numpy array: %s", ve)
        return None, [], {}, 0

    try:
        windows_dataset = _get_tf().data.Dataset.from_tensor_slices(windows_array)
        windows_dataset = windows_dataset.batch(batch_size)
        windows_dataset = windows_dataset.prefetch(_get_tf().data.AUTOTUNE)
        logger.info("Successfully created TensorFlow dataset for prediction.")
    except Exception as e:
        logger.error("Failed to create TensorFlow dataset: %s", e)
        return None, [], {}, 0

    return windows_dataset, window_metadata, location_indices_map, total_windows


def prepare_flat_features(df: pd.DataFrame, feature_columns: list[str]) -> tuple[np.ndarray, list[int]]:
    """
    Prepare flat (non-windowed) features for traditional ML models.

    NOTE: This function is called when IS_WINDOWING=False in the model config.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features
    feature_columns : list
        List of feature column names

    Returns:
    --------
    tuple: (np.ndarray, list)
        - Feature array (n_samples, n_features)
        - List of original DataFrame indices
    """
    logger.info("Preparing flat features (no windowing) with %s features", len(feature_columns))

    df_features = df[feature_columns].copy()
    valid_mask = ~df_features.isna().any(axis=1)

    X = df_features[valid_mask].values
    indices = df.index[valid_mask].tolist()

    logger.info("Prepared %s samples with shape %s", len(X), X.shape)

    return X, indices


# =============================================================================
# Prediction Mapping Functions
# =============================================================================


def map_predictions_to_df_improved(df, predictions, window_metadata, model_name):
    """
    Maps predictions back to the original DataFrame using explicit metadata.
    Now includes metadata columns for verification of mapping correctness.

    Parameters:
    -----------
    df : pandas.DataFrame
        The original DataFrame
    predictions : numpy.ndarray
        Array of model predictions
    window_metadata : list
        List of metadata dictionaries for each window
    model_name : str
        Name of the model for the prediction column

    Returns:
    --------
    pandas.DataFrame
        DataFrame with predictions and metadata mapped back to original indices
    """
    logger.info("Mapping %s %s predictions back to DataFrame", len(predictions), model_name)

    if len(predictions) != len(window_metadata):
        logger.error("Prediction count (%s) doesn't match metadata count (%s)", len(predictions), len(window_metadata))
        n_predictions = min(len(predictions), len(window_metadata))
        logger.warning("Will map only %s predictions", n_predictions)
    else:
        n_predictions = len(predictions)

    target_indices = []
    pred_values = []
    location_ids = []
    window_starts = []
    window_ends = []
    window_positions = []

    for i in range(n_predictions):
        target_idx = window_metadata[i]["prediction_target_idx"]
        if target_idx is not None:
            target_indices.append(target_idx)
            pred_values.append(predictions[i])
            location_ids.append(window_metadata[i]["location_id"])
            window_starts.append(window_metadata[i]["window_start_idx"])
            window_ends.append(window_metadata[i]["window_end_idx"])
            window_positions.append(window_metadata[i]["window_position"])

    if len(target_indices) != len(set(target_indices)):
        n_dups = len(target_indices) - len(set(target_indices))
        logger.info("Found %s duplicate target indices. Averaging overlapping predictions.", n_dups)
        # Average predictions for duplicate indices instead of silently overwriting
        agg_df = pd.DataFrame(
            {
                "pred": pred_values,
                "location": location_ids,
                "window_start": window_starts,
                "window_end": window_ends,
                "window_pos": window_positions,
            },
            index=target_indices,
        )
        agg_df = agg_df.groupby(level=0).agg(
            {
                "pred": "mean",
                "location": "first",
                "window_start": "first",
                "window_end": "last",
                "window_pos": "last",
            }
        )
        pred_series = agg_df["pred"]
        location_series = agg_df["location"]
        window_start_series = agg_df["window_start"]
        window_end_series = agg_df["window_end"]
        window_pos_series = agg_df["window_pos"]
    else:
        pred_series = pd.Series(pred_values, index=target_indices, dtype=float)
        location_series = pd.Series(location_ids, index=target_indices)
        window_start_series = pd.Series(window_starts, index=target_indices)
        window_end_series = pd.Series(window_ends, index=target_indices)
        window_pos_series = pd.Series(window_positions, index=target_indices)

    mapped_count = len(pred_series)
    logger.info("Successfully mapped %s %s predictions", mapped_count, model_name)

    pred_df = pd.DataFrame(index=df.index)
    column_name = f"sap_velocity_{model_name}"

    missing_indices = [idx for idx in target_indices if idx not in df.index]
    if missing_indices:
        logger.warning("Found %s target indices that don't exist in the DataFrame index", len(missing_indices))
        logger.debug("First few missing indices: %s", missing_indices[:5])

    pred_df[column_name] = pred_series
    pred_df[f"{column_name}_location"] = location_series
    pred_df[f"{column_name}_window_start"] = window_start_series
    pred_df[f"{column_name}_window_end"] = window_end_series
    pred_df[f"{column_name}_window_pos"] = window_pos_series

    filled_count = pred_df[column_name].notna().sum()
    if filled_count != mapped_count:
        logger.warning(
            f"Only {filled_count} of {mapped_count} predictions were added to DataFrame. Check index alignment."
        )

    mapped_pct = mapped_count / n_predictions * 100 if n_predictions > 0 else 0
    if mapped_pct < 80:
        logger.warning(
            f"Only {mapped_pct:.1f}% of {model_name} predictions were mapped. Check index alignment and shift value."
        )

    return pred_df


def map_flat_predictions_to_df(
    df: pd.DataFrame, predictions: np.ndarray, indices: list[int], model_name: str
) -> pd.DataFrame:
    """
    Maps flat (non-windowed) predictions back to the original DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The original DataFrame
    predictions : np.ndarray
        Array of model predictions
    indices : list
        List of DataFrame indices corresponding to predictions
    model_name : str
        Name of the model for the prediction column

    Returns:
    --------
    pd.DataFrame
        DataFrame with predictions mapped back to original indices
    """
    logger.info("Mapping %s flat %s predictions back to DataFrame", len(predictions), model_name)

    pred_df = pd.DataFrame(index=df.index)
    column_name = f"sap_velocity_{model_name}"

    pred_series = pd.Series(predictions.flatten(), index=indices, dtype=float)
    pred_df[column_name] = pred_series

    filled_count = pred_df[column_name].notna().sum()
    logger.info("Successfully mapped %s %s predictions", filled_count, model_name)

    return pred_df


def verify_prediction_mapping(df_predictions, model_name):
    """
    Performs basic verification of prediction mapping correctness using metadata.

    Parameters:
    -----------
    df_predictions : pandas.DataFrame
        DataFrame with predictions and metadata
    model_name : str
        Name of the model prediction column to verify

    Returns:
    --------
    bool
        True if basic verification passes, False otherwise
    """
    pred_col = f"sap_velocity_{model_name}"
    loc_col = f"{pred_col}_location"

    if pred_col not in df_predictions.columns:
        logger.error("Prediction column %s not found", pred_col)
        return False

    if loc_col not in df_predictions.columns:
        logger.warning("Metadata columns for %s not found. Verification not possible.", model_name)
        return False

    pred_rows = df_predictions[df_predictions[pred_col].notna()]
    pred_count = len(pred_rows)

    if pred_count == 0:
        logger.warning("No predictions found for %s", model_name)
        return False

    logger.info("Verifying %s predictions for %s", pred_count, model_name)

    if "name" in df_predictions.columns:
        loc_rows = pred_rows.dropna(subset=["name", loc_col])
        location_matches = loc_rows["name"] == loc_rows[loc_col]
        match_pct = location_matches.mean() * 100

        logger.info("Location match: %s% of predictions mapped to correct locations", match_pct)

        if match_pct < 95:
            mismatches = loc_rows[~location_matches].head(5)
            logger.warning("Examples of location mismatches:\n%s", mismatches[["name", loc_col, pred_col]])

    if loc_col in pred_rows.columns:
        location_summary = pred_rows[loc_col].value_counts().head(10)
        logger.info("Predictions by location (top 10):\n%s", location_summary)

    window_start_col = f"{pred_col}_window_start"
    window_end_col = f"{pred_col}_window_end"

    if window_start_col in pred_rows.columns and window_end_col in pred_rows.columns:
        valid_windows = pred_rows[window_end_col] >= pred_rows[window_start_col]
        valid_pct = valid_windows.mean() * 100
        logger.info("Window coherence: %s% of prediction windows are valid", valid_pct)

    logger.info("Prediction mapping verification completed for %s", model_name)
    return True


# =============================================================================
# Feature Preparation (CONFIG-DRIVEN)
# =============================================================================


def get_feature_columns_from_config(df: pd.DataFrame, config: ModelConfig | None) -> list[str]:
    """
    Get feature columns based on model config, with fallback to default detection.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    config : ModelConfig, optional
        Model configuration containing feature_names

    Returns:
    --------
    list
        List of feature column names to use
    """
    if config and config.feature_names:
        # Use features from config
        available = []
        missing = []

        for feature in config.feature_names:
            if feature in df.columns:
                available.append(feature)
            else:
                # Try case-insensitive match
                matched = False
                for col in df.columns:
                    if col.lower() == feature.lower():
                        available.append(col)
                        matched = True
                        logger.info("Matched feature '%s' to column '%s' (case-insensitive)", feature, col)
                        break
                if not matched:
                    missing.append(feature)

        if missing:
            logger.warning("Features from config not found in data: %s", missing)

        logger.info("Using %s features from config", len(available))
        return available
    else:
        # Fallback to default feature detection
        logger.info("No feature_names in config. Using default feature detection.")
        return None  # Signal to use default detection


def prepare_features_from_preprocessed(df, feature_scaler=None, input_width=8, config=None):
    """
    Prepare features from preprocessed data with improved error handling.

    IMPROVED: Now uses config.feature_names if available, otherwise falls back to default detection.

    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed data
    feature_scaler : sklearn.preprocessing.StandardScaler, optional
        Fitted scaler for features
    input_width : int
        Number of time steps for input window
    config : ModelConfig, optional
        Model configuration for feature selection

    Returns:
    --------
    tuple: (pandas.DataFrame, list)
        - Prepared DataFrame with features scaled
        - List of actual feature columns used
    """
    try:
        logger.info("Preparing features from preprocessed data...")
        if df is None or df.empty:
            logger.error("Input DataFrame is None or empty.")
            return None, []

        df_processed = df.copy()

        # NOTE: Temperature conversion K->C is handled by GEE/CDS processors.
        # No conversion here to avoid double-conversion.

        # NOTE: Column names are kept as-is from GEE/CDS processors.
        # Model config feature_names handles name mapping.

        # IMPROVED: Try to get features from config first
        config_features = get_feature_columns_from_config(df_processed, config)

        if config_features:
            # Use features from config
            feature_columns = config_features
            logger.info("Using %s features from model config: %s", len(feature_columns), feature_columns)
        else:
            # Fallback to default ordered features (original behavior)
            ordered_features = [
                "ext_rad",
                "sw_in",
                "ta",
                "ws",
                "vpd",
                "ppfd_in",
                "annual_mean_temperature",
                "annual_precipitation",
                "day sin",
                "week sin",
                "month sin",
                "year sin",
            ]

            available_features = [col for col in ordered_features if col in df_processed.columns]
            missing_features = [col for col in ordered_features if col not in df_processed.columns]
            if missing_features:
                logger.warning("Some default features are missing: %s", missing_features)

            if len(available_features) < 3:
                logger.warning("Very few features available (%s). Results may be unreliable.", len(available_features))
                numeric_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
                excluded_cols = ["sap_velocity", "latitude", "longitude"]
                additional_features = [
                    col for col in numeric_cols if col not in available_features and col not in excluded_cols
                ]
                if additional_features:
                    logger.info("Adding %s additional numeric features.", len(additional_features))
                    available_features.extend(additional_features)

            feature_columns = available_features
            logger.info("Using %s default features: %s", len(feature_columns), feature_columns)

        # Apply scaling if scaler is provided
        if feature_scaler is not None:
            try:
                # Case 1: Scaler knows its feature names (Scikit-Learn 1.0+)
                if hasattr(feature_scaler, "feature_names_in_"):
                    scaler_features = feature_scaler.feature_names_in_
                    # Only scale the columns that exist in the dataframe
                    valid_features = [f for f in scaler_features if f in df_processed.columns]

                    if len(valid_features) == len(scaler_features):
                        logger.info("Scaling %s specific features matching scaler definition.", len(valid_features))
                        df_processed[valid_features] = feature_scaler.transform(df_processed[valid_features])
                    else:
                        missing = set(scaler_features) - set(df_processed.columns)
                        logger.warning("Cannot scale: Data is missing features expected by scaler: %s", missing)

                # Case 2: Scaler only knows the count (Old Scikit-Learn or specific save formats)
                elif hasattr(feature_scaler, "n_features_in_"):
                    n_expected = feature_scaler.n_features_in_
                    n_current = len(feature_columns)

                    if n_current == n_expected:
                        # Exact match
                        df_processed[feature_columns] = feature_scaler.transform(df_processed[feature_columns])

                    elif n_current > n_expected:
                        # DATA has MORE features than SCALER.
                        # Logic: Identify the PFT/Categorical columns and exclude them from scaling.
                        pft_cols = ["MF", "DNF", "ENF", "EBF", "WSA", "WET", "DBF", "SAV"]

                        # Filter to get only continuous columns
                        continuous_cols = [col for col in feature_columns if col not in pft_cols]

                        if len(continuous_cols) == n_expected:
                            logger.info(
                                f"Smart Scaling: Found {len(continuous_cols)} continuous features matching scaler count ({n_expected}). Scaling these only."
                            )
                            df_processed[continuous_cols] = feature_scaler.transform(df_processed[continuous_cols])
                        else:
                            # Fallback: Try to guess based on standard ordering (usually scaling happens before one-hot encoding)
                            logger.warning(
                                f"Could not automatically match {n_expected} scaler features from {n_current} columns. Skipping scaling to avoid errors."
                            )
                    else:
                        logger.warning(
                            f"Feature count mismatch: data has {n_current}, scaler expects {n_expected}. Skipping scaling."
                        )
                else:
                    logger.warning("Feature scaler doesn't appear to be properly fitted. Skipping scaling.")
            except Exception as e:
                logger.error("Error applying feature scaling: %s", e)
                logger.error(traceback.format_exc())
        else:
            logger.info("No feature scaler provided. Using unscaled features.")

        return df_processed, feature_columns

    except Exception as e:
        logger.error("Error in prepare_features_from_preprocessed: %s", e)
        logger.error(traceback.format_exc())
        return None, []


# =============================================================================
# Main Prediction Function (CONFIG-DRIVEN)
# =============================================================================


def make_predictions_improved(
    df, models, feature_columns, label_scaler, input_width=8, shift=1, model_configs=None, transformer=None
):
    """
    Make predictions with improved mapping approach.

    IMPROVED: Now conditionally creates windows based on model config IS_WINDOWING setting.
    Handles optional label_scaler - if None, skips inverse scaling.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the features
    models : dict
        Dictionary of loaded models {model_type: model_object}
    feature_columns : list
        List of feature column names
    label_scaler : sklearn.preprocessing.StandardScaler or None
        Fitted scaler for the target variable (optional - if None, scaling is skipped)
    input_width : int
        Width of the input window (used if windowing enabled)
    shift : int
        Prediction shift
    model_configs : dict, optional
        Dictionary of ModelConfig objects for each model type
    transformer : DataTransformer, optional
        Transformer for applying inverse target transforms

    Returns:
    --------
    pandas.DataFrame
        DataFrame with added prediction columns
    """
    logger.info("Making predictions with improved mapping...")

    if df is None or df.empty:
        logger.error("Input DataFrame is empty or None")
        return pd.DataFrame()

    if not models:
        logger.error("No models provided")
        return df

    if not feature_columns:
        logger.error("No feature columns specified")
        return df

    if model_configs is None:
        model_configs = {}

    if label_scaler is None:
        logger.info("No label scaler provided. Predictions will be in scaled space.")

    results_df = df.copy()

    # Make predictions with each model
    for model_type, model in models.items():
        logger.info("Making predictions with %s model", model_type)

        # Get config for this model
        config = model_configs.get(model_type)

        # Determine if windowing is needed based on config
        if config:
            is_windowing = config.is_windowing
            model_input_width = config.input_width or input_width
            model_shift = config.shift or shift
            logger.info(
                "  Config: IS_WINDOWING=%s, input_width=%s, shift=%s", is_windowing, model_input_width, model_shift
            )
        else:
            # Fallback: determine based on model type
            is_deep_model = model_type.lower() in ["cnn_lstm", "lstm", "transformer", "ann", "gru"]
            is_windowing = is_deep_model
            model_input_width = input_width
            model_shift = shift
            logger.info("  No config found. Using default: IS_WINDOWING=%s (based on model type)", is_windowing)

        try:
            if is_windowing:
                # ========== WINDOWED PREDICTIONS ==========
                logger.info("  Creating windowed predictions for %s", model_type)

                windows_dataset, window_metadata, location_map, total_windows = create_prediction_windows_improved(
                    df, feature_columns, input_width=model_input_width, shift=model_shift
                )

                if windows_dataset is None or not window_metadata:
                    logger.error("Failed to create prediction windows for %s", model_type)
                    continue

                # Handle different model types
                is_keras_model = isinstance(model, _get_tf().keras.Model)

                if is_keras_model:
                    preds_scaled = model.predict(windows_dataset, verbose=0)
                elif hasattr(model, "predict"):
                    # For traditional ML models with windowed input, flatten the windows
                    windows_np = np.concatenate([batch.numpy() for batch in windows_dataset], axis=0)
                    X_flattened = windows_np.reshape(windows_np.shape[0], -1)
                    preds_scaled = model.predict(X_flattened)
                else:
                    logger.error("Model %s lacks a predict method. Skipping.", model_type)
                    continue

                # Standardize prediction shape
                if len(preds_scaled.shape) == 3:
                    logger.info("Model %s produced 3D output with shape %s.", model_type, preds_scaled.shape)
                    preds_scaled = preds_scaled[:, -1, :].reshape(-1, 1)
                elif len(preds_scaled.shape) == 2 and preds_scaled.shape[1] > 1:
                    logger.info(
                        "Model %s produced 2D output with multiple features: %s.", model_type, preds_scaled.shape
                    )
                    preds_scaled = preds_scaled[:, 0].reshape(-1, 1)
                elif len(preds_scaled.shape) == 1:
                    preds_scaled = preds_scaled.reshape(-1, 1)

                # Inverse transform to get actual sap velocity values (only if scaler exists)
                if label_scaler is not None:
                    preds_unscaled = label_scaler.inverse_transform(preds_scaled).flatten()
                    logger.debug("  Applied inverse label scaling for %s", model_type)
                else:
                    preds_unscaled = preds_scaled.flatten()

                # Apply inverse target transform if transformer provided
                if transformer and transformer.should_transform_target():
                    logger.info("  Applying inverse target transform: %s", transformer.target_transform)
                    preds_unscaled = transformer.transform_target(preds_unscaled, inverse=True)

                # Map predictions back to DataFrame with explicit metadata
                mapped_results = map_predictions_to_df_improved(df, preds_unscaled, window_metadata, model_type)

                # Copy ALL columns including metadata
                pred_col = f"sap_velocity_{model_type}"
                results_df[pred_col] = mapped_results[pred_col]

                meta_cols = [
                    f"{pred_col}_location",
                    f"{pred_col}_window_start",
                    f"{pred_col}_window_end",
                    f"{pred_col}_window_pos",
                ]

                for meta_col in meta_cols:
                    if meta_col in mapped_results.columns:
                        results_df[meta_col] = mapped_results[meta_col]

            else:
                # ========== FLAT (NON-WINDOWED) PREDICTIONS ==========
                logger.info("  Creating flat predictions for %s (no windowing)", model_type)

                X, indices = prepare_flat_features(df, feature_columns)

                if len(X) == 0:
                    logger.error("No valid samples for %s", model_type)
                    continue

                # Make predictions
                if hasattr(model, "predict"):
                    preds_scaled = model.predict(X)
                else:
                    logger.error("Model %s lacks a predict method. Skipping.", model_type)
                    continue

                # Ensure 2D for inverse transform
                if len(preds_scaled.shape) == 1:
                    preds_scaled = preds_scaled.reshape(-1, 1)

                # Inverse transform (only if scaler exists)
                if label_scaler is not None:
                    preds_unscaled = label_scaler.inverse_transform(preds_scaled).flatten()
                    logger.debug("  Applied inverse label scaling for %s", model_type)
                else:
                    preds_unscaled = preds_scaled.flatten()

                # Apply inverse target transform if transformer provided
                if transformer and transformer.should_transform_target():
                    logger.info("  Applying inverse target transform: %s", transformer.target_transform)
                    preds_unscaled = transformer.transform_target(preds_unscaled, inverse=True)

                # Map predictions back to DataFrame
                mapped_results = map_flat_predictions_to_df(df, preds_unscaled, indices, model_type)

                pred_col = f"sap_velocity_{model_type}"
                results_df[pred_col] = mapped_results[pred_col]

            # Log prediction statistics
            pred_col = f"sap_velocity_{model_type}"
            non_nan_count = results_df[pred_col].notna().sum()
            logger.info(
                f"Added {non_nan_count} predictions for {model_type} ({non_nan_count / len(results_df) * 100:.1f}% coverage)"
            )

            if non_nan_count > 0:
                logger.info(
                    f"{model_type} predictions range: [{results_df[pred_col].min():.2f}, {results_df[pred_col].max():.2f}]"
                )

        except Exception as e:
            logger.error("Error making predictions with %s: %s", model_type, e)
            logger.error(traceback.format_exc())

    # Create ensemble prediction if multiple models succeeded
    pred_cols = [
        col
        for col in results_df.columns
        if col.startswith("sap_velocity_")
        and not col.startswith("sap_velocity_ensemble")
        and not any(col.endswith(x) for x in ["_location", "_window_start", "_window_end", "_window_pos"])
    ]

    if len(pred_cols) > 1:
        logger.info("Creating ensemble prediction from %s models", len(pred_cols))
        results_df["sap_velocity_ensemble"] = results_df[pred_cols].median(axis=1)
        ensemble_count = results_df["sap_velocity_ensemble"].notna().sum()
        logger.info(
            f"Ensemble has {ensemble_count} predictions ({ensemble_count / len(results_df) * 100:.1f}% coverage)"
        )
    elif len(pred_cols) == 1:
        logger.info("Only one model available (%s). Using it as the ensemble.", pred_cols[0])
        results_df["sap_velocity_ensemble"] = results_df[pred_cols[0]]
    else:
        logger.warning("No predictions were generated by any model.")

    return results_df


# =============================================================================
# Validation Functions (KEPT FROM ORIGINAL)
# =============================================================================


def validate_predictions(df_predictions, feature_columns):
    """
    Validate predictions to detect potential mapping issues.

    Parameters:
    -----------
    df_predictions : pandas.DataFrame
        DataFrame with predictions
    feature_columns : list
        List of feature columns used

    Returns:
    --------
    bool
        True if all validations pass, False otherwise
    """
    logger.info("Validating predictions...")

    pred_cols = [col for col in df_predictions.columns if col.startswith("sap_velocity_")]
    if not pred_cols:
        logger.warning("No prediction columns found to validate")
        return False

    validation_results = {}
    validation_passed = True

    # 1. Check for reasonable value ranges for sap velocity
    for col in pred_cols:
        # Skip metadata columns
        if any(x in col for x in ["_location", "_window_start", "_window_end", "_window_pos"]):
            continue

        pred_data = df_predictions[col].dropna()
        if len(pred_data) == 0:
            logger.error("No predictions in column %s", col)
            validation_results[f"{col}_empty"] = False
            validation_passed = False
            continue

        min_val, max_val = pred_data.min(), pred_data.max()
        if min_val < 0:
            logger.warning("Column %s has negative predictions: min=%s", col, min_val)
            validation_results[f"{col}_negative"] = False
            validation_passed = False
        if max_val > MAX_EXPECTED_SAP_VELOCITY:
            logger.warning(
                f"Column {col} has unusually high values: max={max_val:.2f} (threshold={MAX_EXPECTED_SAP_VELOCITY})"
            )

    # 2. Check for temporal consistency if time column exists
    time_col = None
    if isinstance(df_predictions.index, pd.DatetimeIndex):
        time_col = df_predictions.index
    else:
        time_cols = [col for col in df_predictions.columns if "time" in col.lower() or "date" in col.lower()]
        if time_cols and time_cols[0] in df_predictions.columns:
            try:
                time_col = pd.to_datetime(df_predictions[time_cols[0]])
            except (ValueError, TypeError, KeyError):
                pass

    if time_col is not None:
        for col in pred_cols:
            if any(x in col for x in ["_location", "_window_start", "_window_end", "_window_pos"]):
                continue

            if len(df_predictions[col].dropna()) > 10:
                pred_series = df_predictions[col].dropna()
                if isinstance(time_col, pd.DatetimeIndex):
                    temp_df = pd.DataFrame({col: pred_series}, index=time_col)
                else:
                    temp_df = pd.DataFrame({col: pred_series, "time": time_col})
                    temp_df.set_index("time", inplace=True)

                temp_df = temp_df.sort_index()
                abs_diff = temp_df[col].diff().abs()

                if not abs_diff.empty:
                    max_diff = abs_diff.max()
                    if max_diff > 50:
                        logger.warning("Column %s shows potentially abrupt changes (max diff: %s)", col, max_diff)
                        validation_results[f"{col}_temporal_consistency"] = False
                        validation_passed = False

    # 3. Check correlation with expected driving variables
    key_features = [f for f in ["ta", "vpd", "ppfd_in"] if f in df_predictions.columns]
    for feat in key_features:
        for col in pred_cols:
            if any(x in col for x in ["_location", "_window_start", "_window_end", "_window_pos"]):
                continue

            valid_data = df_predictions[[feat, col]].dropna()
            if len(valid_data) > 10:
                try:
                    corr = valid_data.corr().iloc[0, 1]
                    if abs(corr) < 0.05:
                        logger.warning("Column %s shows very low correlation with %s: %s", col, feat, corr)
                        validation_results[f"{col}_{feat}_correlation"] = False
                        validation_passed = False
                except (ValueError, ZeroDivisionError, IndexError) as exc:
                    logger.debug("Correlation check skipped for %s/%s: %s", feat, col, exc)

    if validation_passed:
        logger.info("All prediction validations passed.")
    else:
        logger.warning("Some prediction validations failed. Check mapping and model outputs.")
        logger.warning("Failed validations: %s", [k for k, v in validation_results.items() if not v])

    return validation_passed


# =============================================================================
# Argument Parsing
# =============================================================================


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Predict sap velocity from preprocessed data using config-driven approach.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory with preprocessed data files.")
    parser.add_argument(
        "--output", type=str, default=None, help="Output directory. Defaults to ./outputs/<input_file_stem>/"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models",
        help="Base directory containing model subdirectories (structure: models/{model_type}/{run_id}/).",
    )
    parser.add_argument("--model-type", type=str, required=True, help="Type of model to use (e.g., xgb, rf, cnn_lstm).")
    parser.add_argument("--run-id", type=str, required=True, help="Run identifier for the model.")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to explicit model config JSON file (overrides auto-detection)."
    )
    parser.add_argument(
        "--input-width",
        type=int,
        default=DEFAULT_PARAMS["INPUT_WIDTH"],
        help="Time steps for sequence model input window (used if not in config).",
    )
    parser.add_argument(
        "--shift",
        type=int,
        default=DEFAULT_PARAMS["SHIFT"],
        help="Prediction shift steps ahead (used if not in config).",
    )
    parser.add_argument(
        "--label-width",
        type=int,
        default=DEFAULT_PARAMS["LABEL_WIDTH"],
        help="Time steps in the label window (informational).",
    )
    parser.add_argument("--validate", action="store_true", help="Perform validation checks on predictions.")
    parser.add_argument(
        "--list-runs", action="store_true", help="List available run_ids for the specified model_type and exit."
    )
    parser.add_argument(
        "--time-scale", type=str, default="daily", choices=["daily", "hourly"], help="Temporal scale of predictions."
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="parquet",
        choices=["parquet", "csv"],
        help="Output file format (parquet for ~70%% size reduction).",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="gzip",
        choices=["gzip", "snappy", "none"],
        help="Compression for Parquet output (gzip=smaller, snappy=faster).",
    )

    return parser.parse_args()


# =============================================================================
# Main Function
# =============================================================================


def _extract_year_from_filename(stem: str) -> str:
    """Extract 4-digit year from filename like prediction_2015_01_daily.

    Parameters
    ----------
    stem : str
        Filename stem (no extension).

    Returns
    -------
    str
        4-digit year string, or 'unknown' if not found.
    """
    match = re.search(r"((?:19|20)\d{2})", stem)
    return match.group(1) if match else "unknown"


def main():
    """Main execution function"""
    start_time = time.time()
    args = parse_args()

    # Validate CLI path components (L2: prevent path traversal via model_type/run_id)
    try:
        model_type = validate_path_component(args.model_type, "model_type")
        run_id = validate_path_component(args.run_id, "run_id")
    except ValueError as e:
        logger.error("Invalid argument: %s", e)
        sys.exit(1)

    # Setup directories
    models_dir = Path(args.models_dir).resolve()

    # Handle --list-runs option
    if args.list_runs:
        available_runs = find_available_run_ids(models_dir, model_type)
        if available_runs:
            logger.info("Available run_ids for %s:", model_type)
            for rid in available_runs:
                logger.info("  - %s", rid)
        else:
            logger.info("No runs found for model type: %s", model_type)
        sys.exit(0)

    # Input directory validation and file discovery
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        logger.error("Input directory not found: %s", input_dir)
        sys.exit(1)
    input_files = list(input_dir.glob("*.csv"))
    logger.info("Found %s input files in %s", len(input_files), input_dir)
    if not input_files:
        logger.error("No CSV files found in %s", input_dir)
        sys.exit(1)
    for input_file in input_files:
        file_start_time = time.time()
        logger.info("Processing input file: %s", input_file)

        if args.output:
            output_dir = Path(args.output)
        else:
            year = _extract_year_from_filename(input_file.stem)
            output_dir = Path("./outputs/prediction") / args.time_scale / year
        output_dir.mkdir(exist_ok=True, parents=True)

        # Log setup information
        logger.info("=" * 60)
        logger.info(" Sap Velocity Prediction Script (Config-Driven) ")
        logger.info("=" * 60)
        logger.info("Input: %s", input_file)
        logger.info("Output: %s", output_dir)
        logger.info("Model: %s/%s", model_type, run_id)
        logger.info("Models dir: %s", models_dir)
        logger.info("Default window params: width=%s, shift=%s", args.input_width, args.shift)

        # Verify model directory exists
        model_dir = get_model_dir(models_dir, model_type, run_id)
        if not model_dir.exists():
            logger.error("Model directory not found: %s", model_dir)
            available_runs = find_available_run_ids(models_dir, model_type)
            if available_runs:
                logger.info("Available run_ids for %s: %s", model_type, available_runs)
            sys.exit(1)

        # Load model
        models = load_models(models_dir, model_type, run_id)
        if not models:
            logger.error("No models were loaded. Exiting.")
            sys.exit(1)
        logger.info("Successfully loaded %s models: %s", len(models), list(models.keys()))

        # Load model config
        if args.config:
            # Use explicit config if provided
            model_configs = {}
            explicit_config = load_model_config(Path(args.config))
            if explicit_config:
                logger.info("Using explicit config from %s", args.config)
                model_configs[model_type] = explicit_config
        else:
            # Load config from model directory
            model_configs = load_model_configs(models_dir, model_type, run_id)

        logger.info("Loaded configs for %s models: %s", len(model_configs), list(model_configs.keys()))

        # Create transformer based on config
        transformer = None
        if model_configs:
            first_config = list(model_configs.values())[0]
            transformer = DataTransformer(first_config)
            if transformer.should_transform_target():
                logger.info("Will apply inverse target transform: %s", transformer.target_transform)

        # Load scalers from model directory (optional - if not found, skip scaling)
        feature_scaler, label_scaler = load_scalers(models_dir, model_type, run_id)

        if label_scaler is None:
            logger.info("No label scaler found. Predictions will not be inverse-scaled.")
        if feature_scaler is None:
            logger.info("No feature scaler found. Features will not be scaled.")

        # Load and prepare data
        df_raw = load_preprocessed_data(input_file)
        if df_raw is None or df_raw.empty:
            logger.error("Failed to load input data. Skipping this input file.")
            continue
        logger.info("Loaded data with %s rows and %s columns", len(df_raw), len(df_raw.columns))

        # Save original index
        original_index = df_raw.index

        # Get primary config for feature preparation
        primary_config = list(model_configs.values())[0] if model_configs else None

        # Prepare features (CONFIG-DRIVEN)
        df_prepared, feature_columns = prepare_features_from_preprocessed(
            df_raw,
            feature_scaler,
            input_width=args.input_width,
            config=primary_config,  # Pass config for feature selection
        )

        if df_prepared is None or not feature_columns:
            logger.error("Failed to prepare features. Skipping this input file.")
            continue
        logger.info("Prepared features: %s", feature_columns)

        # Make predictions (CONFIG-DRIVEN)
        df_predictions = make_predictions_improved(
            df_prepared,
            models,
            feature_columns,
            label_scaler,
            input_width=args.input_width,
            shift=args.shift,
            model_configs=model_configs,
            transformer=transformer,
        )

        # Verify prediction mapping for windowed models
        for mt, config in model_configs.items():
            if config and config.is_windowing:
                verify_prediction_mapping(df_predictions, mt)

        if df_predictions.empty:
            logger.error("Prediction failed or returned empty DataFrame. Skipping this input file.")
            continue

        # Restore original index if possible
        if len(original_index) == len(df_predictions):
            try:
                df_final = df_predictions.set_index(original_index)
                logger.info("Restored original index to predictions DataFrame.")
            except Exception as e:
                logger.warning("Could not restore original index: %s. Using current index.", e)
                df_final = df_predictions
        else:
            logger.warning(
                f"Length mismatch between original ({len(original_index)}) and "
                f"predictions ({len(df_predictions)}). Using current index."
            )
            df_final = df_predictions

        # Validate predictions
        if args.validate:
            validation_passed = validate_predictions(df_final, feature_columns)
            logger.info("Prediction validation %s", "passed" if validation_passed else "failed")

        # Save predictions
        if args.output_format == "parquet":
            ext = ".parquet"
            compression = args.compression if args.compression != "none" else None
        else:
            ext = ".csv"
            compression = None
        output_file = output_dir / f"{input_file.stem}_predictions_{model_type}_{run_id}{ext}"
        try:
            if args.output_format == "parquet":
                df_final.to_parquet(
                    output_file,
                    compression=compression,
                    engine="pyarrow",
                    index=False,
                )
            else:
                df_final.to_csv(output_file, index=False)
            logger.info("Saved predictions to %s", output_file)
        except Exception as e:
            logger.error("Error saving predictions: %s", e)

        # Save config summary
        config_summary_file = output_dir / f"prediction_config_{model_type}_{run_id}.json"
        try:
            config_summary = {
                "model_type": model_type,
                "run_id": run_id,
                "feature_columns_used": feature_columns,
                "default_input_width": args.input_width,
                "default_shift": args.shift,
                "has_feature_scaler": feature_scaler is not None,
                "has_label_scaler": label_scaler is not None,
                "model_config": model_configs[model_type].to_dict() if model_type in model_configs else None,
                "input_file": str(input_file),
                "timestamp": datetime.now().isoformat(),
            }
            with open(config_summary_file, "w") as f:
                json.dump(config_summary, f, indent=2)
            logger.info("Saved config summary to %s", config_summary_file)
        except Exception as e:
            logger.warning("Could not save config summary: %s", e)

        # Finish
        end_time = time.time()
        execution_time = end_time - file_start_time
        logger.info("=" * 60)
        logger.info(" Prediction Script Completed ")
        logger.info(" Execution time: %s seconds (%s minutes)", execution_time, execution_time / 60)
        logger.info("=" * 60)

    # Total execution time
    total_time = time.time() - start_time
    logger.info("All files processed in %ss (%s min)", total_time, total_time / 60)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as se:
        if se.code != 0:
            logger.info("Script exited with code: %s", se.code)
        sys.exit(se.code)
    except Exception as e:
        logger.error("Unhandled error: %s", e)
        logger.error(traceback.format_exc())
        sys.exit(1)
