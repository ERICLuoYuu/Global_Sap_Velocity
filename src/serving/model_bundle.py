"""A FINAL model as serving loads it: its files, the checks that make it servable, and prediction.

The trainer (``test_hyperparameter_tuning_ML_spatial_stratified.py``) writes, per run,
``FINAL_<type>_<run>.joblib``, ``FINAL_scaler_<run>_feature.pkl``, ``FINAL_config_<run>.json`` and, when the
target was transformed, ``FINAL_target_transformer_<run>.pkl``. It scales only the non-PFT columns, with a
``StandardScaler`` fitted on a DataFrame, so the scaler records their names and order.

:func:`load_bundle` refuses, rather than warns about, every state in which a prediction would be silently
wrong: a model trained under another feature contract, a windowed model, retired soil-moisture names, or a
scaler whose columns are not the model's non-PFT columns in the model's order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from src.feature_contract.definitions import SERVABLE_CONTRACT, assert_no_retired_swc_features
from src.hyperparameter_optimization.shap_constants import PFT_COLUMNS

DAILY = "daily"


@dataclass(frozen=True)
class BundlePaths:
    model: Path
    scaler: Path
    config: Path
    transformer: Path


def bundle_paths(models_root, run_id: str, model_type: str = "xgb") -> BundlePaths:
    directory = Path(models_root) / model_type / run_id
    return BundlePaths(
        model=directory / f"FINAL_{model_type}_{run_id}.joblib",
        scaler=directory / f"FINAL_scaler_{run_id}_feature.pkl",
        config=directory / f"FINAL_config_{run_id}.json",
        transformer=directory / f"FINAL_target_transformer_{run_id}.pkl",
    )


@dataclass(frozen=True)
class ModelBundle:
    run_id: str
    model: Any
    scaler: Any
    transformer: Optional[Any]
    feature_names: tuple
    numeric_names: tuple
    numeric_index: tuple
    config: dict

    def predict(self, matrix: pd.DataFrame) -> np.ndarray:
        """Target on the original scale per row; NaN where any feature is missing (the trainer dropped those)."""
        if list(matrix.columns) != list(self.feature_names):
            raise ValueError(f"{self.run_id}: matrix columns must be the model's feature names in the trained order")
        values = matrix.to_numpy(dtype=np.float32, copy=True)
        out = np.full(len(values), np.nan, dtype=float)
        valid = np.isfinite(values).all(axis=1)
        if not valid.any():
            return out
        rows = values[valid]
        index = list(self.numeric_index)
        numeric = pd.DataFrame(rows[:, index], columns=list(self.numeric_names))
        rows[:, index] = self.scaler.transform(numeric)
        raw = np.asarray(self.model.predict(rows), dtype=float)
        out[valid] = self.transformer.inverse_transform(raw) if self.transformer is not None else raw
        return out


def validate_config(config: dict, run_id: str, require_contract: str = SERVABLE_CONTRACT) -> tuple:
    """The model's feature names, after refusing a config that cannot be served; see the module docstring."""
    contract = config.get("feature_contract")
    if contract != require_contract:
        raise ValueError(f"{run_id} was trained under feature contract {contract!r}; serving needs {require_contract!r}")
    if (config.get("data_info") or {}).get("IS_WINDOWING"):
        raise NotImplementedError(f"{run_id} is a windowed model; serving builds one row per day, not windows")
    if config.get("time_scale") != DAILY:
        raise ValueError(f"{run_id} has time_scale {config.get('time_scale')!r}; serving builds daily rows")
    names = config.get("feature_names")
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
        raise ValueError(f"{run_id}: FINAL_config has no usable feature_names")
    if len(set(names)) != len(names):
        raise ValueError(f"{run_id}: FINAL_config feature_names holds duplicates")
    assert_no_retired_swc_features(names, f"FINAL_config of {run_id}")
    return tuple(names)


def _require_files(paths: BundlePaths, needs_transformer: bool) -> None:
    wanted = [paths.model, paths.scaler, paths.config] + ([paths.transformer] if needs_transformer else [])
    missing = [str(path) for path in wanted if not path.exists()]
    if missing:
        raise FileNotFoundError(f"model files missing: {missing}")


def load_bundle(
    models_root, run_id: str, model_type: str = "xgb", require_contract: str = SERVABLE_CONTRACT
) -> ModelBundle:
    paths = bundle_paths(models_root, run_id, model_type)
    if not paths.config.exists():
        raise FileNotFoundError(f"model config missing: {paths.config}")
    with open(paths.config) as fh:
        config = json.load(fh)
    names = validate_config(config, run_id, require_contract)
    transform = (config.get("preprocessing") or {}).get("target_transform")
    _require_files(paths, needs_transformer=bool(transform))

    numeric_names = tuple(name for name in names if name not in PFT_COLUMNS)
    # joblib unpickles: these are this project's own trainer outputs under models_root, never external files.
    scaler = joblib.load(paths.scaler)
    scaler_names = tuple(getattr(scaler, "feature_names_in_", ()))
    if scaler_names != numeric_names:
        raise ValueError(
            f"{run_id}: the scaler was fitted on {list(scaler_names)}, "
            f"not on the model's non-PFT features in order {list(numeric_names)}"
        )
    transformer = None
    if transform:
        transformer = joblib.load(paths.transformer)
        if getattr(transformer, "method", None) != transform:
            raise ValueError(
                f"{run_id}: target transformer method {getattr(transformer, 'method', None)!r} "
                f"is not the config's {transform!r}"
            )
    model = joblib.load(paths.model)
    n_in = getattr(model, "n_features_in_", len(names))
    if n_in != len(names):
        raise ValueError(f"{run_id}: the model expects {n_in} features and the config names {len(names)}")
    return ModelBundle(
        run_id=run_id,
        model=model,
        scaler=scaler,
        transformer=transformer,
        feature_names=names,
        numeric_names=numeric_names,
        numeric_index=tuple(i for i, name in enumerate(names) if name not in PFT_COLUMNS),
        config=config,
    )
