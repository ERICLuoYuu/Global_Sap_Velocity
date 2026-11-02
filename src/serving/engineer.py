"""From a daily base frame to a model matrix, by the trainer's own steps.

A transcription of the per-site loop of ``test_hyperparameter_tuning_ML_spatial_stratified.py``, which
``forward_selection/data_loader.py`` repeats: index on ``solar_TIMESTAMP`` in time order, the eight
cyclical time features, every feature-engineering group under the run's contract, the PFT one-hot,
``float32``, infinities read as missing. The functions called are the trainer's; nothing is re-derived.

The trainer drops a row when any feature is missing. :func:`model_matrix` keeps such a row and leaves
the decision to :meth:`src.serving.model_bundle.ModelBundle.predict`, which does not predict it. A feature
the frame lacks altogether raises :class:`AbsentFeatures`; the trainer skips a site that lacks one.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.feature_contract.definitions import SERVABLE_CONTRACT, validate_contract
from src.hyperparameter_optimization.feature_engineering import apply_all_feature_engineering
from src.hyperparameter_optimization.shap_constants import PFT_COLUMNS
from src.hyperparameter_optimization.training_utils import add_time_features, build_pft_onehot
from src.serving.base_columns import PFT_COLUMN, TIME_COLUMN

TIME_SCALE = "daily"
REQUIRED_COLUMNS = (TIME_COLUMN, "latitude", "longitude")


class AbsentFeatures(KeyError):
    """Model features an engineered frame lacks: feature engineering builds a feature only when its inputs allow.

    ``missing`` names them in the model's order. :func:`src.serving.grid_predict.predict_cells` leaves such a
    cell unpredicted.
    """

    def __init__(self, cell_id: str, missing: Sequence[str]):
        self.cell_id = cell_id
        self.missing = tuple(missing)
        super().__init__(f"{cell_id}: the engineered frame cannot supply {list(self.missing)}")

    def __reduce__(self):
        return type(self), (self.cell_id, self.missing)


def engineer_cell(daily: pd.DataFrame, contract: str = SERVABLE_CONTRACT) -> pd.DataFrame:
    """Every engineered feature of one place, indexed by ``solar_TIMESTAMP`` in time order.

    ``daily`` holds one row per day with the base columns of :mod:`src.serving.base_columns`;
    ``latitude`` and ``longitude`` are constant columns, as the trainer writes them.
    """
    contract = validate_contract(contract)
    missing = [column for column in REQUIRED_COLUMNS if column not in daily.columns]
    if missing:
        raise KeyError(f"daily frame lacks required column(s) {missing}")
    frame = daily.set_index(TIME_COLUMN).sort_index()
    frame = add_time_features(frame, datetime_column=None)
    frame, _ = apply_all_feature_engineering(frame, TIME_SCALE, contract=contract)
    return frame


def model_matrix(
    engineered: pd.DataFrame,
    feature_names: Sequence[str],
    cell_id: str,
    pft_column: str = PFT_COLUMN,
) -> pd.DataFrame:
    """The model's columns in the model's order, ``float32``, infinities as NaN.

    :class:`AbsentFeatures` names every feature the frame cannot supply: a model must never be fed a shorter row.
    A frame with no ``pft_column`` value on any row supplies no one-hot, so the model's PFT features are absent too.
    """
    names = list(feature_names)
    if len(set(names)) != len(names):
        raise ValueError(f"{cell_id}: feature list holds duplicates")
    frame = engineered
    if any(name in PFT_COLUMNS for name in names):
        if pft_column not in frame.columns:
            raise KeyError(f"{cell_id}: the model uses PFT one-hot columns and the frame has no {pft_column!r}")
        if frame[pft_column].isna().all():
            raise AbsentFeatures(cell_id, [name for name in names if name in PFT_COLUMNS])
        frame = build_pft_onehot(frame.copy(), pft_column, cell_id, PFT_COLUMNS)
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise AbsentFeatures(cell_id, missing)
    return frame[names].astype(np.float32).replace([np.inf, -np.inf], np.nan)
