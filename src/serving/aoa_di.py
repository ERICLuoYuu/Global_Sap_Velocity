"""The dissimilarity index (Meyer & Pebesma 2021) of a served model matrix.

:func:`src.serving.grid_predict.predict_cells` hands this the very rows it predicts, so a cell-day has a DI
exactly when it has a prediction: a row with any missing feature, which the model does not predict, has none.
The arithmetic is :mod:`src.aoa.core`'s; this module only binds a reference to a model's feature order.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.aoa import core
from src.aoa.prepare import load_aoa_reference


@dataclass(frozen=True)
class DIReference:
    path: str
    sha256: str
    feature_names: tuple
    means: np.ndarray
    stds: np.ndarray
    weights: np.ndarray
    d_bar: float
    threshold: float
    tree: Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_di_reference(path, feature_names: Sequence[str]) -> DIReference:
    """A reference whose features are the model's, in the model's order; anything else is refused."""
    raw = load_aoa_reference(Path(path))
    names = list(feature_names)
    if raw["feature_names"] != names:
        raise ValueError(
            f"{path}: reference feature names {raw['feature_names']} are not the model's feature names {names} in order"
        )
    return DIReference(
        path=str(path),
        sha256=_sha256(Path(path)),
        feature_names=tuple(names),
        means=np.asarray(raw["feature_means"], dtype=float),
        stds=np.asarray(raw["feature_stds"], dtype=float),
        weights=np.asarray(raw["feature_weights"], dtype=float),
        d_bar=raw["d_bar"],
        threshold=raw["threshold"],
        tree=core.build_kdtree(raw["reference_cloud_weighted"]),
    )


def matrix_di(reference: DIReference, values: np.ndarray) -> np.ndarray:
    """float32 DI per row; NaN where any feature is not finite."""
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != len(reference.feature_names):
        raise ValueError(f"matrix has shape {values.shape}; the reference has {len(reference.feature_names)} columns")
    out = np.full(len(values), np.nan, dtype=np.float32)
    valid = np.isfinite(values).all(axis=1)
    if not valid.any():
        return out
    weighted = core.apply_importance_weights(
        core.standardize_features(values[valid].astype(float), reference.means, reference.stds), reference.weights
    )
    out[valid] = core.compute_prediction_di(weighted, reference.tree, reference.d_bar, workers=1)
    return out
