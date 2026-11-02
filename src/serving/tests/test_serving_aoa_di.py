"""Tests for src/serving/aoa_di.py: the dissimilarity index of a served model matrix."""

import numpy as np
import pytest
from src.serving.aoa_di import load_di_reference, matrix_di

from src.aoa import core
from src.aoa.prepare import build_aoa_reference, load_aoa_reference

NAMES = ["ta", "vpd", "sw_in", "ENF"]


@pytest.fixture
def reference_path(tmp_path):
    rng = np.random.default_rng(5)
    X = rng.normal(size=(60, len(NAMES)))
    X[:, 3] = rng.integers(0, 2, 60)
    folds = np.tile(np.arange(3), 20)
    return build_aoa_reference(X, folds, np.array([0.5, 1.0, 0.2, 0.1]), NAMES, tmp_path, "unit")


def test_matrix_di_is_the_core_prediction_di(reference_path):
    reference = load_di_reference(reference_path, NAMES)
    raw = load_aoa_reference(reference_path)
    values = np.random.default_rng(9).normal(size=(25, len(NAMES))).astype(np.float32)
    weighted = core.apply_importance_weights(
        core.standardize_features(values.astype(float), raw["feature_means"], raw["feature_stds"]),
        raw["feature_weights"],
    )
    expected = core.compute_prediction_di(weighted, core.build_kdtree(raw["reference_cloud_weighted"]), raw["d_bar"])
    got = matrix_di(reference, values)
    assert got.dtype == np.float32  # nosec B101
    np.testing.assert_allclose(got, expected, rtol=1e-5)


def test_a_row_with_a_missing_feature_has_no_di(reference_path):
    reference = load_di_reference(reference_path, NAMES)
    values = np.ones((4, len(NAMES)), dtype=np.float32)
    values[1, 2] = np.nan
    values[3, 0] = np.inf
    got = matrix_di(reference, values)
    assert np.isnan(got[[1, 3]]).all()  # nosec B101
    assert np.isfinite(got[[0, 2]]).all()  # nosec B101


def test_no_finite_row_gives_all_nan(reference_path):
    reference = load_di_reference(reference_path, NAMES)
    assert np.isnan(matrix_di(reference, np.full((3, len(NAMES)), np.nan, dtype=np.float32))).all()  # nosec B101


def test_refuses_a_reference_in_another_feature_order(reference_path):
    with pytest.raises(ValueError, match="feature"):
        load_di_reference(reference_path, NAMES[::-1])


def test_refuses_a_reference_for_other_features(reference_path):
    with pytest.raises(ValueError, match="feature"):
        load_di_reference(reference_path, [*NAMES[:-1], "WSA"])


def test_refuses_a_matrix_of_another_width(reference_path):
    reference = load_di_reference(reference_path, NAMES)
    with pytest.raises(ValueError, match="columns"):
        matrix_di(reference, np.ones((2, 3), dtype=np.float32))


def test_the_reference_carries_its_identity(reference_path):
    reference = load_di_reference(reference_path, NAMES)
    raw = load_aoa_reference(reference_path)
    assert reference.threshold == raw["threshold"]  # nosec B101
    assert reference.d_bar == raw["d_bar"]  # nosec B101
    assert len(reference.sha256) == 64  # nosec B101
    assert reference.path == str(reference_path)  # nosec B101
