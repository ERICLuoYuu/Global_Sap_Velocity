"""Tests for the per-site PyTorch ANN ensemble (Task 1)."""

from __future__ import annotations

import numpy as np
from src.fu_ann_sensitivity.ann import ensemble_predict, train_ensemble


def test_ensemble_learns_linear_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4))
    # SM = col 2 (positive driver), VPD = col 1 (negative driver)
    y = 1.0 * X[:, 2] - 1.0 * X[:, 1] + 0.1 * rng.normal(size=400)
    models, test_r = train_ensemble(X, y, n_repeats=3, hidden=10, seed=42)
    assert len(models) == 3
    assert test_r > 0.5
    preds = ensemble_predict(models, X)
    assert preds.shape == (400,)
    assert np.corrcoef(preds, y)[0, 1] > 0.7


def test_train_ensemble_is_reproducible():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 4))
    y = X[:, 2] - X[:, 1] + 0.1 * rng.normal(size=300)
    m1, _ = train_ensemble(X, y, n_repeats=2, seed=11)
    m2, _ = train_ensemble(X, y, n_repeats=2, seed=11)
    p1 = ensemble_predict(m1, X)
    p2 = ensemble_predict(m2, X)
    np.testing.assert_allclose(p1, p2, rtol=1e-4, atol=1e-4)
