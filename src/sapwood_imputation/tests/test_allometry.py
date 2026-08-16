"""Tests for per-functional-type log-log allometric fitting."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from src.sapwood_imputation.allometry import (
    GroupFit,
    fit_loglog_by_group,
    predict_asw,
)


def _power_law_frame() -> pd.DataFrame:
    """Noise-free power-law data for two groups.

    angiosperm: ASW = 2.0 * DBH ** 1.5
    gymnosperm: ASW = 0.5 * DBH ** 2.0
    """
    dbh = np.array([5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0], dtype=float)
    rows = []
    for d in dbh:
        rows.append({"group": "angiosperm", "dbh": d, "asw": 2.0 * d**1.5})
        rows.append({"group": "gymnosperm", "dbh": d, "asw": 0.5 * d**2.0})
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_recovers_synthetic_parameters() -> None:
    fits = fit_loglog_by_group(_power_law_frame(), "group", "dbh", "asw")

    ang = fits["angiosperm"]
    gym = fits["gymnosperm"]

    # slope == exponent b
    assert ang.slope == pytest.approx(1.5, rel=1e-6)
    assert gym.slope == pytest.approx(2.0, rel=1e-6)
    # intercept == ln(a)
    assert math.exp(ang.intercept) == pytest.approx(2.0, rel=1e-6)
    assert math.exp(gym.intercept) == pytest.approx(0.5, rel=1e-6)
    # noise-free => R^2 == 1
    assert ang.r2 == pytest.approx(1.0, abs=1e-9)


@pytest.mark.unit
def test_correction_factor_at_least_one() -> None:
    fits = fit_loglog_by_group(_power_law_frame(), "group", "dbh", "asw")
    for fit in fits.values():
        assert fit.correction_factor >= 1.0
    # noise-free => CF ~ 1
    assert fits["angiosperm"].correction_factor == pytest.approx(1.0, abs=1e-6)


@pytest.mark.unit
def test_predict_matches_truth_with_correction() -> None:
    fits = fit_loglog_by_group(_power_law_frame(), "group", "dbh", "asw")
    # held-out DBH
    pred = predict_asw(25.0, "gymnosperm", fits)
    assert pred == pytest.approx(0.5 * 25.0**2.0, rel=1e-3)
    assert pred > 0


@pytest.mark.unit
def test_corrected_prediction_ge_plain_exp() -> None:
    # Inject noise so MSE > 0 and CF > 1; corrected estimate must exceed naive exp.
    rng = np.random.default_rng(0)
    dbh = rng.uniform(5, 50, size=200)
    asw = 1.3 * dbh**1.7 * np.exp(rng.normal(0, 0.3, size=dbh.size))
    df = pd.DataFrame({"group": "angiosperm", "dbh": dbh, "asw": asw})
    fits = fit_loglog_by_group(df, "group", "dbh", "asw")
    fit = fits["angiosperm"]
    assert fit.correction_factor > 1.0
    plain = math.exp(fit.intercept + fit.slope * math.log(20.0))
    corrected = predict_asw(20.0, "angiosperm", fits)
    assert corrected > plain


@pytest.mark.unit
def test_group_with_too_few_points_is_skipped() -> None:
    df = pd.DataFrame(
        {
            "group": ["angiosperm", "angiosperm", "angiosperm", "gymnosperm"],
            "dbh": [10.0, 20.0, 30.0, 25.0],
            "asw": [100.0, 200.0, 300.0, 400.0],
        }
    )
    fits = fit_loglog_by_group(df, "group", "dbh", "asw", min_n=2)
    assert "angiosperm" in fits
    assert "gymnosperm" not in fits  # only 1 point


@pytest.mark.unit
def test_predict_unknown_group_raises() -> None:
    fits = fit_loglog_by_group(_power_law_frame(), "group", "dbh", "asw")
    with pytest.raises(KeyError):
        predict_asw(20.0, "unknown", fits)


@pytest.mark.unit
def test_groupfit_is_immutable() -> None:
    fit = GroupFit(
        group="angiosperm",
        intercept=0.0,
        slope=1.0,
        correction_factor=1.0,
        r2=1.0,
        mse=0.0,
        n=5,
    )
    with pytest.raises((AttributeError, TypeError)):
        fit.slope = 2.0  # type: ignore[misc]
