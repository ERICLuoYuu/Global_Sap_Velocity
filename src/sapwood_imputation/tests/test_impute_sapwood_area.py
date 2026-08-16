"""Tests for the dataframe-level ASW imputation orchestration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.sapwood_imputation.functional_type import ANGIOSPERM, GYMNOSPERM
from src.sapwood_imputation.impute_sapwood_area import (
    SOURCE_IMPLAUSIBLE,
    SOURCE_IMPUTED,
    SOURCE_MEASURED,
    SOURCE_UNRESOLVED,
    add_functional_type,
    fit_frame_from_dataframe,
    fit_from_dataframe,
    impute_dataframe,
    predict_asw_additive,
)


def _training_frame() -> pd.DataFrame:
    """Many measured trees so the global fit is well determined."""
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(80):
        d = rng.uniform(5, 60)
        rows.append({"pl_species": "Pinus sylvestris", "pl_dbh": d, "pl_sapw_area": 0.6 * d**1.9})
        d2 = rng.uniform(5, 60)
        rows.append({"pl_species": "Quercus ilex", "pl_dbh": d2, "pl_sapw_area": 1.4 * d2**1.6})
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_add_functional_type() -> None:
    df = pd.DataFrame({"pl_species": ["Pinus sylvestris", "Quercus ilex", ""]})
    out = add_functional_type(df)
    assert out["pl_functional_type"].tolist() == [GYMNOSPERM, ANGIOSPERM, "unknown"]
    # input not mutated
    assert "pl_functional_type" not in df.columns


@pytest.mark.unit
def test_measured_rows_preserved() -> None:
    train = _training_frame()
    fits = fit_from_dataframe(train)
    out = impute_dataframe(train, fits)
    measured = out[out["pl_sapw_area"].notna()]
    assert (measured["pl_sapw_area_source"] == SOURCE_MEASURED).all()
    # filled equals original where measured
    assert np.allclose(measured["pl_sapw_area_filled"], measured["pl_sapw_area"])


@pytest.mark.unit
def test_missing_asw_is_imputed() -> None:
    train = _training_frame()
    fits = fit_from_dataframe(train)
    target = pd.DataFrame({"pl_species": ["Pinus sylvestris"], "pl_dbh": [30.0], "pl_sapw_area": [np.nan]})
    out = impute_dataframe(target, fits)
    row = out.iloc[0]
    assert row["pl_sapw_area_source"] == SOURCE_IMPUTED
    assert row["pl_sapw_area_filled"] > 0
    # within a reasonable factor of the true power law
    assert row["pl_sapw_area_filled"] == pytest.approx(0.6 * 30.0**1.9, rel=0.15)


@pytest.mark.unit
def test_original_column_never_mutated() -> None:
    train = _training_frame()
    fits = fit_from_dataframe(train)
    target = pd.DataFrame({"pl_species": ["Pinus sylvestris"], "pl_dbh": [30.0], "pl_sapw_area": [np.nan]})
    before = target["pl_sapw_area"].copy()
    impute_dataframe(target, fits)
    pd.testing.assert_series_equal(target["pl_sapw_area"], before)


@pytest.mark.unit
def test_missing_dbh_is_unresolved() -> None:
    train = _training_frame()
    fits = fit_from_dataframe(train)
    target = pd.DataFrame({"pl_species": ["Pinus sylvestris"], "pl_dbh": [np.nan], "pl_sapw_area": [np.nan]})
    out = impute_dataframe(target, fits)
    row = out.iloc[0]
    assert row["pl_sapw_area_source"] == SOURCE_UNRESOLVED
    assert pd.isna(row["pl_sapw_area_filled"])


@pytest.mark.unit
def test_unknown_functional_type_is_unresolved() -> None:
    train = _training_frame()
    fits = fit_from_dataframe(train)
    target = pd.DataFrame({"pl_species": ["Myrtaceae"], "pl_dbh": [30.0], "pl_sapw_area": [np.nan]})
    out = impute_dataframe(target, fits)
    assert out.iloc[0]["pl_sapw_area_source"] == SOURCE_UNRESOLVED


@pytest.mark.unit
def test_output_schema_has_new_columns() -> None:
    train = _training_frame()
    fits = fit_from_dataframe(train)
    out = impute_dataframe(train, fits)
    for col in ("pl_sapw_area_filled", "pl_sapw_area_source", "pl_functional_type"):
        assert col in out.columns


@pytest.mark.unit
def test_nonpositive_asw_excluded_from_fit() -> None:
    # A zero / negative ASW must not poison the log fit.
    train = _training_frame()
    bad = pd.DataFrame({"pl_species": ["Pinus sylvestris"], "pl_dbh": [30.0], "pl_sapw_area": [0.0]})
    combined = pd.concat([train, bad], ignore_index=True)
    fit = fit_from_dataframe(combined)
    # fit still finite and sensible
    assert np.isfinite(fit.slope)
    assert dict(fit.n_by_group)[GYMNOSPERM] == 80  # the zero-ASW row excluded


@pytest.mark.unit
def test_functional_type_shifts_intercept_not_slope() -> None:
    """The imputer must use the additive form, not two independent regressions.

    Under one shared slope the predicted angiosperm/gymnosperm ratio is the
    same at every DBH; under separate slopes it drifts (and can invert).
    """
    fit = fit_from_dataframe(_training_frame())
    ratios = [
        predict_asw_additive(d, ANGIOSPERM, fit) / predict_asw_additive(d, GYMNOSPERM, fit)
        for d in (10.0, 25.0, 50.0, 80.0)
    ]
    assert ratios[0] == pytest.approx(ratios[-1], rel=1e-9)
    assert ratios[0] == pytest.approx(np.exp(fit.offset(ANGIOSPERM) - fit.offset(GYMNOSPERM)), rel=1e-9)


# ---------------------------------------------------------------------------
# The unit-error gate, seen from the imputer
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_implausible_measurement_is_flagged_not_passed_through() -> None:
    """A measured ASW its own DBH contradicts must not reach the output."""
    fit = fit_from_dataframe(_training_frame())
    # DBH in mm (293 instead of 29.3) against a correct ASW -- the NOR_HUR case.
    target = pd.DataFrame({"pl_species": ["Picea abies"], "pl_dbh": [293.0], "pl_sapw_area": [242.2]})
    row = impute_dataframe(target, fit).iloc[0]
    assert row["pl_sapw_area_source"] == SOURCE_IMPLAUSIBLE
    assert np.isnan(row["pl_sapw_area_filled"])


@pytest.mark.unit
def test_implausible_row_is_not_silently_imputed_instead() -> None:
    """The suspect DBH must not be reused: imputing from it would trust it.

    A 293 cm 'DBH' fed to the allometry yields an ASW an order of magnitude too
    large, which is worse than admitting the tree is unusable.
    """
    fit = fit_from_dataframe(_training_frame())
    target = pd.DataFrame({"pl_species": ["Picea abies"], "pl_dbh": [293.0], "pl_sapw_area": [242.2]})
    row = impute_dataframe(target, fit).iloc[0]
    assert row["pl_sapw_area_source"] != SOURCE_IMPUTED
    assert row["pl_sapw_area_source"] != SOURCE_MEASURED


@pytest.mark.unit
def test_missing_asw_with_large_dbh_still_imputes() -> None:
    """The gate needs both values; an unmeasured tree stays eligible."""
    fit = fit_from_dataframe(_training_frame())
    target = pd.DataFrame({"pl_species": ["Picea abies"], "pl_dbh": [40.0], "pl_sapw_area": [np.nan]})
    assert impute_dataframe(target, fit).iloc[0]["pl_sapw_area_source"] == SOURCE_IMPUTED


@pytest.mark.unit
def test_implausible_trees_are_excluded_from_the_fit() -> None:
    """One mis-united tree must not be allowed to bend the regression."""
    clean = _training_frame()
    poisoned = pd.concat(
        [
            clean,
            pd.DataFrame(
                # ASW 1000x too small, at a leverage point in ln(DBH).
                {"pl_species": ["Pinus sylvestris"] * 3, "pl_dbh": [55.0, 58.0, 60.0], "pl_sapw_area": [1.0, 1.1, 1.2]}
            ),
        ],
        ignore_index=True,
    )
    assert len(fit_frame_from_dataframe(poisoned)) == len(fit_frame_from_dataframe(clean))
    assert fit_from_dataframe(poisoned).slope == pytest.approx(fit_from_dataframe(clean).slope, rel=1e-9)


@pytest.mark.unit
def test_asw_exceeding_basal_area_still_votes() -> None:
    """Archive convention, not an error -- see geometry module docstring."""
    clean = _training_frame()
    oversized = pd.concat(
        [clean, pd.DataFrame({"pl_species": ["Quercus ilex"], "pl_dbh": [30.0], "pl_sapw_area": [3.0 * np.pi * 225]})],
        ignore_index=True,
    )
    assert len(fit_frame_from_dataframe(oversized)) == len(fit_frame_from_dataframe(clean)) + 1
