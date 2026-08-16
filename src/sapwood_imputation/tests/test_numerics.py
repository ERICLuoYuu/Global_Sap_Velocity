"""Tests for unit-aware coercion of provider metadata cells.

These cover the failure that motivated the module: a cell like ``'16.5 cm'``
is truthy under ``pd.notna`` but raises ``TypeError`` when compared to a
number, and becomes ``NaN`` under a plain ``pd.to_numeric``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.sapwood_imputation.impute_sapwood_area import (
    ASW_COL,
    DBH_COL,
    SOURCE_IMPUTED,
    SOURCE_MEASURED,
    SOURCE_UNRESOLVED,
    coerce_inputs,
    fit_from_dataframe,
    impute_dataframe,
)
from src.sapwood_imputation.numerics import coerce_series, defect_count


@pytest.mark.unit
def test_unit_bearing_dbh_is_converted_not_dropped() -> None:
    values = pd.Series(["16.5 cm", "165 mm", "0.165 m", "30"])
    coerced, _ = coerce_series(values, field=DBH_COL)
    # every form is the same tree diameter in cm
    assert coerced.iloc[0] == pytest.approx(16.5)
    assert coerced.iloc[1] == pytest.approx(16.5)
    assert coerced.iloc[2] == pytest.approx(16.5)
    assert coerced.iloc[3] == pytest.approx(30.0)
    assert coerced.dtype == float


@pytest.mark.unit
def test_unit_bearing_sapwood_area_is_converted() -> None:
    values = pd.Series(["207.2 cm2", "20720 mm2"])
    coerced, _ = coerce_series(values, field=ASW_COL)
    assert coerced.iloc[0] == pytest.approx(207.2)
    assert coerced.iloc[1] == pytest.approx(207.2)


@pytest.mark.unit
def test_absent_tokens_are_not_defects() -> None:
    values = pd.Series(["n/a", "", "NA", None])
    coerced, counts = coerce_series(values, field=DBH_COL)
    assert coerced.isna().all()
    # 'absent' means the provider never measured it -- data, not a defect.
    assert defect_count(counts) == 0


@pytest.mark.unit
def test_unparseable_cell_is_counted_as_a_defect() -> None:
    values = pd.Series(["see notes", "12.0"])
    _, counts = coerce_series(values, field=DBH_COL)
    assert defect_count(counts) == 1


@pytest.mark.unit
def test_coerce_inputs_tolerates_missing_columns() -> None:
    df = pd.DataFrame({"pl_species": ["Pinus sylvestris"]})
    measured, dbh, counts = coerce_inputs(df)
    assert measured.isna().all()
    assert dbh.isna().all()
    assert counts[DBH_COL]["missing_column"] == 1


# ---------------------------------------------------------------------------
# Regression: the string cell must not crash, and must still be usable
# ---------------------------------------------------------------------------
def _training_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(80):
        d = rng.uniform(5, 60)
        rows.append({"pl_species": "Pinus sylvestris", "pl_dbh": d, "pl_sapw_area": 0.6 * d**1.9})
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_string_dbh_does_not_raise_and_is_imputed() -> None:
    """Regression: '30 cm' used to raise TypeError in the > 0 guard."""
    fits = fit_from_dataframe(_training_frame())
    target = pd.DataFrame({"pl_species": ["Pinus sylvestris"], "pl_dbh": ["30 cm"], "pl_sapw_area": [np.nan]})
    out = impute_dataframe(target, fits)  # must not raise
    row = out.iloc[0]
    assert row["pl_sapw_area_source"] == SOURCE_IMPUTED
    # the tree is recovered, not lost: same answer as a bare numeric 30.0
    assert row["pl_sapw_area_filled"] == pytest.approx(0.6 * 30.0**1.9, rel=0.15)


@pytest.mark.unit
def test_string_measured_asw_does_not_raise_and_counts_as_measured() -> None:
    """Regression: '207.2 cm2' used to raise TypeError in the > 0 guard."""
    fits = fit_from_dataframe(_training_frame())
    target = pd.DataFrame({"pl_species": ["Pinus sylvestris"], "pl_dbh": [30.0], "pl_sapw_area": ["207.2 cm2"]})
    out = impute_dataframe(target, fits)  # must not raise
    row = out.iloc[0]
    assert row["pl_sapw_area_source"] == SOURCE_MEASURED
    assert row["pl_sapw_area_filled"] == pytest.approx(207.2)


@pytest.mark.unit
def test_unit_bearing_trees_contribute_to_the_fit() -> None:
    """A 'cm'-suffixed cohort must widen the fit, not be silently discarded."""
    numeric = _training_frame()
    stringy = numeric.copy()
    stringy["pl_dbh"] = [f"{v} cm" for v in stringy["pl_dbh"]]
    stringy["pl_sapw_area"] = [f"{v} cm2" for v in stringy["pl_sapw_area"]]

    fit_numeric = fit_from_dataframe(numeric)
    fit_stringy = fit_from_dataframe(stringy)

    assert fit_stringy.n == fit_numeric.n
    assert dict(fit_stringy.n_by_group) == dict(fit_numeric.n_by_group)
    assert fit_stringy.slope == pytest.approx(fit_numeric.slope, rel=1e-9)
    assert fit_stringy.offset("gymnosperm") == pytest.approx(fit_numeric.offset("gymnosperm"), rel=1e-9)


@pytest.mark.unit
def test_unparseable_dbh_yields_unresolved_not_crash() -> None:
    fits = fit_from_dataframe(_training_frame())
    target = pd.DataFrame({"pl_species": ["Pinus sylvestris"], "pl_dbh": ["see notes"], "pl_sapw_area": [np.nan]})
    out = impute_dataframe(target, fits)
    assert out.iloc[0]["pl_sapw_area_source"] == SOURCE_UNRESOLVED
