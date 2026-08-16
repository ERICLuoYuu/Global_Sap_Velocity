"""Tests for the ASW/DBH unit-error gate.

The gate exists to keep a mis-united tree out of a log-log regression, where a
100x error in ASW is a 4.6-unit displacement in the response. These tests pin
both halves of its asymmetry: the low side rejects, the high side only counts.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.sapwood_imputation.geometry import (
    MIN_SAPWOOD_FRACTION,
    OVERSIZED_SAPWOOD_FRACTION,
    basal_area,
    describe,
    implausible_mask,
    is_implausible,
    oversized_mask,
    sapwood_fraction,
)


def _asw_for_fraction(dbh: float, fraction: float) -> float:
    return fraction * math.pi * (dbh / 2.0) ** 2


@pytest.mark.unit
def test_basal_area_matches_circle_formula() -> None:
    assert basal_area(pd.Series([20.0])).iloc[0] == pytest.approx(math.pi * 100.0)


@pytest.mark.unit
def test_non_positive_dbh_gives_no_fraction() -> None:
    """Zero or negative DBH must not divide -- it yields 'unknown', not 'bad'."""
    asw = pd.Series([100.0, 100.0])
    dbh = pd.Series([0.0, -20.0])
    assert sapwood_fraction(asw, dbh).isna().all()
    assert not implausible_mask(asw, dbh).any()


@pytest.mark.unit
def test_typical_tree_passes() -> None:
    dbh = pd.Series([30.0])
    asw = pd.Series([_asw_for_fraction(30.0, 0.6)])  # 60%: the archive median
    assert not implausible_mask(asw, dbh).any()
    assert not oversized_mask(asw, dbh).any()


@pytest.mark.unit
def test_dbh_recorded_in_millimetres_is_rejected() -> None:
    """The real NOR_HUR failure: DBH in mm scales the fraction by 1/100."""
    true_dbh, fraction = 29.3, 0.5
    asw = pd.Series([_asw_for_fraction(true_dbh, fraction)])
    as_mm = pd.Series([true_dbh * 10])
    assert implausible_mask(asw, as_mm).all()
    # ... and the same tree with the correct unit sails through.
    assert not implausible_mask(asw, pd.Series([true_dbh])).any()


@pytest.mark.unit
def test_threshold_is_exclusive_at_the_boundary() -> None:
    dbh = pd.Series([40.0, 40.0])
    asw = pd.Series(
        [
            _asw_for_fraction(40.0, MIN_SAPWOOD_FRACTION * 0.99),
            _asw_for_fraction(40.0, MIN_SAPWOOD_FRACTION * 1.01),
        ]
    )
    assert list(implausible_mask(asw, dbh)) == [True, False]


@pytest.mark.unit
def test_missing_values_are_unknown_not_implausible() -> None:
    """A tree awaiting imputation must never be pre-emptively condemned."""
    asw = pd.Series([np.nan, 200.0, np.nan])
    dbh = pd.Series([30.0, np.nan, np.nan])
    assert not implausible_mask(asw, dbh).any()


@pytest.mark.unit
def test_oversized_is_reported_but_never_rejected() -> None:
    """ASW > basal area is impossible, yet 3.9% of the archive reports it.

    Poyatos et al. (2021) fitted on those trees, so refusing them here would
    hold new data to a stricter standard than the relationship being reproduced.
    """
    dbh = pd.Series([25.0, 25.0])
    asw = pd.Series([_asw_for_fraction(25.0, 1.5), _asw_for_fraction(25.0, 5.0)])
    assert oversized_mask(asw, dbh).all()
    assert not implausible_mask(asw, dbh).any()
    assert OVERSIZED_SAPWOOD_FRACTION == 1.0


@pytest.mark.unit
def test_asw_equal_to_basal_area_is_kept() -> None:
    """Eight archive sites set ASW to the whole cross-section for every tree."""
    dbh = pd.Series([31.1])
    asw = pd.Series([_asw_for_fraction(31.1, 1.0)])
    assert not implausible_mask(asw, dbh).any()


@pytest.mark.unit
def test_scalar_predicate_agrees_with_the_mask() -> None:
    dbh = [12.0, 40.0, 40.0, 40.0]
    asw = [
        _asw_for_fraction(12.0, 0.55),
        _asw_for_fraction(40.0, 0.001),
        _asw_for_fraction(40.0, 2.0),
        _asw_for_fraction(40.0, MIN_SAPWOOD_FRACTION / 2),
    ]
    mask = implausible_mask(pd.Series(asw), pd.Series(dbh))
    scalar = [is_implausible(a, d) for a, d in zip(asw, dbh)]  # noqa: B905 - 3.9 has no strict=
    assert list(mask) == scalar == [False, True, False, True]


@pytest.mark.unit
def test_scalar_predicate_tolerates_junk() -> None:
    for asw, dbh in ((None, 30.0), (200.0, None), (float("nan"), 30.0), ("junk", 30.0), (200.0, "junk")):
        assert is_implausible(asw, dbh) is False


@pytest.mark.unit
def test_describe_counts_both_tails() -> None:
    dbh = pd.Series([30.0, 30.0, 30.0, 30.0])
    asw = pd.Series(
        [
            _asw_for_fraction(30.0, 0.001),  # implausible
            _asw_for_fraction(30.0, 0.6),  # fine
            _asw_for_fraction(30.0, 2.0),  # oversized
            np.nan,  # not checkable
        ]
    )
    summary = describe(asw, dbh)
    assert summary["checkable_trees"] == 3
    assert summary["implausible"] == 1
    assert summary["oversized"] == 1
    assert summary["max_fraction"] == pytest.approx(2.0)
