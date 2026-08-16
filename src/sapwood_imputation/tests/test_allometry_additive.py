"""Tests for the additive form: ln(ASW) ~ ln(DBH) + functional type.

The distinguishing property of this model against two independent regressions
is that the group shifts the intercept only, so the predicted ratio between
groups is constant in DBH. Several tests below pin exactly that.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.sapwood_imputation.allometry import (
    AdditiveFit,
    fit_loglog_by_group,
    fit_loglog_with_group_offset,
    predict_asw_additive,
)

SHARED_SLOPE = 1.7
A_ANGIO = 1.3
A_GYMNO = 0.8


def _parallel_frame(n: int = 60, noise: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Two groups sharing one exponent, differing only in the coefficient.

    This is data the additive model can fit exactly (when noise == 0).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for coefficient, group in ((A_ANGIO, "angiosperm"), (A_GYMNO, "gymnosperm")):
        dbh = rng.uniform(5.0, 60.0, size=n)
        asw = coefficient * dbh**SHARED_SLOPE
        if noise:
            asw = asw * np.exp(rng.normal(0.0, noise, size=n))
        rows.append(pd.DataFrame({"group": group, "dbh": dbh, "asw": asw}))
    return pd.concat(rows, ignore_index=True)


def _diverging_frame() -> pd.DataFrame:
    """Two groups with genuinely different exponents (1.5 vs 2.0)."""
    dbh = np.array([5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0], dtype=float)
    rows = []
    for d in dbh:
        rows.append({"group": "angiosperm", "dbh": d, "asw": 2.0 * d**1.5})
        rows.append({"group": "gymnosperm", "dbh": d, "asw": 0.5 * d**2.0})
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_recovers_shared_slope_and_offset() -> None:
    fit = fit_loglog_with_group_offset(_parallel_frame(), "group", "dbh", "asw")

    assert fit.slope == pytest.approx(SHARED_SLOPE, rel=1e-9)
    # exp(intercept + offset) is each group's multiplicative coefficient.
    recovered = {name: math.exp(fit.intercept + offset) for name, offset in fit.offsets}
    assert recovered["angiosperm"] == pytest.approx(A_ANGIO, rel=1e-9)
    assert recovered["gymnosperm"] == pytest.approx(A_GYMNO, rel=1e-9)
    assert fit.r2 == pytest.approx(1.0, abs=1e-12)


@pytest.mark.unit
def test_reference_offset_is_exactly_zero() -> None:
    fit = fit_loglog_with_group_offset(_parallel_frame(), "group", "dbh", "asw")
    assert fit.offset(fit.reference) == 0.0
    assert fit.reference in fit.groups


@pytest.mark.unit
def test_predictions_are_invariant_to_the_reference_choice() -> None:
    """The reference is a parameterisation detail, never a modelling choice."""
    df = _parallel_frame(noise=0.25)
    as_angio = fit_loglog_with_group_offset(df, "group", "dbh", "asw", reference="angiosperm")
    as_gymno = fit_loglog_with_group_offset(df, "group", "dbh", "asw", reference="gymnosperm")

    assert as_angio.reference != as_gymno.reference
    for group in ("angiosperm", "gymnosperm"):
        for dbh in (8.0, 30.0, 70.0):
            assert predict_asw_additive(dbh, group, as_angio) == pytest.approx(
                predict_asw_additive(dbh, group, as_gymno), rel=1e-9
            )


@pytest.mark.unit
def test_group_ratio_is_constant_across_dbh() -> None:
    """The defining behaviour: one slope => a DBH-independent group ratio.

    Two independent regressions on the same data give a ratio that drifts, so
    this test fails if the imputer ever silently reverts to the interaction form.
    """
    df = _parallel_frame(noise=0.3, seed=5)
    fit = fit_loglog_with_group_offset(df, "group", "dbh", "asw")
    ratios = [
        predict_asw_additive(d, "angiosperm", fit) / predict_asw_additive(d, "gymnosperm", fit)
        for d in (5.0, 15.0, 40.0, 100.0)
    ]
    assert ratios == pytest.approx([ratios[0]] * len(ratios), rel=1e-9)

    expected = math.exp(fit.offset("angiosperm") - fit.offset("gymnosperm"))
    assert ratios[0] == pytest.approx(expected, rel=1e-9)


@pytest.mark.unit
def test_additive_cannot_fit_diverging_slopes_but_interaction_can() -> None:
    """Documents the cost of the constraint on data that violates it."""
    df = _diverging_frame()
    additive = fit_loglog_with_group_offset(df, "group", "dbh", "asw")
    separate = fit_loglog_by_group(df, "group", "dbh", "asw")

    assert additive.r2 < 1.0  # one slope cannot serve exponents 1.5 and 2.0
    assert separate["angiosperm"].r2 == pytest.approx(1.0, abs=1e-9)
    assert separate["gymnosperm"].r2 == pytest.approx(1.0, abs=1e-9)
    # The shared slope lands between the two true exponents.
    assert 1.5 < additive.slope < 2.0


@pytest.mark.unit
def test_adding_group_never_lowers_r2() -> None:
    fit = fit_loglog_with_group_offset(_parallel_frame(noise=0.2), "group", "dbh", "asw")
    assert fit.r2 >= fit.r2_without_group
    # Parallel groups with different coefficients => the group term earns its place.
    assert fit.r2 > fit.r2_without_group


@pytest.mark.unit
def test_residual_variance_uses_all_estimated_parameters() -> None:
    """dof = n - 3 for two groups (intercept, slope, one indicator)."""
    df = _parallel_frame(noise=0.3, seed=11)
    fit = fit_loglog_with_group_offset(df, "group", "dbh", "asw")

    x = np.log(df["dbh"].to_numpy(dtype=float))
    y = np.log(df["asw"].to_numpy(dtype=float))
    offsets = dict(fit.offsets)
    predicted = fit.intercept + fit.slope * x + df["group"].map(offsets).to_numpy(dtype=float)
    sse = float(np.sum((y - predicted) ** 2))

    assert fit.n == len(df)
    assert fit.mse == pytest.approx(sse / (fit.n - 3), rel=1e-9)
    assert fit.correction_factor == pytest.approx(math.exp(fit.mse / 2.0), rel=1e-12)


@pytest.mark.unit
def test_correction_factor_is_shared_and_at_least_one() -> None:
    fit = fit_loglog_with_group_offset(_parallel_frame(noise=0.4), "group", "dbh", "asw")
    assert fit.correction_factor > 1.0
    # Noise-free data => no bias to correct.
    clean = fit_loglog_with_group_offset(_parallel_frame(), "group", "dbh", "asw")
    assert clean.correction_factor == pytest.approx(1.0, abs=1e-9)


@pytest.mark.unit
def test_corrected_prediction_exceeds_plain_exponentiation() -> None:
    fit = fit_loglog_with_group_offset(_parallel_frame(noise=0.35), "group", "dbh", "asw")
    plain = math.exp(fit.intercept + fit.slope * math.log(20.0) + fit.offset("angiosperm"))
    assert predict_asw_additive(20.0, "angiosperm", fit) > plain


@pytest.mark.unit
def test_group_below_min_n_is_excluded_and_unpredictable() -> None:
    df = pd.DataFrame(
        {
            "group": ["angiosperm"] * 6 + ["gymnosperm"],
            "dbh": [10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 22.0],
            "asw": [80.0, 150.0, 240.0, 350.0, 480.0, 800.0, 300.0],
        }
    )
    fit = fit_loglog_with_group_offset(df, "group", "dbh", "asw", min_n=2)
    assert fit.groups == ("angiosperm",)
    with pytest.raises(KeyError):
        predict_asw_additive(22.0, "gymnosperm", fit)


@pytest.mark.unit
def test_single_group_still_fits() -> None:
    df = _parallel_frame()
    df = df[df["group"] == "angiosperm"]
    fit = fit_loglog_with_group_offset(df, "group", "dbh", "asw")
    assert fit.groups == ("angiosperm",)
    assert fit.offset("angiosperm") == 0.0
    assert fit.slope == pytest.approx(SHARED_SLOPE, rel=1e-9)
    # With no group term there is nothing to add, so both R2 values coincide.
    assert fit.r2 == pytest.approx(fit.r2_without_group, rel=1e-12)


@pytest.mark.unit
def test_no_fittable_group_raises() -> None:
    df = pd.DataFrame({"group": ["angiosperm"], "dbh": [20.0], "asw": [200.0]})
    with pytest.raises(ValueError, match="cannot fit the allometry"):
        fit_loglog_with_group_offset(df, "group", "dbh", "asw", min_n=2)


@pytest.mark.unit
def test_constant_dbh_raises_instead_of_returning_a_minimum_norm_solution() -> None:
    """lstsq happily returns garbage for a rank-deficient design; refuse it."""
    df = pd.DataFrame(
        {
            "group": ["angiosperm"] * 3 + ["gymnosperm"] * 3,
            "dbh": [20.0] * 6,
            "asw": [180.0, 200.0, 220.0, 120.0, 130.0, 140.0],
        }
    )
    with pytest.raises(ValueError, match="Rank-deficient"):
        fit_loglog_with_group_offset(df, "group", "dbh", "asw")


@pytest.mark.unit
def test_unknown_reference_raises() -> None:
    with pytest.raises(ValueError, match="not among the fitted groups"):
        fit_loglog_with_group_offset(_parallel_frame(), "group", "dbh", "asw", reference="fern")


@pytest.mark.unit
def test_nonpositive_dbh_rejected_at_prediction() -> None:
    fit = fit_loglog_with_group_offset(_parallel_frame(), "group", "dbh", "asw")
    for bad in (0.0, -5.0, float("nan")):
        with pytest.raises(ValueError, match="positive finite"):
            predict_asw_additive(bad, "angiosperm", fit)


@pytest.mark.unit
def test_additive_fit_is_immutable() -> None:
    fit = fit_loglog_with_group_offset(_parallel_frame(), "group", "dbh", "asw")
    assert isinstance(fit, AdditiveFit)
    with pytest.raises((AttributeError, TypeError)):
        fit.slope = 2.0  # type: ignore[misc]
    # offsets is a tuple of pairs, not a dict, so it cannot be edited in place.
    with pytest.raises((AttributeError, TypeError)):
        fit.offsets[0] = ("angiosperm", 99.0)  # type: ignore[index]


# ---------------------------------------------------------------------------
# Back-transform: which correction factor is applied, and is it entitled to be
# ---------------------------------------------------------------------------
def _skewed_frame(seed: int = 17) -> pd.DataFrame:
    """A fit whose log residuals are strongly left-skewed, as the real pool is.

    A few trees with far less sapwood than the law predicts -- the thin-sapwood
    ring-porous pattern -- is what pulls mean(exp(r)) below exp(sigma^2/2).
    """
    rng = np.random.default_rng(seed)
    dbh = rng.uniform(10, 60, size=300)
    noise = rng.normal(0.0, 0.2, size=300)
    noise[:25] -= rng.uniform(1.0, 2.5, size=25)  # long negative tail
    return pd.DataFrame(
        {
            "group": "angiosperm",
            "dbh": dbh,
            "asw": 1.2 * dbh**1.7 * np.exp(noise),
        }
    )


@pytest.mark.unit
def test_both_correction_factors_are_always_reported() -> None:
    fit = fit_loglog_with_group_offset(_skewed_frame(), "group", "dbh", "asw")
    assert fit.baskerville_factor > 1.0
    assert fit.smearing_factor > 1.0
    # On skewed residuals the closed form overshoots the distribution-free one.
    assert fit.baskerville_factor > fit.smearing_factor


@pytest.mark.unit
def test_applied_factor_follows_the_requested_back_transform() -> None:
    df = _skewed_frame()
    baskerville = fit_loglog_with_group_offset(df, "group", "dbh", "asw", back_transform="baskerville")
    smearing = fit_loglog_with_group_offset(df, "group", "dbh", "asw", back_transform="smearing")
    none = fit_loglog_with_group_offset(df, "group", "dbh", "asw", back_transform="none")

    assert baskerville.correction_factor == pytest.approx(baskerville.baskerville_factor)
    assert smearing.correction_factor == pytest.approx(smearing.smearing_factor)
    assert none.correction_factor == 1.0
    # Only the back-transform moves; the regression itself is untouched.
    assert smearing.slope == pytest.approx(baskerville.slope)
    assert smearing.intercept == pytest.approx(baskerville.intercept)


@pytest.mark.unit
def test_back_transform_changes_the_prediction_it_is_supposed_to() -> None:
    df = _skewed_frame()
    baskerville = fit_loglog_with_group_offset(df, "group", "dbh", "asw", back_transform="baskerville")
    none = fit_loglog_with_group_offset(df, "group", "dbh", "asw", back_transform="none")

    ratio = predict_asw_additive(30.0, "angiosperm", baskerville) / predict_asw_additive(30.0, "angiosperm", none)
    assert ratio == pytest.approx(baskerville.baskerville_factor)


@pytest.mark.unit
def test_unknown_back_transform_is_refused_not_silently_ignored() -> None:
    with pytest.raises(ValueError, match="unknown back_transform"):
        fit_loglog_with_group_offset(_parallel_frame(), "group", "dbh", "asw", back_transform="magic")


@pytest.mark.unit
def test_residual_normality_flag_tracks_the_residuals() -> None:
    skewed = fit_loglog_with_group_offset(_skewed_frame(), "group", "dbh", "asw")
    assert skewed.residual_skew < -0.5
    assert skewed.residuals_look_normal is False

    rng = np.random.default_rng(4)
    dbh = rng.uniform(10, 60, size=400)
    clean = pd.DataFrame(
        {
            "group": "angiosperm",
            "dbh": dbh,
            "asw": 1.2 * dbh**1.7 * np.exp(rng.normal(0.0, 0.2, size=400)),
        }
    )
    assert fit_loglog_with_group_offset(clean, "group", "dbh", "asw").residuals_look_normal is True


@pytest.mark.unit
def test_fit_records_the_dbh_range_it_saw() -> None:
    """Predicting outside this range is extrapolating a power law."""
    fit = fit_loglog_with_group_offset(_skewed_frame(), "group", "dbh", "asw")
    assert 10.0 <= fit.dbh_min < fit.dbh_max <= 60.0
