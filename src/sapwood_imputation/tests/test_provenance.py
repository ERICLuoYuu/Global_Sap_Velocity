"""Tests for telling a measured sapwood area from a provider-computed one.

The detector's whole job is to notice that a value is *too good* -- that it lies
exactly on a curve. So the positive controls here are noise-free power laws, and
the negative controls are the same laws with realistic scatter.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.sapwood_imputation.provenance import (
    ALLOMETRY_DERIVED,
    ASSUMED_BASAL_AREA,
    MEASURED,
    MIN_GROUP_SIZE,
    NOT_ASSESSED,
    POLICY_ALL,
    POLICY_DECLARED,
    POLICY_NONE,
    PROVIDER_DECLARED,
    assess,
    declared_estimate,
    estimated_mask,
    excluded_verdicts,
    summarise,
)

SITE, SPECIES, DBH, ASW = "_site", "pl_species", "pl_dbh", "pl_sapw_area"


def _frame(site: str, dbh, asw, species: str = "Pinus sylvestris") -> pd.DataFrame:
    return pd.DataFrame({SITE: site, SPECIES: species, DBH: list(dbh), ASW: list(asw)})


def _power_law(n: int = 10, a: float = 0.6, b: float = 1.9, sigma: float = 0.0, seed: int = 0):
    dbh = np.linspace(12.0, 50.0, n)
    asw = a * dbh**b
    if sigma:
        asw = asw * np.exp(np.random.default_rng(seed).normal(0.0, sigma, n))
    return dbh, asw


@pytest.mark.unit
def test_noise_free_power_law_is_flagged() -> None:
    """POSITIVE CONTROL: ASW = a*DBH^b exactly is a formula, not a measurement."""
    dbh, asw = _power_law()
    verdicts = assess(_frame("X", dbh, asw), SITE, SPECIES, DBH, ASW)
    assert [v.verdict for v in verdicts] == [ALLOMETRY_DERIVED]
    assert verdicts[0].r2 == pytest.approx(1.0, abs=1e-12)
    assert verdicts[0].is_estimated


@pytest.mark.unit
def test_realistic_scatter_is_not_flagged() -> None:
    """NEGATIVE CONTROL: real trees scatter, and must survive."""
    dbh, asw = _power_law(sigma=0.15, seed=1)
    verdicts = assess(_frame("X", dbh, asw), SITE, SPECIES, DBH, ASW)
    assert [v.verdict for v in verdicts] == [MEASURED]
    assert not verdicts[0].is_estimated


@pytest.mark.unit
def test_asw_equal_to_basal_area_gets_its_own_verdict() -> None:
    """ASW == pi*r^2 also gives R2 == 1, but it is a different provider act."""
    dbh = np.linspace(10.0, 45.0, 8)
    asw = math.pi * (dbh / 2.0) ** 2
    verdicts = assess(_frame("X", dbh, asw), SITE, SPECIES, DBH, ASW)
    assert [v.verdict for v in verdicts] == [ASSUMED_BASAL_AREA]
    assert verdicts[0].slope == pytest.approx(2.0, abs=1e-9)
    assert verdicts[0].median_fraction == pytest.approx(1.0, abs=1e-9)


@pytest.mark.unit
def test_small_group_is_not_assessed_and_is_kept() -> None:
    """Absence of evidence is not evidence: a 3-tree site must not be discarded."""
    dbh, asw = _power_law(n=MIN_GROUP_SIZE - 1)
    verdicts = assess(_frame("X", dbh, asw), SITE, SPECIES, DBH, ASW)
    assert [v.verdict for v in verdicts] == [NOT_ASSESSED]
    assert not verdicts[0].is_estimated


@pytest.mark.unit
def test_groups_are_per_site_and_species() -> None:
    """A derived species must not condemn a measured one at the same site."""
    d1, a1 = _power_law(n=8)  # derived
    d2, a2 = _power_law(n=8, sigma=0.2, seed=2)  # measured
    df = pd.concat(
        [_frame("X", d1, a1, "Pinus sylvestris"), _frame("X", d2, a2, "Quercus ilex")],
        ignore_index=True,
    )
    got = {(v.species, v.verdict) for v in assess(df, SITE, SPECIES, DBH, ASW)}
    assert got == {("Pinus sylvestris", ALLOMETRY_DERIVED), ("Quercus ilex", MEASURED)}


@pytest.mark.unit
def test_mask_selects_only_the_derived_rows_under_policy_all() -> None:
    d1, a1 = _power_law(n=8)
    d2, a2 = _power_law(n=8, sigma=0.2, seed=2)
    df = pd.concat(
        [_frame("X", d1, a1, "Pinus sylvestris"), _frame("X", d2, a2, "Quercus ilex")],
        ignore_index=True,
    )
    mask = estimated_mask(df, SITE, SPECIES, DBH, ASW, policy=POLICY_ALL)
    assert mask[: len(d1)].all()
    assert not mask[len(d1) :].any()


# ---------------------------------------------------------------------------
# Detecting is not excluding: the policy decides how far a finding travels.
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_default_policy_keeps_a_statistically_derived_site() -> None:
    """THE DEFAULT. A perfect power law is still *detected* -- but not excluded.

    Only ESTIMATED_SITES rests on a provider document; the R2 cut is an
    inference, so acting on it is opt-in.
    """
    dbh, asw = _power_law(n=8)
    df = _frame("SOME_SITE", dbh, asw)

    assert [v.verdict for v in assess(df, SITE, SPECIES, DBH, ASW)] == [ALLOMETRY_DERIVED]
    assert not estimated_mask(df, SITE, SPECIES, DBH, ASW).any()


@pytest.mark.unit
def test_default_policy_still_excludes_the_two_documented_sites() -> None:
    for site in ("ESP_GDN", "AUT_ZOE"):
        dbh, asw = _power_law(n=8, sigma=0.2, seed=5)  # looks measured
        mask = estimated_mask(_frame(site, dbh, asw), SITE, SPECIES, DBH, ASW)
        assert mask.all(), f"{site} is documented as estimated and must always be barred"


@pytest.mark.unit
def test_policy_none_excludes_nothing_at_all() -> None:
    """Reproduces the published Poyatos pool, documented sites included."""
    dbh, asw = _power_law(n=8)
    df = pd.concat([_frame("ESP_GDN", dbh, asw), _frame("X", dbh, asw)], ignore_index=True)
    assert not estimated_mask(df, SITE, SPECIES, DBH, ASW, policy=POLICY_NONE).any()


@pytest.mark.unit
def test_policies_are_ordered_by_strictness() -> None:
    assert excluded_verdicts(POLICY_NONE) < excluded_verdicts(POLICY_DECLARED)
    assert excluded_verdicts(POLICY_DECLARED) < excluded_verdicts(POLICY_ALL)


@pytest.mark.unit
def test_unknown_policy_is_refused_not_silently_ignored() -> None:
    """A typo must not quietly widen or narrow what gets dropped from the fit."""
    with pytest.raises(ValueError, match="unknown exclusion policy"):
        excluded_verdicts("declared_only")
    dbh, asw = _power_law(n=8)
    with pytest.raises(ValueError, match="unknown exclusion policy"):
        estimated_mask(_frame("X", dbh, asw), SITE, SPECIES, DBH, ASW, policy="strict")


@pytest.mark.unit
def test_declared_site_is_flagged_despite_scattered_values() -> None:
    """ES-Gdn's estimate does not track DBH, so only the registry can catch it."""
    dbh = [114.0, 139.0, 107.0, 150.0]
    asw = [10.05, 10.68, 10.68, 10.05]
    verdicts = assess(_frame("ESP_GDN", dbh, asw, "Olea europaea"), SITE, SPECIES, DBH, ASW)
    assert [v.verdict for v in verdicts] == [PROVIDER_DECLARED]
    assert verdicts[0].is_estimated


@pytest.mark.unit
def test_declared_registry_survives_separator_drift() -> None:
    for spelling in ("ESP_GDN", "esp-gdn", "Esp_Gdn"):
        assert declared_estimate(spelling)
    assert declared_estimate("ESP_ABR") is None


@pytest.mark.unit
def test_declaration_outranks_a_measured_looking_series() -> None:
    """Even a perfectly scattered series stays flagged if the provider said so."""
    dbh, asw = _power_law(sigma=0.25, seed=9)
    verdicts = assess(_frame("ESP_GDN", dbh, asw), SITE, SPECIES, DBH, ASW)
    assert verdicts[0].verdict == PROVIDER_DECLARED


@pytest.mark.unit
def test_missing_site_column_degrades_to_no_flags() -> None:
    """A bare frame (unit tests, single-file callers) must not raise."""
    dbh, asw = _power_law()
    df = pd.DataFrame({SPECIES: "Pinus sylvestris", DBH: dbh, ASW: asw})
    assert not estimated_mask(df, SITE, SPECIES, DBH, ASW).any()


@pytest.mark.unit
def test_constant_dbh_group_is_not_assessed() -> None:
    """No DBH spread means no line to be suspiciously close to."""
    df = _frame("X", [30.0] * 6, [200.0, 210.0, 190.0, 205.0, 195.0, 201.0])
    verdicts = assess(df, SITE, SPECIES, DBH, ASW)
    assert verdicts[0].verdict == NOT_ASSESSED


@pytest.mark.unit
def test_summarise_reports_thresholds_and_sites() -> None:
    d1, a1 = _power_law(n=8)
    d2, a2 = _power_law(n=8, sigma=0.2, seed=2)
    df = pd.concat(
        [_frame("DERIVED_SITE", d1, a1), _frame("REAL_SITE", d2, a2, "Quercus ilex")],
        ignore_index=True,
    )
    report = summarise(assess(df, SITE, SPECIES, DBH, ASW))
    assert report["by_verdict"][ALLOMETRY_DERIVED]["trees"] == 8
    assert report["by_verdict"][MEASURED]["trees"] == 8
    assert report["estimated_sites"][ALLOMETRY_DERIVED] == ["DERIVED_SITE"]
    # The artifact states the rule it applied, so a reader need not open the code.
    assert report["thresholds"]["min_group_size"] == MIN_GROUP_SIZE
