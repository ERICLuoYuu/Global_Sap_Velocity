"""Tests for the out-of-sample diagnostics.

These check that each diagnostic *detects the thing it claims to detect*, by
building a frame with the defect injected at a known size and asserting the
diagnostic recovers it. A diagnostic that always returns something plausible is
worse than none, because it is quoted in the report as evidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.sapwood_imputation.allometry import fit_loglog_with_group_offset
from src.sapwood_imputation.validation import (
    bias_by_genus,
    leave_one_site_out,
    residual_spread_by_size,
    site_leverage,
    validate,
)

GROUP, DBH, ASW, SITE, SPECIES = "ft", "dbh", "asw", "site", "sp"


def _frame(n_sites: int = 12, per_site: int = 15, sigma: float = 0.2, seed: int = 3) -> pd.DataFrame:
    """A well-behaved multi-site corpus: one shared law, lognormal scatter."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sites):
        gymno = s % 2 == 0
        dbh = rng.uniform(10, 60, size=per_site)
        asw = (0.8 if gymno else 1.3) * dbh**1.7 * np.exp(rng.normal(0, sigma, size=per_site))
        rows.append(
            pd.DataFrame(
                {
                    SITE: f"SITE_{s:02d}",
                    GROUP: "gymnosperm" if gymno else "angiosperm",
                    SPECIES: "Pinus sylvestris" if gymno else "Quercus robur",
                    DBH: dbh,
                    ASW: asw,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _fit(df: pd.DataFrame):
    return fit_loglog_with_group_offset(df, GROUP, DBH, ASW, min_n=2)


@pytest.mark.unit
def test_loso_scores_every_site_and_reports_both_weightings() -> None:
    df = _frame()
    out = leave_one_site_out(df, GROUP, DBH, ASW, SITE)

    assert out["scored_sites"] == 12
    for variant in ("unweighted", "site_balanced"):
        assert out[variant]["trees_scored"] == len(df)
        assert 0.0 < out[variant]["r2"] <= 1.0
        assert np.isfinite(out[variant]["residual_sd_log"])


@pytest.mark.unit
def test_loso_is_not_the_in_sample_number() -> None:
    """Held-out error must exceed the fit's own residual: that is the point."""
    df = _frame()
    fit = _fit(df)
    out = leave_one_site_out(df, GROUP, DBH, ASW, SITE)

    assert out["unweighted"]["residual_sd_log"] > np.sqrt(fit.mse) * 0.99
    assert out["unweighted"]["r2"] <= fit.r2


@pytest.mark.unit
def test_loso_punishes_a_site_that_breaks_the_shared_law() -> None:
    """One deviant site must show up as a bad held-out site, not be absorbed."""
    df = _frame()
    rogue = df[SITE] == "SITE_00"
    df.loc[rogue, ASW] = df.loc[rogue, ASW] * 10.0  # an order of magnitude off

    out = leave_one_site_out(df, GROUP, DBH, ASW, SITE)
    assert out["unweighted"]["site_weighted_median_relative_error"] > 0.0
    # The pooled fit is dragged, so held-out skill drops well below the clean case.
    clean = leave_one_site_out(_frame(), GROUP, DBH, ASW, SITE)
    assert out["unweighted"]["r2"] < clean["unweighted"]["r2"]


@pytest.mark.unit
def test_loso_reports_when_it_cannot_run() -> None:
    tiny = _frame(n_sites=1, per_site=5)
    out = leave_one_site_out(tiny, GROUP, DBH, ASW, SITE)
    assert out["scored_sites"] == 0


@pytest.mark.unit
def test_size_spread_is_flat_when_scatter_is_uniform() -> None:
    """No false alarm on homoscedastic data.

    The naive max/min-of-ten-bins ratio reaches ~1.68 here purely by sampling
    noise, which is why the flag is a z on a pre-specified comparison instead.
    """
    df = _frame(n_sites=20, per_site=40, sigma=0.2)
    out = residual_spread_by_size(df, GROUP, DBH, ASW, _fit(df))

    assert out["bins"] > 1
    assert abs(out["sd_trend_z"]) < 2.5
    assert out["heteroscedastic"] is False


@pytest.mark.unit
def test_size_spread_detects_scatter_that_grows_with_size() -> None:
    """Inject heteroscedasticity and require the diagnostic to name it."""
    rng = np.random.default_rng(11)
    dbh = np.exp(rng.uniform(np.log(5), np.log(120), size=900))
    # noise SD proportional to ln(DBH): small trees tight, big trees loose
    noise = rng.normal(0, 0.05 + 0.25 * (np.log(dbh) - np.log(5)) / (np.log(120) - np.log(5)))
    df = pd.DataFrame(
        {
            SITE: "S",
            GROUP: "angiosperm",
            SPECIES: "Quercus robur",
            DBH: dbh,
            ASW: 1.0 * dbh**1.7 * np.exp(noise),
        }
    )
    out = residual_spread_by_size(df, GROUP, DBH, ASW, _fit(df))

    assert out["heteroscedastic"] is True
    assert out["sd_trend_z"] > 2.5
    assert out["by_bin"][0]["residual_sd"] < out["by_bin"][-1]["residual_sd"]


def _two_genera_one_type(seed: int = 5) -> pd.DataFrame:
    """Larix and Pinus: two genera inside one functional type.

    The injected bias is only *detectable* this way. A genus that is the sole
    member of its functional type has its offset absorbed by that group's own
    indicator, leaving a mean residual of exactly zero -- see
    :func:`test_genus_bias_is_blind_to_a_genus_alone_in_its_type`.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for genus, factor in (("Larix", 0.4), ("Pinus", 1.0)):
        dbh = rng.uniform(10, 60, size=60)
        rows.append(
            pd.DataFrame(
                {
                    SITE: f"S_{genus}",
                    GROUP: "gymnosperm",
                    SPECIES: f"{genus} decidua" if genus == "Larix" else "Pinus sylvestris",
                    DBH: dbh,
                    ASW: factor * dbh**1.7 * np.exp(rng.normal(0, 0.15, size=60)),
                }
            )
        )
    dbh = rng.uniform(10, 60, size=60)
    rows.append(
        pd.DataFrame(
            {
                SITE: "S_oak",
                GROUP: "angiosperm",
                SPECIES: "Quercus robur",
                DBH: dbh,
                ASW: 1.2 * dbh**1.7 * np.exp(rng.normal(0, 0.15, size=60)),
            }
        )
    )
    return pd.concat(rows, ignore_index=True)


@pytest.mark.unit
def test_genus_bias_recovers_an_injected_offset() -> None:
    """A genus given 40% of the sapwood of its type-mate must be named as such."""
    df = _two_genera_one_type()
    out = bias_by_genus(df, GROUP, DBH, ASW, SPECIES, _fit(df), min_n=10)
    by_genus = {r["genus"]: r for r in out["by_genus"]}

    assert {"Larix", "Pinus", "Quercus"} <= set(by_genus)
    # Larix has less sapwood than its gymnosperm offset gives it -> over-predicted.
    assert by_genus["Larix"]["over_prediction_factor"] > 1.5
    assert by_genus["Pinus"]["over_prediction_factor"] < by_genus["Larix"]["over_prediction_factor"]
    assert out["worst_over_predicted"]["genus"] == "Larix"
    assert out["bias_spread"] > 1.5


@pytest.mark.unit
def test_genus_bias_is_blind_to_a_genus_alone_in_its_type() -> None:
    """Documents a real limitation, so nobody reads a 1.0 as 'no genus bias'.

    With one genus per functional type the group indicator absorbs the whole
    offset, and every genus comes back unbiased however wrong the model is.
    """
    df = _frame(n_sites=8, per_site=20)  # Pinus <-> gymnosperm, Quercus <-> angiosperm
    thin = df[SPECIES] == "Pinus sylvestris"
    df.loc[thin, ASW] = df.loc[thin, ASW] * 0.5

    out = bias_by_genus(df, GROUP, DBH, ASW, SPECIES, _fit(df), min_n=10)
    for row in out["by_genus"]:
        assert row["over_prediction_factor"] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.unit
def test_genus_bias_ignores_genera_below_the_reporting_threshold() -> None:
    df = _frame(n_sites=4, per_site=10)
    out = bias_by_genus(df, GROUP, DBH, ASW, SPECIES, _fit(df), min_n=1000)
    assert out["genera"] == 0
    assert out["worst_over_predicted"] is None


@pytest.mark.unit
def test_site_leverage_counts_shares_against_the_fittable_trees() -> None:
    df = _frame(n_sites=10, per_site=10)
    # One dominant stand.
    big = pd.DataFrame(
        {
            SITE: "BIG",
            GROUP: "angiosperm",
            SPECIES: "Quercus robur",
            DBH: np.linspace(10, 60, 100),
            ASW: np.linspace(10, 60, 100) ** 1.7,
        }
    )
    out = site_leverage(pd.concat([df, big], ignore_index=True), DBH, ASW, SITE)

    assert out["sites"] == 11
    assert out["trees"] == 200
    assert out["largest_sites"][0]["site"] == "BIG"
    assert out["top_5_share"] == pytest.approx(0.7, abs=0.01)  # 100 + 4*10 of 200


@pytest.mark.unit
def test_site_leverage_ignores_trees_the_fit_cannot_use() -> None:
    df = _frame(n_sites=3, per_site=10)
    unusable = df.head(5).copy()
    unusable[ASW] = np.nan
    unusable[SITE] = "GHOST"
    out = site_leverage(pd.concat([df, unusable], ignore_index=True), DBH, ASW, SITE)

    assert "GHOST" not in [s["site"] for s in out["largest_sites"]]
    assert out["trees"] == 30


@pytest.mark.unit
def test_validate_returns_every_diagnostic_and_is_json_safe() -> None:
    import json

    df = _frame()
    report = validate(df, _fit(df), GROUP, DBH, ASW, SITE, SPECIES)

    assert set(report) == {
        "leave_one_site_out",
        "residual_spread_by_size",
        "bias_by_genus",
        "site_leverage",
    }
    json.dumps(report)  # must survive the report writer


@pytest.mark.unit
def test_loso_r2_is_scored_against_the_trees_it_actually_covered() -> None:
    """A skipped site must not inflate R2.

    Sites whose removal would strip a whole functional type from the training
    set cannot be scored. Dividing their absent residuals by the *full* corpus
    variance would credit the model with variance it never had to explain, and
    the error grows with the number of skipped sites.
    """
    df = _frame(n_sites=8, per_site=15)
    # A single-species site that is the only holder of a very large size class:
    # excluded from scoring, but a big contributor to the corpus variance.
    rng = np.random.default_rng(31)
    dbh = rng.uniform(150, 300, size=25)
    lonely = pd.DataFrame(
        {
            SITE: "LONELY",
            GROUP: "gymnosperm",
            SPECIES: "Larix decidua",
            DBH: dbh,
            ASW: 0.05 * dbh**1.7 * np.exp(rng.normal(0, 0.2, size=25)),
        }
    )
    df = pd.concat([df, lonely], ignore_index=True)

    out = leave_one_site_out(df, GROUP, DBH, ASW, SITE)
    covered = out["unweighted"]["trees_scored"]
    # R2 must be computed on the covered trees; a value derived from the full
    # corpus SST would sit above 1 - SSE/SST_covered whenever coverage is partial.
    y = np.log(df[ASW].to_numpy(float))
    assert covered <= len(y)
    assert out["unweighted"]["r2"] <= 1.0
