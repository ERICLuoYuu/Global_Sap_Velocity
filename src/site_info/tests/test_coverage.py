"""Tests for src/site_info/coverage.py.

The allowlist is the one place where this pipeline is allowed to say "missing
is fine".  If it is too permissive it hides the exact failure the coverage
check was built to catch, so its behaviour is pinned down here.
"""

from __future__ import annotations

from src.site_info.coverage import (
    KNOWN_ABSENT,
    classify_missing,
)

# --------------------------------------------------------------------------
# the three outcomes
# --------------------------------------------------------------------------


def test_a_listed_site_is_explained_and_carries_its_reason():
    report = classify_missing("globmap_lai", ["AUT_MMG"])

    assert report.ok
    assert report.problem() is None
    assert [site for site, _ in report.explained] == ["AUT_MMG"]
    assert "2020-12-26" in report.explained[0][1]


def test_an_unlisted_site_is_a_failure():
    """The whole point: a silent extraction failure must not pass."""
    report = classify_missing("globmap_lai", ["ARG_MAZ"])

    assert not report.ok
    assert report.unexplained == ("ARG_MAZ",)
    assert "unexplained" in report.problem()


def test_a_listed_site_that_is_actually_present_is_a_failure():
    """A stale entry would silently excuse a real regression later."""
    report = classify_missing("globmap_lai", missing=[], present=["AUT_MMG"])

    assert not report.ok
    assert report.stale == ("AUT_MMG",)
    assert "stale" in report.problem()


def test_explained_and_unexplained_are_reported_together():
    report = classify_missing("globmap_lai", ["AUT_MMG", "ARG_MAZ", "DEU_HAI"])

    assert not report.ok
    assert [site for site, _ in report.explained] == ["AUT_MMG", "DEU_HAI"]
    assert report.unexplained == ("ARG_MAZ",)


def test_nothing_missing_is_clean():
    report = classify_missing("globmap_lai", [])

    assert report.ok
    assert report.explained == ()
    assert report.lines() == []


# --------------------------------------------------------------------------
# defaults that must fail safe
# --------------------------------------------------------------------------


def test_a_product_with_no_allowlist_excuses_nothing():
    """era5 has no entry, so every absence there is a genuine failure."""
    report = classify_missing("era5", ["ARG_MAZ", "AUS_KAR"])

    assert not report.ok
    assert report.unexplained == ("ARG_MAZ", "AUS_KAR")
    assert report.explained == ()


def test_an_unknown_product_name_excuses_nothing():
    report = classify_missing("not_a_product", ["AUT_MMG"])

    assert not report.ok
    assert report.unexplained == ("AUT_MMG",)


def test_present_defaults_to_empty_so_omitting_it_cannot_excuse_anything():
    """Callers that do not pass `present` must not get a free pass."""
    report = classify_missing("globmap_lai", ["AUT_MMG"])

    assert report.stale == ()
    assert report.ok


# --------------------------------------------------------------------------
# the allowlist's own content
# --------------------------------------------------------------------------


def test_only_globmap_lai_is_excused_at_all():
    """A new entry here is a claim about data availability -- make it visible."""
    assert set(KNOWN_ABSENT) == {"globmap_lai"}


def test_the_four_excused_sites_are_exactly_the_expected_ones():
    assert set(KNOWN_ABSENT["globmap_lai"]) == {
        "AUT_MMG",
        "DEU_HAI",
        "DEU_HAR",
        "NOR_HUR",
    }


def test_every_reason_names_the_source_and_its_end_date():
    """A reason like "no data" would be useless to whoever reads this next."""
    for site, reason in KNOWN_ABSENT["globmap_lai"].items():
        assert "GlobMap LAI" in reason, site
        assert "2020-12-26" in reason, site
        assert "starts" in reason, site


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def test_lines_are_empty_when_there_is_nothing_to_say():
    assert classify_missing("globmap_lai", []).lines() == []


def test_lines_mention_every_finding():
    report = classify_missing("globmap_lai", missing=["AUT_MMG", "ARG_MAZ"], present=["DEU_HAI"])
    text = " ".join(report.lines())

    assert "AUT_MMG" in text
    assert "ARG_MAZ" in text
    assert "DEU_HAI" in text


def test_duplicates_in_the_input_do_not_duplicate_the_output():
    report = classify_missing("globmap_lai", ["AUT_MMG", "AUT_MMG"])

    assert len(report.explained) == 1
