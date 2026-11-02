"""Tests for the per-site IGBP land-cover resolution."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.site_info.pft_resolution import (  # noqa: E402
    IGBP_BY_LC_TYPE1,
    LandcoverVocabularyError,
    igbp_code,
    is_absent,
    load_site_landcover_modes,
    modal_class,
    resolve_pft,
    site_landcover_modes,
    site_year_classes,
    summarise,
)


def hourly(spec):
    """Build an hourly-expanded landcover frame from {year: (class, n_rows)}."""
    rows = []
    for site, per_year in spec.items():
        for year, (cls, n) in per_year.items():
            stamps = pd.date_range(f"{year}-01-01", periods=n, freq="h")
            for stamp in stamps:
                rows.append({"site_name": site, "timestamp": stamp, "landcover_type": cls})
    return pd.DataFrame(rows)


def annual(spec):
    """Build a one-row-per-site-year frame from {site: {year: class}}."""
    rows = [
        {"site_name": site, "year": year, "landcover_type": cls}
        for site, per_year in spec.items()
        for year, cls in per_year.items()
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- the mode


def test_mode_weights_years_not_rows():
    """A single long year must not outvote two short ones.

    This is the property the reduction exists for.  Grouping the hourly product
    row-wise would hand this site class 5 on 8000 rows against class 1 on 400,
    when class 1 holds two of the three years.
    """
    frame = hourly({"SITE_A": {2001: (5, 8000), 2002: (1, 200), 2003: (1, 200)}})
    assert site_landcover_modes(frame) == {"SITE_A": IGBP_BY_LC_TYPE1[1]}


def test_constant_site_returns_that_class():
    frame = annual({"SITE_A": {2001: 4, 2002: 4, 2003: 4}})
    assert site_landcover_modes(frame) == {"SITE_A": "DBF"}


def test_single_year_site():
    frame = annual({"SITE_A": {2015: 2}})
    assert site_landcover_modes(frame) == {"SITE_A": "EBF"}


def test_tie_breaks_to_the_more_recent_year():
    assert modal_class({2001: 9, 2002: 8}) == 8
    assert modal_class({2001: 8, 2002: 9}) == 9
    # three-way tie: still the most recent
    assert modal_class({2001: 1, 2002: 4, 2003: 5}) == 5


def test_tie_break_is_by_the_tied_class_latest_year_not_the_global_latest():
    """Class 1 holds 2001+2005, class 8 holds 2002+2003 -- 1 wins on recency."""
    assert modal_class({2001: 1, 2002: 8, 2003: 8, 2005: 1}) == 1


def test_majority_beats_recency():
    assert modal_class({2001: 1, 2002: 1, 2003: 8}) == 1


def test_modal_class_of_nothing_is_none():
    assert modal_class({}) is None


def test_davos_pattern_recovers_evergreen_needleleaf():
    """The real CHE_DAV series measured on the corpus: 9 years of 1, 6 of 8."""
    per_year = {
        2010: 1,
        2011: 1,
        2012: 1,
        2013: 1,
        2014: 1,
        2015: 1,
        2016: 8,
        2017: 8,
        2018: 1,
        2019: 8,
        2020: 8,
        2021: 8,
        2022: 1,
        2023: 8,
        2024: 1,
    }
    assert modal_class(per_year) == 1
    assert IGBP_BY_LC_TYPE1[modal_class(per_year)] == "ENF"


# ------------------------------------------------------- input shapes


def test_both_input_shapes_agree():
    spec = {"SITE_A": {2001: 5, 2002: 1, 2003: 1}}
    from_annual = site_landcover_modes(annual(spec))
    from_hourly = site_landcover_modes(hourly({"SITE_A": {2001: (5, 3), 2002: (1, 3), 2003: (1, 3)}}))
    assert from_annual == from_hourly == {"SITE_A": "ENF"}


def test_missing_time_column_is_an_error():
    frame = pd.DataFrame({"site_name": ["A"], "landcover_type": [1]})
    with pytest.raises(KeyError):
        site_landcover_modes(frame)


def test_site_year_classes_reports_the_year_map():
    frame = annual({"SITE_A": {2001: 5, 2002: 1}})
    assert site_year_classes(frame) == {"SITE_A": {2001: 5, 2002: 1}}


def test_rows_with_no_class_are_dropped_not_counted():
    frame = pd.DataFrame(
        {
            "site_name": ["A", "A", "A"],
            "year": [2001, 2002, 2003],
            "landcover_type": [4, 4, None],
        }
    )
    assert site_year_classes(frame) == {"A": {2001: 4, 2002: 4}}
    assert site_landcover_modes(frame) == {"A": "DBF"}


def test_site_with_no_usable_rows_is_absent_from_the_result():
    frame = pd.DataFrame({"site_name": ["A", "B"], "year": [2001, 2002], "landcover_type": [4, None]})
    assert site_landcover_modes(frame) == {"A": "DBF"}


# ------------------------------------------------------- vocabulary


def test_every_published_igbp_code_maps():
    assert sorted(IGBP_BY_LC_TYPE1) == list(range(1, 18))
    assert len(set(IGBP_BY_LC_TYPE1.values())) == 17


def test_unmapped_landcover_code_raises_rather_than_becoming_null():
    frame = annual({"SITE_A": {2001: 42}})
    with pytest.raises(LandcoverVocabularyError) as exc:
        site_landcover_modes(frame)
    assert "42" in str(exc.value)
    assert "SITE_A" in str(exc.value)


def test_zero_is_not_a_valid_igbp_class():
    frame = annual({"SITE_A": {2001: 0}})
    with pytest.raises(LandcoverVocabularyError):
        site_landcover_modes(frame)


# ------------------------------------------------------- absence


@pytest.mark.parametrize(
    "value",
    [None, float("nan"), "", "  ", "NA", "na", "n/a", "None", "null", "-9999", "unknown"],
)
def test_absent_values(value):
    assert is_absent(value)


@pytest.mark.parametrize("value", ["ENF", "DBF", " MF ", "WET"])
def test_present_values(value):
    assert not is_absent(value)


# ------------------------------------------------------- resolution


def test_provider_value_wins():
    assert resolve_pft("DBF", "ENF") == ("DBF", "provider")


def test_provider_value_is_stripped():
    assert resolve_pft("  DBF  ", None) == ("DBF", "provider")


@pytest.mark.parametrize("missing", [float("nan"), None, "", "NA"])
def test_fallback_used_when_provider_missing(missing):
    assert resolve_pft(missing, "ENF") == ("ENF", "modis")


def test_unresolved_when_both_missing():
    assert resolve_pft(float("nan"), None) == (None, "unresolved")


def test_unresolved_when_fallback_is_itself_absent():
    assert resolve_pft(float("nan"), "NA") == (None, "unresolved")


def test_summarise_names_the_borrowed_and_unresolved_sites():
    text = summarise({"A": "provider", "B": "modis", "C": "unresolved"})
    assert "1 provider" in text
    assert "1 modis" in text
    assert "1 unresolved" in text
    assert "'B'" in text
    assert "'C'" in text


# ------------------------------------------------------- file loading


def test_load_reads_only_the_needed_columns(tmp_path):
    path = tmp_path / "landcover_output.csv"
    frame = hourly({"SITE_A": {2001: (5, 4), 2002: (1, 4), 2003: (1, 4)}})
    frame["lon"] = 7.0
    frame["lat"] = 51.0
    frame["a_column_nobody_wants"] = "x" * 40
    frame.to_csv(path, index=False)
    assert load_site_landcover_modes(path) == {"SITE_A": "ENF"}


def test_load_accepts_the_annual_shape(tmp_path):
    path = tmp_path / "landcover_output.csv"
    annual({"SITE_A": {2001: 4}, "SITE_B": {2001: 1, 2002: 1}}).to_csv(path, index=False)
    assert load_site_landcover_modes(path) == {"SITE_A": "DBF", "SITE_B": "ENF"}


def test_load_rejects_a_file_without_a_site_column(tmp_path):
    path = tmp_path / "landcover_output.csv"
    pd.DataFrame({"year": [2001], "landcover_type": [4]}).to_csv(path, index=False)
    with pytest.raises(KeyError):
        load_site_landcover_modes(path)


def test_load_rejects_a_file_without_a_class_column(tmp_path):
    path = tmp_path / "landcover_output.csv"
    pd.DataFrame({"site_name": ["A"], "year": [2001]}).to_csv(path, index=False)
    with pytest.raises(KeyError):
        load_site_landcover_modes(path)


# ------------------------------------------- naming at the generator
#
# The extracted product used to hold bare LC_Type1 integers, and the only place
# that turned them into words was a long-name map in the merge that was never
# joined to anything.  The name is now written at extraction, in si_igbp's own
# short vocabulary.  These tests pin the mapping and the reader that prefers it.


def annual_with_code(spec):
    """One row per site-year carrying BOTH the raw integer and its short code."""
    rows = [
        {
            "site_name": site,
            "year": year,
            "landcover_type": cls,
            "pft": IGBP_BY_LC_TYPE1[cls],
        }
        for site, per_year in spec.items()
        for year, cls in per_year.items()
    ]
    return pd.DataFrame(rows)


def test_every_published_lc_type1_maps_to_a_short_code():
    for value in range(1, 18):
        assert igbp_code(value) == IGBP_BY_LC_TYPE1[value]


def test_the_eight_provider_classes_are_all_reachable():
    """si_igbp's measured vocabulary over the 226-site corpus."""
    provider_vocabulary = {"ENF", "EBF", "DNF", "DBF", "MF", "WSA", "SAV", "WET"}
    assert provider_vocabulary <= set(IGBP_BY_LC_TYPE1.values())


def test_igbp_code_accepts_the_float_earth_engine_hands_back():
    assert igbp_code(4.0) == "DBF"


def test_igbp_code_rejects_a_value_outside_the_scheme():
    with pytest.raises(LandcoverVocabularyError):
        igbp_code(18)


def test_igbp_code_rejects_none():
    """A point over masked water reduces to None; a blank name must not be written."""
    with pytest.raises(LandcoverVocabularyError):
        igbp_code(None)


def test_igbp_code_rejects_nan():
    with pytest.raises(LandcoverVocabularyError):
        igbp_code(float("nan"))


def test_igbp_code_never_returns_a_long_name():
    """The regression this whole change exists to prevent."""
    for value in range(1, 18):
        code = igbp_code(value)
        assert " " not in code
        assert code.isupper()
        assert len(code) <= 4


def test_a_stored_code_column_is_used(tmp_path):
    path = tmp_path / "landcover_output.csv"
    annual_with_code({"SITE_A": {2001: 4, 2002: 4, 2003: 1}}).to_csv(path, index=False)
    assert load_site_landcover_modes(path) == {"SITE_A": "DBF"}


def test_the_legacy_integer_only_product_still_loads(tmp_path):
    """The artifact currently on disk has no pft column; it must keep working."""
    path = tmp_path / "landcover_output.csv"
    annual({"SITE_A": {2001: 4}, "SITE_B": {2001: 1, 2002: 1}}).to_csv(path, index=False)
    assert load_site_landcover_modes(path) == {"SITE_A": "DBF", "SITE_B": "ENF"}


def test_a_product_carrying_only_the_code_column_loads(tmp_path):
    path = tmp_path / "landcover_output.csv"
    pd.DataFrame({"site_name": ["S"], "year": [2001], "pft": ["ENF"]}).to_csv(path, index=False)
    assert load_site_landcover_modes(path) == {"S": "ENF"}


def test_the_stored_code_takes_the_mode_across_years(tmp_path):
    path = tmp_path / "landcover_output.csv"
    annual_with_code({"S": {2001: 1, 2002: 1, 2003: 8}}).to_csv(path, index=False)
    assert load_site_landcover_modes(path) == {"S": "ENF"}


def test_a_code_that_contradicts_its_raw_integer_raises(tmp_path):
    """The two columns are redundant on purpose -- so drift is detectable."""
    path = tmp_path / "landcover_output.csv"
    frame = annual_with_code({"SITE_A": {2001: 4}})
    frame.loc[0, "pft"] = "ENF"  # the raw value 4 is DBF
    frame.to_csv(path, index=False)
    with pytest.raises(LandcoverVocabularyError):
        load_site_landcover_modes(path)


def test_a_long_name_in_the_code_column_raises(tmp_path):
    path = tmp_path / "landcover_output.csv"
    frame = pd.DataFrame(
        {"site_name": ["S"], "year": [2001], "pft": ["Deciduous Broadleaf Forests"]}
    )
    frame.to_csv(path, index=False)
    with pytest.raises(LandcoverVocabularyError):
        load_site_landcover_modes(path)


def test_positive_control_the_code_column_really_is_what_is_read(tmp_path):
    """Ignoring `pft` and mapping the integer would answer DBF, not WET."""
    path = tmp_path / "landcover_output.csv"
    frame = pd.DataFrame({"site_name": ["S"], "year": [2001], "pft": ["WET"]})
    frame.to_csv(path, index=False)
    assert load_site_landcover_modes(path) == {"S": "WET"}
    with_raw = pd.DataFrame({"site_name": ["S"], "year": [2001], "landcover_type": [4]})
    with_raw.to_csv(path, index=False)
    assert load_site_landcover_modes(path) == {"S": "DBF"}


def test_an_absent_code_row_is_dropped_not_guessed(tmp_path):
    path = tmp_path / "landcover_output.csv"
    frame = pd.DataFrame(
        {"site_name": ["S", "S"], "year": [2001, 2002], "pft": ["ENF", ""]}
    )
    frame.to_csv(path, index=False)
    assert load_site_landcover_modes(path) == {"S": "ENF"}
