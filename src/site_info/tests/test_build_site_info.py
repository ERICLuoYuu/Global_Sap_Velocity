"""Tests for src/site_info/build_site_info.py.

The tests that matter here assert things the module does not itself decide:
that two timestamp *spellings* denote the same instant, that a prefix-colliding
site code is not absorbed by its longer sibling, and that a coordinate outside
the domain of latitude is reported.  Reading back a value the code just wrote
would prove nothing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from src.site_info.build_site_info import (
    SITE_INFO_COLUMNS,
    SiteInfoError,
    build_site_info,
    discover_sites,
    read_coordinates,
    read_time_span,
    validate,
    write_site_info,
)


def make_site(
    directory: Path,
    site: str,
    *,
    timestamps: list[str],
    lat: float = 48.0,
    lon: float = 7.0,
) -> None:
    """Write the two files build_site_info needs for one site."""
    directory.mkdir(parents=True, exist_ok=True)
    sapf = pd.DataFrame(
        {
            "TIMESTAMP": timestamps,
            "solar_TIMESTAMP": timestamps,
            f"{site}_Qpe_Js_1": [1.0] * len(timestamps),
        }
    )
    sapf.to_csv(directory / f"{site}_sapf_data.csv", index=False)
    pd.DataFrame([{"si_code": site, "si_lat": lat, "si_long": lon, "si_elev": 300}]).to_csv(
        directory / f"{site}_site_md.csv", index=False
    )


AWARE = ["2015-04-22T22:00:00Z", "2015-04-23T00:00:00Z"]
NAIVE = ["2015-04-22 22:00:00", "2015-04-23 00:00:00"]


# --------------------------------------------------------------------------
# discover_sites
# --------------------------------------------------------------------------


def test_discovers_every_site_with_sap_flow(tmp_path):
    make_site(tmp_path, "ARG_MAZ", timestamps=AWARE)
    make_site(tmp_path, "DEU_HOH", timestamps=NAIVE)

    assert discover_sites(tmp_path) == ["ARG_MAZ", "DEU_HOH"]


def test_a_prefix_of_another_site_is_its_own_site(tmp_path):
    """CHE_DAV must not be swallowed by CHE_DAV_SEE, nor invent a third code.

    The combined corpus is one flat directory holding both source corpora, so
    prefix collisions exist there that exist in neither corpus alone.
    """
    make_site(tmp_path, "CHE_DAV", timestamps=AWARE)
    make_site(tmp_path, "CHE_DAV_SEE", timestamps=AWARE)
    make_site(tmp_path, "SWE_NOR", timestamps=AWARE)
    make_site(tmp_path, "SWE_NOR_ST1_AF1", timestamps=AWARE)

    assert discover_sites(tmp_path) == [
        "CHE_DAV",
        "CHE_DAV_SEE",
        "SWE_NOR",
        "SWE_NOR_ST1_AF1",
    ]


def test_metadata_only_directory_entries_are_not_sites(tmp_path):
    make_site(tmp_path, "ARG_MAZ", timestamps=AWARE)
    (tmp_path / "ARG_MAZ_env_data.csv").write_text("TIMESTAMP\n2015-01-01\n")
    (tmp_path / "MANIFEST.json").write_text("{}")
    (tmp_path / "site_info.csv").write_text("site_name\n")

    assert discover_sites(tmp_path) == ["ARG_MAZ"]


def test_empty_directory_raises_rather_than_yielding_nothing(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SiteInfoError, match="no \\*_sapf_data.csv"):
        discover_sites(tmp_path / "empty")


def test_absent_directory_raises(tmp_path):
    with pytest.raises(SiteInfoError, match="does not exist"):
        discover_sites(tmp_path / "no_such_scale")


# --------------------------------------------------------------------------
# read_time_span — the corpus's two timestamp dialects
# --------------------------------------------------------------------------


def test_zulu_and_naive_spellings_denote_the_same_instant(tmp_path):
    """The claim that makes utc=True necessary, stated against absolute time.

    '2015-04-22T22:00:00Z' and '2015-04-22 22:00:00' are the same moment,
    because the naive form is UTC by the corpus's contract.  If the naive form
    were ever localised to anything else, these two spans would differ.
    """
    make_site(tmp_path, "AWARE_SITE", timestamps=AWARE)
    make_site(tmp_path, "NAIVE_SITE", timestamps=NAIVE)

    aware_start, aware_end = read_time_span(tmp_path, "AWARE_SITE")
    naive_start, naive_end = read_time_span(tmp_path, "NAIVE_SITE")

    assert aware_start == naive_start == pd.Timestamp("2015-04-22 22:00:00+00:00")
    assert aware_end == naive_end == pd.Timestamp("2015-04-23 00:00:00+00:00")


def test_both_dialects_come_back_timezone_aware(tmp_path):
    make_site(tmp_path, "NAIVE_SITE", timestamps=NAIVE)
    start, end = read_time_span(tmp_path, "NAIVE_SITE")
    assert start.tzinfo is not None
    assert end.tzinfo is not None


def test_span_is_min_and_max_not_first_and_last(tmp_path):
    make_site(
        tmp_path,
        "SHUFFLED",
        timestamps=[
            "2015-06-01T12:00:00Z",
            "2015-01-01T00:00:00Z",
            "2015-12-31T23:00:00Z",
            "2015-03-01T06:00:00Z",
        ],
    )
    start, end = read_time_span(tmp_path, "SHUFFLED")
    assert start == pd.Timestamp("2015-01-01 00:00:00+00:00")
    assert end == pd.Timestamp("2015-12-31 23:00:00+00:00")


def test_unparseable_timestamp_raises_rather_than_becoming_nat(tmp_path):
    make_site(tmp_path, "BROKEN", timestamps=["2015-04-22T22:00:00Z", "not-a-date"])
    with pytest.raises(SiteInfoError, match="did not parse"):
        read_time_span(tmp_path, "BROKEN")


def test_missing_timestamp_column_names_the_columns_it_did_find(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp": ["2015-01-01"], "value": [1.0]}).to_csv(tmp_path / "LOWER_sapf_data.csv", index=False)
    with pytest.raises(SiteInfoError, match="no 'TIMESTAMP' column"):
        read_time_span(tmp_path, "LOWER")


def test_header_only_sap_file_raises(tmp_path):
    (tmp_path / "HOLLOW_sapf_data.csv").write_text("TIMESTAMP,value\n")
    with pytest.raises(SiteInfoError, match="no rows"):
        read_time_span(tmp_path, "HOLLOW")


def test_absent_sap_file_raises(tmp_path):
    with pytest.raises(SiteInfoError, match="no sapf_data file"):
        read_time_span(tmp_path, "GHOST")


# --------------------------------------------------------------------------
# read_coordinates
# --------------------------------------------------------------------------


def test_reads_latitude_and_longitude_from_site_md(tmp_path):
    make_site(tmp_path, "ESP_GDN", timestamps=AWARE, lat=41.2937, lon=-3.2278)
    assert read_coordinates(tmp_path, "ESP_GDN") == (41.2937, -3.2278)


def test_coordinates_survive_a_byte_order_mark(tmp_path):
    """A provider's spreadsheet export can glue a BOM to the first header."""
    make_site(tmp_path, "BOMMED", timestamps=AWARE, lat=50.0, lon=6.0)
    path = tmp_path / "BOMMED_site_md.csv"
    path.write_text(path.read_text(), encoding="utf-8-sig")

    assert read_coordinates(tmp_path, "BOMMED") == (50.0, 6.0)


def test_non_numeric_coordinate_raises_and_shows_the_value(tmp_path):
    make_site(tmp_path, "TEXTY", timestamps=AWARE)
    pd.DataFrame([{"si_lat": "48.5 N", "si_long": 7.0}]).to_csv(tmp_path / "TEXTY_site_md.csv", index=False)
    with pytest.raises(SiteInfoError, match="48.5 N"):
        read_coordinates(tmp_path, "TEXTY")


def test_missing_coordinate_column_raises(tmp_path):
    make_site(tmp_path, "NOCOORD", timestamps=AWARE)
    pd.DataFrame([{"si_code": "NOCOORD", "si_elev": 300}]).to_csv(tmp_path / "NOCOORD_site_md.csv", index=False)
    with pytest.raises(SiteInfoError, match="si_lat"):
        read_coordinates(tmp_path, "NOCOORD")


def test_absent_site_md_raises(tmp_path):
    make_site(tmp_path, "ORPHAN", timestamps=AWARE)
    (tmp_path / "ORPHAN_site_md.csv").unlink()
    with pytest.raises(SiteInfoError, match="no site_md file"):
        read_coordinates(tmp_path, "ORPHAN")


# --------------------------------------------------------------------------
# build_site_info
# --------------------------------------------------------------------------


def test_column_order_matches_what_the_extractors_concatenate(tmp_path):
    """extract_canopyheight does pd.concat([df, results], axis=1) — order is API."""
    make_site(tmp_path, "ARG_MAZ", timestamps=AWARE)
    frame = build_site_info(tmp_path)
    assert list(frame.columns) == SITE_INFO_COLUMNS


def test_mixed_dialect_corpus_yields_one_datetime_dtype(tmp_path):
    """The regression guard: object dtype here means two formats in the CSV."""
    make_site(tmp_path, "ARG_MAZ", timestamps=AWARE)
    make_site(tmp_path, "DEU_HOH", timestamps=NAIVE)

    frame = build_site_info(tmp_path)

    assert str(frame["start_date"].dtype) == "datetime64[ns, UTC]"
    assert str(frame["end_date"].dtype) == "datetime64[ns, UTC]"


def test_written_file_reparses_to_the_same_instants(tmp_path):
    """A round trip through CSV must not shift or drop the zone."""
    make_site(tmp_path, "ARG_MAZ", timestamps=AWARE)
    make_site(tmp_path, "DEU_HOH", timestamps=NAIVE)
    frame = build_site_info(tmp_path)

    out = write_site_info(frame, tmp_path / "out" / "site_info.csv")
    reloaded = pd.read_csv(out)
    reparsed = pd.to_datetime(reloaded["start_date"], utc=True)

    assert list(reparsed) == list(frame["start_date"])
    # One dialect on disk, so a bulk parse cannot infer a format that NaTs rows.
    assert reparsed.notna().all()


def test_only_restricts_output_to_the_named_sites(tmp_path):
    make_site(tmp_path, "ARG_MAZ", timestamps=AWARE)
    make_site(tmp_path, "DEU_HOH", timestamps=NAIVE)
    make_site(tmp_path, "ITA_CP2", timestamps=NAIVE)

    frame = build_site_info(tmp_path, only=["DEU_HOH", "ITA_CP2"])

    assert sorted(frame["site_name"]) == ["DEU_HOH", "ITA_CP2"]


def test_requesting_an_absent_site_raises_instead_of_returning_fewer_rows(tmp_path):
    """A delta run that silently drops a site extracts nothing for it, forever."""
    make_site(tmp_path, "ARG_MAZ", timestamps=AWARE)
    with pytest.raises(SiteInfoError, match="NOT_A_SITE"):
        build_site_info(tmp_path, only=["ARG_MAZ", "NOT_A_SITE"])


def test_only_is_deduplicated(tmp_path):
    make_site(tmp_path, "ARG_MAZ", timestamps=AWARE)
    frame = build_site_info(tmp_path, only=["ARG_MAZ", "ARG_MAZ"])
    assert len(frame) == 1


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start_date": pd.to_datetime(["2015-01-01T00:00:00Z"], utc=True),
            "end_date": pd.to_datetime(["2015-12-31T00:00:00Z"], utc=True),
            "lat": [48.0],
            "lon": [7.0],
            "site_name": ["FRA_HES"],
        }
    )


def test_a_good_frame_has_no_problems():
    assert validate(valid_frame()) == []


def test_latitude_beyond_the_pole_is_reported():
    frame = valid_frame()
    frame.loc[0, "lat"] = 95.0
    assert any("lat outside" in p for p in validate(frame))


def test_longitude_beyond_the_antimeridian_is_reported():
    frame = valid_frame()
    frame.loc[0, "lon"] = 200.0
    assert any("lon outside" in p for p in validate(frame))


def test_null_island_is_reported():
    """(0, 0) is open ocean — it is what a failed coordinate lookup returns."""
    frame = valid_frame()
    frame.loc[0, "lat"] = 0.0
    frame.loc[0, "lon"] = 0.0
    assert any("(0, 0)" in p for p in validate(frame))


def test_a_site_on_the_equator_is_not_mistaken_for_null_island():
    frame = valid_frame()
    frame.loc[0, "lat"] = 0.0
    frame.loc[0, "lon"] = -51.43  # BRA_CAX_CON sits near the equator
    assert validate(frame) == []


def test_reversed_span_is_reported():
    frame = valid_frame()
    frame.loc[0, "start_date"] = pd.Timestamp("2016-01-01T00:00:00Z")
    assert any("start_date after end_date" in p for p in validate(frame))


def test_duplicate_site_names_are_reported():
    frame = pd.concat([valid_frame(), valid_frame()], ignore_index=True)
    assert any("duplicate site_name" in p for p in validate(frame))


def test_missing_columns_are_reported_without_crashing():
    assert validate(pd.DataFrame({"site_name": ["X"]})) == ["missing columns: ['start_date', 'end_date', 'lat', 'lon']"]
