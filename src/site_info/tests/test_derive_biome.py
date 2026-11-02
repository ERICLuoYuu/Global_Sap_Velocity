"""Tests for ``derive_biome``.

The load-bearing one is ``test_archive_regenerates_its_own_labels``: it feeds
the 202 archive sites their OWN stored si_mat/si_map and requires the parsed
polygons to return the stored si_biome.  That is a positive control on the
whole chain -- R parse, vertex order, ray casting, the Mediterranean rename --
against data this module never touches.  Without it the parser could be quietly
wrong and every unit test below would still pass.
"""

from __future__ import annotations

import csv
import os

import pytest

from src.site_info import derive_biome as db

CORPUS = "/scratch/tmp/yluo2/Global_Sap_Velocity/data/raw/0.1.5/0.1.5/csv/sapwood_combined"
R_SOURCE = "/scratch/tmp/yluo2/gsv/qc_coordinates_functions.R"

needs_corpus = pytest.mark.skipif(not os.path.isdir(CORPUS), reason="corpus not present")
needs_r_source = pytest.mark.skipif(not os.path.isfile(R_SOURCE), reason="R source not present")

SYNTHETIC_R = """
biomes_df <- data.frame(
    mat = c(0, 0, 10, 10,
            -10, -10, 0, 0),
    map = c(0, 100, 100, 0,
            200, 300, 300, 200),
    biome = c(rep('Mediterranean', 4),
              rep('Tundra', 4))
)
# STEP 2
"""

HEADER = ["si_code", "si_lat", "si_long", "si_remarks", "si_mat", "si_map", "si_biome"]


def _write_site(directory, site, **overrides):
    row = {name: "NA" for name in HEADER}
    row["si_code"] = site
    row.update(overrides)
    path = os.path.join(directory, site + db.SITE_SUFFIX)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerow(row)
    return path


def _read_site(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))[0]


@pytest.fixture()
def synthetic_r(tmp_path):
    path = tmp_path / "qc.R"
    path.write_text(SYNTHETIC_R, encoding="utf-8")
    return str(path)


# --------------------------------------------------------------- polygon parse


def test_parses_blocks_in_source_order(synthetic_r):
    polygons = db.parse_whittaker_polygons(synthetic_r)
    assert list(polygons) == ["Mediterranean", "Tundra"]  # nosec B101
    assert len(polygons["Mediterranean"]) == 4  # nosec B101
    assert polygons["Mediterranean"][0] == (0.0, 0.0)  # nosec B101
    assert polygons["Mediterranean"][1] == (100.0, 0.0)  # nosec B101


def test_parse_rejects_mismatched_vector_lengths(tmp_path):
    """A vertex-count mismatch must raise, not silently truncate a polygon."""
    bad = tmp_path / "bad.R"
    bad.write_text(
        "biomes_df <- data.frame(\n"
        "    mat = c(0, 0, 10),\n"
        "    map = c(0, 100, 100, 0),\n"
        "    biome = c(rep('Tundra', 4))\n)\n# STEP 2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        db.parse_whittaker_polygons(str(bad))


def test_parse_rejects_blocks_not_covering_vertices(tmp_path):
    bad = tmp_path / "bad.R"
    bad.write_text(
        "biomes_df <- data.frame(\n"
        "    mat = c(0, 0, 10, 10),\n"
        "    map = c(0, 100, 100, 0),\n"
        "    biome = c(rep('Tundra', 3))\n)\n# STEP 2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        db.parse_whittaker_polygons(str(bad))


@needs_r_source
def test_real_source_has_nine_polygons():
    polygons = db.parse_whittaker_polygons(R_SOURCE)
    assert len(polygons) == 9  # nosec B101
    assert sum(len(ring) for ring in polygons.values()) == 106  # nosec B101
    assert "Mediterranean" in polygons  # nosec B101


# ------------------------------------------------------------------- geometry


def test_point_in_ring_unit_square():
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert db.point_in_ring(0.5, 0.5, square)  # nosec B101
    assert not db.point_in_ring(1.5, 0.5, square)  # nosec B101
    assert not db.point_in_ring(0.5, 1.5, square)  # nosec B101
    assert not db.point_in_ring(-0.5, 0.5, square)  # nosec B101


def test_classify_applies_the_mediterranean_rename(synthetic_r):
    """The QC package says 'Mediterranean'; the released database does not."""
    polygons = db.parse_whittaker_polygons(synthetic_r)
    assert db.classify(50, 5, polygons) == "Woodland/Shrubland"  # nosec B101


def test_classify_returns_unrenamed_names_unchanged(synthetic_r):
    polygons = db.parse_whittaker_polygons(synthetic_r)
    assert db.classify(250, -5, polygons) == "Tundra"  # nosec B101


def test_classify_outside_every_polygon_is_none(synthetic_r):
    """CHN_ARG_GWD and GWS really do fall outside; None is a valid outcome."""
    polygons = db.parse_whittaker_polygons(synthetic_r)
    assert db.classify(5000, 40, polygons) is None  # nosec B101


def test_classify_rejects_unusable_numbers(synthetic_r):
    polygons = db.parse_whittaker_polygons(synthetic_r)
    assert db.classify(float("nan"), 5, polygons) is None  # nosec B101
    assert db.classify("", 5, polygons) is None  # nosec B101
    assert db.classify(None, 5, polygons) is None  # nosec B101


# ------------------------------------------------------- the positive control


@needs_corpus
@needs_r_source
def test_archive_regenerates_its_own_labels():
    """Stored si_mat/si_map must reproduce stored si_biome for the archive.

    200/202 is the known ceiling: CHN_ARG_GWD and CHN_ARG_GWS are hyper-arid
    Tarim Basin sites that fall outside every polygon, where sp::over returns
    NA and the archive filled the nearest class.
    """
    polygons = db.parse_whittaker_polygons(R_SOURCE)
    hits = misses = 0
    for site in db.site_codes(CORPUS):
        _, rows = db._read_table(os.path.join(CORPUS, site + db.SITE_SUFFIX))
        if not rows:
            continue
        row = rows[0]
        if db.is_absent(row.get("si_biome")) or db.is_absent(row.get("si_mat")):
            continue
        predicted = db.classify(row.get("si_map"), row.get("si_mat"), polygons)
        if predicted == row.get("si_biome"):
            hits += 1
        else:
            misses += 1
    assert hits + misses >= 200, "positive control is vacuous without the archive labels"  # nosec B101
    assert hits >= 200, f"only {hits}/{hits + misses} archive labels regenerate"  # nosec B101


# ----------------------------------------------------------------- site_codes


def test_site_codes_strips_by_exact_suffix(tmp_path):
    """CHE_DAV is a prefix of CHE_DAV_SEE; a prefix glob would conflate them."""
    _write_site(str(tmp_path), "CHE_DAV")
    _write_site(str(tmp_path), "CHE_DAV_SEE")
    assert db.site_codes(str(tmp_path)) == ["CHE_DAV", "CHE_DAV_SEE"]  # nosec B101


# --------------------------------------------------------------------- absent


@pytest.mark.parametrize("value", ["", "NA", "na", "NaN", "none", "-9999", None, "  "])
def test_is_absent_covers_the_archive_spellings(value):
    assert db.is_absent(value)  # nosec B101


def test_is_absent_does_not_swallow_real_values():
    assert not db.is_absent("Temperate forest")  # nosec B101
    assert not db.is_absent("0")  # nosec B101


# --------------------------------------------------------------------- remarks


def test_note_replaces_an_empty_remark():
    assert db._note_remarks("NA") == db.PROVENANCE_NOTE  # nosec B101


def test_note_appends_to_an_existing_remark():
    noted = db._note_remarks("read from workbook X")
    assert noted.startswith("read from workbook X; ")  # nosec B101
    assert noted.endswith(db.PROVENANCE_NOTE)  # nosec B101


def test_note_is_idempotent():
    once = db._note_remarks("read from workbook X")
    assert db._note_remarks(once) == once  # nosec B101


# ----------------------------------------------------------------------- apply


def _apply(tmp_path, synthetic_r, climate_rows):
    climate = tmp_path / "climate.csv"
    with open(climate, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["site_name", "bio1", "bio12"])
        writer.writerows(climate_rows)
    derived = db.derive_for_scale(str(tmp_path), str(climate), synthetic_r)
    return derived, db.apply_to_site_md(str(tmp_path), derived)


def test_apply_writes_all_three_columns_and_the_note(tmp_path, synthetic_r):
    path = _write_site(str(tmp_path), "AUT_ZOE")
    _, written = _apply(tmp_path, synthetic_r, [["AUT_ZOE", "5", "50"]])
    assert written == [("AUT_ZOE", "Woodland/Shrubland")]
    row = _read_site(path)
    assert row["si_biome"] == "Woodland/Shrubland"
    assert row["si_mat"] == "5"
    assert row["si_map"] == "50"
    assert db.PROVENANCE_NOTE in row["si_remarks"]


def test_apply_refuses_to_overwrite_a_present_archive_value(tmp_path, synthetic_r):
    path = _write_site(str(tmp_path), "FRA_PUE", si_biome="Temperate forest", si_mat="11.1", si_map="700")
    _, written = _apply(tmp_path, synthetic_r, [["FRA_PUE", "5", "50"]])
    assert written == []
    row = _read_site(path)
    assert row["si_biome"] == "Temperate forest"
    assert row["si_mat"] == "11.1"


def test_apply_skips_a_site_outside_every_polygon(tmp_path, synthetic_r):
    path = _write_site(str(tmp_path), "CHN_ARG_GWD")
    _, written = _apply(tmp_path, synthetic_r, [["CHN_ARG_GWD", "40", "5000"]])
    assert written == []
    assert db.is_absent(_read_site(path)["si_biome"])


def test_apply_leaves_every_other_field_untouched(tmp_path, synthetic_r):
    path = _write_site(str(tmp_path), "AUT_ZOE", si_lat="47.1", si_long="11.3")
    before = _read_site(path)
    _apply(tmp_path, synthetic_r, [["AUT_ZOE", "5", "50"]])
    after = _read_site(path)
    moved = {k for k in before if before[k] != after[k]}
    assert moved == {"si_biome", "si_mat", "si_map", "si_remarks"}


def test_apply_makes_a_backup_holding_the_original(tmp_path, synthetic_r):
    path = _write_site(str(tmp_path), "AUT_ZOE")
    _apply(tmp_path, synthetic_r, [["AUT_ZOE", "5", "50"]])
    backup = path + db.BACKUP_SUFFIX
    assert os.path.exists(backup)
    assert db.is_absent(_read_site(backup)["si_biome"])


def test_apply_does_not_clobber_an_existing_backup(tmp_path, synthetic_r):
    """A second run must not overwrite the pre-fill snapshot with a filled one."""
    path = _write_site(str(tmp_path), "AUT_ZOE")
    _apply(tmp_path, synthetic_r, [["AUT_ZOE", "5", "50"]])
    _apply(tmp_path, synthetic_r, [["AUT_ZOE", "5", "50"]])
    assert db.is_absent(_read_site(path + db.BACKUP_SUFFIX)["si_biome"])


@pytest.mark.skipif(not hasattr(os, "link"), reason="no hardlink support")
def test_apply_writes_through_the_hardlink(tmp_path, synthetic_r):
    """The Stage-1 twin shares an inode; the fill must reach it too.

    A temp-and-rename would break the link and leave format_unitcon stale,
    so this asserts both the shared content and the surviving link count.
    """
    path = _write_site(str(tmp_path), "AUT_ZOE")
    twin = str(tmp_path / "twin_copy.csv")
    os.link(path, twin)
    assert os.stat(path).st_nlink == 2

    _apply(tmp_path, synthetic_r, [["AUT_ZOE", "5", "50"]])

    assert os.stat(path).st_nlink == 2, "hardlink was broken by the write"
    assert _read_site(twin)["si_biome"] == "Woodland/Shrubland"


def test_derive_reports_missing_climate_rather_than_guessing(tmp_path, synthetic_r):
    _write_site(str(tmp_path), "SWE_NOR")
    derived, written = _apply(tmp_path, synthetic_r, [])
    assert derived["SWE_NOR"].label is None
    assert derived["SWE_NOR"].source == "no WorldClim 2.1 climate"
    assert written == []
