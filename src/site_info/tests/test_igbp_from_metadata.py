"""Tests for deriving ``si_igbp`` from stand and species metadata.

The two load-bearing cases are named after the sites that motivated them:

* ``FIN_HYY`` -- a "Coniferous" stand whose instrumented species include a
  birch.  If the habit test looked at every species rather than only the
  needleleaf ones, the stand would come back unresolved and fall through to
  MODIS, which calls it Woody Savanna.
* ``AUT_ZOE`` -- a "Mixed beech-spruce forest" whose instrumented species are
  both broadleaf deciduous, because nobody put a sensor on the spruce.  If the
  species tier ran first, the site would be labelled DBF and be wrong.
"""

from __future__ import annotations

import os

import pytest

from src.site_info import igbp_from_metadata as igbp
from src.site_info import pft_resolution

# --------------------------------------------------------------------------
# leaf type and habit
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["Pinus sylvestris", "Picea abies", "Larix decidua", "Pseudotsuga menziesii", "abies alba"],
)
def test_conifer_genera_are_needleleaf(name):
    assert igbp.leaf_type(name) == igbp.NEEDLELEAF


@pytest.mark.parametrize("name", ["Fagus sylvatica", "Quercus ilex", "Betula pendula", "Theobroma cacao"])
def test_angiosperm_genera_are_broadleaf(name):
    assert igbp.leaf_type(name) == igbp.BROADLEAF


def test_an_unknown_genus_falls_to_broadleaf_and_says_so():
    """The default is documented by a module constant, not buried in a branch."""
    assert igbp.UNKNOWN_GENUS_IS_BROADLEAF is True
    assert igbp.leaf_type("Wollemia nobilis") == igbp.BROADLEAF


def test_an_empty_species_name_does_not_crash():
    assert igbp.leaf_type("") == igbp.BROADLEAF
    assert igbp.leaf_type("   ") == igbp.BROADLEAF


@pytest.mark.parametrize(
    "value,expected",
    [
        ("evergreen", igbp.EVERGREEN),
        ("cold deciduous", igbp.DECIDUOUS),
        ("drought deciduous", igbp.DECIDUOUS),
        ("brevideciduous", igbp.DECIDUOUS),
        ("Cold Deciduous", igbp.DECIDUOUS),
        ("NA", igbp.UNKNOWN_HABIT),
        ("", igbp.UNKNOWN_HABIT),
        (None, igbp.UNKNOWN_HABIT),
        (float("nan"), igbp.UNKNOWN_HABIT),
    ],
)
def test_leaf_habit_vocabulary(value, expected):
    assert igbp.leaf_habit(value) == expected


def test_absent_vocabulary_matches_pft_resolution():
    """Two copies of the same idea drift; a test is what keeps them together."""
    assert igbp._ABSENT == pft_resolution._ABSENT


# --------------------------------------------------------------------------
# the stand tier
# --------------------------------------------------------------------------


def test_coniferous_plus_evergreen_is_enf():
    species = [(igbp.NEEDLELEAF, igbp.EVERGREEN)]
    assert igbp.class_from_stand_type("Coniferous", species) == "ENF"


def test_coniferous_plus_deciduous_is_dnf():
    """A pure larch stand is Deciduous Needleleaf, a class this corpus can reach."""
    species = [(igbp.NEEDLELEAF, igbp.DECIDUOUS)]
    assert igbp.class_from_stand_type("Coniferous", species) == "DNF"


@pytest.mark.parametrize("stated", ["Broadleaf deciduous", "Deciduous broadleaf", "broadleaved"])
def test_broadleaf_deciduous_stands_are_dbf(stated):
    species = [(igbp.BROADLEAF, igbp.DECIDUOUS)]
    assert igbp.class_from_stand_type(stated, species) == "DBF"


def test_broadleaf_evergreen_stand_is_ebf():
    species = [(igbp.BROADLEAF, igbp.EVERGREEN)]
    assert igbp.class_from_stand_type("Broadleaf evergreen", species) == "EBF"


@pytest.mark.parametrize("stated", ["Mixed", "Mixed beech-spruce forest", "mixed forest"])
def test_mixed_stands_are_mf(stated):
    assert igbp.class_from_stand_type(stated, []) == "MF"


@pytest.mark.parametrize(
    "stated,expected",
    [
        ("Woody Savanna", "WSA"),
        ("Open Holm Oak woodland / Savanna", "WSA"),
        ("Savanna", "SAV"),
        ("savanna grassland", "SAV"),
    ],
)
def test_savanna_is_split_on_the_woody_word(stated, expected):
    """Cover outranks leaf type in IGBP, so these never reach the leaf branch."""
    species = [(igbp.BROADLEAF, igbp.EVERGREEN)]
    assert igbp.class_from_stand_type(stated, species) == expected


def test_a_dehesa_is_savanna_not_evergreen_broadleaf():
    """ESP_ABR: MODIS called it Croplands, species alone would call it EBF.

    The provider says Woody Savanna and reports 22 stems/ha at 6.6 m, so the
    open-canopy class is the right one.
    """
    species = [(igbp.BROADLEAF, igbp.EVERGREEN)]  # Quercus ilex
    assert igbp.class_from_stand_type("Woody Savanna", species) == "WSA"


@pytest.mark.parametrize("stated", ["NA", "", None, "   "])
def test_an_absent_stand_type_is_silent_rather_than_wrong(stated):
    species = [(igbp.NEEDLELEAF, igbp.EVERGREEN)]
    assert igbp.class_from_stand_type(stated, species) is None


def test_an_unrecognised_stand_type_is_silent():
    """Unknown wording must fall through to the species tier, not guess."""
    species = [(igbp.NEEDLELEAF, igbp.EVERGREEN)]
    assert igbp.class_from_stand_type("Endoleptic Rendzic Phaeozem", species) is None


def test_fin_hyy_only_the_needleleaf_habit_counts():
    """A Coniferous stand with an instrumented birch is still ENF.

    Without the type filter the habit set is {E, D}, ``_forest_class`` returns
    None, and the site falls through to MODIS -- which calls Hyytiala WSA.
    """
    species = [
        (igbp.NEEDLELEAF, igbp.EVERGREEN),  # Pinus sylvestris
        (igbp.BROADLEAF, igbp.DECIDUOUS),  # Betula pendula
    ]
    assert igbp.class_from_stand_type("Coniferous", species) == "ENF"


def test_a_coniferous_stand_with_no_conifer_instrumented_is_silent():
    """Negative control for the type filter: an empty habit set must not guess."""
    species = [(igbp.BROADLEAF, igbp.DECIDUOUS)]
    assert igbp.class_from_stand_type("Coniferous", species) is None


def test_two_habits_inside_one_leaf_type_is_silent():
    """Larch among spruce: IGBP splits ENF/DNF on cover, which is unmeasured."""
    species = [
        (igbp.NEEDLELEAF, igbp.EVERGREEN),
        (igbp.NEEDLELEAF, igbp.DECIDUOUS),
    ]
    assert igbp.class_from_stand_type("Coniferous", species) is None


# --------------------------------------------------------------------------
# the species tier
# --------------------------------------------------------------------------


def test_species_tier_calls_two_leaf_types_mixed():
    species = [(igbp.BROADLEAF, igbp.DECIDUOUS), (igbp.NEEDLELEAF, igbp.EVERGREEN)]
    assert igbp.class_from_species(species) == "MF"


@pytest.mark.parametrize(
    "pair,expected",
    [
        ((igbp.NEEDLELEAF, igbp.EVERGREEN), "ENF"),
        ((igbp.NEEDLELEAF, igbp.DECIDUOUS), "DNF"),
        ((igbp.BROADLEAF, igbp.EVERGREEN), "EBF"),
        ((igbp.BROADLEAF, igbp.DECIDUOUS), "DBF"),
    ],
)
def test_species_tier_single_leaf_type(pair, expected):
    assert igbp.class_from_species([pair]) == expected


def test_species_tier_is_silent_with_no_species():
    assert igbp.class_from_species([]) is None


def test_species_tier_is_silent_when_habit_is_unknown():
    assert igbp.class_from_species([(igbp.BROADLEAF, igbp.UNKNOWN_HABIT)]) is None


# --------------------------------------------------------------------------
# tier ordering -- the whole point of the module
# --------------------------------------------------------------------------


def test_aut_zoe_the_stand_beats_the_instrumented_species():
    """Positive control for tier order.

    Both instrumented species are broadleaf deciduous, so the species tier
    alone says DBF.  The provider describes a mixed beech-spruce forest, and
    the spruce simply never got a sensor.  Stand wins.
    """
    rows = [("Fagus sylvatica", "cold deciduous"), ("Acer pseudoplatanus", "cold deciduous")]
    assert igbp.class_from_species([(igbp.BROADLEAF, igbp.DECIDUOUS)]) == "DBF"

    code, tier, evidence = igbp.derive_igbp("Mixed beech-spruce forest", rows)
    assert code == "MF"
    assert tier == igbp.TIER_STAND
    assert "Mixed beech-spruce forest" in evidence


def test_the_species_tier_runs_when_the_stand_is_silent():
    rows = [("Picea abies", "evergreen")]
    code, tier, evidence = igbp.derive_igbp("NA", rows)
    assert code == "ENF"
    assert tier == igbp.TIER_SPECIES
    assert "st_forest_type" in evidence


def test_a_site_with_no_evidence_is_unresolved_not_guessed():
    code, tier, evidence = igbp.derive_igbp("NA", [])
    assert code is None
    assert tier == igbp.TIER_NONE
    assert "no species_md rows" in evidence


def test_derive_never_raises_on_junk():
    """The merge calls this inside a try whose except skips the whole site."""
    code, tier, _ = igbp.derive_igbp(object(), [(None, None), ("", 3.5)])
    assert tier in {igbp.TIER_STAND, igbp.TIER_SPECIES, igbp.TIER_NONE}
    assert code is None or code in pft_resolution.IGBP_CODES


# --------------------------------------------------------------------------
# vocabulary agreement with the column this feeds
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forest_type",
    [
        "Coniferous",
        "Broadleaf deciduous",
        "Mixed",
        "Woody Savanna",
        "Savanna",
        "Permanent wetland",
    ],
)
def test_every_derived_code_is_in_the_si_igbp_vocabulary(forest_type):
    """A long name here would one-hot into its own category beside the codes."""
    species = [(igbp.NEEDLELEAF, igbp.EVERGREEN), (igbp.BROADLEAF, igbp.DECIDUOUS)]
    code = igbp.class_from_stand_type(forest_type, species)
    assert code is None or code in pft_resolution.IGBP_CODES


def test_summarise_reports_every_site_and_a_tally():
    rows = {
        "FIN_HYY": ("ENF", igbp.TIER_STAND, "st_forest_type='Coniferous'"),
        "CHE_LAE": ("MF", igbp.TIER_SPECIES, "species=BLD+NLE"),
    }
    lines = igbp.summarise(rows)
    assert any("FIN_HYY" in line and "ENF" in line for line in lines)
    assert any("CHE_LAE" in line and "MF" in line for line in lines)
    assert "2 site(s)" in lines[-1]


# --------------------------------------------------------------------------
# against the real corpus
# --------------------------------------------------------------------------

CORPUS = os.environ.get(
    "IGBP_TEST_CORPUS",
    "/scratch/tmp/yluo2/Global_Sap_Velocity/data/raw/0.1.5/0.1.5/csv/sapwood_combined",
)


@pytest.mark.skipif(not os.path.isdir(CORPUS), reason="corpus not present on this host")
def test_every_internal_site_resolves_without_modis():
    """The whole point: 23 of 23, no fallback, no unresolved."""
    from src.site_info.derive_igbp import derive_for_scale

    rows = derive_for_scale(CORPUS)
    internal = {site: value for site, value in rows.items() if value[1] != "provider"}
    assert len(internal) == 23, sorted(internal)
    unresolved = [site for site, value in internal.items() if value[0] is None]
    assert unresolved == []
    assert all(value[0] in pft_resolution.IGBP_CODES for value in internal.values())


@pytest.mark.skipif(not os.path.isdir(CORPUS), reason="corpus not present on this host")
def test_the_named_sites_get_the_classes_the_evidence_supports():
    from src.site_info.derive_igbp import derive_for_scale

    rows = derive_for_scale(CORPUS)
    expected = {
        "FIN_HYY": "ENF",  # stand says Coniferous; the birch does not vote
        "AUT_ZOE": "MF",  # stand says mixed beech-spruce
        "DEU_HAR": "MF",  # stand says Mixed
        "FRA_BIL": "ENF",  # MODIS said WSA
        "ESP_ABR": "WSA",  # MODIS said CRO
        "ESP_LM1": "WSA",
        "ESP_LMA": "WSA",  # "Open Holm Oak woodland / Savanna"
        "CHE_DAV": "ENF",  # no stand type; Picea abies only
        "CHE_LAE": "MF",  # no stand type; both leaf types instrumented
        "DEU_HOH": "DBF",  # no stand type; three broadleaf deciduous species
        "CZE_LNZ": "DBF",
        "SWE_NOR": "ENF",
    }
    for site, code in expected.items():
        assert rows[site][0] == code, (site, rows[site])


# --------------------------------------------------------------------------
# --apply: writing si_igbp back into the site metadata
# --------------------------------------------------------------------------


def _fake_site(directory, site, igbp="NA", forest_type="Coniferous"):
    """A minimal three-file site, with a comma inside a text field.

    The comma matters: apply rewrites the whole file through the csv writer, so
    a field that needs quoting is exactly where a rewrite could corrupt data.
    """
    import csv as _csv

    def write(name, header, row):
        with open(directory / name, "w", newline="", encoding="utf-8") as handle:
            writer = _csv.writer(handle)
            writer.writerow(header)
            writer.writerow(row)

    write(f"{site}_site_md.csv", ["si_code", "si_igbp", "si_remarks"],
          [site, igbp, "a remark, with a comma"])
    write(f"{site}_stand_md.csv", ["st_site_id", "st_forest_type"], [site, forest_type])
    write(f"{site}_species_md.csv", ["sp_name", "sp_leaf_habit"],
          ["Pinus sylvestris", "evergreen"])


def _read_row(path):
    import csv as _csv

    with open(path, newline="", encoding="utf-8") as handle:
        return next(_csv.DictReader(handle))


def test_apply_fills_si_igbp_and_leaves_a_backup(tmp_path):
    from src.site_info.derive_igbp import BACKUP_SUFFIX, apply_to_site_md, derive_for_scale

    _fake_site(tmp_path, "TST_ONE")
    rows = derive_for_scale(str(tmp_path))
    assert rows["TST_ONE"][0] == "ENF"

    written = apply_to_site_md(str(tmp_path), rows)
    assert written == [("TST_ONE", "ENF")]
    assert _read_row(tmp_path / "TST_ONE_site_md.csv")["si_igbp"] == "ENF"

    backup = tmp_path / ("TST_ONE_site_md.csv" + BACKUP_SUFFIX)
    assert backup.exists()
    assert _read_row(backup)["si_igbp"] == "NA"


def test_a_quoted_field_survives_the_rewrite(tmp_path):
    """The value must be identical, whatever the writer does about quoting."""
    from src.site_info.derive_igbp import apply_to_site_md, derive_for_scale

    _fake_site(tmp_path, "TST_TWO")
    apply_to_site_md(str(tmp_path), derive_for_scale(str(tmp_path)))
    row = _read_row(tmp_path / "TST_TWO_site_md.csv")
    assert row["si_remarks"] == "a remark, with a comma"
    assert row["si_igbp"] == "ENF"


def test_a_second_apply_is_a_no_op_and_keeps_the_first_backup(tmp_path):
    """Re-running must not overwrite a good backup with a modified file."""
    from src.site_info.derive_igbp import BACKUP_SUFFIX, apply_to_site_md, derive_for_scale

    _fake_site(tmp_path, "TST_TRE")
    apply_to_site_md(str(tmp_path), derive_for_scale(str(tmp_path)))
    backup = tmp_path / ("TST_TRE_site_md.csv" + BACKUP_SUFFIX)
    first = backup.read_text(encoding="utf-8")

    again = apply_to_site_md(str(tmp_path), derive_for_scale(str(tmp_path)))
    assert again == []
    assert backup.read_text(encoding="utf-8") == first


def test_apply_refuses_to_overwrite_a_provider_value(tmp_path):
    """Negative control: the guard fires, and the file is left alone."""
    from src.site_info.derive_igbp import ApplyRefusedError, apply_to_site_md, BACKUP_SUFFIX

    _fake_site(tmp_path, "TST_FOR", igbp="DBF")
    forced = {"TST_FOR": ("ENF", "stand", "forced by the test")}
    with pytest.raises(ApplyRefusedError):
        apply_to_site_md(str(tmp_path), forced)

    assert _read_row(tmp_path / "TST_FOR_site_md.csv")["si_igbp"] == "DBF"
    assert not (tmp_path / ("TST_FOR_site_md.csv" + BACKUP_SUFFIX)).exists()
