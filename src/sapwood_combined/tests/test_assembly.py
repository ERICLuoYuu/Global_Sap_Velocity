"""Placing one site into the combined directory.

Two invariants: an unchanged file arrives byte-identical, and a changed file
arrives changed everywhere it should be.
"""

from __future__ import annotations

import filecmp
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.sapwood_combined.assembly import FILE_KINDS, assemble_site, place_file
from src.sapwood_combined.conversion import CONVERSION_COL, CONVERTED, NATIVE, SAPWOOD_UNITS, UNITS_COL
from src.sapwood_combined.membership import EXTERNAL, ROUTE_CONVERTED, ROUTE_NATIVE, Corpus, resolve_routes

# INTERNAL_KINDS is deliberately NOT FILE_KINDS: the internal corpus ships seven
# of the nine kinds, and equating them would make the test below assert the very
# thing it exists to refuse.
from src.sapwood_combined.tests.conftest import INTERNAL_KINDS, write_site


def _routes(root: Path):
    return {
        r.site: r
        for r in resolve_routes(
            (
                Corpus(origin=EXTERNAL, plant_dir=root / "csv/plant", sapwood_dir=root / "csv/sapwood"),
                Corpus(origin="internal", plant_dir=root / "internal/plant", sapwood_dir=root / "internal/sapwood"),
            ),
            plant_asw_dir=root / "csv/plant_asw_filled",
            sapwood_asw_dir=root / "csv/sapwood_asw_filled",
        )
    }


@pytest.mark.unit
def test_place_file_copies_bytes_exactly(tmp_path: Path) -> None:
    src = tmp_path / "a.csv"
    src.write_text("TIMESTAMP,x\n2015-01-01T00:00:00Z,1.5\n", encoding="utf-8")
    dst = tmp_path / "out" / "a.csv"
    dst.parent.mkdir()
    place_file(src, dst)
    assert filecmp.cmp(src, dst, shallow=False)


@pytest.mark.unit
def test_a_native_site_arrives_byte_identical(corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "combined"
    out.mkdir()
    outcome = assemble_site(_routes(corpus)["EXT_SAP"], out)

    assert outcome.route == ROUTE_NATIVE
    for kind in FILE_KINDS:
        name = f"EXT_SAP_{kind}.csv"
        if kind == "plant_md":
            continue  # taken from sapwood_asw_filled, checked separately
        assert filecmp.cmp(corpus / "csv/sapwood" / name, out / name, shallow=False), kind


@pytest.mark.unit
def test_an_external_site_writes_all_nine_kinds(corpus: Path, tmp_path: Path) -> None:
    """Including sapf_flags and env_flags, which an earlier draft missed entirely.

    Dropping them is silent: sap_analyzer resolves the flags file by name beside
    the data file and falls back to ``flags = None`` when it is absent, so QC
    step 1 would simply never run on any archive site.
    """
    out = tmp_path / "combined"
    out.mkdir()
    assemble_site(_routes(corpus)["EXT_SAP"], out)
    assert sorted(p.name for p in out.glob("EXT_SAP_*.csv")) == sorted(f"EXT_SAP_{k}.csv" for k in FILE_KINDS)
    assert len(FILE_KINDS) == 9


@pytest.mark.unit
def test_an_internal_site_writes_the_seven_kinds_it_has_and_invents_no_flags(corpus: Path, tmp_path: Path) -> None:
    """The internal corpus generated no flags files. Absent by provenance, not by loss."""
    out = tmp_path / "combined"
    out.mkdir()
    outcome = assemble_site(_routes(corpus)["INT_SAP"], out)

    written = sorted(p.name for p in out.glob("INT_SAP_*.csv"))
    assert written == sorted(f"INT_SAP_{k}.csv" for k in INTERNAL_KINDS)
    assert "INT_SAP_sapf_flags.csv" not in written
    assert "sapf_flags" not in outcome.kinds


@pytest.mark.unit
def test_every_site_reports_the_kinds_it_actually_wrote(corpus: Path, tmp_path: Path) -> None:
    """The manifest's per-site `kinds` is what tells a consumer 7-vs-9 is expected."""
    out = tmp_path / "combined"
    out.mkdir()
    for site in ("EXT_SAP", "EXT_PLANT", "INT_SAP", "INT_PLANT"):
        outcome = assemble_site(_routes(corpus)[site], out)
        on_disk = {p.name[len(site) + 1 : -len(".csv")] for p in out.glob(f"{site}_*.csv")}
        assert set(outcome.kinds) == on_disk, site


@pytest.mark.unit
def test_a_native_sites_flags_arrive_byte_identical(corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "combined"
    out.mkdir()
    assemble_site(_routes(corpus)["EXT_SAP"], out)
    for kind in ("sapf_flags", "env_flags"):
        name = f"EXT_SAP_{kind}.csv"
        assert filecmp.cmp(corpus / "csv/sapwood" / name, out / name, shallow=False), kind


@pytest.mark.unit
def test_a_converted_site_drops_the_same_tree_from_data_and_flags(corpus: Path, tmp_path: Path) -> None:
    """The pair must agree on which trees the site has."""

    write_site(
        corpus / "csv/plant_asw_filled",
        "EXT_PLANT",
        ["EXT_PLANT_Psy_Jt_1", "EXT_PLANT_Psy_Jt_2"],
        areas={"EXT_PLANT_Psy_Jt_1": 50.0, "EXT_PLANT_Psy_Jt_2": None},
        kinds=("plant_md",),
    )
    out = tmp_path / "combined"
    out.mkdir()
    assemble_site(_routes(corpus)["EXT_PLANT"], out)

    data = pd.read_csv(out / "EXT_PLANT_sapf_data.csv")
    flags = pd.read_csv(out / "EXT_PLANT_sapf_flags.csv")
    assert list(data.columns) == list(flags.columns)
    assert "EXT_PLANT_Psy_Jt_2" not in flags.columns


@pytest.mark.unit
def test_a_converted_site_that_drops_nothing_leaves_its_flags_byte_identical(corpus: Path, tmp_path: Path) -> None:
    """Only sapf_data is arithmetic. Flags are labels, and here nothing is filtered.

    Byte-identity rather than value-equality: with no tree dropped there is no
    column to remove, so the file must be copied, not re-serialised.
    """
    out = tmp_path / "combined"
    out.mkdir()
    outcome = assemble_site(_routes(corpus)["EXT_PLANT"], out)

    assert outcome.dropped == ()
    name = "EXT_PLANT_sapf_flags.csv"
    assert filecmp.cmp(corpus / "csv/plant" / name, out / name, shallow=False)


@pytest.mark.unit
def test_flag_labels_survive_a_filtering_rewrite(corpus: Path, tmp_path: Path) -> None:
    """When a tree IS dropped the file is rewritten -- the surviving labels must not move."""
    write_site(
        corpus / "csv/plant_asw_filled",
        "EXT_PLANT",
        ["EXT_PLANT_Psy_Jt_1", "EXT_PLANT_Psy_Jt_2"],
        areas={"EXT_PLANT_Psy_Jt_1": 50.0, "EXT_PLANT_Psy_Jt_2": None},
        kinds=("plant_md",),
    )
    out = tmp_path / "combined"
    out.mkdir()
    assemble_site(_routes(corpus)["EXT_PLANT"], out)

    before = pd.read_csv(corpus / "csv/plant/EXT_PLANT_sapf_flags.csv", keep_default_na=False, dtype=str)
    after = pd.read_csv(out / "EXT_PLANT_sapf_flags.csv", keep_default_na=False, dtype=str)
    pd.testing.assert_series_equal(before["EXT_PLANT_Psy_Jt_1"], after["EXT_PLANT_Psy_Jt_1"])
    pd.testing.assert_series_equal(before["TIMESTAMP"], after["TIMESTAMP"])


@pytest.mark.unit
def test_a_native_sites_metadata_is_marked_native(corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "combined"
    out.mkdir()
    assemble_site(_routes(corpus)["EXT_SAP"], out)
    md = pd.read_csv(out / "EXT_SAP_plant_md.csv")
    assert set(md[CONVERSION_COL]) == {NATIVE}
    assert set(md[UNITS_COL]) == {SAPWOOD_UNITS}


@pytest.mark.unit
def test_a_converted_site_divides_its_flux(corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "combined"
    out.mkdir()
    outcome = assemble_site(_routes(corpus)["EXT_PLANT"], out)

    assert outcome.route == ROUTE_CONVERTED
    assert outcome.n_trees_in == 2 and outcome.n_trees_out == 2
    before = pd.read_csv(corpus / "csv/plant/EXT_PLANT_sapf_data.csv")
    after = pd.read_csv(out / "EXT_PLANT_sapf_data.csv")
    np.testing.assert_allclose(after["EXT_PLANT_Psy_Jt_1"], before["EXT_PLANT_Psy_Jt_1"] / 50.0)
    np.testing.assert_allclose(after["EXT_PLANT_Psy_Jt_2"], before["EXT_PLANT_Psy_Jt_2"] / 25.0)


@pytest.mark.unit
def test_a_converted_site_declares_sapwood_units(corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "combined"
    out.mkdir()
    assemble_site(_routes(corpus)["EXT_PLANT"], out)
    md = pd.read_csv(out / "EXT_PLANT_plant_md.csv")
    assert set(md[UNITS_COL]) == {SAPWOOD_UNITS}
    assert set(md[CONVERSION_COL]) == {CONVERTED}


@pytest.mark.unit
def test_a_converted_site_copies_its_untouched_kinds_verbatim(corpus: Path, tmp_path: Path) -> None:
    """Three files are rewritten -- sapf_data, plant_md, sapf_flags -- and no others.

    sapf_flags joins them because a dropped tree must lose its flags column too,
    so the file is re-serialised even when (as here) nothing is dropped.
    """
    out = tmp_path / "combined"
    out.mkdir()
    assemble_site(_routes(corpus)["EXT_PLANT"], out)
    for kind in FILE_KINDS:
        if kind in ("sapf_data", "plant_md", "sapf_flags"):
            continue
        name = f"EXT_PLANT_{kind}.csv"
        assert filecmp.cmp(corpus / "csv/plant" / name, out / name, shallow=False), kind


@pytest.mark.unit
def test_a_dropped_tree_leaves_the_flux_file(corpus: Path, tmp_path: Path) -> None:
    """The SE_Nor guard, end to end."""

    write_site(
        corpus / "csv/plant_asw_filled",
        "EXT_PLANT",
        ["EXT_PLANT_Psy_Jt_1", "EXT_PLANT_Psy_Jt_2"],
        areas={"EXT_PLANT_Psy_Jt_1": 50.0, "EXT_PLANT_Psy_Jt_2": None},
        kinds=("plant_md",),
    )
    out = tmp_path / "combined"
    out.mkdir()
    outcome = assemble_site(_routes(corpus)["EXT_PLANT"], out)

    after = pd.read_csv(out / "EXT_PLANT_sapf_data.csv")
    assert "EXT_PLANT_Psy_Jt_2" not in after.columns
    assert outcome.n_trees_in == 2 and outcome.n_trees_out == 1
    assert [code for code, _ in outcome.dropped] == ["EXT_PLANT_Psy_Jt_2"]


@pytest.mark.unit
def test_a_site_with_no_convertible_tree_is_skipped_and_writes_nothing(corpus: Path, tmp_path: Path) -> None:
    """ESP_GDN. An all-timestamp file would read downstream as a real but empty site."""

    write_site(
        corpus / "csv/plant_asw_filled",
        "EXT_PLANT",
        ["EXT_PLANT_Psy_Jt_1", "EXT_PLANT_Psy_Jt_2"],
        areas={"EXT_PLANT_Psy_Jt_1": None, "EXT_PLANT_Psy_Jt_2": None},
        kinds=("plant_md",),
    )
    out = tmp_path / "combined"
    out.mkdir()
    outcome = assemble_site(_routes(corpus)["EXT_PLANT"], out)

    assert outcome.skipped_reason is not None
    assert outcome.n_trees_out == 0
    assert list(out.glob("EXT_PLANT_*")) == []


@pytest.mark.unit
def test_source_flag_counts_are_reported(corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "combined"
    out.mkdir()
    outcome = assemble_site(_routes(corpus)["EXT_PLANT"], out)
    assert outcome.source_flags == {"measured": 2}


@pytest.mark.unit
def test_assembly_is_idempotent(corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "combined"
    out.mkdir()
    assemble_site(_routes(corpus)["EXT_PLANT"], out)
    first = (out / "EXT_PLANT_sapf_data.csv").read_bytes()
    assemble_site(_routes(corpus)["EXT_PLANT"], out)
    assert (out / "EXT_PLANT_sapf_data.csv").read_bytes() == first
