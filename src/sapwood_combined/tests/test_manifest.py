"""The manifest is the product's own account of itself.

It is written last on purpose: its absence is how a consumer tells an interrupted
run from a finished one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.sapwood_combined.assembly import FILE_KINDS, SiteOutcome
from src.sapwood_combined.manifest import MANIFEST_NAME, build_manifest, write_manifest
from src.sapwood_combined.membership import EXTERNAL, INTERNAL, ROUTE_CONVERTED, ROUTE_NATIVE

NINE_KINDS = FILE_KINDS
SEVEN_KINDS = FILE_KINDS[:-2]

#: What the archive ships, and what the internal corpus ships.
NINE_KINDS = (
    "sapf_data",
    "plant_md",
    "env_data",
    "env_md",
    "site_md",
    "stand_md",
    "species_md",
    "sapf_flags",
    "env_flags",
)
SEVEN_KINDS = NINE_KINDS[:7]

OUTCOMES = (
    SiteOutcome(
        site="EXT_SAP",
        origin=EXTERNAL,
        route=ROUTE_NATIVE,
        n_trees_in=3,
        n_trees_out=3,
        dropped=(),
        source_flags={"measured": 3},
        files_written=9,
        kinds=NINE_KINDS,
    ),
    SiteOutcome(
        site="EXT_PLANT",
        origin=EXTERNAL,
        route=ROUTE_CONVERTED,
        n_trees_in=4,
        n_trees_out=3,
        dropped=(("EXT_PLANT_Psy_Jt_4", "no usable sapwood area"),),
        source_flags={"imputed": 3, "implausible": 1},
        files_written=9,
        kinds=NINE_KINDS,
    ),
    SiteOutcome(
        site="INT_NATIVE",
        origin=INTERNAL,
        route=ROUTE_NATIVE,
        n_trees_in=2,
        n_trees_out=2,
        dropped=(),
        source_flags={"measured": 2},
        files_written=7,
        kinds=SEVEN_KINDS,
    ),
    SiteOutcome(
        site="ESP_GDN",
        origin=INTERNAL,
        route=ROUTE_CONVERTED,
        n_trees_in=4,
        n_trees_out=0,
        dropped=(("a", "no usable sapwood area"),),
        source_flags={"implausible": 4},
        files_written=0,
        skipped_reason="no tree has a usable sapwood area",
    ),
)


@pytest.mark.unit
def test_manifest_counts_only_written_sites() -> None:
    m = build_manifest(OUTCOMES, asw_model={})
    assert m["summary"]["sites_written"] == 3
    assert m["summary"]["sites_skipped"] == 1
    assert m["summary"]["native"] == 2
    assert m["summary"]["converted"] == 1


@pytest.mark.unit
def test_skipped_sites_are_named_with_a_reason() -> None:
    m = build_manifest(OUTCOMES, asw_model={})
    assert "ESP_GDN" in m["skipped"]
    assert "usable sapwood area" in m["skipped"]["ESP_GDN"]
    assert "ESP_GDN" not in m["sites"]


@pytest.mark.unit
def test_every_written_site_reports_its_route_and_tree_counts() -> None:
    m = build_manifest(OUTCOMES, asw_model={})
    entry = m["sites"]["EXT_PLANT"]
    assert entry["route"] == ROUTE_CONVERTED
    assert entry["origin"] == EXTERNAL
    assert entry["n_trees_in"] == 4
    assert entry["n_trees_out"] == 3
    assert entry["dropped"] == [["EXT_PLANT_Psy_Jt_4", "no usable sapwood area"]]
    assert entry["asw_source_flags"] == {"imputed": 3, "implausible": 1}


@pytest.mark.unit
def test_the_asw_model_is_recorded_so_the_output_can_be_traced() -> None:
    """A run against unrefined ASW must not be indistinguishable from a refined one."""
    m = build_manifest(OUTCOMES, asw_model={"taxon_refinement": True, "n": 2139})
    assert m["asw_model"]["taxon_refinement"] is True
    assert m["asw_model"]["n"] == 2139


@pytest.mark.unit
def test_stem_overlaps_are_reported_by_default() -> None:
    """The sweep's findings must reach the product, since nothing acts on them."""
    m = build_manifest(OUTCOMES, asw_model={})
    overlaps = m["stem_overlaps"]
    assert "SWE_NOR" in overlaps and "ESP_LMA" in overlaps
    for entry in overlaps.values():
        assert entry["overlaps"] and entry["relation"] and entry["evidence"]


@pytest.mark.unit
def test_overlapping_sites_are_still_written() -> None:
    """Advisory means advisory: a flagged site is not filtered out."""
    from src.sapwood_combined.membership import OVERLAP_NOTES, overlap_note

    assert overlap_note("SWE_NOR") is not None
    assert overlap_note("swe-nor") is not None, "lookup must survive separator drift"
    assert overlap_note("NOT_A_SITE") is None
    # No site in the registry is treated as excluded anywhere in routing.
    assert "ESP_GDN" not in OVERLAP_NOTES, "ESP_GDN is skipped for missing area, not overlap"


@pytest.mark.unit
def test_duplicate_stem_sites_are_flagged_for_the_consumer() -> None:
    """They are real sites, but training them alongside their archive twin double-counts."""
    m = build_manifest(OUTCOMES, asw_model={}, duplicate_sites={"SWE_NOR": "SWE_NOR_ST1_AF1"})
    assert m["duplicate_stems"]["SWE_NOR"] == "SWE_NOR_ST1_AF1"


@pytest.mark.unit
def test_manifest_is_json_serialisable_and_lands_in_the_output_dir(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, build_manifest(OUTCOMES, asw_model={}))
    assert path == tmp_path / MANIFEST_NAME
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["summary"]["sites_written"] == 3


@pytest.mark.unit
def test_manifest_records_the_scale_it_claims() -> None:
    m = build_manifest(OUTCOMES, asw_model={})
    assert m["units"] == "cm3 cm-2 h-1"


@pytest.mark.unit
def test_each_site_records_which_kinds_it_wrote() -> None:
    """7 vs 9 is a property of the source corpus, so the product must state it."""
    m = build_manifest(OUTCOMES, asw_model={})
    assert m["sites"]["EXT_SAP"]["kinds"] == list(NINE_KINDS)
    assert m["sites"]["INT_NATIVE"]["kinds"] == list(SEVEN_KINDS)


@pytest.mark.unit
def test_missing_flags_are_declared_as_provenance_not_loss() -> None:
    """Absent flags are indistinguishable from dropped ones unless the manifest says so.

    QC treats a missing flags file as 'no flags' rather than an error, so nothing
    downstream would ever complain about the gap.
    """
    m = build_manifest(OUTCOMES, asw_model={})
    flags = m["flags_available"]

    assert flags["sites_with_flags"] == 2
    assert flags["sites_without_flags"] == 1
    assert flags["without_flags"] == ["INT_NATIVE"]
    assert "never generated them" in flags["note"]
    assert "ESP_GDN" not in flags["without_flags"], "a skipped site is not a flagless one"
