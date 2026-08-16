"""Phase 5 R3 -- the whole CLI, end to end, over a miniature two-corpus tree.

The unit tests prove each part in isolation. What only an end-to-end run can
prove is that the parts agree: that the manifest describes the directory that
was actually written, that a second run does not drift, and that the ordering
guarantee holds -- MANIFEST.json last, so its absence marks an interrupted run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.sapwood_combined import build_sapwood_combined as cli
from src.sapwood_combined.conversion import SAPWOOD_UNITS, UNITS_COL
from src.sapwood_combined.manifest import MANIFEST_NAME
from src.sapwood_combined.tests.conftest import EXTERNAL_KINDS, INTERNAL_KINDS, write_site


@pytest.fixture
def stub_paths(monkeypatch, corpus: Path):
    """Point the CLI's PathConfig at the fixture corpus.

    The internal corpus is repository-relative rather than under
    ``base_data_dir``, so it cannot be redirected by argument alone.
    """
    import path_config

    class StubPaths:
        def __init__(self, **_kw) -> None:
            self.raw_csv_dir = corpus / "csv" / "sapwood"
            self.internal_metadata_dir = corpus / "internal" / "sapwood"

    monkeypatch.setattr(path_config, "PathConfig", StubPaths)
    return corpus


def _run(root: Path, **kw) -> int:
    return cli.run(base_data_dir=str(root), base_output_dir=str(root / "outputs"), **kw)


def _out(root: Path) -> Path:
    return root / "csv" / "sapwood_combined"


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_a_full_run_writes_every_routed_site(stub_paths: Path) -> None:
    written = _run(stub_paths)
    out = _out(stub_paths)

    assert written == 5
    manifest = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert set(manifest["sites"]) == {"EXT_BOTH", "EXT_PLANT", "EXT_SAP", "INT_PLANT", "INT_SAP"}
    assert manifest["summary"]["native"] == 3
    assert manifest["summary"]["converted"] == 2


@pytest.mark.integration
def test_the_manifest_describes_the_directory_that_was_written(stub_paths: Path) -> None:
    """Every claim in the manifest is checked against the files on disk."""
    _run(stub_paths)
    out = _out(stub_paths)
    manifest = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))

    for site, entry in manifest["sites"].items():
        on_disk = sorted(p.name[len(site) + 1 : -len(".csv")] for p in out.glob(f"{site}_*.csv"))
        assert sorted(entry["kinds"]) == on_disk, site
        assert entry["files"] == len(on_disk), site

    total = sum(1 for p in out.glob("*.csv"))
    assert manifest["summary"]["files_written"] == total


@pytest.mark.integration
def test_the_product_is_uniformly_at_sapwood_scale(stub_paths: Path) -> None:
    """The single claim the whole module exists to make."""
    _run(stub_paths)
    for md_path in _out(stub_paths).glob("*_plant_md.csv"):
        md = pd.read_csv(md_path)
        assert set(md[UNITS_COL]) == {SAPWOOD_UNITS}, md_path.name


@pytest.mark.integration
def test_external_sites_carry_flags_and_internal_ones_declare_why_they_do_not(
    stub_paths: Path,
) -> None:
    _run(stub_paths)
    out = _out(stub_paths)
    manifest = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))

    for site in ("EXT_BOTH", "EXT_PLANT", "EXT_SAP"):
        assert sorted(manifest["sites"][site]["kinds"]) == sorted(EXTERNAL_KINDS)
        assert (out / f"{site}_sapf_flags.csv").exists()

    for site in ("INT_PLANT", "INT_SAP"):
        assert sorted(manifest["sites"][site]["kinds"]) == sorted(INTERNAL_KINDS)
        assert not (out / f"{site}_sapf_flags.csv").exists()

    assert sorted(manifest["flags_available"]["without_flags"]) == ["INT_PLANT", "INT_SAP"]
    assert manifest["flags_available"]["sites_with_flags"] == 3


@pytest.mark.integration
def test_a_site_measured_at_sapwood_scale_is_never_derived(stub_paths: Path) -> None:
    """EXT_BOTH exists at both scales. Its measured series must survive untouched."""
    _run(stub_paths)
    out = _out(stub_paths)

    before = pd.read_csv(stub_paths / "csv/sapwood/EXT_BOTH_sapf_data.csv")
    after = pd.read_csv(out / "EXT_BOTH_sapf_data.csv")
    pd.testing.assert_frame_equal(before, after)

    manifest = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["sites"]["EXT_BOTH"]["route"] == "native"


@pytest.mark.integration
def test_the_overlap_findings_reach_the_product(stub_paths: Path) -> None:
    """Recorded, not applied: every flagged site is still written elsewhere."""
    _run(stub_paths)
    manifest = json.loads((_out(stub_paths) / MANIFEST_NAME).read_text(encoding="utf-8"))

    overlaps = manifest["stem_overlaps"]
    assert len(overlaps) == 9
    for name, entry in overlaps.items():
        assert entry["overlaps"] and entry["relation"] and entry["evidence"], name


@pytest.mark.integration
def test_the_asw_model_behind_the_areas_is_recorded(stub_paths: Path) -> None:
    """Division destroys the denominator, so provenance must be captured here."""
    (stub_paths / "csv/plant_asw_filled/asw_allometry_fit_report.json").write_text(
        json.dumps({"fit": {"n": 2139, "slope": 1.8, "r2": 0.94}, "taxon_offsets": {"offsets": [1, 2, 3]}}),
        encoding="utf-8",
    )
    _run(stub_paths)
    manifest = json.loads((_out(stub_paths) / MANIFEST_NAME).read_text(encoding="utf-8"))

    assert manifest["asw_model"]["taxon_refinement"] is True
    assert manifest["asw_model"]["taxon_offsets"] == 3
    assert manifest["asw_model"]["n"] == 2139


@pytest.mark.integration
def test_an_unrefined_asw_model_is_distinguishable_in_the_output(stub_paths: Path) -> None:
    """The fit statistics are identical either way; only this block differs."""
    (stub_paths / "csv/plant_asw_filled/asw_allometry_fit_report.json").write_text(
        json.dumps({"fit": {"n": 2139}}), encoding="utf-8"
    )
    _run(stub_paths)
    manifest = json.loads((_out(stub_paths) / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["asw_model"]["taxon_refinement"] is False


@pytest.mark.integration
def test_a_missing_fit_report_is_reported_not_guessed(stub_paths: Path) -> None:
    _run(stub_paths)
    manifest = json.loads((_out(stub_paths) / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["asw_model"]["available"] is False
    assert "asw_allometry_fit_report" in manifest["asw_model"]["reason"]


# --------------------------------------------------------------------------
# Re-runs
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_a_second_run_reproduces_the_first_byte_for_byte(stub_paths: Path) -> None:
    """Everything except the manifest, whose timestamp is meant to move."""
    _run(stub_paths)
    out = _out(stub_paths)
    first = {p.name: p.read_bytes() for p in out.glob("*.csv")}

    _run(stub_paths, force=True)
    second = {p.name: p.read_bytes() for p in out.glob("*.csv")}

    assert set(first) == set(second)
    for name in first:
        assert first[name] == second[name], name


@pytest.mark.integration
def test_the_manifest_is_written_last(stub_paths: Path) -> None:
    """Its absence is the only signal that a run was interrupted."""
    _run(stub_paths)
    out = _out(stub_paths)
    manifest = out / MANIFEST_NAME
    newest_csv = max(p.stat().st_mtime_ns for p in out.glob("*.csv"))
    assert manifest.stat().st_mtime_ns >= newest_csv


@pytest.mark.integration
def test_a_dry_run_writes_nothing_at_all(stub_paths: Path) -> None:
    assert _run(stub_paths, dry_run=True) == 0
    assert not _out(stub_paths).exists()


@pytest.mark.integration
def test_linking_produces_the_same_content_as_copying(stub_paths: Path) -> None:
    """--link is a storage optimisation; it must not change a single byte."""
    _run(stub_paths)
    out = _out(stub_paths)
    copied = {p.name: p.read_bytes() for p in out.glob("*.csv")}

    _run(stub_paths, force=True, link=True)
    linked = {p.name: p.read_bytes() for p in out.glob("*.csv")}
    assert copied == linked


# --------------------------------------------------------------------------
# A site that cannot be converted
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_a_site_with_no_usable_area_is_skipped_and_named(stub_paths: Path) -> None:
    """ESP_GDN's shape. Absent from the product, present in the manifest."""
    write_site(
        stub_paths / "csv/plant_asw_filled",
        "EXT_PLANT",
        ["EXT_PLANT_Psy_Jt_1", "EXT_PLANT_Psy_Jt_2"],
        areas={"EXT_PLANT_Psy_Jt_1": None, "EXT_PLANT_Psy_Jt_2": None},
        kinds=("plant_md",),
    )
    written = _run(stub_paths)
    out = _out(stub_paths)

    assert written == 4
    assert list(out.glob("EXT_PLANT_*")) == []
    manifest = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert "usable sapwood area" in manifest["skipped"]["EXT_PLANT"]
    assert "EXT_PLANT" not in manifest["sites"]


@pytest.mark.integration
def test_the_cli_entry_point_parses_and_runs(stub_paths: Path) -> None:
    assert cli.main(["--base-data-dir", str(stub_paths), "--dry-run"]) == 0


# --------------------------------------------------------------------------
# Independence from PathConfig drift
#
# The canonical environment's path_config.py is older than this machine's and
# has no `internal_metadata_dir`. Depending on that attribute made the CLI fail
# on Palma with an AttributeError -- a deployment breaking on a detail of a
# shared config file it did not need to read.
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_an_older_path_config_without_the_internal_attribute_still_runs(monkeypatch, corpus: Path) -> None:
    """Palma's path_config.py predates `internal_metadata_dir`."""
    import path_config

    class OldPaths:
        def __init__(self, **_kw) -> None:
            self.raw_csv_dir = corpus / "csv" / "sapwood"
            # deliberately no internal_metadata_dir

    monkeypatch.setattr(path_config, "PathConfig", OldPaths)
    written = cli.run(
        base_data_dir=str(corpus),
        base_output_dir=str(corpus / "outputs"),
        internal_root=corpus / "internal",
    )
    assert written == 5


@pytest.mark.integration
def test_skip_internal_builds_the_archive_alone(stub_paths: Path) -> None:
    """Useful wherever the internal corpus is absent or stale."""
    written = _run(stub_paths, skip_internal=True)
    manifest = json.loads((_out(stub_paths) / MANIFEST_NAME).read_text(encoding="utf-8"))

    assert written == 3
    assert set(manifest["sites"]) == {"EXT_BOTH", "EXT_PLANT", "EXT_SAP"}
    assert all(e["origin"] == "external" for e in manifest["sites"].values())


@pytest.mark.integration
def test_skip_internal_is_recorded_so_a_partial_build_is_not_mistaken_for_a_full_one(
    stub_paths: Path,
) -> None:
    """A 202-site product and a 226-site product must not look alike."""
    _run(stub_paths, skip_internal=True)
    manifest = json.loads((_out(stub_paths) / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["corpora"]["internal_included"] is False

    _run(stub_paths, force=True)
    manifest = json.loads((_out(stub_paths) / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["corpora"]["internal_included"] is True
