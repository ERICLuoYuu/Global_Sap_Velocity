"""Tests for corpus discovery and the duplicate-stem registry."""

from __future__ import annotations

import pytest

from src.sapwood_imputation.sources import (
    DUPLICATE_SITES,
    EXTERNAL,
    INTERNAL,
    SAME_FOREST_NO_MEASURED_ASW,
    canonical_site,
    discover,
    duplicate_holder,
    summarise,
)


def _write(directory, site: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{site}_plant_md.csv").write_text(
        "pl_species,pl_dbh,pl_sapw_area\nPinus sylvestris,30,400\n", encoding="utf-8"
    )


@pytest.mark.unit
def test_external_only_when_no_internal_dir(tmp_path) -> None:
    _write(tmp_path / "ext", "ARG_MAZ")
    files = discover(tmp_path / "ext")
    assert [f.origin for f in files] == [EXTERNAL]
    assert files[0].may_vote


@pytest.mark.unit
def test_missing_internal_dir_is_skipped_not_fatal(tmp_path) -> None:
    """A checkout without the internal working tree must still run."""
    _write(tmp_path / "ext", "ARG_MAZ")
    files = discover(tmp_path / "ext", tmp_path / "nope")
    assert len(files) == 1


@pytest.mark.unit
def test_internal_dir_present_but_empty_is_an_error(tmp_path) -> None:
    """An existing-but-empty dir means the path is wrong; do not silently skip."""
    _write(tmp_path / "ext", "ARG_MAZ")
    (tmp_path / "int").mkdir()
    with pytest.raises(FileNotFoundError, match="internal metadata dir"):
        discover(tmp_path / "ext", tmp_path / "int")


@pytest.mark.unit
def test_internal_files_are_tagged_and_appended(tmp_path) -> None:
    _write(tmp_path / "ext", "ARG_MAZ")
    _write(tmp_path / "int", "AUT_MMG")
    files = discover(tmp_path / "ext", tmp_path / "int")
    assert [f.origin for f in files] == [EXTERNAL, INTERNAL]
    assert files[1].site == "AUT_MMG"
    assert files[1].may_vote


@pytest.mark.unit
def test_duplicate_site_is_processed_but_may_not_vote(tmp_path) -> None:
    """SWE_NOR's 26 stems are byte-identical to two archive sites."""
    _write(tmp_path / "ext", "ARG_MAZ")
    _write(tmp_path / "int", "SWE_NOR")
    files = discover(tmp_path / "ext", tmp_path / "int")
    duplicate = files[1]
    assert duplicate.may_vote is False
    assert duplicate.duplicate_of == "SWE_NOR_ST1_AF1 + SWE_NOR_ST1_AF2"
    # It is still in the list: the site must appear in the output directory.
    assert len(files) == 2


@pytest.mark.unit
def test_filename_collision_raises(tmp_path) -> None:
    """Both corpora write into one output dir keyed by filename."""
    _write(tmp_path / "ext", "SWE_NOR")
    _write(tmp_path / "int", "SWE_NOR")
    with pytest.raises(ValueError, match="share filenames"):
        discover(tmp_path / "ext", tmp_path / "int")


@pytest.mark.unit
def test_registry_lookup_survives_case_and_separator_drift() -> None:
    """A site-keyed registry that misses silently is the documented hazard."""
    for spelling in ("SWE_NOR", "swe_nor", "SWE-NOR", "Swe-Nor"):
        assert duplicate_holder(spelling) == DUPLICATE_SITES["SWE_NOR"]
    assert duplicate_holder("SWE_SKO_MIN") is None


@pytest.mark.unit
def test_canonical_site_folds_separators() -> None:
    assert canonical_site("swe-nor") == canonical_site("SWE_NOR") == "SWE_NOR"


@pytest.mark.unit
def test_the_two_registries_are_disjoint() -> None:
    """A site either duplicates measured stems or it does not; never both."""
    assert not set(DUPLICATE_SITES) & set(SAME_FOREST_NO_MEASURED_ASW)


@pytest.mark.unit
def test_summarise_names_the_external_holder_not_the_origin(tmp_path) -> None:
    """Regression: reporting f.origin here would print 'internal' as the holder."""
    _write(tmp_path / "ext", "ARG_MAZ")
    _write(tmp_path / "int", "ESP_LMA")
    report = summarise(discover(tmp_path / "ext", tmp_path / "int"))
    assert report["files_by_origin"] == {EXTERNAL: 1, INTERNAL: 1}
    assert report["duplicate_sites_excluded_from_fit"] == {"ESP_LMA": "ESP_MAJ_MAI"}
