"""Phase 5 R2 -- what must fail, and fail loudly.

The governing principle: in this pipeline a silent wrong answer is far more
expensive than a crash. Every case here is one where continuing would produce a
plausible-looking file that is wrong in a way nothing downstream can detect --
so each one raises, and each test pins the raise rather than the recovery.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.sapwood_combined.assembly import assemble_site
from src.sapwood_combined.conversion import (
    align_flags,
    areas_from_metadata,
    check_flags_columns,
    convert_sapflow,
)
from src.sapwood_combined.membership import EXTERNAL, INTERNAL, Corpus, resolve_routes
from src.sapwood_combined.tests.conftest import metadata_frame, sapf_flags_frame, sapflow_frame, write_site

CODES = ["S_Psy_Jt_1", "S_Psy_Jt_2"]


def _corpora(root: Path):
    return (
        Corpus(origin=EXTERNAL, plant_dir=root / "csv/plant", sapwood_dir=root / "csv/sapwood"),
        Corpus(origin=INTERNAL, plant_dir=root / "internal/plant", sapwood_dir=root / "internal/sapwood"),
    )


def _routes(root: Path):
    return {
        r.site: r
        for r in resolve_routes(
            _corpora(root),
            plant_asw_dir=root / "csv/plant_asw_filled",
            sapwood_asw_dir=root / "csv/sapwood_asw_filled",
        )
    }


# --------------------------------------------------------------------------
# Join failures
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_metadata_without_the_pl_code_column_raises() -> None:
    md = metadata_frame("S", CODES).drop(columns=["pl_code"])
    with pytest.raises(ValueError, match="pl_code"):
        areas_from_metadata(md)


@pytest.mark.unit
def test_metadata_from_the_raw_archive_rather_than_asw_filled_raises() -> None:
    """The raw archive has no filled column, so every tree would silently drop.

    Naming the expected directory matters: the failure is a wrong input path,
    and an error that only said 'column missing' would send someone hunting in
    the wrong place.
    """
    md = metadata_frame("S", CODES).drop(columns=["pl_sapw_area_filled"])
    with pytest.raises(ValueError, match="asw_filled"):
        areas_from_metadata(md)


@pytest.mark.unit
def test_two_metadata_rows_for_one_tree_raises() -> None:
    md = pd.concat([metadata_frame("S", CODES)] * 2, ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        areas_from_metadata(md)


@pytest.mark.unit
def test_a_flux_column_with_no_metadata_row_raises_naming_the_column() -> None:
    frame = sapflow_frame([*CODES, "S_Psy_Jt_ORPHAN"])
    with pytest.raises(ValueError, match="S_Psy_Jt_ORPHAN"):
        convert_sapflow(frame, metadata_frame("S", CODES))


# --------------------------------------------------------------------------
# Flags disagreements
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_flags_missing_a_converted_tree_raises() -> None:
    with pytest.raises(ValueError, match="disagree"):
        align_flags(sapf_flags_frame(["S_Psy_Jt_1"]), converted=tuple(CODES))


@pytest.mark.unit
def test_flags_and_data_headers_that_differ_raise_in_both_directions() -> None:
    with pytest.raises(ValueError, match="have data but no flags"):
        check_flags_columns(sapf_flags_frame(["S_Psy_Jt_1"]).columns, sapflow_frame(CODES).columns)
    with pytest.raises(ValueError, match="have flags but no data"):
        check_flags_columns(sapf_flags_frame(CODES).columns, sapflow_frame(["S_Psy_Jt_1"]).columns)


@pytest.mark.unit
def test_an_unaligned_flags_file_is_refused_even_on_the_copy_path(corpus: Path, tmp_path: Path) -> None:
    """The fast path must be as strict as the rewriting one, or it becomes a hole."""
    sapf_flags_frame(["EXT_PLANT_Psy_Jt_1"]).to_csv(corpus / "csv/plant/EXT_PLANT_sapf_flags.csv", index=False)
    out = tmp_path / "combined"
    out.mkdir()
    with pytest.raises(ValueError, match="disagree"):
        assemble_site(_routes(corpus)["EXT_PLANT"], out)


@pytest.mark.unit
def test_an_unaligned_flags_file_is_refused_on_the_native_route_too(corpus: Path, tmp_path: Path) -> None:
    """A native site is copied verbatim, which is exactly why it needs checking.

    The converted route validates flags against data and raises; the native route
    used to copy both without looking, so 181 of the corpus's 226 sites passed
    through the one guard that was never applied to them. Nothing downstream can
    detect the result: :mod:`src.Analyzers.sap_analyzer` pairs the two files by
    column name, and a tree present in one but not the other silently loses its
    QC record or acquires someone else's.
    """
    sapf_flags_frame(["EXT_SAP_Psy_Jt_1", "EXT_SAP_Psy_Jt_GHOST"]).to_csv(
        corpus / "csv/sapwood/EXT_SAP_sapf_flags.csv", index=False
    )
    out = tmp_path / "combined"
    out.mkdir()
    with pytest.raises(ValueError, match="disagree"):
        assemble_site(_routes(corpus)["EXT_SAP"], out)


# --------------------------------------------------------------------------
# Scale failures
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_a_native_site_whose_metadata_denies_being_at_sapwood_scale_raises(corpus: Path, tmp_path: Path) -> None:
    """Presence in the sapwood directory is a claim, and claims get checked.

    The native route *assigns* ``pl_sap_units = cm3 cm-2 h-1`` rather than
    verifying it, so a plant-scale series filed under ``csv/sapwood/`` would be
    relabelled instead of refused -- and would then sit in the product roughly
    100x too large under a header asserting otherwise. That is the same failure
    :mod:`src.sapwood_combined.conversion` exists to prevent, arriving by the one
    route that does no arithmetic.

    It also makes the assertion checkable at all: a test that reads
    ``pl_sap_units`` back out of the output cannot fail while the code writes it
    unconditionally.
    """
    write_site(
        corpus / "csv/sapwood_asw_filled",
        "EXT_SAP",
        ["EXT_SAP_Psy_Jt_1"],
        units="cm3 h-1",
        kinds=("plant_md",),
    )
    out = tmp_path / "combined"
    out.mkdir()
    with pytest.raises(ValueError, match="cm3 h-1"):
        assemble_site(_routes(corpus)["EXT_SAP"], out)


@pytest.mark.unit
def test_a_native_site_whose_metadata_states_no_units_at_all_raises(corpus: Path, tmp_path: Path) -> None:
    """Silence is not consent: assigning a unit no one declared is the same hole."""
    md = metadata_frame("EXT_SAP", ["EXT_SAP_Psy_Jt_1"], units="cm3 cm-2 h-1").drop(columns=["pl_sap_units"])
    md.to_csv(corpus / "csv/sapwood_asw_filled/EXT_SAP_plant_md.csv", index=False)
    out = tmp_path / "combined"
    out.mkdir()
    with pytest.raises(ValueError, match="pl_sap_units"):
        assemble_site(_routes(corpus)["EXT_SAP"], out)


# --------------------------------------------------------------------------
# Membership failures
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_a_site_in_both_corpora_raises_rather_than_overwriting(corpus: Path) -> None:
    write_site(corpus / "internal/sapwood", "EXT_SAP", ["EXT_SAP_Psy_Jt_1"], units="cm3 cm-2 h-1")
    with pytest.raises(ValueError, match="EXT_SAP"):
        _routes(corpus)


@pytest.mark.unit
def test_a_cross_corpus_collision_that_differs_only_in_spelling_still_raises(corpus: Path) -> None:
    """One flat output directory: 'ext-sap' and 'EXT_SAP' are the same filename risk."""
    write_site(corpus / "internal/sapwood", "ext-sap", ["EXT_SAP_Psy_Jt_1"], units="cm3 cm-2 h-1")
    write_site(corpus / "csv/sapwood_asw_filled", "ext-sap", ["EXT_SAP_Psy_Jt_1"], kinds=("plant_md",))
    with pytest.raises(ValueError, match="(?i)ext_sap"):
        _routes(corpus)


@pytest.mark.unit
def test_a_native_site_without_filled_asw_raises_naming_the_path_and_the_scale(corpus: Path) -> None:
    (corpus / "csv/sapwood_asw_filled/EXT_SAP_plant_md.csv").unlink()
    with pytest.raises(FileNotFoundError, match="sapwood"):
        _routes(corpus)


@pytest.mark.unit
def test_a_converted_site_without_filled_asw_names_the_plant_scale(corpus: Path) -> None:
    (corpus / "csv/plant_asw_filled/EXT_PLANT_plant_md.csv").unlink()
    with pytest.raises(FileNotFoundError, match="plant"):
        _routes(corpus)


@pytest.mark.unit
def test_both_corpora_absent_yields_no_routes_rather_than_an_error(tmp_path: Path) -> None:
    """An empty checkout is a legitimate state; it must not look like a crash."""
    assert (
        resolve_routes(
            _corpora(tmp_path),
            plant_asw_dir=tmp_path / "csv/plant_asw_filled",
            sapwood_asw_dir=tmp_path / "csv/sapwood_asw_filled",
        )
        == []
    )


# --------------------------------------------------------------------------
# CLI guards
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_rebuilding_into_a_non_empty_directory_requires_force(tmp_path: Path, monkeypatch) -> None:
    """Half of an old build merged into a new one would look complete and be wrong."""
    from src.sapwood_combined import build_sapwood_combined as cli

    _install_stub_paths(monkeypatch, tmp_path)
    out = tmp_path / "csv" / "sapwood_combined"
    out.mkdir(parents=True)
    (out / "stale_file.csv").write_text("x\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="--force"):
        cli.run(base_data_dir=str(tmp_path), base_output_dir=str(tmp_path / "outputs"))


def _install_stub_paths(monkeypatch, root: Path) -> None:
    """Point the CLI's PathConfig at a fixture tree.

    The internal corpus lives in the repository rather than under
    ``base_data_dir``, so it cannot be redirected by argument alone.
    """
    import path_config

    class StubPaths:
        def __init__(self, **_kw) -> None:
            self.raw_csv_dir = root / "csv" / "sapwood"
            self.internal_metadata_dir = root / "internal" / "sapwood"

    monkeypatch.setattr(path_config, "PathConfig", StubPaths)
