"""Phase 5 R1 -- boundaries and degenerate shapes.

Each case here is one the real corpus either contains or is one provider
correction away from containing: a site with a single tree, a column that is
entirely missing, an area at the edge of representable, a flux file with no
trees left at all.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.sapwood_combined.assembly import assemble_site
from src.sapwood_combined.conversion import (
    DROPPED_NO_AREA,
    align_flags,
    areas_from_metadata,
    convert_metadata,
    convert_sapflow,
    data_columns,
    timestamp_columns,
)
from src.sapwood_combined.membership import EXTERNAL, INTERNAL, Corpus, canonical_site, resolve_routes
from src.sapwood_combined.tests.conftest import metadata_frame, sapf_flags_frame, sapflow_frame, write_site

CODES = ["S_Psy_Jt_1", "S_Psy_Jt_2"]


def _md(areas=None, **kw):
    return metadata_frame("S", CODES, areas=areas, **kw)


def _routes(root: Path):
    return {
        r.site: r
        for r in resolve_routes(
            (
                Corpus(origin=EXTERNAL, plant_dir=root / "csv/plant", sapwood_dir=root / "csv/sapwood"),
                Corpus(origin=INTERNAL, plant_dir=root / "internal/plant", sapwood_dir=root / "internal/sapwood"),
            ),
            plant_asw_dir=root / "csv/plant_asw_filled",
            sapwood_asw_dir=root / "csv/sapwood_asw_filled",
        )
    }


# --------------------------------------------------------------------------
# Degenerate frames
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_a_sapf_data_with_no_tree_columns_converts_to_itself() -> None:
    """Timestamps only. Nothing to divide, and nothing that can go wrong."""
    frame = sapflow_frame([])
    result = convert_sapflow(frame, _md())

    assert result.converted == ()
    assert result.dropped == ()
    assert list(result.frame.columns) == ["TIMESTAMP", "solar_TIMESTAMP"]
    assert len(result.frame) == 3


@pytest.mark.unit
def test_a_frame_with_no_rows_keeps_its_columns() -> None:
    """An empty site still has to declare its shape."""
    frame = sapflow_frame(CODES).iloc[0:0]
    result = convert_sapflow(frame, _md({"S_Psy_Jt_1": 50.0, "S_Psy_Jt_2": 25.0}))

    assert result.converted == tuple(CODES)
    assert list(result.frame.columns) == ["TIMESTAMP", "solar_TIMESTAMP", *CODES]
    assert len(result.frame) == 0


@pytest.mark.unit
def test_an_all_missing_column_divides_to_all_missing() -> None:
    """A tree that never reported is still a tree; it must not become zeros."""
    frame = sapflow_frame(["S_Psy_Jt_1"], values={"S_Psy_Jt_1": [np.nan] * 3})
    result = convert_sapflow(frame, _md({"S_Psy_Jt_1": 10.0}))

    assert result.converted == ("S_Psy_Jt_1",)
    assert result.frame["S_Psy_Jt_1"].isna().all()


@pytest.mark.unit
def test_a_single_tree_site_converts(corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "combined"
    out.mkdir()
    outcome = assemble_site(_routes(corpus)["INT_PLANT"], out)
    assert outcome.n_trees_in == 1 and outcome.n_trees_out == 1


@pytest.mark.unit
def test_zero_flux_stays_zero() -> None:
    """0 / area is 0, not missing -- a real reading of no flow."""
    frame = sapflow_frame(["S_Psy_Jt_1"], values={"S_Psy_Jt_1": [0.0, 0.0, 0.0]})
    result = convert_sapflow(frame, _md({"S_Psy_Jt_1": 10.0}))
    assert (result.frame["S_Psy_Jt_1"] == 0.0).all()


@pytest.mark.unit
def test_negative_flux_is_preserved() -> None:
    """Reverse flow is a real measurement; the converter must not clip it."""
    frame = sapflow_frame(["S_Psy_Jt_1"], values={"S_Psy_Jt_1": [-50.0, 0.0, 100.0]})
    result = convert_sapflow(frame, _md({"S_Psy_Jt_1": 10.0}))
    np.testing.assert_allclose(result.frame["S_Psy_Jt_1"], [-5.0, 0.0, 10.0])


# --------------------------------------------------------------------------
# Areas at the edges
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("area", [1e-6, 1e-3, 1e6])
def test_extreme_but_usable_areas_still_divide(area: float) -> None:
    """Only non-finite and non-positive are refused; small and large are legal."""
    frame = sapflow_frame(["S_Psy_Jt_1"], values={"S_Psy_Jt_1": [1.0, 2.0, 3.0]})
    result = convert_sapflow(frame, _md({"S_Psy_Jt_1": area}))
    assert result.converted == ("S_Psy_Jt_1",)
    np.testing.assert_allclose(result.frame["S_Psy_Jt_1"], np.array([1.0, 2.0, 3.0]) / area)


@pytest.mark.unit
def test_an_area_given_as_a_numeric_string_is_usable() -> None:
    """CSV round-trips make this ordinary; refusing it would drop a real tree."""
    md = _md({"S_Psy_Jt_1": 50.0, "S_Psy_Jt_2": 25.0})
    md["pl_sapw_area_filled"] = md["pl_sapw_area_filled"].astype(str)
    assert areas_from_metadata(md) == {"S_Psy_Jt_1": 50.0, "S_Psy_Jt_2": 25.0}


@pytest.mark.unit
def test_an_area_that_is_not_a_number_at_all_is_refused() -> None:
    md = _md({"S_Psy_Jt_1": 50.0, "S_Psy_Jt_2": None})
    md.loc[md["pl_code"] == "S_Psy_Jt_2", "pl_sapw_area_filled"] = "not a number"
    assert areas_from_metadata(md) == {"S_Psy_Jt_1": 50.0}


# --------------------------------------------------------------------------
# Column classification
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_timestamp_detection_is_case_insensitive_and_substring_based() -> None:
    cols = ["TIMESTAMP", "solar_TIMESTAMP", "timestamp_utc", "TREE_1"]
    assert timestamp_columns(cols) == ("TIMESTAMP", "solar_TIMESTAMP", "timestamp_utc")
    assert data_columns(cols) == ("TREE_1",)


@pytest.mark.unit
def test_column_order_is_preserved_through_conversion() -> None:
    """Downstream reads some files positionally; reordering silently breaks them."""
    codes = ["S_Psy_Jt_2", "S_Psy_Jt_1"]
    frame = sapflow_frame(codes)
    result = convert_sapflow(frame, metadata_frame("S", codes, areas=dict.fromkeys(codes, 10.0)))
    assert list(result.frame.columns) == ["TIMESTAMP", "solar_TIMESTAMP", *codes]


# --------------------------------------------------------------------------
# Flags at the edges
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_flags_with_no_trees_survive_alignment() -> None:
    aligned = align_flags(sapf_flags_frame([]), converted=())
    assert list(aligned.columns) == ["TIMESTAMP", "solar_TIMESTAMP"]


@pytest.mark.unit
def test_empty_flag_cells_are_not_confused_with_missing() -> None:
    """'' means unflagged. It must not round-trip into NaN."""
    flags = sapf_flags_frame(["S_Psy_Jt_1"])
    flags.loc[0, "S_Psy_Jt_1"] = ""
    aligned = align_flags(flags, converted=("S_Psy_Jt_1",))
    assert aligned.loc[0, "S_Psy_Jt_1"] == ""


# --------------------------------------------------------------------------
# Name folding
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw,folded",
    [
        ("DEU_HOH", "DEU_HOH"),
        ("deu-hoh", "DEU_HOH"),
        ("DEU HOH", "DEU_HOH"),
        ("_DEU__HOH_", "DEU_HOH"),
        ("DEU.HOH", "DEU_HOH"),
    ],
)
def test_site_folding_absorbs_separator_and_case_drift(raw: str, folded: str) -> None:
    assert canonical_site(raw) == folded


@pytest.mark.unit
def test_folding_keeps_genuinely_different_sites_apart() -> None:
    """Over-folding would merge two stands, which is worse than missing a match."""
    assert canonical_site("CZE_BK1") != canonical_site("CZE_BIK")
    assert canonical_site("FRA_HES") != canonical_site("FRA_HES_HE1_NON")


# --------------------------------------------------------------------------
# Metadata edges
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_metadata_with_no_source_column_reports_no_flags() -> None:
    """Older filled files predate the provenance column; that is not an error."""
    from src.sapwood_combined.conversion import source_flag_counts

    md = _md().drop(columns=["pl_sapw_area_source"])
    assert source_flag_counts(md) == {}


@pytest.mark.unit
def test_metadata_describing_trees_the_flux_file_lacks_still_records_their_fate() -> None:
    """A row with no column is tolerated, but must not be labelled 'converted'."""
    md = convert_metadata(_md(), converted=("S_Psy_Jt_1",), dropped=())
    fate = dict(md.set_index("pl_code")["pl_sapwood_scale_origin"])
    assert fate["S_Psy_Jt_1"] == "converted"
    assert fate["S_Psy_Jt_2"] == DROPPED_NO_AREA


@pytest.mark.unit
def test_a_site_whose_every_tree_is_dropped_writes_nothing_at_all(corpus: Path, tmp_path: Path) -> None:
    """ESP_GDN. Not even the untouched kinds -- a partial site is worse than none."""
    write_site(
        corpus / "csv/plant_asw_filled",
        "EXT_PLANT",
        ["EXT_PLANT_Psy_Jt_1", "EXT_PLANT_Psy_Jt_2"],
        areas={"EXT_PLANT_Psy_Jt_1": 0.0, "EXT_PLANT_Psy_Jt_2": np.nan},
        kinds=("plant_md",),
    )
    out = tmp_path / "combined"
    out.mkdir()
    outcome = assemble_site(_routes(corpus)["EXT_PLANT"], out)

    assert outcome.skipped_reason is not None
    assert outcome.kinds == ()
    assert outcome.files_written == 0
    assert list(out.iterdir()) == []


@pytest.mark.unit
def test_float_precision_survives_the_write(corpus: Path, tmp_path: Path) -> None:
    """%.6g keeps six significant digits across ten orders of magnitude.

    The tolerance is the format's own guarantee, not a wish: six significant
    digits bound the relative error at 0.5e-5 in the worst case (mantissa near
    1), so 1e-5 is the tightest honest assertion. Values here span 1e-4 to 1e5
    to show the bound is relative, not absolute -- an absolute tolerance would
    pass trivially at the small end and fail at the large one.
    """
    write_site(
        corpus / "csv/plant",
        "EXT_PLANT",
        ["EXT_PLANT_Psy_Jt_1", "EXT_PLANT_Psy_Jt_2"],
        values={"EXT_PLANT_Psy_Jt_1": [123.456789, 0.000123456, 98765.4321]},
    )
    out = tmp_path / "combined"
    out.mkdir()
    assemble_site(_routes(corpus)["EXT_PLANT"], out)

    after = pd.read_csv(out / "EXT_PLANT_sapf_data.csv")
    expected = np.array([123.456789, 0.000123456, 98765.4321]) / 50.0
    np.testing.assert_allclose(after["EXT_PLANT_Psy_Jt_1"], expected, rtol=1e-5)
    # and the error really is at the format's bound, not merely inside a loose one
    np.testing.assert_array_less(
        np.abs(after["EXT_PLANT_Psy_Jt_1"] - expected) / np.abs(expected),
        1e-5,
    )
