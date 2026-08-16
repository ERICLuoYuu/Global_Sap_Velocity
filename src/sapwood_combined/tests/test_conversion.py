"""cm3 h-1 -> cm3 cm-2 h-1, one tree at a time.

The load-bearing test here is
:func:`test_a_tree_without_a_usable_area_is_dropped_not_passed_through`. Passing
a column through undivided is the one failure this module exists to prevent: it
leaves plant-level values under a sapwood-level header, where nothing downstream
can tell they are 100x too large. ``plant_to_sapwood_converter`` shipped exactly
that once -- SE_Nor's median read 122 instead of 0.30.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.sapwood_combined.conversion import (
    CONVERSION_COL,
    CONVERTED,
    DROPPED_NO_AREA,
    SAPWOOD_UNITS,
    UNITS_COL,
    align_flags,
    areas_from_metadata,
    check_flags_columns,
    convert_metadata,
    convert_sapflow,
    timestamp_columns,
)
from src.sapwood_combined.tests.conftest import metadata_frame, sapf_flags_frame, sapflow_frame

CODES = ["S_Psy_Jt_1", "S_Psy_Jt_2"]


def _md(areas=None, flags=None):
    return metadata_frame("S", CODES, areas=areas, flags=flags)


@pytest.mark.unit
def test_timestamp_columns_are_found_by_name() -> None:
    cols = ["TIMESTAMP", "solar_TIMESTAMP", "S_Psy_Jt_1"]
    assert timestamp_columns(cols) == ("TIMESTAMP", "solar_TIMESTAMP")


@pytest.mark.unit
def test_each_tree_is_divided_by_its_own_area() -> None:
    frame = sapflow_frame(CODES)
    result = convert_sapflow(frame, _md({"S_Psy_Jt_1": 50.0, "S_Psy_Jt_2": 25.0}))

    assert result.converted == tuple(CODES)
    assert result.dropped == ()
    np.testing.assert_allclose(result.frame["S_Psy_Jt_1"], np.array([100.0, 200.0, 300.0]) / 50.0)
    np.testing.assert_allclose(result.frame["S_Psy_Jt_2"], np.array([200.0, 400.0, 600.0]) / 25.0)


@pytest.mark.unit
def test_timestamps_survive_verbatim() -> None:
    """They are identity, not data: parsing and re-rendering them loses the offset."""
    frame = sapflow_frame(CODES)
    result = convert_sapflow(frame, _md())

    assert list(result.frame.columns[:2]) == ["TIMESTAMP", "solar_TIMESTAMP"]
    pd.testing.assert_series_equal(result.frame["TIMESTAMP"], frame["TIMESTAMP"])
    pd.testing.assert_series_equal(result.frame["solar_TIMESTAMP"], frame["solar_TIMESTAMP"])


@pytest.mark.unit
def test_a_tree_without_a_usable_area_is_dropped_not_passed_through() -> None:
    """The SE_Nor guard. An undivided column is worse than a missing one."""
    frame = sapflow_frame(CODES)
    result = convert_sapflow(frame, _md({"S_Psy_Jt_1": 50.0, "S_Psy_Jt_2": None}))

    assert result.converted == ("S_Psy_Jt_1",)
    assert [code for code, _ in result.dropped] == ["S_Psy_Jt_2"]
    assert "S_Psy_Jt_2" not in result.frame.columns
    # and the survivor really was divided
    np.testing.assert_allclose(result.frame["S_Psy_Jt_1"], np.array([100.0, 200.0, 300.0]) / 50.0)


@pytest.mark.unit
@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_an_area_that_cannot_divide_is_refused(bad: float) -> None:
    result = convert_sapflow(sapflow_frame(CODES), _md({"S_Psy_Jt_1": 50.0, "S_Psy_Jt_2": bad}))
    assert "S_Psy_Jt_2" not in result.frame.columns
    assert result.dropped[0][1] == DROPPED_NO_AREA


@pytest.mark.unit
def test_a_column_absent_from_the_metadata_raises() -> None:
    """A tree with no metadata row is a join failure, not a missing measurement."""
    frame = sapflow_frame(CODES + ["S_Psy_Jt_99"])
    with pytest.raises(ValueError, match="S_Psy_Jt_99"):
        convert_sapflow(frame, _md())


@pytest.mark.unit
def test_a_metadata_row_with_no_column_is_tolerated() -> None:
    """Metadata routinely describes trees the flux file does not carry."""
    frame = sapflow_frame(["S_Psy_Jt_1"])
    result = convert_sapflow(frame, _md())
    assert result.converted == ("S_Psy_Jt_1",)


@pytest.mark.unit
def test_non_numeric_cells_become_missing_not_an_error() -> None:
    frame = sapflow_frame(["S_Psy_Jt_1"])
    frame.loc[1, "S_Psy_Jt_1"] = "NA"
    result = convert_sapflow(frame, _md({"S_Psy_Jt_1": 10.0}))
    assert pd.isna(result.frame.loc[1, "S_Psy_Jt_1"])
    assert result.frame.loc[0, "S_Psy_Jt_1"] == pytest.approx(10.0)


@pytest.mark.unit
def test_areas_from_metadata_keeps_only_usable_values() -> None:
    areas = areas_from_metadata(_md({"S_Psy_Jt_1": 50.0, "S_Psy_Jt_2": 0.0}))
    assert areas == {"S_Psy_Jt_1": 50.0}


@pytest.mark.unit
def test_duplicate_pl_codes_are_refused() -> None:
    """Two rows for one tree means we cannot say which area is its own."""
    md = pd.concat([_md(), _md()], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        areas_from_metadata(md)


@pytest.mark.unit
def test_metadata_declares_the_new_scale_and_keeps_the_provider_original() -> None:
    md = convert_metadata(_md(), converted=("S_Psy_Jt_1",), dropped=(("S_Psy_Jt_2", DROPPED_NO_AREA),))

    assert set(md[UNITS_COL]) == {SAPWOOD_UNITS}
    # pl_sap_units_orig records what the provider delivered; it is not ours to rewrite.
    assert set(md["pl_sap_units_orig"]) == {"kg h-1"}


@pytest.mark.unit
def test_metadata_records_the_fate_of_every_tree() -> None:
    md = convert_metadata(_md(), converted=("S_Psy_Jt_1",), dropped=(("S_Psy_Jt_2", DROPPED_NO_AREA),))
    fate = dict(md.set_index("pl_code")[CONVERSION_COL])
    assert fate == {"S_Psy_Jt_1": CONVERTED, "S_Psy_Jt_2": DROPPED_NO_AREA}


@pytest.mark.unit
def test_conversion_does_not_mutate_its_inputs() -> None:
    frame, md = sapflow_frame(CODES), _md()
    before_frame, before_md = frame.copy(), md.copy()
    convert_sapflow(frame, md)
    convert_metadata(md, converted=tuple(CODES), dropped=())
    pd.testing.assert_frame_equal(frame, before_frame)
    pd.testing.assert_frame_equal(md, before_md)


@pytest.mark.unit
def test_a_site_with_no_convertible_tree_yields_no_converted_columns() -> None:
    """ESP_GDN's real shape: every tree's area refused by the geometric gate."""
    result = convert_sapflow(sapflow_frame(CODES), _md({"S_Psy_Jt_1": None, "S_Psy_Jt_2": None}))
    assert result.converted == ()
    assert len(result.dropped) == 2


# --------------------------------------------------------------------------
# sapf_flags: filtered alongside sapf_data, never divided.
#
# Flags are the QC record -- NA_ADDED / OUT_WARN / NA_PRESENT -- and step 1 of
# sap_analyzer's six-step QC reads them by filename beside the data file. They
# are categorical, so conversion must not touch their values; but a tree dropped
# from sapf_data has to leave sapf_flags too, or the two files disagree about
# which trees the site has.
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_flags_lose_exactly_the_columns_the_data_lost() -> None:
    flags = sapf_flags_frame(CODES)
    aligned = align_flags(flags, converted=("S_Psy_Jt_1",))

    assert list(aligned.columns) == ["TIMESTAMP", "solar_TIMESTAMP", "S_Psy_Jt_1"]
    assert "S_Psy_Jt_2" not in aligned.columns


@pytest.mark.unit
def test_flag_values_are_carried_through_untouched() -> None:
    """Categorical. Dividing a flag, or coercing it to numeric, destroys it."""
    flags = sapf_flags_frame(CODES)
    aligned = align_flags(flags, converted=tuple(CODES))
    pd.testing.assert_frame_equal(aligned, flags)


@pytest.mark.unit
def test_flags_keep_their_timestamps() -> None:
    aligned = align_flags(sapf_flags_frame(CODES), converted=())
    assert list(aligned.columns) == ["TIMESTAMP", "solar_TIMESTAMP"]
    assert len(aligned) == 3


@pytest.mark.unit
def test_a_flags_file_that_disagrees_with_its_data_file_raises() -> None:
    """359/359 archive files align exactly today, so a mismatch is a source change.

    Silently intersecting would let a renamed or dropped tree pass unnoticed --
    the same class of failure as an undivided column, in the QC record instead
    of the data.
    """
    flags = sapf_flags_frame(CODES)
    with pytest.raises(ValueError, match="S_Psy_Jt_9"):
        align_flags(flags, converted=("S_Psy_Jt_1", "S_Psy_Jt_9"))


@pytest.mark.unit
def test_matched_flags_and_data_headers_are_accepted() -> None:
    check_flags_columns(sapf_flags_frame(CODES).columns, sapflow_frame(CODES).columns)


@pytest.mark.unit
def test_a_tree_with_data_but_no_flags_is_refused() -> None:
    """It would be QC'd as if it had no flags, which is not the same as unflagged."""
    with pytest.raises(ValueError, match="S_Psy_Jt_2"):
        check_flags_columns(sapf_flags_frame(["S_Psy_Jt_1"]).columns, sapflow_frame(CODES).columns)


@pytest.mark.unit
def test_a_tree_with_flags_but_no_data_is_refused() -> None:
    """The reverse direction: a flags column describing a tree that is not there."""
    with pytest.raises(ValueError, match="S_Psy_Jt_2"):
        check_flags_columns(sapf_flags_frame(CODES).columns, sapflow_frame(["S_Psy_Jt_1"]).columns)


@pytest.mark.unit
def test_align_flags_does_not_mutate_its_input() -> None:
    flags = sapf_flags_frame(CODES)
    before = flags.copy()
    align_flags(flags, converted=("S_Psy_Jt_1",))
    pd.testing.assert_frame_equal(flags, before)
