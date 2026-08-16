"""Phase 5 round 1 -- edge cases.

Degenerate inputs that exist in the corpus or plausibly could: an env file with
no variables, a variable that is entirely NaN, a QC column whose variable was
dropped upstream, and QC columns arriving as int at one site and float at
another (pandas types the column from whether it contains a NaN, which is a
property of the site, not of the schema).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.qc_flags.columns import split_qc_columns
from src.qc_flags.config import QcFlagConfig
from src.qc_flags.masking import apply_mask, build_mask, count_masked
from src.qc_flags.mechanisms import Mechanism, skip_reason


@pytest.fixture()
def index():
    return pd.date_range("2020-06-01", periods=4, freq="h", name="TIMESTAMP")


class TestEmptyAndDegenerateFrames:
    def test_a_frame_with_no_rows_yields_an_empty_mask(self):
        data = pd.DataFrame({"ta": pd.Series(dtype=float)})
        mask = build_mask(data, None, Mechanism.QC_COLUMNS, QcFlagConfig())
        assert mask.empty
        assert list(mask.columns) == ["ta"]

    def test_a_frame_with_no_variables_yields_a_mask_with_no_columns(self, index):
        data = pd.DataFrame(index=index)
        mask = build_mask(data, None, Mechanism.NONE, QcFlagConfig())
        assert list(mask.columns) == []
        assert len(mask) == len(index)

    def test_swe_nor_shaped_frame_is_skipped_not_processed(self):
        # TIMESTAMP + solar_TIMESTAMP and nothing else: a fully-built scaffold
        # whose ICOS extraction never filled anything in.
        data = pd.DataFrame({"TIMESTAMP": [], "solar_TIMESTAMP": []})
        assert skip_reason(data) is not None

    def test_apply_mask_on_an_empty_mask_returns_the_data_unchanged(self, index):
        data = pd.DataFrame({"ta": [1.0, 2.0, 3.0, 4.0]}, index=index)
        out = apply_mask(data, pd.DataFrame(index=index))
        pd.testing.assert_frame_equal(out, data)


class TestAllNaNVariables:
    def test_an_all_nan_variable_masks_nothing_extra(self, index):
        data = pd.DataFrame({"ta": [np.nan] * 4, "ta_qc": [1, 1, 1, 1]}, index=index)
        mask = build_mask(data, None, Mechanism.QC_COLUMNS, QcFlagConfig())
        # The mask is still True -- the QC verdict does not depend on presence --
        # but the COUNT must be zero, because masking a NaN removes nothing.
        assert mask["ta"].all()
        masked, present = count_masked(mask, data)
        assert masked["ta"] == 0
        assert present["ta"] == 0

    def test_masked_counts_never_exceed_present_counts(self, index):
        data = pd.DataFrame({"ta": [1.0, np.nan, 3.0, np.nan], "ta_qc": [2, 2, 2, 2]}, index=index)
        mask = build_mask(data, None, Mechanism.QC_COLUMNS, QcFlagConfig())
        masked, present = count_masked(mask, data)
        assert masked["ta"] == 2
        assert present["ta"] == 2
        assert masked["ta"] <= present["ta"]


class TestQcColumnWithoutItsVariable:
    def test_an_orphan_qc_column_masks_nothing(self, index):
        data = pd.DataFrame({"pa_qc": [2, 2, 2, 2], "ta": [1.0] * 4}, index=index)
        mask = build_mask(data, None, Mechanism.QC_COLUMNS, QcFlagConfig())
        assert list(mask.columns) == ["ta"]
        assert not mask["ta"].any()

    def test_an_orphan_qc_column_is_still_classified_as_qc(self, index):
        variables, qc = split_qc_columns(pd.DataFrame({"pa_qc": [], "ta": []}))
        assert qc == ["pa_qc"]
        assert variables == ["ta"]


class TestQcDtypeVariation:
    """pandas types a QC column from whether it happens to contain a NaN."""

    @pytest.mark.parametrize(
        "values,expected",
        [
            ([0, 1, 2, 0], [False, True, True, False]),  # int64
            ([0.0, 1.0, 2.0, 0.0], [False, True, True, False]),  # float64
            (["0", "1", "2", "0"], [False, True, True, False]),  # object
            ([0, np.nan, 2, 0], [False, True, True, False]),  # float64 with NaN
        ],
    )
    def test_threshold_behaves_the_same_across_dtypes(self, index, values, expected):
        data = pd.DataFrame({"ta": [1.0] * 4, "ta_qc": values}, index=index)
        mask = build_mask(data, None, Mechanism.QC_COLUMNS, QcFlagConfig())
        assert list(mask["ta"]) == expected

    def test_a_qc_level_above_two_is_masked(self, index):
        # No level 3 exists in the corpus today. If one appears it must mask,
        # not fall through -- the rule is "monotone in quality", not "in {1,2}".
        data = pd.DataFrame({"ta": [1.0] * 4, "ta_qc": [0, 3, 9, 0]}, index=index)
        mask = build_mask(data, None, Mechanism.QC_COLUMNS, QcFlagConfig())
        assert list(mask["ta"]) == [False, True, True, False]

    def test_a_negative_qc_level_is_not_masked_at_the_default_threshold(self, index):
        data = pd.DataFrame({"ta": [1.0] * 4, "ta_qc": [-1, 0, 1, 0]}, index=index)
        mask = build_mask(data, None, Mechanism.QC_COLUMNS, QcFlagConfig())
        assert list(mask["ta"]) == [False, False, True, False]


class TestIntegerVariableDtype:
    def test_masking_an_int_column_does_not_raise(self, index):
        # A variable whose values are all integral is read as int64; assigning
        # NaN into it warns in pandas 2.x and will raise in 3.0.
        data = pd.DataFrame({"precip": [0, 1, 2, 3], "precip_qc": [0, 1, 0, 0]}, index=index)
        mask = build_mask(data, None, Mechanism.QC_COLUMNS, QcFlagConfig())
        out = apply_mask(data, mask)
        assert np.isnan(out["precip"].iloc[1])
        assert out["precip"].iloc[0] == 0


class TestFlagFrameEdges:
    def test_an_empty_flag_frame_masks_nothing(self, index):
        data = pd.DataFrame({"ta": [1.0] * 4}, index=index)
        flags = pd.DataFrame(index=index)
        mask = build_mask(data, flags, Mechanism.SAPFLUXNET_FLAGS, QcFlagConfig())
        assert not mask["ta"].any()

    def test_duplicate_timestamps_in_the_flag_frame_are_deduplicated(self, index):
        data = pd.DataFrame({"ta": [1.0] * 4}, index=index)
        dup_index = pd.DatetimeIndex([index[0], index[0], index[1], index[2], index[3]])
        flags = pd.DataFrame({"ta": ["RANGE_WARN", "", "", "", ""]}, index=dup_index)
        mask = build_mask(data, flags, Mechanism.SAPFLUXNET_FLAGS, QcFlagConfig())
        # keep="first", so the RANGE_WARN survives.
        assert list(mask["ta"]) == [True, False, False, False]

    def test_duplicate_timestamps_in_the_data_broadcast_one_flag_to_each_repeat(self, index):
        # Both analyzers dedupe the data index before masking, so this is a
        # precondition rather than a supported input -- but four real sites do
        # carry duplicates (KOR_TAE_TC{1,2,3} at 68,111 each, PRT_LEZ_ARN at 4),
        # so the behaviour is pinned rather than left to be rediscovered. One
        # flag cell applies to every row sharing its timestamp, which is the
        # only defensible reading: the flag describes the instant, not the row.
        dup_index = pd.DatetimeIndex([index[0], index[0], index[1]])
        data = pd.DataFrame({"ta": [1.0, 2.0, 3.0]}, index=dup_index)
        flags = pd.DataFrame({"ta": ["RANGE_WARN", "RANGE_WARN", ""]}, index=dup_index)
        mask = build_mask(data, flags, Mechanism.SAPFLUXNET_FLAGS, QcFlagConfig())
        assert list(mask["ta"]) == [True, True, False]

    def test_flag_frame_covering_more_timestamps_than_the_data(self, index):
        data = pd.DataFrame({"ta": [1.0] * 4}, index=index)
        wide = pd.date_range("2020-06-01", periods=10, freq="h")
        flags = pd.DataFrame({"ta": ["RANGE_WARN"] * 10}, index=wide)
        mask = build_mask(data, flags, Mechanism.SAPFLUXNET_FLAGS, QcFlagConfig())
        assert len(mask) == 4
        assert mask["ta"].all()
