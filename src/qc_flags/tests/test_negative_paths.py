"""Phase 5 round 2 -- negative paths.

Malformed input, and the difference between *degrading* and *lying*.  The rule
throughout: a missing or unreadable quality verdict is masked, never trusted.
An absent QC value is not evidence of quality, and the conservative direction
costs a handful of rows while the permissive one lets unlabelled data into
training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.qc_flags.config import QcFlagConfig
from src.qc_flags.masking import apply_mask, build_mask, qc_mask_for_column
from src.qc_flags.mechanisms import Mechanism, detect_mechanism
from src.qc_flags.removal_log_validator import RowStatus, validate_log
from src.qc_flags.vocabulary import actionable_mask, is_actionable


@pytest.fixture()
def index():
    return pd.date_range("2020-06-01", periods=4, freq="h", name="TIMESTAMP")


class TestMalformedQcValues:
    @pytest.mark.parametrize("bad", ["", "  ", "n/a", "NA", "bad", "1.5.2", None])
    def test_unreadable_verdicts_are_masked_not_trusted(self, bad):
        series = pd.Series([0, bad, 0])
        mask = qc_mask_for_column(series, threshold=1)
        assert list(mask) == [False, True, False]

    def test_a_float_that_is_not_a_whole_level_still_compares(self):
        # 1.5 is not a level the schema defines, but it is >= 1 and must mask
        # rather than fall through as unrecognised.
        assert list(qc_mask_for_column(pd.Series([0.4, 1.5]), threshold=1)) == [False, True]

    def test_infinity_is_masked(self):
        assert list(qc_mask_for_column(pd.Series([0, np.inf]), threshold=1)) == [False, True]

    def test_an_entirely_unparseable_qc_column_masks_the_whole_variable(self, index):
        data = pd.DataFrame({"ta": [1.0] * 4, "ta_qc": ["x", "y", "z", "w"]}, index=index)
        mask = build_mask(data, None, Mechanism.QC_COLUMNS, QcFlagConfig())
        assert mask["ta"].all()


class TestMalformedFlagTokens:
    @pytest.mark.parametrize("bad", ["", ";", ";;", "  ;  ", None, np.nan])
    def test_empty_and_separator_only_cells_are_not_actionable(self, bad):
        assert is_actionable(bad) is False

    def test_a_token_that_merely_contains_range_warn_is_not_actionable(self):
        # Substring matching would fire on all of these.
        for lookalike in ("RANGE_WARNING", "NOT_RANGE_WARN_REALLY", "XRANGE_WARN"):
            assert is_actionable(lookalike) is False, lookalike

    def test_whitespace_around_a_token_is_tolerated(self):
        assert is_actionable("NA_PRESENT ; RANGE_WARN ") is True

    def test_lowercase_is_not_actionable(self):
        # The archive is uppercase. Accepting lowercase would be inventing a
        # vocabulary the data does not have.
        assert is_actionable("range_warn") is False

    def test_actionable_mask_on_a_numeric_series_does_not_raise(self):
        # A flag column that arrived as numbers is malformed, not actionable.
        assert not actionable_mask(pd.Series([1, 2, 3])).any()

    def test_actionable_mask_on_an_all_nan_series(self):
        assert not actionable_mask(pd.Series([np.nan, np.nan])).any()


class TestFlagFrameDoesNotMatchData:
    def test_flag_columns_naming_nothing_in_the_data_are_ignored(self, index):
        data = pd.DataFrame({"ta": [1.0] * 4}, index=index)
        flags = pd.DataFrame({"something_else": ["RANGE_WARN"] * 4}, index=index)
        mask = build_mask(data, flags, Mechanism.SAPFLUXNET_FLAGS, QcFlagConfig())
        assert not mask["ta"].any()
        assert list(mask.columns) == ["ta"]

    def test_a_disjoint_time_index_masks_nothing(self, index):
        data = pd.DataFrame({"ta": [1.0] * 4}, index=index)
        elsewhere = pd.date_range("2021-01-01", periods=4, freq="h")
        flags = pd.DataFrame({"ta": ["RANGE_WARN"] * 4}, index=elsewhere)
        mask = build_mask(data, flags, Mechanism.SAPFLUXNET_FLAGS, QcFlagConfig())
        assert not mask["ta"].any()

    def test_sapfluxnet_mechanism_with_no_flag_frame_degrades_to_no_mask(self, index):
        # Declared mechanism and available data disagree. Masking nothing is
        # the safe direction; raising would kill a site for a caller's slip.
        data = pd.DataFrame({"ta": [1.0] * 4}, index=index)
        mask = build_mask(data, None, Mechanism.SAPFLUXNET_FLAGS, QcFlagConfig())
        assert not mask["ta"].any()


class TestMissingSolarTimestamp:
    def test_a_flag_frame_without_solar_timestamp_is_usable(self, index):
        # D-4: env_analyzer accessed flags['solar_TIMESTAMP'] unconditionally.
        # The masking layer must not depend on that column at all.
        data = pd.DataFrame({"ta": [1.0] * 4}, index=index)
        flags = pd.DataFrame({"ta": ["RANGE_WARN", "", "", ""]}, index=index)
        assert "solar_TIMESTAMP" not in flags.columns
        mask = build_mask(data, flags, Mechanism.SAPFLUXNET_FLAGS, QcFlagConfig())
        assert list(mask["ta"]) == [True, False, False, False]


class TestMechanismDetectionNegatives:
    def test_a_missing_site_raises_filenotfound(self, tmp_path):
        # A site that was asked for but is not on disk is a caller bug, not a
        # site without a mechanism. Conflating the two is how D-1 stayed
        # invisible for as long as it did.
        with pytest.raises(FileNotFoundError):
            detect_mechanism("ABSENT", "env", tmp_path)

    def test_garbage_content_reports_none_rather_than_crashing(self, tmp_path):
        # Documenting real behaviour rather than an invented one: pandas parses
        # a line of prose as a single-column header, so detection sees a file
        # with no *_qc columns and says NONE. That is the right answer to the
        # question detection asks -- "which mechanism?" -- and it is not
        # detection's job to decide the file is unusable. The frame having no
        # TIMESTAMP is caught downstream, where that contract lives.
        (tmp_path / "BAD_SITE_env_data.csv").write_text("this is not a csv\n")
        assert detect_mechanism("BAD_SITE", "env", tmp_path) is Mechanism.NONE

    def test_an_empty_file_raises_rather_than_being_classified(self, tmp_path):
        # Zero bytes is not a header. pandas raises EmptyDataError, and letting
        # it through is right: a truncated delivery must not be reported as a
        # site that simply has no flags.
        (tmp_path / "EMPTY_SITE_env_data.csv").write_text("")
        with pytest.raises(pd.errors.EmptyDataError):
            detect_mechanism("EMPTY_SITE", "env", tmp_path)


class TestValidatorNegatives:
    def test_a_log_with_no_rows_validates_to_nothing(self, tmp_path):
        log = tmp_path / "empty.csv"
        log.write_text("entry_id,location,plant_type,column,start_time,end_time,removal_type\n")
        assert validate_log(log, tmp_path) == []

    def test_a_missing_log_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validate_log(tmp_path / "nope.csv", tmp_path)

    def test_a_row_with_no_location_is_reported_not_skipped(self, tmp_path):
        log = tmp_path / "log.csv"
        log.write_text("entry_id,location,plant_type,column,start_time,end_time,removal_type\n1,,,,,,column_entire\n")
        verdicts = validate_log(log, tmp_path)
        assert len(verdicts) == 1
        assert verdicts[0].status is RowStatus.NO_SUCH_FILE


class TestApplyMaskNegatives:
    def test_a_mask_column_absent_from_the_data_is_ignored(self, index):
        data = pd.DataFrame({"ta": [1.0] * 4}, index=index)
        mask = pd.DataFrame({"ta": [False] * 4, "ghost": [True] * 4}, index=index)
        out = apply_mask(data, mask)
        assert list(out.columns) == ["ta"]

    def test_a_mask_with_a_shorter_index_leaves_the_rest_alone(self, index):
        data = pd.DataFrame({"ta": [1.0, 2.0, 3.0, 4.0]}, index=index)
        mask = pd.DataFrame({"ta": [True]}, index=index[:1])
        out = apply_mask(data, mask)
        assert np.isnan(out["ta"].iloc[0])
        assert list(out["ta"].iloc[1:]) == [2.0, 3.0, 4.0]
