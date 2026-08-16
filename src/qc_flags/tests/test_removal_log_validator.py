"""A row that names nothing must be reported, not silently skipped.

Criteria 11 and 13.  Thirty-three of 371 entries were inert when this was first
measured -- seven of them reviewed decisions that had quietly stopped applying
-- and nothing in the pipeline could have said so.  The point of these tests is
that each *kind* of inertness is detected separately: "points at a real file" is
not the same claim as "points at a real column", which is not the same as
"overlaps the data".
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.qc_flags.removal_log_validator import (
    RowStatus,
    summarise,
    validate_log,
    validate_row,
)

HEADER = [
    "entry_id",
    "location",
    "plant_type",
    "column",
    "start_time",
    "end_time",
    "issue_category",
    "removal_type",
    "notes",
    "decision_date",
    "reviewer",
]


@pytest.fixture()
def corpus(tmp_path):
    frame = pd.DataFrame(
        {
            "TIMESTAMP": pd.date_range("2020-01-01", periods=48, freq="h"),
            "SITE_AA_Qil_Js_1": range(48),
            "SITE_AA_Qil_Js_2": range(48),
        }
    )
    frame.to_csv(tmp_path / "SITE_AA_sapf_data.csv", index=False)
    return tmp_path


def row(**kwargs):
    base = {key: "" for key in HEADER}
    base.update(
        {
            "entry_id": "1",
            "location": "SITE_AA",
            "plant_type": "sapf_data",
            "column": "SITE_AA_Qil_Js_1",
            "removal_type": "column_entire",
        }
    )
    base.update(kwargs)
    return base


class TestRowStatus:
    def test_a_good_column_entire_row_acts(self, corpus):
        assert validate_row(row(), corpus).status is RowStatus.ACTS

    def test_a_missing_file_is_reported(self, corpus):
        verdict = validate_row(row(location="NO_SITE"), corpus)
        assert verdict.status is RowStatus.NO_SUCH_FILE

    def test_a_missing_column_is_reported(self, corpus):
        # The sensor-level rows were exactly this shape: a real file, a column
        # name that the B1 sensor collapse had removed.
        verdict = validate_row(row(column="SITE_AA_SF185_20_NW_1"), corpus)
        assert verdict.status is RowStatus.NO_SUCH_COLUMN
        assert "SF185" in verdict.detail

    def test_a_blank_removal_type_is_inert(self, corpus):
        verdict = validate_row(row(removal_type=""), corpus)
        assert verdict.status is RowStatus.INERT_TYPE

    def test_an_unknown_removal_type_is_inert(self, corpus):
        assert validate_row(row(removal_type="probably"), corpus).status is RowStatus.INERT_TYPE

    def test_site_entire_needs_no_column(self, corpus):
        verdict = validate_row(row(removal_type="site_entire", column=""), corpus)
        assert verdict.status is RowStatus.ACTS

    def test_an_overlapping_period_acts(self, corpus):
        verdict = validate_row(
            row(
                removal_type="period",
                start_time=pd.Timestamp("2020-01-01 06:00"),
                end_time=pd.Timestamp("2020-01-01 12:00"),
            ),
            corpus,
        )
        assert verdict.status is RowStatus.ACTS

    def test_a_period_entirely_after_the_data_does_not_act(self, corpus):
        verdict = validate_row(
            row(
                removal_type="period",
                start_time=pd.Timestamp("2021-01-01"),
                end_time=pd.Timestamp("2021-02-01"),
            ),
            corpus,
        )
        assert verdict.status is RowStatus.NO_OVERLAP

    def test_a_period_entirely_before_the_data_does_not_act(self, corpus):
        verdict = validate_row(
            row(
                removal_type="period",
                start_time=pd.Timestamp("2019-01-01"),
                end_time=pd.Timestamp("2019-02-01"),
            ),
            corpus,
        )
        assert verdict.status is RowStatus.NO_OVERLAP

    def test_a_period_with_unparsed_dates_is_inert(self, corpus):
        # The most dangerous shape: it looks complete. Ten CZE_STN rows were
        # here after the reformat, and the only symptom was a count moving.
        verdict = validate_row(
            row(removal_type="period", start_time=pd.NaT, end_time=pd.NaT),
            corpus,
        )
        assert verdict.status is RowStatus.NO_OVERLAP
        assert "unparsed" in verdict.detail

    def test_tz_aware_data_compares_against_a_naive_log(self, corpus, tmp_path):
        # Some corpus files carry a UTC offset in TIMESTAMP; the log's dates
        # never do. Comparing the two raises TypeError, which took out the
        # whole validation run rather than one row.
        aware = pd.DataFrame(
            {
                "TIMESTAMP": pd.date_range("2020-01-01", periods=48, freq="h", tz="UTC"),
                "SITE_TZ_Qil_Js_1": range(48),
            }
        )
        aware.to_csv(tmp_path / "SITE_TZ_sapf_data.csv", index=False)
        verdict = validate_row(
            row(
                location="SITE_TZ",
                column="SITE_TZ_Qil_Js_1",
                removal_type="period",
                start_time=pd.Timestamp("2020-01-01 06:00"),
                end_time=pd.Timestamp("2020-01-01 12:00"),
            ),
            tmp_path,
        )
        assert verdict.status is RowStatus.ACTS

    def test_a_period_partially_overlapping_acts(self, corpus):
        verdict = validate_row(
            row(
                removal_type="period",
                start_time=pd.Timestamp("2019-12-25"),
                end_time=pd.Timestamp("2020-01-01 03:00"),
            ),
            corpus,
        )
        assert verdict.status is RowStatus.ACTS


class TestValidateLog:
    def test_reads_a_file_and_classifies_every_row(self, corpus, tmp_path):
        log = tmp_path / "log.csv"
        pd.DataFrame(
            [
                row(entry_id="1"),
                row(entry_id="2", column="GONE"),
                row(entry_id="3", location="NO_SITE"),
            ]
        ).to_csv(log, index=False)
        verdicts = validate_log(log, corpus)
        assert len(verdicts) == 3
        assert [v.status for v in verdicts] == [
            RowStatus.ACTS,
            RowStatus.NO_SUCH_COLUMN,
            RowStatus.NO_SUCH_FILE,
        ]

    def test_mixed_date_spellings_both_parse(self, corpus, tmp_path):
        # The validator must not reintroduce the whole-column parse it exists
        # to catch.
        log = tmp_path / "log.csv"
        pd.DataFrame(
            [
                row(entry_id="1", removal_type="period", start_time="2020-01-01", end_time="2020-01-02"),
                row(
                    entry_id="2",
                    removal_type="period",
                    start_time="2020-01-01 06:00:00",
                    end_time="2020-01-01 12:00:00",
                ),
            ]
        ).to_csv(log, index=False)
        verdicts = validate_log(log, corpus)
        assert all(v.status is RowStatus.ACTS for v in verdicts)


class TestSummarise:
    def test_counts_and_lists_the_inert_rows(self, corpus, tmp_path):
        log = tmp_path / "log.csv"
        pd.DataFrame([row(entry_id="1"), row(entry_id="2", column="GONE")]).to_csv(log, index=False)
        report = summarise(validate_log(log, corpus))
        assert report["total"] == 2
        assert report["acts"] == 1
        assert report["inert"] == 1
        assert report["inert_rows"][0]["entry_id"] == "2"
        assert report["inert_rows"][0]["status"] == "no_such_column"

    def test_a_fully_valid_log_reports_zero_inert(self, corpus, tmp_path):
        log = tmp_path / "log.csv"
        pd.DataFrame([row(entry_id="1"), row(entry_id="2", column="SITE_AA_Qil_Js_2")]).to_csv(log, index=False)
        report = summarise(validate_log(log, corpus))
        assert report["inert"] == 0
