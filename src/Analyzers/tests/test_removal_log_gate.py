"""The removal-log validator runs at the entry point, and can actually fail.

Criterion 11: the validator runs as part of the QC entry point, not as a script
someone remembers to invoke.

The second class of test here exists because the first version of this wiring
was silently replaced with a stub that reported ``inert=0`` unconditionally. A
validator that always says "all clear" passes every test that only checks a
clean log, so these feed it a log that is *known* to be broken and require the
count to move.
"""

from __future__ import annotations

import pandas as pd
import pytest

from path_config import PathConfig
from src.Analyzers.sap_analyzer import SapFlowAnalyzer

HEADER = (
    "entry_id,location,plant_type,column,start_time,end_time,issue_category,removal_type,notes,decision_date,reviewer\n"
)


@pytest.fixture()
def corpus(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    frame = pd.DataFrame(
        {
            "TIMESTAMP": pd.date_range("2020-01-01", periods=24, freq="h"),
            "SITE_AA_Qil_Js_1": range(24),
        }
    )
    frame.to_csv(corpus_dir / "SITE_AA_sapf_data.csv", index=False)
    return corpus_dir


@pytest.fixture()
def analyzer(tmp_path):
    paths = PathConfig(scale="sapwood", base_data_dir=str(tmp_path / "data"), base_output_dir=str(tmp_path / "out"))
    return SapFlowAnalyzer(paths=paths)


def write_log(path, rows):
    path.write_text(HEADER + "".join(rows))
    return path


GOOD = "1,SITE_AA,sapf_data,SITE_AA_Qil_Js_1,,,bad,column_entire,note,2026-08-16,me\n"
DEAD_COLUMN = "2,SITE_AA,sapf_data,SITE_AA_SF185_20_NW_1,,,bad,column_entire,note,2026-08-16,me\n"
DEAD_SITE = "3,NO_SUCH,sapf_data,NO_SUCH_Qil_Js_1,,,bad,column_entire,note,2026-08-16,me\n"


class TestValidatorRunsAtTheEntryPoint:
    def test_set_removal_log_validates_without_being_asked(self, analyzer, corpus, tmp_path):
        log = write_log(tmp_path / "log.csv", [GOOD])
        analyzer.set_removal_log(log)
        analyzer.validate_removal_log_against_corpus(corpus_dir=corpus)
        assert analyzer.removal_log_validation["total"] == 1
        assert analyzer.removal_log_validation["inert"] == 0

    def test_a_clean_log_reports_no_inert_rows(self, analyzer, corpus, tmp_path):
        log = write_log(tmp_path / "log.csv", [GOOD, GOOD.replace("1,SITE", "9,SITE")])
        analyzer.set_removal_log(log)
        summary = analyzer.validate_removal_log_against_corpus(corpus_dir=corpus)
        assert summary["acts"] == 2
        assert summary["inert"] == 0


class TestTheGateCanActuallyFail:
    """A validator that cannot report a problem is not a validator."""

    def test_a_dead_column_is_counted_as_inert(self, analyzer, corpus, tmp_path):
        log = write_log(tmp_path / "log.csv", [GOOD, DEAD_COLUMN])
        analyzer.set_removal_log(log)
        summary = analyzer.validate_removal_log_against_corpus(corpus_dir=corpus)
        assert summary["inert"] == 1
        assert summary["by_status"]["no_such_column"] == 1

    def test_a_dead_site_is_counted_as_inert(self, analyzer, corpus, tmp_path):
        log = write_log(tmp_path / "log.csv", [GOOD, DEAD_SITE])
        analyzer.set_removal_log(log)
        summary = analyzer.validate_removal_log_against_corpus(corpus_dir=corpus)
        assert summary["inert"] == 1
        assert summary["by_status"]["no_such_file"] == 1

    def test_the_inert_row_is_named_not_just_counted(self, analyzer, corpus, tmp_path):
        log = write_log(tmp_path / "log.csv", [DEAD_COLUMN])
        analyzer.set_removal_log(log)
        summary = analyzer.validate_removal_log_against_corpus(corpus_dir=corpus)
        assert summary["inert_rows"][0]["entry_id"] == "2"
        assert "SF185" in summary["inert_rows"][0]["column"]

    def test_strict_raises_on_an_inert_row(self, analyzer, corpus, tmp_path):
        log = write_log(tmp_path / "log.csv", [DEAD_COLUMN])
        analyzer.set_removal_log(log)
        with pytest.raises(ValueError):
            analyzer.validate_removal_log_against_corpus(corpus_dir=corpus, strict=True)

    def test_strict_does_not_raise_on_a_clean_log(self, analyzer, corpus, tmp_path):
        log = write_log(tmp_path / "log.csv", [GOOD])
        analyzer.set_removal_log(log)
        analyzer.validate_removal_log_against_corpus(corpus_dir=corpus, strict=True)

    def test_set_removal_log_defaults_to_reporting_not_raising(self, analyzer, corpus, tmp_path):
        # strict must not be tied to validate: turning validation on should not
        # turn a partial corpus into a hard failure.
        log = write_log(tmp_path / "log.csv", [DEAD_SITE])
        analyzer.set_removal_log(log)  # must not raise
        assert analyzer.removal_processor is not None
