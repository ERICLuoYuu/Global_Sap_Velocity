"""The removal log loader must not let one row's format silence another's.

Acceptance criterion 11c.  `load_removal_log` parsed `start_time`/`end_time`
with a single whole-column `pd.to_datetime`.  pandas 2.x infers one format from
the first element, so a column mixing `2011-09-21` with `2016-05-04 12:41:00`
coerces the second spelling to `NaT` -- and `errors="coerce"` means it does so
without a word.

That cost ten `CZE_STN` rows the moment the log was reformatted.  Before the
reformat they survived only because the column also contained semicolons, which
sent the loader down a value-by-value path instead.  The semicolons were
load-bearing, and tidying them broke the rows they were accidentally protecting.

The file is uniform today.  These tests are about the next hand-edit.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.Analyzers.mannual_removal_processor import RemovalLogProcessor

HEADER = (
    "entry_id,location,plant_type,column,start_time,end_time,issue_category,removal_type,notes,decision_date,reviewer\n"
)


def write_log(path: Path, rows) -> Path:
    path.write_text(HEADER + "".join(rows))
    return path


@pytest.fixture()
def processor():
    return RemovalLogProcessor()


class TestMixedDatetimeSpellings:
    def test_both_spellings_survive_in_one_column(self, tmp_path, processor):
        # The exact shape that cost the CZE_STN rows: a bare date first, so the
        # inferred format is date-only, then a datetime that does not fit it.
        log = write_log(
            tmp_path / "removal_log.csv",
            [
                "1,ESP_TIL,sapf_data,ESP_TIL_Qil_Js_1,2020-05-01,2020-05-10,bad,period,note,2026-08-16,me\n",
                "2,CZE_STN,sapf_data,CZE_STN_Pab_Js_1,2020-05-01 12:41:00,2020-05-10 06:00:00,bad,period,note,2026-08-16,me\n",
            ],
        )
        processor.load_removal_log(log, validate=False)
        periods = [e for e in processor.removal_entries if e.removal_type == "period"]
        assert len(periods) == 2
        assert all(pd.notna(e.start_time) for e in periods), "a spelling was coerced to NaT"

    def test_the_reverse_order_also_survives(self, tmp_path, processor):
        # Whichever row comes first must not decide the other's fate.
        log = write_log(
            tmp_path / "removal_log.csv",
            [
                "1,CZE_STN,sapf_data,CZE_STN_Pab_Js_1,2020-05-01 12:41:00,2020-05-10 06:00:00,bad,period,note,2026-08-16,me\n",
                "2,ESP_TIL,sapf_data,ESP_TIL_Qil_Js_1,2020-05-01,2020-05-10,bad,period,note,2026-08-16,me\n",
            ],
        )
        processor.load_removal_log(log, validate=False)
        periods = [e for e in processor.removal_entries if e.removal_type == "period"]
        assert len(periods) == 2
        assert all(pd.notna(e.start_time) for e in periods)

    def test_times_are_parsed_not_truncated_to_midnight(self, tmp_path, processor):
        log = write_log(
            tmp_path / "removal_log.csv",
            [
                "1,ESP_TIL,sapf_data,ESP_TIL_Qil_Js_1,2020-05-01,2020-05-10,bad,period,note,2026-08-16,me\n",
                "2,CZE_STN,sapf_data,CZE_STN_Pab_Js_1,2020-05-01 12:41:00,2020-05-10 06:00:00,bad,period,note,2026-08-16,me\n",
            ],
        )
        processor.load_removal_log(log, validate=False)
        stn = [e for e in processor.removal_entries if e.location == "CZE_STN"][0]
        assert stn.start_time == pd.Timestamp("2020-05-01 12:41:00")

    def test_semicolon_rows_still_expand(self, tmp_path, processor):
        # The value-by-value path must keep working; it is the one that was
        # accidentally protecting the mixed spellings before.
        log = write_log(
            tmp_path / "removal_log.csv",
            [
                "1,ESP_TIL,sapf_data,ESP_TIL_Qil_Js_1,2020-05-01;2020-07-01,2020-05-10;2020-07-10,bad,period,note,2026-08-16,me\n",
            ],
        )
        processor.load_removal_log(log, validate=False)
        assert len([e for e in processor.removal_entries if e.removal_type == "period"]) == 2

    def test_an_unparseable_date_does_not_take_its_neighbours_with_it(self, tmp_path, processor):
        log = write_log(
            tmp_path / "removal_log.csv",
            [
                "1,ESP_TIL,sapf_data,ESP_TIL_Qil_Js_1,not-a-date,2020-05-10,bad,period,note,2026-08-16,me\n",
                "2,CZE_STN,sapf_data,CZE_STN_Pab_Js_1,2020-05-01 12:41:00,2020-05-10 06:00:00,bad,period,note,2026-08-16,me\n",
            ],
        )
        processor.load_removal_log(log, validate=False)
        stn = [e for e in processor.removal_entries if e.location == "CZE_STN"]
        assert len(stn) == 1
        assert pd.notna(stn[0].start_time)


class TestRealLogParsesCompletely:
    """The delivered file, not a fixture."""

    @pytest.fixture()
    def real_log(self):
        path = Path(__file__).resolve().parents[3] / "removal_log.csv"
        if not path.exists():
            pytest.skip("removal_log.csv not present")
        return path

    def test_no_period_entry_loses_its_dates(self, real_log, processor):
        processor.load_removal_log(real_log, validate=False)
        periods = [e for e in processor.removal_entries if e.removal_type == "period"]
        natted = [e for e in periods if pd.isna(e.start_time) or pd.isna(e.end_time)]
        assert natted == [], f"period entries with unparsed dates: {[e.entry_id for e in natted]}"

    def test_period_count_matches_the_validated_figure(self, real_log, processor):
        # 154 of 154 parsed and overlapped their data in job 46188083.
        processor.load_removal_log(real_log, validate=False)
        periods = [e for e in processor.removal_entries if e.removal_type == "period"]
        assert len(periods) == 154

    def test_no_entry_has_a_blank_removal_type(self, real_log, processor):
        # Criterion 11b: six CZE_LIZ rows were parsed and never actioned.
        processor.load_removal_log(real_log, validate=False)
        blank = [e for e in processor.removal_entries if not e.removal_type]
        assert blank == []
