"""Applying clock offsets at the Stage-3 load boundary (plan WP-A2; failure modes F2, F3, F6, F9)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.clock_offsets.apply import correct_clock, correct_loaded_frames, site_code
from src.clock_offsets.registry import ClockOffset

HOUR = pd.Timedelta(hours=1)
STAMPS = ("2008-06-15 10:00", "2008-06-15 10:30", "2008-06-15 11:00")


def applied(offset, start=None, end=None):
    return ClockOffset(
        site="SITE",
        component="env",
        offset_hours=offset,
        status="applied",
        start=start,
        end=end,
        evidence="synthetic",
    )


def frame(stamps, tz="UTC"):
    index = pd.DatetimeIndex(pd.to_datetime(list(stamps)))
    labels = index.tz_localize(tz) if tz else index
    return pd.DataFrame(
        {
            "TIMESTAMP": labels,
            "solar_TIMESTAMP": labels + pd.Timedelta(minutes=97),
            "sw_in": np.arange(len(labels), dtype=float),
        }
    )


class TestSiteCode:
    @pytest.mark.parametrize(
        "name, code",
        [
            ("USA_ORN_ST1_AMB_env_data.csv", "USA_ORN_ST1_AMB"),
            ("USA_ORN_ST2_AMB_sapf_data.csv", "USA_ORN_ST2_AMB"),
            ("CHE_DAV_env_flags.csv", "CHE_DAV"),
            ("CHE_DAV_SEE_sapf_flags.csv", "CHE_DAV_SEE"),
            ("FIN_PET_env_data_outliers_removed.csv", "FIN_PET"),
            ("FIN_PET_sapf_data_outliers_removed.csv", "FIN_PET"),
        ],
    )
    def test_the_full_code_comes_from_an_exact_suffix(self, name, code):
        assert site_code(Path("/corpus") / name) == code
        assert site_code(name) == code

    @pytest.mark.parametrize(
        "name", ["FIN_PET_site_md.csv", "FIN_PET_env_md.csv", "_env_data.csv", "FIN_PET_env_data.txt"]
    )
    def test_a_name_without_a_data_suffix_raises(self, name):
        with pytest.raises(ValueError):
            site_code(name)


class TestCorrectClock:
    def test_a_late_label_moves_back_by_the_offset(self):
        source = frame(STAMPS)
        out = correct_clock(source, [applied(5.0)])
        pd.testing.assert_series_equal(out["TIMESTAMP"], source["TIMESTAMP"] - 5 * HOUR)
        pd.testing.assert_series_equal(out["solar_TIMESTAMP"], source["solar_TIMESTAMP"] - 5 * HOUR)
        pd.testing.assert_series_equal(out["sw_in"], source["sw_in"])

    def test_an_early_label_moves_forward(self):
        source = frame(STAMPS)
        out = correct_clock(source, [applied(-6.0)])
        pd.testing.assert_series_equal(out["TIMESTAMP"], source["TIMESTAMP"] + 6 * HOUR)

    def test_a_timestamp_index_moves_and_keeps_its_name(self):
        source = frame(STAMPS).set_index("TIMESTAMP")
        out = correct_clock(source, [applied(3.5)])
        pd.testing.assert_index_equal(out.index, source.index - pd.Timedelta(hours=3.5))
        assert out.index.name == "TIMESTAMP"
        # The column's own index is the corrected TIMESTAMP, so compare values, not index labels.
        pd.testing.assert_series_equal(
            out["solar_TIMESTAMP"], source["solar_TIMESTAMP"] - pd.Timedelta(hours=3.5), check_index=False
        )

    def test_only_rows_inside_the_half_open_period_move(self):
        source = frame(STAMPS)
        period = applied(2.0, start="2008-06-15T10:30:00Z", end="2008-06-15T11:00:00Z")
        out = correct_clock(source, [period])
        assert (out["TIMESTAMP"] != source["TIMESTAMP"]).tolist() == [False, True, False]

    def test_membership_is_judged_on_the_original_label(self):
        # Two adjoining periods: a row moved by the first entry must not be picked up by the second.
        source = frame(("2009-01-01 05:00", "2009-01-01 07:00"))
        entries = [
            applied(-6.0, end="2009-01-01T06:00:00Z"),
            applied(-5.25, start="2009-01-01T06:00:00Z"),
        ]
        out = correct_clock(source, entries)
        moved = (out["TIMESTAMP"] - source["TIMESTAMP"]).tolist()
        assert moved == [pd.Timedelta(hours=6), pd.Timedelta(hours=5.25)]

    def test_timezone_and_dtype_are_preserved(self):
        source = frame(STAMPS)
        out = correct_clock(source, [applied(5.0)])
        assert out["TIMESTAMP"].dtype == source["TIMESTAMP"].dtype
        assert str(out["TIMESTAMP"].dt.tz) == "UTC"

    def test_naive_labels_are_read_as_utc_and_stay_naive(self):
        source = frame(STAMPS, tz=None)
        out = correct_clock(source, [applied(1.0, start="2008-06-15T10:30:00Z")])
        assert out["TIMESTAMP"].dt.tz is None
        assert (out["TIMESTAMP"] != source["TIMESTAMP"]).tolist() == [False, True, True]

    def test_nat_stays_nat(self):
        source = frame(STAMPS)
        source.loc[1, "TIMESTAMP"] = pd.NaT
        out = correct_clock(source, [applied(5.0)])
        assert pd.isna(out.loc[1, "TIMESTAMP"])
        assert out.loc[0, "TIMESTAMP"] == source.loc[0, "TIMESTAMP"] - 5 * HOUR

    def test_the_input_frame_is_not_modified(self):
        source = frame(STAMPS).set_index("TIMESTAMP")
        before = source.copy(deep=True)
        correct_clock(source, [applied(5.0)])
        pd.testing.assert_frame_equal(source, before)

    def test_no_entries_returns_an_equal_copy(self):
        source = frame(STAMPS)
        out = correct_clock(source, [])
        pd.testing.assert_frame_equal(out, source)
        assert out is not source

    def test_a_missing_optional_time_column_is_skipped(self):
        source = frame(STAMPS).drop(columns=["solar_TIMESTAMP"])
        out = correct_clock(source, [applied(5.0)])
        assert list(out.columns) == ["TIMESTAMP", "sw_in"]
        pd.testing.assert_series_equal(out["TIMESTAMP"], source["TIMESTAMP"] - 5 * HOUR)

    def test_a_frame_without_timestamp_raises(self):
        with pytest.raises(ValueError):
            correct_clock(frame(STAMPS).drop(columns=["TIMESTAMP"]), [applied(5.0)])

    def test_an_unparsed_time_column_raises(self):
        source = frame(STAMPS)
        source["solar_TIMESTAMP"] = source["solar_TIMESTAMP"].astype(str)
        with pytest.raises(TypeError):
            correct_clock(source, [applied(5.0)])

    def test_an_entry_that_is_not_applied_is_refused(self):
        acknowledged = ClockOffset(site="SITE", component="env", offset_hours=5.0, status="acknowledged")
        with pytest.raises(ValueError):
            correct_clock(frame(STAMPS), [acknowledged])

    def test_flags_and_data_with_the_same_labels_stay_aligned(self):
        data = frame(STAMPS)
        flags = frame(STAMPS).rename(columns={"sw_in": "sw_in_flag"})
        out_data = correct_clock(data, [applied(5.0)])
        out_flags = correct_clock(flags, [applied(5.0)])
        pd.testing.assert_series_equal(out_data["TIMESTAMP"], out_flags["TIMESTAMP"])

    @pytest.mark.parametrize("label, hour", [("2008-06-15 10:15", 10), ("2008-06-15 10:10", 9)])
    def test_a_quarter_hour_offset_lands_in_the_right_hour(self, label, hour):
        out = correct_clock(frame((label,)), [applied(0.25)])
        assert out["TIMESTAMP"].dt.floor("h").dt.hour.iloc[0] == hour


class TestCorrectLoadedFrames:
    REGISTRY = (
        ClockOffset(site="USA_ORN_ST2_AMB", component="env", offset_hours=5.0, status="applied", evidence="synthetic"),
    )

    def test_data_and_flags_move_together(self):
        data, flags = frame(STAMPS), frame(STAMPS)
        path = Path("/corpus/USA_ORN_ST2_AMB_env_data.csv")
        out_data, out_flags = correct_loaded_frames(path, "env", data, flags, self.REGISTRY)
        pd.testing.assert_series_equal(out_data["TIMESTAMP"], data["TIMESTAMP"] - 5 * HOUR)
        pd.testing.assert_series_equal(out_flags["TIMESTAMP"], out_data["TIMESTAMP"])

    def test_a_site_without_entries_gets_its_own_objects_back(self):
        data, flags = frame(STAMPS), frame(STAMPS)
        path = Path("/corpus/USA_ORN_ST1_AMB_env_data.csv")
        out_data, out_flags = correct_loaded_frames(path, "env", data, flags, self.REGISTRY)
        assert out_data is data
        assert out_flags is flags

    def test_missing_flags_stay_missing(self):
        path = Path("/corpus/USA_ORN_ST2_AMB_env_data.csv")
        _out_data, out_flags = correct_loaded_frames(path, "env", frame(STAMPS), None, self.REGISTRY)
        assert out_flags is None

    def test_an_entry_for_the_other_component_moves_nothing(self):
        data = frame(STAMPS)
        path = Path("/corpus/USA_ORN_ST2_AMB_sapf_data.csv")
        out_data, _out_flags = correct_loaded_frames(path, "sap", data, None, self.REGISTRY)
        assert out_data is data

    def test_a_file_name_without_a_site_suffix_raises(self):
        with pytest.raises(ValueError):
            correct_loaded_frames(Path("/corpus/USA_ORN_ST2_AMB_env_md.csv"), "env", frame(STAMPS), None, self.REGISTRY)
