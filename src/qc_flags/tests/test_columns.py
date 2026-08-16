"""Column classification and flag-column resolution.

Two defects live here.

D-3: `*_qc` columns are not excluded from the env variable set, so they get
z-scored, outlier-filtered and daily-resampled as if they were weather.  Stage 4
then joins the whole frame wholesale, so they would arrive in `merged_data.csv`
as feature-shaped columns.

D-5: flag columns are matched by substring and the first hit taken, so a column
that is a prefix of a later one resolves to the wrong tree.  The corpus already
contains the hazardous ordering (`SWE_NOR_Psy_Jt_10` before `_Jt_1`); only the
absence of a flag file at that site hides it.  Criterion 10.
"""

import pandas as pd
import pytest

from src.qc_flags import columns as c


class TestSplitQcColumns:
    def test_separates_qc_from_variables(self):
        cols = ["ta", "ta_qc", "vpd", "vpd_qc", "sw_in", "sw_in_qc"]
        variables, qc = c.split_qc_columns(cols)
        assert variables == ["ta", "vpd", "sw_in"]
        assert qc == ["ta_qc", "vpd_qc", "sw_in_qc"]

    def test_timestamps_and_coordinates_are_neither(self):
        cols = ["TIMESTAMP", "solar_TIMESTAMP", "lat", "long", "ta", "ta_qc"]
        variables, qc = c.split_qc_columns(cols)
        assert variables == ["ta"]
        assert qc == ["ta_qc"]

    def test_no_qc_columns_yields_empty_list(self):
        variables, qc = c.split_qc_columns(["ta", "vpd"])
        assert variables == ["ta", "vpd"]
        assert qc == []

    def test_qc_column_without_its_variable_is_still_qc(self):
        # Edge case: a QC column whose variable was dropped upstream. It must
        # not silently become a variable just because its partner is missing.
        variables, qc = c.split_qc_columns(["ta", "pa_qc"])
        assert variables == ["ta"]
        assert qc == ["pa_qc"]

    def test_accepts_a_dataframe(self):
        df = pd.DataFrame(columns=["TIMESTAMP", "ta", "ta_qc"])
        variables, qc = c.split_qc_columns(df)
        assert variables == ["ta"]
        assert qc == ["ta_qc"]

    def test_order_is_preserved(self):
        variables, _ = c.split_qc_columns(["vpd", "ta", "sw_in"])
        assert variables == ["vpd", "ta", "sw_in"]


class TestQcColumnFor:
    def test_names_the_partner_column(self):
        assert c.qc_column_for("ta") == "ta_qc"

    def test_variable_for_inverts_it(self):
        assert c.variable_for("ta_qc") == "ta"

    def test_variable_for_rejects_a_non_qc_name(self):
        with pytest.raises(ValueError):
            c.variable_for("ta")


class TestResolveFlagColumn:
    """The D-5 fix: exact equality, never containment."""

    def test_exact_match(self):
        assert c.resolve_flag_column("ta", ["ta", "vpd"]) == "ta"

    def test_returns_none_when_absent(self):
        assert c.resolve_flag_column("ta", ["vpd", "sw_in"]) is None

    def test_prefix_column_does_not_match_a_longer_name(self):
        # The substring idiom would return SWE_NOR_Psy_Jt_10 here.
        assert c.resolve_flag_column("SWE_NOR_Psy_Jt_1", ["SWE_NOR_Psy_Jt_10"]) is None

    def test_swe_nor_ordering_resolves_to_the_right_tree(self):
        # Criterion 10(b): the fixture is the ordering that actually exists on
        # disk in SWE_NOR_sapf_data.csv, so this cannot drift into testing a
        # case the corpus never contains.
        flag_cols = [
            "SWE_NOR_Psy_Jt_10",
            "SWE_NOR_Psy_Jt_11",
            "SWE_NOR_Psy_Jt_12",
            "SWE_NOR_Psy_Jt_1",
        ]
        assert c.resolve_flag_column("SWE_NOR_Psy_Jt_1", flag_cols) == "SWE_NOR_Psy_Jt_1"
        assert c.resolve_flag_column("SWE_NOR_Psy_Jt_10", flag_cols) == "SWE_NOR_Psy_Jt_10"

    def test_every_tree_resolves_to_itself_over_the_whole_ordering(self):
        flag_cols = [f"SWE_NOR_Psy_Jt_{i}" for i in [10, 11, 12, 1, 2, 3]]
        for col in flag_cols:
            assert c.resolve_flag_column(col, flag_cols) == col

    def test_accepts_a_dataframe_columns_index(self):
        flags = pd.DataFrame(columns=["ta", "vpd"])
        assert c.resolve_flag_column("ta", flags.columns) == "ta"
