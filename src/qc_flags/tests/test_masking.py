"""Building the mask -- one algorithm, three mechanisms.

`build_mask` returns a boolean frame aligned to the data's index and variable
columns, where True means "set this cell to NaN".  It never mutates its input:
deciding what to mask and applying the mask are separate steps, so the decision
can be counted and reported before anything is destroyed.
"""

import numpy as np
import pandas as pd
import pytest

from src.qc_flags.config import QcFlagConfig
from src.qc_flags.masking import apply_mask, build_mask
from src.qc_flags.mechanisms import Mechanism


@pytest.fixture()
def index():
    return pd.date_range("2020-06-01", periods=6, freq="h", name="TIMESTAMP")


@pytest.fixture()
def env_data(index):
    return pd.DataFrame(
        {
            "ta": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "vpd": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "sw_in": [0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
        },
        index=index,
    )


class TestSapfluxnetMechanism:
    def test_range_warn_is_masked(self, env_data, index):
        flags = pd.DataFrame(
            {
                "ta": ["", "RANGE_WARN", "", "", "", ""],
                "vpd": [""] * 6,
                "sw_in": [""] * 6,
            },
            index=index,
        )
        mask = build_mask(env_data, flags, Mechanism.SAPFLUXNET_FLAGS, QcFlagConfig())
        assert list(mask["ta"]) == [False, True, False, False, False, False]
        assert not mask["vpd"].any()

    def test_out_warn_is_not_masked(self, env_data, index):
        # Criterion 15(b) in miniature: a cell carrying OUT_WARN alone survives.
        flags = pd.DataFrame({c: ["OUT_WARN"] * 6 for c in env_data.columns}, index=index)
        mask = build_mask(env_data, flags, Mechanism.SAPFLUXNET_FLAGS, QcFlagConfig())
        assert not mask.any().any()

    def test_variable_with_no_flag_column_is_untouched(self, env_data, index):
        flags = pd.DataFrame({"ta": ["RANGE_WARN"] * 6}, index=index)
        mask = build_mask(env_data, flags, Mechanism.SAPFLUXNET_FLAGS, QcFlagConfig())
        assert mask["ta"].all()
        assert not mask["vpd"].any()

    def test_flags_are_aligned_by_timestamp_not_by_position(self, env_data, index):
        # The flag frame is reversed. A positional zip would mask the wrong hour.
        flags = pd.DataFrame(
            {"ta": ["", "", "", "", "", "RANGE_WARN"], "vpd": [""] * 6, "sw_in": [""] * 6},
            index=index,
        ).iloc[::-1]
        mask = build_mask(env_data, flags, Mechanism.SAPFLUXNET_FLAGS, QcFlagConfig())
        assert list(mask["ta"]) == [False, False, False, False, False, True]

    def test_flag_rows_missing_for_some_timestamps_default_to_unmasked(self, env_data, index):
        flags = pd.DataFrame({"ta": ["RANGE_WARN"], "vpd": [""], "sw_in": [""]}, index=index[:1])
        mask = build_mask(env_data, flags, Mechanism.SAPFLUXNET_FLAGS, QcFlagConfig())
        assert list(mask["ta"]) == [True, False, False, False, False, False]


class TestQcColumnMechanism:
    @pytest.fixture()
    def icos_data(self, index):
        return pd.DataFrame(
            {
                "ta": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                "ta_qc": [0, 1, 2, 0, 1, 0],
                "vpd": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "vpd_qc": [0, 0, 0, 2, 2, 0],
            },
            index=index,
        )

    def test_default_threshold_masks_one_and_two(self, icos_data):
        mask = build_mask(icos_data, None, Mechanism.QC_COLUMNS, QcFlagConfig())
        assert list(mask["ta"]) == [False, True, True, False, True, False]
        assert list(mask["vpd"]) == [False, False, False, True, True, False]

    def test_threshold_two_keeps_level_one(self, icos_data):
        mask = build_mask(icos_data, None, Mechanism.QC_COLUMNS, QcFlagConfig(qc_threshold=2))
        assert list(mask["ta"]) == [False, False, True, False, False, False]

    def test_qc_columns_are_not_themselves_maskable_variables(self, icos_data):
        mask = build_mask(icos_data, None, Mechanism.QC_COLUMNS, QcFlagConfig())
        assert "ta_qc" not in mask.columns
        assert list(mask.columns) == ["ta", "vpd"]

    def test_variable_without_a_qc_partner_is_untouched(self, index):
        data = pd.DataFrame({"ta": [1.0] * 6, "ta_qc": [2] * 6, "ws": [3.0] * 6}, index=index)
        mask = build_mask(data, None, Mechanism.QC_COLUMNS, QcFlagConfig())
        assert mask["ta"].all()
        assert not mask["ws"].any()

    def test_float_qc_values_compare_correctly(self, index):
        # pandas reads a QC column with any NaN as float64; 1.0 must behave as 1.
        data = pd.DataFrame({"ta": [1.0] * 3, "ta_qc": [0.0, 1.0, 2.0]}, index=index[:3])
        mask = build_mask(data, None, Mechanism.QC_COLUMNS, QcFlagConfig())
        assert list(mask["ta"]) == [False, True, True]

    def test_missing_qc_value_is_masked_conservatively(self, index):
        # An absent QC verdict is not evidence of quality. Mask it.
        data = pd.DataFrame({"ta": [1.0] * 3, "ta_qc": [0.0, np.nan, 0.0]}, index=index[:3])
        mask = build_mask(data, None, Mechanism.QC_COLUMNS, QcFlagConfig())
        assert list(mask["ta"]) == [False, True, False]

    def test_non_numeric_qc_value_is_masked_conservatively(self, index):
        data = pd.DataFrame({"ta": [1.0] * 3, "ta_qc": ["0", "bogus", "2"]}, index=index[:3])
        mask = build_mask(data, None, Mechanism.QC_COLUMNS, QcFlagConfig())
        assert list(mask["ta"]) == [False, True, True]


class TestPropagationToDerivedVariables:
    """Criterion 6.  `ppfd_in` and `rh` are computed, not measured."""

    def test_ppfd_in_masked_exactly_where_sw_in_is(self, index):
        data = pd.DataFrame(
            {
                "sw_in": [100.0] * 6,
                "sw_in_qc": [0, 1, 0, 2, 0, 0],
                "ppfd_in": [200.0] * 6,
            },
            index=index,
        )
        mask = build_mask(data, None, Mechanism.QC_COLUMNS, QcFlagConfig())
        # Set equality, not count equality.
        assert set(mask.index[mask["ppfd_in"]]) == set(mask.index[mask["sw_in"]])

    def test_rh_masked_where_either_ta_or_vpd_is(self, index):
        data = pd.DataFrame(
            {
                "ta": [1.0] * 6,
                "ta_qc": [0, 1, 0, 0, 0, 0],
                "vpd": [1.0] * 6,
                "vpd_qc": [0, 0, 0, 2, 0, 0],
                "rh": [50.0] * 6,
            },
            index=index,
        )
        mask = build_mask(data, None, Mechanism.QC_COLUMNS, QcFlagConfig())
        expected = set(mask.index[mask["ta"]]) | set(mask.index[mask["vpd"]])
        assert set(mask.index[mask["rh"]]) == expected

    def test_derived_variable_absent_is_not_an_error(self, index):
        data = pd.DataFrame({"sw_in": [1.0] * 3, "sw_in_qc": [1, 0, 0]}, index=index[:3])
        mask = build_mask(data, None, Mechanism.QC_COLUMNS, QcFlagConfig())
        assert "ppfd_in" not in mask.columns

    def test_propagation_does_not_unmask_a_derived_variables_own_flag(self, index):
        # ppfd_in has no _qc column of its own in the corpus, but if a future
        # delivery adds one, propagation must union with it, never replace it.
        data = pd.DataFrame(
            {
                "sw_in": [1.0] * 3,
                "sw_in_qc": [1, 0, 0],
                "ppfd_in": [1.0] * 3,
                "ppfd_in_qc": [0, 0, 2],
            },
            index=index[:3],
        )
        mask = build_mask(data, None, Mechanism.QC_COLUMNS, QcFlagConfig())
        assert list(mask["ppfd_in"]) == [True, False, True]

    def test_propagation_does_not_apply_to_the_sapfluxnet_mechanism(self, index):
        # Measured, not assumed: 19 of 176 external sites carry `ppfd_in` on
        # rows where `sw_in` is absent (154,155 rows in total; USA_DUK_HAR
        # alone 36,047 of 70,128 present), and 35 of 200 do the same for `rh`.
        # A value present where its source is missing was not computed from it,
        # so at these sites both are independent measurements carrying their
        # own flag columns. The ICOS sites have zero such rows, which is why
        # propagation is right there and wrong here.
        data = pd.DataFrame({"sw_in": [1.0] * 3, "ppfd_in": [1.0] * 3}, index=index[:3])
        flags = pd.DataFrame({"sw_in": ["RANGE_WARN", "", ""], "ppfd_in": ["", "", ""]}, index=index[:3])
        mask = build_mask(data, flags, Mechanism.SAPFLUXNET_FLAGS, QcFlagConfig())
        assert list(mask["sw_in"]) == [True, False, False]
        assert list(mask["ppfd_in"]) == [False, False, False]

    def test_ppfd_in_keeps_its_own_flag_under_the_sapfluxnet_mechanism(self, index):
        # The flip side: not propagating must not mean ignoring ppfd_in's own
        # flags, which is the failure mode a blanket "no propagation" would hit.
        data = pd.DataFrame({"sw_in": [1.0] * 3, "ppfd_in": [1.0] * 3}, index=index[:3])
        flags = pd.DataFrame({"sw_in": ["", "", ""], "ppfd_in": ["", "RANGE_WARN", ""]}, index=index[:3])
        mask = build_mask(data, flags, Mechanism.SAPFLUXNET_FLAGS, QcFlagConfig())
        assert list(mask["ppfd_in"]) == [False, True, False]


class TestNoMechanism:
    def test_returns_an_all_false_mask_of_the_right_shape(self, env_data):
        mask = build_mask(env_data, None, Mechanism.NONE, QcFlagConfig())
        assert not mask.any().any()
        assert list(mask.columns) == list(env_data.columns)
        assert list(mask.index) == list(env_data.index)

    def test_a_site_with_no_mechanism_is_not_an_error(self, env_data):
        # D-1: this is the case that has been killing all 24 internal sites.
        build_mask(env_data, None, Mechanism.NONE, QcFlagConfig())


class TestApplyMask:
    def test_does_not_mutate_the_input(self, env_data, index):
        mask = pd.DataFrame({c: [False] * 6 for c in env_data.columns}, index=index)
        mask.loc[index[0], "ta"] = True
        before = env_data.copy()
        apply_mask(env_data, mask)
        pd.testing.assert_frame_equal(env_data, before)

    def test_sets_masked_cells_to_nan(self, env_data, index):
        mask = pd.DataFrame({c: [False] * 6 for c in env_data.columns}, index=index)
        mask.loc[index[0], "ta"] = True
        out = apply_mask(env_data, mask)
        assert np.isnan(out.loc[index[0], "ta"])
        assert out.loc[index[1], "ta"] == 11.0

    def test_columns_outside_the_mask_are_carried_through_untouched(self, index):
        data = pd.DataFrame({"ta": [1.0] * 3, "ta_qc": [0, 1, 2]}, index=index[:3])
        mask = pd.DataFrame({"ta": [False, True, False]}, index=index[:3])
        out = apply_mask(data, mask)
        assert list(out["ta_qc"]) == [0, 1, 2]
