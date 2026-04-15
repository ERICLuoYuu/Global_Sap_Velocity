# ruff: noqa: N999
"""Unit tests for src.Analyzers.calibration_corrector (Flo 2019 SFD correction).

The calibration module multiplies SFD columns of non-calibrated HD/CHD plants
by per-method multipliers derived from Flo et al. 2019 Table 1:

    HD  (TD)  multiplier = exp(0.519) ≈ 1.6804
    CHD (TTD) multiplier = exp(0.493) ≈ 1.6373

Tests verify: per-method multiplier application, calibration-flag respect,
non-Dissipation method pass-through, timestamp preservation, immutability,
NaN handling, audit record shape, None/empty plant_md no-op (ICOS
pass-through), missing-plant-md-row warning, and custom multiplier override.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import pytest

from src.Analyzers.calibration_corrector import (
    FLO2019_CHD_MULTIPLIER,
    FLO2019_HD_MULTIPLIER,
    FLO2019_MULTIPLIERS,
    _is_calibrated,
    apply_flo2019_correction,
)

# --------------------------------------------------------------------------
# Constant sanity checks
# --------------------------------------------------------------------------


class TestFlo2019Constants:
    def test_hd_multiplier_matches_flo2019_table1(self) -> None:
        """HD multiplier = exp(0.519), derived from Flo 2019 Ln-Ratio for TD."""
        assert math.isclose(FLO2019_HD_MULTIPLIER, math.exp(0.519), rel_tol=1e-4)

    def test_chd_multiplier_matches_flo2019_table1(self) -> None:
        """CHD multiplier = exp(0.493), derived from Flo 2019 Ln-Ratio for TTD."""
        assert math.isclose(FLO2019_CHD_MULTIPLIER, math.exp(0.493), rel_tol=1e-4)

    def test_multipliers_dict_has_hd_and_chd_only(self) -> None:
        """Only the Dissipation-family methods get corrected."""
        assert set(FLO2019_MULTIPLIERS.keys()) == {"HD", "CHD"}

    def test_multipliers_dict_maps_to_constants(self) -> None:
        assert FLO2019_MULTIPLIERS["HD"] == FLO2019_HD_MULTIPLIER
        assert FLO2019_MULTIPLIERS["CHD"] == FLO2019_CHD_MULTIPLIER


class TestIsCalibrated:
    """Defensive checks on the _is_calibrated normaliser. SAPFLUXNET ships
    pl_sens_calib as strings ('TRUE'/'FALSE'/'NA') in v0.1.5, but a future
    CSV reader could produce Python bool, int, or NaN — make sure all paths
    behave."""

    def test_string_true_uppercase(self) -> None:
        assert _is_calibrated("TRUE") is True

    def test_string_true_titlecase(self) -> None:
        assert _is_calibrated("True") is True

    def test_string_true_lowercase(self) -> None:
        assert _is_calibrated("true") is True

    def test_string_false(self) -> None:
        assert _is_calibrated("FALSE") is False

    def test_string_na(self) -> None:
        assert _is_calibrated("NA") is False

    def test_python_bool_true(self) -> None:
        assert _is_calibrated(True) is True

    def test_python_bool_false(self) -> None:
        assert _is_calibrated(False) is False

    def test_int_one(self) -> None:
        assert _is_calibrated(1) is True

    def test_int_zero(self) -> None:
        assert _is_calibrated(0) is False

    def test_none(self) -> None:
        assert _is_calibrated(None) is False

    def test_nan(self) -> None:
        assert _is_calibrated(float("nan")) is False


# --------------------------------------------------------------------------
# Core correction behaviour
# --------------------------------------------------------------------------


class TestApplyFlo2019Correction:
    def test_hd_noncalib_multiplied_by_hd_multiplier(
        self, sapf_wide: pd.DataFrame, plant_md: pd.DataFrame, base_value: float
    ) -> None:
        """P01 (HD, FALSE) and P02 (HD, NA) should be × FLO2019_HD_MULTIPLIER."""
        original_p01 = sapf_wide["P01"].to_numpy(copy=True)
        original_p02 = sapf_wide["P02"].to_numpy(copy=True)

        out, _ = apply_flo2019_correction(sapf_wide, plant_md)

        # P01 row 0 is NaN (seeded in the fixture); compare non-NaN rows only.
        np.testing.assert_allclose(
            out["P01"].to_numpy()[1:],
            original_p01[1:] * FLO2019_HD_MULTIPLIER,
            rtol=1e-9,
        )
        assert pd.isna(out["P01"].iloc[0])

        np.testing.assert_allclose(
            out["P02"].to_numpy(),
            original_p02 * FLO2019_HD_MULTIPLIER,
            rtol=1e-9,
        )
        assert base_value > 0  # sanity — fixture baseline is positive

    def test_chd_noncalib_multiplied_by_chd_multiplier(self, sapf_wide: pd.DataFrame, plant_md: pd.DataFrame) -> None:
        """P06 (CHD, FALSE) should be × FLO2019_CHD_MULTIPLIER."""
        original_p06 = sapf_wide["P06"].to_numpy(copy=True)

        out, _ = apply_flo2019_correction(sapf_wide, plant_md)

        np.testing.assert_allclose(
            out["P06"].to_numpy(),
            original_p06 * FLO2019_CHD_MULTIPLIER,
            rtol=1e-9,
        )

    def test_hd_calibrated_unchanged_uppercase_true(self, sapf_wide: pd.DataFrame, plant_md: pd.DataFrame) -> None:
        """P03 (HD, pl_sens_calib='TRUE') must NOT be corrected."""
        original_p03 = sapf_wide["P03"].to_numpy(copy=True)
        out, _ = apply_flo2019_correction(sapf_wide, plant_md)
        np.testing.assert_array_equal(out["P03"].to_numpy(), original_p03)

    def test_hd_calibrated_unchanged_titlecase_true(self, sapf_wide: pd.DataFrame, plant_md: pd.DataFrame) -> None:
        """P04 (HD, pl_sens_calib='True') must be treated as calibrated — case-insensitive."""
        original_p04 = sapf_wide["P04"].to_numpy(copy=True)
        out, _ = apply_flo2019_correction(sapf_wide, plant_md)
        np.testing.assert_array_equal(out["P04"].to_numpy(), original_p04)

    def test_non_dissipation_methods_unchanged(self, sapf_wide: pd.DataFrame, plant_md: pd.DataFrame) -> None:
        """HR (P05) and HPTM (P07) are outside the Dissipation family — no correction."""
        original_p05 = sapf_wide["P05"].to_numpy(copy=True)
        original_p07 = sapf_wide["P07"].to_numpy(copy=True)
        out, _ = apply_flo2019_correction(sapf_wide, plant_md)
        np.testing.assert_array_equal(out["P05"].to_numpy(), original_p05)
        np.testing.assert_array_equal(out["P07"].to_numpy(), original_p07)

    def test_timestamp_column_preserved(self, sapf_wide: pd.DataFrame, plant_md: pd.DataFrame) -> None:
        out, _ = apply_flo2019_correction(sapf_wide, plant_md)
        pd.testing.assert_series_equal(out["TIMESTAMP"], sapf_wide["TIMESTAMP"])

    def test_output_has_same_columns_and_shape(self, sapf_wide: pd.DataFrame, plant_md: pd.DataFrame) -> None:
        out, _ = apply_flo2019_correction(sapf_wide, plant_md)
        assert list(out.columns) == list(sapf_wide.columns)
        assert out.shape == sapf_wide.shape

    def test_input_dataframe_not_mutated(self, sapf_wide: pd.DataFrame, plant_md: pd.DataFrame) -> None:
        """Immutability: the caller's DataFrame must be untouched."""
        pristine = sapf_wide.copy(deep=True)
        apply_flo2019_correction(sapf_wide, plant_md)
        pd.testing.assert_frame_equal(sapf_wide, pristine)

    def test_nan_cells_preserved(self, sapf_wide: pd.DataFrame, plant_md: pd.DataFrame) -> None:
        """NaN × multiplier = NaN. P01 row 0 is NaN in the fixture."""
        out, _ = apply_flo2019_correction(sapf_wide, plant_md)
        assert pd.isna(out["P01"].iloc[0])

    def test_returns_audit_records(self, sapf_wide: pd.DataFrame, plant_md: pd.DataFrame) -> None:
        """Audit list contains one dict per corrected plant.

        Note: this module is tested in isolation. P08/P09/P13/P15 are HD
        non-calibrated, so calibration would touch them, but in the composed
        merge pipeline the treatment filter drops them BEFORE calibration runs.
        Their presence in this expected set reflects the per-module contract,
        not the end-to-end pipeline behaviour.
        """
        _, audit = apply_flo2019_correction(sapf_wide, plant_md)

        corrected_codes = {r["pl_code"] for r in audit}
        # HD non-calib: P01, P02, P08, P09, P11, P12, P13, P14, P15
        # CHD non-calib: P06
        expected_hd = {"P01", "P02", "P08", "P09", "P11", "P12", "P13", "P14", "P15"}
        expected_chd = {"P06"}
        assert corrected_codes == expected_hd | expected_chd

        for record in audit:
            assert set(record.keys()) >= {"pl_code", "method", "multiplier"}
            if record["pl_code"] in expected_hd:
                assert record["method"] == "HD"
                assert record["multiplier"] == FLO2019_HD_MULTIPLIER
            elif record["pl_code"] in expected_chd:
                assert record["method"] == "CHD"
                assert record["multiplier"] == FLO2019_CHD_MULTIPLIER


# --------------------------------------------------------------------------
# Defensive / edge-case behaviour
# --------------------------------------------------------------------------


class TestEdgeCases:
    def test_none_plant_md_is_noop(self, sapf_wide: pd.DataFrame) -> None:
        """plant_md=None → return unchanged copy + empty audit (ICOS pass-through)."""
        out, audit = apply_flo2019_correction(sapf_wide, None)
        pd.testing.assert_frame_equal(out, sapf_wide)
        assert audit == []
        # Still immutable — out must be a copy, not the same object
        assert out is not sapf_wide

    def test_empty_plant_md_is_noop(self, sapf_wide: pd.DataFrame) -> None:
        empty_md = pd.DataFrame(columns=["pl_code", "pl_sens_meth", "pl_sens_calib"])
        out, audit = apply_flo2019_correction(sapf_wide, empty_md)
        pd.testing.assert_frame_equal(out, sapf_wide)
        assert audit == []

    def test_plant_column_missing_from_plant_md_logs_warning(
        self,
        sapf_wide: pd.DataFrame,
        plant_md: pd.DataFrame,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A plant column present in sapf_wide but missing from plant_md is passed through
        unchanged with a warning log."""
        sapf_extra = sapf_wide.copy()
        sapf_extra["P99_GHOST"] = 42.0
        original_ghost = sapf_extra["P99_GHOST"].to_numpy(copy=True)

        with caplog.at_level(logging.WARNING):
            out, _ = apply_flo2019_correction(sapf_extra, plant_md)

        np.testing.assert_array_equal(out["P99_GHOST"].to_numpy(), original_ghost)
        assert any("P99_GHOST" in rec.message for rec in caplog.records)

    def test_custom_multiplier_override_passthrough(self, sapf_wide: pd.DataFrame, plant_md: pd.DataFrame) -> None:
        """multipliers={'HD': 1.0, 'CHD': 1.0} is a no-op on HD/CHD columns."""
        out, _ = apply_flo2019_correction(sapf_wide, plant_md, multipliers={"HD": 1.0, "CHD": 1.0})
        # Every numeric column should equal the input
        for col in sapf_wide.columns:
            if col == "TIMESTAMP":
                continue
            left = out[col].to_numpy()
            right = sapf_wide[col].to_numpy()
            mask = ~pd.isna(left)
            np.testing.assert_array_equal(left[mask], right[mask])
            assert pd.isna(left).equals(pd.isna(right)) if hasattr(pd.isna(left), "equals") else True

    def test_custom_multiplier_method_specific(self, sapf_wide: pd.DataFrame, plant_md: pd.DataFrame) -> None:
        """Custom per-method multiplier is honored — HD×2, CHD×3."""
        original_p01 = sapf_wide["P01"].to_numpy(copy=True)
        original_p06 = sapf_wide["P06"].to_numpy(copy=True)

        out, _ = apply_flo2019_correction(sapf_wide, plant_md, multipliers={"HD": 2.0, "CHD": 3.0})

        np.testing.assert_allclose(out["P01"].to_numpy()[1:], original_p01[1:] * 2.0, rtol=1e-12)
        np.testing.assert_allclose(out["P06"].to_numpy(), original_p06 * 3.0, rtol=1e-12)

    def test_duplicate_pl_code_in_plant_md_logs_warning_and_keeps_first(
        self,
        sapf_wide: pd.DataFrame,
        plant_md: pd.DataFrame,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Duplicate pl_code rows in plant_md (rare but possible when two
        sensors are mounted on one tree) should not crash; first row wins
        with a warning."""
        dup_row = plant_md[plant_md["pl_code"] == "P01"].copy()
        dup_row["pl_sens_calib"] = "TRUE"  # the second row would mark it calibrated
        plant_md_dupes = pd.concat([plant_md, dup_row], ignore_index=True)
        original_p01 = sapf_wide["P01"].to_numpy(copy=True)

        with caplog.at_level(logging.WARNING):
            out, _ = apply_flo2019_correction(sapf_wide, plant_md_dupes)

        # First row wins — first row has FALSE → P01 IS corrected ×1.6804.
        np.testing.assert_allclose(
            out["P01"].to_numpy()[1:],
            original_p01[1:] * FLO2019_HD_MULTIPLIER,
            rtol=1e-9,
        )
        assert any("Duplicate pl_code" in rec.message for rec in caplog.records)
