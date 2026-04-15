# ruff: noqa: N999
"""Unit tests for src.Analyzers.treatment_filter.

The treatment filter drops plants whose experimental manipulation decouples
local water / energy forcing from what ERA5-Land / WorldClim / SoilGrids can
see at 0.1°. The rule is label-driven with control-token precedence over
drop-token substrings (so 'Pre Irrigation' — a baseline phase — is kept
even though it contains the 'irrig' substring), plant-label precedence over
stand-label (so EUC_ELE ambient-ring trees are kept even though their stand
label says 'Elevated atmospheric CO2'), and hard site overrides for the
known edge-case sites.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.Analyzers.treatment_filter import (
    CONTROL_TOKENS,
    DROP_TOKENS,
    SAMPLING_ONLY,
    SITE_OVERRIDES,
    classify_label,
    classify_plant,
    filter_by_treatment,
)

# --------------------------------------------------------------------------
# Constants / token lists — sanity checks
# --------------------------------------------------------------------------


class TestTokenConstants:
    def test_control_tokens_include_pre_irrigation(self) -> None:
        assert any("pre irrigat" in t for t in CONTROL_TOKENS)

    def test_control_tokens_include_ambient(self) -> None:
        assert "ambient" in CONTROL_TOKENS

    def test_drop_tokens_include_irrig(self) -> None:
        assert "irrig" in DROP_TOKENS

    def test_drop_tokens_include_drought(self) -> None:
        assert "drought" in DROP_TOKENS

    def test_drop_tokens_include_elevated_co2(self) -> None:
        assert any("elevated" in t and "co2" in t for t in DROP_TOKENS)

    def test_sampling_only_includes_increment_cores(self) -> None:
        assert "increment cores" in SAMPLING_ONLY

    def test_site_overrides_include_euc_ele(self) -> None:
        assert "AUS_RIC_EUC_ELE" in SITE_OVERRIDES
        assert SITE_OVERRIDES["AUS_RIC_EUC_ELE"] == "keep_all"

    def test_site_overrides_include_sen_sou_baseline_phases(self) -> None:
        assert "ESP_SEN_SOU_PRE" in SITE_OVERRIDES
        assert "ESP_SEN_SOU_POS" in SITE_OVERRIDES


# --------------------------------------------------------------------------
# classify_label — precedence rules
# --------------------------------------------------------------------------


class TestClassifyLabel:
    def test_control_precedence_over_drop_pre_irrigation(self) -> None:
        """'Pre Irrigation' contains 'irrig' substring but must return 'keep'."""
        assert classify_label("Pre Irrigation") == "keep"

    def test_control_precedence_over_drop_post_irrigation(self) -> None:
        assert classify_label("Post Irrigation") == "keep"

    def test_control_ambient(self) -> None:
        assert classify_label("Ambient") == "keep"
        assert classify_label("Ambient CO2") == "keep"

    def test_control_case_insensitive(self) -> None:
        assert classify_label("CONTROL") == "keep"
        assert classify_label("control") == "keep"

    def test_drop_irrigation_plain(self) -> None:
        assert classify_label("Irrigation") == "drop"

    def test_drop_drought_factorial(self) -> None:
        assert classify_label("drought_2") == "drop"
        assert classify_label("Drought") == "drop"

    def test_drop_root_trenching(self) -> None:
        assert classify_label("Root trenching") == "drop"

    def test_drop_elevated_co2(self) -> None:
        assert classify_label("Elevated atmospheric CO2") == "drop"

    def test_drop_shade(self) -> None:
        assert classify_label("Shade") == "drop"

    def test_sampling_only_returns_unknown(self) -> None:
        assert classify_label("Increment cores") == "unknown"
        assert classify_label("Destructive sampling") == "unknown"

    def test_na_and_empty_return_unknown(self) -> None:
        assert classify_label(None) == "unknown"
        assert classify_label("") == "unknown"
        assert classify_label("NA") == "unknown"
        assert classify_label("none") == "unknown"
        assert classify_label(float("nan")) == "unknown"

    def test_unknown_label_defaults_to_keep(self) -> None:
        """Labels we've never seen default to 'keep' (conservative)."""
        assert classify_label("Some Novel Label") == "keep"

    def test_post_thinning_is_keep(self) -> None:
        assert classify_label("Post-thinning") == "keep"


# --------------------------------------------------------------------------
# classify_plant — plant-first / stand-fallback logic
# --------------------------------------------------------------------------


class TestClassifyPlant:
    def test_plant_label_trusted_first_euc_ele_edge(self) -> None:
        """EUC_ELE: pl_treatment='Ambient CO2', st_treatment='Elevated atmospheric CO2'.
        Plant label wins → keep."""
        assert classify_plant("Ambient CO2", "Elevated atmospheric CO2") == "keep"

    def test_plant_drop_wins_over_stand_keep(self) -> None:
        assert classify_plant("Irrigation", "Control") == "drop"

    def test_stand_fallback_on_plant_unknown(self) -> None:
        """pl_treatment NA → use st_treatment."""
        assert classify_plant(None, "Drought") == "drop"
        assert classify_plant(None, "Control") == "keep"

    def test_sampling_only_falls_through_to_stand(self) -> None:
        """'Increment cores' is sampling-only → stand label decides."""
        assert classify_plant("Increment cores", "Drought") == "drop"
        assert classify_plant("Increment cores", None) == "keep"

    def test_both_unknown_defaults_keep(self) -> None:
        assert classify_plant(None, None) == "keep"
        assert classify_plant("NA", "NA") == "keep"

    def test_stand_drop_does_not_override_plant_keep(self) -> None:
        assert classify_plant("Control", "Drought") == "keep"


# --------------------------------------------------------------------------
# filter_by_treatment — DataFrame-level behaviour
# --------------------------------------------------------------------------


class TestFilterByTreatment:
    def test_drops_expected_plants(
        self, sapf_wide: pd.DataFrame, plant_md: pd.DataFrame, stand_md: pd.DataFrame
    ) -> None:
        """Expected drops from the 15-plant fixture:
        - P08 (pl=Irrigation)
        - P09 (st=Drought)
        - P13 (pl=drought_1)
        - P15 (pl=Increment cores, st=Drought → stand fallback to drop)

        Note: P10, P11 get site-override exemption, but in this test we don't
        pass site_code, so the edge-case sites fall back to label logic. P10
        (pl=Ambient CO2) keeps via plant-first rule. P11 (st=Pre Irrigation)
        keeps via control-token precedence.
        """
        out, report = filter_by_treatment(sapf_wide, plant_md, stand_md)

        dropped = set(report["dropped"])
        kept = set(report["kept"])

        assert dropped == {"P08", "P09", "P13", "P15"}
        assert kept == {
            "P01",
            "P02",
            "P03",
            "P04",
            "P05",
            "P06",
            "P07",
            "P10",
            "P11",
            "P12",
            "P14",
        }
        for pc in dropped:
            assert pc not in out.columns
        for pc in kept:
            assert pc in out.columns

    def test_report_shape(self, sapf_wide: pd.DataFrame, plant_md: pd.DataFrame, stand_md: pd.DataFrame) -> None:
        _, report = filter_by_treatment(sapf_wide, plant_md, stand_md)
        assert set(report.keys()) >= {"kept", "dropped", "reason_by_plant"}
        assert isinstance(report["kept"], list)
        assert isinstance(report["dropped"], list)
        assert isinstance(report["reason_by_plant"], dict)
        for pc in report["dropped"]:
            assert pc in report["reason_by_plant"]
            assert isinstance(report["reason_by_plant"][pc], str)
            assert len(report["reason_by_plant"][pc]) > 0

    def test_input_not_mutated(self, sapf_wide: pd.DataFrame, plant_md: pd.DataFrame, stand_md: pd.DataFrame) -> None:
        pristine = sapf_wide.copy(deep=True)
        filter_by_treatment(sapf_wide, plant_md, stand_md)
        pd.testing.assert_frame_equal(sapf_wide, pristine)

    def test_preserves_timestamp_column(
        self, sapf_wide: pd.DataFrame, plant_md: pd.DataFrame, stand_md: pd.DataFrame
    ) -> None:
        out, _ = filter_by_treatment(sapf_wide, plant_md, stand_md)
        assert "TIMESTAMP" in out.columns
        pd.testing.assert_series_equal(out["TIMESTAMP"], sapf_wide["TIMESTAMP"])

    def test_preserves_timestamp_when_all_plants_dropped(self) -> None:
        """A site where every plant is under drought returns a frame with just
        the timestamp column — the merge-script caller will then skip it."""
        ts = pd.date_range("2020-06-01", periods=5, freq="h")
        df = pd.DataFrame({"TIMESTAMP": ts, "PA": [1.0] * 5, "PB": [2.0] * 5})
        pmd = pd.DataFrame(
            [
                {
                    "pl_code": "PA",
                    "pl_sens_meth": "HD",
                    "pl_sens_calib": "FALSE",
                    "pl_treatment": "Irrigation",
                    "si_code": "X",
                },
                {
                    "pl_code": "PB",
                    "pl_sens_meth": "HD",
                    "pl_sens_calib": "FALSE",
                    "pl_treatment": "Drought",
                    "si_code": "X",
                },
            ]
        )
        smd = pd.DataFrame([{"si_code": "X", "st_treatment": None}])

        out, report = filter_by_treatment(df, pmd, smd)
        assert "TIMESTAMP" in out.columns
        assert "PA" not in out.columns
        assert "PB" not in out.columns
        assert set(report["dropped"]) == {"PA", "PB"}
        assert report["kept"] == []

    def test_missing_plant_md_row_defaults_unknown_with_stand_fallback(
        self, sapf_wide: pd.DataFrame, plant_md: pd.DataFrame, stand_md: pd.DataFrame
    ) -> None:
        """A plant column absent from plant_md is treated as pl_treatment=unknown
        and falls through to default-keep (no stand lookup available)."""
        sap_extra = sapf_wide.copy()
        sap_extra["P99_GHOST"] = 7.0
        out, report = filter_by_treatment(sap_extra, plant_md, stand_md)
        assert "P99_GHOST" in out.columns
        assert "P99_GHOST" in report["kept"]

    def test_none_plant_md_is_noop(self, sapf_wide: pd.DataFrame) -> None:
        """plant_md=None → every plant kept, empty report (ICOS pass-through)."""
        out, report = filter_by_treatment(sapf_wide, None, None)
        pd.testing.assert_frame_equal(out, sapf_wide)
        assert report["dropped"] == []
        assert out is not sapf_wide  # defensive copy

    def test_site_override_keep_all_euc_ele(
        self, sapf_wide: pd.DataFrame, plant_md: pd.DataFrame, stand_md: pd.DataFrame
    ) -> None:
        """When site_code is in SITE_OVERRIDES with 'keep_all', the stand-level
        'Elevated atmospheric CO2' label is ignored for ALL plants."""
        out, report = filter_by_treatment(sapf_wide, plant_md, stand_md, site_code="AUS_RIC_EUC_ELE")
        # All 15 fixture plants should survive when the override fires,
        # because the override is whole-site.
        assert set(report["dropped"]) == set()
        for pc in ["P01", "P08", "P09", "P13"]:
            assert pc in out.columns

    def test_site_override_keep_all_sen_sou_pre(
        self, sapf_wide: pd.DataFrame, plant_md: pd.DataFrame, stand_md: pd.DataFrame
    ) -> None:
        out, report = filter_by_treatment(sapf_wide, plant_md, stand_md, site_code="ESP_SEN_SOU_PRE")
        assert set(report["dropped"]) == set()

    def test_site_code_not_in_overrides_uses_normal_logic(
        self, sapf_wide: pd.DataFrame, plant_md: pd.DataFrame, stand_md: pd.DataFrame
    ) -> None:
        """An arbitrary site_code not in SITE_OVERRIDES behaves identically to
        site_code=None."""
        out1, r1 = filter_by_treatment(sapf_wide, plant_md, stand_md, site_code="FRA_HES_HE2")
        out2, r2 = filter_by_treatment(sapf_wide, plant_md, stand_md)
        assert set(r1["dropped"]) == set(r2["dropped"])
        assert set(r1["kept"]) == set(r2["kept"])

    def test_duplicate_pl_code_in_plant_md_logs_warning_and_keeps_first(
        self,
        sapf_wide: pd.DataFrame,
        plant_md: pd.DataFrame,
        stand_md: pd.DataFrame,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Duplicate pl_code rows in plant_md should not crash; first row wins
        with a warning. Mirrors the matching test in calibration_corrector."""
        import logging

        # Duplicate P08 (Irrigation → drop) but mark the duplicate as Control.
        # First-row-wins → P08 must still be dropped.
        dup_row = plant_md[plant_md["pl_code"] == "P08"].copy()
        dup_row["pl_treatment"] = "Control"
        plant_md_dupes = pd.concat([plant_md, dup_row], ignore_index=True)

        with caplog.at_level(logging.WARNING):
            _, report = filter_by_treatment(sapf_wide, plant_md_dupes, stand_md)

        assert "P08" in report["dropped"]
        assert any("Duplicate pl_code" in rec.message for rec in caplog.records)


class TestTokenInvariants:
    """The module-level _validate_token_lists assertion runs at import time;
    these tests document the invariant in the test file so a future maintainer
    sees it explicitly."""

    def test_no_control_token_is_substring_of_drop_token(self) -> None:
        for ct in CONTROL_TOKENS:
            for dt in DROP_TOKENS:
                assert ct not in dt, f"CONTROL_TOKEN {ct!r} must not be substring of DROP_TOKEN {dt!r}"
