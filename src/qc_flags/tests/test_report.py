"""Every site ends in a recorded state.

D-1's real damage was not the `AttributeError`; it was that the error became one
line of stdout among thousands and nobody could tell a site that was skipped
from a site that died.  Acceptance criteria 1 and 2 turn "did every site get
processed?" into a number, so the answer stops depending on reading a log.
"""

import json

import pytest

from src.qc_flags.mechanisms import Mechanism
from src.qc_flags.report import RunReport, SiteFlagOutcome


def outcome(site, mechanism=Mechanism.NONE, masked=None, present=None, skipped=None):
    return SiteFlagOutcome(
        site=site,
        kind="env",
        mechanism=mechanism,
        masked_counts=masked or {},
        present_counts=present or {},
        skipped_reason=skipped,
    )


class TestSiteFlagOutcome:
    def test_is_immutable(self):
        o = outcome("ARG_MAZ")
        with pytest.raises(AttributeError):
            o.site = "OTHER"

    def test_masked_fraction_is_of_present_values_only(self):
        # Masking an already-NaN cell is a no-op, so the honest denominator is
        # the count of values that were actually there.
        o = outcome("CHE_DAV", masked={"ta": 10}, present={"ta": 1000})
        assert o.masked_fraction("ta") == pytest.approx(0.01)

    def test_masked_fraction_of_a_variable_with_nothing_present_is_zero(self):
        o = outcome("X_Y", masked={"ta": 0}, present={"ta": 0})
        assert o.masked_fraction("ta") == 0.0

    def test_a_skipped_site_is_accounted_for(self):
        o = outcome("SWE_NOR", skipped="no environmental variables")
        assert o.is_accounted_for is True

    def test_a_processed_site_is_accounted_for(self):
        o = outcome("ARG_MAZ", mechanism=Mechanism.SAPFLUXNET_FLAGS, masked={"ta": 1}, present={"ta": 2})
        assert o.is_accounted_for is True

    def test_a_site_with_neither_counts_nor_a_reason_is_not_accounted_for(self):
        # This is the D-1 shape: the site vanished and nothing says why.
        o = outcome("GHOST_SITE")
        assert o.is_accounted_for is False


class TestRunReport:
    def test_mechanism_census_counts_each_kind(self):
        report = RunReport()
        for i in range(3):
            report.add(outcome(f"EXT_{i}", Mechanism.SAPFLUXNET_FLAGS, {"ta": 1}, {"ta": 9}))
        report.add(outcome("ICOS_1", Mechanism.QC_COLUMNS, {"ta": 1}, {"ta": 9}))
        report.add(outcome("PLAIN_1", Mechanism.NONE, {}, {"ta": 9}))
        report.add(outcome("SWE_NOR", skipped="no environmental variables"))

        census = report.mechanism_census()
        assert census["sapfluxnet_flags"] == 3
        assert census["qc_columns"] == 1
        assert census["none"] == 1
        assert census["skipped"] == 1

    def test_every_site_accounted_for(self):
        report = RunReport()
        report.add(outcome("A_B", Mechanism.NONE, {}, {"ta": 1}))
        report.add(outcome("SWE_NOR", skipped="no env"))
        assert report.all_accounted_for() is True

    def test_an_unaccounted_site_is_reported_not_hidden(self):
        report = RunReport()
        report.add(outcome("GHOST_SITE"))
        assert report.all_accounted_for() is False
        assert "GHOST_SITE" in report.unaccounted_sites()

    def test_adding_the_same_site_twice_raises(self):
        # Silent overwriting would make sites_in == sites_reported hold while a
        # site was lost, which is the exact accounting criterion 2 asks for.
        report = RunReport()
        report.add(outcome("A_B", Mechanism.NONE, {}, {"ta": 1}))
        with pytest.raises(ValueError):
            report.add(outcome("A_B", Mechanism.NONE, {}, {"ta": 1}))

    def test_same_site_different_kind_is_allowed(self):
        report = RunReport()
        report.add(SiteFlagOutcome("A_B", "env", Mechanism.NONE, {}, {"ta": 1}, None))
        report.add(SiteFlagOutcome("A_B", "sapf", Mechanism.NONE, {}, {"t1": 1}, None))
        assert len(report) == 2

    def test_json_round_trips(self, tmp_path):
        report = RunReport()
        report.add(outcome("A_B", Mechanism.QC_COLUMNS, {"ta": 3}, {"ta": 100}))
        path = tmp_path / "report.json"
        report.write_json(path)
        loaded = json.loads(path.read_text())
        assert loaded["census"]["qc_columns"] == 1
        assert loaded["sites"][0]["site"] == "A_B"
        assert loaded["sites"][0]["masked_counts"]["ta"] == 3

    def test_table_names_the_skip_reason(self):
        report = RunReport()
        report.add(outcome("SWE_NOR", skipped="no environmental variables"))
        assert "no environmental variables" in report.to_table()

    def test_table_survives_an_empty_report(self):
        assert RunReport().to_table() is not None
