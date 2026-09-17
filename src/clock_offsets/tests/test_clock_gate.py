"""The clock gate over a Stage-3 tree (plan WP-A5; failure modes F1, F3, F10; acceptance criteria AC2, AC3).

The corpus here is synthetic and small: five sites on time to make a control population, and one site for each
thing the gate has to say -- an unregistered clock, a correction that was skipped, one applied twice, an
acknowledged site, a dead sensor and a polar day.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.clock_offsets.detect import Estimate, toa_for
from src.clock_offsets.gate import (
    APPLIED_ERROR_HOURS,
    ERROR,
    WARN,
    SiteClock,
    findings,
    main,
    measure_site,
    zero_points,
)
from src.clock_offsets.registry import ClockOffset

REPO = Path(__file__).resolve().parents[3]
LAT, LON = 45.0, 5.0
DAYS = 120
GATES = {"min_clear_days": 20, "min_edge_days": 20}

CONTROLS = ("CTL_A", "CTL_B", "CTL_C", "CTL_D", "CTL_E")
UNREGISTERED = "NEW_SHIFT"
SKIPPED = "REG_SKIPPED"
TWICE = "REG_TWICE"
CORRECTED = "REG_OK"
ACKNOWLEDGED = "ACK_SITE"
DEAD = "DEAD_SENSOR"
TWO_PERIODS = "REG_PERIODS"
ONE_WITNESS = "ONE_WITNESS"
ACK_ONE_WITNESS = "ACK_ONE_WITNESS"

REGISTRY = (
    ClockOffset(site=SKIPPED, component="env", offset_hours=5.0, status="applied", evidence="synthetic"),
    ClockOffset(site=TWICE, component="env", offset_hours=5.0, status="applied", evidence="synthetic"),
    ClockOffset(site=CORRECTED, component="env", offset_hours=5.0, status="applied", evidence="synthetic"),
    ClockOffset(site=ACKNOWLEDGED, component="env", offset_hours=1.25, status="acknowledged", evidence="synthetic"),
    # Two periods, as ISR_YAT_YAT has: a logger that was reset, corrected in two pieces.
    ClockOffset(site=TWO_PERIODS, component="env", offset_hours=-5.0, status="applied", evidence="synthetic",
                start="2015-03-01T00:00:00Z", end="2015-05-01T00:00:00Z"),
    ClockOffset(site=TWO_PERIODS, component="env", offset_hours=-6.0, status="applied", evidence="synthetic",
                start="2015-05-01T00:00:00Z"),
    ClockOffset(site=ONE_WITNESS, component="env", offset_hours=5.0, status="applied", evidence="synthetic"),
    ClockOffset(site=ACK_ONE_WITNESS, component="env", offset_hours=1.25, status="acknowledged", evidence="synthetic"),
)


def write_site(corpus, site, offset, lat=LAT, lon=LON, clearness=0.75, days=DAYS, start="2015-03-01", sap_lag=3):
    """One site's Stage-3 output: sw_in stamped ``offset`` hours late, its sap, and a coordinate."""
    index = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=days * 48, freq="30min")
    sw_in = clearness * toa_for(index - pd.Timedelta(hours=offset), lat, lon).to_numpy()
    stamps = index.strftime("%Y-%m-%dT%H:%M:%SZ")
    pd.DataFrame({"TIMESTAMP": stamps, "sw_in": sw_in, "ta": 15.0}).to_csv(
        corpus / f"{site}_env_data_outliers_removed.csv", index=False
    )
    sap = pd.Series(sw_in).shift(sap_lag).to_numpy()
    pd.DataFrame({"TIMESTAMP": stamps, f"{site}_Tree_1": sap, f"{site}_Tree_2": 0.5 * sap}).to_csv(
        corpus / f"{site}_sapf_data_outliers_removed.csv", index=False
    )
    pd.DataFrame({"si_lat": [lat], "si_long": [lon], "si_elev": [100.0]}).to_csv(
        corpus / f"{site}_site_md.csv", index=False
    )


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    directory = tmp_path_factory.mktemp("stage3")
    for site in CONTROLS:
        write_site(directory, site, 0.0)
    write_site(directory, UNREGISTERED, 5.0)
    write_site(directory, SKIPPED, 5.0)  # the registry says it was corrected; the data say otherwise
    write_site(directory, TWICE, -5.0)  # corrected a second time, so it overshot the other way
    write_site(directory, CORRECTED, 0.0)
    write_site(directory, ACKNOWLEDGED, 1.25)
    write_site(directory, DEAD, 0.0, clearness=0.02)
    write_site(directory, TWO_PERIODS, -6.0)  # the registry corrected it in two pieces; neither landed
    write_site(directory, ONE_WITNESS, 5.0, days=30)  # too short for clear days, long enough for edges
    write_site(directory, ACK_ONE_WITNESS, 1.25, days=30)
    return directory


@pytest.fixture(scope="module")
def clocks(corpus):
    suffix = "_env_data_outliers_removed.csv"
    sites = sorted(path.name[: -len(suffix)] for path in corpus.glob("*" + suffix))
    return [measure_site(site, corpus, corpus, corpus, **GATES) for site in sites]


@pytest.fixture(scope="module")
def report(clocks):
    return findings(clocks, zero_points(clocks), REGISTRY)


def levels_for(report, site):
    return {finding.level for finding in report if finding.site == site}


def messages_for(report, site):
    return " ".join(finding.message for finding in report if finding.site == site)


class TestMeasurement:
    def test_every_site_is_measured_or_says_why(self, clocks):
        assert len(clocks) == len(CONTROLS) + 9
        for clock in clocks:
            assert clock.clear_day or clock.clear_day.reason

    def test_the_control_population_sets_the_zero_points(self, clocks):
        zeros = zero_points(clocks)
        assert zeros["n_controls"] >= len(CONTROLS)
        assert abs(zeros["clear_day"]) <= 0.25
        assert abs(zeros["edge"]) <= 0.25
        assert abs(zeros["pair"] - 1.5) <= 0.25  # the synthetic physiological delay

    def test_a_dead_sensor_is_not_a_control(self, clocks):
        dead = next(clock for clock in clocks if clock.site == DEAD)
        assert not dead.clear_day and "dead" in dead.clear_day.reason


class TestFindings:
    def test_a_site_on_time_says_nothing(self, report):
        for site in CONTROLS:
            assert not levels_for(report, site)

    def test_an_unregistered_clock_is_an_error(self, report):
        assert ERROR in levels_for(report, UNREGISTERED)
        assert "5" in messages_for(report, UNREGISTERED)

    def test_a_correction_that_never_ran_is_an_error(self, report):
        assert ERROR in levels_for(report, SKIPPED)

    def test_a_correction_applied_twice_is_an_error(self, report):
        assert ERROR in levels_for(report, TWICE)

    def test_a_site_the_registry_fixed_says_nothing(self, report):
        assert not levels_for(report, CORRECTED)

    def test_an_acknowledged_site_warns_and_does_not_fail(self, report):
        assert levels_for(report, ACKNOWLEDGED) == {WARN}

    def test_a_dead_sensor_raises_nothing_at_all(self, report):
        assert not levels_for(report, DEAD)

    def test_two_period_entries_are_listed_and_never_summed(self, report):
        # ISR_YAT_YAT carries -5 h and -6 h over two periods. Their sum, -11 h, is a number that was
        # never applied to anything, and a gate that prints it teaches the reader to distrust it.
        message = messages_for(report, TWO_PERIODS)
        assert ERROR in levels_for(report, TWO_PERIODS)
        assert "-11" not in message
        assert "-5.00" in message and "-6.00" in message
        assert "twice" not in message  # the two-entry case cannot diagnose a double application

    def test_one_witness_at_a_corrected_site_still_says_something(self, report):
        # Only the edges can read this site. That is not enough to fail on, and far too much to swallow:
        # a correction that never landed would otherwise leave the gate silent.
        findings_here = [finding for finding in report if finding.site == ONE_WITNESS]
        assert levels_for(report, ONE_WITNESS) == {WARN}
        assert "applied" in " ".join(finding.message for finding in findings_here)
        assert "5" in messages_for(report, ONE_WITNESS)

    def test_one_witness_at_an_acknowledged_site_says_it_is_acknowledged(self, report):
        assert levels_for(report, ACK_ONE_WITNESS) == {WARN}
        assert "acknowledged" in messages_for(report, ACK_ONE_WITNESS)

    def test_the_findings_name_the_component(self, report):
        assert all(finding.component in ("env", "sap") for finding in report)


class TestDecisionRules:
    """The rules themselves, on measurements built by hand: no corpus, no estimator, no ambiguity."""

    ZEROS = {
        "clear_day": 0.0, "edge": 0.0, "pair": 1.5, "pair_p10": 1.5, "pair_p90": 1.5,
        "n_controls": 6, "n_pair_controls": 6,
    }

    @staticmethod
    def clock(site, clear, edge, pair=1.5):
        def estimate(hours):
            return Estimate(reason="could not measure") if hours is None else Estimate(hours=hours, sample=5000)

        return SiteClock(site, LAT, LON, estimate(clear), estimate(edge), estimate(pair))

    def test_a_corrected_site_whose_estimators_disagree_is_still_an_error(self):
        # Both estimators read this site and they contradict each other; one says the clock is three hours
        # out AFTER the registry corrected it. Which of the two is wrong does not matter here, because
        # neither reading is what a correction that landed looks like.
        found = findings([self.clock(SKIPPED, 3.0, 0.1)], self.ZEROS, REGISTRY)
        assert [finding.level for finding in found] == [ERROR]
        assert "+3.00" in found[0].message

    def test_a_corrected_site_only_one_estimator_can_read_is_a_warning(self):
        # FIN_PET's shape: 22 clear days of the 40 the correlation needs, so only the edges can speak.
        found = findings([self.clock(SKIPPED, None, 3.0)], self.ZEROS, REGISTRY)
        assert [finding.level for finding in found] == [WARN]

    def test_a_corrected_site_that_measures_nothing_left_says_nothing(self):
        assert findings([self.clock(SKIPPED, 0.2, -0.1)], self.ZEROS, REGISTRY) == []

    def test_two_estimators_that_agree_the_site_is_on_time_do_not_fail_it(self):
        # USA_ORN_ST2_AMB after its +5 h correction, to the decimal: clear-day -0.17 h, edges -0.50 h.
        # Its twin USA_ORN_ST1_AMB shares the met station, was never corrected, and the whole-corpus run
        # measures it at -0.174 / -0.500 -- the same numbers -- and says nothing about it. A rule that
        # fails one and not the other is reading the registry, not the clock. The residual here is the
        # station's own estimator bias, and the edge shift cannot resolve below 0.25 h anyway.
        assert findings([self.clock(SKIPPED, -0.17, -0.50)], self.ZEROS, REGISTRY) == []

    @pytest.mark.parametrize(
        "clear,edge",
        [(3.0, 0.1), (1.2, 1.5), (-0.17, -0.50), (0.6, 0.55), (5.0, 4.8), (0.49, 0.51)],
    )
    def test_the_residual_it_fails_on_is_the_residual_it_prints(self, clear, edge):
        # The failure this pins: the rule decided on the LOUDEST estimator and the message reported the
        # MEAN, so a corrected site could be failed for 0.50 h and told it measures 0.34 h -- a verdict
        # its own number does not support. Whatever the rule judges, the message must say that number.
        found = findings([self.clock(SKIPPED, clear, edge)], self.ZEROS, REGISTRY)
        if not found or found[0].level != ERROR:
            return
        printed = re.search(r"still measures ([+-]\d+\.\d+) h", found[0].message)
        assert printed, found[0].message
        assert abs(float(printed.group(1))) >= APPLIED_ERROR_HOURS, found[0].message

    def test_an_unregistered_site_needs_both_estimators_over_the_threshold(self):
        # They agree and their mean reaches 1.0 h, but neither alone does. The rule is "both reach 1 h":
        # a pair that only averages its way across the line is a warning, not a failure.
        modest = findings([self.clock("NEW_MODEST", 0.8, 1.2)], self.ZEROS, REGISTRY)
        assert [finding.level for finding in modest] == [WARN]
        real = findings([self.clock("NEW_REAL", 1.05, 1.3)], self.ZEROS, REGISTRY)
        assert [finding.level for finding in real] == [ERROR]


class TestEdges:
    """The shapes a real corpus has that a synthetic one has to be told about."""

    def test_a_tree_with_no_control_site_at_all_still_reports(self, tmp_path):
        # Every site shifted the same way: there is no population to calibrate on. The zero points fall
        # back to nothing rather than to the median of the shifted sites, which would hide all of them.
        for site in CONTROLS:
            write_site(tmp_path, site, 5.0)
        clocks = [measure_site(site, tmp_path, tmp_path, tmp_path, **GATES) for site in sorted(CONTROLS)]
        zeros = zero_points(clocks)
        assert zeros["n_controls"] == 0
        assert zeros["clear_day"] == 0.0 and zeros["edge"] == 0.0
        assert all(finding.level == ERROR for finding in findings(clocks, zeros, REGISTRY))

    def test_a_site_without_a_usable_coordinate_says_so_and_does_not_crash(self, tmp_path):
        write_site(tmp_path, CONTROLS[0], 0.0)
        md = tmp_path / f"{CONTROLS[0]}_site_md.csv"
        pd.DataFrame({"si_lat": [np.nan], "si_long": [np.nan], "si_elev": [100.0]}).to_csv(md, index=False)
        clock = measure_site(CONTROLS[0], tmp_path, tmp_path, tmp_path, **GATES)
        assert not clock.clear_day and "si_lat" in clock.clear_day.reason
        assert clock.latitude is None
        assert findings([clock], zero_points([clock]), REGISTRY) == []

    def test_a_sap_file_whose_trees_are_all_empty_is_not_a_pair_lag(self, tmp_path):
        write_site(tmp_path, CONTROLS[0], 0.0)
        sap = pd.read_csv(tmp_path / f"{CONTROLS[0]}_sapf_data_outliers_removed.csv")
        for col in [c for c in sap.columns if c != "TIMESTAMP"]:
            sap[col] = np.nan
        sap.to_csv(tmp_path / f"{CONTROLS[0]}_sapf_data_outliers_removed.csv", index=False)
        clock = measure_site(CONTROLS[0], tmp_path, tmp_path, tmp_path, **GATES)
        assert not clock.pair and clock.pair.reason
        assert clock.clear_day  # the env side is untouched by an empty sap file

    def test_only_the_named_sites_are_measured(self, tmp_path, capsys):
        for site in CONTROLS:
            write_site(tmp_path, site, 0.0)
        argv = TestMain.argv(tmp_path) + ["--sites", CONTROLS[0], CONTROLS[1]]
        assert main(argv) == 0
        assert "2/2 sites measured" in capsys.readouterr().out


class TestMain:
    def test_a_clean_tree_exits_zero(self, tmp_path):
        for site in CONTROLS:
            write_site(tmp_path, site, 0.0)
        code = main(self.argv(tmp_path))
        assert code == 0

    def test_a_tree_with_an_unregistered_clock_exits_one_and_writes_its_report(self, tmp_path, capsys):
        for site in CONTROLS:
            write_site(tmp_path, site, 0.0)
        write_site(tmp_path, UNREGISTERED, 5.0)
        out = tmp_path / "clock_gate.json"
        assert main(self.argv(tmp_path) + ["--out", str(out)]) == 1
        assert ERROR in capsys.readouterr().out
        report = json.loads(out.read_text())
        assert report["summary"]["errors"] == 1
        assert report["registry_sha256"]
        assert any(site["site"] == UNREGISTERED for site in report["sites"])

    def test_it_runs_as_a_module(self, tmp_path):
        write_site(tmp_path, CONTROLS[0], 0.0)
        result = subprocess.run(
            [sys.executable, "-m", "src.clock_offsets.gate"] + self.argv(tmp_path),
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_a_thin_control_population_is_said_out_loud(self, tmp_path, capsys):
        # Every offset in the report is calibrated on the median of the controls. Run the gate over a
        # handful of sites -- a re-run subset, say -- and that median is noise, quietly moving every number
        # in the report. The verdict stands; the caution says what it rests on.
        for site in CONTROLS[:3]:
            write_site(tmp_path, site, 0.0)
        assert main(self.argv(tmp_path)) == 0
        printed = capsys.readouterr().out
        assert "CAUTION" in printed and "3 control" in printed

    def test_an_empty_tree_is_a_failure_not_a_pass(self, tmp_path, capsys):
        # The most dangerous thing a gate can do is agree with a directory that holds nothing. A wrong
        # --env-dir, a tree that never finished: both arrive here as "no findings".
        assert main(self.argv(tmp_path)) == 1
        assert "no site" in capsys.readouterr().out

    def test_the_report_directory_is_created_if_it_is_missing(self, tmp_path):
        write_site(tmp_path, CONTROLS[0], 0.0)
        out = tmp_path / "reports" / "nested" / "clock_gate.json"
        assert main(self.argv(tmp_path) + ["--out", str(out)]) == 0
        assert json.loads(out.read_text())["summary"]["sites"] == 1

    def test_an_unreadable_env_file_is_a_reason_not_a_traceback(self, tmp_path, capsys):
        # One corrupt file in a 225-site corpus must not take the whole report down with it, and must
        # not pass unmentioned either.
        for site in CONTROLS:
            write_site(tmp_path, site, 0.0)
        broken = tmp_path / f"{CONTROLS[0]}_env_data_outliers_removed.csv"
        broken.write_text("TIMESTAMP,sw_in" + chr(10) + "not-a-time,1,2,3,4" + chr(10))
        assert main(self.argv(tmp_path)) == 0
        assert "4/5 sites measured" in capsys.readouterr().out

    @staticmethod
    def argv(directory):
        return [
            "--env-dir", str(directory),
            "--sap-dir", str(directory),
            "--md-dir", str(directory),
            "--min-clear-days", str(GATES["min_clear_days"]),
            "--min-edge-days", str(GATES["min_edge_days"]),
        ]
