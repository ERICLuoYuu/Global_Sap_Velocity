"""The clock stamp beside a Stage-3 output: what it records, and what the merge refuses without it.

Review finding 2, 2026-09-17. ``clock_registry_sha256`` records the registry the MERGE imported, and the
env daily gate checked only that the field was non-empty. So the ordinary workflow -- edit the registry,
re-run the merge alone, because Stage 3 is a morning and the merge is an hour -- produced a tree whose
block asserted corrections its inputs never received, and ~14 h of selection and training were spent
attributing a null result to a fix that never landed. ``registry_sha256(())`` is 64 valid hex characters,
so even a tree built with every correction stripped out passed.

The tests below are written against the two halves of that: a stamp is written by the run that moves the
rows, and the merge fails when the stamp and its own registry disagree -- in either direction.
"""

from __future__ import annotations

import json

import pytest

from src.clock_offsets.provenance import (
    MAX_REPORTED_SITES,
    STAMP_VERSION,
    describe,
    offsets_digest,
    provenance_errors,
    read_stamp,
    stamp_content,
    stamp_errors,
    stamp_path,
    tree_provenance,
    write_stamp,
)
from src.clock_offsets.registry import REGISTRY, ClockOffset

ENV = "_env_data_outliers_removed.csv"
SAP = "_sapf_data_outliers_removed.csv"


def entry(**overrides):
    fields = {
        "site": "FIN_PET",
        "component": "env",
        "offset_hours": 3.5,
        "status": "applied",
        "evidence": "synthetic",
    }
    fields.update(overrides)
    return ClockOffset(**fields)


@pytest.fixture()
def tree(tmp_path):
    """A Stage-3 output directory: one site the registry moves, and three it does not."""
    for name in ("FIN_PET" + ENV, "FIN_PET" + SAP, "DEU_HIN_TER" + ENV, "USA_ORN_ST1_AMB" + ENV):
        (tmp_path / name).write_text("TIMESTAMP\n")
    return tmp_path


class TestWriteStamp:
    def test_a_corrected_file_gets_a_stamp_naming_its_offsets(self, tree):
        path = tree / ("FIN_PET" + ENV)
        written = write_stamp(path, "env", (entry(),))
        assert written == stamp_path(path) and written.is_file()
        stamp = json.loads(written.read_text())
        assert stamp["site"] == "FIN_PET" and stamp["component"] == "env"
        assert [row["offset_hours"] for row in stamp["offsets"]] == [3.5]

    def test_an_uncorrected_file_gets_no_stamp(self, tree):
        path = tree / ("DEU_HIN_TER" + ENV)
        assert write_stamp(path, "env", (entry(),)) is None
        assert not stamp_path(path).exists()

    def test_the_other_component_of_a_corrected_site_gets_no_stamp(self, tree):
        # FIN_PET's env is moved and its sap is not; one stamp, not two.
        assert write_stamp(tree / ("FIN_PET" + SAP), "sap", (entry(),)) is None

    def test_a_stamp_the_registry_no_longer_supports_is_removed(self, tree):
        path = tree / ("FIN_PET" + ENV)
        write_stamp(path, "env", (entry(),))
        assert write_stamp(path, "env", ()) is None
        assert not stamp_path(path).exists(), "a withdrawn correction must not leave its claim behind"

    def test_the_site_is_read_from_the_file_name_and_never_from_a_prefix(self, tree):
        # The registry names the ST2 twin; ST1 stands at the same location and shares its sensors.
        table = (entry(site="USA_ORN_ST2_AMB", offset_hours=5.0),)
        assert write_stamp(tree / ("USA_ORN_ST1_AMB" + ENV), "env", table) is None

    def test_the_stamp_sits_beside_the_file_it_describes(self, tree):
        path = tree / ("FIN_PET" + ENV)
        assert stamp_path(path).parent == path.parent
        assert stamp_path(path).name == "FIN_PET_env_data_outliers_removed.clock.json"


class TestReadStamp:
    def test_no_stamp_reads_as_none(self, tree):
        assert read_stamp(tree / ("DEU_HIN_TER" + ENV)) is None

    def test_a_stamp_of_an_unknown_version_is_refused(self, tree):
        path = tree / ("FIN_PET" + ENV)
        write_stamp(path, "env", (entry(),))
        stamp_path(path).write_text(json.dumps({"version": STAMP_VERSION + 1}))
        with pytest.raises(ValueError):
            read_stamp(path)

    def test_a_stamp_that_is_not_an_object_is_refused(self, tree):
        path = tree / ("FIN_PET" + ENV)
        stamp_path(path).write_text("[]")
        with pytest.raises(ValueError):
            read_stamp(path)


class TestStampErrors:
    def test_a_corrected_file_written_under_the_same_registry_passes(self, tree):
        path = tree / ("FIN_PET" + ENV)
        write_stamp(path, "env", (entry(),))
        assert stamp_errors(path, "env", (entry(),)) == []

    def test_an_uncorrected_file_with_no_stamp_passes(self, tree):
        assert stamp_errors(tree / ("DEU_HIN_TER" + ENV), "env", (entry(),)) == []

    def test_a_corrected_file_with_no_stamp_fails(self, tree):
        # The workflow gap itself: the registry names the site, Stage 3 was never re-run.
        errors = stamp_errors(tree / ("FIN_PET" + ENV), "env", (entry(),))
        assert errors and "no clock stamp" in errors[0]

    def test_changing_the_offset_after_stage_3_fails(self, tree):
        path = tree / ("FIN_PET" + ENV)
        write_stamp(path, "env", (entry(),))
        errors = stamp_errors(path, "env", (entry(offset_hours=5.0),))
        assert errors and "+3.5 h" in errors[0] and "+5 h" in errors[0]

    def test_narrowing_the_period_after_stage_3_fails(self, tree):
        path = tree / ("FIN_PET" + ENV)
        write_stamp(path, "env", (entry(),))
        assert stamp_errors(path, "env", (entry(start="2008-07-01T00:00:00Z"),))

    def test_withdrawing_the_entry_after_stage_3_fails(self, tree):
        path = tree / ("FIN_PET" + ENV)
        write_stamp(path, "env", (entry(),))
        errors = stamp_errors(path, "env", ())
        assert errors and "no longer applies" in errors[0]

    def test_rewriting_the_evidence_alone_does_not_fail(self, tree):
        # Prose is corrected far more often than a clock is; a sentence must not cost a rebuild.
        path = tree / ("FIN_PET" + ENV)
        write_stamp(path, "env", (entry(),))
        assert stamp_errors(path, "env", (entry(evidence="reworded, same measurement"),)) == []

    def test_a_damaged_stamp_fails_rather_than_passing(self, tree):
        path = tree / ("FIN_PET" + ENV)
        write_stamp(path, "env", (entry(),))
        stamp_path(path).write_text("{not json")
        errors = stamp_errors(path, "env", (entry(),))
        assert errors and "cannot be read" in errors[0]

    def test_a_stamp_belonging_to_another_site_fails(self, tree):
        path = tree / ("FIN_PET" + ENV)
        write_stamp(path, "env", (entry(),))
        content = json.loads(stamp_path(path).read_text())
        content["site"] = "DEU_HIN_TER"
        stamp_path(path).write_text(json.dumps(content))
        errors = stamp_errors(path, "env", (entry(),))
        assert errors and "DEU_HIN_TER" in errors[0]

    def test_a_damaged_stamp_is_reported_and_never_raised(self, tree):
        # The merge's per-site except-continue would turn a raise into a silently dropped site.
        path = tree / ("FIN_PET" + ENV)
        stamp_path(path).write_text("{not json")
        assert isinstance(stamp_errors(path, "env", (entry(),)), list)

    def test_every_applied_site_in_the_shipped_registry_round_trips(self, tmp_path):
        applied = {(e.site, e.component) for e in REGISTRY if e.status == "applied"}
        assert applied, "the shipped registry applies nothing; this test would prove nothing"
        for site, component in sorted(applied):
            path = tmp_path / (site + (ENV if component == "env" else SAP))
            path.write_text("TIMESTAMP\n")
            assert write_stamp(path, component) is not None, site
            assert stamp_errors(path, component) == [], site


class TestDigestAndDescription:
    def test_the_digest_ignores_the_order_of_entries(self):
        first, second = entry(start="2008-01-01T00:00:00Z"), entry(start="2009-01-01T00:00:00Z")
        assert offsets_digest((first, second)) == offsets_digest((second, first))

    def test_the_digest_ignores_the_evidence(self):
        assert offsets_digest((entry(),)) == offsets_digest((entry(evidence="other words"),))

    @pytest.mark.parametrize(
        "changed",
        [{"offset_hours": 5.0}, {"status": "acknowledged"}, {"site": "RUS_FYO"}, {"start": "2008-01-01T00:00:00Z"}],
    )
    def test_any_acting_field_changes_the_digest(self, changed):
        assert offsets_digest((entry(**changed),)) != offsets_digest((entry(),))

    def test_a_stamp_and_the_registry_agree_on_the_digest(self):
        content = stamp_content("FIN_PET", "env", (entry(),), (entry(),))
        assert content["offsets_sha256"] == offsets_digest(content["offsets"])

    def test_every_entry_is_described_never_summed(self):
        # ISR_YAT_YAT carries two clocks, -5 h then -6 h. A reader must see both, never their sum.
        told = describe((entry(offset_hours=-5.0), entry(offset_hours=-6.0)))
        assert told.split("; ") == ["-5 h", "-6 h"]

    def test_a_period_entry_names_its_bounds(self):
        assert describe((entry(offset_hours=-5.0, end="2009-11-23T12:00:00Z"),)) == (
            "-5 h until 2009-11-23T12:00:00Z"
        )

    def test_no_offset_is_described_as_such(self):
        assert describe(()) == "no offset"


class TestTreeProvenance:
    def test_it_counts_every_site_written_not_only_the_corrected_ones(self):
        record = tree_provenance({"FIN_PET": [], "DEU_HIN_TER": []}, (entry(),))
        assert record["n_sites_checked"] == 2

    def test_it_names_the_corrected_sites_of_this_tree(self):
        record = tree_provenance({"FIN_PET": [], "DEU_HIN_TER": []}, (entry(),))
        assert record["sites_corrected"] == ["FIN_PET"]

    def test_a_site_the_merge_skipped_is_absent_rather_than_clean(self):
        record = tree_provenance({"DEU_HIN_TER": []}, (entry(),))
        assert "FIN_PET" not in record["mismatches"] and record["sites_corrected"] == []

    def test_only_the_sites_that_failed_are_recorded_as_mismatches(self):
        record = tree_provenance({"FIN_PET": ["moved by nothing"], "DEU_HIN_TER": []}, (entry(),))
        assert list(record["mismatches"]) == ["FIN_PET"]


class TestProvenanceErrors:
    def test_a_tree_whose_inputs_all_matched_passes(self):
        assert provenance_errors(tree_provenance({"FIN_PET": [], "DEU_HIN_TER": []}, (entry(),))) == []

    @pytest.mark.parametrize("missing", [None, "", 64 * "0", []])
    def test_a_tree_that_never_checked_fails(self, missing):
        assert provenance_errors(missing)

    @pytest.mark.parametrize("checked", [0, -1, None, True, "212"])
    def test_a_tree_that_checked_no_site_fails(self, checked):
        assert provenance_errors({"n_sites_checked": checked, "mismatches": {}})

    def test_a_record_without_a_mismatches_map_fails(self):
        assert provenance_errors({"n_sites_checked": 212})

    def test_a_mismatch_is_reported_with_the_reason_the_merge_gave(self):
        record = tree_provenance({"USA_HUY_LIN_NON": ["USA_HUY_LIN_NON/env: no clock stamp"]}, ())
        errors = provenance_errors(record)
        assert errors and "USA_HUY_LIN_NON" in errors[0] and "no clock stamp" in errors[0]

    def test_a_long_list_is_capped_but_the_count_is_complete(self):
        mismatches = {"SITE_{0}".format(i): ["off"] for i in range(MAX_REPORTED_SITES + 3)}
        errors = provenance_errors({"n_sites_checked": 200, "mismatches": mismatches})
        assert any("3 further site" in error for error in errors)
