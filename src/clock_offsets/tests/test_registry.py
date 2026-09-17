"""The reviewed clock-offset registry: what it accepts, what it refuses, and what it hands a loader.

Plan WP-A1, acceptance criterion AC1. Sign convention: a positive offset means the label is LATE.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from src.clock_offsets.registry import (
    MAX_OFFSET_HOURS,
    REGISTRY,
    ClockOffset,
    ClockRegistryError,
    offsets_for,
    registry_sha256,
    require_valid_registry,
    validate_registry,
)

CORPUS = Path(__file__).resolve().parents[3] / "data/raw/0.1.5/0.1.5/csv/sapwood_combined"

CORPUS_FILES = (
    "FIN_PET_env_data.csv",
    "FIN_PET_sapf_data.csv",
    "USA_ORN_ST1_AMB_env_data.csv",
    "USA_ORN_ST2_AMB_env_data.csv",
    "CHE_DAV_SEE_env_data.csv",
)


def entry(**overrides):
    fields = {
        "site": "FIN_PET",
        "component": "env",
        "offset_hours": 3.0,
        "status": "applied",
        "evidence": "synthetic",
    }
    fields.update(overrides)
    return ClockOffset(**fields)


@pytest.fixture()
def corpus(tmp_path):
    for name in CORPUS_FILES:
        (tmp_path / name).write_text("TIMESTAMP\n")
    return tmp_path


def only_error(errors):
    assert len(errors) == 1, errors
    return errors[0]


class TestAccepted:
    def test_a_valid_registry_has_no_errors(self, corpus):
        assert validate_registry((entry(), entry(component="sap")), corpus) == []

    def test_an_empty_registry_is_valid(self, corpus):
        assert validate_registry((), corpus) == []

    def test_quarter_hour_and_negative_offsets_are_accepted(self, corpus):
        assert validate_registry((entry(offset_hours=-5.25),), corpus) == []

    def test_an_offset_of_exactly_the_maximum_is_accepted(self, corpus):
        assert validate_registry((entry(offset_hours=MAX_OFFSET_HOURS),), corpus) == []

    def test_adjoining_half_open_periods_do_not_overlap(self, corpus):
        before = entry(offset_hours=-6.0, end="2009-01-01T00:00:00Z")
        after = entry(offset_hours=-5.25, start="2009-01-01T00:00:00Z")
        assert validate_registry((before, after), corpus) == []

    def test_an_acknowledged_entry_may_carry_no_offset_and_no_evidence(self, corpus):
        assert validate_registry((entry(offset_hours=0.0, status="acknowledged", evidence=""),), corpus) == []

    def test_without_a_corpus_directory_the_file_check_is_skipped(self):
        assert validate_registry((entry(site="NOT_A_SITE"),), None) == []

    def test_entries_are_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            entry().offset_hours = 4.0


class TestRefused:
    @pytest.mark.parametrize(
        "overrides, fragment",
        [
            ({"component": "soil"}, "component"),
            ({"status": "maybe"}, "status"),
            ({"offset_hours": MAX_OFFSET_HOURS + 0.25}, "exceeds"),
            ({"offset_hours": -(MAX_OFFSET_HOURS + 0.25)}, "exceeds"),
            ({"offset_hours": 3.1}, "0.25"),
            ({"offset_hours": float("nan")}, "finite"),
            ({"offset_hours": 0.0}, "zero"),
            ({"evidence": "  "}, "evidence"),
            ({"start": "2010-01-01", "end": "2009-01-01"}, "before"),
            ({"start": "2010-01-01", "end": "2010-01-01"}, "before"),
            ({"start": "not a date"}, "parse"),
            ({"site": ""}, "site"),
            ({"site": "FIN/PET"}, "site"),
        ],
    )
    def test_a_malformed_entry_is_refused_with_one_clear_error(self, corpus, overrides, fragment):
        assert fragment in only_error(validate_registry((entry(**overrides),), corpus))

    def test_overlapping_whole_record_entries_are_refused(self, corpus):
        assert "overlap" in only_error(validate_registry((entry(), entry(offset_hours=3.5)), corpus))

    def test_partially_overlapping_periods_are_refused(self, corpus):
        first = entry(end="2010-06-01")
        second = entry(offset_hours=3.5, start="2010-01-01", end="2011-01-01")
        assert "overlap" in only_error(validate_registry((first, second), corpus))

    def test_an_acknowledged_entry_overlapping_an_applied_one_is_refused(self, corpus):
        acknowledged = entry(status="acknowledged", offset_hours=0.0, evidence="")
        assert "overlap" in only_error(validate_registry((entry(), acknowledged), corpus))

    def test_the_same_period_for_the_other_component_is_not_an_overlap(self, corpus):
        assert validate_registry((entry(), entry(component="sap")), corpus) == []

    def test_a_site_with_no_corpus_file_is_refused(self, corpus):
        errors = validate_registry((entry(site="USA_HUY_LIN_NON"),), corpus)
        assert "USA_HUY_LIN_NON_env_data.csv" in only_error(errors)

    def test_the_component_decides_which_file_must_exist(self, corpus):
        errors = validate_registry((entry(site="USA_ORN_ST2_AMB", component="sap"),), corpus)
        assert "USA_ORN_ST2_AMB_sapf_data.csv" in only_error(errors)

    def test_a_prefix_twin_does_not_satisfy_the_file_check(self, corpus):
        # CHE_DAV_SEE_env_data.csv exists; CHE_DAV_env_data.csv does not.
        assert "CHE_DAV_env_data.csv" in only_error(validate_registry((entry(site="CHE_DAV"),), corpus))

    def test_every_error_is_reported_not_just_the_first(self, corpus):
        errors = validate_registry((entry(component="soil"), entry(site="NOT_A_SITE", offset_hours=3.3)), corpus)
        assert len(errors) == 3, errors

    def test_require_valid_registry_raises_listing_every_error(self, corpus):
        with pytest.raises(ClockRegistryError) as caught:
            require_valid_registry((entry(component="soil"), entry(site="NOT_A_SITE")), corpus)
        message = str(caught.value)
        assert "component" in message
        assert "NOT_A_SITE_env_data.csv" in message

    def test_require_valid_registry_is_silent_when_valid(self, corpus):
        require_valid_registry((entry(),), corpus)


class TestOffsetsFor:
    REGISTRY = (
        entry(site="USA_ORN_ST2_AMB", offset_hours=5.0),
        entry(site="USA_ORN_ST2_AMB", component="sap", offset_hours=5.0),
        entry(site="SWE_SVA_MIX_NON", offset_hours=0.0, status="acknowledged", evidence=""),
    )

    def test_returns_the_applied_entries_for_that_site_and_component(self):
        found = offsets_for("USA_ORN_ST2_AMB", "env", self.REGISTRY)
        assert found == (self.REGISTRY[0],)

    def test_the_result_is_a_tuple(self):
        assert isinstance(offsets_for("USA_ORN_ST2_AMB", "sap", self.REGISTRY), tuple)

    def test_acknowledged_entries_are_never_handed_to_a_loader(self):
        assert offsets_for("SWE_SVA_MIX_NON", "env", self.REGISTRY) == ()

    def test_the_twin_site_gets_nothing(self):
        assert offsets_for("USA_ORN_ST1_AMB", "env", self.REGISTRY) == ()

    def test_the_two_token_location_key_gets_nothing(self):
        # The Stage-3 loaders key `location` on two tokens: USA_ORN for both twins.
        assert offsets_for("USA_ORN", "env", self.REGISTRY) == ()

    def test_an_unknown_component_raises(self):
        with pytest.raises(ValueError):
            offsets_for("USA_ORN_ST2_AMB", "soil", self.REGISTRY)


class TestRegistryHash:
    def test_the_hash_is_a_stable_sha256(self):
        registry = (entry(), entry(component="sap"))
        assert registry_sha256(registry) == registry_sha256(tuple(registry))
        assert len(registry_sha256(registry)) == 64

    def test_any_field_change_changes_the_hash(self):
        base = registry_sha256((entry(),))
        changed = (
            entry(offset_hours=3.5),
            entry(status="acknowledged"),
            entry(start="2010-01-01"),
            entry(evidence="other"),
            entry(site="USA_ORN_ST2_AMB"),
        )
        assert all(registry_sha256((variant,)) != base for variant in changed)

    def test_the_order_of_entries_does_not_change_the_hash(self):
        first, second = entry(), entry(component="sap")
        assert registry_sha256((first, second)) == registry_sha256((second, first))


class TestShippedRegistry:
    def test_the_shipped_registry_is_a_tuple_of_entries(self):
        assert isinstance(REGISTRY, tuple)
        assert all(isinstance(item, ClockOffset) for item in REGISTRY)

    def test_the_shipped_registry_is_valid_without_a_corpus(self):
        assert validate_registry(REGISTRY, None) == []

    @pytest.mark.skipif(not CORPUS.is_dir(), reason="corpus not on this machine")
    def test_the_shipped_registry_is_valid_against_the_corpus(self):
        assert validate_registry(REGISTRY, CORPUS) == []
