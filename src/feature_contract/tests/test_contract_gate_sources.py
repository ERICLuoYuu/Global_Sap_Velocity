"""Where the gate is allowed to learn that a tree is servable.

Two artefacts can say so: the ``feature_contract`` block inside RUN_PROVENANCE.json, written by the
sweep, and the sidecar the Stage-4 merge writes for itself. A tree built by the merge carries the
second from the moment it is finished, and must be usable then -- the whole point of building the
contract into the merge was to remove the pass that came afterwards.
"""

import json

import pytest

from src.feature_contract.definitions import (
    PROVENANCE_FILE,
    PROVENANCE_KEY,
    SERVABLE_CONTRACT,
    SIDECAR_FILE,
    assert_dataset_contract,
)

BLOCK = {"name": SERVABLE_CONTRACT, "swvl": {"n_fallback_cells": 0}}


def _provenance(directory, block=None):
    payload = {"config_name": "fullyear_fullday_tx"}
    if block is not None:
        payload[PROVENANCE_KEY] = block
    (directory / PROVENANCE_FILE).write_text(json.dumps(payload))


def _sidecar(directory, block=BLOCK):
    (directory / SIDECAR_FILE).write_text(json.dumps(block))


class TestATreeTheSweepFinished:
    def test_the_provenance_block_is_accepted(self, tmp_path):
        _provenance(tmp_path, BLOCK)
        assert_dataset_contract(tmp_path, SERVABLE_CONTRACT)

    def test_and_the_site_contract_is_refused_against_it(self, tmp_path):
        _provenance(tmp_path, BLOCK)
        with pytest.raises(ValueError, match="servable"):
            assert_dataset_contract(tmp_path, "site")


class TestATreeTheMergeJustFinished:
    """No sweep has run yet: the sidecar is the only record, and it is enough."""

    def test_the_sidecar_alone_is_accepted(self, tmp_path):
        _sidecar(tmp_path)
        assert_dataset_contract(tmp_path, SERVABLE_CONTRACT)

    def test_the_sidecar_beside_a_provenance_that_has_no_block_is_accepted(self, tmp_path):
        """The real state after the merge and the sweep, before anyone folds the block in."""
        _provenance(tmp_path, None)
        _sidecar(tmp_path)
        assert_dataset_contract(tmp_path, SERVABLE_CONTRACT)

    def test_the_site_contract_is_refused_against_it(self, tmp_path):
        """Its _zscore columns are climatological anomalies. Read as per-site z-scores they would
        train a model that looks servable and is not -- and before this, nothing said so."""
        _provenance(tmp_path, None)
        _sidecar(tmp_path)
        with pytest.raises(ValueError, match="servable"):
            assert_dataset_contract(tmp_path, "site")


class TestATreeThatSaysNothing:
    def test_servable_is_refused_and_names_both_artefacts(self, tmp_path):
        with pytest.raises(ValueError, match=SIDECAR_FILE):
            assert_dataset_contract(tmp_path, SERVABLE_CONTRACT)

    def test_a_legacy_tree_is_still_fine_under_the_site_contract(self, tmp_path):
        _provenance(tmp_path, None)
        assert_dataset_contract(tmp_path, "site")

    def test_a_tree_with_no_files_at_all_is_fine_under_the_site_contract(self, tmp_path):
        assert_dataset_contract(tmp_path, "site")


class TestASidecarThatNamesSomethingElse:
    def test_it_is_refused_under_servable(self, tmp_path):
        _sidecar(tmp_path, {"name": "site"})
        with pytest.raises(ValueError, match="servable"):
            assert_dataset_contract(tmp_path, SERVABLE_CONTRACT)

    def test_the_provenance_block_wins_when_both_are_present(self, tmp_path):
        """The sweep's fold is the later, more considered record; the sidecar is the fallback."""
        _provenance(tmp_path, BLOCK)
        _sidecar(tmp_path, {"name": "site"})
        assert_dataset_contract(tmp_path, SERVABLE_CONTRACT)
