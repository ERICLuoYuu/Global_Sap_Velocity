"""The contract badge must never outlive the data, nor arrive after it.

The dataset gate treats feature_contract.json as authoritative, so these two are data-corruption
paths rather than tidiness: a badge left behind by an earlier servable run presents per-site z-scores
as climatological ones, and a badge that arrives only at the end leaves an interrupted servable run
looking like a site tree, which the site contract accepts.
"""

import json

import numpy as np
import pytest

from src.feature_contract.definitions import (
    CELLS_FILE,
    SERVABLE_CONTRACT,
    SIDECAR_FILE,
    assert_dataset_contract,
)
from src.feature_contract.merge_contract import (
    cell_rows,
    clear_sidecar,
    contract_block,
    resolve_site_cells,
    write_sidecar,
)
from src.feature_contract.swvl_climatology import SwvlClimatology

LAYERS = (1, 2, 3, 4)


def _clim():
    lats, lons = np.array([51.0, 50.9, 50.8]), np.array([7.9, 8.0, 8.1])
    return SwvlClimatology(
        lats=lats,
        lons=lons,
        sigma={layer: np.full((3, 3), 0.05) for layer in LAYERS},
        mean={layer: np.full((3, 3), 0.30) for layer in LAYERS},
        path="/fake/cube.nc",
        sha256="0" * 64,
        attrs={},
    )


def _rows(site="A"):
    return cell_rows(site, 50.9, 8.0, resolve_site_cells(_clim(), site, 50.9, 8.0, LAYERS))


def _badge(tmp_path, complete=True):
    rows = _rows()
    write_sidecar(tmp_path, contract_block(_clim(), LAYERS, 3, rows, "sha", complete=complete), rows)
    return tmp_path


class TestClearingAStaleBadge:
    def test_it_removes_both_files(self, tmp_path):
        _badge(tmp_path)
        removed = clear_sidecar(tmp_path)
        assert {p.name for p in removed} == {SIDECAR_FILE, CELLS_FILE}
        assert not (tmp_path / SIDECAR_FILE).exists()
        assert not (tmp_path / CELLS_FILE).exists()

    def test_a_directory_with_no_badge_is_left_alone(self, tmp_path):
        assert clear_sidecar(tmp_path) == []

    def test_after_clearing_the_gate_no_longer_calls_the_tree_servable(self, tmp_path):
        """The scenario: a servable run, then a site re-merge into the same directory. Without the
        clear, the site tree would keep a badge the gate believes."""
        _badge(tmp_path)
        assert_dataset_contract(tmp_path, SERVABLE_CONTRACT)  # before
        clear_sidecar(tmp_path)
        with pytest.raises(ValueError):
            assert_dataset_contract(tmp_path, SERVABLE_CONTRACT)
        assert_dataset_contract(tmp_path, "site")  # and it reads as the site tree it now is


class TestABadgeThatSaysWhetherItIsFinal:
    def test_the_provisional_badge_is_marked_incomplete(self, tmp_path):
        _badge(tmp_path, complete=False)
        assert json.loads((tmp_path / SIDECAR_FILE).read_text())["complete"] is False

    def test_the_final_badge_is_marked_complete(self, tmp_path):
        _badge(tmp_path, complete=True)
        assert json.loads((tmp_path / SIDECAR_FILE).read_text())["complete"] is True

    def test_an_interrupted_run_still_reads_as_servable_not_as_a_site_tree(self, tmp_path):
        """This is the point of writing the badge first. The numbers are servable either way; what
        must not happen is the site contract accepting them because no badge arrived."""
        _badge(tmp_path, complete=False)
        assert_dataset_contract(tmp_path, SERVABLE_CONTRACT)
        with pytest.raises(ValueError, match="servable"):
            assert_dataset_contract(tmp_path, "site")

    def test_the_final_badge_replaces_the_provisional_one(self, tmp_path):
        _badge(tmp_path, complete=False)
        rows = _rows("A") + _rows("B")
        write_sidecar(tmp_path, contract_block(_clim(), LAYERS, 3, rows, "sha", complete=True), rows)
        block = json.loads((tmp_path / SIDECAR_FILE).read_text())
        assert block["complete"] is True
        assert block["swvl"]["n_sites"] == 2
