"""The trainer-side gate: a run may only read a dataset built under its own contract."""

import json

import pytest
from src.feature_contract.definitions import FEATURE_CONTRACTS, assert_dataset_contract, validate_contract


def _config_dir(tmp_path, block):
    """A merged config dir. block=None -> no RUN_PROVENANCE.json; "absent" -> provenance without the block."""
    config_dir = tmp_path / "cfg"
    (config_dir / "daily").mkdir(parents=True)
    if block is not None:
        provenance = {"config_name": "fullyear_fullday_tx", "flags": {}}
        if block != "absent":
            provenance["feature_contract"] = block
        (config_dir / "RUN_PROVENANCE.json").write_text(json.dumps(provenance))
    return config_dir


def test_known_contracts():
    assert FEATURE_CONTRACTS == ("site", "servable")


def test_validate_returns_the_name():
    assert validate_contract("servable") == "servable"


def test_validate_rejects_unknown():
    with pytest.raises(ValueError, match="contract"):
        validate_contract("clim")


def test_site_accepts_a_legacy_tree_without_provenance(tmp_path):
    assert_dataset_contract(_config_dir(tmp_path, None), "site")


def test_site_accepts_provenance_without_a_contract_block(tmp_path):
    assert_dataset_contract(_config_dir(tmp_path, "absent"), "site")


def test_site_rejects_a_servable_tree(tmp_path):
    with pytest.raises(ValueError, match="servable"):
        assert_dataset_contract(_config_dir(tmp_path, {"name": "servable"}), "site")


def test_servable_accepts_a_servable_tree(tmp_path):
    assert_dataset_contract(_config_dir(tmp_path, {"name": "servable"}), "servable")


def test_servable_rejects_a_tree_without_provenance(tmp_path):
    with pytest.raises(ValueError, match="RUN_PROVENANCE"):
        assert_dataset_contract(_config_dir(tmp_path, None), "servable")


def test_servable_rejects_provenance_without_a_contract_block(tmp_path):
    with pytest.raises(ValueError, match="feature_contract"):
        assert_dataset_contract(_config_dir(tmp_path, "absent"), "servable")


def test_rejects_an_unknown_contract_before_reading_anything(tmp_path):
    with pytest.raises(ValueError, match="contract"):
        assert_dataset_contract(tmp_path / "does-not-exist", "clim")
