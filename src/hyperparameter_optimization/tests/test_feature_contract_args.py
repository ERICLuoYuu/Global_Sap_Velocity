"""--FEATURE_CONTRACT on the ML trainer's parser, and its record in FINAL_config; --SWC_SCALE is gone."""

import argparse
import sys

import pytest

from src.hyperparameter_optimization.training_utils import build_run_provenance, parse_args

BASE_ARGV = ["trainer", "--selected_features", "features.json"]


def _parse(monkeypatch, *extra):
    monkeypatch.setattr(sys, "argv", BASE_ARGV + list(extra))
    return parse_args()


def test_the_default_is_servable(monkeypatch):
    """The Stage-4 merge builds servable by default; a consumer defaulting to the other contract
    would make every fresh tree fail the dataset gate until someone passed a flag."""
    assert _parse(monkeypatch).FEATURE_CONTRACT == "servable"


def test_site_is_still_accepted(monkeypatch):
    assert _parse(monkeypatch, "--FEATURE_CONTRACT", "site").FEATURE_CONTRACT == "site"


def test_servable_is_accepted(monkeypatch):
    assert _parse(monkeypatch, "--FEATURE_CONTRACT", "servable").FEATURE_CONTRACT == "servable"


def test_an_unknown_contract_is_rejected_by_the_parser(monkeypatch):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "--FEATURE_CONTRACT", "clim")


def test_provenance_records_the_contract():
    args = argparse.Namespace(FEATURE_CONTRACT="servable", selected_features="f.json")
    assert build_run_provenance(args)["feature_contract"] == "servable"


def test_provenance_of_a_parser_without_the_flag_records_none():
    assert build_run_provenance(argparse.Namespace())["feature_contract"] is None


def test_the_swc_scale_flag_is_gone(monkeypatch):
    """Soil moisture has one definition per quantity per layer; there is no scale left to choose."""
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "--SWC_SCALE", "ratio")


def test_provenance_records_no_swc_scale():
    assert "swc_scale" not in build_run_provenance(argparse.Namespace(FEATURE_CONTRACT="servable"))
