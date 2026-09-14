"""--FEATURE_CONTRACT and --SWC_SCALE on the ML trainer's parser, and their record in FINAL_config."""

import argparse
import sys

import pytest

from src.hyperparameter_optimization.training_utils import build_run_provenance, parse_args

BASE_ARGV = ["trainer", "--selected_features", "features.json"]


def _parse(monkeypatch, *extra):
    monkeypatch.setattr(sys, "argv", BASE_ARGV + list(extra))
    return parse_args()


def test_default_contract_is_site(monkeypatch):
    assert _parse(monkeypatch).FEATURE_CONTRACT == "site"


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


def test_default_swc_scale_is_ratio(monkeypatch):
    assert _parse(monkeypatch).SWC_SCALE == "ratio"


@pytest.mark.parametrize("scale", ["zscore", "raw"])
def test_the_other_swc_scales_are_accepted(monkeypatch, scale):
    assert scale == _parse(monkeypatch, "--FEATURE_CONTRACT", "servable", "--SWC_SCALE", scale).SWC_SCALE


def test_an_unknown_swc_scale_is_rejected_by_the_parser(monkeypatch):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "--SWC_SCALE", "minmax")


def test_provenance_records_the_swc_scale():
    args = argparse.Namespace(FEATURE_CONTRACT="servable", SWC_SCALE="raw")
    assert build_run_provenance(args)["swc_scale"] == "raw"


def test_provenance_of_a_parser_without_the_swc_flag_records_none():
    assert build_run_provenance(argparse.Namespace())["swc_scale"] is None
