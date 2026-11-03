from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from src.static_features import results
from src.static_features.checks import reproduce
from src.static_features.checks.reproduce import classify
from src.static_features.results import NO_DATA, OK, ProductResult, RunContext, write_part
from src.static_features.sites import site_list_hash


def test_classify_hand_computed():
    old = pd.Series([1.0, np.nan, 2.0, np.nan, 5.0])
    new = pd.Series([1.0 + 1e-8, np.nan, np.nan, 3.0, 6.0])
    assert list(classify(old, new)) == ["match", "match", "value_to_empty", "empty_to_value", "differs"]


def test_classify_tolerance_is_relative():
    old = pd.Series([1000.0, 1000.0])
    new = pd.Series([1000.0005, 1000.01])
    assert list(classify(old, new)) == ["match", "differs"]


def test_classify_compares_strings_exactly():
    old = pd.Series(["Cfb", None, "Dfc", "Cfb", None])
    new = pd.Series(["Cfb", None, "dfc", None, "Cfa"])
    assert list(classify(old, new)) == ["match", "match", "differs", "value_to_empty", "empty_to_value"]


def test_classify_compares_booleans():
    assert list(classify(pd.Series([True, False]), pd.Series([True, True]))) == ["match", "differs"]


SITES = pd.DataFrame({"site_name": ["A", "B"], "lat": [1.0, 2.0], "lon": [3.0, 4.0]})
SCALE = "demo"


def _part(root, site_hash, product, columns):
    """columns: name -> [value at A, value at B]."""
    index = pd.Index(["A", "B"], name="site_name")
    values = pd.DataFrame(columns, index=index)
    status = values.notna().replace({True: OK, False: NO_DATA})
    result = ProductResult(
        product=product,
        values=values,
        status=status,
        coverage=pd.DataFrame(index=index),
        by_year=None,
        extras={},
        provenance={},
    )
    write_part(result, root, site_hash)


def _current(extracted, directory, name, columns):
    path = extracted / directory / SCALE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"site_name": ["A", "B"], **columns}).to_csv(path, index=False)


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """Parts and current files that agree everywhere, with the tree laid out as on Palma."""
    extracted = tmp_path / "extracted_data"
    context = RunContext(sites=SITES, raw_root=tmp_path, output_root=extracted / "static_1km" / SCALE, project="p")
    monkeypatch.setattr(results.RunContext, "for_scale", staticmethod(lambda scale, project: context))
    site_hash = site_list_hash(SITES)

    def build(stand_age_part=(np.nan, 5.0), stand_age_current=(np.nan, 5.0)):
        _part(context.parts_root, site_hash, "worldclim", {"bio1": [1.5, 2.5], "koppen_name": ["Cfb", "Dfc"]})
        _part(context.parts_root, site_hash, "fan", {"root_depth": [3.5, 4.5]})
        _part(context.parts_root, site_hash, "stand_age", {"stand_age_TC000": list(stand_age_part)})
        _current(
            extracted,
            "env_site_data",
            "site_info_with_env_data.csv",
            {"bio1": [1.5, 2.5], "koppen_name": ["Cfb", "Dfc"]},
        )
        _current(extracted, "root_depth_site_data", "site_info_with_root_depth_fan2017.csv", {"root_depth": [3.5, 4.5]})
        _current(extracted, "terrain_site_data", "stand_age_data.csv", {"stand_age_TC000": list(stand_age_current)})
        return context

    return build


def test_reproduce_exits_zero_when_every_site_matches(tree):
    context = tree()
    assert reproduce.main(["--scale", SCALE]) == 0
    assert (context.output_root / "checks" / "reproduce_summary.csv").exists()


def test_reproduce_exits_nonzero_and_names_site_on_empty_to_value(tree, caplog):
    context = tree(stand_age_part=(7.0, 5.0))
    with caplog.at_level(logging.ERROR):
        assert reproduce.main(["--scale", SCALE]) != 0
    assert "empty_to_value" in caplog.text and "stand_age_TC000" in caplog.text and "A" in caplog.text
    report = pd.read_csv(context.output_root / "checks" / "reproduce.csv")
    assert list(report["site_name"]) == ["A"] and list(report["outcome"]) == ["empty_to_value"]


def test_reproduce_exits_nonzero_on_a_lost_value(tree):
    tree(stand_age_current=(9.0, 5.0))
    assert reproduce.main(["--scale", SCALE]) != 0


def test_reproduce_refuses_a_part_from_another_site_list(tree, tmp_path, monkeypatch, caplog):
    context = tree()
    moved = SITES.assign(lat=[1.5, 2.0])
    monkeypatch.setattr(
        results.RunContext,
        "for_scale",
        staticmethod(
            lambda scale, project: RunContext(
                sites=moved, raw_root=tmp_path, output_root=context.output_root, project="p"
            )
        ),
    )
    with caplog.at_level(logging.ERROR):
        assert reproduce.main(["--scale", SCALE]) != 0
    assert "site list" in caplog.text


EXPECTED_AT_A = {"A": {"outcomes": (reproduce.DIFFERS,), "reason": "because"}}


def test_a_named_difference_exits_zero_and_is_still_reported(tree, monkeypatch, caplog):
    monkeypatch.setattr(reproduce, "EXPECTED_DIFFERENCES", EXPECTED_AT_A)
    context = tree(stand_age_part=(7.0, 5.0), stand_age_current=(9.0, 5.0))

    with caplog.at_level(logging.WARNING):
        assert reproduce.main(["--scale", SCALE]) == 0
    assert "stand_age_TC000" in caplog.text and "because" in caplog.text
    report = pd.read_csv(context.output_root / "checks" / "reproduce.csv")
    assert list(report["site_name"]) == ["A"]


def test_another_outcome_at_a_named_site_still_stops(tree, monkeypatch, caplog):
    monkeypatch.setattr(reproduce, "EXPECTED_DIFFERENCES", EXPECTED_AT_A)
    tree(stand_age_part=(7.0, 5.0))

    with caplog.at_level(logging.ERROR):
        assert reproduce.main(["--scale", SCALE]) != 0
    assert "empty_to_value" in caplog.text


def test_a_difference_at_another_site_still_stops(tree, monkeypatch):
    monkeypatch.setattr(reproduce, "EXPECTED_DIFFERENCES", {"B": EXPECTED_AT_A["A"]})
    tree(stand_age_part=(7.0, 5.0), stand_age_current=(9.0, 5.0))

    assert reproduce.main(["--scale", SCALE]) != 0


def test_the_named_sites_each_give_a_reason():
    assert set(reproduce.EXPECTED_DIFFERENCES) == {"CHE_DAV_SEE", "CHN_YUN_YUN", "GBR_DEV_CON", "GBR_DEV_DRO"}
    assert all(entry["reason"] and entry["outcomes"] for entry in reproduce.EXPECTED_DIFFERENCES.values())
    # Only the moved Devilla stands may gain a value: it replaces the merge's fallback to ground st_age.
    gaining = {site for site, entry in reproduce.EXPECTED_DIFFERENCES.items() if reproduce.EMPTY_TO_VALUE in entry["outcomes"]}
    assert gaining == {"GBR_DEV_CON", "GBR_DEV_DRO"}
