from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from src.static_features.checks import agreement
from src.static_features.checks.agreement import PAIRS, check_units, main, pair_stats, report
from src.static_features.results import ExtractionFailure, RunContext

DRAWS = 200


def _good_provenance():
    return {
        "products": {
            "lusk": {
                "traits": {
                    "6": {"column": "root_depth_lusk", "unit": "m"},
                    "4": {"column": "wood_density_lusk", "unit": "g cm3"},
                    "14": {"column": "lusk_14", "unit": "mg g-1"},
                }
            }
        }
    }


def test_duplicate_pairs_collapse():
    a = pd.Series([1.0, 1.0, 1.0, 2.0, 3.0])
    b = pd.Series([2.0, 2.0, 2.0, 3.0, 5.0])
    assert pair_stats(a, b, draws=DRAWS)["n"] == 3


def test_a_shared_value_on_one_side_is_not_a_duplicate():
    assert pair_stats(pd.Series([1.0, 1.0, 2.0]), pd.Series([2.0, 3.0, 4.0]), draws=DRAWS)["n"] == 3


def test_missing_values_drop_the_pair():
    assert pair_stats(pd.Series([1.0, np.nan, 3.0, 4.0]), pd.Series([1.0, 2.0, np.nan, 4.0]), draws=DRAWS)["n"] == 2


def test_bias_and_rmse_hand_computed():
    stats = pair_stats(pd.Series([1.0, 2.0, 3.0]), pd.Series([2.0, 2.0, 2.0]), draws=DRAWS)
    assert stats["bias"] == 0.0 and stats["rmse"] == pytest.approx(0.8165, abs=1e-4)


def test_bias_is_a_minus_b():
    assert pair_stats(pd.Series([3.0, 4.0]), pd.Series([1.0, 1.0]), draws=DRAWS)["bias"] == 2.5


def test_pearson_and_spearman_hand_computed():
    stats = pair_stats(pd.Series([1.0, 2.0, 3.0, 4.0]), pd.Series([1.0, 4.0, 9.0, 16.0]), draws=DRAWS)

    assert stats["spearman"] == pytest.approx(1.0)
    assert stats["pearson"] == pytest.approx(0.98437, abs=1e-4)


def test_constant_series_gives_nan_r_not_a_crash():
    stats = pair_stats(pd.Series([1.0, 1.0, 1.0]), pd.Series([1.0, 2.0, 3.0]), draws=DRAWS)

    assert np.isnan(stats["pearson"]) and np.isnan(stats["spearman"])
    assert np.isnan(stats["pearson_low"]) and np.isnan(stats["pearson_high"])
    assert stats["n"] == 3


def test_no_pair_at_all_gives_nan_not_a_crash():
    stats = pair_stats(pd.Series([np.nan]), pd.Series([1.0]), draws=DRAWS)

    assert stats["n"] == 0
    assert np.isnan(stats["bias"]) and np.isnan(stats["rmse"]) and np.isnan(stats["bias_low"])


def test_bootstrap_is_repeatable_with_seed_42():
    rng = np.random.default_rng(1)
    a, b = pd.Series(rng.normal(size=40)), pd.Series(rng.normal(size=40))

    first = pair_stats(a, b, seed=42, draws=DRAWS)
    assert first == pair_stats(a, b, seed=42, draws=DRAWS)
    assert first != pair_stats(a, b, seed=7, draws=DRAWS)


def test_bootstrap_interval_brackets_the_estimate():
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(size=200))
    b = a + pd.Series(rng.normal(scale=0.5, size=200))
    stats = pair_stats(a, b, draws=DRAWS)

    for name in ("pearson", "spearman", "bias", "rmse"):
        assert stats[f"{name}_low"] <= stats[name] <= stats[f"{name}_high"], name


def test_agreement_refuses_an_unexpected_unit():
    provenance = _good_provenance()
    provenance["products"]["lusk"]["traits"]["6"]["unit"] = "cm"
    with pytest.raises(ExtractionFailure, match="root_depth_lusk"):
        check_units(provenance)


def test_agreement_accepts_the_expected_units():
    check_units(_good_provenance())


def test_agreement_refuses_when_a_trait_is_not_recorded():
    provenance = _good_provenance()
    del provenance["products"]["lusk"]["traits"]["4"]
    with pytest.raises(ExtractionFailure, match="wood_density_lusk"):
        check_units(provenance)


def test_pairs_are_the_pinned_four():
    assert PAIRS == (
        ("root_depth_lusk", "root_depth_han"),
        ("root_depth_lusk", "root_depth"),
        ("root_depth_han", "root_depth"),
        ("wood_density_lusk", "wood_density"),
    )


def test_report_has_a_row_per_igbp_and_overall():
    index = pd.Index(["A", "B", "C", "D"], name="site_name")
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 3.0, 2.0, 5.0]}, index=index)
    igbp = pd.Series(["ENF", "ENF", "EBF", np.nan], index=index)
    table = report(frame, [("x", "y")], igbp).set_index("group")

    assert sorted(table.index) == ["EBF", "ENF", "all"]
    assert table.loc["all", "n"] == 4  # a site with no class still counts overall
    assert table.loc["ENF", "n"] == 2
    assert table.loc["EBF", "n"] == 1
    assert table.loc["ENF", "bias"] == pytest.approx(-0.5)
    assert set(table["a"]) == {"x"} and set(table["b"]) == {"y"}


def _files(tmp_path, provenance):
    output = tmp_path / "static_1km" / "scale"
    output.mkdir(parents=True)
    names = ["A", "B", "C"]
    pd.DataFrame(
        {
            "site_name": names,
            "root_depth_lusk": [1.0, 2.0, 3.0],
            "root_depth_han": [1.0, 2.5, 3.5],
            "wood_density_lusk": [0.4, 0.5, 0.6],
            "wood_density": [0.5, 0.5, 0.7],
        }
    ).to_csv(output / "new_static_features.csv", index=False)
    pd.DataFrame({"site_name": names, "root_depth": [2.0, 3.0, 4.0]}).to_csv(
        output / "site_info_with_root_depth_fan2017.csv", index=False
    )
    (output / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    sites = pd.DataFrame({"site_name": names, "si_igbp": ["ENF", "ENF", "EBF"]})
    return RunContext(sites=sites, raw_root=tmp_path, output_root=output, project="p")


def _run(context, monkeypatch):
    monkeypatch.setattr(agreement.RunContext, "for_scale", classmethod(lambda cls, scale, project: context))
    monkeypatch.setattr(agreement, "DRAWS", DRAWS)
    return main(["--scale", "scale"])


def test_main_writes_every_pair_by_group(tmp_path, monkeypatch):
    context = _files(tmp_path, _good_provenance())

    assert _run(context, monkeypatch) == 0
    table = pd.read_csv(context.output_root / "checks" / "agreement.csv")
    assert len(table) == 4 * 3
    assert set(table["group"]) == {"all", "ENF", "EBF"}
    fan = table[(table["a"] == "root_depth_han") & (table["b"] == "root_depth") & (table["group"] == "all")]
    assert fan["bias"].iloc[0] == pytest.approx(-2 / 3)


def test_main_returns_nonzero_on_a_wrong_unit_and_writes_nothing(tmp_path, monkeypatch):
    provenance = _good_provenance()
    provenance["products"]["lusk"]["traits"]["4"]["unit"] = "kg m-3"
    context = _files(tmp_path, provenance)

    assert _run(context, monkeypatch) == 1
    assert not (context.output_root / "checks" / "agreement.csv").exists()
