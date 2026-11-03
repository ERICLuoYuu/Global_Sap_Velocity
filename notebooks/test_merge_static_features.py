"""The merge always reads the full 1 km static set, and stops rather than fill in NaN columns."""

import inspect
import json
import sys

import numpy as np
import pandas as pd
import pytest

import notebooks.merge_gap_filled_hourly_orginal as merge
from notebooks.merge_gap_filled_hourly_orginal import (
    STATIC_FILES,
    add_new_static,
    calculate_soil_hydraulics,
    get_weighted_soil_props,
    load_new_static,
    load_stand_age_map,
    merge_sap_env_data_site,
    new_static_row,
    paths,
    read_static_provenance,
    require_static_sites,
    sap_site_names,
    static_exact_columns,
    static_features_root,
    static_input_paths,
)
from src.feature_contract.merge_contract import FeatureContractError
from src.static_features import results


def _root_with(tmp_path, names):
    for name in names:
        (tmp_path / name).write_text("site_name\n", encoding="utf-8")
    return tmp_path


def _new_static(path, names=("A", "B")):
    frame = pd.DataFrame({"site_name": list(names), "wood_density": 0.5, "root_depth_han_type": "NETTe"})
    frame.to_csv(path, index=False)
    return path


def test_root_is_where_the_package_assembles(monkeypatch):
    monkeypatch.setattr(results, "load_sites", lambda *_: pd.DataFrame())
    assert static_features_root() == results.RunContext.for_scale(paths.scale, "project").output_root


def test_returns_the_static_files_under_the_root(tmp_path):
    root = _root_with(tmp_path, STATIC_FILES.values())
    found = static_input_paths(root)

    assert set(found) == {"env", "soil", "stand_age", "fan", "new", "provenance"}
    assert found["env"] == root / "site_info_with_env_data.csv"
    assert found["soil"] == root / "soilgrids_data.csv"
    assert found["stand_age"] == root / "stand_age_data.csv"
    assert found["fan"] == root / "site_info_with_root_depth_fan2017.csv"
    assert found["new"] == root / "new_static_features.csv"
    assert found["provenance"] == root / "provenance.json"


def test_missing_file_raises_naming_it(tmp_path):
    root = _root_with(tmp_path, [name for name in STATIC_FILES.values() if name != "soilgrids_data.csv"])
    with pytest.raises(FileNotFoundError, match="soilgrids_data.csv"):
        static_input_paths(root)


def test_names_every_absent_file(tmp_path):
    with pytest.raises(FileNotFoundError) as caught:
        static_input_paths(tmp_path)
    for name in STATIC_FILES.values():
        assert name in str(caught.value)


def test_merge_stops_when_the_features_are_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(merge, "static_features_root", lambda: tmp_path / "absent")
    monkeypatch.setattr(merge, "load_cube", lambda _: None)
    with pytest.raises(FileNotFoundError, match="new_static_features.csv"):
        merge_sap_env_data_site(tmp_path / "out")


def test_there_is_no_way_to_turn_the_features_off(monkeypatch, tmp_path):
    assert "static_root" not in inspect.signature(merge_sap_env_data_site).parameters
    monkeypatch.setattr(merge, "merge_sap_env_data_site", lambda output, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["merge", "--output-dir", str(tmp_path), "--static-root", str(tmp_path)])
    with pytest.raises(SystemExit):
        merge.main()


def test_stand_age_map_averages_the_given_file(tmp_path):
    path = tmp_path / "stand_age_data.csv"
    pd.DataFrame({"site_name": ["A"], "stand_age_TC000": [1.0], "stand_age_TC010": [3.0]}).to_csv(path, index=False)
    assert load_stand_age_map(path) == {"A": 2.0}


def test_stand_age_file_without_age_columns_stops_naming_it(tmp_path):
    path = tmp_path / "stand_age_data.csv"
    pd.DataFrame({"site_name": ["A"], "age": [1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="stand_age_data.csv"):
        load_stand_age_map(path)


def test_new_static_missing_a_site_raises_naming_it(tmp_path):
    path = _new_static(tmp_path / "new.csv")
    with pytest.raises(ValueError, match="CCC"):
        load_new_static(path, ["A", "B", "CCC"])


def test_new_static_repeating_a_site_raises_naming_it(tmp_path):
    path = _new_static(tmp_path / "new.csv", names=("A", "A"))
    with pytest.raises(ValueError, match="A"):
        load_new_static(path, ["A"])


def test_new_static_returns_the_frame_and_its_value_columns(tmp_path):
    frame, columns = load_new_static(_new_static(tmp_path / "new.csv"), ["A"])

    assert columns == ["wood_density", "root_depth_han_type"]
    assert frame.index.name == "site_name"
    assert frame.loc["B", "wood_density"] == 0.5


def test_new_columns_join_the_static_columns():
    assert "wood_density" in static_exact_columns(["wood_density"])


def test_the_earlier_static_columns_are_all_kept():
    expected = {
        "latitude",
        "longitude",
        "elevation",
        "slope",
        "aspect_sin",
        "aspect_cos",
        "mean_annual_temp",
        "mean_annual_precip",
        "temp_seasonality",
        "precip_seasonality",
        "canopy_height",
        "stand_age",
        "root_depth",
        "infiltration_depth",
        "deep_drainage",
        "drainage_frequency",
        "regolith_flush_rate",
        "gw_residence_time",
    }
    assert set(static_exact_columns([])) == expected


def test_static_exact_columns_can_be_extended_by_the_caller():
    columns = static_exact_columns([])
    columns.append("pl_code")
    assert "pl_code" not in static_exact_columns([])


def test_row_adds_every_column_to_every_row(tmp_path):
    new_static, _ = load_new_static(_new_static(tmp_path / "new.csv"), ["A"])
    frame = pd.DataFrame({"x": [1, 2]}, index=[7, 9])
    out = add_new_static(frame, new_static_row(new_static, "A"))

    assert list(out.columns) == ["x", "wood_density", "root_depth_han_type"]
    assert out["wood_density"].tolist() == [0.5, 0.5]
    assert out["root_depth_han_type"].tolist() == ["NETTe", "NETTe"]
    assert out.index.tolist() == [7, 9]


def test_a_new_column_clashing_with_the_frame_ends_the_run(tmp_path):
    new_static, _ = load_new_static(_new_static(tmp_path / "new.csv"), ["A"])
    frame = pd.DataFrame({"wood_density": [1.0]})
    with pytest.raises(FeatureContractError, match="wood_density"):
        add_new_static(frame, new_static_row(new_static, "A"))


SITE_FILES = ("env", "soil", "stand_age", "fan", "new")


def _site_files(tmp_path, **names_by_key):
    """Each site file holding sites A and B, unless ``names_by_key`` says otherwise."""
    found = {}
    for key in SITE_FILES:
        found[key] = tmp_path / STATIC_FILES[key]
        pd.DataFrame({"site_name": names_by_key.get(key, ["A", "B"]), "x": 1.0}).to_csv(found[key], index=False)
    return found


def test_every_site_once_in_every_file_passes(tmp_path):
    require_static_sites(_site_files(tmp_path), ["A", "B"])


def test_a_file_may_hold_sites_with_no_sap_file(tmp_path):
    require_static_sites(_site_files(tmp_path), ["A"])


@pytest.mark.parametrize("key", SITE_FILES)
def test_a_site_absent_from_a_static_file_stops_the_merge_naming_both(tmp_path, key):
    with pytest.raises(FeatureContractError) as caught:
        require_static_sites(_site_files(tmp_path, **{key: ["A"]}), ["A", "B"])
    assert STATIC_FILES[key] in str(caught.value)
    assert "'B'" in str(caught.value)


@pytest.mark.parametrize("key", SITE_FILES)
def test_a_site_repeated_in_a_static_file_stops_the_merge_naming_both(tmp_path, key):
    with pytest.raises(FeatureContractError) as caught:
        require_static_sites(_site_files(tmp_path, **{key: ["A", "B", "B"]}), ["A", "B"])
    assert STATIC_FILES[key] in str(caught.value)
    assert "'B'" in str(caught.value)


def test_every_file_with_a_problem_is_named_at_once(tmp_path):
    with pytest.raises(FeatureContractError) as caught:
        require_static_sites(_site_files(tmp_path, soil=["A"], fan=["A"]), ["A", "B"])
    assert STATIC_FILES["soil"] in str(caught.value)
    assert STATIC_FILES["fan"] in str(caught.value)


def test_sites_are_named_as_the_main_loop_names_them(tmp_path):
    for stem in ("AUS_CAN_ST1_EUC_sapf_data_outliers_removed", "ARG_MAZ_sapf_data_outliers_removed"):
        (tmp_path / f"{stem}.csv").write_text("x\n", encoding="utf-8")
    assert sap_site_names(tmp_path) == ["ARG_MAZ", "AUS_CAN_ST1_EUC"]


def test_the_site_loop_has_no_soft_path_for_a_static_miss():
    source = inspect.getsource(merge_sap_env_data_site)
    assert "curr_site_info.empty" not in source
    assert "soil_data.empty" not in source
    assert "else np.nan" not in source.split("fan2017_features = {")[1].split("}")[0]
    assert "require_static_sites(" in source
    assert "read_static_provenance(" in source


def _soil(**values):
    columns = {}
    for prop in ("sand", "clay", "soc", "bdod", "cfvo"):
        for depth in ("0_5", "5_15", "15_30", "30_60", "60_100"):
            columns[f"{prop}_{depth}"] = values.get(f"{prop}_{depth}", values.get(prop, 10.0))
    return pd.Series(columns)


def test_a_soil_property_with_no_data_stays_empty_rather_than_zero():
    props = get_weighted_soil_props(_soil(sand=60.0, clay=20.0, soc=np.nan, bdod=np.nan, cfvo=5.0))

    assert props["sand"] == pytest.approx(60.0)
    assert props["clay"] == pytest.approx(20.0)
    assert np.isnan(props["soc"])
    assert np.isnan(props["bdod"])


def test_hydraulics_without_organic_matter_stay_empty():
    props = get_weighted_soil_props(_soil(sand=60.0, clay=20.0, soc=np.nan, cfvo=5.0))
    organic_matter = props["soc"] / 10.0 * 1.72
    theta = calculate_soil_hydraulics(props["sand"] / 100.0, props["clay"] / 100.0, organic_matter, props["cfvo"])

    assert all(np.isnan(value) for value in theta)


def test_a_property_is_weighted_only_over_the_depths_that_hold_it():
    row = _soil(soc=np.nan, soc_0_5=10.0, soc_5_15=40.0)

    # (10 * 5 cm + 40 * 10 cm) / 15 cm, not divided by the full 100 cm.
    assert get_weighted_soil_props(row)["soc"] == pytest.approx(30.0)


def test_a_complete_soil_row_keeps_its_thickness_weighting():
    row = _soil(sand_0_5=10.0, sand_5_15=20.0, sand_15_30=30.0, sand_30_60=40.0, sand_60_100=50.0)

    # (10*5 + 20*10 + 30*15 + 40*30 + 50*40) / 100
    assert get_weighted_soil_props(row)["sand"] == pytest.approx(39.0)


def _provenance(tmp_path, site_hash="today"):
    path = tmp_path / "provenance.json"
    path.write_text(
        json.dumps(
            {
                "site_list_hash": site_hash,
                "code_commit": "abc",
                "products": {"soil": {"code": {"commit": "def", "package_clean": True}}, "yang": {}},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_the_static_provenance_says_what_was_read(tmp_path):
    assert read_static_provenance(_provenance(tmp_path), "today") == {
        "site_list_hash": "today",
        "code_commit": "abc",
        "product_code": {"soil": {"commit": "def", "package_clean": True}, "yang": None},
    }


def test_a_root_from_another_site_list_stops_the_merge_naming_it(tmp_path):
    with pytest.raises(FeatureContractError, match="provenance.json"):
        read_static_provenance(_provenance(tmp_path, site_hash="earlier"), "today")


class PastTheChecksError(Exception):
    """Raised by the first load after the root's checks, so a run stops there without any real data."""


@pytest.fixture
def merge_on_a_root(monkeypatch, tmp_path):
    """Run the merge against a root in ``tmp_path`` whose site list hashes to "today"; sap sites are A and B."""
    sap_directories = []

    def run(site_hash="today", **names_by_key):
        _site_files(tmp_path, **names_by_key)
        _provenance(tmp_path, site_hash=site_hash)
        monkeypatch.setattr(merge, "static_features_root", lambda: tmp_path)
        monkeypatch.setattr(merge, "load_cube", lambda _: None)
        monkeypatch.setattr(merge, "load_sites", lambda *_: None)
        monkeypatch.setattr(merge, "site_list_hash", lambda _: "today")
        monkeypatch.setattr(merge, "sap_site_names", lambda directory: sap_directories.append(directory) or ["A", "B"])

        def stop(*_):
            raise PastTheChecksError

        monkeypatch.setattr(merge, "load_stand_age_map", stop)
        merge_sap_env_data_site(tmp_path / "out")

    run.sap_directories = sap_directories
    return run


def test_the_merge_stops_on_a_site_absent_from_a_static_file(merge_on_a_root):
    with pytest.raises(FeatureContractError, match="soilgrids_data.csv"):
        merge_on_a_root(soil=["A"])


def test_the_merge_stops_on_a_root_from_another_site_list(merge_on_a_root):
    with pytest.raises(FeatureContractError, match="provenance.json"):
        merge_on_a_root(site_hash="earlier")


def test_the_merge_goes_on_from_a_good_root_and_checks_the_sites_it_will_loop_over(merge_on_a_root):
    with pytest.raises(PastTheChecksError):
        merge_on_a_root()
    assert merge_on_a_root.sap_directories == [paths.sap_outliers_removed_dir]
