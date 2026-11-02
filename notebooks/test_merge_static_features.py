"""The merge always reads the full 1 km static set, and stops rather than fill in NaN columns."""

import inspect
import sys

import pandas as pd
import pytest

import notebooks.merge_gap_filled_hourly_orginal as merge
from notebooks.merge_gap_filled_hourly_orginal import (
    STATIC_FILES,
    add_new_static,
    load_new_static,
    load_stand_age_map,
    merge_sap_env_data_site,
    new_static_row,
    paths,
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


def test_returns_the_five_files_under_the_root(tmp_path):
    root = _root_with(tmp_path, STATIC_FILES.values())
    found = static_input_paths(root)

    assert set(found) == {"env", "soil", "stand_age", "fan", "new"}
    assert found["env"] == root / "site_info_with_env_data.csv"
    assert found["soil"] == root / "soilgrids_data.csv"
    assert found["stand_age"] == root / "stand_age_data.csv"
    assert found["fan"] == root / "site_info_with_root_depth_fan2017.csv"
    assert found["new"] == root / "new_static_features.csv"


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
