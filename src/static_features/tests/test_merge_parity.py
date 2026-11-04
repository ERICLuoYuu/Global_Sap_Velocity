from __future__ import annotations

import numpy as np
import pandas as pd
from src.static_features.checks import merge_parity
from src.static_features.checks.merge_parity import compare, main, static_columns_of
from src.static_features.results import RunContext


def _daily(site="A", days=("2010-01-01", "2010-01-02"), **columns):
    frame = pd.DataFrame(
        {"site_name": site, "TIMESTAMP": list(days), "sap_velocity": 1.0, "elevation": 100.0}, index=range(len(days))
    )
    return frame.assign(**columns)


def test_parity_passes_when_only_static_columns_change():
    report = compare(_daily(), _daily(elevation=120.0), ["elevation"])

    assert report["same_keys"] is True
    assert report["dynamic_columns_differing"] == {}
    assert report["static_columns_differing"] == {"elevation": 2}
    assert report["added_columns"] == []
    assert report["removed_columns"] == []


def test_parity_reports_a_changed_dynamic_column():
    report = compare(_daily(), _daily(sap_velocity=[1.0, 2.0]), ["elevation"])

    assert report["same_keys"] is True
    assert report["dynamic_columns_differing"] == {"sap_velocity": 1}


def test_parity_reports_a_lost_row():
    report = compare(_daily(), _daily(days=("2010-01-01",)), ["elevation"])

    assert report["same_keys"] is False
    assert report["lost_keys"] == 1
    assert report["gained_keys"] == 0


def test_parity_reports_a_gained_row():
    report = compare(_daily(days=("2010-01-01",)), _daily(), ["elevation"])

    assert report["same_keys"] is False
    assert report["lost_keys"] == 0
    assert report["gained_keys"] == 1


def test_parity_compares_rows_by_key_not_position():
    old = _daily(sap_velocity=[1.0, 2.0])
    new = old.iloc[::-1].reset_index(drop=True)
    report = compare(old, new, [])

    assert report["same_keys"] is True
    assert report["dynamic_columns_differing"] == {}


def test_parity_treats_both_empty_as_the_same():
    report = compare(_daily(sap_velocity=[np.nan, 1.0]), _daily(sap_velocity=[np.nan, 1.0]), [])
    assert report["dynamic_columns_differing"] == {}


def test_parity_counts_one_side_empty_as_a_difference():
    report = compare(_daily(sap_velocity=[np.nan, 1.0]), _daily(sap_velocity=[2.0, 1.0]), [])
    assert report["dynamic_columns_differing"] == {"sap_velocity": 1}


def test_parity_names_added_and_removed_columns():
    report = compare(_daily(old_column=1.0), _daily(new_column=2.0), [])

    assert report["added_columns"] == ["new_column"]
    assert report["removed_columns"] == ["old_column"]


def test_parity_compares_text_columns():
    report = compare(_daily(pft="ENF"), _daily(pft=["ENF", "DBF"]), [])
    assert report["dynamic_columns_differing"] == {"pft": 1}


def test_static_columns_hold_the_soil_props_but_not_era5_soil_temperature():
    columns = ["elevation", "soil_sand_0_5", "soil_theta_wp", "soil_temperature_level_1", "sap_velocity", "stand_age"]
    assert static_columns_of(columns) == ["elevation", "soil_sand_0_5", "soil_theta_wp", "stand_age"]


def _tree(tmp_path, name, **columns):
    daily = tmp_path / name / "daily"
    daily.mkdir(parents=True)
    _daily(**columns).to_csv(daily / "A_daily.csv", index=False)
    return tmp_path / name


def _run(tmp_path, monkeypatch, off, on, new_columns=("wood_density",)):
    output = tmp_path / "static_1km" / "scale"
    output.mkdir(parents=True)
    pd.DataFrame(columns=["site_name", *new_columns]).to_csv(output / "new_static_features.csv", index=False)
    context = RunContext(sites=pd.DataFrame({"site_name": ["A"]}), raw_root=tmp_path, output_root=output, project="p")
    monkeypatch.setattr(merge_parity.RunContext, "for_scale", classmethod(lambda cls, scale, project: context))
    return main(["--scale", "scale", "--before", str(off), "--after", str(on)]), output / "checks"


def test_main_exits_zero_when_only_the_expected_things_differ(tmp_path, monkeypatch):
    off = _tree(tmp_path, "off")
    on = _tree(tmp_path, "on", elevation=120.0, wood_density=0.5)
    code, checks = _run(tmp_path, monkeypatch, off, on)

    assert code == 0
    table = pd.read_csv(checks / "merge_parity.csv").set_index("column")
    assert table.loc["elevation", "kind"] == "static_differs"
    assert table.loc["elevation", "rows"] == 2
    assert table.loc["wood_density", "kind"] == "added"


def test_main_exits_nonzero_and_names_a_dynamic_column(tmp_path, monkeypatch, caplog):
    off = _tree(tmp_path, "off")
    on = _tree(tmp_path, "on", sap_velocity=2.0, wood_density=0.5)
    code, _ = _run(tmp_path, monkeypatch, off, on)

    assert code == 1
    assert "sap_velocity" in caplog.text


def test_main_exits_nonzero_when_a_new_column_did_not_arrive(tmp_path, monkeypatch, caplog):
    off = _tree(tmp_path, "off")
    on = _tree(tmp_path, "on")
    code, _ = _run(tmp_path, monkeypatch, off, on)

    assert code == 1
    assert "wood_density" in caplog.text


def test_main_exits_nonzero_when_a_site_is_on_one_side_only(tmp_path, monkeypatch, caplog):
    off = _tree(tmp_path, "off")
    on = _tree(tmp_path, "on", wood_density=0.5)
    _daily(site="B").to_csv(off / "daily" / "B_daily.csv", index=False)
    code, _ = _run(tmp_path, monkeypatch, off, on)

    assert code == 1
    assert "B_daily.csv" in caplog.text


def test_main_exits_nonzero_when_a_tree_holds_no_daily_files(tmp_path, monkeypatch):
    off = tmp_path / "off"
    on = tmp_path / "on"
    (off / "daily").mkdir(parents=True)
    (on / "daily").mkdir(parents=True)
    code, _ = _run(tmp_path, monkeypatch, off, on)
    assert code == 1


def test_main_exits_nonzero_on_an_added_column_the_file_does_not_name(tmp_path, monkeypatch, caplog):
    off = _tree(tmp_path, "off")
    on = _tree(tmp_path, "on", wood_density=0.5, wood_density_sum=1.0)
    code, _ = _run(tmp_path, monkeypatch, off, on)

    assert code == 1
    assert "wood_density_sum" in caplog.text


def test_main_exits_nonzero_on_a_removed_column(tmp_path, monkeypatch, caplog):
    off = _tree(tmp_path, "off", old_column=1.0)
    on = _tree(tmp_path, "on", wood_density=0.5)
    code, _ = _run(tmp_path, monkeypatch, off, on)

    assert code == 1
    assert "old_column" in caplog.text


def test_the_aggregate_across_sites_is_not_a_site_file(tmp_path):
    daily = tmp_path / "daily"
    daily.mkdir()
    (daily / "AAA_BBB_daily.csv").write_text("site_name,TIMESTAMP\n")
    (daily / "all_biomes_merged_daily.csv").write_text("site_name,TIMESTAMP\n")
    assert list(merge_parity._daily_files(str(tmp_path))) == ["AAA_BBB_daily.csv"]


def test_parity_counts_a_static_value_that_became_empty():
    report = compare(_daily(), _daily(elevation=np.nan), ["elevation"])

    assert report["static_columns_emptied"] == {"elevation": 2}


def test_a_static_value_that_only_moved_or_arrived_is_not_emptied():
    moved = compare(_daily(), _daily(elevation=120.0), ["elevation"])
    arrived = compare(_daily(elevation=np.nan), _daily(), ["elevation"])

    assert moved["static_columns_emptied"] == {}
    assert arrived["static_columns_emptied"] == {}


def test_main_exits_nonzero_naming_site_and_column_when_a_static_value_empties(tmp_path, monkeypatch, caplog):
    off = _tree(tmp_path, "off")
    on = _tree(tmp_path, "on", elevation=np.nan, wood_density=0.5)
    code, checks = _run(tmp_path, monkeypatch, off, on)

    assert code == 1
    assert "A: elevation" in caplog.text
    table = pd.read_csv(checks / "merge_parity.csv")
    assert set(table.loc[table["column"] == "elevation", "kind"]) == {"static_differs", "emptied"}


def test_main_allows_a_named_site_to_empty_and_still_reports_it(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(merge_parity, "EXPECTED_EMPTIED", {"A": {"columns": ("elevation",), "reason": "because"}})
    off = _tree(tmp_path, "off")
    on = _tree(tmp_path, "on", elevation=np.nan, wood_density=0.5)
    with caplog.at_level("WARNING"):
        code, _ = _run(tmp_path, monkeypatch, off, on)

    assert code == 0
    assert "A: elevation" in caplog.text and "because" in caplog.text


def test_main_stops_when_a_named_site_empties_another_column(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(merge_parity, "EXPECTED_EMPTIED", {"A": {"columns": ("slope",), "reason": "because"}})
    off = _tree(tmp_path, "off")
    on = _tree(tmp_path, "on", elevation=np.nan, wood_density=0.5)
    code, _ = _run(tmp_path, monkeypatch, off, on)

    assert code == 1
    assert "A: elevation" in caplog.text


def test_no_site_may_lose_a_static_value_now_that_soil_gaps_are_filled():
    # CHN_YUN_YUN was named while its cell had no soc or bdod; it now borrows them from a cell 0.18 km away.
    assert merge_parity.EXPECTED_EMPTIED == {}
