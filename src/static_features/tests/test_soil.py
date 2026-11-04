from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.static_features.neighbours import BORROWED_COLUMNS, Borrowed, Node
from src.static_features.products.soil import DEPTHS, PROPERTIES, cell_frame, empty_columns, soil_result, texture_class
from src.static_features.results import NEIGHBOUR, NO_DATA, OK, validate


def _property_cells(prop, raw, count=16.0, all_count=16.0, site="A"):
    row = {"all_count": all_count}
    for depth in DEPTHS:
        row[f"{prop}_{depth}_mean_mean"] = raw
        row[f"{prop}_{depth}_mean_count"] = count
    return pd.DataFrame([row], index=pd.Index([site], name="site_name"))


def _all_properties(**raw_by_property):
    return {prop: _property_cells(prop, raw_by_property.get(prop, 100.0)) for prop in PROPERTIES}


def test_soil_applies_each_conversion():
    result = soil_result(_all_properties(bdod=140.0, sand=400.0, nitrogen=2500.0, soc=300.0, phh2o=65.0, cfvo=120.0))
    values = result.values.loc["A"]

    assert values["bdod_0_5"] == pytest.approx(1.4)
    assert values["sand_0_5"] == pytest.approx(40.0)
    assert values["nitrogen_100_200"] == pytest.approx(2.5)
    assert values["soc_30_60"] == pytest.approx(30.0)
    assert values["phh2o_60_100"] == pytest.approx(6.5)
    assert values["cfvo_15_30"] == pytest.approx(12.0)


def test_soil_has_every_property_at_every_depth():
    result = soil_result(_all_properties())
    expected = {
        f"{prop}_{depth}"
        for prop in ("clay", "sand", "silt", "bdod", "cfvo", "soc", "phh2o", "nitrogen")
        for depth in ("0_5", "5_15", "15_30", "30_60", "60_100", "100_200")
    }
    assert set(result.values.columns) == expected | {"texture_class_0_5"}
    validate(result, ["A"])


def test_soil_depths_are_named_from_the_band_depths():
    assert DEPTHS == ("0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm")


def test_texture_class_matches_old_rules():
    assert texture_class(45, 20, 35) == "Clay"
    assert texture_class(5, 90, 5) == "Sand"
    assert texture_class(10, 5, 85) == "Silt"
    assert texture_class(30, 40, 30) == "Clay Loam"
    assert texture_class(22, 50, 28) == "Sandy Clay Loam"
    assert texture_class(10, 30, 60) == "Silt Loam"
    assert texture_class(10, 75, 15) == "Sandy Loam"
    assert texture_class(15, 40, 45) == "Loam"
    assert texture_class(None, 20, 35) is None
    assert texture_class(float("nan"), 20, 35) is None


def test_texture_class_comes_from_the_converted_percentages():
    result = soil_result(_all_properties(clay=450.0, sand=200.0, silt=350.0))
    assert result.values.loc["A", "texture_class_0_5"] == "Clay"
    assert result.status.loc["A", "texture_class_0_5"] == OK


def test_masked_soil_is_no_data_with_zero_coverage():
    cells = _all_properties()
    cells["clay"] = _property_cells("clay", np.nan, count=0.0)
    result = soil_result(cells)

    assert np.isnan(result.values.loc["A", "clay_0_5"])
    assert result.status.loc["A", "clay_0_5"] == NO_DATA
    assert result.coverage.loc["A", "clay_0_5"] == 0.0
    assert pd.isna(result.values.loc["A", "texture_class_0_5"])  # one fraction missing, no class
    assert result.status.loc["A", "texture_class_0_5"] == NO_DATA
    validate(result, ["A"])


def test_soil_coverage_is_per_property_and_depth():
    cells = _all_properties()
    cells["sand"] = _property_cells("sand", 400.0, count=12.0, all_count=16.0)
    result = soil_result(cells)
    assert result.coverage.loc["A", "sand_30_60"] == 0.75
    assert result.coverage.loc["A", "clay_30_60"] == 1.0


NODE = Node(distance_km=0.18, row=7, col=9, lat=1.5, lon=2.5)


def _empty_soc():
    cells = _all_properties()
    cells["soc"] = _property_cells("soc", np.nan, count=0.0)
    return cells


def test_empty_columns_are_the_value_columns_without_data():
    empty = empty_columns(soil_result(_empty_soc()))

    assert "texture_class_0_5" not in empty.columns
    assert bool(empty.loc["A", "soc_0_5"]) and bool(empty.loc["A", "soc_100_200"])
    assert not bool(empty.loc["A", "clay_0_5"])


def test_a_borrowed_value_takes_its_place_with_status_neighbour():
    result = soil_result(_empty_soc(), {"A": {"soc_0_5": Borrowed(12.5, NODE)}})

    assert result.values.loc["A", "soc_0_5"] == 12.5
    assert result.status.loc["A", "soc_0_5"] == NEIGHBOUR
    assert result.coverage.loc["A", "soc_0_5"] == 0.0  # coverage stays the site's own cell's
    assert np.isnan(result.values.loc["A", "soc_5_15"])
    assert result.status.loc["A", "soc_5_15"] == NO_DATA
    validate(result, ["A"])


def test_the_borrowed_values_are_listed_with_their_distance():
    table = soil_result(_empty_soc(), {"A": {"soc_0_5": Borrowed(12.5, NODE)}}).extras["soil_neighbours"]

    assert list(table.columns) == BORROWED_COLUMNS
    assert table.iloc[0].tolist() == ["A", "soc_0_5", 0.18, 1.5, 2.5]
    assert soil_result(_all_properties()).extras["soil_neighbours"].empty


def test_texture_from_a_borrowed_fraction_is_neighbour():
    cells = _all_properties(clay=450.0, sand=200.0, silt=350.0)
    cells["clay"] = _property_cells("clay", np.nan, count=0.0)
    result = soil_result(cells, {"A": {"clay_0_5": Borrowed(45.0, NODE)}})

    assert result.values.loc["A", "texture_class_0_5"] == "Clay"
    assert result.status.loc["A", "texture_class_0_5"] == NEIGHBOUR
    validate(result, ["A"])


def test_cell_frame_names_each_cell_once_in_the_order_given():
    nodes = [NODE, Node(distance_km=1.0, row=7, col=10, lat=1.5, lon=2.6)]
    frame = cell_frame(nodes)

    assert frame["row"].tolist() == [7, 7] and frame["col"].tolist() == [9, 10]
    assert frame["site_name"].is_unique
