from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.static_features.products.soil import DEPTHS, PROPERTIES, soil_result, texture_class
from src.static_features.results import NO_DATA, OK, validate


def _property_cells(prop, raw, count=16.0, all_count=16.0, site="A"):
    row = {"all_count": all_count}
    for depth in DEPTHS:
        row[f"{prop}_{depth}_mean_mean"] = raw
        row[f"{prop}_{depth}_mean_count"] = count
    return pd.DataFrame([row], index=pd.Index([site], name="site_name"))


def _all_properties(**raw_by_property):
    return {prop: _property_cells(prop, raw_by_property.get(prop, 100.0)) for prop in PROPERTIES}


def test_soil_applies_each_conversion():
    result = soil_result(_all_properties(bdod=140.0, sand=400.0, nitrogen=250.0, soc=300.0, phh2o=65.0, cfvo=120.0))
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
