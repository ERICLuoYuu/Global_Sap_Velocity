from __future__ import annotations

import numpy as np
import pandas as pd
from src.static_features.products.canopy import canopy_result
from src.static_features.results import NO_DATA, OK, validate


def _cells(mean, p95, count, all_count):
    row = {"h_mean": mean, "h_p95": p95, "h_count": float(count), "all_count": float(all_count)}
    return pd.DataFrame([row], index=pd.Index(["A"], name="site_name"))


def test_canopy_zero_is_a_value_not_missing():
    result = canopy_result(_cells(0.0, 0.0, 700000, 700000))

    assert result.values.loc["A", "canopy_height_m"] == 0.0
    assert result.status.loc["A", "canopy_height_m"] == OK
    assert result.values.loc["A", "canopy_height_p95_m"] == 0.0
    assert result.status.loc["A", "canopy_height_p95_m"] == OK
    validate(result, ["A"])


def test_canopy_reads_mean_and_percentile_from_their_own_outputs():
    result = canopy_result(_cells(12.5, 28.0, 350000, 700000))

    assert result.values.loc["A", "canopy_height_m"] == 12.5
    assert result.values.loc["A", "canopy_height_p95_m"] == 28.0
    assert result.coverage.loc["A", "canopy_height_m"] == 0.5
    assert result.coverage.loc["A", "canopy_height_p95_m"] == 0.5


def test_cell_with_no_land_is_no_data():
    result = canopy_result(_cells(np.nan, np.nan, 0, 700000))

    assert result.status.loc["A", "canopy_height_m"] == NO_DATA
    assert result.coverage.loc["A", "canopy_height_m"] == 0.0
    validate(result, ["A"])
