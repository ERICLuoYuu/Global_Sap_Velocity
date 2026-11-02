import pytest

from src.feature_contract.definitions import SERVABLE_CONTRACT
from src.forward_selection.feature_registry import CANDIDATE_FEATURES, MANDATORY_FEATURES
from src.hyperparameter_optimization.shap_constants import PFT_COLUMNS
from src.serving.base_columns import BASE_COLUMNS
from src.serving.dependencies import (
    base_dependencies,
    feature_history,
    is_hourly_feature,
    servability_report,
    spin_up_days,
    synthetic_cell,
)
from src.serving.engineer import engineer_cell

FEATURES = [
    "ta_lag1d",
    "precip_sum_7d",
    "tropical",
    "boreal",
    "Year sin",
    "et0",
    "ENF",
    "ta_roll7d_mean",
    "sw_in_roll14d_std",
    "cumulative_gdd",
    "days_since_rain",
    "volumetric_soil_water_layer_1_zscore",
]


@pytest.fixture(scope="module")
def dependencies():
    return base_dependencies(FEATURES)


@pytest.fixture(scope="module")
def history():
    return feature_history(FEATURES)


def test_synthetic_cell_carries_every_base_column():
    assert set(BASE_COLUMNS) <= set(synthetic_cell().columns)


def test_dependencies_are_measured_from_the_feature_code(dependencies):
    assert dependencies["ta_lag1d"] == ("ta",)
    assert dependencies["precip_sum_7d"] == ("total_precipitation_hourly",)
    assert dependencies["tropical"] == ("latitude",)
    assert dependencies["boreal"] == ("latitude",)
    assert dependencies["Year sin"] == ("solar_TIMESTAMP",)
    assert dependencies["ENF"] == ("pft",)
    assert dependencies["volumetric_soil_water_layer_1_zscore"] == ("volumetric_soil_water_layer_1_zscore",)


def test_fallback_inputs_are_found_by_perturbation(dependencies):
    assert {"rh_max", "rh_min", "ta_max", "ta_min", "elevation", "ext_rad", "sw_in", "ws"} <= set(dependencies["et0"])


def test_history_needed_by_lags_windows_and_record_starts(history):
    assert history["ta_lag1d"].days == 1
    assert history["ta_roll7d_mean"].days == 6
    assert history["sw_in_roll14d_std"].days == 13
    assert history["tropical"].days == 0
    assert history["cumulative_gdd"].never_servable
    assert history["days_since_rain"].event_dependent
    assert not history["ta_lag1d"].event_dependent


def test_unknown_or_unservable_feature_is_refused():
    with pytest.raises(KeyError):
        base_dependencies(["not_a_feature"])
    with pytest.raises(KeyError, match="soil_moisture_rel"):
        base_dependencies(["ta", "soil_moisture_rel"], SERVABLE_CONTRACT)


def test_spin_up_refuses_a_record_dependent_feature():
    report = servability_report(["ta_lag1d", "cumulative_gdd"])
    with pytest.raises(ValueError, match="cumulative_gdd"):
        spin_up_days(report)
    assert spin_up_days(report[report["feature"] == "ta_lag1d"]) == 1


def test_every_daily_ffs_candidate_is_produced_or_known_unservable():
    produced = set(engineer_cell(synthetic_cell(), SERVABLE_CONTRACT).columns) | set(PFT_COLUMNS)
    names = list(MANDATORY_FEATURES) + [n for group in CANDIDATE_FEATURES.values() for n in group]
    daily = [n for n in names if not is_hourly_feature(n)]
    assert sorted(set(daily) - produced) == ["soil_dry", "soil_moisture_rel"]
