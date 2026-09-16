"""The FFS registry and loader know soil moisture only by its level and anomaly columns."""

import pytest

import src.forward_selection.data_loader as data_loader
from src.feature_contract.definitions import RETIRED_SWC_FEATURES, SWC_LAYERS, swc_anomaly_col, swc_level_col
from src.forward_selection.feature_registry import (
    ADDITIONAL_FEATURES,
    CANDIDATE_FEATURES,
    INTERMEDIATE_ONLY,
    MANDATORY_FEATURES,
    get_all_expected_columns,
)

LEVELS = {swc_level_col(layer) for layer in SWC_LAYERS}
ANOMALIES = {swc_anomaly_col(layer) for layer in SWC_LAYERS}


def _soil_moisture(names):
    return {n for n in names if n.startswith("volumetric_soil_water_layer_") or n.startswith("swc_layer")}


class TestRegistry:
    def test_the_selectable_soil_moisture_is_exactly_level_and_anomaly_per_layer(self):
        assert _soil_moisture(get_all_expected_columns()) == LEVELS | ANOMALIES

    def test_the_top_layer_anomaly_is_mandatory_and_its_level_a_candidate(self):
        assert swc_anomaly_col(1) in MANDATORY_FEATURES
        assert swc_level_col(1) not in MANDATORY_FEATURES
        assert CANDIDATE_FEATURES[swc_level_col(1)] == [swc_level_col(1)]

    @pytest.mark.parametrize(
        "label,names",
        [
            ("MANDATORY_FEATURES", MANDATORY_FEATURES),
            ("ADDITIONAL_FEATURES", ADDITIONAL_FEATURES),
            ("INTERMEDIATE_ONLY", INTERMEDIATE_ONLY),
            ("get_all_expected_columns()", get_all_expected_columns()),
        ],
    )
    def test_no_registry_list_names_a_retired_feature(self, label, names):
        assert not set(names) & set(RETIRED_SWC_FEATURES), label

    def test_level_columns_are_features_not_intermediates(self):
        assert not LEVELS & set(INTERMEDIATE_ONLY)

    def test_every_soil_moisture_feature_is_loaded(self):
        assert LEVELS | ANOMALIES <= set(ADDITIONAL_FEATURES)


class _ReachedTheFiles(Exception):
    pass


def _stop_at_read_csv(*args, **kwargs):
    raise _ReachedTheFiles


class TestLoaderRefusesRetiredNames:
    @pytest.fixture
    def data_dir(self, tmp_path, monkeypatch):
        (tmp_path / "SITE_A_daily.csv").write_text("TIMESTAMP\n2020-01-01\n")
        monkeypatch.setattr(data_loader.pd, "read_csv", _stop_at_read_csv)
        return tmp_path

    def test_the_default_column_list_passes_the_guard(self, data_dir):
        # Reading is stubbed to fail AFTER the guard, so every site is dropped: reaching that is the pass.
        with pytest.raises(RuntimeError, match="No valid site data"):
            data_loader.load_and_cache_features(data_dir, data_dir / "cache.npz")

    def test_a_retired_name_is_refused_before_any_file_is_read(self, data_dir, monkeypatch):
        stale = list(data_loader.ADDITIONAL_FEATURES) + ["volumetric_soil_water_layer_1"]
        monkeypatch.setattr(data_loader, "ADDITIONAL_FEATURES", stale)
        with pytest.raises(ValueError, match="retired"):
            data_loader.load_and_cache_features(data_dir, data_dir / "cache.npz")
