"""A feature cache or feature list from before the soil-moisture cleanup is refused, not silently used.

A cache built before 2026-09-14 holds the retired bare and swc_layer*_norm columns and none of the
level / anomaly columns. Selection on it would drop the mandatory anomaly (absent) and every other
soil-moisture feature (no longer in the registry) without a word; train_from_cache would train on
whatever retired columns the list names.
"""

import json

import numpy as np
import pytest

import src.forward_selection.train_from_cache as train_from_cache
from src.forward_selection.data_loader import load_cache

STALE_NAMES = ["vpd", "volumetric_soil_water_layer_1", "swc_layer3_norm"]
CANONICAL_NAMES = ["vpd", "volumetric_soil_water_layer_1_raw", "volumetric_soil_water_layer_1_zscore"]


def _write_cache(path, names, with_matrix=True):
    n = 4
    arrays = {
        "y": np.arange(n, dtype=float),
        "groups": np.zeros(n, dtype=int),
        "pfts_encoded": np.zeros(n, dtype=int),
        "feature_names": np.array(names),
        "pft_categories": np.array(["ENF"]),
    }
    if with_matrix:
        arrays["X"] = np.arange(n * len(names), dtype=float).reshape(n, len(names))
    np.savez_compressed(path, **arrays)
    return path


def _write_list(path, names):
    path.write_text(json.dumps({"selected_features": names}))
    return path


class TestLoadCache:
    def test_a_cache_naming_retired_features_is_refused(self, tmp_path):
        cache = _write_cache(tmp_path / "feature_cache.npz", STALE_NAMES)
        with pytest.raises(ValueError) as err:
            load_cache(cache)
        message = str(err.value)
        assert "retired" in message
        assert "--build_cache" in message
        assert "swc_layer3_norm ->" in message
        assert "volumetric_soil_water_layer_1 ->" in message

    def test_the_refusal_comes_before_the_matrix_is_read(self, tmp_path):
        """No X in the file: reading the matrix first would raise KeyError instead."""
        cache = _write_cache(tmp_path / "feature_cache.npz", STALE_NAMES, with_matrix=False)
        with pytest.raises(ValueError, match="retired"):
            load_cache(cache)

    def test_a_canonical_cache_loads(self, tmp_path):
        """POSITIVE CONTROL: the guard must not refuse a cache built after the cleanup."""
        cache = _write_cache(tmp_path / "feature_cache.npz", CANONICAL_NAMES)
        data = load_cache(cache)
        assert data["feature_names"] == CANONICAL_NAMES
        assert data["X"].shape == (4, 3)


class TestTrainFromCache:
    def test_a_stale_cache_is_refused(self, tmp_path):
        cache = _write_cache(tmp_path / "feature_cache.npz", STALE_NAMES)
        selected = _write_list(tmp_path / "selected.json", ["vpd"])
        with pytest.raises(ValueError, match="retired"):
            train_from_cache.load_selected_features(cache, selected)

    def test_a_stale_feature_list_is_refused_by_name(self, tmp_path):
        """On a new cache the retired name is merely absent, which used to be only a warning."""
        cache = _write_cache(tmp_path / "feature_cache.npz", CANONICAL_NAMES)
        selected = _write_list(tmp_path / "selected.json", ["vpd", "volumetric_soil_water_layer_1"])
        with pytest.raises(ValueError) as err:
            train_from_cache.load_selected_features(cache, selected)
        assert "retired" in str(err.value)
        assert str(selected) in str(err.value)

    def test_selection_by_name_in_cache_order(self, tmp_path):
        """POSITIVE CONTROL: a canonical cache and list select exactly the named columns."""
        cache = _write_cache(tmp_path / "feature_cache.npz", CANONICAL_NAMES)
        selected = _write_list(tmp_path / "selected.json", ["volumetric_soil_water_layer_1_zscore", "vpd"])
        data, sel_idx, sel_names = train_from_cache.load_selected_features(cache, selected)
        assert sel_idx == [0, 2]
        assert sel_names == ["vpd", "volumetric_soil_water_layer_1_zscore"]
        np.testing.assert_array_equal(data["X"][:, sel_idx], data["X"][:, [0, 2]])
