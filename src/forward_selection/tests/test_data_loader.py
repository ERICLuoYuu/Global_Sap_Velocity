"""Tests for data_loader.py — cache roundtrip."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.forward_selection.data_loader import load_cache


class TestLoadCache:
    def _make_cache(self, tmp_path: Path) -> Path:
        cache_path = tmp_path / "test_cache.npz"
        rng = np.random.RandomState(42)
        np.savez_compressed(
            cache_path,
            X=rng.randn(100, 5).astype(np.float32),
            y=rng.rand(100).astype(np.float32),
            groups=np.repeat([0, 1, 2, 3, 4], 20),
            pfts_encoded=np.repeat([0, 1, 2], [40, 30, 30]),
            feature_names=np.array(["f1", "f2", "f3", "f4", "f5"]),
            pft_categories=np.array(["ENF", "DBF", "EBF"]),
        )
        return cache_path

    def test_loads_all_keys(self, tmp_path: Path) -> None:
        cache_path = self._make_cache(tmp_path)
        data = load_cache(cache_path)
        expected_keys = {"X", "y", "groups", "pfts_encoded", "feature_names", "pft_categories"}
        assert set(data.keys()) == expected_keys

    def test_shapes_match(self, tmp_path: Path) -> None:
        cache_path = self._make_cache(tmp_path)
        data = load_cache(cache_path)
        assert data["X"].shape == (100, 5)
        assert data["y"].shape == (100,)
        assert data["groups"].shape == (100,)
        assert data["pfts_encoded"].shape == (100,)

    def test_feature_names_are_list(self, tmp_path: Path) -> None:
        cache_path = self._make_cache(tmp_path)
        data = load_cache(cache_path)
        assert isinstance(data["feature_names"], list)
        assert data["feature_names"] == ["f1", "f2", "f3", "f4", "f5"]

    def test_pft_categories_are_list(self, tmp_path: Path) -> None:
        cache_path = self._make_cache(tmp_path)
        data = load_cache(cache_path)
        assert isinstance(data["pft_categories"], list)
        assert data["pft_categories"] == ["ENF", "DBF", "EBF"]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_cache(tmp_path / "nonexistent.npz")

    def test_dtypes_preserved(self, tmp_path: Path) -> None:
        cache_path = self._make_cache(tmp_path)
        data = load_cache(cache_path)
        assert data["X"].dtype == np.float32
        assert data["y"].dtype == np.float32
