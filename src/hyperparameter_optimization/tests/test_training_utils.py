"""Tests for training_utils extracted module."""

import numpy as np
import pandas as pd
import pytest

from src.hyperparameter_optimization.training_utils import (
    add_time_features,
    convert_windows_to_numpy,
    create_spatial_groups,
)


class TestAddTimeFeatures:
    """Test cyclical time feature creation."""

    @pytest.fixture
    def time_df(self):
        dates = pd.date_range("2020-01-01", periods=365, freq="D")
        return pd.DataFrame({"solar_TIMESTAMP": dates, "ta": np.random.randn(365)})

    def test_adds_time_columns(self, time_df):
        result = add_time_features(time_df)
        for col in ["Day sin", "Day cos", "Year sin", "Year cos"]:
            assert col in result.columns, f"Missing {col}"

    def test_cyclical_range(self, time_df):
        result = add_time_features(time_df)
        # Sin/cos should be in [-1, 1]
        assert result["Year sin"].between(-1, 1).all()
        assert result["Year cos"].between(-1, 1).all()


class TestConvertWindowsToNumpy:
    """Test window list to numpy conversion."""

    def test_basic_conversion(self):
        rng = np.random.RandomState(42)
        windows = [
            (rng.randn(5, 3), rng.randn(1)),
            (rng.randn(5, 3), rng.randn(1)),
        ]
        X, y = convert_windows_to_numpy(windows)
        # Reshapes to 2D: (n_windows, input_width * n_features)
        assert X.shape == (2, 15)
        assert y.shape == (2,)

    def test_empty_input(self):
        X, y = convert_windows_to_numpy([])
        assert len(X) == 0
        assert len(y) == 0


class TestCreateSpatialGroups:
    """Test spatial group creation."""

    def test_returns_tuple(self):
        lat = np.array([51.0] * 50 + [52.0] * 50)
        lon = np.array([7.0] * 50 + [8.0] * 50)
        groups, info = create_spatial_groups(lat, lon, method="grid", n_groups=2)
        assert isinstance(groups, np.ndarray)
        assert len(groups) == 100

    def test_group_count(self):
        lat = np.array([51.0] * 50 + [52.0] * 50)
        lon = np.array([7.0] * 50 + [8.0] * 50)
        groups, _ = create_spatial_groups(lat, lon, method="grid", n_groups=4)
        # Should have at most n_groups unique groups
        assert len(np.unique(groups)) <= 4


class TestBoundaryValues:
    """Round 1: Edge cases and boundary values."""

    def test_single_timestamp(self):
        """Single date produces valid time features."""
        df = pd.DataFrame({"solar_TIMESTAMP": pd.to_datetime(["2020-06-15"]), "ta": [25.0]})
        result = add_time_features(df)
        assert result["Year sin"].notna().all()
        assert result["Day cos"].notna().all()

    def test_windows_single_element(self):
        """Single window element."""
        windows = [(np.array([[1.0, 2.0]]), np.array([3.0]))]
        X, y = convert_windows_to_numpy(windows)
        assert X.shape == (1, 2)
        assert y.shape == (1,)

    def test_spatial_groups_single_point(self):
        """Single coordinate."""
        groups, info = create_spatial_groups(np.array([51.0]), np.array([7.0]), method="grid")
        assert len(groups) == 1
        assert len(info) == 1

    def test_spatial_groups_invalid_method(self):
        """Unknown method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            create_spatial_groups(np.array([51.0]), np.array([7.0]), method="invalid")


class TestNegativePaths:
    """Round 2: Negative paths and error handling."""

    def test_add_time_features_missing_column(self):
        """Missing datetime column raises ValueError."""
        df = pd.DataFrame({"ta": [1.0, 2.0]})
        with pytest.raises(ValueError, match="not found"):
            add_time_features(df, datetime_column="solar_TIMESTAMP")

    def test_add_time_features_nan_timestamps(self):
        """NaN timestamps are dropped with warning."""
        df = pd.DataFrame(
            {
                "solar_TIMESTAMP": [pd.Timestamp("2020-01-01"), pd.NaT, pd.Timestamp("2020-01-03")],
                "ta": [1.0, 2.0, 3.0],
            }
        )
        result = add_time_features(df)
        assert len(result) == 2  # NaT row dropped

    def test_add_time_features_non_datetime_index(self):
        """Non-convertible index raises ValueError."""
        df = pd.DataFrame({"ta": [1.0]}, index=["not_a_date"])
        with pytest.raises(ValueError, match="cannot be converted"):
            add_time_features(df, datetime_column=None)

    def test_spatial_groups_default_method(self):
        """Default method with balanced groups."""
        lat = np.array([51.0, 52.0, 53.0, 54.0])
        lon = np.array([7.0, 8.0, 9.0, 10.0])
        counts = np.array([100, 50, 75, 25])
        groups, info = create_spatial_groups(lat, lon, data_counts=counts, n_groups=2, method="default")
        assert len(groups) == 4
        assert len(np.unique(groups)) <= 2
