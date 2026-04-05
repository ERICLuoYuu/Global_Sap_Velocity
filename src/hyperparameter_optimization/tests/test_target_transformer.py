"""Tests for TargetTransformer extracted module."""

import numpy as np
import pytest

from src.hyperparameter_optimization.target_transformer import TargetTransformer


class TestTargetTransformerInit:
    """Test initialization and validation."""

    def test_valid_methods(self):
        for method in TargetTransformer.VALID_METHODS:
            t = TargetTransformer(method=method)
            assert t.method == method

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Invalid method"):
            TargetTransformer(method="invalid")

    def test_default_method_is_log1p(self):
        t = TargetTransformer()
        assert t.method == "log1p"

    def test_repr(self):
        t = TargetTransformer(method="sqrt")
        assert "sqrt" in repr(t)
        assert "fitted=False" in repr(t)


class TestRoundTrip:
    """Test that inverse_transform(transform(y)) ≈ y for all methods."""

    @pytest.fixture
    def positive_data(self):
        rng = np.random.RandomState(42)
        return rng.exponential(scale=5.0, size=200) + 0.1

    @pytest.fixture
    def mixed_data(self):
        rng = np.random.RandomState(42)
        return rng.normal(loc=5.0, scale=2.0, size=200)

    @pytest.mark.parametrize("method", ["log1p", "sqrt", "box-cox", "yeo-johnson", "none"])
    def test_round_trip(self, method, positive_data):
        t = TargetTransformer(method=method)
        y_t = t.fit_transform(positive_data)
        y_back = t.inverse_transform(y_t)
        np.testing.assert_allclose(y_back, positive_data, rtol=1e-5, atol=1e-5)

    def test_yeo_johnson_with_negatives(self, mixed_data):
        t = TargetTransformer(method="yeo-johnson")
        y_t = t.fit_transform(mixed_data)
        y_back = t.inverse_transform(y_t)
        np.testing.assert_allclose(y_back, mixed_data, rtol=1e-5, atol=1e-5)

    def test_none_is_identity(self, positive_data):
        t = TargetTransformer(method="none")
        y_t = t.fit_transform(positive_data)
        np.testing.assert_array_equal(y_t, positive_data)


class TestEdgeCases:
    """Test edge cases: zeros, single values, large arrays."""

    def test_log1p_with_zeros(self):
        y = np.array([0.0, 0.0, 1.0, 2.0])
        t = TargetTransformer(method="log1p")
        y_t = t.fit_transform(y)
        assert y_t[0] == 0.0  # log1p(0) = 0
        y_back = t.inverse_transform(y_t)
        np.testing.assert_allclose(y_back, y, atol=1e-10)

    def test_sqrt_with_zeros(self):
        y = np.array([0.0, 1.0, 4.0, 9.0])
        t = TargetTransformer(method="sqrt")
        y_t = t.fit_transform(y)
        np.testing.assert_allclose(y_t, [0.0, 1.0, 2.0, 3.0], atol=1e-10)

    def test_single_value(self):
        y = np.array([5.0])
        t = TargetTransformer(method="log1p")
        y_t = t.fit_transform(y)
        y_back = t.inverse_transform(y_t)
        np.testing.assert_allclose(y_back, y, atol=1e-10)

    def test_boxcox_unfitted_raises(self):
        t = TargetTransformer(method="box-cox")
        with pytest.raises(RuntimeError, match="must be fitted"):
            t.transform(np.array([1.0, 2.0]))

    def test_yeo_johnson_unfitted_raises(self):
        t = TargetTransformer(method="yeo-johnson")
        with pytest.raises(RuntimeError, match="must be fitted"):
            t.transform(np.array([1.0, 2.0]))


class TestSerialization:
    """Test get_params / from_params round-trip."""

    @pytest.mark.parametrize("method", ["log1p", "sqrt", "box-cox", "yeo-johnson", "none"])
    def test_params_round_trip(self, method):
        rng = np.random.RandomState(42)
        y = rng.exponential(scale=5.0, size=200) + 0.1

        t1 = TargetTransformer(method=method)
        t1.fit(y)
        params = t1.get_params()

        t2 = TargetTransformer.from_params(params)
        assert t2.method == t1.method
        assert t2._fitted == t1._fitted
        assert t2._lambda == t1._lambda
        assert t2._shift == t1._shift

    def test_from_params_preserves_transform(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        t1 = TargetTransformer(method="log1p")
        t1.fit(y)

        t2 = TargetTransformer.from_params(t1.get_params())
        np.testing.assert_array_equal(t1.transform(y), t2.transform(y))


class TestBoundaryValues:
    """Round 1: Edge cases and boundary values."""

    def test_large_array_round_trip(self):
        """Stress test with 1M elements."""
        rng = np.random.RandomState(42)
        y = rng.exponential(scale=10.0, size=100_000) + 0.01
        t = TargetTransformer(method="log1p")
        y_back = t.inverse_transform(t.fit_transform(y))
        np.testing.assert_allclose(y_back, y, rtol=1e-5)

    def test_all_zeros(self):
        """log1p and sqrt should handle all-zero input."""
        y = np.zeros(10)
        for method in ["log1p", "sqrt", "none"]:
            t = TargetTransformer(method=method)
            y_t = t.fit_transform(y)
            y_back = t.inverse_transform(y_t)
            np.testing.assert_allclose(y_back, y, atol=1e-10)

    def test_constant_array(self):
        """Constant positive array should round-trip."""
        y = np.full(50, 3.14)
        for method in ["log1p", "sqrt", "none"]:
            t = TargetTransformer(method=method)
            y_back = t.inverse_transform(t.fit_transform(y))
            np.testing.assert_allclose(y_back, y, atol=1e-8)

    def test_very_small_values(self):
        """Values near machine epsilon."""
        y = np.array([1e-15, 1e-10, 1e-5, 1e-1])
        t = TargetTransformer(method="log1p")
        y_back = t.inverse_transform(t.fit_transform(y))
        np.testing.assert_allclose(y_back, y, rtol=1e-5)

    def test_very_large_values(self):
        """Very large values."""
        y = np.array([1e6, 1e8, 1e10])
        t = TargetTransformer(method="log1p")
        y_back = t.inverse_transform(t.fit_transform(y))
        np.testing.assert_allclose(y_back, y, rtol=1e-5)

    def test_boxcox_negative_input_shifts(self):
        """Box-Cox should auto-shift negative data."""
        y = np.array([-5.0, -1.0, 0.0, 1.0, 5.0, 10.0])
        t = TargetTransformer(method="box-cox")
        y_t = t.fit_transform(y)
        assert t._shift > 0
        y_back = t.inverse_transform(y_t)
        np.testing.assert_allclose(y_back, y, rtol=1e-4)

    def test_sqrt_negative_input_shifts(self):
        """Sqrt should auto-shift negative data."""
        y = np.array([-2.0, 0.0, 1.0, 4.0])
        t = TargetTransformer(method="sqrt")
        t.fit(y)
        assert t._shift > 0
        y_t = t.transform(y)
        y_back = t.inverse_transform(y_t)
        np.testing.assert_allclose(y_back, y, atol=1e-6)


class TestNegativePaths:
    """Round 2: Negative paths and error handling."""

    def test_transform_before_fit_log1p(self):
        """log1p doesn't require fitting — should not raise."""
        t = TargetTransformer(method="log1p")
        result = t.transform(np.array([1.0, 2.0]))
        assert len(result) == 2

    def test_inverse_transform_boxcox_unfitted(self):
        """inverse_transform without fit on box-cox should raise."""
        t = TargetTransformer(method="box-cox")
        with pytest.raises(RuntimeError, match="must be fitted"):
            t.inverse_transform(np.array([1.0, 2.0]))

    def test_inverse_transform_yeo_unfitted(self):
        """inverse_transform without fit on yeo-johnson should raise."""
        t = TargetTransformer(method="yeo-johnson")
        with pytest.raises(RuntimeError, match="must be fitted"):
            t.inverse_transform(np.array([1.0, 2.0]))

    def test_from_params_missing_key(self):
        """from_params with incomplete dict should raise KeyError."""
        with pytest.raises(KeyError):
            TargetTransformer.from_params({"method": "log1p"})

    def test_fit_empty_array(self):
        """Fitting on empty array — scipy may raise."""
        t = TargetTransformer(method="log1p")
        # log1p doesn't use scipy, so fit should succeed
        t.fit(np.array([]))
        assert t._fitted

    def test_list_input_accepted(self):
        """Plain list input should be auto-converted to ndarray."""
        t = TargetTransformer(method="log1p")
        y_t = t.fit_transform([1.0, 2.0, 3.0])
        assert isinstance(y_t, np.ndarray)
        assert len(y_t) == 3
