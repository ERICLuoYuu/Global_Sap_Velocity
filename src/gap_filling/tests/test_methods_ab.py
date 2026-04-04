import numpy as np
import pandas as pd
import pytest


def _make_gapped_series(gap_size=3, total=100, freq="h"):
    """Create a series with a synthetic gap in the middle."""
    idx = pd.date_range("2020-01-01", periods=total, freq=freq)
    vals = np.sin(np.linspace(0, 4 * np.pi, total)) + 2  # positive sap-like
    s = pd.Series(vals, index=idx)
    mid = total // 2
    s.iloc[mid : mid + gap_size] = np.nan
    return s, mid, gap_size


class TestGroupA:
    def test_linear_fills_gap(self):
        from src.gap_filling.methods.interpolation import fill_linear

        s, mid, gs = _make_gapped_series()
        filled = fill_linear(s)
        assert filled.isna().sum() == 0
        assert filled.iloc[mid] >= 0

    def test_linear_no_mutation(self):
        from src.gap_filling.methods.interpolation import fill_linear

        s, _, _ = _make_gapped_series()
        original_nans = s.isna().sum()
        _ = fill_linear(s)
        assert s.isna().sum() == original_nans  # original unchanged

    def test_cubic_fills_gap(self):
        from src.gap_filling.methods.interpolation import fill_cubic

        s, _, _ = _make_gapped_series()
        filled = fill_cubic(s)
        assert filled.isna().sum() == 0

    def test_akima_fills_gap(self):
        from src.gap_filling.methods.interpolation import fill_akima

        s, _, _ = _make_gapped_series()
        filled = fill_akima(s)
        assert filled.isna().sum() == 0

    def test_nearest_fills_gap(self):
        from src.gap_filling.methods.interpolation import fill_nearest

        s, _, _ = _make_gapped_series()
        filled = fill_nearest(s)
        assert filled.isna().sum() == 0

    def test_all_non_negative(self):
        """All Group A methods must return non-negative values."""
        from src.gap_filling.methods.interpolation import (
            fill_akima,
            fill_cubic,
            fill_linear,
            fill_nearest,
        )

        s, _, _ = _make_gapped_series()
        for fn in [fill_linear, fill_cubic, fill_akima, fill_nearest]:
            filled = fn(s)
            assert (filled >= 0).all(), f"{fn.__name__} returned negative values"

    def test_short_series_akima_fallback(self):
        """Akima needs >=4 points; should fallback to linear with fewer."""
        from src.gap_filling.methods.interpolation import fill_akima

        idx = pd.date_range("2020-01-01", periods=3, freq="h")
        s = pd.Series([1.0, np.nan, 3.0], index=idx)
        filled = fill_akima(s)
        assert filled.isna().sum() == 0


class TestGroupB:
    def test_mdv_fills_gap(self):
        from src.gap_filling.methods.statistical import fill_mdv

        s, _, _ = _make_gapped_series(gap_size=6, total=200)
        filled = fill_mdv(s)
        assert filled.isna().sum() == 0

    def test_rolling_fills_gap(self):
        from src.gap_filling.methods.statistical import fill_rolling_mean

        s, _, _ = _make_gapped_series()
        filled = fill_rolling_mean(s)
        assert filled.isna().sum() == 0

    def test_stl_fills_gap(self):
        from src.gap_filling.methods.statistical import fill_stl

        s, _, _ = _make_gapped_series(total=200)
        filled = fill_stl(s)
        assert filled.isna().sum() == 0

    def test_stl_short_series_fallback(self):
        """STL needs >= 2*period points; should fallback to rolling."""
        from src.gap_filling.methods.statistical import fill_stl

        s, _, _ = _make_gapped_series(total=20)
        filled = fill_stl(s)
        assert filled.isna().sum() == 0
