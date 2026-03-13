"""Tests for AOA visualization / plotting utilities."""

import numpy as np
import pytest
from src.aoa.plotting import (
    plot_aoa_map,
    plot_di_histogram,
)


class TestPlotDIHistogram:
    """Tests for DI histogram with threshold line."""

    def test_returns_figure_and_axes(self):
        """Should return a matplotlib (fig, ax) tuple."""
        import matplotlib

        di = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        threshold = 2.0

        fig, ax = plot_di_histogram(di, threshold)

        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt = matplotlib.pyplot
        plt.close(fig)

    def test_threshold_line_drawn(self):
        """Axes should contain a vertical line at the threshold value."""
        di = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        threshold = 2.0

        fig, ax = plot_di_histogram(di, threshold)

        # Check that at least one vertical line exists at threshold
        vlines = [line for line in ax.get_lines() if _is_vertical_at(line, threshold)]
        assert len(vlines) >= 1, "No vertical line found at threshold"
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_includes_training_di(self):
        """When di_train is provided, histogram should have more patches."""
        di = np.array([0.5, 1.0, 1.5])
        di_train = np.array([0.3, 0.7, 1.1, 1.4])
        threshold = 2.0

        fig, ax = plot_di_histogram(di, threshold, di_train=di_train)

        # Should have rendered without error; at minimum we have bars
        assert len(ax.patches) > 0
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_custom_bins(self):
        """bins parameter should control histogram resolution."""
        di = np.random.default_rng(0).random(100) * 3
        threshold = 2.0

        fig1, ax1 = plot_di_histogram(di, threshold, bins=10)
        fig2, ax2 = plot_di_histogram(di, threshold, bins=50)

        # More bins → more patches
        assert len(ax2.patches) > len(ax1.patches)
        import matplotlib.pyplot as plt

        plt.close(fig1)
        plt.close(fig2)

    def test_does_not_show_by_default(self):
        """Figure should be created but not displayed (non-interactive)."""
        import matplotlib

        matplotlib.use("Agg")  # ensure non-interactive backend

        di = np.array([0.5, 1.0, 1.5])
        threshold = 2.0

        fig, ax = plot_di_histogram(di, threshold)
        # If we got here without error on Agg backend, it didn't try to show
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPlotAOAMap:
    """Tests for spatial AOA map (2D grid)."""

    def test_returns_figure_and_axes(self):
        """Should return a matplotlib (fig, ax) tuple."""
        import matplotlib

        lats = np.linspace(40, 50, 5)
        lons = np.linspace(-10, 0, 6)
        di_grid = np.random.default_rng(0).random((5, 6))
        threshold = 0.5

        fig, ax = plot_aoa_map(lats, lons, di_grid, threshold)

        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_grid_shape_mismatch_raises(self):
        """di_grid shape must be (len(lats), len(lons))."""
        lats = np.linspace(40, 50, 5)
        lons = np.linspace(-10, 0, 6)
        di_grid = np.random.default_rng(0).random((3, 3))  # wrong shape
        threshold = 0.5

        with pytest.raises(ValueError):
            plot_aoa_map(lats, lons, di_grid, threshold)

    def test_colorbar_present(self):
        """Figure should include a colorbar."""
        lats = np.linspace(40, 50, 5)
        lons = np.linspace(-10, 0, 6)
        di_grid = np.random.default_rng(0).random((5, 6))
        threshold = 0.5

        fig, ax = plot_aoa_map(lats, lons, di_grid, threshold)

        # Colorbar adds an extra axes to the figure
        assert len(fig.axes) >= 2
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_nan_values_handled(self):
        """NaN cells (e.g., ocean) should not crash the plot."""
        lats = np.linspace(40, 50, 5)
        lons = np.linspace(-10, 0, 6)
        di_grid = np.random.default_rng(0).random((5, 6))
        di_grid[0, :] = np.nan  # simulate ocean row
        threshold = 0.5

        fig, ax = plot_aoa_map(lats, lons, di_grid, threshold)
        # Should render without error
        import matplotlib.pyplot as plt

        plt.close(fig)


def _is_vertical_at(line, x_val, tol=1e-6):
    """Check if a matplotlib Line2D is a vertical line at x_val."""
    xdata = line.get_xdata()
    if len(xdata) < 2:
        return False
    return all(abs(x - x_val) < tol for x in xdata)
