"""Two figures were deleted because their inputs carry no information.

Deleted 2026-09-16, on the finding that both are computed from a per-pixel
Z-score, whose temporal mean is 0 by construction:

  taylor_global.png   fed on zscore.mean(dim="time"). Measured over the April
                      run: max|mean| = 2.2e-6 (sap velocity), 2.9e-7 (GLEAM),
                      mean std exactly 1.000000. Every product therefore landed
                      on the same point of the diagram -- the figure was plotting
                      floating-point noise, not skill.

  trend_agreement.png fit a straight line through 3 annual means of a variable
                      whose 3-year mean is 0. With 3 points and one constraint
                      the slope sign is close to arbitrary, and the "agreement"
                      map then counts how often two arbitrary signs coincide.

These tests exist so neither comes back without the underlying input changing
first. A Taylor diagram over a *spatial* pattern (seasonal amplitude, say) would
be legitimate; one over the temporal mean of a Z-score never is.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import src.transpiration_comparison.comparison.temporal_comp as temporal_comp
import src.transpiration_comparison.pipeline as pipeline


class TestTaylorIsGone:
    def test_the_module_is_deleted(self):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("src.transpiration_comparison.visualization.taylor")

    def test_the_pipeline_no_longer_writes_it(self):
        source = Path(pipeline.__file__).read_text()
        assert "taylor" not in source.lower()  # nosec B101


class TestTrendAgreementIsGone:
    def test_the_function_is_deleted(self):
        assert not hasattr(temporal_comp, "trend_agreement")  # nosec B101

    def test_the_pipeline_no_longer_writes_it(self):
        source = Path(pipeline.__file__).read_text()
        assert "trend_agreement" not in source  # nosec B101
