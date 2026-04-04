"""Tests for pipeline orchestrator and CLI.

Requires matplotlib to be functional (pipeline.py imports visualization).
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# pipeline.py imports visualization modules which need matplotlib.pyplot.
# Stub it out if matplotlib is broken in this environment.
try:
    import matplotlib.pyplot  # noqa: F401
except (ImportError, AttributeError):
    sys.modules.setdefault("matplotlib", MagicMock())
    sys.modules.setdefault("matplotlib.pyplot", MagicMock())
    sys.modules.setdefault("matplotlib.colors", MagicMock())
    sys.modules.setdefault("cartopy", MagicMock())
    sys.modules.setdefault("cartopy.crs", MagicMock())

from src.transpiration_comparison.cli import main, setup_logging
from src.transpiration_comparison.config import (
    DEFAULT_DOMAIN,
    DEFAULT_PATHS,
    DEFAULT_PERIOD,
    PRODUCTS,
    Paths,
)
from src.transpiration_comparison.pipeline import (
    TranspirationComparisonPipeline,
    _get_product_handler,
)

# ── Product handler factory ───────────────────────────────────────


class TestGetProductHandler:
    def test_sap_velocity(self):
        h = _get_product_handler("sap_velocity", DEFAULT_PATHS, DEFAULT_PERIOD, DEFAULT_DOMAIN)
        assert h.config.name == "Sap Velocity (XGBoost)"

    def test_gleam(self):
        h = _get_product_handler("gleam", DEFAULT_PATHS, DEFAULT_PERIOD, DEFAULT_DOMAIN)
        assert h.config.short_name == "gleam"

    def test_era5land(self):
        h = _get_product_handler("era5land", DEFAULT_PATHS, DEFAULT_PERIOD, DEFAULT_DOMAIN)
        assert h.config.short_name == "era5"

    def test_pmlv2(self):
        h = _get_product_handler("pmlv2", DEFAULT_PATHS, DEFAULT_PERIOD, DEFAULT_DOMAIN)
        assert h.config.short_name == "pmlv2"

    def test_gldas(self):
        h = _get_product_handler("gldas", DEFAULT_PATHS, DEFAULT_PERIOD, DEFAULT_DOMAIN)
        assert h.config.short_name == "gldas"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown product"):
            _get_product_handler("nonexistent", DEFAULT_PATHS, DEFAULT_PERIOD, DEFAULT_DOMAIN)


# ── Pipeline init ─────────────────────────────────────────────────


class TestPipelineInit:
    def test_init_stores_config(self):
        p = TranspirationComparisonPipeline(
            paths=DEFAULT_PATHS,
            period=DEFAULT_PERIOD,
            domain=DEFAULT_DOMAIN,
        )
        assert p.paths is DEFAULT_PATHS
        assert p.period is DEFAULT_PERIOD
        assert p.domain is DEFAULT_DOMAIN

    def test_download_skips_sap_velocity(self, tmp_path):
        paths = Paths(
            base=tmp_path,
            products_dir=tmp_path / "products",
            preprocessed_dir=tmp_path / "preproc",
            figures_dir=tmp_path / "figs",
            reports_dir=tmp_path / "reports",
        )
        p = TranspirationComparisonPipeline(paths=paths, period=DEFAULT_PERIOD, domain=DEFAULT_DOMAIN)
        # download_products with only sap_velocity should not raise
        p.download_products(["sap_velocity"])

    def test_preprocess_skips_existing(self, tmp_path):
        preproc = tmp_path / "preproc"
        preproc.mkdir()
        # Create a fake preprocessed file
        dummy = preproc / "gleam_preprocessed.nc"
        dummy.write_text("fake")

        paths = Paths(
            base=tmp_path,
            products_dir=tmp_path / "products",
            preprocessed_dir=preproc,
            figures_dir=tmp_path / "figs",
            reports_dir=tmp_path / "reports",
        )
        p = TranspirationComparisonPipeline(paths=paths, period=DEFAULT_PERIOD, domain=DEFAULT_DOMAIN)
        # Should skip gleam because file exists (no error)
        p.preprocess_products(["gleam"])

    def test_run_comparison_fails_without_data(self, tmp_path):
        preproc = tmp_path / "preproc"
        preproc.mkdir()
        paths = Paths(
            base=tmp_path,
            products_dir=tmp_path / "products",
            preprocessed_dir=preproc,
            predictions_parquet=tmp_path / "parquet",
            figures_dir=tmp_path / "figs",
            reports_dir=tmp_path / "reports",
        )
        p = TranspirationComparisonPipeline(paths=paths, period=DEFAULT_PERIOD, domain=DEFAULT_DOMAIN)
        with pytest.raises(RuntimeError, match="No preprocessed data"):
            p.run_comparison("spatial")


# ── CLI ───────────────────────────────────────────────────────────


class TestSetupLogging:
    def test_verbose_sets_debug(self):
        import logging

        root = logging.getLogger()
        root.handlers.clear()  # basicConfig is no-op if handlers exist
        setup_logging(verbose=True)
        assert root.level == logging.DEBUG

    def test_normal_sets_info(self):
        import logging

        root = logging.getLogger()
        root.handlers.clear()
        setup_logging(verbose=False)
        assert root.level == logging.INFO


class TestCLIArgParsing:
    def test_no_command_exits(self):
        with pytest.raises(SystemExit):
            main([])

    def test_unknown_command_exits(self):
        with pytest.raises(SystemExit):
            main(["unknown_cmd"])

    def test_compare_requires_phase(self):
        with pytest.raises(SystemExit):
            main(["compare"])

    def test_download_valid_parse(self, tmp_path, monkeypatch):
        """CLI parses download args correctly but fails on actual download."""
        # main() will try to run the pipeline which needs data — expect exit 1
        result = main(["--base-dir", str(tmp_path), "download", "--products", "gleam"])
        # Returns 1 because download will fail (no credentials/network)
        assert result == 1
