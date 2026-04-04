"""Tests for product handlers (instantiation, config, unit logic)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from src.transpiration_comparison.config import (
    DEFAULT_DOMAIN,
    DEFAULT_PATHS,
    DEFAULT_PERIOD,
    PRODUCTS,
    UNIT_TO_MM_DAY,
)
from src.transpiration_comparison.products.base import ProductBase
from src.transpiration_comparison.products.era5land import ERA5LandProduct
from src.transpiration_comparison.products.gldas import GLDASProduct
from src.transpiration_comparison.products.gleam import GLEAMProduct
from src.transpiration_comparison.products.pmlv2 import PMLv2Product
from src.transpiration_comparison.products.sap_velocity import SapVelocityProduct


class TestProductInstantiation:
    """All product classes should instantiate without external I/O."""

    def test_sap_velocity_init(self):
        p = SapVelocityProduct()
        assert p.config == PRODUCTS["sap_velocity"]
        assert p.config.variable == "sap_velocity_xgb"

    def test_era5land_init(self):
        p = ERA5LandProduct()
        assert p.config == PRODUCTS["era5land"]
        assert p.config.units == "m/day"

    def test_gleam_init(self):
        p = GLEAMProduct()
        assert p.config == PRODUCTS["gleam"]
        assert p.config.source_type == "sftp"

    def test_pmlv2_init(self):
        p = PMLv2Product()
        assert p.config == PRODUCTS["pmlv2"]
        assert p.config.native_resolution == 0.05

    def test_gldas_init(self):
        p = GLDASProduct()
        assert p.config == PRODUCTS["gldas"]
        assert p.config.native_temporal == "3-hourly"


class TestProductRepr:
    def test_repr_format(self):
        p = ERA5LandProduct()
        r = repr(p)
        assert "ERA5LandProduct" in r
        assert "ERA5-Land" in r


class TestProductOutputDir:
    def test_output_dir_uses_short_name(self, tmp_path):
        from src.transpiration_comparison.config import Paths

        paths = Paths(
            base=tmp_path,
            products_dir=tmp_path / "products",
            preprocessed_dir=tmp_path / "preproc",
        )
        p = ERA5LandProduct(paths=paths)
        d = p.output_dir
        assert d.name == "era5"
        assert d.exists()


class TestProductPreprocessedPath:
    def test_preprocessed_path_format(self, tmp_path):
        from src.transpiration_comparison.config import Paths

        paths = Paths(
            base=tmp_path,
            preprocessed_dir=tmp_path / "preproc",
        )
        p = GLEAMProduct(paths=paths)
        assert p.preprocessed_path.name == "gleam_preprocessed.nc"


class TestUnitConversions:
    def test_era5land_m_to_mm(self):
        assert UNIT_TO_MM_DAY["m/day"] == 1000.0

    def test_gldas_kgm2s_to_mmday(self):
        assert UNIT_TO_MM_DAY["kg/m2/s"] == 86400.0

    def test_gleam_already_mm(self):
        assert UNIT_TO_MM_DAY["mm/day"] == 1.0

    def test_sap_velocity_not_convertible(self):
        assert UNIT_TO_MM_DAY["cm3/cm2/h"] is None


class TestSapVelocityDownloadCheck:
    def test_download_raises_if_dir_missing(self, tmp_path):
        from src.transpiration_comparison.config import Paths

        paths = Paths(
            base=tmp_path,
            predictions_parquet=tmp_path / "nonexistent_parquet",
        )
        p = SapVelocityProduct(paths=paths)
        with pytest.raises(FileNotFoundError, match="Predictions not found"):
            p.download()


class TestSapVelocityLoadCheck:
    def test_load_raises_if_no_parquet(self, tmp_path):
        from src.transpiration_comparison.config import Paths

        parquet_dir = tmp_path / "parquet"
        parquet_dir.mkdir()
        paths = Paths(base=tmp_path, predictions_parquet=parquet_dir)
        p = SapVelocityProduct(paths=paths)
        with pytest.raises(FileNotFoundError, match="No parquet files"):
            p.load()


class TestERA5LandLoadCheck:
    def test_load_raises_if_no_nc_files(self, tmp_path):
        from src.transpiration_comparison.config import Paths

        paths = Paths(base=tmp_path, products_dir=tmp_path / "products")
        p = ERA5LandProduct(paths=paths)
        with pytest.raises(FileNotFoundError, match="No ERA5-Land files"):
            p.load()


class TestGLEAMLoadCheck:
    def test_load_raises_if_no_files(self, tmp_path):
        from src.transpiration_comparison.config import Paths

        paths = Paths(base=tmp_path, products_dir=tmp_path / "products")
        p = GLEAMProduct(paths=paths)
        with pytest.raises(FileNotFoundError, match="No GLEAM files"):
            p.load()


class TestGLDASLoadCheck:
    def test_load_raises_if_no_files(self, tmp_path):
        from src.transpiration_comparison.config import Paths

        paths = Paths(base=tmp_path, products_dir=tmp_path / "products")
        p = GLDASProduct(paths=paths)
        with pytest.raises(FileNotFoundError, match="No GLDAS files"):
            p.load()


class TestGLEAMDownloadCredentials:
    def test_download_raises_without_creds(self, monkeypatch):
        monkeypatch.delenv("GLEAM_USER", raising=False)
        monkeypatch.delenv("GLEAM_PASS", raising=False)
        p = GLEAMProduct()
        with pytest.raises(RuntimeError, match="GLEAM credentials not set"):
            p.download()
