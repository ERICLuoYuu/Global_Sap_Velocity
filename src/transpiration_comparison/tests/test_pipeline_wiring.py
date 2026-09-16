"""The pipeline must actually use the guard rails, not merely ship them.

alignment.assert_common_time_axis, alignment.align_to_reference, the provenance
sidecar and the serving loader all exist; these tests pin them into
TranspirationComparisonPipeline so a run cannot bypass them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from src.transpiration_comparison.products.serving_sap_velocity import (
    ServingSapVelocityProduct,
)
from src.transpiration_comparison.provenance import RunProvenance, read_provenance, write_provenance

from src.transpiration_comparison.config import Paths, SpatialDomain, TimePeriod
from src.transpiration_comparison.pipeline import (
    TranspirationComparisonPipeline,
    _downloadable,
    _get_product_handler,
)
from src.transpiration_comparison.products.sap_velocity import SapVelocityProduct

PERIOD = TimePeriod(start="2016-01-01", end="2018-12-31")


def _write_preprocessed(directory, key, var, start, periods, lat, lon):
    """A preprocessed file shaped the way preprocess_products writes them."""
    time = pd.date_range(start, periods=periods, freq="ME")
    values = np.ones((periods, len(lat), len(lon)), dtype=np.float32)
    ds = xr.Dataset(
        {var: (["time", "lat", "lon"], values), f"{var}_zscore": (["time", "lat", "lon"], values)},
        coords={"time": time, "lat": lat, "lon": lon},
    )
    directory.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(directory / f"{key}_preprocessed.nc")


def _pipeline(tmp_path, **kwargs):
    paths = Paths(base=tmp_path, **{k: kwargs.pop(k) for k in ("run_tag",) if k in kwargs})
    paths.ensure_dirs()
    return TranspirationComparisonPipeline(paths=paths, period=PERIOD, domain=SpatialDomain(), **kwargs)


class TestLoadPreprocessedGuards:
    def test_a_period_mismatch_between_caches_raises(self, tmp_path):
        """The old reindex_like turned this into an all-NaN gleam, silently."""
        pipeline = _pipeline(tmp_path)
        lat, lon = np.arange(45.0, 46.0, 0.1), np.arange(5.0, 6.0, 0.1)
        pre = pipeline.paths.preprocessed_dir
        _write_preprocessed(pre, "sap_velocity", "sap_velocity", "2016-01-31", 36, lat, lon)
        _write_preprocessed(pre, "gleam", "transpiration", "2015-01-31", 12, lat, lon)

        with pytest.raises(ValueError, match="gleam"):
            pipeline._load_preprocessed()

    def test_matching_caches_are_aligned_onto_the_sap_grid(self, tmp_path):
        pipeline = _pipeline(tmp_path)
        lat, lon = np.arange(45.0, 46.0, 0.1), np.arange(5.0, 6.0, 0.1)
        pre = pipeline.paths.preprocessed_dir
        _write_preprocessed(pre, "sap_velocity", "sap_velocity", "2016-01-31", 36, lat, lon)
        _write_preprocessed(pre, "gleam", "transpiration", "2016-01-31", 36, lat + 0.05, lon + 0.05)

        datasets = pipeline._load_preprocessed()

        assert np.allclose(datasets["gleam"].lat.values, datasets["sap_velocity"].lat.values)  # nosec B101

    def test_the_alignment_method_is_configurable(self, tmp_path):
        pipeline = _pipeline(tmp_path, alignment="linear")
        assert pipeline.alignment == "linear"  # nosec B101


class TestProvenance:
    def test_preprocessing_records_the_run(self, tmp_path):
        pipeline = _pipeline(tmp_path)
        pipeline.preprocess_products([], allow_missing=[])

        stored = read_provenance(pipeline.paths.preprocessed_dir)
        assert stored is not None  # nosec B101
        assert stored.period_start == "2016-01-01"  # nosec B101
        assert stored.sap_source == "parquet"  # nosec B101

    def test_a_cache_from_another_run_stops_the_pipeline(self, tmp_path):
        pipeline = _pipeline(tmp_path)
        write_provenance(
            pipeline.paths.preprocessed_dir,
            RunProvenance(
                sap_source="serving",
                run_id="a_different_model",
                period_start="2016-01-01",
                period_end="2018-12-31",
            ),
        )
        with pytest.raises(ValueError, match="a_different_model"):
            pipeline.preprocess_products([], allow_missing=[])


class TestSapSource:
    def test_the_default_source_is_the_legacy_parquet(self, tmp_path):
        paths = Paths(base=tmp_path)
        handler = _get_product_handler("sap_velocity", paths, PERIOD, SpatialDomain())
        assert isinstance(handler, SapVelocityProduct)  # nosec B101

    def test_serving_selects_the_netcdf_loader(self, tmp_path):
        paths = Paths(base=tmp_path)
        handler = _get_product_handler(
            "sap_velocity", paths, PERIOD, SpatialDomain(), sap_source="serving", run_id="smzp"
        )
        assert isinstance(handler, ServingSapVelocityProduct)  # nosec B101
        assert handler.run_id == "smzp"  # nosec B101

    def test_serving_without_a_run_id_is_refused(self, tmp_path):
        paths = Paths(base=tmp_path)
        with pytest.raises(ValueError, match="run_id"):
            _get_product_handler("sap_velocity", paths, PERIOD, SpatialDomain(), sap_source="serving")

    def test_the_pipeline_passes_its_source_to_the_factory(self, tmp_path):
        pipeline = _pipeline(tmp_path, sap_source="serving", run_id="smzp")
        handler = pipeline._sap_handler()
        assert isinstance(handler, ServingSapVelocityProduct)  # nosec B101


class TestRunTag:
    def test_a_run_tag_gives_the_run_its_own_directories(self, tmp_path):
        tagged = Paths(base=tmp_path, run_tag="servable_smzp")
        untagged = Paths(base=tmp_path)
        assert tagged.preprocessed_dir != untagged.preprocessed_dir  # nosec B101
        assert tagged.preprocessed_dir.name == "servable_smzp"  # nosec B101
        assert tagged.figures_dir.name == "servable_smzp"  # nosec B101
        assert tagged.reports_dir.name == "servable_smzp"  # nosec B101

    def test_paths_follow_the_base_directory(self, tmp_path):
        paths = Paths(base=tmp_path)
        for field in ("products_dir", "preprocessed_dir", "figures_dir", "reports_dir", "serving_store"):
            assert str(getattr(paths, field)).startswith(str(tmp_path))  # nosec B101


class TestDownloadSelection:
    """Which products a download phase attempts -- decided without touching one.

    Asserting this through download_products() itself would have opened a real
    SFTP session and pulled ~1 GB per GLEAM year whenever GLEAM_USER happened to
    be set in the environment. The choice is a pure function; test that.
    """

    def test_an_allowed_missing_product_is_not_attempted(self):
        assert _downloadable(["gleam", "gldas"], {"gldas"}) == ["gleam"]

    def test_sap_velocity_is_never_downloaded(self):
        assert _downloadable(["sap_velocity", "gleam"], set()) == ["gleam"]

    def test_nothing_allowed_means_everything_is_attempted(self):
        assert _downloadable(["gleam", "era5land"], set()) == ["gleam", "era5land"]
