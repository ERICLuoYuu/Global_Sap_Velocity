from __future__ import annotations

"""Full pipeline orchestrator for transpiration comparison."""

import logging

import xarray as xr

from .comparison.agreement import consensus_map, unique_signal_detection
from .comparison.collocation import triple_collocation
from .comparison.regional import all_regional_analyses
from .comparison.spatial import pft_stratified_metrics, pixel_temporal_correlation, spatial_rmse_map, zonal_mean_profile
from .comparison.temporal_comp import seasonal_cycle_comparison, trend_agreement
from .config import PRODUCTS, REGIONS, Paths, SpatialDomain, TimePeriod
from .preprocessing.normalize import compute_anomaly, compute_climatology, zscore_normalize
from .preprocessing.regrid import regrid_to_common
from .preprocessing.temporal import aggregate_to_daily
from .products.sap_velocity import SapVelocityProduct
from .visualization.maps import (
    _fmt_name,
    plot_consensus_map,
    plot_correlation_map,
    plot_error_map,
    plot_global_map,
    plot_multi_product_maps,
)
from .visualization.taylor import taylor_diagram
from .visualization.timeseries import (
    plot_pft_heatmap,
    plot_regional_timeseries,
    plot_seasonal_overlay,
    plot_zonal_mean,
)

logger = logging.getLogger(__name__)


def _get_product_handler(product_key: str, paths: Paths, period: TimePeriod, domain: SpatialDomain):
    """Factory to get the right product handler."""
    if product_key == "sap_velocity":
        return SapVelocityProduct(paths=paths, period=period, domain=domain)
    elif product_key == "gleam":
        from .products.gleam import GLEAMProduct

        return GLEAMProduct(paths=paths, period=period, domain=domain)
    elif product_key == "era5land":
        from .products.era5land import ERA5LandProduct

        return ERA5LandProduct(paths=paths, period=period, domain=domain)
    elif product_key == "pmlv2":
        from .products.pmlv2 import PMLv2Product

        return PMLv2Product(paths=paths, period=period, domain=domain)
    elif product_key == "gldas":
        from .products.gldas import GLDASProduct

        return GLDASProduct(paths=paths, period=period, domain=domain)
    else:
        raise ValueError(f"Unknown product: {product_key}")


class TranspirationComparisonPipeline:
    """Orchestrates the full comparison pipeline."""

    def __init__(
        self,
        paths: Paths,
        period: TimePeriod,
        domain: SpatialDomain,
    ):
        self.paths = paths
        self.period = period
        self.domain = domain

    def download_products(self, product_keys: list[str]) -> None:
        """Phase 2: Download transpiration products."""
        logger.info(f"=== Phase 2: Downloading {len(product_keys)} products ===")
        for key in product_keys:
            if key == "sap_velocity":
                logger.info("Sap velocity: already on disk, skipping download")
                continue
            logger.info(f"Downloading {key}...")
            handler = _get_product_handler(key, self.paths, self.period, self.domain)
            handler.download()
        logger.info("=== Download complete ===")

    def preprocess_products(self, product_keys: list[str]) -> None:
        """Preprocess: load, regrid, aggregate, normalize, save."""
        logger.info(f"=== Preprocessing {len(product_keys)} products ===")
        self.paths.preprocessed_dir.mkdir(parents=True, exist_ok=True)

        for key in product_keys:
            output_nc = self.paths.preprocessed_dir / f"{key}_preprocessed.nc"
            if output_nc.exists():
                logger.info(f"{key}: preprocessed file exists, skipping")
                continue

            logger.info(f"Preprocessing {key}...")
            try:
                handler = _get_product_handler(key, self.paths, self.period, self.domain)
                ds = handler.load()

                # Determine the primary variable name
                var_name = _find_primary_var(ds, key)

                # Validate dimensions
                if "lat" not in ds.dims or "lon" not in ds.dims:
                    raise ValueError(f"Missing lat/lon dims. Got: {list(ds.dims)}")

                # Regrid if needed
                config = PRODUCTS[key]
                if abs(config.native_resolution - self.domain.resolution) > 0.01:
                    ds = regrid_to_common(ds, target_resolution=self.domain.resolution, domain=self.domain)

                # Temporal aggregation if needed
                if config.native_temporal != "daily":
                    ds = aggregate_to_daily(ds, config.native_temporal, var=var_name)

                # Aggregate to monthly means to reduce data volume
                if len(ds.time) > 100:
                    logger.info(f"  Aggregating {len(ds.time)} timesteps to monthly means...")
                    ds = ds.resample(time="1ME").mean()
                    logger.info(f"  Now {len(ds.time)} monthly timesteps")

                # Z-score normalize
                ds = zscore_normalize(ds, var_name)

                # Compute anomaly
                ds = compute_anomaly(ds, var_name)

                # Save preprocessed
                ds.to_netcdf(output_nc)
                logger.info(f"  Saved: {output_nc} ({output_nc.stat().st_size / 1e6:.1f} MB)")
            except Exception as e:
                logger.error(f"{key}: preprocessing failed: {e}. Skipping.")
                if output_nc.exists():
                    output_nc.unlink()
                continue

        logger.info("=== Preprocessing complete ===")

    def _load_preprocessed(self) -> dict[str, xr.Dataset]:
        """Load all preprocessed datasets and align grids."""
        datasets = {}
        for nc_file in sorted(self.paths.preprocessed_dir.glob("*_preprocessed.nc")):
            key = nc_file.stem.replace("_preprocessed", "")
            datasets[key] = xr.open_dataset(nc_file, chunks={"time": 365})
            logger.info(f"  Loaded {key}: {dict(datasets[key].dims)}")

        # Align all datasets to sap_velocity grid using nearest-neighbor
        # (grids offset by 0.05 deg: SAPV at x.0 vs GLEAM at x.05)
        if "sap_velocity" in datasets and len(datasets) > 1:
            ref = datasets["sap_velocity"]
            for key in list(datasets.keys()):
                if key == "sap_velocity":
                    continue
                ds = datasets[key]
                datasets[key] = ds.reindex_like(ref, method="nearest", tolerance=0.06)
                logger.info(f"  Reindexed {key} to sap_velocity grid: {dict(datasets[key].sizes)}")
        return datasets

    def _load_pft_map(self) -> xr.DataArray:
        """Load PFT map from sap velocity predictions."""
        handler = SapVelocityProduct(paths=self.paths, period=self.period, domain=self.domain)
        return handler.load_pft_map()

    def run_comparison(self, phase: str, region: str | None = None) -> None:
        """Run a specific comparison phase."""
        datasets = self._load_preprocessed()
        if not datasets:
            raise RuntimeError("No preprocessed data found. Run 'preprocess' first.")

        pft_map = self._load_pft_map()

        if phase == "spatial":
            self._run_spatial(datasets, pft_map)
        elif phase == "temporal":
            self._run_temporal(datasets, pft_map)
        elif phase == "regional":
            self._run_regional(datasets, pft_map, region)
        elif phase == "agreement":
            self._run_agreement(datasets, pft_map)
        elif phase == "collocation":
            self._run_collocation(datasets)

    def _run_spatial(self, datasets: dict[str, xr.Dataset], pft_map: xr.DataArray) -> None:
        """Phase 3: Spatial pattern comparison."""
        logger.info("=== Phase 3: Spatial comparison ===")
        sapv_ds = datasets.get("sap_velocity")
        if sapv_ds is None:
            raise KeyError("sap_velocity not in preprocessed datasets")

        results_dir = self.paths.reports_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        fig_dir = self.paths.figures_dir / "global_maps"

        corr_maps = {}
        pft_metrics_all = {}

        for key, ds in datasets.items():
            if key == "sap_velocity":
                continue

            # Pixel-wise temporal correlation
            var_comp = _find_zscore_var(ds)
            var_ref = _find_zscore_var(sapv_ds)
            corr = pixel_temporal_correlation(sapv_ds, ds, var_ref, var_comp)
            corr_maps[key] = corr

            # Plot correlation map
            plot_correlation_map(
                corr["pearson_r"],
                title=f"Temporal Correlation: Sap Velocity vs {key}",
                output_path=fig_dir / f"corr_sapv_vs_{key}.png",
            )

            # RMSE map
            rmse = spatial_rmse_map(sapv_ds, ds, var_ref, var_comp)
            plot_error_map(
                rmse,
                title=f"Z-score RMSE \u2014 Sap Velocity vs {_fmt_name(key)}",
                output_path=fig_dir / f"rmse_sapv_vs_{key}.png",
                units="RMSE (Z-score)",
                vmin=0,
                vmax=2,
            )

            # PFT metrics
            pft_df = pft_stratified_metrics(sapv_ds, ds, pft_map, var_ref, var_comp)
            pft_metrics_all[key] = pft_df
            pft_df.to_csv(results_dir / f"pft_metrics_{key}.csv", index=False)

        # Zonal means
        zonal_df = zonal_mean_profile(datasets)
        zonal_df.to_csv(results_dir / "zonal_means.csv", index=False)
        plot_zonal_mean(zonal_df, fig_dir / "zonal_mean_profiles.png")

        # Multi-product signal strength maps (temporal std, not mean which is ~0)
        std_maps = {}
        for key, ds in datasets.items():
            zsvar = _find_zscore_var(ds)
            if zsvar:
                std_maps[key] = ds[zsvar].std(dim="time").compute()
        plot_multi_product_maps(
            std_maps,
            title="Temporal Variability (Z-score Std)",
            output_path=fig_dir / "multi_product_signal_strength.png",
            cmap="YlOrRd",
            vmin=0,
            vmax=1.5,
            units="Z-score std",
        )

        # PFT heatmap
        if pft_metrics_all:
            plot_pft_heatmap(
                pft_metrics_all,
                "pearson_r",
                self.paths.figures_dir / "pft_stratified" / "pft_correlation_heatmap.png",
            )

        logger.info("=== Phase 3 complete ===")

    def _run_temporal(self, datasets: dict[str, xr.Dataset], pft_map: xr.DataArray) -> None:
        """Phase 4: Temporal pattern comparison."""
        logger.info("=== Phase 4: Temporal comparison ===")
        fig_dir = self.paths.figures_dir / "temporal"

        # Seasonal cycle comparison needs climatologies of Z-scores
        # (not raw values, which have incompatible units across products)
        clim_datasets = {}
        for key, ds in datasets.items():
            var = _find_zscore_var(ds)
            clim_datasets[key] = compute_climatology(ds, var)

        result = seasonal_cycle_comparison(clim_datasets, pft_map)

        # Plot seasonal overlays per PFT
        from .config import PFT_COLUMNS

        for pft_code in PFT_COLUMNS:
            plot_seasonal_overlay(
                result["seasonal_profiles"],
                pft_code,
                fig_dir / f"seasonal_{pft_code}.png",
            )

        # Save correlations and peak months
        result["correlations"].to_csv(self.paths.reports_dir / "seasonal_correlations.csv", index=False)
        result["peak_months"].to_csv(self.paths.reports_dir / "peak_months.csv", index=False)

        # Trend agreement
        trend_ds = trend_agreement(datasets)
        if "trend_agreement" in trend_ds:
            plot_global_map(
                trend_ds["trend_agreement"],
                title="Trend Sign Agreement Across Products",
                output_path=fig_dir / "trend_agreement.png",
                cmap="RdYlGn",
                vmin=0,
                vmax=1,
                units="Fraction agreeing",
            )

        logger.info("=== Phase 4 complete ===")

    def _run_regional(self, datasets: dict[str, xr.Dataset], pft_map: xr.DataArray, region: str | None) -> None:
        """Phase 5: Regional deep dives."""
        logger.info("=== Phase 5: Regional analysis ===")

        if region:
            regions = {region: REGIONS[region]}
        else:
            regions = REGIONS

        fig_dir = self.paths.figures_dir / "regional"

        for rkey, rval in regions.items():
            result = all_regional_analyses(datasets, pft_map).get(rkey, {})
            if not result:
                continue

            if not result["time_series"].empty:
                plot_regional_timeseries(
                    result["time_series"],
                    rval.name,
                    fig_dir / f"ts_{rkey}.png",
                )

        logger.info("=== Phase 5 complete ===")

    def _run_agreement(self, datasets: dict[str, xr.Dataset], pft_map: xr.DataArray) -> None:
        """Phase 6: Product agreement analysis."""
        logger.info("=== Phase 6: Product agreement ===")
        fig_dir = self.paths.figures_dir / "summary"
        sapv_ds = datasets.get("sap_velocity")
        if sapv_ds is None:
            raise KeyError("sap_velocity not in preprocessed datasets")

        # Recompute correlation maps
        corr_maps = {}
        for key, ds in datasets.items():
            if key == "sap_velocity":
                continue
            var_comp = _find_zscore_var(ds)
            var_ref = _find_zscore_var(sapv_ds)
            corr_maps[key] = pixel_temporal_correlation(sapv_ds, ds, var_ref, var_comp)

        # Consensus map — mask non-forest pixels (ocean/bare shows as false disagreement)
        consensus = consensus_map(corr_maps)
        forest_mask = pft_map.notnull()
        for var in consensus.data_vars:
            consensus[var] = consensus[var].where(forest_mask)
        plot_consensus_map(consensus, fig_dir / "consensus_zones.png")

        # Unique signal detection
        unique_df = unique_signal_detection(datasets, pft_map)
        if not unique_df.empty:
            unique_df.to_csv(self.paths.reports_dir / "unique_signals.csv", index=False)

        # Taylor diagram (global, time-mean patterns)
        ref_mean = sapv_ds[_find_zscore_var(sapv_ds)].mean(dim="time")
        prod_means = {}
        for key, ds in datasets.items():
            if key == "sap_velocity":
                continue
            zsvar = _find_zscore_var(ds)
            if zsvar:
                prod_means[key] = ds[zsvar].mean(dim="time")

        if prod_means:
            taylor_diagram(ref_mean, prod_means, "Taylor Diagram: Global Patterns", fig_dir / "taylor_global.png")

        logger.info("=== Phase 6 complete ===")

    def _run_collocation(self, datasets: dict[str, xr.Dataset]) -> None:
        """Triple collocation analysis (requires exactly 3 transpiration products)."""
        logger.info("=== Collocation analysis ===")
        transp = {k: v for k, v in datasets.items() if k != "sap_velocity"}
        keys = list(transp.keys())

        if len(keys) < 3:
            logger.warning(f"Need ≥3 transpiration products, have {len(keys)}. Skipping.")
            return

        # Use first 3 products
        a_key, b_key, c_key = keys[0], keys[1], keys[2]
        var_a = _find_zscore_var(transp[a_key])
        var_b = _find_zscore_var(transp[b_key])
        var_c = _find_zscore_var(transp[c_key])

        result = triple_collocation(transp[a_key], transp[b_key], transp[c_key], var_a, var_b, var_c)

        # Save results
        tc_ds = xr.Dataset(
            {
                f"err_var_{a_key}": result["err_var_a"],
                f"err_var_{b_key}": result["err_var_b"],
                f"err_var_{c_key}": result["err_var_c"],
            }
        )
        tc_ds.to_netcdf(self.paths.preprocessed_dir / "triple_collocation.nc")
        logger.info("=== Collocation complete ===")

    def generate_figures(self) -> None:
        """Phase 7: Generate all visualization figures."""
        logger.info("=== Phase 7: Generating figures ===")
        datasets = self._load_preprocessed()
        pft_map = self._load_pft_map()

        self._run_spatial(datasets, pft_map)
        self._run_temporal(datasets, pft_map)
        self._run_regional(datasets, pft_map, region=None)
        self._run_agreement(datasets, pft_map)
        logger.info("=== All figures generated ===")

    def run_all(self) -> None:
        """Run the full pipeline end-to-end."""
        logger.info("========== FULL PIPELINE START ==========")
        product_keys = [k for k in PRODUCTS if PRODUCTS[k].enabled]

        self.download_products([k for k in product_keys if k != "sap_velocity"])
        self.preprocess_products(product_keys)
        self.generate_figures()
        self._run_collocation(self._load_preprocessed())
        logger.info("========== FULL PIPELINE COMPLETE ==========")


def _find_primary_var(ds: xr.Dataset, product_key: str) -> str:
    """Find the primary data variable in a dataset."""
    if "sap_velocity" in ds.data_vars:
        return "sap_velocity"
    if "transpiration" in ds.data_vars:
        return "transpiration"
    config = PRODUCTS.get(product_key)
    if config and config.variable in ds.data_vars:
        return config.variable
    # Fallback: first non-coordinate variable
    data_vars = [v for v in ds.data_vars if not v.endswith("_zscore") and not v.endswith("_anomaly")]
    if data_vars:
        return data_vars[0]
    return list(ds.data_vars)[0]


def _find_zscore_var(ds: xr.Dataset) -> str:
    """Find the Z-score variable in a dataset."""
    zscore_vars = [v for v in ds.data_vars if v.endswith("_zscore")]
    if zscore_vars:
        return zscore_vars[0]
    return list(ds.data_vars)[0]
