"""
Explore Relationship Observations
===================================
Scatter plots with LOWESS fitted lines exploring:

1. Tree size (DBH and height) vs sapwood area
   - Overall, by biome, and by PFT

2. Sap flow density vs environmental variables
   (vpd, ws, ta, ext_rad, volumetric_soil_water_layer_1)
   - Overall, by biome, and by PFT

Data is loaded from merged daytime-only daily files
(outputs/processed_data/sapwood/merged_daytime_only/daily/).

Usage:
    python explore_relationship_observations.py
    python explore_relationship_observations.py --scale sapwood
    python explore_relationship_observations.py --output-dir ./my_output
"""

from __future__ import annotations  # PEP 604 `X | None` on Python 3.9 (HPC venv)

import argparse
import logging
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
except ImportError:
    lowess = None
    warnings.warn(
        "statsmodels not installed — LOWESS fitting will be unavailable.",
        stacklevel=2,
    )

# Add project root to path (insert at front so project path_config.py
# shadows any stale copy that may live elsewhere on sys.path, e.g. inside .venv/).
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir in sys.path:
    sys.path.remove(parent_dir)
sys.path.insert(0, parent_dir)

from path_config import PathConfig  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Visual constants ─────────────────────────────────────────────────────────

BIOME_COLORS = {
    "Boreal forest": "#1b7837",
    "Subtropical desert": "#d73027",
    "Temperate forest": "#4575b4",
    "Temperate grassland desert": "#fdae61",
    "Temperate rain forest": "#74add1",
    "Tropical forest savanna": "#f46d43",
    "Tropical rain forest": "#006837",
    "Tundra": "#abd9e9",
    "Woodland/Shrubland": "#a6761d",
}

PFT_COLORS = {
    "DBF": "#1f78b4",
    "EBF": "#33a02c",
    "ENF": "#e31a1c",
    "MF": "#ff7f00",
    "DNF": "#6a3d9a",
    "WSA": "#b15928",
    "SAV": "#a6cee3",
    "WET": "#b2df8a",
    "GRA": "#fb9a99",
    "CSH": "#fdbf6f",
    "OSH": "#cab2d6",
    "CRO": "#ffff99",
}

PFT_FULL_NAMES = {
    "DBF": "Deciduous Broadleaf Forest",
    "EBF": "Evergreen Broadleaf Forest",
    "ENF": "Evergreen Needleleaf Forest",
    "MF": "Mixed Forest",
    "DNF": "Deciduous Needleleaf Forest",
    "WSA": "Woody Savanna",
    "SAV": "Savanna",
    "WET": "Permanent Wetland",
    "GRA": "Grassland",
    "CSH": "Closed Shrubland",
    "OSH": "Open Shrubland",
    "CRO": "Cropland",
}
ENV_VARIABLES = {
    "vpd": "VPD (kPa)",
    "ws": "Wind Speed (m/s)",
    "ta": "Air Temperature (°C)",
    "ext_rad": "Extraterrestrial Radiation (W/m²)",
    # Two SWC representations plotted here — both come from the merge pipeline's
    # three-variant output (see `merge_gap_filled_hourly_orginal.py`):
    #   *_raw    → original ERA5-Land m³/m³ (physically meaningful, cross-site comparable).
    #   *_zscore → proper z-score (x − μ_site) / σ_site (centred, symmetric across sites).
    # The pipeline's default `volumetric_soil_water_layer_1` column is the x/σ
    # variant; we skip it here in favour of z-score for interpretability. If
    # running against old CSVs that lack the `_raw`/`_zscore` columns, the
    # plotting loops silently skip these entries.
    "volumetric_soil_water_layer_1_raw": "Soil Water Content Layer 1 (m³/m³) [ERA5-Land, raw]",
    "volumetric_soil_water_layer_1_zscore": "Soil Water Content Layer 1 (z-score) [(x − μ) / σ, per site]",
}


class RelationshipExplorer:
    """
    Explores relationships between tree/environmental variables and sap flow
    via scatter plots with LOWESS fitted lines.
    """

    def __init__(
        self,
        scale: str = "sapwood",
        output_dir: str | None = None,
        max_scatter_points: int = 50_000,
        use_raw: bool = True,
        per_site: bool = True,
        drydown: bool = True,
    ):
        valid_scales = {"sapwood", "plant", "site"}
        if scale not in valid_scales:
            raise ValueError(f"Invalid scale '{scale}'. Must be one of {valid_scales}.")
        self.paths = PathConfig(scale=scale)
        self.max_scatter_points = max_scatter_points
        self.use_raw = use_raw
        self.per_site = per_site
        self.drydown = drydown

        if output_dir is not None:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.paths.figures_root / "relationship_exploration" / "merged_daytime_only"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # DataFrames populated by load methods
        self.tree_df: pd.DataFrame | None = None
        self.sapflow_env_df: pd.DataFrame | None = None
        # Site metadata cache — populated lazily by _load_site_metadata_cache()
        self._site_md_cache: dict | None = None

    # ─────────────────────────────────────────────────────────────────────────
    # Data loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load_site_metadata_cache(self) -> dict[str, tuple]:
        """
        Read all *_site_md.csv files once and return a dict mapping
        site_code -> (biome, pft). Missing values are np.nan.
        Result is cached on self._site_md_cache for subsequent calls.
        """
        if self._site_md_cache is not None:
            return self._site_md_cache

        cache: dict[str, tuple] = {}
        for f in sorted(self.paths.raw_csv_dir.glob("*_site_md.csv")):
            site_code = f.stem.replace("_site_md", "")
            try:
                md = pd.read_csv(f)
                biome = md["si_biome"].iloc[0] if "si_biome" in md.columns and len(md) > 0 else np.nan
                pft = md["si_igbp"].iloc[0] if "si_igbp" in md.columns and len(md) > 0 else np.nan
            except Exception as e:
                logger.warning(f"Could not read site metadata {f.name}: {e}")
                biome, pft = np.nan, np.nan
            cache[site_code] = (biome, pft)

        logger.info(f"Site metadata cache built: {len(cache)} sites.")
        self._site_md_cache = cache
        return cache

    def load_tree_metadata(self) -> pd.DataFrame:
        """
        Load plant-level metadata (pl_dbh, pl_height, pl_sapw_area) from all
        *_plant_md.csv files, joined with biome/PFT from *_site_md.csv.
        """
        logger.info("Loading tree metadata from raw plant/site metadata files …")
        records: list[dict] = []

        plant_md_files = sorted(self.paths.raw_csv_dir.glob("*_plant_md.csv"))
        logger.info(f"Found {len(plant_md_files)} plant metadata files.")

        site_md_cache = self._load_site_metadata_cache()

        for pf in plant_md_files:
            try:
                parts = pf.stem.replace("_plant_md", "")
                site_code = parts

                plant_md = pd.read_csv(pf)
                if plant_md.empty:
                    continue

                biome, pft = site_md_cache.get(site_code, (np.nan, np.nan))

                for _, row in plant_md.iterrows():
                    records.append(
                        {
                            "site_code": site_code,
                            "pl_dbh": row.get("pl_dbh", np.nan),
                            "pl_height": row.get("pl_height", np.nan),
                            "pl_sapw_area": row.get("pl_sapw_area", np.nan),
                            "pl_species": row.get("pl_species", ""),
                            "biome": biome,
                            "pft": pft,
                        }
                    )
            except Exception as e:
                logger.warning(f"Error processing {pf.name}: {e}")

        df = pd.DataFrame(records)
        # Convert to numeric (handles mixed types / NA strings)
        for col in ["pl_dbh", "pl_height", "pl_sapw_area"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        n_total = len(df)
        n_dbh = df["pl_dbh"].notna().sum()
        n_height = df["pl_height"].notna().sum()
        n_sapw = df["pl_sapw_area"].notna().sum()
        logger.info(f"Tree metadata loaded: {n_total} trees | DBH: {n_dbh}, Height: {n_height}, Sapwood area: {n_sapw}")
        self.tree_df = df
        return df

    def load_sapflow_env_data(self) -> pd.DataFrame:
        """
        Load sap flow + environmental data from the canonical growing-season,
        daytime-only, daily merged files. Each site CSV already contains
        sap_velocity, biome, pft, site_name, and ERA5-Land environmental
        variables (raw + z-scored SWC variants when the merge pipeline was run
        with the three-column SWC output).

        Prefers ``merged/daytime_only/growing_season/daily`` (canonical HPC
        location per feedback memory); falls back to ``merged_daytime_only/daily``
        when the growing-season dir isn't present (e.g., old local copies).
        """
        # Prefer the canonical growing-season + daytime-only daily dir on HPC.
        candidates = [
            self.paths.processed_root / self.paths.scale / "merged" / "daytime_only" / "growing_season" / "daily",
            self.paths.processed_root / self.paths.scale / "merged_daytime_only" / "growing_season" / "daily",
            self.paths.processed_root / self.paths.scale / "merged_daytime_only" / "daily",
            self.paths.merged_daytime_only_dir / "daily",
        ]
        daily_dir: Path | None = None
        for c in candidates:
            try:
                if c.exists() and any(c.glob("*_daily.csv")):
                    daily_dir = c
                    break
            except (OSError, PermissionError):
                continue

        if daily_dir is None:
            logger.error("No merged daily dir found. Tried: " + ", ".join(str(c) for c in candidates))
            self.sapflow_env_df = pd.DataFrame()
            return self.sapflow_env_df

        logger.info(f"Loading sap flow and environmental data from {daily_dir} …")

        site_files = sorted(daily_dir.glob("*_daily.csv"))
        # Exclude the all-biomes concatenated file
        site_files = [f for f in site_files if f.name != "all_biomes_merged_daily.csv"]
        logger.info(f"Found {len(site_files)} site daily files in {daily_dir}")

        # Columns to keep from each file
        keep_cols = ["TIMESTAMP", "sap_velocity", "biome", "pft", "site_name"] + list(ENV_VARIABLES.keys())

        all_dfs: list[pd.DataFrame] = []
        processed = 0

        for sf in site_files:
            try:
                df = pd.read_csv(sf, parse_dates=["TIMESTAMP"])
                if df.empty:
                    continue

                # Keep only columns we need (that exist in this file)
                available = [c for c in keep_cols if c in df.columns]
                df = df[available]

                if "sap_velocity" not in df.columns:
                    continue

                all_dfs.append(df)
                processed += 1

                if processed % 20 == 0:
                    logger.info(f"  Loaded {processed} sites …")

            except Exception as e:
                logger.warning(f"Error reading {sf.name}: {e}")

        if not all_dfs:
            logger.error("No sap flow + env data could be loaded!")
            self.sapflow_env_df = pd.DataFrame()
            return self.sapflow_env_df

        combined = pd.concat(all_dfs, ignore_index=True)

        # Rename sap_velocity → sap_flow_density for consistency with plot code
        combined.rename(columns={"sap_velocity": "sap_flow_density"}, inplace=True)

        # Ensure numeric types
        numeric_cols = ["sap_flow_density"] + list(ENV_VARIABLES.keys())
        for c in numeric_cols:
            if c in combined.columns:
                combined[c] = pd.to_numeric(combined[c], errors="coerce")

        swc_variants = [
            c
            for c in (
                "volumetric_soil_water_layer_1_raw",
                "volumetric_soil_water_layer_1_zscore",
            )
            if c in combined.columns and combined[c].notna().any()
        ]
        logger.info(
            f"Sap flow + env data loaded: {len(combined):,} rows from "
            f"{processed} sites | biomes: {combined['biome'].nunique()}, "
            f"PFTs: {combined['pft'].nunique()} | SWC variants present: {swc_variants or 'none'}"
        )
        self.sapflow_env_df = combined
        return combined

    # ─────────────────────────────────────────────────────────────────────────
    # Plotting helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _subsample(df: pd.DataFrame, max_n: int = 50_000, random_state: int = 42) -> pd.DataFrame:
        """Randomly subsample if df is larger than max_n."""
        if len(df) > max_n:
            return df.sample(n=max_n, random_state=random_state)
        return df

    @staticmethod
    def _add_linear_trend(
        ax: plt.Axes,
        x: np.ndarray,
        y: np.ndarray,
        color: str = "#111111",
        linestyle: str = "--",
        linewidth: float = 2.0,
        alpha: float = 0.9,
        zorder: int = 6,
    ) -> dict | None:
        """
        Fit and draw a linear regression line on *ax* over the range of x.
        Returns fit statistics or None if not enough finite pairs (<3).
        """
        mask = np.isfinite(x) & np.isfinite(y)
        xm, ym = x[mask], y[mask]
        if len(xm) < 3 or np.ptp(xm) == 0:
            return None
        try:
            slope, intercept, r_value, p_value, _ = stats.linregress(xm, ym)
        except ValueError:
            return None
        x_range = np.array([xm.min(), xm.max()])
        ax.plot(
            x_range,
            slope * x_range + intercept,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )
        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_value**2),
            "p_value": float(p_value),
            "n": int(len(xm)),
        }

    @staticmethod
    def _trim_to_percentile(df: pd.DataFrame, col: str, percentile: float = 95.0) -> pd.DataFrame:
        """
        Keep rows where ``df[col]`` ≤ the given percentile of ``df[col]``.

        Used to strip extreme high-end outliers from environmental x-axes so
        scatter plots and LOWESS fits are not distorted by a long right tail.
        NaN values in *col* are dropped (they can't be plotted anyway).
        """
        if col not in df.columns or df.empty:
            return df
        vals = pd.to_numeric(df[col], errors="coerce")
        finite = vals.dropna()
        if finite.empty:
            return df
        cutoff = float(np.percentile(finite, percentile))
        # NaN comparisons return False → NaN rows are dropped, which is desired.
        return df[vals <= cutoff].copy()

    @staticmethod
    def _scatter_with_lowess(
        ax: plt.Axes,
        x: np.ndarray,
        y: np.ndarray,
        color: str = "#4575b4",
        alpha: float = 0.25,
        lowess_color: str = "#d73027",
        lowess_frac: float = 0.3,
        label: str | None = None,
        point_size: float = 6,
        trend_line: bool = False,
        trend_color: str = "#111111",
    ):
        """
        Draw scatter + LOWESS line on *ax*.
        Annotate with n and Spearman rho. If *trend_line* is True, also fit
        and overlay a linear regression line; annotation includes R² and slope.
        """
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        n = len(x)
        if n < 10:
            ax.text(
                0.5,
                0.5,
                f"n = {n}\n(too few)",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="grey",
            )
            return

        ax.scatter(x, y, s=point_size, alpha=alpha, color=color, edgecolors="none", rasterized=True, label=label)

        # LOWESS fit
        if lowess is not None:
            try:
                frac = min(1.0, max(lowess_frac, 30 / n))
                fitted = lowess(y, x, frac=frac, is_sorted=False, return_sorted=True)
                ax.plot(fitted[:, 0], fitted[:, 1], color=lowess_color, linewidth=2, zorder=5)
            except Exception as e:
                logger.debug(f"LOWESS failed: {e}")

        # Linear trend line
        trend_stats = None
        if trend_line:
            trend_stats = RelationshipExplorer._add_linear_trend(ax, x, y, color=trend_color)

        # Spearman correlation
        try:
            rho, p = stats.spearmanr(x, y)
            rho_str = f"ρ = {rho:.2f}"
            if p < 0.001:
                rho_str += "***"
            elif p < 0.01:
                rho_str += "**"
            elif p < 0.05:
                rho_str += "*"
        except Exception:
            rho_str = ""

        annotation = f"n = {n:,}\n{rho_str}"
        if trend_stats is not None:
            p_str = (
                f"{trend_stats['p_value']:.1e}" if trend_stats["p_value"] < 1e-3 else f"{trend_stats['p_value']:.3f}"
            )
            annotation += f"\nslope = {trend_stats['slope']:.3g}\nR² = {trend_stats['r_squared']:.2f}  p={p_str}"

        ax.annotate(
            annotation,
            xy=(0.03, 0.95),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.85),
        )

    def _make_faceted_plot(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        group_col: str,
        x_label: str,
        y_label: str,
        title: str,
        save_name: str,
        color_map: dict[str, str] | None = None,
        max_cols: int = 4,
        trend_line: bool = False,
    ):
        """Create a grid of scatter + LOWESS (optional trend-line) subplots, one per group."""
        groups = sorted(df[group_col].dropna().unique())
        n_groups = len(groups)
        if n_groups == 0:
            logger.warning(f"No groups found for {group_col} — skipping {save_name}")
            return

        n_cols = min(max_cols, n_groups)
        n_rows = int(np.ceil(n_groups / n_cols))

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(5 * n_cols, 4.5 * n_rows),
            squeeze=False,
        )
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

        for idx, group in enumerate(groups):
            row_i, col_i = divmod(idx, n_cols)
            ax = axes[row_i, col_i]

            sub = df[df[group_col] == group].dropna(subset=[x_col, y_col])
            sub = self._subsample(sub, self.max_scatter_points)

            color = (color_map or {}).get(group, "#4575b4")
            self._scatter_with_lowess(
                ax,
                sub[x_col].values,
                sub[y_col].values,
                color=color,
                label=group,
                trend_line=trend_line,
            )
            ax.set_xlabel(x_label, fontsize=9)
            ax.set_ylabel(y_label, fontsize=9)
            ax.set_title(str(group), fontsize=10, fontweight="bold")
            ax.tick_params(labelsize=8)

        # Hide unused axes
        for idx in range(n_groups, n_rows * n_cols):
            row_i, col_i = divmod(idx, n_cols)
            axes[row_i, col_i].set_visible(False)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved → {save_path}")

    def _make_overall_plot(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        x_label: str,
        y_label: str,
        title: str,
        save_name: str,
        hue_col: str | None = None,
        color_map: dict[str, str] | None = None,
        suppress_legend: bool = False,
        trend_line: bool = False,
    ):
        """Single overall scatter + LOWESS plot, optionally colored by hue."""
        valid_full = df.dropna(subset=[x_col, y_col]).copy()
        valid_scatter = self._subsample(valid_full, self.max_scatter_points)

        fig, ax = plt.subplots(figsize=(8, 6))

        if hue_col and hue_col in valid_scatter.columns:
            groups = sorted(valid_scatter[hue_col].dropna().unique())
            for g in groups:
                g_df = valid_scatter[valid_scatter[hue_col] == g]
                color = (color_map or {}).get(g, None)
                ax.scatter(
                    g_df[x_col],
                    g_df[y_col],
                    s=10,
                    alpha=0.3,
                    color=color,
                    edgecolors="none",
                    rasterized=True,
                    label=g,
                )
            if not suppress_legend:
                ax.legend(fontsize=7, loc="upper left", framealpha=0.9, ncol=2, markerscale=2)
        else:
            ax.scatter(
                valid_scatter[x_col],
                valid_scatter[y_col],
                s=10,
                alpha=0.25,
                color="#4575b4",
                edgecolors="none",
                rasterized=True,
            )

        # Spearman on full data
        x_arr_full = valid_full[x_col].values.astype(float)
        y_arr_full = valid_full[y_col].values.astype(float)
        full_mask = np.isfinite(x_arr_full) & np.isfinite(y_arr_full)
        x_arr_full = x_arr_full[full_mask]
        y_arr_full = y_arr_full[full_mask]
        n = len(x_arr_full)

        # LOWESS on a capped subsample for performance
        _MAX_LOWESS = 10_000
        valid_lowess = self._subsample(valid_full, _MAX_LOWESS)
        x_lowess = valid_lowess[x_col].values.astype(float)
        y_lowess = valid_lowess[y_col].values.astype(float)
        lowess_mask = np.isfinite(x_lowess) & np.isfinite(y_lowess)
        x_lowess = x_lowess[lowess_mask]
        y_lowess = y_lowess[lowess_mask]

        if lowess is not None and len(x_lowess) >= 30:
            try:
                frac = min(1.0, max(0.3, 30 / len(x_lowess)))
                fitted = lowess(y_lowess, x_lowess, frac=frac, is_sorted=False, return_sorted=True)
                ax.plot(fitted[:, 0], fitted[:, 1], color="#d73027", linewidth=2.5, zorder=10, label="LOWESS")
            except Exception as e:
                logger.debug(f"LOWESS failed: {e}")

        # Linear trend line on full-data finite pairs
        trend_stats = None
        if trend_line and n >= 3:
            trend_stats = self._add_linear_trend(ax, x_arr_full, y_arr_full, color="#111111", linewidth=2.5, zorder=11)

        # Spearman
        try:
            rho, p = stats.spearmanr(x_arr_full, y_arr_full)
            rho_str = f"ρ = {rho:.2f}"
            if p < 0.001:
                rho_str += "***"
            elif p < 0.01:
                rho_str += "**"
            elif p < 0.05:
                rho_str += "*"
        except Exception:
            rho_str = ""

        annotation = f"n = {n:,}\n{rho_str}"
        if trend_stats is not None:
            p_str = (
                f"{trend_stats['p_value']:.1e}" if trend_stats["p_value"] < 1e-3 else f"{trend_stats['p_value']:.3f}"
            )
            annotation += f"\nslope = {trend_stats['slope']:.3g}\nR² = {trend_stats['r_squared']:.2f}  p={p_str}"

        ax.annotate(
            annotation,
            xy=(0.03, 0.95),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.85),
        )

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        plt.tight_layout()

        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved → {save_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Plot Set 1: Tree size vs Sapwood area
    # ─────────────────────────────────────────────────────────────────────────

    def plot_tree_size_vs_sapwood_area(self):
        """
        Generate scatter + LOWESS plots for:
          - DBH vs sapwood area (overall, by biome, by PFT)
          - Height vs sapwood area (overall, by biome, by PFT)
        """
        if self.tree_df is None:
            self.load_tree_metadata()

        df = self.tree_df.copy()
        logger.info("Generating tree-size vs sapwood-area plots …")

        for x_col, x_label, tag in [
            ("pl_dbh", "DBH (cm)", "dbh"),
            ("pl_height", "Height (m)", "height"),
        ]:
            valid = df.dropna(subset=[x_col, "pl_sapw_area"])
            if valid.empty:
                logger.warning(f"No valid data for {x_col} vs pl_sapw_area")
                continue

            y_label = "Sapwood Area (cm²) [per plant]"
            n_plants = len(valid)
            n_sites = valid["site_code"].nunique()

            # ── Overall ──
            # Color by biome (9 cats → legible legend) rather than site_code
            # (~165 sites → legend unreadable). Biome also matches the colour
            # convention used by `_plot_sapflow_env_overall_grid`.
            self._make_overall_plot(
                valid,
                x_col,
                "pl_sapw_area",
                x_label=x_label,
                y_label=y_label,
                title=f"{x_label} vs Sapwood Area — {n_plants:,} plants, {n_sites} sites",
                save_name=f"{tag}_vs_sapwood_area_overall.png",
                hue_col="biome",
                color_map=BIOME_COLORS,
            )

            # ── By biome ──
            biome_valid = valid.dropna(subset=["biome"])
            if not biome_valid.empty:
                self._make_faceted_plot(
                    biome_valid,
                    x_col,
                    "pl_sapw_area",
                    group_col="biome",
                    x_label=x_label,
                    y_label=y_label,
                    title=f"{x_label} vs {y_label} — by Biome",
                    save_name=f"{tag}_vs_sapwood_area_by_biome.png",
                    color_map=BIOME_COLORS,
                )

            # ── By PFT ──
            pft_valid = valid.dropna(subset=["pft"])
            if not pft_valid.empty:
                self._make_faceted_plot(
                    pft_valid,
                    x_col,
                    "pl_sapw_area",
                    group_col="pft",
                    x_label=x_label,
                    y_label=y_label,
                    title=f"{x_label} vs {y_label} — by PFT",
                    save_name=f"{tag}_vs_sapwood_area_by_pft.png",
                    color_map=PFT_COLORS,
                )

        logger.info("Tree-size plots complete.")

    # ─────────────────────────────────────────────────────────────────────────
    # Plot Set 2: Sap flow density vs Environmental variables
    # ─────────────────────────────────────────────────────────────────────────

    def plot_sapflow_vs_env(self):
        """
        Generate scatter + LOWESS plots for sap flow density vs each of
        the environmental variables (vpd, ws, ta, ext_rad, volumetric_soil_water_layer_1).
        Each variable gets: overall plot, by-biome faceted, by-PFT faceted.

        Env x-axes are trimmed to the lower 95% of each variable's distribution
        (top 5% dropped) so scatter + LOWESS are not distorted by long right tails.
        """
        if self.sapflow_env_df is None:
            self.load_sapflow_env_data()

        df = self.sapflow_env_df.copy()
        if df.empty:
            logger.error("No sap-flow + env data available — skipping plots.")
            return

        logger.info("Generating sap-flow vs environment plots …")

        y_col = "sap_flow_density"
        y_label = "Sap Flow Density (cm³ cm⁻² h⁻¹)"

        # ── Overall summary (2 × 3 grid, all 6 env vars) ──
        self._plot_sapflow_env_overall_grid(df, y_col, y_label)

        # ── Per env variable: by biome and by PFT ──
        for env_var, env_label in ENV_VARIABLES.items():
            if env_var not in df.columns:
                logger.warning(f"Env variable '{env_var}' not found — skipping.")
                continue

            valid = df.dropna(subset=[y_col, env_var])
            if valid.empty:
                continue
            # Trim top 5% of env distribution (outliers distort LOWESS / axes)
            valid = self._trim_to_percentile(valid, env_var, 95.0)

            # By biome
            biome_valid = valid.dropna(subset=["biome"])
            if not biome_valid.empty:
                self._make_faceted_plot(
                    biome_valid,
                    env_var,
                    y_col,
                    group_col="biome",
                    x_label=env_label,
                    y_label=y_label,
                    title=f"Sap Flow Density vs {env_label} — by Biome",
                    save_name=f"sapflow_vs_{env_var}_by_biome.png",
                    color_map=BIOME_COLORS,
                )

            # By PFT
            pft_valid = valid.dropna(subset=["pft"])
            if not pft_valid.empty:
                self._make_faceted_plot(
                    pft_valid,
                    env_var,
                    y_col,
                    group_col="pft",
                    x_label=env_label,
                    y_label=y_label,
                    title=f"Sap Flow Density vs {env_label} — by PFT",
                    save_name=f"sapflow_vs_{env_var}_by_pft.png",
                    color_map=PFT_COLORS,
                )

        logger.info("Sap-flow vs environment plots complete.")

    def _plot_sapflow_env_overall_grid(self, df: pd.DataFrame, y_col: str, y_label: str):
        """
        Create a single 2x3 figure with one subplot per env variable,
        scatter + LOWESS, colored by biome.
        """
        env_vars_present = [v for v in ENV_VARIABLES if v in df.columns]
        n_vars = len(env_vars_present)
        if n_vars == 0:
            return

        n_cols = 3
        n_rows = int(np.ceil(n_vars / n_cols))

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(6 * n_cols, 5 * n_rows),
            squeeze=False,
        )
        fig.suptitle(
            "Sap Flow Density vs Environmental Variables (Overall)",
            fontsize=15,
            fontweight="bold",
            y=1.01,
        )

        for idx, env_var in enumerate(env_vars_present):
            row_i, col_i = divmod(idx, n_cols)
            ax = axes[row_i, col_i]

            valid_full = df.dropna(subset=[y_col, env_var])
            # Trim top 5% of env distribution so LOWESS/scatter aren't distorted
            valid_full = self._trim_to_percentile(valid_full, env_var, 95.0)
            valid_scatter = self._subsample(valid_full, self.max_scatter_points)

            # Scatter colored by biome (subsampled data only)
            biome_groups = sorted(valid_scatter["biome"].dropna().unique())
            for biome in biome_groups:
                b_df = valid_scatter[valid_scatter["biome"] == biome]
                ax.scatter(
                    b_df[env_var],
                    b_df[y_col],
                    s=6,
                    alpha=0.2,
                    color=BIOME_COLORS.get(biome, "#888888"),
                    edgecolors="none",
                    rasterized=True,
                    label=biome,
                )

            # Spearman on full data (n shows all valid pairs)
            x_arr_full = valid_full[env_var].values.astype(float)
            y_arr_full = valid_full[y_col].values.astype(float)
            full_mask = np.isfinite(x_arr_full) & np.isfinite(y_arr_full)
            x_arr_full = x_arr_full[full_mask]
            y_arr_full = y_arr_full[full_mask]
            n = len(x_arr_full)

            # LOWESS on a capped subsample — trend curve doesn't need all n rows
            _MAX_LOWESS = 10_000
            valid_lowess = self._subsample(valid_full, _MAX_LOWESS)
            x_lowess = valid_lowess[env_var].values.astype(float)
            y_lowess = valid_lowess[y_col].values.astype(float)
            lowess_mask = np.isfinite(x_lowess) & np.isfinite(y_lowess)
            x_lowess = x_lowess[lowess_mask]
            y_lowess = y_lowess[lowess_mask]

            if lowess is not None and len(x_lowess) >= 30:
                try:
                    frac = min(1.0, max(0.3, 30 / len(x_lowess)))
                    fitted = lowess(y_lowess, x_lowess, frac=frac, is_sorted=False, return_sorted=True)
                    ax.plot(fitted[:, 0], fitted[:, 1], color="#d73027", linewidth=2, zorder=10)
                except Exception as e:
                    logger.debug(f"LOWESS failed for {env_var}: {e}")

            # Spearman
            try:
                rho, p = stats.spearmanr(x_arr_full, y_arr_full)
                rho_str = f"ρ={rho:.2f}"
                if p < 0.001:
                    rho_str += "***"
                elif p < 0.01:
                    rho_str += "**"
                elif p < 0.05:
                    rho_str += "*"
            except Exception:
                rho_str = ""

            ax.annotate(
                f"n={n:,}  {rho_str}",
                xy=(0.03, 0.95),
                xycoords="axes fraction",
                ha="left",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey", alpha=0.85),
            )

            ax.set_xlabel(ENV_VARIABLES[env_var], fontsize=9)
            ax.set_ylabel(y_label if col_i == 0 else "", fontsize=9)
            ax.set_title(ENV_VARIABLES[env_var], fontsize=10, fontweight="bold")
            ax.tick_params(labelsize=8)

        # Hide unused axes
        for idx in range(n_vars, n_rows * n_cols):
            row_i, col_i = divmod(idx, n_cols)
            axes[row_i, col_i].set_visible(False)

        # Shared legend — collect from all axes to avoid missing biomes
        seen_labels: dict[str, object] = {}
        for ax_row in axes:
            for ax_cell in ax_row:
                handles, labels = ax_cell.get_legend_handles_labels()
                for handle, label in zip(handles, labels):  # noqa: B905 — 3.9 compat
                    if label not in seen_labels:
                        seen_labels[label] = handle
        handles = list(seen_labels.values())
        labels = list(seen_labels.keys())
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=min(5, len(handles)),
                fontsize=8,
                markerscale=2,
                bbox_to_anchor=(0.5, -0.02),
            )

        plt.tight_layout()
        save_path = self.output_dir / "sapflow_vs_env_overall.png"
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved → {save_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Plot Set 3: Per-site sap flow density vs Environmental variables
    # ─────────────────────────────────────────────────────────────────────────

    def plot_sapflow_vs_env_per_site(self):
        """
        Generate one 2x3 sap-flow-vs-env grid per site, saved **flat** to
        ``{output_dir}/per_site/{site_code}_sapflow_vs_env.png`` (no per-site
        subdirectory — one PNG per site sits alongside the rest).

        Faceting by biome/PFT is meaningless per site (both are constant),
        so this produces a single grid per site with all 6 env variables
        (including raw and normalised SWC side-by-side in the grid).
        Scatter color is taken from ``PFT_COLORS`` so the plot tags its PFT
        visually without needing a legend.
        """
        if self.sapflow_env_df is None:
            self.load_sapflow_env_data()

        df = self.sapflow_env_df
        if df is None or df.empty:
            logger.error("No sap-flow + env data available — skipping per-site plots.")
            return

        y_col = "sap_flow_density"
        y_label = "Sap Flow Density (cm³ cm⁻² h⁻¹)"

        per_site_root = self.output_dir / "per_site"
        per_site_root.mkdir(parents=True, exist_ok=True)

        sites = sorted(df["site_name"].dropna().unique())
        logger.info(f"Generating per-site sap-flow vs env plots for {len(sites)} sites …")

        processed = 0
        for site in sites:
            site_df = df[df["site_name"] == site]
            if site_df.empty or site_df[y_col].dropna().empty:
                continue

            biome_series = site_df["biome"].dropna() if "biome" in site_df.columns else pd.Series(dtype=object)
            pft_series = site_df["pft"].dropna() if "pft" in site_df.columns else pd.Series(dtype=object)
            biome = str(biome_series.iloc[0]) if not biome_series.empty else "—"
            pft = str(pft_series.iloc[0]) if not pft_series.empty else "—"

            save_path = per_site_root / f"{site}_sapflow_vs_env.png"

            self._plot_sapflow_env_grid_for_site(
                site_df,
                y_col,
                y_label,
                str(site),
                biome,
                pft,
                save_path,
            )
            processed += 1

            if processed % 20 == 0:
                logger.info(f"  Processed {processed}/{len(sites)} sites …")

        logger.info(f"Per-site sap-flow vs env plots complete: {processed} sites saved under {per_site_root}")

    def _plot_sapflow_env_grid_for_site(
        self,
        df: pd.DataFrame,
        y_col: str,
        y_label: str,
        site_code: str,
        biome: str,
        pft: str,
        save_path: Path,
    ):
        """Render a 2x3 scatter + LOWESS grid for a single site."""
        env_vars_present = [v for v in ENV_VARIABLES if v in df.columns]
        n_vars = len(env_vars_present)
        if n_vars == 0:
            return

        n_cols = 3
        n_rows = int(np.ceil(n_vars / n_cols))

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(6 * n_cols, 5 * n_rows),
            squeeze=False,
        )

        site_color = PFT_COLORS.get(pft, "#4575b4")
        total_n = int(df[y_col].notna().sum())
        fig.suptitle(
            f"{site_code}  —  {biome} / {pft}  (n = {total_n:,} daily rows)",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )

        for idx, env_var in enumerate(env_vars_present):
            row_i, col_i = divmod(idx, n_cols)
            ax = axes[row_i, col_i]

            valid = df.dropna(subset=[y_col, env_var])
            # Per-site top-5% trim on the env x-axis (this site's own distribution)
            valid = self._trim_to_percentile(valid, env_var, 95.0)
            self._scatter_with_lowess(
                ax,
                valid[env_var].values.astype(float),
                valid[y_col].values.astype(float),
                color=site_color,
                alpha=0.5,
                point_size=12,
            )
            ax.set_xlabel(ENV_VARIABLES[env_var], fontsize=10)
            ax.set_ylabel(y_label if col_i == 0 else "", fontsize=10)
            ax.set_title(ENV_VARIABLES[env_var], fontsize=11, fontweight="bold")
            ax.tick_params(labelsize=8)

        for idx in range(n_vars, n_rows * n_cols):
            row_i, col_i = divmod(idx, n_cols)
            axes[row_i, col_i].set_visible(False)

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # Plot Set 4: Sap flow density vs SWC on dry-down days
    # ─────────────────────────────────────────────────────────────────────────
    #
    # Following the paper's algorithm (refs 13, 26, 27, 61, 62):
    #   * Rain source: ERA5-Land `total_precipitation_hourly_sum` (m/day, ×1000 → mm).
    #   * No-rain threshold: 1.0 mm/day (WMO trace).
    #   * A dry-down is a run of ≥10 consecutive no-rain days, preceded by a
    #     rain day, over which SM shows a decreasing trend (variant B):
    #         SM[end] < SM[start], linear slope < 0, ≤20% daily-diff violations.
    #   * Cropland sites (PFT=CRO) are dropped entirely.
    #   * Data source: peak-growing-season daytime-only dailies when available,
    #     falling back to `merged_daytime_only/daily` with a warning.

    def _resolve_drydown_daily_dir(self) -> tuple[Path, bool]:
        """
        Resolve the source daily dir for dry-down analysis.
        Returns (dir, is_growing_season). Falls back to merged_daytime_only/daily
        with a warning if no growing-season dir exists.
        """
        candidates = [
            self.paths.merged_growing_season_dir / "daily",
            self.paths.processed_root / self.paths.scale / "merged" / "daytime_only" / "growing_season" / "daily",
            self.paths.processed_root / self.paths.scale / "merged_daytime_only" / "growing_season" / "daily",
        ]
        for c in candidates:
            try:
                if c.exists() and any(c.glob("*_daily.csv")):
                    logger.info(f"Dry-down source (growing-season): {c}")
                    return c, True
            except (OSError, PermissionError):
                continue

        fallback = self.paths.processed_root / self.paths.scale / "merged_daytime_only" / "daily"
        if not fallback.exists():
            fallback = self.paths.merged_daytime_only_dir / "daily"
        logger.warning(
            "Growing-season daily dir NOT found — falling back to "
            f"{fallback}. Dry-down detection will run on daytime-only "
            "(not peak-GS-restricted) data."
        )
        return fallback, False

    def _load_drydown_input_data(self) -> pd.DataFrame:
        """
        Load per-site daily CSVs with columns needed for dry-down detection:
        TIMESTAMP, site_name, biome, pft, sap_velocity,
        volumetric_soil_water_layer_1, total_precipitation_hourly_sum.

        Converts ERA5 precip (m/day) to mm/day in a new column `precip_era5_mm`.
        """
        daily_dir, _is_gs = self._resolve_drydown_daily_dir()
        site_files = [f for f in sorted(daily_dir.glob("*_daily.csv")) if f.name != "all_biomes_merged_daily.csv"]
        logger.info(f"Dry-down loader: scanning {len(site_files)} site files in {daily_dir}")

        keep_cols = [
            "TIMESTAMP",
            "site_name",
            "biome",
            "pft",
            "sap_velocity",
            "volumetric_soil_water_layer_1",
            "volumetric_soil_water_layer_1_raw",
            "volumetric_soil_water_layer_1_zscore",
            "total_precipitation_hourly_sum",
        ]

        all_dfs: list[pd.DataFrame] = []
        skipped_no_precip = 0
        for sf in site_files:
            try:
                df = pd.read_csv(sf, parse_dates=["TIMESTAMP"])
                if df.empty:
                    continue
                available = [c for c in keep_cols if c in df.columns]
                df = df[available]
                if "sap_velocity" not in df.columns or "volumetric_soil_water_layer_1" not in df.columns:
                    continue
                if "total_precipitation_hourly_sum" not in df.columns:
                    skipped_no_precip += 1
                    continue
                all_dfs.append(df)
            except Exception as e:
                logger.warning(f"Error reading {sf.name}: {e}")

        if skipped_no_precip:
            logger.warning(f"Skipped {skipped_no_precip} files missing ERA5 precip column.")

        if not all_dfs:
            logger.error("No valid dry-down input data found.")
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)
        combined.rename(columns={"sap_velocity": "sap_flow_density"}, inplace=True)
        combined["precip_era5_mm"] = pd.to_numeric(combined["total_precipitation_hourly_sum"], errors="coerce") * 1000.0
        for c in ["sap_flow_density", "volumetric_soil_water_layer_1", "precip_era5_mm"]:
            combined[c] = pd.to_numeric(combined[c], errors="coerce")

        logger.info(f"Dry-down input loaded: {len(combined):,} rows from {combined['site_name'].nunique()} sites.")
        return combined

    @staticmethod
    def _identify_dry_downs(
        site_df: pd.DataFrame,
        precip_col: str = "precip_era5_mm",
        sm_col: str = "volumetric_soil_water_layer_1",
        min_days: int = 10,
        rain_threshold_mm: float = 1.0,  # WMO trace threshold — matches paper methodology
        monotonic_tolerance: float = 0.20,
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Detect dry-down episodes in a single-site daily dataframe.

        Returns:
            mask   : boolean array (len == len(site_df sorted by TIMESTAMP))
                     marking rows that belong to a valid dry-down episode.
            events : per-event diagnostics with start/end dates, length, SM
                     endpoints, slope, violation fraction.

        Rules (paper variant B):
            * No-rain day:  precip ≤ rain_threshold_mm AND precip is finite.
            * Runs must be calendar-consecutive (day gap == 1 day).
            * Run must be preceded by a rain day (first-row runs rejected).
            * Run length ≥ min_days, all SM values finite.
            * SM[end] < SM[start] AND linear slope < 0 AND
              fraction of positive daily diffs ≤ monotonic_tolerance.
        """
        df = site_df.sort_values("TIMESTAMP").reset_index(drop=True)
        n = len(df)
        mask = np.zeros(n, dtype=bool)
        events: list[dict] = []
        if n < min_days + 1:
            return mask, events

        precip = df[precip_col].to_numpy(dtype=float)
        sm = df[sm_col].to_numpy(dtype=float)
        ts = pd.to_datetime(df["TIMESTAMP"])
        day_diffs = ts.diff().dt.days.to_numpy()  # NaN at index 0

        no_rain = (precip <= rain_threshold_mm) & np.isfinite(precip)

        i = 0
        while i < n:
            if not no_rain[i]:
                i += 1
                continue
            run_start = i
            j = i + 1
            while j < n and no_rain[j] and day_diffs[j] == 1:
                j += 1
            run_end = j  # exclusive
            i = run_end

            run_len = run_end - run_start
            if run_len < min_days:
                continue
            # Preceding rain day required → reject run_start == 0
            # (also reject if calendar gap immediately before run_start)
            if run_start == 0 or day_diffs[run_start] != 1:
                continue
            # By run construction, no_rain[run_start - 1] is False (rain), good.

            sm_run = sm[run_start:run_end]
            if not np.isfinite(sm_run).all():
                continue

            diffs = np.diff(sm_run)
            if diffs.size == 0:
                continue
            n_violations = int((diffs > 0).sum())
            frac_violations = n_violations / diffs.size
            slope = float(np.polyfit(np.arange(run_len), sm_run, 1)[0])

            if sm_run[-1] < sm_run[0] and slope < 0 and frac_violations <= monotonic_tolerance:
                mask[run_start:run_end] = True
                events.append(
                    {
                        "start_date": ts.iloc[run_start].strftime("%Y-%m-%d"),
                        "end_date": ts.iloc[run_end - 1].strftime("%Y-%m-%d"),
                        "n_days": int(run_len),
                        "sm_start": float(sm_run[0]),
                        "sm_end": float(sm_run[-1]),
                        "slope_per_day": slope,
                        "frac_violations": frac_violations,
                    }
                )

        return mask, events

    def plot_sapflow_vs_swc_drydown(self):
        """
        Build dry-down-filtered dataset and generate plots for BOTH raw (m³/m³)
        and site-normalised (x/σ) SWC, written to a flat output structure:

            - {output_dir}/swc_drydown/overall_{raw|normalized}.png
            - {output_dir}/swc_drydown/by_biome_{raw|normalized}.png
            - {output_dir}/swc_drydown/by_pft_{raw|normalized}.png
            - {output_dir}/swc_drydown/per_site/{site}_{raw|normalized}.png
            - {output_dir}/swc_drydown/drydown_events.csv

        Dry-down detection runs once (on the normalised column) — events are
        invariant under division by σ, so switching units doesn't change which
        rows are dry-down days.
        """
        df = self._load_drydown_input_data()
        if df.empty:
            logger.error("Dry-down dataset empty — skipping SWC dry-down plots.")
            return

        # Drop cropland sites (irrigation would bias the dry-down detection)
        n_before = len(df)
        df = df[df["pft"] != "CRO"].copy()
        n_dropped = n_before - len(df)
        if n_dropped:
            logger.info(f"Dropped {n_dropped} rows at cropland sites (PFT=CRO).")

        # Per-site dry-down detection
        drydown_rows: list[pd.DataFrame] = []
        all_events: list[dict] = []
        for site_name, g in df.groupby("site_name"):
            mask, events = self._identify_dry_downs(g)
            if mask.sum() == 0:
                continue
            g_sorted = g.sort_values("TIMESTAMP").reset_index(drop=True)
            selected = g_sorted.loc[mask].copy()
            drydown_rows.append(selected)

            biome_val = g_sorted["biome"].dropna()
            pft_val = g_sorted["pft"].dropna()
            biome_tag = str(biome_val.iloc[0]) if not biome_val.empty else None
            pft_tag = str(pft_val.iloc[0]) if not pft_val.empty else None
            for ev in events:
                ev["site_name"] = site_name
                ev["biome"] = biome_tag
                ev["pft"] = pft_tag
            all_events.extend(events)

        if not drydown_rows:
            logger.warning("No dry-down events detected at any site — skipping plots.")
            return

        dd_df = pd.concat(drydown_rows, ignore_index=True)
        n_events = len(all_events)
        n_sites = dd_df["site_name"].nunique()
        logger.info(f"Dry-down events: {n_events} across {n_sites} sites (total {len(dd_df):,} day-rows retained).")

        # Output dir + diagnostic CSV
        drydown_dir = self.output_dir / "swc_drydown"
        drydown_dir.mkdir(parents=True, exist_ok=True)
        events_csv = drydown_dir / "drydown_events.csv"
        pd.DataFrame(all_events).to_csv(events_csv, index=False)
        logger.info(f"Saved → {events_csv}")

        y_col = "sap_flow_density"
        y_label = "Sap Flow Density (cm³ cm⁻² h⁻¹)"

        # Loop over BOTH SWC representations — raw m³/m³ and per-site z-score.
        # Each variant produces its own overall / by-biome / by-PFT / per-site plots.
        variants = [
            ("volumetric_soil_water_layer_1_raw", "raw"),
            ("volumetric_soil_water_layer_1_zscore", "zscore"),
        ]
        per_site_root = drydown_dir / "per_site"
        per_site_root.mkdir(parents=True, exist_ok=True)

        for sm_col, tag in variants:
            if sm_col not in dd_df.columns or dd_df[sm_col].dropna().empty:
                logger.warning(f"SWC variant '{tag}' unavailable ({sm_col} missing or empty) — skipping.")
                continue
            sm_label = ENV_VARIABLES.get(sm_col, sm_col)
            logger.info(f"Rendering SWC dry-down plots: variant={tag}, column={sm_col}")

            # ── Overall ──
            self._make_overall_plot(
                dd_df,
                sm_col,
                y_col,
                x_label=sm_label,
                y_label=y_label,
                title=(
                    f"Sap Flow Density vs SWC ({tag}) on Dry-Down Days — "
                    f"{n_events} events, {n_sites} sites, {len(dd_df):,} days"
                ),
                save_name=f"swc_drydown/overall_{tag}.png",
                hue_col="biome",
                color_map=BIOME_COLORS,
            )

            # ── By biome ──
            biome_df = dd_df.dropna(subset=["biome"])
            if not biome_df.empty:
                self._make_faceted_plot(
                    biome_df,
                    sm_col,
                    y_col,
                    group_col="biome",
                    x_label=sm_label,
                    y_label=y_label,
                    title=f"Sap Flow Density vs SWC ({tag}) on Dry-Down Days — by Biome",
                    save_name=f"swc_drydown/by_biome_{tag}.png",
                    color_map=BIOME_COLORS,
                )

            # ── By PFT ──
            pft_df = dd_df.dropna(subset=["pft"])
            if not pft_df.empty:
                self._make_faceted_plot(
                    pft_df,
                    sm_col,
                    y_col,
                    group_col="pft",
                    x_label=sm_label,
                    y_label=y_label,
                    title=f"Sap Flow Density vs SWC ({tag}) on Dry-Down Days — by PFT",
                    save_name=f"swc_drydown/by_pft_{tag}.png",
                    color_map=PFT_COLORS,
                )

            # ── Per site (flat; one PNG per site per variant) ──
            self._plot_swc_drydown_per_site(dd_df, per_site_root, sm_col, y_col, sm_label, y_label, tag)

        logger.info("SWC dry-down plots complete.")

    def _plot_swc_drydown_per_site(
        self,
        dd_df: pd.DataFrame,
        per_site_root: Path,
        sm_col: str,
        y_col: str,
        sm_label: str,
        y_label: str,
        tag: str,
    ):
        """
        Per-site scatter+LOWESS for dry-down days, saved **flat** as
        ``{per_site_root}/{site}_{tag}.png``. No per-site subdirectory.
        """
        processed = 0
        for site, g in dd_df.groupby("site_name"):
            g_valid = g.dropna(subset=[sm_col, y_col])
            if g_valid.empty:
                continue

            biome_s = g_valid["biome"].dropna()
            pft_s = g_valid["pft"].dropna()
            biome = str(biome_s.iloc[0]) if not biome_s.empty else "—"
            pft = str(pft_s.iloc[0]) if not pft_s.empty else "—"
            site_color = PFT_COLORS.get(pft, "#4575b4")

            fig, ax = plt.subplots(figsize=(7, 6))
            self._scatter_with_lowess(
                ax,
                g_valid[sm_col].to_numpy(dtype=float),
                g_valid[y_col].to_numpy(dtype=float),
                color=site_color,
                alpha=0.6,
                point_size=16,
            )
            ax.set_xlabel(sm_label, fontsize=11)
            ax.set_ylabel(y_label, fontsize=11)
            ax.set_title(
                f"{site}  —  {biome} / {pft}  [{tag}]\ndry-down days: {len(g_valid):,}",
                fontsize=12,
                fontweight="bold",
            )
            plt.tight_layout()

            fig.savefig(per_site_root / f"{site}_{tag}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            processed += 1

        logger.info(f"Per-site SWC dry-down plots ({tag}): {processed} sites saved under {per_site_root}")

    # ─────────────────────────────────────────────────────────────────────────
    # Main runner
    # ─────────────────────────────────────────────────────────────────────────

    def run(self):
        """Execute the full analysis: load data, generate all plots."""
        logger.info("=" * 70)
        logger.info("RELATIONSHIP EXPLORATION — START")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 70)

        # ── Part 1: Tree size vs Sapwood area ──
        logger.info("\n▶ Part 1: Tree size vs Sapwood area")
        self.load_tree_metadata()
        self.plot_tree_size_vs_sapwood_area()

        # ── Part 2: Sap flow density vs Environmental variables ──
        logger.info("\n▶ Part 2: Sap flow density vs Environmental variables")
        self.load_sapflow_env_data()
        self.plot_sapflow_vs_env()

        # ── Part 3: Per-site sap flow density vs Environmental variables ──
        if self.per_site:
            logger.info("\n▶ Part 3: Per-site sap flow density vs Environmental variables")
            self.plot_sapflow_vs_env_per_site()

        # ── Part 4: Sap flow density vs SWC on dry-down days (ERA5 precip) ──
        if self.drydown:
            logger.info("\n▶ Part 4: Sap flow density vs SWC on dry-down days (ERA5 precip)")
            self.plot_sapflow_vs_swc_drydown()

        logger.info("=" * 70)
        logger.info("RELATIONSHIP EXPLORATION — COMPLETE")
        logger.info(f"All figures saved to: {self.output_dir}")
        logger.info("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Explore relationships: tree size vs sapwood area, sap flow density vs environmental variables."
    )
    parser.add_argument(
        "--scale", type=str, default="sapwood", help="Data scale: sapwood, plant, or site (default: sapwood)"
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory for figures")
    parser.add_argument("--max-points", type=int, default=50_000, help="Max scatter points per panel (default: 50000)")
    parser.add_argument(
        "--use-raw",
        action="store_true",
        default=False,
        help="Use raw data directories without outlier removal (default: False, i.e. use outlier-removed data when available)",
    )
    parser.add_argument(
        "--no-per-site",
        dest="per_site",
        action="store_false",
        default=True,
        help="Skip the per-site sap-flow-vs-env grid plots (default: generate them).",
    )
    parser.add_argument(
        "--no-drydown",
        dest="drydown",
        action="store_false",
        default=True,
        help="Skip the SWC dry-down plot set (default: generate them).",
    )
    args = parser.parse_args()

    explorer = RelationshipExplorer(
        scale=args.scale,
        output_dir=args.output_dir,
        max_scatter_points=args.max_points,
        use_raw=args.use_raw,
        per_site=args.per_site,
        drydown=args.drydown,
    )
    explorer.run()


if __name__ == "__main__":
    main()
