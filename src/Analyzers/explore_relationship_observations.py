"""
Explore Relationship Observations
===================================
Scatter plots with LOWESS fitted lines exploring:

1. Tree size (DBH and height) vs sapwood area
   - Overall, by biome, and by PFT

2. Sap flow density vs environmental variables
   (vpd, ws, sw_in, ta, ppfd_in, ext_rad)
   - Overall, by biome, and by PFT

Data is loaded from raw SAPFLUXNET files individually.

Usage:
    python explore_relationship_observations.py
    python explore_relationship_observations.py --scale sapwood
    python explore_relationship_observations.py --output-dir ./my_output
"""

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
    warnings.warn("statsmodels not installed — LOWESS fitting will be unavailable.")

# Add project root to path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from path_config import PathConfig

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
    "sw_in": "Shortwave Radiation (W/m²)",
    "ta": "Air Temperature (°C)",
    "ppfd_in": "PPFD (µmol/m²/s)",
    "ext_rad": "Extraterrestrial Radiation (W/m²)",
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
    ):
        valid_scales = {"sapwood", "plant", "site"}
        if scale not in valid_scales:
            raise ValueError(f"Invalid scale '{scale}'. Must be one of {valid_scales}.")
        self.paths = PathConfig(scale=scale)
        self.max_scatter_points = max_scatter_points
        self.use_raw = use_raw

        if output_dir is not None:
            self.output_dir = Path(output_dir)
        else:
            data_subfolder = "raw" if self.use_raw else "outlier_removed"
            self.output_dir = self.paths.figures_root / "relationship_exploration" / data_subfolder
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
        Load sap flow + environmental data from outlier-removed files.
        For each site, compute site-mean sap flow density per timestamp,
        merge with environmental variables, and tag with biome/PFT.
        """
        logger.info("Loading sap flow and environmental data …")

        sap_dir = self.paths.sap_outliers_removed_dir
        env_dir = self.paths.env_outliers_removed_dir

        if self.use_raw or not sap_dir.exists():
            if not self.use_raw:
                logger.warning(f"Sap outlier-removed dir not found: {sap_dir}")
                logger.info("Falling back to raw sap data directory …")
            else:
                logger.info("Using raw data directories (no outlier removal) …")
            sap_dir = self.paths.raw_csv_dir
            env_dir = self.paths.raw_csv_dir
            sap_files = sorted(sap_dir.glob("*_sapf_data.csv"))
        else:
            sap_files = sorted(sap_dir.glob("*_sapf_data_outliers_removed.csv"))

        logger.info(f"Found {len(sap_files)} sap flow files in {sap_dir}")

        site_md_cache = self._load_site_metadata_cache()
        all_dfs: list[pd.DataFrame] = []
        processed = 0

        for sf in sap_files:
            try:
                # Determine site code from filename
                stem = sf.stem
                if "_outliers_removed" in stem:
                    site_code = stem.replace("_sapf_data_outliers_removed", "")
                else:
                    site_code = stem.replace("_sapf_data", "")

                # ── Load sap flow data ──
                sap_df = pd.read_csv(sf, parse_dates=["TIMESTAMP"])
                if sap_df.empty:
                    continue

                # Compute site-mean sap flow density (mean across all tree columns)
                ts_col = "TIMESTAMP"
                value_cols = [c for c in sap_df.columns if c != ts_col]
                if not value_cols:
                    continue

                sap_df[value_cols] = sap_df[value_cols].apply(pd.to_numeric, errors="coerce")
                # Require ≥50% of sensors valid per timestamp
                min_valid = max(1, len(value_cols) // 2)
                n_valid = sap_df[value_cols].notna().sum(axis=1)
                sap_df["sap_flow_density"] = sap_df[value_cols].mean(axis=1).where(n_valid >= min_valid)
                sap_ts = sap_df[[ts_col, "sap_flow_density"]].copy()

                # ── Load environmental data ──
                if "_outliers_removed" in stem:
                    env_file = env_dir / f"{site_code}_env_data_outliers_removed.csv"
                else:
                    env_file = env_dir / f"{site_code}_env_data.csv"

                if not env_file.exists():
                    continue

                env_df = pd.read_csv(env_file, parse_dates=["TIMESTAMP"])
                if env_df.empty:
                    continue

                # Standardise env column names (lowercase)
                env_df.columns = [c.lower() if c != "TIMESTAMP" else c for c in env_df.columns]

                # Keep only needed env columns
                env_keep = ["TIMESTAMP"] + [c for c in ENV_VARIABLES if c in env_df.columns]
                env_df = env_df[env_keep]

                # ── Merge sap + env on TIMESTAMP ──
                merged = pd.merge(sap_ts, env_df, on="TIMESTAMP", how="inner")
                if merged.empty:
                    continue

                # ── Get biome / PFT from cache (np.nan when missing) ──
                biome, pft = site_md_cache.get(site_code, (np.nan, np.nan))

                merged["biome"] = biome
                merged["pft"] = pft
                merged["site_name"] = site_code

                all_dfs.append(merged)
                processed += 1

                if processed % 20 == 0:
                    logger.info(f"  Processed {processed} sites …")

            except Exception as e:
                logger.warning(f"Error processing {sf.name}: {e}")

        if not all_dfs:
            logger.error("No sap flow + env data could be loaded!")
            self.sapflow_env_df = pd.DataFrame()
            return self.sapflow_env_df

        combined = pd.concat(all_dfs, ignore_index=True)

        # Ensure numeric types
        numeric_cols = ["sap_flow_density"] + list(ENV_VARIABLES.keys())
        for c in numeric_cols:
            if c in combined.columns:
                combined[c] = pd.to_numeric(combined[c], errors="coerce")

        logger.info(
            f"Sap flow + env data loaded: {len(combined)} rows from "
            f"{processed} sites | biomes: {combined['biome'].nunique()}, "
            f"PFTs: {combined['pft'].nunique()}"
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
    ):
        """
        Draw scatter + LOWESS line on *ax*.
        Annotate with n and Spearman rho.
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

        ax.annotate(
            f"n = {n:,}\n{rho_str}",
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
    ):
        """Create a grid of scatter + LOWESS subplots, one per group."""
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

        ax.annotate(
            f"n = {n:,}\n{rho_str}",
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
            self._make_overall_plot(
                valid,
                x_col,
                "pl_sapw_area",
                x_label=x_label,
                y_label=y_label,
                title=f"{x_label} vs Sapwood Area — {n_plants:,} plants, {n_sites} sites",
                save_name=f"{tag}_vs_sapwood_area_overall.png",
                hue_col="site_code",
                color_map=None,  # auto color-cycle; too many sites for a legend
                suppress_legend=True,
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
        6 environmental variables (vpd, ws, sw_in, ta, ppfd_in, ext_rad).
        Each variable gets: overall plot, by-biome faceted, by-PFT faceted.
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
                for h, l in zip(*ax_cell.get_legend_handles_labels()):
                    if l not in seen_labels:
                        seen_labels[l] = h
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
    parser.add_argument("--use-raw", action="store_true", default=False, help="Use raw data directories without outlier removal (default: False, i.e. use outlier-removed data when available)",
    )
    args = parser.parse_args()

    explorer = RelationshipExplorer(
        scale=args.scale,
        output_dir=args.output_dir,
        max_scatter_points=args.max_points,
        use_raw=args.use_raw,
    )
    explorer.run()


if __name__ == "__main__":
    main()
