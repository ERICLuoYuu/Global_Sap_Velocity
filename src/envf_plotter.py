"""Interactive Environmental Data Plotter with Plotly.

Generates one HTML per env variable per site, showing:
  - Clean data line (outliers_removed)
  - Flagged data markers (red x)
  - Outlier markers (orange circle)
  - Gap-filled markers (green circle-open) [if gap_filled_masks dir provided]
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

# Env variable units
ENV_UNITS = {
    "ta": "\u00b0C",
    "rh": "%",
    "vpd": "kPa",
    "sw_in": "W\u00b7m\u207b\u00b2",
    "netrad": "W\u00b7m\u207b\u00b2",
    "ws": "m\u00b7s\u207b\u00b9",
    "precip": "mm",
    "swc_shallow": "m\u00b3\u00b7m\u207b\u00b3",
    "ppfd_in": "\u00b5mol\u00b7m\u207b\u00b2\u00b7s\u207b\u00b9",
    "ext_rad": "W\u00b7m\u207b\u00b2",
}

# Columns to skip when iterating env variables
SKIP_COLS = {"TIMESTAMP", "solar_TIMESTAMP", "TIMESTAMP_LOCAL", "lat", "long"}


class EnvPlotter:
    """Interactive environmental data plotter."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        outlier_dir: str | None = None,
        flagged_dir: str | None = None,
        gap_filled_dir: str | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.outlier_dir = Path(outlier_dir) if outlier_dir else None
        self.flagged_dir = Path(flagged_dir) if flagged_dir else None
        self.gap_filled_dir = Path(gap_filled_dir) if gap_filled_dir else None
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _parse_site_name(self, file_path: Path) -> str:
        """Extract site name from env filename.

        e.g. AUS_WOM from AUS_WOM_env_data_outliers_removed.csv
        """
        stem = file_path.stem
        for suffix in ["_env_data_outliers_removed", "_env_data"]:
            if stem.endswith(suffix):
                return stem[: -len(suffix)]
        if "_env_" in stem:
            return stem[: stem.index("_env_")]
        return stem

    def _load_outliers(self, site: str, var: str) -> pd.DataFrame | None:
        """Load per-variable outlier file: {site}_env_data_{var}_outliers.csv."""
        if self.outlier_dir is None:
            return None
        path = self.outlier_dir / f"{site}_env_data_{var}_outliers.csv"
        if not path.exists():
            return None
        try:
            return pd.read_csv(path, parse_dates=["timestamp"])
        except Exception:
            return None

    def _load_flagged(self, site: str, var: str) -> pd.DataFrame | None:
        """Load per-variable flagged file: {site}_env_data_{var}_flagged.csv."""
        if self.flagged_dir is None:
            return None
        path = self.flagged_dir / f"{site}_env_data_{var}_flagged.csv"
        if not path.exists():
            return None
        try:
            return pd.read_csv(path, parse_dates=["TIMESTAMP"])
        except Exception:
            return None

    def _load_gap_filled(self, site: str, var: str) -> pd.DataFrame | None:
        """Load per-variable gap-fill mask: {site}_env_data_{var}_gap_filled.csv."""
        if self.gap_filled_dir is None:
            return None
        path = self.gap_filled_dir / f"{site}_env_data_{var}_gap_filled.csv"
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path, parse_dates=["timestamp"])
            if "is_gap_filled" in df.columns:
                return df[df["is_gap_filled"]]
            return None
        except Exception:
            return None

    def create_plot(
        self,
        env_df: pd.DataFrame,
        site: str,
        var: str,
    ) -> go.Figure | None:
        """Create a single interactive plot for one env variable."""
        ts_col = "solar_TIMESTAMP" if "solar_TIMESTAMP" in env_df.columns else "TIMESTAMP"
        if ts_col not in env_df.columns or var not in env_df.columns:
            return None

        env_df[var] = pd.to_numeric(env_df[var], errors="coerce")
        valid = env_df[[var, ts_col]].dropna()
        if valid.empty:
            return None

        fig = go.Figure()

        # 1. Main trace (clean data)
        fig.add_trace(
            go.Scatter(
                x=valid[ts_col],
                y=valid[var],
                mode="lines+markers",
                name=f"{var} (clean)",
                line=dict(color="blue", width=1),
                marker=dict(size=2, color="blue"),
                hovertemplate=f"<b>Time:</b> %{{x}}<br><b>{var}:</b> %{{y:.3f}}<extra></extra>",
            )
        )

        # 2. Flagged (red x)
        flagged = self._load_flagged(site, var)
        flagged_count = 0
        if flagged is not None and var in flagged.columns:
            flagged[var] = pd.to_numeric(flagged[var], errors="coerce")
            flagged_valid = flagged[["TIMESTAMP", var]].dropna()
            if not flagged_valid.empty:
                flagged_count = len(flagged_valid)
                fig.add_trace(
                    go.Scatter(
                        x=flagged_valid["TIMESTAMP"],
                        y=flagged_valid[var],
                        mode="markers",
                        name="Flagged",
                        marker=dict(size=8, color="red", symbol="x"),
                        hovertemplate=f"<b>Time:</b> %{{x}}<br><b>{var}:</b> %{{y:.3f}}<br><b>Type:</b> Flagged<extra></extra>",
                    )
                )

        # 3. Outliers (orange)
        outliers = self._load_outliers(site, var)
        outlier_count = 0
        if outliers is not None and "is_outlier" in outliers.columns:
            outlier_pts = outliers[outliers["is_outlier"]]
            if not outlier_pts.empty:
                outlier_count = len(outlier_pts)
                fig.add_trace(
                    go.Scatter(
                        x=outlier_pts["timestamp"],
                        y=outlier_pts["value"],
                        mode="markers",
                        name="Outliers",
                        marker=dict(size=6, color="orange", symbol="circle"),
                        hovertemplate=f"<b>Time:</b> %{{x}}<br><b>{var}:</b> %{{y:.3f}}<br><b>Type:</b> Outlier<extra></extra>",
                    )
                )

        # 4. Gap-filled (green)
        gf = self._load_gap_filled(site, var)
        gf_count = 0
        if gf is not None and not gf.empty:
            gf_count = len(gf)
            ts_gf = "solar_TIMESTAMP" if "solar_TIMESTAMP" in gf.columns else "timestamp"
            fig.add_trace(
                go.Scatter(
                    x=gf[ts_gf],
                    y=gf["value"],
                    mode="markers",
                    name="Gap Filled",
                    marker=dict(
                        size=5,
                        color="limegreen",
                        symbol="circle-open",
                        line=dict(width=1, color="green"),
                    ),
                    hovertemplate=f"<b>Time:</b> %{{x}}<br><b>{var}:</b> %{{y:.3f}}<br><b>Type:</b> Gap Filled<extra></extra>",
                )
            )

        unit = ENV_UNITS.get(var, "")
        date_start = env_df.index.min()
        date_end = env_df.index.max()
        if hasattr(date_start, "strftime"):
            date_start = date_start.strftime("%Y-%m-%d")
            date_end = date_end.strftime("%Y-%m-%d")

        fig.update_layout(
            title={
                "text": f"{var.upper()} - {site}<br>"
                + f"Period: {date_start} to {date_end}<br>"
                + f"<sub>Valid: {len(valid)} | Outliers: {outlier_count} | "
                + f"Flagged: {flagged_count} | Gap Filled: {gf_count}</sub>",
                "x": 0.5,
                "xanchor": "center",
            },
            xaxis_title="Timestamp",
            yaxis_title=f"{var} ({unit})" if unit else var,
            hovermode="closest",
            template="plotly_white",
            height=500,
            showlegend=True,
            xaxis=dict(rangeslider=dict(visible=True)),
        )

        return fig

    def process_file(self, file_path: Path) -> None:
        """Process one env outliers_removed CSV: create one plot per variable."""
        site = self._parse_site_name(file_path)
        print(f"\nProcessing env: {site} ({file_path.name})")

        df = pd.read_csv(file_path, parse_dates=["TIMESTAMP"])
        if "solar_TIMESTAMP" in df.columns:
            df["solar_TIMESTAMP"] = pd.to_datetime(df["solar_TIMESTAMP"])
            df.set_index("solar_TIMESTAMP", drop=False, inplace=True)
        else:
            df.set_index("TIMESTAMP", drop=False, inplace=True)

        env_vars = [c for c in df.columns if c not in SKIP_COLS and not c.startswith("Unnamed")]

        site_dir = self.output_dir / site
        site_dir.mkdir(parents=True, exist_ok=True)

        for var in env_vars:
            if df[var].isna().all():
                continue
            fig = self.create_plot(df, site, var)
            if fig is not None:
                out_path = site_dir / f"{var}.html"
                fig.write_html(out_path, config={"displayModeBar": True})
                print(f"    Created: {out_path.name}")
                del fig

    def process_all_files(self, file_pattern: str = "*_env_data_outliers_removed.csv") -> None:
        files = sorted(self.data_dir.glob(file_pattern))
        print(f"Found {len(files)} env files to process")
        for f in files:
            try:
                self.process_file(f)
            except Exception as e:
                print(f"ERROR processing {f.name}: {e}")
                import traceback

                traceback.print_exc()


if __name__ == "__main__":
    DATA_DIR = "./outputs/processed_data/sapwood/env/outliers_removed"
    OUTPUT_DIR = "./outputs/figures/sapwood/env/cleaned_interactive"
    OUTLIER_DIR = "./outputs/processed_data/sapwood/env/outliers"
    FLAGGED_DIR = "./outputs/processed_data/sapwood/env/flagged"

    print("Interactive Environmental Data Plotter")
    plotter = EnvPlotter(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        outlier_dir=OUTLIER_DIR,
        flagged_dir=FLAGGED_DIR,
    )
    plotter.process_all_files()
