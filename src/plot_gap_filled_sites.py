"""Generate interactive visualizations for test sites (sap + env).

This script only generates plots — it does NOT run gap-filling.
Gap-filling is done during data processing (SapFlowAnalyzer).

Sap plots show: clean data, outliers, variability, flagged, reversed, gap-filled.
Env plots show: clean data, outliers, flagged (no sap overlay).
"""

from __future__ import annotations

from pathlib import Path

from src.envf_plotter import EnvPlotter
from src.sapf_plotter import FastSapFlowPlotterDual

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
DATA_ROOT = Path("outputs/processed_data/sapwood")

SAP_DIR = DATA_ROOT / "sap" / "outliers_removed"
ENV_DIR = DATA_ROOT / "env" / "outliers_removed"
GAP_FILLED_MASKS_DIR = DATA_ROOT / "sap" / "gap_filled_masks"

SAP_OUTPUT_DIR = Path("outputs/figures/sapwood/sap/gap_filled_interactive")
ENV_OUTPUT_DIR = Path("outputs/figures/sapwood/env/cleaned_interactive")

OUTLIER_DIR = DATA_ROOT / "sap" / "outliers"
VARIABILITY_DIR = DATA_ROOT / "sap" / "variability_masks"
FLAGGED_DIR = DATA_ROOT / "sap" / "filtered"
REVERSED_DIR = DATA_ROOT / "sap" / "reversed"

ENV_OUTLIER_DIR = DATA_ROOT / "env" / "outliers"
ENV_FLAGGED_DIR = DATA_ROOT / "env" / "flagged"

SITES = ["AUS_WOM", "DEU_HIN_OAK"]


def _find_file(directory: Path, site: str, kind: str) -> Path | None:
    """Find site file matching the naming convention."""
    for pattern in [f"{site}_*_{kind}.csv", f"{site}_{kind}.csv"]:
        matches = list(directory.glob(pattern))
        if matches:
            return matches[0]
    return None


def plot_sap() -> None:
    """Generate sap flow interactive plots for test sites."""
    plotter = FastSapFlowPlotterDual(
        data_dir=str(SAP_DIR),
        env_dir=str(ENV_DIR),
        output_dir=str(SAP_OUTPUT_DIR),
        outlier_dir=str(OUTLIER_DIR) if OUTLIER_DIR.exists() else None,
        variability_dir=str(VARIABILITY_DIR) if VARIABILITY_DIR.exists() else None,
        flagged_dir=str(FLAGGED_DIR) if FLAGGED_DIR.exists() else None,
        reversed_dir=str(REVERSED_DIR) if REVERSED_DIR.exists() else None,
        gap_filled_dir=str(GAP_FILLED_MASKS_DIR) if GAP_FILLED_MASKS_DIR.exists() else None,
        env_var="vpd",
    )

    for site in SITES:
        sap_file = _find_file(SAP_DIR, site, "sapf_data_outliers_removed")
        if sap_file is None:
            print(f"  SKIP sap {site}: file not found")
            continue
        print(f"\n  Plotting sap: {site}")
        plotter.process_file(sap_file)


def plot_env() -> None:
    """Generate env interactive plots for test sites."""
    plotter = EnvPlotter(
        data_dir=str(ENV_DIR),
        output_dir=str(ENV_OUTPUT_DIR),
        outlier_dir=str(ENV_OUTLIER_DIR) if ENV_OUTLIER_DIR.exists() else None,
        flagged_dir=str(ENV_FLAGGED_DIR) if ENV_FLAGGED_DIR.exists() else None,
    )

    for site in SITES:
        env_file = _find_file(ENV_DIR, site, "env_data_outliers_removed")
        if env_file is None:
            print(f"  SKIP env {site}: file not found")
            continue
        print(f"\n  Plotting env: {site}")
        plotter.process_file(env_file)


def main() -> None:
    print("=" * 60)
    print("Visualization Pipeline (Sap + Env)")
    print("=" * 60)

    print("\nStep 1: Sap flow plots...")
    plot_sap()

    print("\nStep 2: Env plots...")
    plot_env()

    print("\nDone.")
    print(f"  Sap plots: {SAP_OUTPUT_DIR}")
    print(f"  Env plots: {ENV_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
