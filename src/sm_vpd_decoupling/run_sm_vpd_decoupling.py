# src/sm_vpd_decoupling/run_sm_vpd_decoupling.py
"""CLI: orchestrate SM-VPD decoupling over responses x SM variants x bin counts
x min-valid-days. Writes per-site CSVs, a depth-profile dissociation table, an
attrition table, and figures.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.sm_vpd_decoupling.aggregate import decouple_all_sites, dominance_summary
from src.sm_vpd_decoupling.loader import (
    load_table,
)
from src.sm_vpd_decoupling.plotting import (
    plot_aggregate_lines,
    plot_cross_site_aggregate,
    plot_depth_dominance,
    plot_example_sites,
    plot_gradient_groups,
    plot_leg_comparison_box,
    plot_leg_comparison_scatter,
    select_example_sites,
)

logger = logging.getLogger(__name__)

T_MIN_PRIMARY: float = 15.0  # Liu 2020 day filter: minimum air temperature (deg C)
T_MIN_SENSITIVITY: float = 5.0  # Sensitivity sweep: lower T_min for a broader sample

SM_VARIANTS = ("swvl1", "swvl2", "swvl3", "swvl4", "root_zone_sm")
RESPONSES = ("E_norm", "Gc_norm")


def _attrition(table: pd.DataFrame, min_valid_days_list: list[int]) -> pd.DataFrame:
    """Sites & rows surviving each downstream threshold (post day-filter table)."""
    rows = [{"stage": "day_filtered", "n_sites": table["site_name"].nunique(), "n_rows": len(table)}]
    per_site_days = table.groupby("site_name").size()
    for mvd in min_valid_days_list:
        rows.append(
            {
                "stage": f"min_valid_days>={mvd}",
                "n_sites": int((per_site_days >= mvd).sum()),
                "n_rows": int(per_site_days[per_site_days >= mvd].sum()),
            }
        )
    return pd.DataFrame(rows)


def run_analysis(
    data_dir: str | None,
    out_dir: str,
    climate_source: str = "site",
    tair_min: float = T_MIN_PRIMARY,
    n_bins_list: list[int] | None = None,
    min_valid_days_list: list[int] | None = None,
    make_figures: bool = True,
) -> None:
    n_bins_list = n_bins_list or [5, 10]
    min_valid_days_list = min_valid_days_list or [120, 240, 360]
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    table = load_table(data_dir, climate_source=climate_source, tair_min=tair_min)
    _attrition(table, min_valid_days_list).to_csv(out / "attrition.csv", index=False)

    depth_rows: list[dict] = []
    for response in RESPONSES:
        for sm_col in SM_VARIANTS:
            for n_bins in n_bins_list:
                for mvd in min_valid_days_list:
                    eff = decouple_all_sites(table, sm_col=sm_col, response=response, n_bins=n_bins, min_valid_days=mvd)
                    tag = f"{response}_{sm_col}_nbins{n_bins}_mvd{mvd}"
                    eff.to_csv(out / f"per_site_{tag}.csv", index=False)
                    pct, n = dominance_summary(eff)
                    depth_rows.append(
                        {
                            "response": response,
                            "sm_variant": sm_col,
                            "n_bins": n_bins,
                            "min_valid_days": mvd,
                            "pct_sm_dominant": pct,
                            "n_sites": n,
                            "mean_sm_given_vpd": eff["sm_given_vpd"].mean(),
                            "mean_vpd_given_sm": eff["vpd_given_sm"].mean(),
                        }
                    )
    depth = pd.DataFrame(depth_rows)
    depth.to_csv(out / "depth_profile.csv", index=False)

    if make_figures:
        _make_figures(table, depth, out, n_bins_list, min_valid_days_list)
    logger.info("Analysis complete -> %s", out)


def _make_figures(table, depth, out, n_bins_list, min_valid_days_list):
    figdir = out / "figures"
    nb, mvd = n_bins_list[0], min_valid_days_list[0]
    example_sites = select_example_sites(table, n=4)
    for response in RESPONSES:
        # (a2) cross-site 2-D decoupling heatmap (+ marginal effect bars) per SM variant,
        # and assemble per-site effects across all SM variants for the comparison box.
        per_variant = []
        for sm_col in SM_VARIANTS:
            plot_cross_site_aggregate(
                table,
                response=response,
                sm_col=sm_col,
                n_bins=nb,
                out_path=str(figdir / f"grid_{response}_{sm_col}.png"),
            )
            plot_aggregate_lines(
                table,
                response=response,
                sm_col=sm_col,
                n_bins=nb,
                out_path=str(figdir / f"aggline_{response}_{sm_col}.png"),
            )
            eff_v = decouple_all_sites(table, sm_col=sm_col, response=response, n_bins=nb, min_valid_days=mvd)
            per_variant.append(eff_v.assign(sm_variant=sm_col))
        combined = pd.concat(per_variant, ignore_index=True)

        # (a1) example-site within-bin lines on the root-zone SM axis
        plot_example_sites(
            table,
            sites=example_sites,
            response=response,
            sm_col="root_zone_sm",
            n_bins=nb,
            out_path=str(figdir / f"examples_{response}.png"),
        )

        # (b) depth dominance
        sub = depth[(depth["response"] == response) & (depth["n_bins"] == nb) & (depth["min_valid_days"] == mvd)]
        if not sub.empty:
            plot_depth_dominance(sub, response=response, out_path=str(figdir / f"depth_{response}.png"))

        # leg comparison: signed paired boxplots (all SM variants) + magnitude scatter (root-zone)
        plot_leg_comparison_box(combined, response=response, out_path=str(figdir / f"compare_box_{response}.png"))
        rz = combined[combined["sm_variant"] == "root_zone_sm"]
        plot_leg_comparison_scatter(
            rz,
            response=response,
            sm_col="root_zone_sm",
            out_path=str(figdir / f"compare_scatter_{response}.png"),
        )

        # (c) per-covariate gradients showing BOTH decoupled legs (root-zone)
        for grp in ("pft", "biome", "aridity", "canopy_height"):
            if grp in rz.columns:
                plot_gradient_groups(
                    rz,
                    group_col=grp,
                    response=response,
                    sm_col="root_zone_sm",
                    out_path=str(figdir / f"grad_{response}_{grp}.png"),
                )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Site-level SM-VPD decoupling.")
    p.add_argument("--data-dir", default=None, help="Daily CSV dir (auto-resolved if omitted)")
    p.add_argument("--out-dir", default="outputs/sm_vpd_decoupling")
    p.add_argument("--climate-source", choices=["site", "era5"], default="site")
    p.add_argument("--tair-min", type=float, default=T_MIN_PRIMARY)
    p.add_argument("--n-bins", type=int, nargs="+", default=[5, 10])
    p.add_argument("--min-valid-days", type=int, nargs="+", default=[120, 240, 360])
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args()
    run_analysis(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        climate_source=args.climate_source,
        tair_min=args.tair_min,
        n_bins_list=args.n_bins,
        min_valid_days_list=args.min_valid_days,
        make_figures=not args.no_figures,
    )


if __name__ == "__main__":
    main()
