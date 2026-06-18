# src/fu_ann_sensitivity/run_fu_ann_sensitivity.py
"""CLI orchestration of the Fu et al. 2022 ANN sensitivity analysis.

For each (response, SM-variant) pair: load + per-site z-score, fit the per-site ANN
ensembles ONCE, then bin the reused per-row sensitivities at every requested bin
count (5x5, 10x10). Writes per-site summaries, cross-site sensitivity maps + dual-leg
tables, a performance/attrition table, and the Fu-style figure set.

Run on Palma via sbatch (job_fu_ann_sensitivity.sh) -- never the login node.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.fu_ann_sensitivity.aggregate import aggregate_at_nbins, fit_site_sensitivities
from src.fu_ann_sensitivity.loader import EXCLUDE_PFT, load_table, zscore_per_site
from src.fu_ann_sensitivity.plotting import (
    plot_dual_legs,
    plot_pft_panels,
    plot_sensitivity_heatmap,
)

logger = logging.getLogger(__name__)

RESPONSES = ("E", "Gc")
SM_VARIANTS = ("swvl1", "swvl2", "swvl3", "swvl4", "root_zone_sm")
BASE_PREDICTORS = ("tair", "vpd", "ppfd")
DEFAULT_N_BINS = (5, 10)
DEFAULT_MIN_VALID_DAYS = 300  # Fu et al. 2022: >= 300 growing-season days
# Canonical growing-season daily dataset (Fu restricts to the growing season).
DEFAULT_DATA_DIR = "outputs/processed_data/sapwood/merged/daytime_only/growing_season/daily"


def _write_maps(agg: dict, out: Path, response: str, sm: str, n_bins: int) -> None:
    """Long-form cell table: one row per (sm_bin, vpd_bin) with both legs + sig."""
    recs = []
    for i in range(n_bins):
        for j in range(n_bins):
            recs.append(
                {
                    "sm_bin": i,
                    "vpd_bin": j,
                    "d_sm": agg["sm_map"][i, j],
                    "d_sm_sig": bool(agg["sm_sig"][i, j]),
                    "d_vpd": agg["vpd_map"][i, j],
                    "d_vpd_sig": bool(agg["vpd_sig"][i, j]),
                }
            )
    pd.DataFrame(recs).to_csv(out / f"maps_{response}_{sm}_nbins{n_bins}.csv", index=False)


def _aggregate_per_pft(per_site, n_bins: int) -> dict:
    """{pft: by_sm_bin} for the per-PFT small-multiples (Fig 3a by PFT)."""
    panels = {}
    for pft in sorted({p["pft"] for p in per_site}):
        grp = [p for p in per_site if p["pft"] == pft]
        try:
            panels[pft] = aggregate_at_nbins(grp, n_bins)["by_sm_bin"]
        except ValueError:
            continue
    return panels


def _make_figures(agg: dict, per_site, out: Path, response: str, sm: str, n_bins: int) -> None:
    tag = f"{response}_{sm}_nbins{n_bins}"
    plot_sensitivity_heatmap(
        agg["sm_map"],
        agg["sm_sig"],
        f"Sensitivity of {response} to SWC ({sm}, {n_bins}x{n_bins})",
        out / f"fig2_{response}_{sm}_SWC_nbins{n_bins}.png",
    )
    plot_sensitivity_heatmap(
        agg["vpd_map"],
        agg["vpd_sig"],
        f"Sensitivity of {response} to VPD ({sm}, {n_bins}x{n_bins})",
        out / f"fig2_{response}_{sm}_VPD_nbins{n_bins}.png",
    )
    plot_dual_legs(agg["by_sm_bin"], agg["by_vpd_bin"], response, out / f"fig3_{tag}.png")
    if sm == "root_zone_sm":
        panels = _aggregate_per_pft(per_site, n_bins)
        if panels:
            plot_pft_panels(panels, response, out / f"fig_pft_{response}_nbins{n_bins}.png")


def run_analysis(
    data_dir,
    out_dir,
    sm_variants=SM_VARIANTS,
    responses=RESPONSES,
    n_bins_list=DEFAULT_N_BINS,
    n_repeats: int = 5,
    min_valid_days: int = DEFAULT_MIN_VALID_DAYS,
    r_threshold: float = 0.5,
    exclude_pft=EXCLUDE_PFT,
    max_sites=None,
    make_figures: bool = True,
) -> Path:
    """Full Fu ANN sensitivity run; returns the output directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    table = load_table(data_dir, exclude_pft=exclude_pft)
    if max_sites is not None:
        keep = list(dict.fromkeys(table["site_name"]))[:max_sites]
        table = table[table["site_name"].isin(keep)].copy()

    zcols = list(BASE_PREDICTORS) + list(sm_variants) + list(responses)
    table_z = zscore_per_site(table, cols=zcols, response=responses[0])

    perf_rows = []
    for response in responses:
        for sm in sm_variants:
            predictors_z = ["tair_z", "vpd_z", f"{sm}_z", "ppfd_z"]
            per_site, n_dropped, median_r = fit_site_sensitivities(
                table_z,
                response=f"{response}_z",
                sm_col=f"{sm}_z",
                vpd_col="vpd_z",
                predictors=predictors_z,
                min_valid_days=min_valid_days,
                n_repeats=n_repeats,
                r_threshold=r_threshold,
            )
            perf_rows.append(
                {
                    "response": response,
                    "sm_variant": sm,
                    "n_sites": len(per_site),
                    "n_dropped": n_dropped,
                    "median_r": median_r,
                }
            )
            pd.DataFrame([{k: p[k] for k in ("site", "pft", "r", "n_days")} for p in per_site]).to_csv(
                out / f"per_site_{response}_{sm}.csv", index=False
            )

            if not per_site:
                logger.warning("No sites survived for %s/%s; skipping maps.", response, sm)
                continue
            for n_bins in n_bins_list:
                agg = aggregate_at_nbins(per_site, n_bins)
                _write_maps(agg, out, response, sm, n_bins)
                agg["by_sm_bin"].to_csv(out / f"by_sm_bin_{response}_{sm}_nbins{n_bins}.csv", index=False)
                agg["by_vpd_bin"].to_csv(out / f"by_vpd_bin_{response}_{sm}_nbins{n_bins}.csv", index=False)
                if make_figures:
                    _make_figures(agg, per_site, out, response, sm, n_bins)

    pd.DataFrame(perf_rows).to_csv(out / "performance.csv", index=False)
    logger.info("Done. Outputs in %s", out)
    return out


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Fu et al. 2022 ANN SWC/VPD sensitivity for E and Gc.")
    p.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Daily per-site CSV dir (default: canonical growing-season dataset).",
    )
    p.add_argument("--out-dir", default="outputs/fu_ann_sensitivity")
    p.add_argument("--responses", nargs="+", default=list(RESPONSES))
    p.add_argument("--sm-variants", nargs="+", default=list(SM_VARIANTS))
    p.add_argument("--n-bins", nargs="+", type=int, default=list(DEFAULT_N_BINS))
    p.add_argument("--n-repeats", type=int, default=5)
    p.add_argument("--min-valid-days", type=int, default=DEFAULT_MIN_VALID_DAYS)
    p.add_argument("--r-threshold", type=float, default=0.5)
    p.add_argument("--keep-all-pft", action="store_true", help="Disable Fu's cropland/wetland exclusion.")
    p.add_argument("--max-sites", type=int, default=None, help="Smoke-run cap on sites.")
    p.add_argument("--no-figures", action="store_true")
    a = p.parse_args(argv)

    run_analysis(
        data_dir=a.data_dir,
        out_dir=a.out_dir,
        sm_variants=a.sm_variants,
        responses=a.responses,
        n_bins_list=a.n_bins,
        n_repeats=a.n_repeats,
        min_valid_days=a.min_valid_days,
        r_threshold=a.r_threshold,
        exclude_pft=() if a.keep_all_pft else EXCLUDE_PFT,
        max_sites=a.max_sites,
        make_figures=not a.no_figures,
    )


if __name__ == "__main__":
    main()
