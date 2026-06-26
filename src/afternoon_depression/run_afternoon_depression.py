"""CLI runner: afternoon depression of transpiration — climate-driver decoupling.

Pipeline (Liu et al. 2024, EC correspondence):
  hourly GS+raw CSVs → EC Text S1 filters → per-site-day ΔSF/C_SF table
  → per-site binned decoupling (Text S2) → RF ±1 SD sensitivity (Text S3)
  → figures + summary CSVs + a markdown verdict.

Usage:
    python src/afternoon_depression/run_afternoon_depression.py \
        --scale sapwood --climate-source era5 --min-valid-days 120
    # local dev against a synthetic/site dir:
    python src/afternoon_depression/run_afternoon_depression.py --data-dir /path/to/hourly --climate-source site
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Repo root on path so both `path_config` and `src.afternoon_depression.*` import.
_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT in sys.path:
    sys.path.remove(_ROOT)
sys.path.insert(0, _ROOT)

from path_config import PathConfig  # noqa: E402
from src.afternoon_depression.data_loader import load_site_day_table, resolve_hourly_dir  # noqa: E402
from src.afternoon_depression.decoupling import decouple_all_sites  # noqa: E402
from src.afternoon_depression.diurnal_metrics import aggregate_monthly  # noqa: E402
from src.afternoon_depression.plotting import (  # noqa: E402
    aridity_class,
    plot_decoupling_lines,
    plot_driver_responses,
    plot_effect_distributions,
    plot_effects_by_group,
)
from src.afternoon_depression.sensitivity import rf_sensitivity  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_EFFECTS = ["vpd_given_tair", "tair_given_vpd", "vpd_given_sm", "sm_given_vpd"]


def _summarise(effects: pd.DataFrame, scope: str) -> dict:
    row = {"scope": scope, "n_sites": int(len(effects))}
    for e in _EFFECTS:
        if e in effects.columns:
            row[f"{e}_median"] = float(effects[e].median())
    return row


def build_summary(effects: pd.DataFrame) -> pd.DataFrame:
    """Overall + per-PFT + per-aridity-class median decoupled effects."""
    rows = [_summarise(effects, "overall")]
    if "pft" in effects.columns:
        for pft, g in effects.groupby("pft"):
            rows.append(_summarise(g, f"pft:{pft}"))
    if "aridity_class" in effects.columns:
        for ac, g in effects.groupby("aridity_class", observed=True):
            rows.append(_summarise(g, f"aridity:{ac}"))
    return pd.DataFrame(rows)


def _verdict(
    effects: pd.DataFrame,
    rf: dict | None,
    rf_sd_scope: str = "site",
    response: str = "sf",
    gc_winsor_top_frac: float = 0.01,
) -> str:
    is_gc = response == "gc"
    dx = "ΔGc" if is_gc else "ΔSF"
    metric = "canopy conductance (Gc)" if is_gc else "sap flow"
    med = {e: float(effects[e].median()) for e in _EFFECTS if e in effects.columns}
    vpd_vs_t, t_vs_v = med.get("vpd_given_tair"), med.get("tair_given_vpd")
    vpd_vs_s, s_vs_v = med.get("vpd_given_sm"), med.get("sm_given_vpd")
    vpd_flag = "  ⚠️ VPD leg confounded (Gc∝1/VPD)" if is_gc else ""
    lines = [
        f"# Afternoon depression of {metric} — driver verdict\n",
        f"- Sites analysed: {len(effects)}",
        f"- {dx}(VPD|Tair) = {vpd_vs_t:.2f}%  vs  {dx}(Tair|VPD) = {t_vs_v:.2f}%  "
        f"→ **{'VPD' if (vpd_vs_t or 0) > (t_vs_v or 0) else 'Tair'} dominates** the Tair–VPD contrast{vpd_flag}",
        f"- {dx}(VPD|SM) = {vpd_vs_s:.2f}%  vs  {dx}(SM|VPD) = {s_vs_v:.2f}%  "
        f"→ **{'VPD' if (vpd_vs_s or 0) > (s_vs_v or 0) else 'SM'} dominates** the SM–VPD contrast{vpd_flag}",
    ]
    if rf:
        order = sorted(rf.items(), key=lambda kv: abs(kv[1]) if kv[1] == kv[1] else -1, reverse=True)
        lines.append(
            f"- RF ±1 SD sensitivity ({rf_sd_scope}-SD, |{dx}|): " + ", ".join(f"{k}={v:.3f}" for k, v in order)
        )
    if is_gc:
        lines.append(
            "\n> **Critical caveat — the 1/VPD confound.** Flo et al. (2021) Eqn 2 makes "
            "Gc ∝ SFD/VPD. Afternoon VPD is structurally higher than morning VPD, so ΔGc is *partly "
            "a mechanical artifact* of dividing by a larger afternoon VPD, not purely stomatal closure. "
            "Binning Gc **by VPD** induces a spurious negative Gc–VPD slope (Oren et al. 1999): the "
            "**VPD|Tair** and **VPD|SM** legs are therefore CONFOUNDED. The **SM|VPD** leg (binned "
            "*within* VPD) and the **Tair|VPD** leg are unaffected and are the interpretable results — "
            "read the SM–VPD contrast off the SM|VPD value, not the VPD|SM value."
        )
        if gc_winsor_top_frac and gc_winsor_top_frac > 0:
            lines.append(
                f"\n> **Artifact removal.** The global top {gc_winsor_top_frac * 100:.3g}% of site-days by Gc "
                "magnitude were winsorised out before analysis. Gc = SFD/VPD blows up when a daytime hour pairs "
                "near-zero VPD with non-zero sap flow (up to ~1.6×10¹⁷ mol m⁻² s⁻¹), producing a tail that sits "
                "above an ~11-order-of-magnitude discontinuity in the distribution (p99 ≈ 1.1×10⁴). The cut is "
                "GLOBAL (not per-site — the explosion is concentrated at a few humid sites) and drops whole "
                "site-days, so AM/PM comparability is preserved."
            )
    lines.append(
        f"\n_Caveats: afternoon depression of {metric} is partly hydraulic (capacitance/hysteresis), "
        "not purely stomatal; effects are per-site percentile-binned then aggregated; residual "
        "VPD–Tair coupling may persist in sparse bins._"
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    paths = PathConfig(scale=args.scale)
    hourly_dir = resolve_hourly_dir(args.scale, args.data_dir, paths.processed_root)
    # Gc outputs go to a `gc/` subdir so the sap-velocity (sf) outputs stay untouched;
    # an explicit --output-dir always wins (used by the threshold sweeps).
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = paths.base_output_dir / "afternoon_depression"
        if args.response == "gc":
            out_dir = out_dir / "gc"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output → %s", out_dir)

    # Figure/verdict labels for the chosen response (defaults reproduce the sf strings).
    resp_label = "ΔGc" if args.response == "gc" else "ΔSF"
    metric_name = "canopy conductance (Gc)" if args.response == "gc" else "sap flow"

    table = load_site_day_table(
        hourly_dir,
        climate_source=args.climate_source,
        response=args.response,
        tair_min=args.tair_min,
        min_daily_sf=args.min_daily_sf,
        min_window_hours=args.min_window_hours,
        min_am_pm_ratio=args.min_am_pm_ratio,
        gc_winsor_top_frac=args.gc_winsor_top_frac,
    )
    if table.empty:
        logger.error("Empty site-day table after filtering — nothing to analyse.")
        return
    if args.restrict_to_qualifying:
        # Pre-filter so Fig 1, the RF cross-check, AND the decoupling all share ONE
        # dense-site set — the "only trust sites with ≥N valid days" scenario. Uses the
        # SAME valid-day count as decouple_all_sites so the kept set matches exactly.
        needed = ["vpd", "tair", "sm", "delta_sf"]
        counts = table.dropna(subset=needed).groupby("site_name").size()
        keep = counts[counts >= args.min_valid_days].index
        n_before = table["site_name"].nunique()
        table = table[table["site_name"].isin(keep)].copy()
        logger.info(
            "Restricted to %d/%d sites with ≥%d valid days (Fig1 + RF + decoupling share one set).",
            table["site_name"].nunique(),
            n_before,
            args.min_valid_days,
        )
        if table.empty:
            logger.error("No site meets --min-valid-days=%d under --restrict-to-qualifying.", args.min_valid_days)
            return
    table.to_csv(out_dir / "site_day_table.csv", index=False)
    logger.info("Site-day table: %d rows, %d sites", len(table), table["site_name"].nunique())

    effects = decouple_all_sites(table, min_valid_days=args.min_valid_days, n_bins=args.n_bins)
    if effects.empty:
        logger.error("No site passed --min-valid-days=%d; lower it or relax filters.", args.min_valid_days)
        return
    # attach per-site aridity class for stratification
    if "aridity" in table.columns:
        site_ai = table.groupby("site_name")["aridity"].median()
        effects["aridity"] = effects["site_name"].map(site_ai)
        effects["aridity_class"] = aridity_class(effects["aridity"])
    effects.to_csv(out_dir / "site_effects.csv", index=False)

    rf = None
    if not args.no_rf:
        rf = rf_sensitivity(table, n_models=args.rf_models, sd_scope=args.rf_sd_scope)
        pd.DataFrame([rf]).to_csv(out_dir / "rf_sensitivity.csv", index=False)

    build_summary(effects).to_csv(out_dir / "decoupling_summary.csv", index=False)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    try:
        plot_driver_responses(
            table, fig_dir / "fig1_driver_responses.png", resp_label=resp_label, metric_name=metric_name
        )
        plot_decoupling_lines(
            table,
            fig_dir / "fig2abc_decoupling_lines.png",
            n_bins=args.n_bins,
            min_valid_days=args.min_valid_days,
            resp_label=resp_label,
        )
        plot_effect_distributions(
            effects, fig_dir / "fig2d_effect_distributions.png", resp_label=resp_label, metric_name=metric_name
        )
        plot_effects_by_group(effects, "pft", fig_dir / "fig2g_effects_by_pft.png", resp_label=resp_label)
        plot_effects_by_group(effects, "aridity_class", fig_dir / "fig2h_effects_by_aridity.png", resp_label=resp_label)

        # Monthly-aggregated figures — Liu et al. (2024) Fig 1/Fig 2 are at MONTHLY
        # pixel scale. Restrict to the verdict's qualifying sites so the daily and
        # monthly views share one site set, then collapse to one row per site-month.
        qualifying = set(effects["site_name"])
        monthly = aggregate_monthly(table[table["site_name"].isin(qualifying)])
        monthly.to_csv(out_dir / "site_month_table.csv", index=False)
        logger.info("Monthly table: %d site-months, %d sites", len(monthly), monthly["site_name"].nunique())
        note = " — MONTHLY aggregated (Liu Fig 1/2 scale)"
        plot_driver_responses(
            monthly,
            fig_dir / "fig1_driver_responses_monthly.png",
            title_note=note,
            resp_label=resp_label,
            metric_name=metric_name,
        )
        plot_decoupling_lines(
            monthly,
            fig_dir / "fig2abc_decoupling_lines_monthly.png",
            n_bins=args.n_bins,
            min_valid_days=args.min_valid_months,
            title_note=note,
            resp_label=resp_label,
        )

        # MEDIAN variants (additive — the mean figures above are kept untouched).
        # ΔSF is a heavy-tailed ratio (bounded −900 %); the median resists that tail,
        # which drags sparse-bin means far negative. Daily keeps its fine bins/deciles
        # (62k samples); monthly uses COARSER bins + quintiles because it has ~23× fewer
        # samples and a compressed driver range, so 0.1 kPa / 10 deciles are over-resolved.
        plot_driver_responses(
            table,
            fig_dir / "fig1_driver_responses_median.png",
            agg="median",
            title_note=" — MEDIAN",
            resp_label=resp_label,
            metric_name=metric_name,
        )
        plot_decoupling_lines(
            table,
            fig_dir / "fig2abc_decoupling_lines_median.png",
            n_bins=args.n_bins,
            min_valid_days=args.min_valid_days,
            agg="median",
            title_note=" — MEDIAN",
            resp_label=resp_label,
        )

        # COARSE-BIN DAILY variants (additive — deciles above kept byte-for-byte).
        # Same daily table, quintiles instead of deciles: at min_valid_days≥120 a 10×10
        # VPD×Tair grid sees ~1 day/cell, so per-site cells are noisy; 5×5 puts ~4× more
        # site-days per cell, filling the grid for sparse and dense-restricted site sets.
        # The MONTHLY-median Fig 2 is already at quintiles (--monthly-n-bins).
        if args.coarse_n_bins and args.coarse_n_bins > 0:
            cnote = f" — {args.coarse_n_bins} bins (quintiles)"
            plot_decoupling_lines(
                table,
                fig_dir / f"fig2abc_decoupling_lines_bins{args.coarse_n_bins}.png",
                n_bins=args.coarse_n_bins,
                min_valid_days=args.min_valid_days,
                title_note=cnote,
                resp_label=resp_label,
            )
            plot_decoupling_lines(
                table,
                fig_dir / f"fig2abc_decoupling_lines_median_bins{args.coarse_n_bins}.png",
                n_bins=args.coarse_n_bins,
                min_valid_days=args.min_valid_days,
                agg="median",
                title_note=f" — MEDIAN,{cnote.split('—', 1)[1]}",
                resp_label=resp_label,
            )

        monthly_widths = {"vpd": 0.2, "tair": 2.0, "sm": 0.04}
        plot_driver_responses(
            monthly,
            fig_dir / "fig1_driver_responses_monthly_median.png",
            bin_widths=monthly_widths,
            min_count=args.monthly_min_count,
            agg="median",
            title_note=" — MONTHLY, MEDIAN, coarse bins (VPD 0.2 kPa, Tair 2 °C)",
            resp_label=resp_label,
            metric_name=metric_name,
        )
        plot_decoupling_lines(
            monthly,
            fig_dir / "fig2abc_decoupling_lines_monthly_median.png",
            n_bins=args.monthly_n_bins,
            min_valid_days=args.min_valid_months,
            agg="median",
            title_note=" — MONTHLY, MEDIAN, quintiles",
            resp_label=resp_label,
        )
    except Exception as e:  # plotting must not sink a completed analysis
        logger.warning("Plotting step failed (%s); CSV outputs are intact.", e)

    report = _verdict(
        effects,
        rf,
        rf_sd_scope=args.rf_sd_scope,
        response=args.response,
        gc_winsor_top_frac=args.gc_winsor_top_frac,
    )
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")
    logger.info("\n%s", report)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scale", default="sapwood", choices=["sapwood", "plant", "site"])
    p.add_argument(
        "--response",
        default="sf",
        choices=["sf", "gc"],
        help="Response variable: 'sf' = sap velocity (default); 'gc' = whole-tree canopy conductance "
        "(Flo et al. 2021 Eqn 2, needs elevation). Gc outputs go to an afternoon_depression/gc/ subdir. "
        "NOTE: Gc∝1/VPD, so the VPD legs of the decoupling are confounded (see the Gc REPORT.md caveat).",
    )
    p.add_argument("--data-dir", default=None, help="Override hourly data dir (else resolved from path_config).")
    p.add_argument(
        "--climate-source",
        default="site",
        choices=["site", "era5"],
        help="'site' = in-situ VPD/Tair (+ ERA5-Land SM); 'era5' = all-ERA5 (needs dewpoint_2m).",
    )
    p.add_argument("--tair-min", type=float, default=5.0, help="Frost-free daily Tair threshold (°C).")
    p.add_argument("--min-daily-sf", type=float, default=0.0, help="Drop days with daily-mean sap flow below this.")
    p.add_argument(
        "--min-am-pm-ratio",
        type=float,
        default=0.10,
        help="Drop days where SF_AM < ratio*SF_PM (bounds ΔSF≥(1-1/ratio)·100%%; Liu S1#3 analog). 0 disables.",
    )
    p.add_argument("--min-window-hours", type=int, default=2, help="Min valid hours per AM/PM window.")
    p.add_argument(
        "--gc-winsor-top-frac",
        type=float,
        default=0.01,
        help="GC ONLY: drop the global top fraction of site-days by Gc magnitude (default 0.01 = top 1%%), "
        "removing the 1/VPD divide-by-near-zero artifact tail above the ~11-order discontinuity. 0 disables. "
        "No effect on --response sf.",
    )
    p.add_argument(
        "--min-valid-days", type=int, default=120, help="Min site-days to include a site (Text S1 #1 analog)."
    )
    p.add_argument("--n-bins", type=int, default=10, help="Percentile bins per driver (deciles).")
    p.add_argument(
        "--coarse-n-bins",
        type=int,
        default=5,
        help="Coarse per-site percentile bins for the additive 5-bin DAILY Fig 2 variants "
        "(quintiles). At dense thresholds the 10-bin grid is thin (~1 day/cell); quintiles put "
        "~4× more site-days per cell. Set 0 to skip the coarse-bin figures.",
    )
    p.add_argument(
        "--min-valid-months",
        type=int,
        default=3,
        help="Monthly figures: min site-months for a site to contribute (per-site percentile stability).",
    )
    p.add_argument(
        "--monthly-n-bins",
        type=int,
        default=5,
        help="Monthly median Fig 2: per-site percentile bins (quintiles — fewer samples than daily).",
    )
    p.add_argument(
        "--monthly-min-count",
        type=int,
        default=10,
        help="Monthly median Fig 1: min observations per fixed-width bin (lower than daily; fewer samples).",
    )
    p.add_argument(
        "--restrict-to-qualifying",
        action="store_true",
        help="Pre-filter the table to sites with ≥ --min-valid-days valid days so Fig 1, the RF "
        "cross-check, and the decoupling all use the SAME dense-site set. Default off = canonical "
        "behaviour (RF/Fig 1 use all sites).",
    )
    p.add_argument("--rf-models", type=int, default=100, help="Number of RF models for sensitivity.")
    p.add_argument(
        "--rf-sd-scope",
        default="site",
        choices=["site", "global"],
        help="RF ±1 SD perturbation size: 'site' = each row by its within-site SD "
        "(per-pixel-consistent with the decoupling); 'global' = one SD over all pooled rows.",
    )
    p.add_argument("--no-rf", action="store_true", help="Skip the RF sensitivity cross-check.")
    p.add_argument("--output-dir", default=None, help="Override output directory.")
    return p


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
