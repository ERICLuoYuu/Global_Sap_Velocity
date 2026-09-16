"""CLI entry point for transpiration comparison pipeline.

Usage:
    python -m src.transpiration_comparison.cli download --products gleam era5land
    python -m src.transpiration_comparison.cli preprocess --products all
    python -m src.transpiration_comparison.cli compare --phase spatial
    python -m src.transpiration_comparison.cli visualize
    python -m src.transpiration_comparison.cli run-all

A servable model lives in the serving store, not the stage-2 parquet dump, and
wants its own cache and figure directories:

    python -m src.transpiration_comparison.cli \\
        --sap-source serving --run-id sweep5_fullyear_fullday_tx_ffs_grid005_hohfix_sv_smzp \\
        --period-start 2016-01-01 --period-end 2018-12-31 \\
        --out-tag servable_smzp run-all
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import DEFAULT_DOMAIN, DEFAULT_PERIOD, PRODUCTS, REGIONS, Paths, TimePeriod
from .pipeline import TranspirationComparisonPipeline
from .preprocessing.alignment import ALIGNMENT_METHODS

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transpiration Product Comparison Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Base directory every other path derives from (default: /scratch/tmp/yluo2/gsv)",
    )

    # --- which sap velocity ---
    parser.add_argument(
        "--sap-source",
        choices=["parquet", "serving"],
        default="parquet",
        help="'parquet': the legacy stage-2 dump. 'serving': the per-year NetCDF "
        "cube under <base>/outputs/serving/store_v1/predictions/<run-id>/ "
        "(needs --run-id).",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Prediction run whose cube to read, e.g. sweep5_fullyear_fullday_tx_ffs_grid005_hohfix_sv_smzp",
    )

    # --- period ---
    parser.add_argument(
        "--period-start",
        default=DEFAULT_PERIOD.start,
        help=f"First day of the comparison period (default {DEFAULT_PERIOD.start})",
    )
    parser.add_argument(
        "--period-end",
        default=DEFAULT_PERIOD.end,
        help=f"Last day of the comparison period (default {DEFAULT_PERIOD.end})",
    )

    # --- where the results go ---
    parser.add_argument(
        "--out-tag",
        default="",
        help="Subdirectory for this run's cache, figures and reports. Without it a second run overwrites the first.",
    )

    # --- how products are put on one grid ---
    parser.add_argument(
        "--alignment",
        choices=list(ALIGNMENT_METHODS),
        default="nearest",
        help="'nearest' reproduces the published run (a product on x.05 centres "
        "is snapped a half cell onto the x.0 prediction grid). 'linear' "
        "averages the straddling cells instead, at the cost of one NaN cell "
        "inward from every data edge.",
    )

    parser.add_argument(
        "--allow-missing",
        default="",
        metavar="P1,P2",
        help="Comma-separated products that may fail without failing the run, "
        "e.g. --allow-missing gldas. (Comma-separated, not space-separated: a "
        "greedy nargs='+' here would swallow the subcommand that follows it.)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    dl = subparsers.add_parser("download", help="Download transpiration products")
    dl.add_argument(
        "--products",
        nargs="+",
        default=["gleam", "era5land", "gldas"],
        choices=list(PRODUCTS.keys()),
        help="Products to download",
    )

    pp = subparsers.add_parser("preprocess", help="Regrid, aggregate, normalize")
    pp.add_argument(
        "--products",
        nargs="+",
        default=["all"],
        help="Products to preprocess ('all' for everything)",
    )

    cmp_parser = subparsers.add_parser("compare", help="Run comparison analysis")
    cmp_parser.add_argument(
        "--phase",
        required=True,
        choices=["spatial", "temporal", "regional", "agreement", "collocation"],
        help="Comparison phase to run",
    )
    cmp_parser.add_argument("--region", default=None, choices=list(REGIONS.keys()))

    subparsers.add_parser("visualize", help="Generate all figures")
    subparsers.add_parser("run-all", help="Run full pipeline end-to-end")

    return parser


def build_pipeline(args) -> TranspirationComparisonPipeline:
    """Assemble the pipeline from parsed arguments.

    Separated from main() so the wiring can be tested without running a phase.
    """
    if args.sap_source == "serving" and not args.run_id:
        raise SystemExit("--sap-source serving needs --run-id naming the prediction to read")

    paths = Paths(
        base=args.base_dir if args.base_dir else Paths().base,
        run_tag=args.out_tag,
    )
    paths.ensure_dirs()

    return TranspirationComparisonPipeline(
        paths=paths,
        period=TimePeriod(start=args.period_start, end=args.period_end),
        domain=DEFAULT_DOMAIN,
        sap_source=args.sap_source,
        run_id=args.run_id,
        alignment=args.alignment,
        allow_missing=[p.strip() for p in args.allow_missing.split(",") if p.strip()],
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    pipeline = build_pipeline(args)
    logger.info(
        f"sap source: {args.sap_source}"
        + (f" (run {args.run_id})" if args.run_id else "")
        + f" | period {args.period_start} .. {args.period_end}"
        + f" | alignment {args.alignment}"
        + f" | cache {pipeline.paths.preprocessed_dir}"
    )

    try:
        if args.command == "download":
            pipeline.download_products(args.products)
        elif args.command == "preprocess":
            products = list(PRODUCTS.keys()) if "all" in args.products else args.products
            pipeline.preprocess_products(products)
        elif args.command == "compare":
            pipeline.run_comparison(args.phase, region=args.region)
        elif args.command == "visualize":
            pipeline.generate_figures()
        elif args.command == "run-all":
            pipeline.run_all()
        else:
            parser.print_help()
            return 1
    except Exception:
        logger.exception("Pipeline failed")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
