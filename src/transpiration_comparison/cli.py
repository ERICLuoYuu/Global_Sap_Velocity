from __future__ import annotations

"""CLI entry point for transpiration comparison pipeline.

Usage:
    python -m src.transpiration_comparison.cli download --products gleam era5land gldas
    python -m src.transpiration_comparison.cli preprocess --products all
    python -m src.transpiration_comparison.cli compare --phase spatial
    python -m src.transpiration_comparison.cli compare --phase temporal
    python -m src.transpiration_comparison.cli compare --phase regional --region amazon
    python -m src.transpiration_comparison.cli compare --phase agreement
    python -m src.transpiration_comparison.cli visualize
    python -m src.transpiration_comparison.cli run-all
"""

import argparse
import logging
import sys
from pathlib import Path

from .config import DEFAULT_DOMAIN, DEFAULT_PATHS, DEFAULT_PERIOD, PRODUCTS, REGIONS, Paths
from .pipeline import TranspirationComparisonPipeline

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Transpiration Product Comparison Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Override base directory (default: /scratch/tmp/yluo2/gsv)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- download ---
    dl = subparsers.add_parser("download", help="Download transpiration products")
    dl.add_argument(
        "--products",
        nargs="+",
        default=["gleam", "era5land", "gldas"],
        choices=list(PRODUCTS.keys()),
        help="Products to download",
    )

    # --- preprocess ---
    pp = subparsers.add_parser("preprocess", help="Regrid, aggregate, normalize")
    pp.add_argument(
        "--products",
        nargs="+",
        default=["all"],
        help="Products to preprocess ('all' for everything)",
    )

    # --- compare ---
    cmp = subparsers.add_parser("compare", help="Run comparison analysis")
    cmp.add_argument(
        "--phase",
        required=True,
        choices=["spatial", "temporal", "regional", "agreement", "collocation"],
        help="Comparison phase to run",
    )
    cmp.add_argument("--region", default=None, choices=list(REGIONS.keys()))

    # --- visualize ---
    subparsers.add_parser("visualize", help="Generate all figures")

    # --- run-all ---
    subparsers.add_parser("run-all", help="Run full pipeline end-to-end")

    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    # Build paths
    paths = DEFAULT_PATHS
    if args.base_dir:
        paths = Paths(base=args.base_dir)
    paths.ensure_dirs()

    pipeline = TranspirationComparisonPipeline(
        paths=paths,
        period=DEFAULT_PERIOD,
        domain=DEFAULT_DOMAIN,
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
