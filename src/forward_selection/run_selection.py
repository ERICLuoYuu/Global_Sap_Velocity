"""CLI entry point for forward feature selection.

Usage:
    # Build feature cache (run once)
    python -m src.forward_selection.run_selection --build_cache \
        --output_dir /scratch/tmp/yluo2/gsv/outputs/forward_selection/

    # Run selection with a specific scoring metric
    python -m src.forward_selection.run_selection \
        --scoring mean_r2 \
        --cache_path /scratch/tmp/yluo2/gsv/outputs/forward_selection/feature_cache.npz \
        --output_dir /scratch/tmp/yluo2/gsv/outputs/forward_selection/ \
        --n_jobs_sfs 32 --n_jobs_xgb 4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.feature_contract.definitions import FEATURE_CONTRACTS, SERVABLE_CONTRACT  # noqa: E402
from src.forward_selection.config import (  # noqa: E402
    HyperparamConfig,
    ScoringMode,
    SelectionConfig,
)
from src.forward_selection.diagnostics import (  # noqa: E402
    export_selected_features,
    plot_feature_ranking,
    plot_performance_curve,
)

logger = logging.getLogger(__name__)


def _setup_logging(output_dir: Path) -> None:
    """Configure logging to file + console."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "forward_selection.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Forward feature selection for sap velocity prediction",
    )
    parser.add_argument(
        "--build_cache",
        action="store_true",
        help="Build feature cache from site CSVs (no selection).",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        choices=[m.value for m in ScoringMode],
        help="Scoring metric for selection.",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        help="Path to the .npz feature cache.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/forward_selection",
        help="Directory for outputs (cache, results, plots).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Override data directory (parent of daily/ or hourly/ subdirectory).",
    )
    parser.add_argument(
        "--time_scale",
        type=str,
        default="daily",
        choices=["daily", "hourly"],
    )
    parser.add_argument(
        "--n_jobs_sfs",
        type=int,
        default=32,
        help="Parallel candidates in mlxtend SFS.",
    )
    parser.add_argument(
        "--n_jobs_xgb",
        type=int,
        default=4,
        help="Threads per XGBoost fit.",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=10,
        help="Number of CV folds.",
    )
    parser.add_argument(
        "--k_features",
        type=str,
        default="best",
        help="Number of features to stop at, 'min,max' range, 'best', or 'parsimonious'.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--spatial_method",
        type=str,
        default="default",
        choices=["default", "grid"],
        help="Spatial grouping method: 'default' (each site = own group) or 'grid'.",
    )
    parser.add_argument(
        "--grid_size",
        type=float,
        default=0.05,
        help="Grid cell size in degrees (only used when --spatial_method=grid).",
    )
    parser.add_argument(
        "--feature_contract",
        type=str,
        default=SERVABLE_CONTRACT,
        choices=list(FEATURE_CONTRACTS),
        help="Feature contract the cache is built under (--build_cache only; the cache records it). "
        "servable (default): definitions a 0.1 deg cell can reproduce, which the Stage-4 merge "
        "writes by default. site: legacy definitions -- per-site soil-moisture statistics and gauge "
        "precip -- for a tree merged with --feature-contract site.",
    )
    return parser.parse_args()


def _parse_k_features(raw: str):
    """Turn the --k_features string into what mlxtend actually accepts.

    argparse hands us a string so that "best" and "parsimonious" work, but
    mlxtend rejects every OTHER string outright -- passing "40" raises
    AttributeError several minutes into the run, after the cache has loaded and
    the CV splits are built. Coerce here, at the boundary, so a numeric cap
    reaches the selector as an int.
    """
    text = str(raw).strip()
    if text.isdigit():
        return int(text)
    if "," in text:
        low, high = (part.strip() for part in text.split(",", 1))
        if low.isdigit() and high.isdigit():
            return (int(low), int(high))
    return text


def _cmd_build_cache(args: argparse.Namespace) -> None:
    """Build the feature cache from site CSVs."""
    from src.forward_selection.data_loader import load_and_cache_features

    output_dir = Path(args.output_dir)

    if args.data_dir:
        data_dir = Path(args.data_dir) / args.time_scale
    else:
        # Default: use PathConfig
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".venv"))
        from path_config import get_default_paths

        paths = get_default_paths()
        data_dir = paths.merged_data_root / args.time_scale

    cache_path = output_dir / "feature_cache.npz"
    logger.info("Building cache from %s -> %s (feature contract: %s)", data_dir, cache_path, args.feature_contract)

    load_and_cache_features(
        data_dir=data_dir,
        cache_path=cache_path,
        time_scale=args.time_scale,
        spatial_method=args.spatial_method,
        grid_size=args.grid_size,
        contract=args.feature_contract,
    )
    logger.info("Cache build complete.")


def _cmd_run_selection(args: argparse.Namespace) -> None:
    """Run forward feature selection."""
    from src.forward_selection.selector import run_forward_selection

    if not args.scoring:
        logger.error("--scoring is required for selection mode")
        sys.exit(1)
    if not args.cache_path:
        logger.error("--cache_path is required for selection mode")
        sys.exit(1)

    config = SelectionConfig(
        scoring=ScoringMode(args.scoring),
        n_splits=args.n_splits,
        random_seed=args.random_seed,
        n_jobs_sfs=args.n_jobs_sfs,
        n_jobs_xgb=args.n_jobs_xgb,
        k_features=_parse_k_features(args.k_features),
        time_scale=args.time_scale,
        output_dir=Path(args.output_dir),
        cache_dir=Path(args.output_dir),
    )
    hyper = HyperparamConfig()

    results = run_forward_selection(
        config=config,
        hyper=hyper,
        cache_path=Path(args.cache_path),
    )

    # Generate diagnostics
    output_dir = Path(args.output_dir)
    scoring_name = args.scoring

    plot_performance_curve(
        results,
        output_path=output_dir / f"ffs_{scoring_name}_curve.png",
    )
    plot_feature_ranking(
        results,
        output_path=output_dir / f"ffs_{scoring_name}_ranking.png",
    )
    export_selected_features(
        results,
        output_path=output_dir / f"ffs_{scoring_name}_selected.json",
    )
    logger.info("All outputs saved to %s", output_dir)


def main() -> None:
    """Entry point."""
    args = _parse_args()
    _setup_logging(Path(args.output_dir))

    if args.build_cache:
        _cmd_build_cache(args)
    else:
        _cmd_run_selection(args)


if __name__ == "__main__":
    main()
