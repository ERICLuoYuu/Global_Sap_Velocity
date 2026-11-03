"""Runs one product: python -m src.static_features.run --product terrain --scale sapwood_combined."""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from src.static_features.results import (
    NEIGHBOUR,
    NO_DATA,
    OK,
    ExtractionFailure,
    ProductResult,
    RunContext,
    code_state,
    validate,
    write_part,
)
from src.static_features.sites import site_list_hash

LOGGER = logging.getLogger(__name__)
DEFAULT_PROJECT = "era5download-447713"

PRODUCTS = {
    "terrain": "src.static_features.products.terrain",
    "canopy": "src.static_features.products.canopy",
    "soil": "src.static_features.products.soil",
    "worldclim": "src.static_features.products.worldclim",
    "stand_age": "src.static_features.products.stand_age",
    "fan": "src.static_features.products.fan",
    "han": "src.static_features.products.han",
    "yang": "src.static_features.products.yang",
    "lusk": "src.static_features.products.lusk",
    "liang": "src.static_features.products.liang",
    "cci": "src.static_features.products.cci",
    "gami": "src.static_features.products.gami",
    "lesiv": "src.static_features.products.lesiv",
    "tree_cover": "src.static_features.products.tree_cover",
    "evi": "src.static_features.products.evi",
    "phenology": "src.static_features.products.phenology",
}


def _log_status_counts(result: ProductResult) -> None:
    """One line per distinct count, so the acceptance numbers can be read from the job log."""
    columns_by_counts: dict[tuple[int, int, int], list[str]] = {}
    for column in result.status.columns:
        counts = result.status[column].value_counts()
        key = (int(counts.get(OK, 0)), int(counts.get(NEIGHBOUR, 0)), int(counts.get(NO_DATA, 0)))
        columns_by_counts.setdefault(key, []).append(column)
    for (ok, neighbour, no_data), columns in sorted(columns_by_counts.items(), reverse=True):
        LOGGER.info("%s: ok=%d neighbour=%d no_data=%d: %s", result.product, ok, neighbour, no_data, ", ".join(columns))


def run_product(name: str, context: RunContext, registry: dict[str, str] = PRODUCTS) -> Path:
    code = code_state()  # before the import: a commit landing during a two-hour request must not claim this part
    module = importlib.import_module(registry[name])
    result = module.extract(context)
    validate(result, list(context.sites["site_name"]))
    stamped = replace(result, provenance={**result.provenance, "code": code})
    path = write_part(stamped, context.parts_root, site_list_hash(context.sites))
    _log_status_counts(result)
    LOGGER.info("%s: wrote %s", name, path)
    return path


def fetch_product(name: str, context: RunContext, registry: dict[str, str] = PRODUCTS) -> None:
    module = importlib.import_module(registry[name])
    fetch_inputs = getattr(module, "fetch_inputs", None)
    if fetch_inputs is None:
        raise ExtractionFailure(f"{name}: nothing to fetch")
    fetch_inputs(context)
    LOGGER.info("%s: inputs are in place", name)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", required=True, choices=sorted(PRODUCTS))
    parser.add_argument("--scale", required=True)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--fetch-only", action="store_true", help="Download and verify inputs; write no part")
    args = parser.parse_args(argv)

    context = RunContext.for_scale(args.scale, args.project)
    try:
        if args.fetch_only:
            fetch_product(args.product, context, registry=PRODUCTS)
        else:
            run_product(args.product, context, registry=PRODUCTS)
    except ExtractionFailure as failure:
        LOGGER.error("%s", failure)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
