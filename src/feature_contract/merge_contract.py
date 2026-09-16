"""The servable feature contract, applied by the Stage-4 merge itself.

A servable tree used to take three passes: the merge normalised soil moisture by each site's own
moments and wrote the gauge's ``precip``; ``rebuild_daily`` copied the whole tree and spliced the
soil-moisture families to the climatology; and ``feature_engineering`` replaced ``precip`` with the
ERA5-Land value at training and serving time. The merge already holds everything the other two
passes need, so it now applies the contract directly and this module holds the decisions.

Two transforms define the contract, and each has exactly one implementation in this repository,
imported here rather than restated::

    soil moisture   normalise_swvl   bare = raw / sigma_clim, zscore = (raw - mu_clim) / sigma_clim
    precipitation   era5_hourly_mean_precip_mm_per_day    mean of hourly metres x 24 x 1000

The soil-moisture moments are handed to ``add_swc_representations`` on the *hourly* frame, before
the daily statistics are taken. The transform is affine with a positive sigma, so it commutes with
the mean, the minimum and the maximum, and distributes over the sum: the daily ``_zscore_sum``
comes out as ``(sum(raw) - n*mu)/sigma`` with the true hour count. ``rebuild_daily`` worked after
aggregation and had to recover ``n`` from ``raw_sum / raw_mean``, which is undefined on a day whose
mean is zero.

``rebuild_daily`` remains the only path for trees built before this change, and refuses a tree that
already carries a ``feature_contract`` block, so the two can never normalise the same values twice.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .definitions import (
    CELLS_FILE,
    ERA5_HOURLY_MEAN_PRECIP_COL,
    PROVENANCE_FILE,
    PROVENANCE_KEY,
    SERVABLE_CONTRACT,
    SERVABLE_RAIN_EVENT_MM_PER_DAY,
    SIDECAR_FILE,
    SWC_LAYERS,
    SWC_LEVEL_SUFFIX,
    era5_hourly_mean_precip_mm_per_day,
    swc_level_col,
)
from .swvl_climatology import DEFAULT_MAX_RADIUS_CELLS, MIN_VALID_SIGMA, load_climatology, lookup_cell

#: The climatology the servable contract normalises soil moisture against. Resolved from this file
#: so the merge finds it whatever the working directory is.
DEFAULT_SWVL_CUBE = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "derived"
    / "swvl_sigma_climatology"
    / "swvl_sigma_climatology_v2_1995_2019.nc"
)

#: ``SIDECAR_FILE`` and ``CELLS_FILE`` are defined in :mod:`src.feature_contract.definitions` and
#: re-exported here, so the rebuild guard and the dataset gate can name the same files without
#: importing this module (which pulls in xarray through the climatology).

#: The cells record, field for field as ``rebuild_daily`` writes it, so both trees read the same.
CELL_FIELDS = (
    "site",
    "site_lat",
    "site_lon",
    "layer",
    "sigma",
    "mean",
    "cell_lat",
    "cell_lon",
    "distance_km",
    "fallback",
)

#: The gauge keeps its daily totals under these names once ``precip`` belongs to ERA5-Land.
GAUGE_INFIX = "_gauge"
GAUGE_SUFFIXES = ("", "_min", "_max", "_sum", "_coverage")

#: Daily statistics of the gauge that have no servable meaning and must not survive under one.
STALE_GAUGE_STATS = ("_min", "_max", "_sum")

SWVL_DEFINITION = {"bare": "raw / sigma_clim", "zscore": "(raw - mu_clim) / sigma_clim"}


class FeatureContractError(RuntimeError):
    """A servable-contract violation.

    The merge's site loop ends in ``except Exception: continue``, which turns a failure into a
    silently missing site. Contract failures are raised as this type and let through, so a tree is
    never quietly short of the sites it claims.
    """


def _swc_column(layer: int) -> str:
    """The bare soil-moisture column of ``layer``, derived from the one definition of the name."""
    return swc_level_col(layer)[: -len(SWC_LEVEL_SUFFIX)]


def load_cube(path=DEFAULT_SWVL_CUBE):
    """The soil-moisture climatology at ``path``.

    Raises rather than returning nothing: a tree labelled servable but normalised by something else
    is indistinguishable from a correct one once it is written.
    """
    cube = Path(path)
    if not cube.is_file():
        raise FeatureContractError(
            f"the servable contract needs the soil-moisture climatology, and {cube} is not a file"
        )
    try:
        return load_climatology(cube)
    except Exception as exc:  # noqa: BLE001 - any read failure is the same refusal
        raise FeatureContractError(f"cannot read the climatology cube {cube}: {exc}") from exc


def resolve_site_cells(clim, site, lat, lon, layers=SWC_LAYERS, max_radius_cells=DEFAULT_MAX_RADIUS_CELLS):
    """``{layer: CellStats}`` for one site: the nearest cell with a usable sigma, per layer."""
    cells = {}
    for layer in layers:
        try:
            cells[layer] = lookup_cell(clim, float(lat), float(lon), layer, max_radius_cells)
        except LookupError as exc:
            raise FeatureContractError(f"{site}: no climatology cell for layer {layer}: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise FeatureContractError(f"{site}: cannot place ({lat}, {lon}) on the climatology grid: {exc}") from exc
    return cells


def preflight_cells(clim, sites, layers=SWC_LAYERS, max_radius_cells=DEFAULT_MAX_RADIUS_CELLS):
    """``{site: {layer: CellStats}}`` for every ``(site, lat, lon)``, resolved before the merge loop.

    Every lookup happens here, outside the loop, because the loop catches everything: a site that
    failed inside it would leave the corpus with one printed line and no other trace.
    """
    return {site: resolve_site_cells(clim, site, lat, lon, layers, max_radius_cells) for site, lat, lon in sites}


def swc_climatology_moments(cells):
    """``{column: (mu, sigma)}`` -- the mapping ``add_swc_representations`` takes, from the cube."""
    return {_swc_column(layer): (cell.mean, cell.sigma) for layer, cell in cells.items()}


def cell_rows(site, lat, lon, cells):
    """One record per layer of which cell the site was normalised against."""
    return [
        {"site": site, "site_lat": float(lat), "site_lon": float(lon), **asdict(cells[layer])}
        for layer in sorted(cells)
    ]


def gauge_column_names(col):
    """The gauge's daily statistics renamed out of the way of the servable column."""
    return {f"{col}{suffix}": f"{col}{GAUGE_INFIX}{suffix}" for suffix in GAUGE_SUFFIXES}


def rename_gauge_columns(frame, col):
    """Return a copy of ``frame`` holding the gauge statistics of ``col`` under their gauge names."""
    return frame.rename(columns=gauge_column_names(col))


def require_era5_precip(era5_data, path):
    """Refuse an ERA5-Land extract that cannot supply servable precipitation.

    Checked once against the whole extract, not per site: a site with no ERA5 rows still receives the
    column (``complete_day_statistics`` returns an empty frame carrying the full column set, and
    ``overwrite_daily_columns`` writes every one of them), so the only way the column can be missing
    is that the extract never had it -- which is true of every site at once, and is worth saying
    before a single site is processed rather than after the first one is finished.
    """
    if ERA5_HOURLY_MEAN_PRECIP_COL not in era5_data.columns:
        raise FeatureContractError(
            f"the servable contract writes precip from {ERA5_HOURLY_MEAN_PRECIP_COL!r}, and the "
            f"ERA5-Land extract {path} does not carry that column"
        )


def apply_servable_precip(daily, col="precip"):
    """Return a copy of ``daily`` whose ``col`` is ERA5-Land precipitation in mm/day.

    A 0.1 degree prediction cell has no rain gauge, so the gauge cannot be the source. Its daily
    minimum, maximum and total describe a different quantity and are dropped rather than left under
    a servable name; the gauge's own totals keep the ``_gauge`` names.
    """
    if ERA5_HOURLY_MEAN_PRECIP_COL not in daily.columns:
        raise FeatureContractError(
            f"the servable contract writes {col!r} from {ERA5_HOURLY_MEAN_PRECIP_COL!r}, which this "
            "frame does not carry: the ERA5-Land extract did not reach this site"
        )
    stale = [f"{col}{suffix}" for suffix in STALE_GAUGE_STATS if f"{col}{suffix}" in daily.columns]
    out = daily.drop(columns=stale)
    return out.assign(**{col: era5_hourly_mean_precip_mm_per_day(out[ERA5_HOURLY_MEAN_PRECIP_COL])})


def provenance_contract(config_dir):
    """The contract ``RUN_PROVENANCE.json`` in ``config_dir`` claims, or ``None`` if it claims none.

    ``None`` covers the directory with no provenance and, just as importantly, the one carrying a
    provenance file with no ``feature_contract`` block -- which is what ``job_merge_sweep5.sh``
    writes after every merge, and every tree between the merge and the fold that adds the block.
    """
    path = Path(config_dir) / PROVENANCE_FILE
    if not path.is_file():
        return None
    try:
        with open(path) as handle:
            provenance = json.load(handle)
    except (OSError, ValueError) as exc:
        raise FeatureContractError(
            f"{path} cannot be read ({exc}), and the dataset gate reads it to decide which contract "
            "built the tree in this directory. Repair or remove it before merging here."
        ) from exc
    block = provenance.get(PROVENANCE_KEY) if isinstance(provenance, dict) else None
    return block.get("name") if isinstance(block, dict) else None


def require_provenance_agrees(config_dir, contract):
    """Refuse to merge into a directory whose provenance names the OTHER contract.

    ``assert_dataset_contract`` reads ``RUN_PROVENANCE.json`` BEFORE the badge beside it, so a block
    left by an earlier run speaks for whatever is written here next, whatever the badge says --
    clearing the badge is not enough on its own. The merge refuses instead of editing the file,
    because the sweep owns it and it records a run that did happen; and it refuses here, before any
    work, rather than letting the disagreement surface at training time hours later.
    """
    claimed = provenance_contract(config_dir)
    if claimed is None or claimed == contract:
        return
    raise FeatureContractError(
        f"{Path(config_dir) / PROVENANCE_FILE} records this directory as built under the {claimed!r} "
        f"contract, and this merge builds {contract!r}. The dataset gate reads that block before the "
        f"badge beside it, so whatever is written here would be read as {claimed!r}. Merge into a "
        f"different --output-dir, or remove the stale {PROVENANCE_KEY!r} block from that file."
    )


def clear_sidecar(config_dir):
    """Remove any contract badge from ``config_dir``; return the paths that were removed.

    A merge under the site contract must leave nothing behind claiming the tree is servable. The
    badge is authoritative for the dataset gate, so one left over from an earlier servable run into
    the same directory would present per-site z-scores and gauge precip as the climatology contract.
    """
    directory = Path(config_dir)
    removed = []
    for name in (SIDECAR_FILE, CELLS_FILE):
        stale = directory / name
        if stale.exists():
            stale.unlink()
            removed.append(stale)
    return removed


def contract_block(clim, layers, max_radius_cells, rows, git_sha, built_by="merge", complete=True):
    """The ``feature_contract`` block of ``RUN_PROVENANCE.json`` for a tree the merge built.

    ``rebuild_daily`` builds its own, describing a text splice of an existing tree. The two agree on
    everything a gate reads -- the contract name, the normalisation, the fallback count and the
    precipitation source -- which ``test_merge_contract`` asserts directly.
    """
    return {
        "name": SERVABLE_CONTRACT,
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "built_by": built_by,
        # False on the badge written before the loop, which names every site the cube placed; True on
        # the one written after it, which names the sites the merge actually kept. A tree left with
        # complete=False was interrupted: its numbers are servable, its site list over-claims.
        "complete": complete,
        "swvl": {
            "cube": clim.path,
            "cube_sha256": clim.sha256,
            "cube_attrs": clim.attrs,
            "layers": list(layers),
            "definition": dict(SWVL_DEFINITION),
            "clip": None,
            "min_valid_sigma": MIN_VALID_SIGMA,
            "max_radius_cells": max_radius_cells,
            "n_fallback_cells": int(sum(1 for row in rows if row["fallback"])),
            "n_sites": len({row["site"] for row in rows}),
            "applied": "to the hourly values, before the daily statistics, so every one of them follows",
        },
        "precip_memory": {
            "source_column": ERA5_HOURLY_MEAN_PRECIP_COL,
            "units": "mm per day",
            "rain_event_mm_per_day": SERVABLE_RAIN_EVENT_MM_PER_DAY,
            "where": (
                "the merge writes precip; feature_engineering recomputes the identical value under "
                "--FEATURE_CONTRACT servable, and serves the grid the same way"
            ),
        },
    }


def write_sidecar(config_dir, block, rows):
    """Write the contract block and the cells record beside ``daily/``; return both paths.

    ``job_merge_sweep5.sh`` reads the block into ``RUN_PROVENANCE.json``, which is what
    ``assert_dataset_contract`` and every shell gate look at.
    """
    directory = Path(config_dir)
    if not rows:
        raise FeatureContractError(
            "a servable tree records the climatology cell every site was normalised against, and no "
            "site was recorded: the merge retained nothing, so the tree must not be labelled servable"
        )
    wrong = next((row for row in rows if set(row) != set(CELL_FIELDS)), None)
    if wrong is not None:
        raise FeatureContractError(f"a cells record must carry exactly {CELL_FIELDS}, got {sorted(wrong)}")
    sidecar, cells = directory / SIDECAR_FILE, directory / CELLS_FILE
    with open(sidecar, "w") as handle:
        json.dump(block, handle, indent=2)
    pd.DataFrame(rows, columns=list(CELL_FIELDS)).to_csv(cells, index=False)
    return sidecar, cells
