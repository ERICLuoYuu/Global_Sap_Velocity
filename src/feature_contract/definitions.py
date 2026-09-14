"""The two feature contracts, and the gate that ties a run to a dataset built under its own.

``site``      legacy. Several features are statistics of each site's OWN record, and
              precipitation memory reads the site gauge:

              * bare ``volumetric_soil_water_layer_L`` = raw / std(site's hourly record)
                (the merge; a 13-day record gives values five times a 3-year one)
              * ``swc_layer{2,3,4}_norm`` = z-score over the site's record
              * ``soil_moisture_rel`` / ``soil_dry`` = the site's 5th-95th percentiles
              * ``precip_sum_*`` / ``*_since_rain`` from gauge ``precip``

              None of these exists for a 0.1 deg prediction cell.

``servable``  each of them is replaced by a definition a grid cell can reproduce:
              soil moisture normalised by the per-cell 1995-2018 ERA5-Land climatology
              (written into the tree by :mod:`src.feature_contract.rebuild_daily`), and
              precipitation memory from ERA5-Land daily precipitation. The two
              site-quantile indicators have no such equivalent and are not emitted.

              Under ``servable`` the soil-moisture SCALE is a further choice (``--SWC_SCALE``):

              * ``ratio``   level columns raw / sigma_clim, ``swc_layer*_norm`` the climatology
                            z-score (the legacy split, with climatology statistics)
              * ``zscore``  every soil-moisture feature from (raw - mu_clim) / sigma_clim
              * ``raw``     every soil-moisture feature from raw m3/m3

Free of xarray so the trainer can run the gate before loading anything heavy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

FEATURE_CONTRACTS = ("site", "servable")
DEFAULT_CONTRACT = "site"
SERVABLE_CONTRACT = "servable"

PROVENANCE_FILE = "RUN_PROVENANCE.json"
PROVENANCE_KEY = "feature_contract"

#: ``precip`` at daily scale is the daily MEAN of hourly gauge mm (``precip_sum / precip == 24``
#: in the merged tree), so this threshold means more than ~12 mm/day. Kept exactly as it was.
SITE_RAIN_EVENT_THRESHOLD = 0.5

#: ERA5-Land hardly ever reports exactly zero precipitation, so an event needs a floor.
#: 1 mm/day follows the soil-moisture dry-down analysis already in this repository.
SERVABLE_RAIN_EVENT_MM_PER_DAY = 1.0

#: Daily sum of ERA5-Land hourly precipitation at the site, in metres (a merge output column).
ERA5_DAILY_PRECIP_COL = "total_precipitation_hourly_sum"
M_TO_MM = 1000.0

SWC_SCALES = ("ratio", "zscore", "raw")
DEFAULT_SWC_SCALE = "ratio"
#: The column family each non-default scale reads its values from.
SWC_SCALE_COMPANION = {"zscore": "_zscore", "raw": "_raw"}
SWC_LEVEL_TEMPLATE = "volumetric_soil_water_layer_{layer}"
SWC_LAYERS = (1, 2, 3, 4)
SWC_STAT_SUFFIXES = ("", "_min", "_max", "_sum")


def validate_contract(contract: str) -> str:
    """Return ``contract`` if it is a known feature contract, else raise ``ValueError``."""
    if contract not in FEATURE_CONTRACTS:
        raise ValueError(f"unknown feature contract {contract!r}; expected one of {FEATURE_CONTRACTS}")
    return contract


def validate_swc_scale(swc_scale: str, contract: str) -> str:
    """Return ``swc_scale`` if it is known and allowed under ``contract``, else raise ``ValueError``.

    Only ``ratio`` exists under ``site``: there the ``_zscore`` columns are per-SITE z-scores, and
    the point of the other scales is a definition a grid cell can reproduce.
    """
    if swc_scale not in SWC_SCALES:
        raise ValueError(f"unknown swc scale {swc_scale!r}; expected one of {SWC_SCALES}")
    if swc_scale != DEFAULT_SWC_SCALE and contract != SERVABLE_CONTRACT:
        raise ValueError(f"swc scale {swc_scale!r} is defined for feature contract 'servable' only, not {contract!r}")
    return swc_scale


def rescale_swc_levels(df: pd.DataFrame, swc_scale: str) -> pd.DataFrame:
    """Put the soil-moisture level columns on the scale ``swc_scale`` selects.

    In a servable tree ``volumetric_soil_water_layer_L{,_min,_max,_sum}`` hold raw / sigma_clim.
    ``zscore`` replaces each with its ``_zscore`` companion and ``raw`` with its ``_raw``
    companion, so every feature later built from a level column (lags, interactions) inherits
    that scale. ``ratio`` returns ``df`` itself. Companion columns are never changed; any other
    scale returns a new frame.

    Raises ``KeyError`` when a level column exists without its companion: keeping it would leave
    that one feature on a different scale from the rest, and nothing downstream would notice.
    """
    if swc_scale not in SWC_SCALES:
        raise ValueError(f"unknown swc scale {swc_scale!r}; expected one of {SWC_SCALES}")
    if swc_scale == DEFAULT_SWC_SCALE:
        return df
    companion = SWC_SCALE_COMPANION[swc_scale]
    replacements = {}
    for layer in SWC_LAYERS:
        base = SWC_LEVEL_TEMPLATE.format(layer=layer)
        for suffix in SWC_STAT_SUFFIXES:
            level = f"{base}{suffix}"
            if level not in df.columns:
                continue
            source = f"{base}{companion}{suffix}"
            if source not in df.columns:
                raise KeyError(f"swc scale {swc_scale!r}: {level!r} has no {source!r} column to take its values from")
            replacements[level] = df[source]
    return df.assign(**replacements)


def era5_precip_mm_per_day(precip_m: pd.Series) -> pd.Series:
    """ERA5-Land daily precipitation in metres -> mm/day, with tiny negative reanalysis values set to 0."""
    return (precip_m * M_TO_MM).clip(lower=0.0)


def assert_dataset_contract(config_dir, contract: str) -> None:
    """Refuse a dataset that was not built under ``contract``.

    ``config_dir`` is the merged config directory (the parent of ``daily/``).

    * ``servable`` needs ``RUN_PROVENANCE.json`` with a ``feature_contract`` block naming it.
      A legacy tree also carries ``*_zscore`` columns, but they are per-SITE z-scores; reading
      them as climatological would train a model that looks servable and is not.
    * ``site`` accepts legacy trees (with or without provenance) and refuses a servable tree,
      whose bare soil-moisture columns are no longer the per-site normalisation.
    """
    contract = validate_contract(contract)
    path = Path(config_dir) / PROVENANCE_FILE
    block = None
    if path.exists():
        with open(path) as fh:
            block = json.load(fh).get(PROVENANCE_KEY)
    built_as = block.get("name") if isinstance(block, dict) else None

    if contract == SERVABLE_CONTRACT:
        if not path.exists():
            raise ValueError(
                f"feature contract 'servable' needs {PROVENANCE_FILE} under {config_dir}, and there is none. "
                "Build the tree with `python -m src.feature_contract.rebuild_daily`."
            )
        if built_as != SERVABLE_CONTRACT:
            raise ValueError(
                f"feature contract 'servable' needs a '{PROVENANCE_KEY}' block naming 'servable' in {path}; "
                f"found {block!r}"
            )
        return

    if built_as is not None and built_as != contract:
        raise ValueError(
            f"feature contract {contract!r} cannot read a tree built as {built_as!r} ({path}); "
            f"pass --FEATURE_CONTRACT {built_as}"
        )
