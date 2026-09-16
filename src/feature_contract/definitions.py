"""The two feature contracts, and the gate that ties a run to a dataset built under its own.

``site``      legacy. Some features are statistics of each site's OWN record, and
              precipitation reads the site gauge:

              * ``volumetric_soil_water_layer_L_zscore`` = z-score over the site's hourly
                record (the merge)
              * ``soil_moisture_rel`` / ``soil_dry`` = the site's 5th-95th percentiles
              * ``precip`` = the gauge's daily MEAN of hourly mm, and ``precip_lag1d``
              * ``precip_sum_*`` / ``*_since_rain`` from gauge ``precip``

              None of these exists for a 0.1 deg prediction cell.

``servable``  each of them is replaced by a definition a grid cell can reproduce:
              the soil-moisture anomaly over the per-cell 1995-2018 ERA5-Land climatology
              (written by the Stage-4 merge under ``--feature-contract servable``; an older tree is
              converted by :mod:`src.feature_contract.rebuild_daily` instead), and
              ERA5-Land daily precipitation in mm/day (the daily mean of the hourly values
              x 24, see :data:`ERA5_HOURLY_MEAN_PRECIP_COL`), the unit the prediction pipeline
              serves, as both ``precip`` and the source of precipitation memory. The two
              site-quantile indicators have no such equivalent and are not emitted.

Soil moisture has ONE definition per quantity per layer L, under both contracts:

* ``volumetric_soil_water_layer_L_raw``     level, m3/m3. The only input of every derived
  soil-moisture feature: lags, day-to-day change, running anomaly, interactions, rew, psi_soil.
* ``volumetric_soil_water_layer_L_zscore``  anomaly, standardised as the contract says above.

Retired on 2026-09-14 (:data:`RETIRED_SWC_FEATURES`): the bare column, raw / sigma, which is
neither a level nor an anomaly, and ``swc_layer{2,3,4}_norm``, a second z-score of the same
layer. The merge still writes the bare column; no feature may read it.

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

#: The Stage-4 merge writes the contract block beside ``daily/`` under this name, and the cells it
#: normalised against under the next. Their presence is the record that a tree was built servable in
#: one pass -- and it is what tells :mod:`src.feature_contract.rebuild_daily` to keep its hands off,
#: since normalising an already-normalised tree cannot be detected from the values afterwards.
SIDECAR_FILE = "feature_contract.json"
CELLS_FILE = "feature_contract_cells.csv"

#: ``precip`` at daily scale is the daily MEAN of hourly gauge mm (``precip_sum / precip == 24``
#: in the merged tree), so this threshold means more than ~12 mm/day. Kept exactly as it was.
SITE_RAIN_EVENT_THRESHOLD = 0.5

#: ERA5-Land hardly ever reports exactly zero precipitation, so an event needs a floor.
#: 1 mm/day follows the soil-moisture dry-down analysis already in this repository.
SERVABLE_RAIN_EVENT_MM_PER_DAY = 1.0

#: Daily sum of ERA5-Land hourly precipitation at the site, in metres (a merge output column).
ERA5_DAILY_PRECIP_COL = "total_precipitation_hourly_sum"
M_TO_MM = 1000.0


def validate_contract(contract: str) -> str:
    """Return ``contract`` if it is a known feature contract, else raise ``ValueError``."""
    if contract not in FEATURE_CONTRACTS:
        raise ValueError(f"unknown feature contract {contract!r}; expected one of {FEATURE_CONTRACTS}")
    return contract


#: ERA5-Land soil layers: 0-7, 7-28, 28-100 and 100-289 cm.
SWC_LAYERS = (1, 2, 3, 4)
SWC_LEVEL_SUFFIX = "_raw"
SWC_ANOMALY_SUFFIX = "_zscore"
#: Daily-statistic suffixes the merged tree appends to either family, e.g. ``..._raw_min``.
SWC_STAT_SUFFIXES = ("", "_min", "_max", "_sum")


def _swc_base(layer: int) -> str:
    if layer not in SWC_LAYERS:
        raise ValueError(f"unknown soil-moisture layer {layer!r}; expected one of {SWC_LAYERS}")
    return f"volumetric_soil_water_layer_{layer}"


def swc_level_col(layer: int) -> str:
    """The level column of ``layer``: volumetric water content, m3/m3."""
    return _swc_base(layer) + SWC_LEVEL_SUFFIX


def swc_anomaly_col(layer: int) -> str:
    """The anomaly column of ``layer``: (raw - mu) / sigma, mu and sigma as the contract defines them."""
    return _swc_base(layer) + SWC_ANOMALY_SUFFIX


#: The one input of every derived soil-moisture feature of the top layer.
SWC_DERIVED_SOURCE = swc_level_col(1)


def _retired_swc_features() -> dict:
    retired = {}
    for layer in SWC_LAYERS:
        for stat in SWC_STAT_SUFFIXES:
            level, anomaly = swc_level_col(layer) + stat, swc_anomaly_col(layer) + stat
            retired[_swc_base(layer) + stat] = f"{level} (level) or {anomaly} (anomaly)"
    for layer in SWC_LAYERS[1:]:
        retired[f"swc_layer{layer}_norm"] = swc_anomaly_col(layer)
    return retired


#: Soil-moisture feature names retired on 2026-09-14, each mapped to its replacement.
RETIRED_SWC_FEATURES = _retired_swc_features()


def assert_no_retired_swc_features(names, source: str) -> None:
    """Raise ``ValueError`` when ``names`` contains a retired soil-moisture feature.

    The merge still writes the bare columns, so a stale feature list does not fail as "missing":
    it would train on the retired definition and nothing downstream would notice. ``source``
    names the list in the message.
    """
    stale = sorted(set(names) & set(RETIRED_SWC_FEATURES))
    if stale:
        detail = "; ".join(f"{name} -> {RETIRED_SWC_FEATURES[name]}" for name in stale)
        raise ValueError(f"{source} names retired soil-moisture features: {detail}")


#: Feature names that hold a quantity another feature already carries, under ``servable`` only.
#:
#: ``precip`` IS the ERA5-Land daily total under that contract -- the merge writes it as the daily
#: MEAN of the hourly values x 24 x 1000 -- and :data:`ERA5_DAILY_PRECIP_COL` is that same total in
#: metres. ``complete_day_statistics`` takes the mean and the sum over the same complete day and
#: emits neither unless every hour is present, so on any row that exists the sum is exactly 24 x the
#: mean: ``precip == M_TO_MM * total_precipitation_hourly_sum``. Offering both spends two selection
#: slots on one driver and splits its SHAP credit between two names.
#:
#: Under ``site`` they are different measurements -- the rain GAUGE and the reanalysis -- and both
#: belong. The column is LOADED under either contract: feature engineering's ``cwd`` group reads it.
SERVABLE_DUPLICATE_FEATURES = {
    ERA5_DAILY_PRECIP_COL: "precip, the same ERA5-Land daily total (precip == 1000 x this column)",
}


def servable_duplicate_features(contract: str) -> dict:
    """``{name: what it duplicates}`` under ``contract``; empty for every contract but servable."""
    return dict(SERVABLE_DUPLICATE_FEATURES) if validate_contract(contract) == SERVABLE_CONTRACT else {}


def assert_no_servable_duplicate_features(names, source: str, contract: str) -> None:
    """Raise ``ValueError`` when ``names`` carries two names for one quantity under ``contract``.

    Like the retired soil-moisture names, these do not announce themselves as "missing": the column
    is present and numeric, so a stale list trains on one driver twice and nothing else notices.
    """
    duplicates = servable_duplicate_features(contract)
    stale = sorted(set(names) & set(duplicates))
    if stale:
        detail = "; ".join(f"{name} -> {duplicates[name]}" for name in stale)
        raise ValueError(
            f"{source} names features that duplicate another under the {contract!r} contract: {detail}"
        )


#: Daily MEAN of the ERA5-Land hourly precipitation values at the site, metres per hour (a merge
#: output column). The daily SUM counts only the hours present in the merged hourly frame, and 2.1%
#: of sap-days in the 2026-09-02 servable tree have fewer than 24 (FIN_HYY_SME and SEN_SOU_* 23 on
#: every day, others down to 1), so the sum undercounts where the mean does not.
ERA5_HOURLY_MEAN_PRECIP_COL = "total_precipitation_hourly"
HOURS_PER_DAY = 24


def era5_hourly_mean_precip_mm_per_day(mean_m_per_hour: pd.Series) -> pd.Series:
    """ERA5-Land daily mean of hourly precipitation, m/h -> mm/day; tiny negative values set to 0.

    The unit of the prediction pipeline's daily total. A day without ERA5 hours stays NaN, where the
    merge's daily sum reads 0 mm.
    """
    return (mean_m_per_hour * HOURS_PER_DAY * M_TO_MM).clip(lower=0.0)


def assert_dataset_contract(config_dir, contract: str) -> None:
    """Refuse a dataset that was not built under ``contract``.

    ``config_dir`` is the merged config directory (the parent of ``daily/``).

    * ``servable`` needs ``RUN_PROVENANCE.json`` with a ``feature_contract`` block naming it.
      A legacy tree also carries ``*_zscore`` columns, but they are per-SITE z-scores; reading
      them as climatological would train a model that looks servable and is not.
    * ``site`` accepts legacy trees (with or without provenance) and refuses a servable tree,
      whose ``*_zscore`` columns hold the climatology anomaly, not the per-site z-score.
    """
    contract = validate_contract(contract)
    directory = Path(config_dir)
    path = directory / PROVENANCE_FILE
    sidecar = directory / SIDECAR_FILE
    block = None
    if path.exists():
        with open(path) as fh:
            block = json.load(fh).get(PROVENANCE_KEY)
    if not isinstance(block, dict) and sidecar.exists():
        # The merge writes its badge here and the sweep folds it into the provenance afterwards. A
        # tree is servable the moment the merge finishes, so the fold is a convenience, not a gate.
        with open(sidecar) as fh:
            block = json.load(fh)
    built_as = block.get("name") if isinstance(block, dict) else None

    if contract == SERVABLE_CONTRACT:
        if not path.exists() and not sidecar.exists():
            raise ValueError(
                f"feature contract 'servable' needs {PROVENANCE_FILE} or {SIDECAR_FILE} under "
                f"{config_dir}, and there is neither. Build the tree with the Stage-4 merge (it is "
                "servable by default), or convert an existing one with "
                "`python -m src.feature_contract.rebuild_daily`."
            )
        if built_as != SERVABLE_CONTRACT:
            raise ValueError(
                f"feature contract 'servable' needs a '{PROVENANCE_KEY}' block naming 'servable' in "
                f"{path}, or a {SIDECAR_FILE} naming it; found {block!r}"
            )
        return

    if built_as is not None and built_as != contract:
        raise ValueError(
            f"feature contract {contract!r} cannot read a tree built as {built_as!r} ({path}); "
            f"pass --FEATURE_CONTRACT {built_as}"
        )
