"""What a model needs from a prediction cell: base columns, days of history, and features no cell can supply.

Measured on the trainer's own feature engineering rather than listed by hand: a hand-written table is a
second definition, and it drifts the first time a group changes.

* **Base columns.** A synthetic cell carrying every base column is engineered once. Then each column is
  removed, and separately perturbed, and every feature that disappears or changes depends on it.
  Perturbing catches fallbacks: ``et0`` reads ``rh_max``/``rh_min`` when present and ``rh`` otherwise.
* **History.** The cell is engineered whole and without its first :data:`SPLIT_DAY` days. A feature needs
  ``k`` days of history when the two versions agree from the ``k``-th shared day on. The comparison has a
  tolerance, because pandas' running sums carry their whole past in the last bits.
* **Never servable.** A feature whose versions do not come to agree depends on where the record starts
  (``cumulative_gdd``, and ``cwd`` without rain to clear the deficit). A cell has no record start.
* **Event-dependent.** A feature whose history need changes when rain turns rare counts from the last
  event (``days_since_rain``); a served cell needs such an event inside its spin-up.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.feature_contract.definitions import SERVABLE_CONTRACT, SWC_LAYERS, swc_anomaly_col, swc_level_col
from src.hyperparameter_optimization.shap_constants import PFT_COLUMNS
from src.serving.base_columns import (
    BASE_COLUMNS,
    DAY_COLUMN,
    PFT_COLUMN,
    SENSOR_SUBSTITUTED,
    SOURCE,
    STATIC_COLUMNS,
    TIME_COLUMN,
)
from src.serving.engineer import REQUIRED_COLUMNS, engineer_cell

N_DAYS = 480
SPLIT_DAY = 120
#: Versions that still differ within this many days of the end are taken never to agree.
MIN_AGREEING_TAIL_DAYS = 60
FREQUENT_RAIN_EVERY_DAYS = 3
RARE_RAIN_EVERY_DAYS = 97
RELATIVE_TOLERANCE = 1e-7
ABSOLUTE_TOLERANCE = 1e-9
PERTURBED_LATITUDES = (62.0, -12.0)
PERTURBED_LONGITUDE = -70.3
TIME_SHIFT = pd.Timedelta(hours=5, minutes=17)
OTHER_PFT = "DBF"

#: Hourly-scale feature names in the FFS registry; a daily cell never produces them.
_HOURLY_NAME = re.compile(r"\d+h(_|$)")
HOURLY_ONLY = frozenset({"hours_since_rain", "is_daytime"})


def is_hourly_feature(name: str) -> bool:
    return bool(_HOURLY_NAME.search(name)) or name in HOURLY_ONLY


def synthetic_cell(n_days: int = N_DAYS, rain_every_days: int = FREQUENT_RAIN_EVERY_DAYS, seed: int = 0) -> pd.DataFrame:
    """A daily frame with every base column, values that cross each threshold feature engineering tests."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2001-01-01", periods=n_days, freq="D", tz="UTC")
    t = np.arange(n_days, dtype=float)
    season = np.sin(2.0 * np.pi * t / 365.25)

    def noise(scale):
        return rng.normal(0.0, scale, n_days)

    ta = 8.0 + 14.0 * season + noise(2.0)
    vpd = np.clip(1.2 + 1.6 * season + noise(0.5), 0.05, None)
    rh = np.clip(70.0 - 20.0 * season + noise(8.0), 5.0, 100.0)
    sw_in = np.clip(160.0 + 110.0 * season + noise(40.0), 5.0, None)
    rain_mm = np.where(t % rain_every_days == 0, 9.0, 0.2)
    pet_mm = np.clip(2.5 + 2.0 * season + noise(0.5), 0.1, None)
    swc = np.clip(0.28 + 0.06 * season + noise(0.01), 0.05, 0.5)

    columns = {
        DAY_COLUMN: days,
        TIME_COLUMN: days + pd.Timedelta(minutes=47),
        "latitude": 47.3,
        "longitude": 11.7,
        PFT_COLUMN: "ENF",
        "ta": ta,
        "ta_min": ta - 4.0 - np.abs(noise(1.0)),
        "ta_max": ta + 5.0 + np.abs(noise(1.0)),
        "vpd": vpd,
        "vpd_min": vpd * 0.25,
        "vpd_max": vpd * 2.1 + np.abs(noise(0.1)),
        "rh": rh,
        "rh_min": np.clip(rh - 25.0 - np.abs(noise(3.0)), 1.0, 100.0),
        "rh_max": np.clip(rh + 15.0 + np.abs(noise(3.0)), 1.0, 100.0),
        "ws": 1.5 + np.abs(noise(1.0)),
        "sw_in": sw_in,
        "ppfd_in": sw_in * 2.114,
        "total_precipitation_hourly": rain_mm / 1000.0 / 24.0,
        "total_precipitation_hourly_sum": rain_mm / 1000.0,
        "potential_evaporation_hourly_sum": -pet_mm / 1000.0,
        "prcip/PET": np.clip(rain_mm / pet_mm, 0.0, 10.0),
        "surface_pressure": 92000.0 + noise(300.0),
        "ext_rad": 280.0 + 160.0 * season,
        "day_length": 12.2 + 3.8 * season,
        "LAI": np.clip(3.0 + 1.5 * season + noise(0.1), 0.1, None),
    }
    for layer in SWC_LAYERS:
        columns[f"soil_temperature_level_{layer}"] = 276.0 + 12.0 * season * (1.0 - 0.2 * (layer - 1)) + noise(0.5)
        raw = np.clip(swc + 0.01 * layer + noise(0.005), 0.02, 0.55)
        columns[swc_level_col(layer)] = raw
        columns[swc_anomaly_col(layer)] = (raw - 0.3) / 0.06
    statics = {
        "elevation": 850.0,
        "canopy_height": 22.0,
        "slope": 8.0,
        "aspect_sin": 0.3,
        "aspect_cos": -0.2,
        "stand_age": 90.0,
        "soil_sand": 45.0,
        "soil_clay": 20.0,
        "soil_soc": 30.0,
        "soil_cfvo": 12.0,
        "soil_theta_wp": 0.13,
        "soil_theta_fc": 0.25,
        "soil_theta_sat": 0.42,
        "mean_annual_temp": 7.5,
        "mean_annual_precip": 900.0,
        "temp_seasonality": 650.0,
        "precip_seasonality": 25.0,
    }
    frame = pd.DataFrame({**columns, **statics})
    absent = sorted(set(BASE_COLUMNS) - set(frame.columns))
    if absent:
        raise AssertionError(f"synthetic cell lacks base column(s) {absent}")
    return frame


def _variants(frame: pd.DataFrame, column: str, rng: np.random.Generator) -> list:
    if column == TIME_COLUMN:
        variants = [frame.assign(**{column: frame[column] + TIME_SHIFT})]
    elif column == PFT_COLUMN:
        variants = [frame.assign(**{column: OTHER_PFT})]
    elif column == "latitude":
        variants = [frame.assign(**{column: value}) for value in PERTURBED_LATITUDES]
    elif column == "longitude":
        variants = [frame.assign(**{column: PERTURBED_LONGITUDE})]
    elif column in STATIC_COLUMNS:
        variants = [frame.assign(**{column: frame[column] * 1.37 + 0.5})]
    else:
        values = frame[column].to_numpy(dtype=float)
        variants = [
            frame.assign(**{column: rng.permutation(values)}),
            frame.assign(**{column: values * 1.37 + 0.11 * float(np.nanstd(values) or 1.0)}),
        ]
    if column not in REQUIRED_COLUMNS:
        variants.append(frame.drop(columns=column))
    return variants


def _agree(a: pd.Series, b: pd.Series) -> np.ndarray:
    return np.isclose(
        a.to_numpy(dtype=float),
        b.to_numpy(dtype=float),
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
        equal_nan=True,
    )


def _changed(base: pd.DataFrame, other: pd.DataFrame, feature: str) -> bool:
    if feature not in other.columns or len(other) != len(base):
        return True
    return not _agree(base[feature], other[feature]).all()


def _split(feature_names: Sequence[str], engineered: pd.DataFrame, contract: str) -> tuple:
    names = list(dict.fromkeys(feature_names))
    onehot = [name for name in names if name in PFT_COLUMNS]
    features = [name for name in names if name not in PFT_COLUMNS]
    unknown = [name for name in features if name not in engineered.columns]
    if unknown:
        raise KeyError(f"feature(s) {unknown} are not produced from the base columns under contract {contract!r}")
    return names, onehot, features


def base_dependencies(feature_names: Sequence[str], contract: str = SERVABLE_CONTRACT) -> dict:
    """Feature -> the base columns it reads, as a sorted tuple."""
    frame = synthetic_cell()
    base = engineer_cell(frame, contract)
    names, onehot, features = _split(feature_names, base, contract)
    found = {name: set() for name in features}
    rng = np.random.default_rng(1)
    for column in BASE_COLUMNS:
        for variant in _variants(frame, column, rng):
            other = engineer_cell(variant, contract)
            for name in features:
                if column not in found[name] and _changed(base, other, name):
                    found[name].add(column)
    for name in onehot:
        found[name] = {PFT_COLUMN}
    return {name: tuple(sorted(found[name])) for name in names}


@dataclass(frozen=True)
class History:
    days: Optional[int]
    event_dependent: bool

    @property
    def never_servable(self) -> bool:
        return self.days is None


def _history_under(frame: pd.DataFrame, features: Sequence[str], contract: str) -> dict:
    whole = engineer_cell(frame, contract).iloc[SPLIT_DAY:]
    tail = engineer_cell(frame.iloc[SPLIT_DAY:], contract)
    needs = {}
    for name in features:
        agree = _agree(whole[name], tail[name])
        if agree.all():
            needs[name] = 0
            continue
        needed = int(np.flatnonzero(~agree)[-1]) + 1
        needs[name] = None if needed > len(tail) - MIN_AGREEING_TAIL_DAYS else needed
    return needs


def feature_history(feature_names: Sequence[str], contract: str = SERVABLE_CONTRACT) -> dict:
    """Feature -> :class:`History` (days of history needed, or never; and whether that depends on rain events)."""
    frequent_cell = synthetic_cell(rain_every_days=FREQUENT_RAIN_EVERY_DAYS)
    names, onehot, features = _split(feature_names, engineer_cell(frequent_cell, contract), contract)
    frequent = _history_under(frequent_cell, features, contract)
    rare = _history_under(synthetic_cell(rain_every_days=RARE_RAIN_EVERY_DAYS), features, contract)
    result = {name: History(days=0, event_dependent=False) for name in onehot}
    for name in features:
        days = None if frequent[name] is None or rare[name] is None else max(frequent[name], rare[name])
        result[name] = History(days=days, event_dependent=frequent[name] != rare[name])
    return {name: result[name] for name in names}


def servability_report(feature_names: Sequence[str], contract: str = SERVABLE_CONTRACT) -> pd.DataFrame:
    """One row per feature: base columns, their sources, sensor substitution, history, and the two flags."""
    dependencies = base_dependencies(feature_names, contract)
    history = feature_history(feature_names, contract)
    rows = []
    for name, columns in dependencies.items():
        rows.append(
            {
                "feature": name,
                "base_columns": ";".join(columns),
                "sources": ";".join(sorted({SOURCE[column] for column in columns})),
                "sensor_substituted": bool(set(columns) & SENSOR_SUBSTITUTED),
                "history_days": np.nan if history[name].never_servable else history[name].days,
                "never_servable": history[name].never_servable,
                "event_dependent": history[name].event_dependent,
            }
        )
    return pd.DataFrame(rows)


def required_base_columns(report: pd.DataFrame) -> tuple:
    """Every base column the report's features read, sorted."""
    columns = set()
    for entry in report["base_columns"]:
        columns.update(part for part in str(entry).split(";") if part)
    return tuple(sorted(columns))


def spin_up_days(report: pd.DataFrame) -> int:
    """Days of history a served cell needs before its first predicted day; refuses a never-servable feature."""
    never = report.loc[report["never_servable"], "feature"].tolist()
    if never:
        raise ValueError(f"feature(s) {never} depend on where a record starts and cannot be served")
    return int(report["history_days"].max()) if len(report) else 0
