# ruff: noqa: N999
"""Shared pytest configuration and fixtures for src/Analyzers tests.

Adds the project root to sys.path so tests can import ``src.Analyzers.*``
regardless of pytest's invocation directory, and provides synthetic
wide-format sap_data + plant_md + stand_md fixtures covering every edge
case exercised by test_calibration_corrector and test_treatment_filter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast, hermetic unit test")
    config.addinivalue_line("markers", "integration: multi-module integration test")
    config.addinivalue_line("markers", "slow: slow smoke tests on real site data")


# --------------------------------------------------------------------------
# Synthetic 15-plant fixture — matches Table §4.2 of the plan doc.
# Each plant column exercises a distinct combination of sensor method,
# calibration flag, and plant/stand treatment. Timestamps are hourly.
# --------------------------------------------------------------------------

_N_ROWS = 24  # 24 hourly readings = one day, plenty for unit tests
_BASE_VALUE = 10.0  # arbitrary SFD baseline; corrections are multiplicative

_PLANT_SPECS: list[dict] = [
    # pl_code, pl_sens_meth, pl_sens_calib, pl_treatment,         st_treatment,
    #                                                             site_code
    {
        "pl_code": "P01",
        "pl_sens_meth": "HD",
        "pl_sens_calib": "FALSE",
        "pl_treatment": None,
        "st_treatment": None,
        "si_code": "FAKE_01",
    },
    {
        "pl_code": "P02",
        "pl_sens_meth": "HD",
        "pl_sens_calib": None,
        "pl_treatment": "Control",
        "st_treatment": None,
        "si_code": "FAKE_01",
    },
    {
        "pl_code": "P03",
        "pl_sens_meth": "HD",
        "pl_sens_calib": "TRUE",
        "pl_treatment": None,
        "st_treatment": None,
        "si_code": "FAKE_01",
    },
    {
        "pl_code": "P04",
        "pl_sens_meth": "HD",
        "pl_sens_calib": "True",
        "pl_treatment": None,
        "st_treatment": None,
        "si_code": "FAKE_01",
    },
    {
        "pl_code": "P05",
        "pl_sens_meth": "HR",
        "pl_sens_calib": "FALSE",
        "pl_treatment": None,
        "st_treatment": None,
        "si_code": "FAKE_01",
    },
    {
        "pl_code": "P06",
        "pl_sens_meth": "CHD",
        "pl_sens_calib": "FALSE",
        "pl_treatment": None,
        "st_treatment": None,
        "si_code": "FAKE_01",
    },
    {
        "pl_code": "P07",
        "pl_sens_meth": "HPTM",
        "pl_sens_calib": None,
        "pl_treatment": None,
        "st_treatment": None,
        "si_code": "FAKE_01",
    },
    {
        "pl_code": "P08",
        "pl_sens_meth": "HD",
        "pl_sens_calib": "FALSE",
        "pl_treatment": "Irrigation",
        "st_treatment": None,
        "si_code": "FAKE_02",
    },
    {
        "pl_code": "P09",
        "pl_sens_meth": "HD",
        "pl_sens_calib": "FALSE",
        "pl_treatment": None,
        "st_treatment": "Drought",
        "si_code": "FAKE_02",
    },
    {
        "pl_code": "P10",
        "pl_sens_meth": "HPTM",
        "pl_sens_calib": "FALSE",
        "pl_treatment": "Ambient CO2",
        "st_treatment": "Elevated atmospheric CO2",
        "si_code": "AUS_RIC_EUC_ELE",
    },
    {
        "pl_code": "P11",
        "pl_sens_meth": "HD",
        "pl_sens_calib": "FALSE",
        "pl_treatment": None,
        "st_treatment": "Pre Irrigation",
        "si_code": "ESP_SEN_SOU_PRE",
    },
    {
        "pl_code": "P12",
        "pl_sens_meth": "HD",
        "pl_sens_calib": "FALSE",
        "pl_treatment": "Post-thinning",
        "st_treatment": None,
        "si_code": "FAKE_03",
    },
    {
        "pl_code": "P13",
        "pl_sens_meth": "HD",
        "pl_sens_calib": "FALSE",
        "pl_treatment": "drought_1",
        "st_treatment": None,
        "si_code": "USA_DUK_HAR",
    },
    {
        "pl_code": "P14",
        "pl_sens_meth": "HD",
        "pl_sens_calib": "FALSE",
        "pl_treatment": "control_1",
        "st_treatment": None,
        "si_code": "USA_DUK_HAR",
    },
    {
        "pl_code": "P15",
        "pl_sens_meth": "HD",
        "pl_sens_calib": "FALSE",
        "pl_treatment": "Increment cores",
        "st_treatment": "Drought",
        "si_code": "FAKE_02",
    },
]


def _make_sapf_wide(plant_codes: list[str]) -> pd.DataFrame:
    ts = pd.date_range("2020-06-01", periods=_N_ROWS, freq="h")
    data = {"TIMESTAMP": ts}
    for i, pc in enumerate(plant_codes):
        # Each plant gets a distinct baseline so we can verify per-column ops.
        data[pc] = np.full(_N_ROWS, _BASE_VALUE + i, dtype=float)
    # Seed one NaN cell in P01 at row 0 so NaN-preservation tests work.
    df = pd.DataFrame(data)
    if "P01" in df.columns:
        df.loc[0, "P01"] = np.nan
    return df


@pytest.fixture
def plant_md() -> pd.DataFrame:
    """Plant-level metadata frame with one row per fixture plant."""
    records = [
        {
            "pl_code": spec["pl_code"],
            "si_code": spec["si_code"],
            "pl_sens_meth": spec["pl_sens_meth"],
            "pl_sens_calib": spec["pl_sens_calib"],
            "pl_treatment": spec["pl_treatment"],
        }
        for spec in _PLANT_SPECS
    ]
    return pd.DataFrame(records)


@pytest.fixture
def stand_md() -> pd.DataFrame:
    """Stand-level metadata — one row per unique si_code in the fixture.

    Real SAPFLUXNET has one st_treatment per site. If multiple plants at the
    same site specify different st_treatment values, take the first non-null
    (matches the spirit of SAPFLUXNET's single-row-per-site stand_md.csv).
    """
    by_site: dict[str, str | None] = {}
    for spec in _PLANT_SPECS:
        site = spec["si_code"]
        st = spec["st_treatment"]
        if site not in by_site or (by_site[site] is None and st is not None):
            by_site[site] = st
    records = [{"si_code": site, "st_treatment": st} for site, st in by_site.items()]
    return pd.DataFrame(records)


@pytest.fixture
def sapf_wide() -> pd.DataFrame:
    """Wide-format sap_data frame with TIMESTAMP + P01..P15 columns."""
    return _make_sapf_wide([spec["pl_code"] for spec in _PLANT_SPECS])


@pytest.fixture
def plant_specs() -> list[dict]:
    """Raw per-plant spec list so tests can assert against expected values."""
    return [dict(s) for s in _PLANT_SPECS]


@pytest.fixture
def base_value() -> float:
    return _BASE_VALUE
