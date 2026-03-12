"""
Unit tests for icos_data_extractor.py
=====================================

All ICOS network calls are mocked — no internet access required.

Run with::

    cd Sapflow-internal
    pytest tests/test_icos_data_extractor.py -v
"""

from __future__ import annotations
import logging
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Stub out ``icoscp`` *before* importing our module so the import-time
# ``from icoscp…`` inside ICOSDataExtractor.__init__ never hits the network.
# ---------------------------------------------------------------------------
_fake_station_mod = types.ModuleType("icoscp.station")
_fake_dobj_mod = types.ModuleType("icoscp.dobj")
_fake_icoscp = types.ModuleType("icoscp")

_fake_station_mod.station = MagicMock()  # type: ignore[attr-defined]
_fake_dobj_mod.Dobj = MagicMock()  # type: ignore[attr-defined]
_fake_icoscp.station = _fake_station_mod  # type: ignore[attr-defined]
_fake_icoscp.dobj = _fake_dobj_mod  # type: ignore[attr-defined]

sys.modules.setdefault("icoscp", _fake_icoscp)
sys.modules.setdefault("icoscp.station", _fake_station_mod)
sys.modules.setdefault("icoscp.dobj", _fake_dobj_mod)

# Make the parent directory importable
PARENT_DIR = Path(__file__).resolve().parent.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from icos_data_extractor import (  # noqa: E402
    ALL_MAPPINGS,
    FluxnetStrategy,
    ICOSDataExtractor,
    ICOSSiteConfig,
    L2Strategy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SAMPLE_SITE = ICOSSiteConfig(sapflow_name="TEST-Site", icos_id="TE-Tst")


def _make_sapflow_csv(directory: Path, site_name: str = "TEST-Site") -> None:
    """Write a minimal sapflow CSV so the extractor can read a time range."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2020-01-01", periods=100, freq="30min"
            ),
            "tree-1": range(100),
        }
    )
    df.to_csv(directory / f"{site_name}_sapflow_sapwood.csv", index=False)


def _make_icos_df(
    start: str = "2019-06-01",
    periods: int = 200,
    freq: str = "30min",
) -> pd.DataFrame:
    """Return a DataFrame that looks like raw ICOS download output."""
    ts = pd.date_range(start, periods=periods, freq=freq)
    return pd.DataFrame(
        {
            "TIMESTAMP": ts,
            "TA_F": range(periods),
            "VPD_F": range(periods),
            "SW_IN_F": range(periods),
            "PA_F": range(periods),
            "WS_F": range(periods),
            "P_F": range(periods),
            "TA_F_QC": [0] * periods,
            "VPD_F_QC": [0] * periods,
            "SW_IN_F_QC": [0] * periods,
            "PA_F_QC": [0] * periods,
            "WS_F_QC": [0] * periods,
            "P_F_QC": [0] * periods,
        }
    )


def _build_extractor(
    tmp_path: Path,
    strategies=None,
    station_mock=None,
    dobj_mock=None,
) -> ICOSDataExtractor:
    """
    Build an ``ICOSDataExtractor`` whose icoscp internals are mocks.
    """
    _make_sapflow_csv(tmp_path)

    ext = ICOSDataExtractor.__new__(ICOSDataExtractor)
    ext.sapflow_dir = tmp_path
    ext.output_dir = tmp_path
    ext.strategies = strategies or [L2Strategy(), FluxnetStrategy()]
    ext._station_mod = station_mock or MagicMock()
    ext._dobj_cls = dobj_mock or MagicMock()
    return ext


def _mock_station(valid: bool = True):
    """Return a mock station module whose ``.get()`` returns a station object."""
    st_obj = MagicMock()
    st_obj.valid = valid
    st_obj.name = "Test Station"
    st_obj.lat = 50.0
    st_obj.lon = 10.0

    mod = MagicMock()
    mod.get.return_value = st_obj
    return mod, st_obj


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestL2Success:
    """L2 strategy succeeds on first attempt — no fallback triggered."""

    def test_produces_correct_output(self, tmp_path, caplog):
        caplog.set_level(logging.INFO)
        icos_df = _make_icos_df(start="2020-01-01", periods=200)

        station_mod, st_obj = _mock_station()
        catalog = pd.DataFrame(
            {"specLabel": ["ETC L2 Fluxnet"], "dobj": ["pid123"]}
        )
        st_obj.data.return_value = catalog

        dobj_cls = MagicMock()
        dobj_cls.return_value.data = icos_df

        ext = _build_extractor(
            tmp_path, station_mock=station_mod, dobj_mock=dobj_cls
        )

        result = ext.process_site(SAMPLE_SITE)

        assert result is True
        out_file = tmp_path / "TEST-Site_env_data.csv"
        assert out_file.exists()

        df_out = pd.read_csv(out_file)
        assert "TIMESTAMP" in df_out.columns
        assert "ta" in df_out.columns
        assert "vpd" in df_out.columns
        assert "precip" in df_out.columns
        assert len(df_out) > 0
        # L2 strategy should be mentioned in the logs
        assert "L2" in caplog.text


class TestFallbackToFluxnet:
    """L2 strategy fails → fallback to Fluxnet Product is triggered."""

    def test_fluxnet_fallback_triggered(self, tmp_path, caplog):
        caplog.set_level(logging.INFO)
        icos_df = _make_icos_df(start="2020-01-01", periods=200)

        station_mod, st_obj = _mock_station()
        # Catalog only has "Fluxnet Product" — no L2 labels at all
        catalog = pd.DataFrame(
            {
                "specLabel": ["Fluxnet Product"],
                "dobj": ["pid_fluxnet"],
                "timeStart": ["2019-01-01"],
                "timeEnd": ["2025-01-01"],
            }
        )
        st_obj.data.return_value = catalog

        dobj_cls = MagicMock()
        dobj_cls.return_value.data = icos_df

        ext = _build_extractor(
            tmp_path, station_mock=station_mod, dobj_mock=dobj_cls
        )

        result = ext.process_site(SAMPLE_SITE)

        assert result is True
        # L2 should have failed (no L2-tier label in catalog)
        assert "L2 strategy failed" in caplog.text
        # Fluxnet Product should have been used
        assert "Fluxnet Product" in caplog.text

        out_file = tmp_path / "TEST-Site_env_data.csv"
        assert out_file.exists()


class TestBothStrategiesFail:
    """Both strategies fail — process_site returns False."""

    def test_returns_false_and_logs_error(self, tmp_path, caplog):
        station_mod, st_obj = _mock_station()
        # Empty catalog → nothing matches
        catalog = pd.DataFrame({"specLabel": pd.Series([], dtype=str), "dobj": pd.Series([], dtype=str)})
        st_obj.data.return_value = catalog

        ext = _build_extractor(tmp_path, station_mock=station_mod)

        result = ext.process_site(SAMPLE_SITE)

        assert result is False
        assert "failed" in caplog.text.lower()


class TestTimeRangeFiltering:
    """ICOS data entirely outside sapflow range → no output."""

    def test_no_overlap_returns_false(self, tmp_path, caplog):
        # Sapflow is 2020-01-01..+100×30min, ICOS data is 2022 only
        icos_df = _make_icos_df(start="2022-06-01", periods=100)

        station_mod, st_obj = _mock_station()
        catalog = pd.DataFrame(
            {"specLabel": ["ETC L2 Fluxnet"], "dobj": ["pid_l2"]}
        )
        st_obj.data.return_value = catalog

        dobj_cls = MagicMock()
        dobj_cls.return_value.data = icos_df

        ext = _build_extractor(
            tmp_path, station_mock=station_mod, dobj_mock=dobj_cls
        )

        result = ext.process_site(SAMPLE_SITE)

        assert result is False
        assert "overlap" in caplog.text.lower()


class TestColumnStandardization:
    """Verify that raw ICOS column names are mapped to standard names."""

    def test_renames_correctly(self):
        df = _make_icos_df(periods=5)
        result = ICOSDataExtractor._standardize_columns(df)

        # Standard names present
        assert "TIMESTAMP" in result.columns
        assert "ta" in result.columns
        assert "vpd" in result.columns
        assert "sw_in" in result.columns
        assert "precip" in result.columns
        assert "ws" in result.columns
        assert "pa" in result.columns
        # QC columns present
        assert "ta_qc" in result.columns
        assert "vpd_qc" in result.columns
        assert "sw_in_qc" in result.columns
        # Original ICOS names removed
        assert "TA_F" not in result.columns
        assert "VPD_F" not in result.columns
        assert "P_F" not in result.columns
