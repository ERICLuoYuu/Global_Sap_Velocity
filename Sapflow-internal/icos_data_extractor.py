#!/usr/bin/env python3
"""
Unified ICOS Environmental Data Extractor for Sapflow Sites
============================================================

Replaces the separate ``icos_env_extractor.py`` (L2 products) and
``icos_fluxnet_extractor.py`` (Fluxnet Product fallback) with a single
script that uses a **strategy pattern**:

  1. Try L2-tier products (ETC L2 Fluxnet, Meteo, etc.)
  2. If L2 fails or has no temporal overlap → fall back to "Fluxnet Product"

Output
------
``{site}_env_data.csv`` written to the ``sapwood/`` directory, overwriting
the existing placeholder stubs.

Required package
----------------
``pip install icoscp``
"""

from __future__ import annotations

import abc
import argparse
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Site registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ICOSSiteConfig:
    """Metadata for a single sapflow ↔ ICOS site pair."""

    sapflow_name: str
    icos_id: str


SITE_REGISTRY: List[ICOSSiteConfig] = [
    # --- from icos_fluxnet_extractor (authoritative IDs) ---
    ICOSSiteConfig("ES-LM1", "ES-LM1"),
    ICOSSiteConfig("ES-LM2", "ES-LM2"),
    ICOSSiteConfig("IT-CP2_sapwood", "IT-Cp2"),
    ICOSSiteConfig("AT_Mmg", "AT-Mmg"),
    ICOSSiteConfig("CH-Lae_daily", "CH-Lae"),
    ICOSSiteConfig("CH-Lae_halfhourly", "CH-Lae"),
    ICOSSiteConfig("ES-Abr", "ES-Abr"),
    # --- from icos_env_extractor ---
    ICOSSiteConfig("CH-Dav_daily", "CH-Dav"),
    ICOSSiteConfig("CH-Dav_halfhourly", "CH-Dav"),
    ICOSSiteConfig("DE-Har", "DE-Har"),
    ICOSSiteConfig("DE-HoH", "DE-HoH"),
    ICOSSiteConfig("ES-LMa", "ES-LMa"),
    ICOSSiteConfig("FI-Hyy", "FI-Hyy"),
    ICOSSiteConfig("FR-BIL", "FR-Bil"),
    ICOSSiteConfig("SE-Nor", "SE-Nor"),
]

# ---------------------------------------------------------------------------
# Unified variable mapping (superset of both original scripts)
# ---------------------------------------------------------------------------

VARIABLE_MAPPING = {
    # --- core (present in both scripts) ---
    "TA_F": "ta",
    "VPD_F": "vpd",
    "SW_IN_F": "sw_in",
    "PA_F": "pa",
    "WS_F": "ws",
    "P_F": "precip",
    # --- additional (env_extractor only) ---
    "PPFD_IN": "ppfd",
    "RH": "rh",
    "TS_F_MDS_1": "ts",
    "SWC_F_MDS_1": "swc",
}

QC_MAPPING = {
    "TA_F_QC": "ta_qc",
    "VPD_F_QC": "vpd_qc",
    "SW_IN_F_QC": "sw_in_qc",
    "PA_F_QC": "pa_qc",
    "WS_F_QC": "ws_qc",
    "P_F_QC": "precip_qc",
}

ALL_MAPPINGS = {**VARIABLE_MAPPING, **QC_MAPPING}

# ---------------------------------------------------------------------------
# Strategy pattern — abstract base + two concrete strategies
# ---------------------------------------------------------------------------


class ExtractionStrategy(abc.ABC):
    """Base class for ICOS data-product extraction strategies."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable strategy name (used in log messages)."""

    @abc.abstractmethod
    def extract(
        self,
        data_catalog: pd.DataFrame,
        dobj_cls,
    ) -> Optional[pd.DataFrame]:
        """
        Attempt to download data from the ICOS Carbon Portal.

        Parameters
        ----------
        data_catalog : pd.DataFrame
            Result of ``station.data()`` for the target station.
        dobj_cls
            The ``icoscp.dobj.Dobj`` class (injected to ease testing).

        Returns
        -------
        pd.DataFrame or None
            Raw data with an ICOS ``TIMESTAMP`` column, or *None* on failure.
        """


class L2Strategy(ExtractionStrategy):
    """Try Level-2 half-hourly / meteo products."""

    _PRIORITIES = [
        r"Fluxnet.*half",
        r"ETC L2 Fluxnet",
        r"ETC L2 Meteo",
        r"ETC L2 ARCHIVE",
    ]

    @property
    def name(self) -> str:  # noqa: D102
        return "L2"

    def extract(  # noqa: D102
        self,
        data_catalog: pd.DataFrame,
        dobj_cls,
    ) -> Optional[pd.DataFrame]:
        for pattern in self._PRIORITIES:
            matches = data_catalog[
                data_catalog["specLabel"].str.contains(
                    pattern, case=False, regex=True
                )
            ]
            if len(matches) > 0:
                row = matches.iloc[0]
                logger.info("  L2 product found: %s", row["specLabel"])
                dobj = dobj_cls(row["dobj"])
                df = dobj.data
                if df is not None and len(df) > 0:
                    return df
        return None


class FluxnetStrategy(ExtractionStrategy):
    """Fall back to the broader "Fluxnet Product" archive."""

    @property
    def name(self) -> str:  # noqa: D102
        return "Fluxnet Product"

    def extract(  # noqa: D102
        self,
        data_catalog: pd.DataFrame,
        dobj_cls,
    ) -> Optional[pd.DataFrame]:
        matches = data_catalog[data_catalog["specLabel"] == "Fluxnet Product"]
        if len(matches) == 0:
            return None
        row = matches.iloc[0]
        time_start = str(row.get("timeStart", ""))[:10]
        time_end = str(row.get("timeEnd", ""))[:10]
        logger.info("  Fluxnet Product found: %s to %s", time_start, time_end)
        dobj = dobj_cls(row["dobj"])
        df = dobj.data
        if df is not None and len(df) > 0:
            return df
        return None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class ICOSDataExtractor:
    """
    Download & standardize ICOS environmental data for sapflow sites.

    Parameters
    ----------
    sapflow_dir : Path
        Directory containing ``*_sapflow_sapwood.csv`` files.
    output_dir : Path, optional
        Where to write ``*_env_data.csv`` files.  Defaults to *sapflow_dir*.
    strategies : list[ExtractionStrategy], optional
        Ordered list of extraction strategies to try.
    """

    def __init__(
        self,
        sapflow_dir: Path,
        output_dir: Optional[Path] = None,
        strategies: Optional[List[ExtractionStrategy]] = None,
    ) -> None:
        self.sapflow_dir = Path(sapflow_dir)
        self.output_dir = Path(output_dir) if output_dir else self.sapflow_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.strategies = strategies or [L2Strategy(), FluxnetStrategy()]

        # Late-import icoscp so tests can mock it
        try:
            from icoscp.station import station as _station_mod  # type: ignore[import-untyped]
            from icoscp.dobj import Dobj as _Dobj_cls  # type: ignore[import-untyped]

            self._station_mod = _station_mod
            self._dobj_cls = _Dobj_cls
            logger.info("icoscp library loaded successfully")
        except ImportError as exc:
            raise ImportError(
                "The 'icoscp' package is required.  Install it with:\n"
                "    pip install icoscp"
            ) from exc

    # ------------------------------------------------------------------ helpers

    def _resolve_sapflow_time_range(
        self, site_config: ICOSSiteConfig
    ) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """Return (start, end) timestamps from the sapflow CSV."""
        sapf_file = (
            self.sapflow_dir
            / f"{site_config.sapflow_name}_sapflow_sapwood.csv"
        )
        if not sapf_file.exists():
            logger.warning("  Sapflow file not found: %s", sapf_file)
            return None, None

        # Read header to find timestamp column (case-insensitive)
        header = pd.read_csv(sapf_file, nrows=0)
        ts_col = next(
            (c for c in header.columns if c.lower() == "timestamp"), None
        )
        if ts_col is None:
            logger.warning("  No timestamp column in %s", sapf_file.name)
            return None, None

        ts_series = pd.read_csv(sapf_file, usecols=[ts_col])[ts_col]
        ts_series = pd.to_datetime(ts_series, format="mixed")
        return ts_series.min(), ts_series.max()

    def _try_strategies(
        self,
        icos_id: str,
        sapf_start: pd.Timestamp,
        sapf_end: pd.Timestamp,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Iterate through strategies until one succeeds **and** has
        temporal overlap with the sapflow measurement period.

        Returns
        -------
        tuple[pd.DataFrame | None, str | None]
            (downloaded_dataframe, strategy_name) or (None, None).
        """
        st = self._station_mod.get(icos_id)
        if st is None or not getattr(st, "valid", False):
            logger.error("  ICOS station '%s' not found or invalid", icos_id)
            return None, None

        logger.info(
            "  Station: %s (lat=%.4f, lon=%.4f)",
            getattr(st, "name", icos_id),
            getattr(st, "lat", 0.0),
            getattr(st, "lon", 0.0),
        )

        data_catalog = st.data()
        if data_catalog is None or len(data_catalog) == 0:
            logger.error("  No data products available for %s", icos_id)
            return None, None

        # Make sapflow bounds timezone-naive for comparison
        start_naive = (
            sapf_start.tz_localize(None) if sapf_start.tzinfo else sapf_start
        )
        end_naive = (
            sapf_end.tz_localize(None) if sapf_end.tzinfo else sapf_end
        )

        for strategy in self.strategies:
            try:
                df = strategy.extract(data_catalog, self._dobj_cls)
                if df is not None:
                    # Quick overlap check before accepting this strategy
                    df_check = self._standardize_columns(df.copy())
                    if (
                        not df_check.empty
                        and "TIMESTAMP" in df_check.columns
                    ):
                        overlap = df_check[
                            (df_check["TIMESTAMP"] >= start_naive)
                            & (df_check["TIMESTAMP"] <= end_naive)
                        ]
                        if len(overlap) > 0:
                            return df, strategy.name
                        logger.warning(
                            "  %s data for %s has no temporal overlap "
                            "with sapflow period, trying next…",
                            strategy.name,
                            icos_id,
                        )
                        continue
                    else:
                        return df, strategy.name
            except Exception:
                logger.debug(
                    "  %s raised an exception for %s",
                    strategy.name,
                    icos_id,
                    exc_info=True,
                )
            logger.warning(
                "  %s strategy failed for %s, trying next…",
                strategy.name,
                icos_id,
            )

        return None, None

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize timestamp handling and rename ICOS columns to the
        project's standard names.
        """
        # --- locate / create a TIMESTAMP column ---
        if "TIMESTAMP" in df.columns:
            df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
        elif "TIMESTAMP_START" in df.columns:
            df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP_START"])
        else:
            ts_cols = [
                c
                for c in df.columns
                if "TIME" in c.upper() or "DATE" in c.upper()
            ]
            if ts_cols:
                df["TIMESTAMP"] = pd.to_datetime(df[ts_cols[0]])
            else:
                logger.error("  No timestamp column found in downloaded data")
                return pd.DataFrame()

        # Strip timezone info
        if df["TIMESTAMP"].dt.tz is not None:
            df["TIMESTAMP"] = df["TIMESTAMP"].dt.tz_localize(None)

        # --- rename environmental columns ---
        rename_map: dict[str, str] = {}
        for icos_name, std_name in ALL_MAPPINGS.items():
            if icos_name in df.columns:
                rename_map[icos_name] = std_name
            else:
                # Fallback: strip common suffixes (_F, _MDS_1) and try again
                alt = icos_name.replace("_F", "").replace("_MDS_1", "")
                if alt in df.columns:
                    rename_map[alt] = std_name

        df = df.rename(columns=rename_map)

        # Keep only TIMESTAMP + successfully mapped columns
        keep = ["TIMESTAMP"] + [
            c for c in df.columns if c in ALL_MAPPINGS.values()
        ]
        return df[[c for c in keep if c in df.columns]].copy()

    # --------------------------------------------------------------- public API

    def process_site(self, site_config: ICOSSiteConfig) -> bool:
        """
        Extract ICOS data for a single site and save to disk.

        Returns
        -------
        bool
            *True* on success, *False* otherwise.
        """
        name = site_config.sapflow_name
        logger.info(
            "\nProcessing: %s (ICOS: %s)", name, site_config.icos_id
        )

        # 1. Sapflow time range
        start, end = self._resolve_sapflow_time_range(site_config)
        if start is None:
            logger.error(
                "  Skipping %s: could not determine sapflow time range", name
            )
            return False
        logger.info("  Sapflow period: %s to %s", start.date(), end.date())

        # 2. Download via strategy chain (with temporal overlap check)
        raw_df, strategy_name = self._try_strategies(
            site_config.icos_id, start, end
        )
        if raw_df is None:
            logger.error("  All extraction strategies failed for %s", name)
            return False

        logger.info(
            "  Downloaded %d rows via %s strategy",
            len(raw_df),
            strategy_name,
        )

        # 3. Standardize columns
        df = self._standardize_columns(raw_df)
        if df.empty or "TIMESTAMP" not in df.columns:
            logger.error(
                "  Column standardization produced empty result for %s", name
            )
            return False

        # 4. Filter to sapflow date range
        start_naive = (
            start.tz_localize(None) if start.tzinfo else start
        )
        end_naive = end.tz_localize(None) if end.tzinfo else end

        mask = (df["TIMESTAMP"] >= start_naive) & (
            df["TIMESTAMP"] <= end_naive
        )
        df = df.loc[mask]

        if df.empty:
            logger.error(
                "  No overlapping data for %s (ICOS vs sapflow date range)",
                name,
            )
            return False

        logger.info(
            "  Filtered to %d rows (%s to %s)",
            len(df),
            df["TIMESTAMP"].min().date(),
            df["TIMESTAMP"].max().date(),
        )

        # 5. Save
        output_file = self.output_dir / f"{name}_env_data.csv"
        df.to_csv(output_file, index=False)
        logger.info(
            "  Saved: %s  |  strategy=%s  |  columns=%s",
            output_file.name,
            strategy_name,
            list(df.columns),
        )
        return True

    def process_all_sites(
        self, sites: Optional[List[ICOSSiteConfig]] = None
    ) -> None:
        """Process every site in *sites* (defaults to ``SITE_REGISTRY``)."""
        sites = sites or SITE_REGISTRY
        results: dict[str, bool] = {}

        for site in sites:
            results[site.sapflow_name] = self.process_site(site)

        # --- summary ---
        ok = sum(v for v in results.values())
        fail = len(results) - ok
        logger.info("\n" + "=" * 60)
        logger.info("EXTRACTION SUMMARY")
        logger.info("=" * 60)
        logger.info("Successful : %d", ok)
        logger.info("Failed     : %d", fail)
        for site_name, success in results.items():
            logger.info("  %s %s", "✓" if success else "✗", site_name)
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run the extractor."""
    parser = argparse.ArgumentParser(
        description="Unified ICOS environmental-data extractor for sapflow sites.",
    )
    parser.add_argument(
        "--site",
        type=str,
        default=None,
        help=(
            "Process a single site (e.g. --site CH-Dav_daily). "
            "Default: process all sites in the registry."
        ),
    )
    args = parser.parse_args()

    sapflow_dir = (
        Path(__file__).parent
        / "Sapflow_SAPFLUXNET_format_unitcon"
        / "sapwood"
    )

    logger.info("=" * 60)
    logger.info("Unified ICOS Data Extractor")
    logger.info("=" * 60)
    logger.info("Sapflow dir : %s", sapflow_dir)
    logger.info("Output dir  : %s", sapflow_dir)

    extractor = ICOSDataExtractor(sapflow_dir=sapflow_dir)

    if args.site:
        match = next(
            (s for s in SITE_REGISTRY if s.sapflow_name == args.site), None
        )
        if match is None:
            logger.error(
                "Site '%s' not found in SITE_REGISTRY", args.site
            )
            return
        extractor.process_site(match)
    else:
        extractor.process_all_sites()


if __name__ == "__main__":
    main()
