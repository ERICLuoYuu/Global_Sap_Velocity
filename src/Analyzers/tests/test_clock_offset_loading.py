"""Clock offsets are applied once, at the Stage-3 load boundary (plan WP-A3; failure modes F1, F3, F5, F6, F7).

The corpus is shaped like the ORN twins: USA_ORN_ST1_AMB and USA_ORN_ST2_AMB share the loaders' two-token
``location`` key (``USA_ORN``). The files are written the way the archive ships them: the physical signal
peaks at true solar noon, ST2's labels run five hours late, and ``solar_TIMESTAMP`` was derived from the late
label. Processing is replaced by a spy, so each test sees exactly the frame QC would see.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from path_config import PathConfig
from src.Analyzers.env_analyzer import EnvironmentalAnalyzer
from src.Analyzers.sap_analyzer import SapFlowAnalyzer
from src.clock_offsets.registry import ClockOffset, ClockRegistryError

TWIN, SHIFTED = "USA_ORN_ST1_AMB", "USA_ORN_ST2_AMB"
LAT, LON, ELEV = 35.9, -84.333, 227.0
OFFSET_H = 5.0
LABELS = pd.date_range("2007-06-01 00:00", periods=96, freq="30min", tz="UTC")
# Whole seconds: the files carry second resolution, and LON / 15 h alone ends in 0.92 s.
SOLAR_SHIFT = pd.Timedelta(hours=LON / 15.0).round("s")
LOG_HEADER = (
    "entry_id,location,plant_type,column,start_time,end_time,issue_category,removal_type,notes,decision_date,reviewer\n"
)


def iso(stamps):
    return [stamp.strftime("%Y-%m-%dT%H:%M:%SZ") for stamp in stamps]


def signal(labels, offset_h):
    """A diurnal curve peaking at TRUE solar noon, zero at night, sampled at LATE labels."""
    solar = labels - pd.Timedelta(hours=offset_h) + SOLAR_SHIFT
    hours = np.asarray(solar.hour + solar.minute / 60.0, dtype=float)
    return np.clip(np.cos((hours - 12.0) / 12.0 * np.pi), 0.0, None)


def entry(component, site=SHIFTED, offset=OFFSET_H):
    return ClockOffset(site=site, component=component, offset_hours=offset, status="applied", evidence="synthetic")


def write_site(corpus, site, offset_h):
    stamps = {"TIMESTAMP": iso(LABELS), "solar_TIMESTAMP": iso(LABELS + SOLAR_SHIFT)}
    curve = signal(LABELS, offset_h)
    pd.DataFrame({"env_time_zone": ["9UTC-05:00, R"], "env_timestep": [30]}).to_csv(
        corpus / f"{site}_env_md.csv", index=False
    )
    md = pd.DataFrame({"si_lat": [LAT], "si_long": [LON], "si_elev": [ELEV]})
    md.to_csv(corpus / f"{site}_site_md.csv", index=False)
    pd.DataFrame({**stamps, "ta": 20.0 + 5.0 * curve, "sw_in": 900.0 * curve}).to_csv(
        corpus / f"{site}_env_data.csv", index=False
    )
    pd.DataFrame({**stamps, "ta": "", "sw_in": ""}).to_csv(corpus / f"{site}_env_flags.csv", index=False)
    trees = [f"{site}_Lst_Jt_1", f"{site}_Lst_Jt_2"]
    pd.DataFrame({**stamps, trees[0]: 1.0 + 10.0 * curve, trees[1]: 1.0 + 4.0 * curve}).to_csv(
        corpus / f"{site}_sapf_data.csv", index=False
    )
    pd.DataFrame({**stamps, trees[0]: "", trees[1]: ""}).to_csv(corpus / f"{site}_sapf_flags.csv", index=False)


@pytest.fixture()
def corpus(tmp_path):
    directory = tmp_path / "corpus"
    directory.mkdir()
    write_site(directory, TWIN, 0.0)
    write_site(directory, SHIFTED, OFFSET_H)
    return directory


@pytest.fixture()
def paths(tmp_path):
    return PathConfig(scale="sapwood", base_data_dir=str(tmp_path / "data"), base_output_dir=str(tmp_path / "out"))


class Spy:
    """Stands in for ``_process_*_data`` and records the frame and flags each site was handed."""

    def __init__(self, returns_frame):
        self.calls = {}
        self.returns_frame = returns_frame

    def __call__(self, df, latitude, longitude, timestep, time_zone, elevation, location, plant_type, flags):
        self.calls[f"{location}_{plant_type}"] = (df.copy(), None if flags is None else flags.copy())
        return df if self.returns_frame else None


def stamps_of(frame, column="TIMESTAMP"):
    return pd.DatetimeIndex(frame[column])


class TestEnvLoader:
    def load(self, paths, corpus, registry):
        # One analyzer per site, as production runs Stage 3 (one site per array task). The env QC report is
        # keyed on the two-token location, so both ORN twins in ONE batch collide there -- clock offsets or not.
        spy = Spy(returns_frame=True)
        for site in (TWIN, SHIFTED):
            analyzer = EnvironmentalAnalyzer(paths=paths, clock_registry=registry)
            analyzer._process_environmental_data = spy
            analyzer._load_environmental_data([corpus / f"{site}_env_data.csv"])
        return spy

    def test_the_registered_site_is_moved_to_true_time(self, paths, corpus):
        df, _flags = self.load(paths, corpus, (entry("env"),)).calls[f"{SHIFTED}_env_data"]
        assert (stamps_of(df) == LABELS - pd.Timedelta(hours=OFFSET_H)).all()
        assert (stamps_of(df, "solar_TIMESTAMP") == LABELS + SOLAR_SHIFT - pd.Timedelta(hours=OFFSET_H)).all()

    def test_its_twin_under_the_same_location_key_is_untouched(self, paths, corpus):
        df, _flags = self.load(paths, corpus, (entry("env"),)).calls[f"{TWIN}_env_data"]
        assert (stamps_of(df) == LABELS).all()

    def test_flags_move_with_their_data(self, paths, corpus):
        df, flags = self.load(paths, corpus, (entry("env"),)).calls[f"{SHIFTED}_env_data"]
        assert (stamps_of(flags) == stamps_of(df)).all()

    def test_an_empty_registry_hands_processing_the_file_as_shipped(self, paths, corpus):
        df, flags = self.load(paths, corpus, ()).calls[f"{SHIFTED}_env_data"]
        assert (stamps_of(df) == LABELS).all()
        assert (stamps_of(flags) == LABELS).all()

    def test_a_sap_entry_does_not_move_env(self, paths, corpus):
        df, _flags = self.load(paths, corpus, (entry("sap"),)).calls[f"{SHIFTED}_env_data"]
        assert (stamps_of(df) == LABELS).all()

    def test_the_corrected_radiation_peaks_at_solar_noon(self, paths, corpus):
        spy = self.load(paths, corpus, (entry("env"),))
        for site in (TWIN, SHIFTED):
            df, _flags = spy.calls[f"{site}_env_data"]
            peak = stamps_of(df)[int(np.argmax(df["sw_in"].to_numpy(dtype=float)))]
            assert (peak + SOLAR_SHIFT).hour in (11, 12), site

    def test_an_invalid_registry_stops_the_load(self, paths, corpus):
        analyzer = EnvironmentalAnalyzer(paths=paths, clock_registry=(entry("env", offset=3.1),))
        analyzer._process_environmental_data = Spy(returns_frame=True)
        with pytest.raises(ClockRegistryError):
            analyzer._load_environmental_data([corpus / f"{SHIFTED}_env_data.csv"])


class TestSapLoader:
    def load(self, paths, corpus, registry, log=None):
        analyzer = SapFlowAnalyzer(paths=paths, clock_registry=registry)
        spy = Spy(returns_frame=False)
        analyzer._process_sapflow_data = spy
        if log is not None:
            analyzer.set_removal_log(log, validate=False)
        analyzer._load_sapflow_data([corpus / f"{TWIN}_sapf_data.csv", corpus / f"{SHIFTED}_sapf_data.csv"])
        return analyzer, spy

    def test_the_registered_site_is_moved_and_its_twin_is_not(self, paths, corpus):
        _analyzer, spy = self.load(paths, corpus, (entry("sap"),))
        shifted, _flags = spy.calls[f"{SHIFTED}_sapf_data"]
        twin, _flags = spy.calls[f"{TWIN}_sapf_data"]
        assert (stamps_of(shifted) == LABELS - pd.Timedelta(hours=OFFSET_H)).all()
        assert (stamps_of(twin) == LABELS).all()

    def test_flags_move_with_their_data(self, paths, corpus):
        _analyzer, spy = self.load(paths, corpus, (entry("sap"),))
        df, flags = spy.calls[f"{SHIFTED}_sapf_data"]
        assert (stamps_of(flags) == stamps_of(df)).all()
        assert (stamps_of(flags, "solar_TIMESTAMP") == stamps_of(df, "solar_TIMESTAMP")).all()

    def test_timestamp_local_is_derived_from_the_corrected_clock(self, paths, corpus):
        _analyzer, spy = self.load(paths, corpus, (entry("sap"),))
        df, _flags = spy.calls[f"{SHIFTED}_sapf_data"]
        assert (stamps_of(df, "TIMESTAMP_LOCAL") == LABELS - pd.Timedelta(hours=OFFSET_H + 5.0)).all()

    def test_an_env_entry_does_not_move_sap(self, paths, corpus):
        _analyzer, spy = self.load(paths, corpus, (entry("env"),))
        df, _flags = spy.calls[f"{SHIFTED}_sapf_data"]
        assert (stamps_of(df) == LABELS).all()

    def test_removal_log_periods_act_on_the_labels_the_reviewer_saw(self, paths, corpus, tmp_path):
        tree = f"{SHIFTED}_Lst_Jt_1"
        log = tmp_path / "removal_log.csv"
        log.write_text(
            LOG_HEADER
            + f"1,USA_ORN,ST2_AMB_sapf_data,{tree},2007-06-01 12:00:00,2007-06-01 18:00:00,bad,period,x,2026-09-16,me\n"
        )
        _analyzer, corrected = self.load(paths, corpus, (entry("sap"),), log)
        _analyzer, uncorrected = self.load(paths, corpus, (), log)
        removed_corrected = corrected.calls[f"{SHIFTED}_sapf_data"][0][tree].isna().to_numpy()
        removed_uncorrected = uncorrected.calls[f"{SHIFTED}_sapf_data"][0][tree].isna().to_numpy()
        np.testing.assert_array_equal(removed_corrected, removed_uncorrected)
        gone = LABELS[removed_uncorrected]
        assert len(gone) == 13
        assert gone.min() == pd.Timestamp("2007-06-01 12:00", tz="UTC")
        assert gone.max() == pd.Timestamp("2007-06-01 18:00", tz="UTC")

    def test_the_qc_night_mask_sees_no_daytime_sap_only_after_correction(self, paths, corpus):
        tree = f"{SHIFTED}_Lst_Jt_1"
        for registry, clean in (((entry("sap"),), True), ((), False)):
            analyzer, spy = self.load(paths, corpus, registry)
            df, _flags = spy.calls[f"{SHIFTED}_sapf_data"]
            day = np.asarray(analyzer._calculate_daytime_mask(stamps_of(df), LAT, LON, ELEV), dtype=bool)
            sap_at_night = bool((df.loc[~day, tree].to_numpy(dtype=float) > 1.0 + 1e-9).any())
            assert sap_at_night is not clean, registry

    def test_an_invalid_registry_stops_the_load(self, paths, corpus):
        analyzer = SapFlowAnalyzer(paths=paths, clock_registry=(entry("sap", offset=3.1),))
        analyzer._process_sapflow_data = Spy(returns_frame=False)
        with pytest.raises(ClockRegistryError):
            analyzer._load_sapflow_data([corpus / f"{SHIFTED}_sapf_data.csv"])
