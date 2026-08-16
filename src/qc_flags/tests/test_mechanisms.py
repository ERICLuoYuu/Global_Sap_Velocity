"""Which of the three flag mechanisms a site has.

The corpus is 202 sites with SAPFLUXNET flag files, 19 internal sites carrying
in-file `*_qc` columns, 4 with neither, and 1 (`SWE_NOR`) whose env file has no
variables at all.  Detection is by inspection of what is on disk, deliberately
independent of what any analyzer decided to do -- acceptance criterion 1
compares the two, and a census derived from the code under test would be a
tautology.
"""

import pandas as pd
import pytest

from src.qc_flags.mechanisms import Mechanism, detect_mechanism, skip_reason


@pytest.fixture()
def corpus(tmp_path):
    """A miniature corpus with one site of each kind."""

    def write(name, cols, rows=2):
        frame = pd.DataFrame({c: [1] * rows for c in cols})
        frame.insert(0, "TIMESTAMP", pd.date_range("2020-01-01", periods=rows, freq="h"))
        frame.to_csv(tmp_path / name, index=False)

    # external: has a flag file
    write("ARG_MAZ_env_data.csv", ["ta", "vpd"])
    write("ARG_MAZ_env_flags.csv", ["ta", "vpd"])
    # internal ICOS: qc columns, no flag file
    write("CHE_DAV_env_data.csv", ["ta", "ta_qc", "vpd", "vpd_qc"])
    # internal non-ICOS: neither
    write("POL_MEZ_env_data.csv", ["ta", "vpd"])
    # degenerate: no variables at all
    pd.DataFrame({"TIMESTAMP": [], "solar_TIMESTAMP": []}).to_csv(tmp_path / "SWE_NOR_env_data.csv", index=False)
    return tmp_path


class TestDetectMechanism:
    def test_flag_file_present_means_sapfluxnet(self, corpus):
        assert detect_mechanism("ARG_MAZ", "env", corpus) is Mechanism.SAPFLUXNET_FLAGS

    def test_qc_columns_without_flag_file(self, corpus):
        assert detect_mechanism("CHE_DAV", "env", corpus) is Mechanism.QC_COLUMNS

    def test_neither_mechanism(self, corpus):
        assert detect_mechanism("POL_MEZ", "env", corpus) is Mechanism.NONE

    def test_empty_env_file_is_none_not_a_crash(self, corpus):
        assert detect_mechanism("SWE_NOR", "env", corpus) is Mechanism.NONE

    def test_flag_file_wins_over_qc_columns(self, corpus, tmp_path):
        # No site has both today, but precedence must be defined rather than
        # discovered: the archive's per-cell tokens are richer than a per-row
        # integer, so they take priority.
        pd.DataFrame({"TIMESTAMP": [], "ta": []}).to_csv(tmp_path / "MIX_SITE_env_data.csv", index=False)
        pd.DataFrame({"TIMESTAMP": [], "ta": []}).to_csv(tmp_path / "MIX_SITE_env_flags.csv", index=False)
        pd.DataFrame({"TIMESTAMP": [], "ta": [], "ta_qc": []}).to_csv(tmp_path / "MIX_SITE_env_data.csv", index=False)
        assert detect_mechanism("MIX_SITE", "env", tmp_path) is Mechanism.SAPFLUXNET_FLAGS

    def test_sapf_kind_uses_sapf_flag_file(self, corpus, tmp_path):
        pd.DataFrame({"TIMESTAMP": [], "TREE_1": []}).to_csv(tmp_path / "XXX_YYY_sapf_data.csv", index=False)
        assert detect_mechanism("XXX_YYY", "sapf", tmp_path) is Mechanism.NONE
        pd.DataFrame({"TIMESTAMP": [], "TREE_1": []}).to_csv(tmp_path / "XXX_YYY_sapf_flags.csv", index=False)
        assert detect_mechanism("XXX_YYY", "sapf", tmp_path) is Mechanism.SAPFLUXNET_FLAGS

    def test_missing_data_file_raises(self, corpus):
        # A site that is asked for but not present is a caller bug, not a
        # site with no mechanism. Never conflate the two.
        with pytest.raises(FileNotFoundError):
            detect_mechanism("NOT_ASITE", "env", corpus)


class TestSkipReason:
    def test_no_variables_yields_a_reason(self):
        frame = pd.DataFrame({"TIMESTAMP": [], "solar_TIMESTAMP": []})
        reason = skip_reason(frame)
        assert reason is not None
        assert "no environmental variables" in reason

    def test_a_frame_with_variables_is_not_skipped(self):
        frame = pd.DataFrame({"TIMESTAMP": [], "ta": []})
        assert skip_reason(frame) is None

    def test_qc_columns_alone_do_not_count_as_variables(self):
        # A file carrying ta_qc but no ta has nothing to clean.
        frame = pd.DataFrame({"TIMESTAMP": [], "ta_qc": []})
        assert skip_reason(frame) is not None
