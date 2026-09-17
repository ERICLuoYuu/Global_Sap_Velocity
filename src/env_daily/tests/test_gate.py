"""``python -m src.env_daily.gate``: an old, interrupted or regressed daily tree cannot pass (plan WP-B3, B4; F21)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.env_daily.gate import main, read_block, tree_errors
from src.env_daily.invariants import KT_BASELINE_TOTAL, KT_RESIDUAL_SITES, env_daily_rule
from src.feature_contract.definitions import PROVENANCE_FILE, PROVENANCE_KEY, SIDECAR_FILE

REPO = Path(__file__).resolve().parents[3]


def block(**overrides):
    """A feature-contract block as the merge writes it after a complete run under the rule."""
    base = {
        "name": "servable",
        "complete": True,
        "env_daily_rule": env_daily_rule(),
        "clock_registry_sha256": "0" * 64,
        "clock_provenance": {
            "registry_sha256": "0" * 64,
            "n_sites_checked": 212,
            "sites_corrected": ["FIN_PET"],
            "mismatches": {},
        },
        "kt_exceedance": {
            "baseline_total": KT_BASELINE_TOTAL,
            "total": KT_BASELINE_TOTAL,
            "by_site": dict(KT_RESIDUAL_SITES),
        },
    }
    base.update(overrides)
    return base


def write(directory, content, name=SIDECAR_FILE):
    (directory / name).write_text(json.dumps(content))
    return directory


class TestTreeErrors:
    def test_a_tree_built_under_the_rule_passes(self):
        assert tree_errors(block()) == []

    def test_no_block_fails(self):
        assert tree_errors(None)

    def test_a_tree_built_before_the_rule_fails(self):
        old = block()
        del old["env_daily_rule"]
        assert any("env_daily_rule" in error for error in tree_errors(old))

    @pytest.mark.parametrize("version", [0, None, "1", True, 0.5])
    def test_a_rule_version_that_is_not_an_integer_of_at_least_one_fails(self, version):
        rule = dict(env_daily_rule(), version=version)
        assert any("env_daily_rule" in error for error in tree_errors(block(env_daily_rule=rule)))

    @pytest.mark.parametrize("sha", [None, ""])
    def test_a_missing_registry_hash_fails(self, sha):
        assert any("clock_registry_sha256" in error for error in tree_errors(block(clock_registry_sha256=sha)))

    @pytest.mark.parametrize("provenance", [None, {}, {"n_sites_checked": 0, "mismatches": {}}])
    def test_a_tree_that_never_checked_its_stage_3_inputs_fails(self, provenance):
        # Review finding 2: the hash alone records what the merge imported, not what its inputs received.
        assert any("clock" in error for error in tree_errors(block(clock_provenance=provenance)))

    def test_a_tree_whose_stage_3_input_was_never_re_run_fails(self):
        stale = dict(block()["clock_provenance"], mismatches={"FIN_PET": ["FIN_PET/env: no clock stamp"]})
        assert any("FIN_PET" in error for error in tree_errors(block(clock_provenance=stale)))

    def test_an_interrupted_merge_fails(self):
        assert any("complete" in error for error in tree_errors(block(complete=False)))

    @pytest.mark.parametrize("kt", [None, {}, {"by_site": None}])
    def test_a_block_without_kt_counts_fails(self, kt):
        assert any("kt_exceedance" in error for error in tree_errors(block(kt_exceedance=kt)))

    def test_a_total_above_the_baseline_fails(self):
        by_site = dict(KT_RESIDUAL_SITES, FIN_HYY_SME=KT_RESIDUAL_SITES["FIN_HYY_SME"] + 1)
        assert tree_errors(block(kt_exceedance={"by_site": by_site}))

    def test_a_new_site_fails(self):
        errors = tree_errors(block(kt_exceedance={"by_site": {"USA_TNO": 1}}))
        assert any("USA_TNO" in error for error in errors)

    def test_every_reason_is_reported_not_only_the_first(self):
        old = block(complete=False, clock_registry_sha256=None)
        del old["env_daily_rule"]
        assert len(tree_errors(old)) >= 3


class TestReadBlock:
    def test_the_sidecar_is_read_when_there_is_no_provenance(self, tmp_path):
        assert read_block(write(tmp_path, block())) == block()

    def test_a_provenance_block_outranks_the_sidecar(self, tmp_path):
        write(tmp_path, block())
        write(tmp_path, {PROVENANCE_KEY: block(complete=False)}, PROVENANCE_FILE)
        assert read_block(tmp_path)["complete"] is False

    def test_a_provenance_file_without_a_block_falls_back_to_the_sidecar(self, tmp_path):
        write(tmp_path, block())
        write(tmp_path, {"sweep": "sweep_x"}, PROVENANCE_FILE)
        assert read_block(tmp_path) == block()

    def test_nothing_on_disk_reads_none(self, tmp_path):
        assert read_block(tmp_path) is None


class TestMain:
    def test_a_passing_tree_exits_zero_with_a_pass_line(self, tmp_path, capsys):
        assert main([str(write(tmp_path, block()))]) == 0
        assert "PASS" in capsys.readouterr().out

    def test_a_failing_tree_exits_one_and_names_every_reason(self, tmp_path, capsys):
        old = block(complete=False)
        del old["env_daily_rule"]
        assert main([str(write(tmp_path, old))]) == 1
        out = capsys.readouterr().out
        assert "env_daily_rule" in out and "complete" in out

    def test_a_provenance_file_that_is_not_json_fails_with_a_line_not_a_traceback(self, tmp_path, capsys):
        (tmp_path / PROVENANCE_FILE).write_text("{ this is not json")
        assert main([str(tmp_path)]) == 1
        printed = capsys.readouterr().out
        assert "ENV DAILY GATE FAIL" in printed
        assert "Traceback" not in printed

    def test_it_runs_as_a_module_from_the_repository_root(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "src.env_daily.gate", str(tmp_path)],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 1, result.stderr
        assert "FAIL" in result.stdout
