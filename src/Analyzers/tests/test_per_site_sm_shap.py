"""Tests for per_site_sm_shap."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "src" / "Analyzers" / "per_site_sm_shap.py"


def test_cli_help_runs_without_error() -> None:
    """--help must exit 0 and print the --sm-variant flag."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--sm-variant" in result.stdout
    assert "--n-jobs" in result.stdout
    assert "--min-rows" in result.stdout
