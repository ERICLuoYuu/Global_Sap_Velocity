"""Shared pytest configuration for src/Extractors tests.

Adds the project root to sys.path so tests can import ``src.Extractors.*``
regardless of pytest's invocation directory, and registers local markers
so pytest does not emit PytestUnknownMarkWarning.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast, hermetic unit test")
    config.addinivalue_line("markers", "integration: multi-module integration test")
    config.addinivalue_line(
        "markers",
        "network: requires live network access (USC Santiago THREDDS); gated by FAN2017_LIVE_TESTS=1",
    )
