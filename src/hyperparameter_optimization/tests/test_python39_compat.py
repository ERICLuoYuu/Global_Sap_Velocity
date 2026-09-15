"""Repo-wide gate: no 3.10+-only syntax may reach the cluster.

Palma runs Python 3.9.25. ``.ruff.toml`` declares ``target-version = "py310"``,
which switches ON ruff's ``B905`` (``zip-without-explicit-strict``) -- so the
linter actively rewrites plain ``zip(a, b)`` into ``zip(a, b, strict=False)``,
which is a hard ``TypeError`` on 3.9. Ten files across this repo carry hand-
written comments or ``# noqa: B905`` markers put there to stop exactly that,
and the rewrite still landed in seven more places, including the
``site_to_group`` line in the DA and prediction trainers -- the statement
immediately after spatial grouping, so both crashed before fitting anything.

A comment is not a gate. This test is.
"""

import ast
from pathlib import Path

import pytest

#: Repo root, from this file at src/hyperparameter_optimization/tests/.
REPO_ROOT = Path(__file__).resolve().parents[3]

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".ruff_cache", ".pytest_cache"}


def _python_files() -> list:
    """Every tracked-looking .py file, minus vendored and cache trees."""
    out = []
    for path in REPO_ROOT.rglob("*.py"):
        if SKIP_DIRS & set(path.parts) or any(part.startswith(".venv") for part in path.parts):
            continue
        if path.suffix != ".py" or path.name.endswith(".pre_provenance"):
            continue
        out.append(path)
    return sorted(out)


def find_zip_strict(source: str) -> list:
    """Line numbers of every ``zip(...)`` call carrying a ``strict=`` keyword.

    Parses rather than greps: the ten defensive comments in this repo all
    contain the literal text ``zip(strict=``, so a grep-based gate would report
    the warnings as violations and the real calls would drown in them.
    """
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "zip"):
            continue
        if any(kw.arg == "strict" for kw in node.keywords):
            found.append(node.lineno)
    return found


class TestNoZipStrict:
    """``zip(..., strict=...)`` is a TypeError on the deployment interpreter."""

    def test_no_python_file_calls_zip_with_strict(self) -> None:
        offenders = []
        for path in _python_files():
            try:
                source = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            try:
                lines = find_zip_strict(source)
            except SyntaxError:
                continue  # not parseable under this interpreter; not our concern
            for lineno in lines:
                offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{lineno}")
        assert offenders == [], (  # nosec B101
            "zip(..., strict=...) is Python 3.10+ and raises TypeError on Palma (3.9.25). "
            "Drop the keyword and mark the line '# noqa: B905' so ruff does not re-add it:\n  "
            + "\n  ".join(offenders)
        )

    def test_the_scan_actually_reaches_the_trainers(self) -> None:
        """An empty or misrooted file list would make the gate above vacuous."""
        names = {p.name for p in _python_files()}
        assert "test_hyperparameter_tuning_ML_spatial_stratified_DA.py" in names  # nosec B101
        assert "preflight_audit.py" in names  # nosec B101
        assert len(_python_files()) > 200  # nosec B101

    def test_detector_fires_on_strict_false(self) -> None:
        """Negative control: strict=False is just as fatal as strict=True."""
        assert find_zip_strict("d = dict(zip(a, b, strict=False))") == [1]  # nosec B101

    def test_detector_fires_on_strict_true(self) -> None:
        assert find_zip_strict("for x, y in zip(a, b, strict=True):\n    pass") == [1]  # nosec B101

    def test_detector_ignores_a_bare_zip(self) -> None:
        assert find_zip_strict("d = dict(zip(a, b))  # noqa: B905") == []  # nosec B101

    def test_detector_ignores_a_comment_about_the_trap(self) -> None:
        """The ten defensive comments must not be reported as violations."""
        source = "# No zip(strict=): Palma runs Python 3.9.\nrows = list(zip(a, b))\n"
        assert find_zip_strict(source) == []  # nosec B101

    def test_detector_ignores_strict_on_something_that_is_not_zip(self) -> None:
        """Only the builtin zip is 3.10-gated; np.strict_whatever is unrelated."""
        assert find_zip_strict("parse(text, strict=True)") == []  # nosec B101


@pytest.mark.parametrize(
    "path",
    [
        "src/hyperparameter_optimization/test_hyperparameter_tuning_ML_spatial_stratified_DA.py",
        "src/hyperparameter_optimization/test_hyperparameter_tuning_ML_spatial_stratified_prediction.py",
    ],
    ids=["DA", "prediction"],
)
def test_site_to_group_survives_python39(path: str) -> None:
    """The two lines that crashed both trainers before a model was ever fitted.

    Named explicitly, not just covered by the sweep: these are the regression,
    and a future sweep-narrowing must not quietly stop checking them.
    """
    full = REPO_ROOT / path
    if not full.exists():
        pytest.skip(f"{path} not present in this checkout")
    source = full.read_text(encoding="utf-8")
    assert "zip(site_ids, spatial_groups, strict=" not in source  # nosec B101
    assert "zip(site_ids, spatial_groups)" in source  # nosec B101
