"""Guard that ``mypy --strict`` actually inspects the library (#16).

Some mypy errors are fatal to the *run*, not just to the file they appear in. A
syntax error inside a stub is one: mypy reports it and stops, and the failure
looks exactly like any other error. So a type-check step can sit in front of a
check that examined nothing at all.

That is what happened here. numpy's bundled stubs use PEP 695 ``type X = ...``
syntax, which mypy cannot parse while analysing at 3.10, so it aborted with

    Found 3 errors in 3 files (errors prevented further checking)

before analysing a single one of the package's modules. Only one of those three
errors was fatal -- the other two were unresolved ``gradio`` imports, which mypy
reports and then carries on past. The distinction matters: counting errors would
not have told anyone the run was empty, and the two harmless ones made the fatal
one look like a third of the problem.

Counting errors is therefore not enough. This module asserts on the *denominator*
-- the number of files mypy says it checked -- so that an optional dependency
added tomorrow cannot quietly reopen the hole.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "adaptshot"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

# "Success: no issues found in 32 source files" / "Found 8 errors ... (checked 32 source files)"
_CHECKED_RE = re.compile(r"(?:no issues found in|checked)\s+(\d+)\s+source files?")


def _module_count() -> int:
    """Number of Python modules mypy is expected to analyse."""

    return len(list(PACKAGE_ROOT.rglob("*.py")))


def _run_mypy() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "mypy", str(PACKAGE_ROOT), "--strict"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _mypy_target_version() -> tuple[int, ...]:
    """The Python version mypy is configured to analyse for."""

    text = PYPROJECT_PATH.read_text(encoding="utf-8")
    match = re.search(r'^python_version\s*=\s*"(\d+)\.(\d+)"', text, flags=re.MULTILINE)
    assert match is not None, "no mypy `python_version` found in pyproject.toml"
    return (int(match.group(1)), int(match.group(2)))


@pytest.fixture(scope="module")
def mypy_result() -> subprocess.CompletedProcess[str]:
    """Run mypy once for the module; skip if the environment cannot support it."""

    pytest.importorskip("mypy", reason="mypy is only installed with the dev extra")
    # The lint environment installs the torch extra precisely so that mypy can
    # resolve `import torch` against real stubs. Without it mypy stops at that
    # import, which is the very condition this module exists to detect -- so the
    # assertion would be true but meaningless. Skip rather than fail.
    pytest.importorskip("torch", reason="mypy needs the torch extra to resolve imports")

    # A type check is only defined relative to one environment. Different Pythons
    # resolve different third-party wheels, and their stubs disagree: on 3.10 pip
    # installs numpy 2.2, whose `ndarray` is generic with no PEP 696 defaults, so
    # every bare `np.ndarray` annotation in the package becomes a `type-arg` error
    # -- 154 of them. On 3.12+ pip installs numpy 2.5, where it does not.
    #
    # Neither verdict is wrong; they are answers to different questions. Running
    # mypy once per matrix entry therefore measures numpy's stubs, not our code.
    # The lint job is the single authority, so pin this to the version it uses.
    # (The bare-`np.ndarray` weakness numpy 2.2 exposes is real, and is tracked
    # in #44 -- it is the same imprecision behind the overload bug in
    # `core/uncertainty.py`. When that is fixed this skip can go.)
    # `>=` rather than `==` so the check still runs during local development on a
    # newer interpreter. That is an empirical claim -- that stub sets at or above
    # the target agree -- and if a future numpy breaks it, this failing is the
    # signal we want, not noise.
    target = _mypy_target_version()
    if sys.version_info[:2] < target:
        pytest.skip(
            f"mypy's verdict is defined against Python {target[0]}.{target[1]} "
            f"or newer (pyproject `python_version`); running under "
            f"{sys.version_info.major}.{sys.version_info.minor}"
        )
    return _run_mypy()


def test_mypy_reaches_every_module(mypy_result: subprocess.CompletedProcess[str]) -> None:
    """mypy must report checking every module in the package, not abort early."""

    output = mypy_result.stdout + mypy_result.stderr
    match = _CHECKED_RE.search(output)

    assert match is not None, (
        "mypy did not report how many files it checked, which means it aborted "
        "before analysing the package. This is #16 recurring.\n\n"
        f"{output.strip()}"
    )

    checked = int(match.group(1))
    expected = _module_count()
    assert checked == expected, (
        f"mypy checked {checked} files but the package contains {expected}. "
        "An unresolved import is cutting the run short; add an "
        "`ignore_missing_imports` override for the optional dependency, or "
        "install it in the environment that runs mypy.\n\n"
        f"{output.strip()}"
    )


def test_mypy_strict_passes(mypy_result: subprocess.CompletedProcess[str]) -> None:
    """The package must be clean under `mypy --strict`."""

    assert mypy_result.returncode == 0, (
        "mypy --strict reported errors:\n\n"
        f"{(mypy_result.stdout + mypy_result.stderr).strip()}"
    )
