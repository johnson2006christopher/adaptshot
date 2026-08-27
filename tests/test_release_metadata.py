"""Release metadata consistency tests.

The package version is declared in two places that must never drift:
``pyproject.toml`` (build metadata) and ``src/adaptshot/__init__.py`` (runtime
metadata). This module derives the expected value from ``pyproject.toml``
rather than hard-coding it, so a release bump cannot leave the test behind.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from adaptshot import __version__

PYPROJECT_PATH = Path(__file__).resolve().parents[1] / "pyproject.toml"

# PEP 440: N.N.N with optional pre/post/dev suffix (e.g. 0.2.0, 0.3.0rc1, 0.2.0-dev).
_PEP440_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-.]?(?:a|b|rc|alpha|beta|dev|post)\.?\d*)?$")


def _declared_version() -> str:
    """Return the version declared in the ``[project]`` table of pyproject.toml."""

    text = PYPROJECT_PATH.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    assert match is not None, "no `version = \"...\"` found in pyproject.toml"
    return match.group(1)


def test_package_version_matches_pyproject() -> None:
    """The importable ``__version__`` must equal the packaged version."""

    declared = _declared_version()
    assert __version__ == declared, (
        f"version drift: src/adaptshot/__init__.py declares {__version__!r} "
        f"but pyproject.toml declares {declared!r}"
    )


def test_version_is_pep440_compliant() -> None:
    """A malformed version string breaks PyPI uploads; catch it before release."""

    assert _PEP440_RE.match(__version__), f"{__version__!r} is not a valid PEP 440 version"


@pytest.mark.parametrize("required_file", ["README.md", "LICENSE", "CHANGELOG.md"])
def test_distribution_files_present(required_file: str) -> None:
    """Files referenced by the package metadata must exist at the repo root."""

    assert (PYPROJECT_PATH.parent / required_file).is_file(), f"missing {required_file}"


def test_changelog_documents_current_version() -> None:
    """Every released version must have a CHANGELOG entry."""

    changelog = (PYPROJECT_PATH.parent / "CHANGELOG.md").read_text(encoding="utf-8")
    base_version = __version__.split("-")[0]
    assert base_version in changelog, (
        f"CHANGELOG.md has no entry for version {base_version}"
    )


# ---------------------------------------------------------------------------
# Supported Python range (#34)
#
# The range is declared in five places that must agree: `requires-python`, the
# trove classifiers, ruff's `target-version`, mypy's `python_version`, and the
# CI matrix. Nothing linked them, so 3.9 was claimed but the matrix stopped at
# 3.12 and 3.13/3.14 were never tested at all. These tests derive every value
# from its source of truth rather than restating it.
# ---------------------------------------------------------------------------

CI_WORKFLOW_PATH = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


def _classifier_versions() -> list[str]:
    """Python versions declared in the trove classifiers, in file order."""

    text = PYPROJECT_PATH.read_text(encoding="utf-8")
    return re.findall(r'"Programming Language :: Python :: (\d+\.\d+)"', text)


def _ci_matrix_versions() -> list[str]:
    """Python versions in the CI test matrix."""

    text = CI_WORKFLOW_PATH.read_text(encoding="utf-8")
    match = re.search(r"python-version:\s*\[([^\]]+)\]", text)
    assert match is not None, "no `python-version: [...]` matrix found in ci.yml"
    return re.findall(r"\d+\.\d+", match.group(1))


def _requires_python_floor() -> str:
    """The minimum Python version declared by `requires-python`."""

    text = PYPROJECT_PATH.read_text(encoding="utf-8")
    match = re.search(r'^requires-python\s*=\s*">=(\d+\.\d+)"', text, flags=re.MULTILINE)
    assert match is not None, "no `requires-python = \">=X.Y\"` found in pyproject.toml"
    return match.group(1)


def test_classifiers_match_ci_matrix() -> None:
    """Every Python we claim to support must be a Python we actually test."""

    classifiers = _classifier_versions()
    matrix = _ci_matrix_versions()
    assert classifiers == matrix, (
        "declared support does not match tested support:\n"
        f"  classifiers: {classifiers}\n"
        f"  CI matrix:   {matrix}\n"
        "A classifier is a promise -- it must follow the proof, not precede it."
    )


def test_requires_python_matches_lowest_tested_version() -> None:
    """`requires-python` must equal the lowest version in the CI matrix."""

    floor = _requires_python_floor()
    lowest = min(_ci_matrix_versions(), key=lambda v: tuple(int(p) for p in v.split(".")))
    assert floor == lowest, (
        f"requires-python declares >={floor} but the lowest tested version is {lowest}"
    )


def test_ruff_and_mypy_target_the_lowest_supported_version() -> None:
    """Static analysis must run against the oldest Python we claim to support.

    Targeting a newer version silently reduces what is verified: syntax and
    typing valid on 3.12 may not be valid on the floor we promise.
    """

    text = PYPROJECT_PATH.read_text(encoding="utf-8")
    floor = _requires_python_floor()
    expected_ruff = "py" + floor.replace(".", "")

    ruff_target = re.search(r'^target-version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    assert ruff_target is not None, "no ruff `target-version` found"
    assert ruff_target.group(1) == expected_ruff, (
        f"ruff targets {ruff_target.group(1)} but requires-python floor is {floor} "
        f"(expected {expected_ruff})"
    )

    mypy_version = re.search(r'^python_version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    assert mypy_version is not None, "no mypy `python_version` found"
    assert mypy_version.group(1) == floor, (
        f"mypy checks against {mypy_version.group(1)} but requires-python floor is {floor}"
    )
