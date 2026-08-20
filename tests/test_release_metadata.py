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

from src.adaptshot import __version__

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
