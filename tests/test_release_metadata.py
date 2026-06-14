"""Release metadata tests for v0.2.0.post0."""

from __future__ import annotations

from pathlib import Path

from src.adaptshot import __version__


def test_package_version_matches_release() -> None:
    """Ensure the imported package version matches the release metadata."""

    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject_text = pyproject_path.read_text(encoding="utf-8")

    assert __version__ == "0.2.0.post0"
    assert 'version = "0.2.0.post0"' in pyproject_text
