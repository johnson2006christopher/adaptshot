"""Smoke tests for the Tambua application package.

These exist to catch the failure mode a move like this invites: the package
imports fine from the repository root, where the old layout happened to be on
`sys.path`, and fails everywhere else. Each test here asserts something that is
only true once the package is genuinely installed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parents[1]


def test_package_imports_without_path_manipulation() -> None:
    """`tambua` must resolve as an installed package, from any directory.

    The previous version inserted the repository root into `sys.path` at import
    time, so it only worked when launched from one place. Running this in a
    subprocess with a different working directory is what makes the test real --
    inside pytest the repository root is already importable.
    """

    pytest.importorskip("tambua", reason="the application is not installed in this environment")

    result = subprocess.run(
        [sys.executable, "-c", "import tambua; print(tambua.__file__)"],
        cwd=Path(sys.prefix),  # deliberately not the repository
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "`import tambua` failed outside the repository, which means it is still "
        f"resolving by path rather than by installation:\n{result.stderr}"
    )
    assert "src/tambua" in result.stdout or "tambua" in result.stdout


def test_app_does_not_manipulate_sys_path() -> None:
    """No module in the package may edit `sys.path`. That is #11's lesson."""

    offenders = [
        f"{path.relative_to(APP_ROOT)}:{n}"
        for path in (APP_ROOT / "src").rglob("*.py")
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "sys.path" in line and not line.lstrip().startswith("#")
    ]
    assert not offenders, "sys.path manipulation in an installed package:\n  " + "\n  ".join(offenders)


def test_console_script_entry_point_resolves() -> None:
    """`tambua = "tambua.app:launch"` must point at something callable.

    A broken entry point is invisible until someone runs the installed command,
    which is the worst moment to discover it.
    """

    pytest.importorskip("tambua", reason="the application is not installed in this environment")
    pytest.importorskip("gradio", reason="app.py imports gradio at module scope")
    from tambua.app import launch

    assert callable(launch)


def test_flagship_config_is_present_and_parses() -> None:
    """The maize config ships with the app and is valid YAML."""

    yaml = pytest.importorskip("yaml")
    config_path = APP_ROOT / "configs" / "maize.yaml"
    assert config_path.is_file(), f"missing flagship config: {config_path}"

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["application"]["name"] == "MziziGuard"
    assert "maize" in config["crops"], "the flagship config must define the maize domain"


def test_synthetic_data_is_labelled_as_synthetic() -> None:
    """The generated-data disclosure must live in the code, not only in the docs.

    A reader who opens `data.py` before the documentation must still learn that
    these are drawn shapes rather than photographs (#17).
    """

    source = (APP_ROOT / "src" / "tambua" / "data.py").read_text(encoding="utf-8")
    assert "synthetic" in source.lower(), "data.py must say that its images are synthetic"
