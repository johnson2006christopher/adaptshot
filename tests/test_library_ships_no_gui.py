"""The library must not ship a web interface (#22).

AdaptShot shipped two Gradio dashboards at once -- `adaptshot.ui.app` and
`adaptshot.studio` -- which is one more than the maximum. `adaptshot.ui.app` is
removed; `studio/` is being extracted to its own repository (#21).

The reason is not tidiness. AdaptShot's premise is that it runs on a CPU, in a
field, with no internet. A web framework in the dependency graph of a library
making that claim is a contradiction someone will eventually notice, and the
answer should be that we noticed first.

This module is the ratchet: it fails if a GUI reappears inside `src/adaptshot/`,
and it will fail on `studio/` too the moment #21 lands, which is intended -- the
exemption below is written to expire.
"""

from __future__ import annotations

from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "adaptshot"

#: Being extracted in #21. Delete this line when it lands; the test then covers
#: the whole package and no further change is needed.
PENDING_EXTRACTION = ("studio",)


def _package_modules() -> list[Path]:
    return [
        path
        for path in PACKAGE_ROOT.rglob("*.py")
        if not any(part in PENDING_EXTRACTION for part in path.relative_to(PACKAGE_ROOT).parts)
    ]


def test_the_removed_interface_is_gone() -> None:
    assert not (PACKAGE_ROOT / "ui").exists(), (
        "src/adaptshot/ui/ is back. The interface belongs in apps/tambua/, "
        "which is its own distribution."
    )


def test_no_module_imports_gradio() -> None:
    """A library that needs a web framework to be imported is not a library."""

    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{n}"
        for path in _package_modules()
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if line.startswith(("import gradio", "from gradio"))
    ]
    assert not offenders, "gradio imported inside the library:\n  " + "\n  ".join(offenders)


def test_the_ui_extra_is_gone() -> None:
    """It duplicated part of `gui` and existed only for the deleted module."""

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]
    assert "ui" not in extras, (
        "the `ui` extra is back; `gui` already covers the studio, and the "
        "application is a separate distribution"
    )


def test_no_documentation_points_at_the_removed_entrypoint() -> None:
    """A tutorial that teaches a deleted entrypoint is worse than no tutorial.

    The one page allowed to name it is the page that explains the removal.
    """

    allowed = {"docs/tutorials/11_ui_pilot_dashboard.md"}
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{n}"
        for path in (REPO_ROOT / "docs").rglob("*.md")
        if str(path.relative_to(REPO_ROOT)) not in allowed
        and "archive/" not in str(path.relative_to(REPO_ROOT))
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "adaptshot.ui" in line or "adaptshot[ui]" in line
    ]
    assert not offenders, (
        "documentation still points at the removed interface:\n  " + "\n  ".join(offenders)
    )
