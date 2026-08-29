"""The library must not ship a web interface (#22).

AdaptShot shipped two Gradio dashboards at once -- `adaptshot.ui.app` and
`adaptshot.studio` -- which is one more than the maximum. `adaptshot.ui.app` is
removed; `studio/` is being extracted to its own repository (#21).

The reason is not tidiness. AdaptShot's premise is that it runs on a CPU, in a
field, with no internet. A web framework in the dependency graph of a library
making that claim is a contradiction someone will eventually notice, and the
answer should be that we noticed first.

This module is the ratchet: it fails if a GUI reappears anywhere inside
`src/adaptshot/`. It carries no exemptions.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "adaptshot"

def _package_modules() -> list[Path]:
    """Every module in the library. No exemptions remain.

    There used to be one, for `studio/`, written to expire. #21 extracted it, so
    it is gone and this test now covers the whole package.
    """

    return list(PACKAGE_ROOT.rglob("*.py"))


def test_the_removed_interfaces_are_gone() -> None:
    assert not (PACKAGE_ROOT / "ui").exists(), (
        "src/adaptshot/ui/ is back. The interface belongs in apps/tambua/, "
        "which is its own distribution (#22)."
    )
    assert not (PACKAGE_ROOT / "studio").exists(), (
        "src/adaptshot/studio/ is back. Its history lives on the "
        "`studio-extract` branch and belongs in its own repository (#21)."
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


def test_the_gui_extras_are_gone() -> None:
    """`ui` existed only for the deleted module; `gui` only for studio.

    Read with a regex rather than `tomllib`, which is 3.11+ while this project
    supports 3.10. `test_release_metadata.py` reads pyproject the same way for
    the same reason.
    """

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for extra in ("ui", "gui"):
        assert not re.search(rf"^{extra}\s*=\s*\[", pyproject, flags=re.MULTILINE), (
            f"the `{extra}` extra is back. The library ships no GUI; the "
            "application is a separate distribution"
        )


def test_no_documentation_teaches_a_removed_entrypoint() -> None:
    """A page may *mention* a removed interface. It may not *teach* it.

    Naming `adaptshot.ui.app` while explaining that it was removed is the
    documentation doing its job -- several pages exist for exactly that, so
    anyone arriving from a search engine is told where the interface went.

    An allowlist of filenames would have to grow every time such a page is
    added, and would stop catching anything. The rule is about content instead:
    a page that names a removed entrypoint must also say it is gone.
    """

    removed = ("adaptshot.ui", "adaptshot[ui]", "adaptshot.studio", "adaptshot[gui]")
    says_it_is_gone = ("removed", "extracted", "moved out", "no longer", "gone")

    offenders = []
    for path in (REPO_ROOT / "docs").rglob("*.md"):
        relative = str(path.relative_to(REPO_ROOT))
        if "archive/" in relative:
            continue
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        if not any(name in text for name in removed):
            continue
        if any(phrase in lowered for phrase in says_it_is_gone):
            continue
        offenders.append(relative)

    assert not offenders, (
        "these documents name a removed interface without saying it was "
        "removed, so a reader would try to use it:\n  " + "\n  ".join(offenders)
    )
