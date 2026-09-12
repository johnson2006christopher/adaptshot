"""The library core must not depend on a web interface (#22, re-drawn by #102).

AdaptShot's premise is that it runs on a CPU, in a field, with no internet. A
web framework in the dependency graph of a *core install* contradicts that, and
the answer should be that we noticed first.

Since v0.3.1 the Tambua application lives inside the package as
``adaptshot.app`` behind the ``app`` extra (#102), so the line this module
holds has moved but not softened:

- ``pip install adaptshot`` stays numpy + Pillow + onnxruntime, and
  ``import adaptshot`` must not touch ``adaptshot.app`` or gradio;
- exactly one module, ``adaptshot/app/ui.py``, may import gradio, and nothing
  imports it at module scope except the lazy import inside the ``tambua`` CLI;
- the retired ``ui`` / ``gui`` extras stay gone.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "adaptshot"

#: The one module allowed to import gradio, as a repo-relative posix path.
GRADIO_BOUNDARY = "src/adaptshot/app/ui.py"


def test_the_removed_interfaces_are_gone() -> None:
    assert not (PACKAGE_ROOT / "ui").exists(), (
        "src/adaptshot/ui/ is back. The interface lives in adaptshot/app/ui.py "
        "behind the `app` extra (#102); a second one is one too many (#22)."
    )
    assert not (PACKAGE_ROOT / "studio").exists(), (
        "src/adaptshot/studio/ is back. Its history lives on the "
        "`studio-extract` branch and belongs in its own repository (#21)."
    )


def test_only_the_ui_module_imports_gradio() -> None:
    """One module holds the gradio boundary; everywhere else stays clean.

    Asserted by path rather than by absence: the invariant is not "gradio is
    rarely imported", it is "the import lives in exactly one place, and this is
    the place".
    """

    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{n}"
        for path in PACKAGE_ROOT.rglob("*.py")
        if path.relative_to(REPO_ROOT).as_posix() != GRADIO_BOUNDARY
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if line.startswith(("import gradio", "from gradio"))
    ]
    assert not offenders, (
        "gradio imported outside the one permitted module "
        f"({GRADIO_BOUNDARY}):\n  " + "\n  ".join(offenders)
    )

    boundary = REPO_ROOT / GRADIO_BOUNDARY
    assert boundary.is_file(), (
        f"{GRADIO_BOUNDARY} is gone. If the UI moved, move this constant with "
        "it; if the app was removed, remove the `app` extra and the `tambua` "
        "script too."
    )


def test_importing_the_library_does_not_load_the_app_or_gradio() -> None:
    """`import adaptshot` must cost a core user nothing app-shaped.

    Run in a subprocess so this process's own imports cannot contaminate the
    measurement. The gradio half of the assertion is vacuous on a core install
    (nothing absent can be imported), and real in every CI job where the app
    extra is installed -- which is exactly where a regression would appear.
    """

    code = (
        "import json, sys\n"
        "import adaptshot\n"
        "loaded = sorted(m for m in sys.modules if m == 'gradio' "
        "or m.startswith('gradio.') or m == 'adaptshot.app' "
        "or m.startswith('adaptshot.app.'))\n"
        "print(json.dumps(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=120, check=True,
    )
    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    assert loaded == [], (
        "importing the library dragged in the app or the web framework: "
        f"{loaded}. The app is an optional extra; the core import path must "
        "not pay for it."
    )


def test_the_cli_module_is_importable_without_gradio() -> None:
    """The `tambua` entry point lands on every install, extras or not.

    pip writes console scripts unconditionally, so `adaptshot.app.cli` must
    import on a core install -- if it pulled gradio in at module scope, a core
    user typing `tambua --help` would get a traceback instead of help text.
    Import it here and assert gradio was not the price.
    """

    code = (
        "import sys\n"
        "import adaptshot.app.cli\n"
        "assert 'gradio' not in sys.modules, 'cli imported gradio at module scope'\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert result.returncode == 0, (
        f"importing adaptshot.app.cli failed or loaded gradio:\n{result.stderr}"
    )


def test_the_gui_extras_are_gone() -> None:
    """`ui` existed only for the deleted module; `gui` only for studio.

    The application's extra is `app`, deliberately not a revival of either
    name. Read with a regex rather than `tomllib`, which is 3.11+ while this
    project supports 3.10. `test_release_metadata.py` reads pyproject the same
    way for the same reason.
    """

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for extra in ("ui", "gui"):
        assert not re.search(rf"^{extra}\s*=\s*\[", pyproject, flags=re.MULTILINE), (
            f"the `{extra}` extra is back. The application's extra is `app`; "
            "the retired names stay retired"
        )


def test_gradio_is_not_a_core_dependency() -> None:
    """gradio may appear under the `app` extra and nowhere else in [project]."""

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    dependencies = re.search(
        r"^dependencies\s*=\s*\[(.*?)\]", pyproject, flags=re.MULTILINE | re.DOTALL
    )
    assert dependencies is not None, "no [project] dependencies block found"
    assert "gradio" not in dependencies.group(1), (
        "gradio is a core dependency. It belongs under the `app` extra: a "
        "library claiming to run offline on a CPU cannot require a web "
        "framework to install."
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
