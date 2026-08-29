"""Smoke tests for the Tambua application package.

These exist to catch the failure mode a move like this invites: the package
imports fine from the repository root, where the old layout happened to be on
`sys.path`, and fails everywhere else. Each test here asserts something that is
only true once the package is genuinely installed.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parents[1]


def _code_lines(path: Path) -> list[tuple[int, str]]:
    """Lines of a module with docstrings and comments removed.

    Prose may name a domain freely -- explaining that swapping the config turns
    maize into solar panels is the documentation doing its job. What matters is
    whether a domain word reaches executable code, so only that is searched.
    """

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    prose: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", [])
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
            and first.end_lineno is not None
        ):
            prose.update(range(first.lineno, first.end_lineno + 1))

    return [
        (n, line.split("#", 1)[0])
        for n, line in enumerate(source.splitlines(), 1)
        if n not in prose
    ]


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


def test_bundled_configs_ship_inside_the_package() -> None:
    """The configs must be package data, not files beside the source tree.

    An installed application whose configs stayed in the repository has nothing
    to run: `pip install tambua` copies the package, not its sibling directories.
    Resolving them through `importlib.resources` is what proves they were built
    into the distribution.
    """

    pytest.importorskip("tambua", reason="the application is not installed in this environment")
    from tambua import bundled_config, load_config

    flagship = load_config(bundled_config("maize"))
    assert flagship.application.name == "MziziGuard"
    assert "maize" in flagship.domains


def test_a_second_domain_ships_and_shares_no_vocabulary() -> None:
    """Generality has to be demonstrated by a second config, not asserted.

    One config proves nothing: any hard-coded assumption would still be
    satisfied by it. Two configs that share no class, no domain and no advice
    are what make the claim falsifiable.
    """

    pytest.importorskip("tambua", reason="the application is not installed in this environment")
    from tambua import bundled_config, load_config

    flagship = load_config(bundled_config("maize"))
    second = load_config(bundled_config("solar_panel"))

    assert second.application.name != flagship.application.name
    assert not set(second.domains) & set(flagship.domains)
    assert not set(second.labels) & set(flagship.labels)


def test_no_domain_vocabulary_is_hard_coded_in_the_application() -> None:
    """No class or domain from either config may appear in the source.

    This is the check behind "the config is the application's identity". If a
    class key can be found in a `.py` file, something is special-casing it, and
    the next config will hit that special case.
    """

    pytest.importorskip("tambua", reason="the application is not installed in this environment")
    from tambua import bundled_config, load_config

    vocabulary = set()
    for name in ("maize", "solar_panel"):
        cfg = load_config(bundled_config(name))
        vocabulary |= set(cfg.labels) | set(cfg.domains)

    # Exactly one line may name a domain: the constant choosing which bundled
    # config loads by default. Pinning it here means adding a second such line
    # fails this test, which is the point.
    permitted = 'DEFAULT_CONFIG = "maize"'

    offenders = [
        f"{path.relative_to(APP_ROOT)}:{n}: {word}"
        for path in (APP_ROOT / "src").rglob("*.py")
        for n, line in _code_lines(path)
        for word in vocabulary
        if word in line and line.strip() != permitted
    ]
    assert not offenders, (
        "domain vocabulary found in application code; it belongs in the config:\n  "
        + "\n  ".join(offenders)
    )


def test_the_package_ships_no_image_generation() -> None:
    """No module in the distribution may draw an image (#53).

    Drawn shapes are not data. A number measured on them is not a result, and
    offering them through the interface as "samples" invites exactly the
    confusion that #17 already cost a release to correct.

    The generator still exists under tests/support/, where deterministic
    licence-free images are the right tool for checking that the pipeline runs.
    This test is what keeps it from drifting back into the product.
    """

    drawing = ("ImageDraw", "Image.new(", "make_placeholder", "generate_samples")
    offenders = [
        f"{path.relative_to(APP_ROOT)}:{n}: {marker}"
        for path in (APP_ROOT / "src").rglob("*.py")
        for n, line in _code_lines(path)
        for marker in drawing
        if marker in line
    ]
    assert not offenders, (
        "the shipped package generates images; it must only read them:\n  "
        + "\n  ".join(offenders)
    )


def test_the_generator_survives_where_it_belongs() -> None:
    """Same class and variant must render identically, and classes must differ.

    Determinism uses `hashlib`, never the builtin `hash()`, whose per-process
    salt would give a different picture every run. Two classes rendering
    identically would make a test that "passes" while measuring nothing.
    """

    pytest.importorskip("tambua", reason="the application is not installed in this environment")
    from support.images import make_placeholder
    from tambua import bundled_config, load_config

    assert make_placeholder("a_class", 2).tobytes() == make_placeholder("a_class", 2).tobytes()

    for name in ("maize", "solar_panel"):
        cfg = load_config(bundled_config(name))
        renders = {key: make_placeholder(key, 0).tobytes() for key in cfg.labels}
        assert len(set(renders.values())) == len(renders), (
            f"two classes in {name}.yaml render to the same image"
        )
