"""Guards the import convention: the suite must exercise the *installed* package.

Importing ``src.adaptshot`` and ``adaptshot`` in the same interpreter loads the
same source files twice, as two unrelated module objects. Classes defined in one
copy are not identical to those in the other, so ``isinstance`` and ``except``
silently fail across the boundary, and module-level state exists twice.

That is not a style preference — it means the tests exercise a second copy of the
library rather than the one users import. See issue #11.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# `apps` holds applications built on the library. They are the most likely place
# for a path-based import to creep back in, because an app run from the repo root
# appears to work while being broken for everyone who installs it.
SCANNED_DIRS = ("tests", "benchmarks", "src", "apps", "examples")

# Matches `import src.adaptshot` / `from src.adaptshot...`, and the dotted form
# used in monkeypatch and mock.patch targets, e.g. "src.adaptshot.core.learner.x".
_SRC_PREFIX = re.compile(r"\bsrc\.adaptshot\b")


def _python_files() -> list[Path]:
    return [
        path
        for directory in SCANNED_DIRS
        for path in sorted((REPO_ROOT / directory).rglob("*.py"))
    ]


def test_no_source_tree_import_prefix() -> None:
    """No Python file may reference the library as ``src.adaptshot``."""

    offenders = []
    for path in _python_files():
        if path == Path(__file__):
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _SRC_PREFIX.search(line):
                rel = path.relative_to(REPO_ROOT)
                offenders.append(f"{rel}:{lineno}: {line.strip()}")

    assert not offenders, (
        "Import the installed package as `adaptshot`, not the source tree as "
        "`src.adaptshot` — the two load as separate modules. Offending lines:\n  "
        + "\n  ".join(offenders)
    )


def test_library_is_loaded_once() -> None:
    """Only one copy of the library may be present in ``sys.modules``."""

    import adaptshot  # noqa: F401

    duplicates = sorted(name for name in sys.modules if name.startswith("src.adaptshot"))
    assert not duplicates, (
        "The library is loaded twice under different names: "
        f"{duplicates}. Every import must go through `adaptshot`."
    )


def test_package_resolves_outside_the_source_tree_convention() -> None:
    """``adaptshot`` must import without the repo root being on ``sys.path``.

    This is what the ``src/`` layout buys: the import resolves through the
    installed distribution, so a file missing from the wheel fails here rather
    than passing because it happened to be sitting in the source tree.
    """

    import adaptshot

    assert adaptshot.__name__ == "adaptshot"
    assert not adaptshot.__name__.startswith("src.")
