"""Every tutorial and how-to page runs as written (#39).

A beginner cannot repair a broken example; they conclude they are the problem
and leave. So each page under ``docs/tutorials`` and ``docs/how-to`` is
executed here, on every push: its ``python`` code blocks are concatenated in
order and run in a fresh interpreter, in an empty working directory, with
outbound sockets disabled -- and, unless the page says it needs torch, with
torch blocked at the import system, so the page proves it works on the core
install.

Conventions the pages follow, enforced here:

- The first lines after the title say who the page is for: a blockquote
  starting ``> **For:**``. One document cannot teach a beginner and answer an
  expert; saying which it is doing is the whole technique.
- Only ```python blocks run. Shell commands go in ```bash blocks, which are
  shown and not executed.
- A block containing the comment ``# docs: not run`` is shown and skipped --
  for the rare block that needs a file the page cannot create. Say why in the
  comment.
- A page that needs the torch extra carries ``<!-- needs: torch -->`` near the
  top. It runs with torch allowed and is skipped where torch is absent.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted(
    list((REPO_ROOT / "docs" / "tutorials").glob("*.md"))
    + list((REPO_ROOT / "docs" / "how-to").glob("*.md"))
)

_PREAMBLE = """
import socket, sys
def _refuse(*a, **k):
    raise OSError("the documentation runs offline; something tried to open a connection")
socket.socket.connect = _refuse
socket.create_connection = _refuse
"""
_BLOCK_TORCH = """
class _NoTorch:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is blocked: this page must work on the core install")
        return None
sys.meta_path.insert(0, _NoTorch())
"""


def python_blocks(page: Path) -> list[str]:
    text = page.read_text(encoding="utf-8")
    blocks = re.findall(r"```python[^\n]*\n(.*?)```", text, flags=re.DOTALL)
    return [b for b in blocks if "# docs: not run" not in b]


def needs_torch(page: Path) -> bool:
    return "<!-- needs: torch -->" in page.read_text(encoding="utf-8")


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.relative_to(REPO_ROOT / "docs").as_posix())
def test_page_says_who_it_is_for(page: Path) -> None:
    lines = [line.strip() for line in page.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines and lines[0].startswith("# "), f"{page.name}: must open with a title"
    head = " ".join(lines[1:4])
    assert "> **For:**" in head or "**For:**" in head, (
        f"{page.name}: the lines after the title must say who the page is for (`> **For:** ...`)"
    )


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.relative_to(REPO_ROOT / "docs").as_posix())
def test_page_code_runs_as_written(page: Path) -> None:
    blocks = python_blocks(page)
    if not blocks:
        pytest.skip("no runnable python on this page")
    torch_page = needs_torch(page)
    if torch_page and importlib.util.find_spec("torch") is None:
        pytest.skip("page needs the torch extra, which is not installed here")

    script = _PREAMBLE + ("" if torch_page else _BLOCK_TORCH) + "\n\n".join(blocks)
    with tempfile.TemporaryDirectory() as workdir:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
            cwd=workdir,
            timeout=600,
        )
    assert completed.returncode == 0, (
        f"{page.relative_to(REPO_ROOT)} does not run as written.\n\n"
        f"--- stderr ---\n{completed.stderr[-3000:]}\n--- stdout ---\n{completed.stdout[-1500:]}"
    )
