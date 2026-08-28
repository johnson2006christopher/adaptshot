"""The README quickstart is executed, not trusted (#28).

A quickstart that no longer runs is worse than none: it is the first thing an
unfamiliar person tries, and when it fails they leave without reporting it.
This test takes the block out of README.md itself -- not a copy -- and runs it
in a fresh interpreter with two things blocked at the import system:

- ``torch``, so it proves the core install is enough;
- outbound sockets, so it proves nothing is downloaded.

If either is needed, this fails here rather than on someone's laptop.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"

_BLOCK_TORCH_AND_NETWORK = """
import socket
import sys


class _Blocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError("torch is blocked: the quickstart must run on a core install")
        return None


sys.meta_path.insert(0, _Blocker())


def _no_network(*args, **kwargs):
    raise OSError("network is blocked: the quickstart must not download anything")


socket.socket.connect = _no_network
socket.create_connection = _no_network
"""


def quickstart_block() -> str:
    """The first Python block after the Quick start heading, verbatim."""

    text = README.read_text(encoding="utf-8")
    section = text[text.index("## Quick start") :]
    match = re.search(r"```python\n(.*?)```", section, flags=re.DOTALL)
    assert match is not None, "README has no python block under Quick start"
    return match.group(1)


def test_quickstart_is_at_most_ten_lines() -> None:
    """The issue's bound. Comments and blanks do not count against it."""

    code = [line for line in quickstart_block().splitlines() if line.strip() and not line.strip().startswith("#")]
    assert len(code) <= 10, f"quickstart is {len(code)} lines of code; the bound is 10"


def test_quickstart_runs_unedited_without_torch_or_network() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _BLOCK_TORCH_AND_NETWORK + quickstart_block()],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        "the README quickstart failed on a core install with the network blocked:\n\n"
        f"{result.stderr[-2000:]}"
    )
    first_word = result.stdout.split()[0] if result.stdout.split() else ""
    assert first_word in {"gray_leaf_spot", "healthy_maize", "northern_leaf_blight"}, (
        f"quickstart printed {result.stdout!r}; expected a class label first"
    )
