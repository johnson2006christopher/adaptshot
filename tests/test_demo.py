"""The conference demo is run, under the conditions it will face (#27).

A demo that fails on stage undercuts the argument the project exists to make,
so it is exercised here the way it will be exercised there: a fresh process,
torch blocked at the import system, the network blocked, on CPU, within the
two-minute bound. And once more with the benchmark artifact deliberately
missing, because "fails visibly and gracefully" is a requirement, not a hope.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO = REPO_ROOT / "examples" / "demo" / "demo.py"

_BLOCK_TORCH = """
import sys


class _Blocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError("torch is blocked: the demo must run on a core install")
        return None


sys.meta_path.insert(0, _Blocker())
sys.argv = [%r, "--no-color"]
import runpy
runpy.run_path(%r, run_name="__main__")
"""


def _run(env_extra: dict[str, str] | None = None) -> tuple[subprocess.CompletedProcess[str], float]:
    import os

    env = dict(os.environ)
    env.update(env_extra or {})
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-c", _BLOCK_TORCH % (str(DEMO), str(DEMO))],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
        env=env,
    )
    return completed, time.perf_counter() - started


def test_demo_runs_offline_without_torch_within_two_minutes() -> None:
    completed, elapsed = _run()
    assert completed.returncode == 0, f"demo exited {completed.returncode}:\n{completed.stderr[-2000:]}"
    assert elapsed < 120, f"demo took {elapsed:.0f}s; the bound is two minutes"

    out = completed.stdout
    assert "network: blocked" in out, "the demo must state that it enforces its own offline claim"
    assert "empirical coverage" in out, "the measured coverage figure was not displayed"
    assert "ask a human" in out, "no abstention was shown; the demo's point is the refusal"
    assert "size 2" in out or "size 3" in out, "no prediction set wider than one was shown"
    assert "Traceback" not in completed.stderr


def test_demo_degrades_visibly_when_the_artifact_is_missing() -> None:
    """Without the benchmark file the demo must still finish, and say what is missing."""

    completed, _ = _run({"ADAPTSHOT_DEMO_RESULTS": str(REPO_ROOT / "does-not-exist.json")})
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "coverage artifact not found" in completed.stdout
    assert "run_plantvillage" in completed.stdout, "the message must say how to produce it"
    assert "Traceback" not in completed.stderr
