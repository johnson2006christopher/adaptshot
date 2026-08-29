"""check_environment() reports what this machine can do, measured here (#38)."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import adaptshot

_BLOCKED = """
import socket, sys
class _B:
    def find_spec(self, n, p=None, t=None):
        if n == "torch" or n.startswith("torch."): raise ImportError("blocked")
        return None
sys.meta_path.insert(0, _B())
def _no(*a, **k): raise OSError("network blocked")
socket.socket.connect = _no; socket.create_connection = _no
"""


def test_report_is_measured_on_this_machine() -> None:
    report = adaptshot.check_environment()
    text = str(report)
    assert "measured here" in text
    assert report.latency_ms_median is not None and report.latency_ms_median > 0
    assert report.latency_backbone in report.bundled_backbones
    assert report.peak_rss_mb is not None and report.peak_rss_mb > 10
    assert "mobilenet_v3_small" in text
    assert "not selected" in text or report.gpu is None


def test_availability_only_path_is_fast_and_measures_nothing() -> None:
    import time

    started = time.perf_counter()
    report = adaptshot.check_environment(measure=False)
    assert time.perf_counter() - started < 0.5
    assert report.latency_ms_median is None and report.peak_rss_mb is None
    assert any(cap.name.startswith("predict") for cap in report.capabilities)


def test_on_a_core_install_it_says_what_needs_torch_and_meets_the_target() -> None:
    """Torch and the network blocked: the report must still measure, name what is
    missing with the exact command, and find the process under 250 MB."""

    script = _BLOCKED + textwrap.dedent(
        """
        import adaptshot
        r = adaptshot.check_environment()
        text = str(r)
        assert r.latency_ms_median is not None, "no latency measured"
        assert not r.torch_loaded_in_process
        assert r.meets_memory_target is True, f"peak {r.peak_rss_mb}"
        missing = {c.name: c.install for c in r.capabilities if not c.available}
        assert "fine-tuning (CA-EWC) via correct()" in missing
        assert missing["fine-tuning (CA-EWC) via correct()"] == 'pip install "adaptshot[torch]"'
        assert "not measured" in text, "a download size was quoted, not measured"
        print("OK", round(r.latency_ms_median, 1), round(r.peak_rss_mb))
        """
    )
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr[-1500:]
    assert completed.stdout.startswith("OK")
