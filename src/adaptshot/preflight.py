"""What AdaptShot can do on this machine, measured here (#38).

**Experimental.** May change in a minor release without a deprecation cycle; see ``adaptshot.api``.

The library's argument is that a model should say when it does not know. This
is the same idea one level up: a library that says what it cannot do in the
environment it finds itself in, before someone commits an afternoon to finding
out by crashing. For the person this project is built for -- a weak laptop, a
metered connection -- "fine-tuning needs a torch install you do not have" is
worth knowing up front, and an ``ImportError`` halfway through is not the way
to learn it.

Two rules, both from the issue. Every figure is **measured on the machine
running the check**: the latency comes from a real inference on the bundled
photographs, the memory from this process's own high-water mark. Nothing is
quoted from documentation or from another machine. And a GPU, if present, is
**mentioned, not selected**: the project's claims hold on a fixed CPU target,
and auto-selecting a device would dissolve the constraint that makes them
meaningful.

    >>> import adaptshot
    >>> print(adaptshot.check_environment())
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import cast

from .core.extractor import bundled_onnx_backbones

_OPTIONAL = ("torch", "onnxruntime", "faiss", "gradio")


@dataclass
class Capability:
    """**Experimental.** May change in a minor release without a deprecation cycle; see ``adaptshot.api``.

    One thing the library can or cannot do here, and how to change that."""

    name: str
    available: bool
    detail: str
    install: str | None = None


@dataclass
class EnvironmentReport:
    """**Experimental.** May change in a minor release without a deprecation cycle; see ``adaptshot.api``.

    Everything ``check_environment`` measured. ``str()`` renders it."""

    python: str
    platform: str
    machine: str
    cpu_count: int | None
    ram_total_mb: float | None
    ram_available_mb: float | None
    dependencies: dict[str, str | None]
    bundled_backbones: list[str]
    capabilities: list[Capability]
    latency_ms_median: float | None
    latency_ms_p95: float | None
    latency_backbone: str | None
    peak_rss_mb: float | None
    torch_loaded_in_process: bool
    gpu: str | None
    notes: list[str] = field(default_factory=list)

    @property
    def meets_memory_target(self) -> bool | None:
        """Whether this process peaked under 250 MB. None if it could not be measured.

        The target is a claim about a core-install process. If torch is loaded
        in this one, the number describes that, and ``notes`` says so.
        """

        if self.peak_rss_mb is None:
            return None
        return self.peak_rss_mb < 250.0

    def __str__(self) -> str:
        from . import __version__

        lines = [f"AdaptShot {__version__} -- environment report (everything below was measured here)"]
        ram = (
            f"{self.ram_available_mb / 1024:.1f} of {self.ram_total_mb / 1024:.1f} GB RAM free"
            if self.ram_total_mb and self.ram_available_mb
            else "RAM: not readable on this platform"
        )
        lines.append(f"  Python {self.python} · {self.platform} · {self.machine} · {self.cpu_count or '?'} cores · {ram}")
        deps = "   ".join(
            f"{name} {version}" if version else f"{name}: not installed" for name, version in self.dependencies.items()
        )
        lines.append(f"  {deps}")
        lines.append(f"  bundled backbones: {', '.join(self.bundled_backbones) or 'none -- this install is missing its package data'}")
        lines.append("  Available now:")
        for cap in self.capabilities:
            if cap.available:
                lines.append(f"    ✓ {cap.name:<36} {cap.detail}")
        missing = [cap for cap in self.capabilities if not cap.available]
        if missing:
            lines.append("  Not available:")
            for cap in missing:
                lines.append(f"    ✗ {cap.name:<36} {cap.detail}")
                if cap.install:
                    lines.append(f"      {' ' * 36} needs: {cap.install}")
        if self.peak_rss_mb is not None:
            verdict = "yes" if self.meets_memory_target else "no"
            loaded = " (torch is loaded in this process; the target describes a core install)" if self.torch_loaded_in_process else ""
            lines.append(f"  Fits the 250 MB target here: {verdict} -- this process peaked at {self.peak_rss_mb:.0f} MB{loaded}")
        if self.gpu:
            lines.append(f"  GPU detected: {self.gpu}. device=\"cuda\" is available and is not selected; the defaults stay CPU.")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


#: Import name -> distribution name, where they differ. find_spec needs the
#: first; the version lives under the second. "Pillow: not installed" on a
#: machine that had just decoded twelve JPEGs with it was the first bug here.
_DISTRIBUTION = {"PIL": "Pillow", "faiss": "faiss-cpu"}


def _version(import_name: str) -> str | None:
    """The installed version, "present" if installed without metadata, None if absent.

    A probe that raises is worse than a wrong answer: this report exists to
    describe a broken environment, so a finder that throws on `torch` -- which
    a shim or a blocked import can do -- reads as "not installed", not a crash.
    """

    try:
        if importlib.util.find_spec(import_name) is None:
            return None
    except (ImportError, ValueError):
        return None
    try:
        return importlib.metadata.version(_DISTRIBUTION.get(import_name, import_name))
    except importlib.metadata.PackageNotFoundError:
        return "present"


def _meminfo() -> tuple[float | None, float | None]:
    total = available = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) / 1024.0
                elif line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return total, available


def _peak_rss_mb() -> float | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as status:
            for line in status:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def _gpu() -> str | None:
    """Named if torch can see one. Never selected."""

    if _version("torch") is None:
        return None
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "Apple MPS"
    except Exception:  # noqa: BLE001 -- a broken torch must not break the report about it
        return None
    return None


def _measure_latency(backbone: str, repeats: int = 5) -> tuple[float, float]:
    """A real inference on the bundled photographs, timed on this machine.

    Twelve support images, one query, repeated. Returns (median, p95) in ms
    for the full predict path, embedding included. It is what this machine
    does *now*: a busy machine reports a slower number, and that is the point.
    """

    from .config.settings import AdaptShotConfig, Backbone
    from .core.learner import FewShotLearner
    from .data import sample_images

    paths, labels = sample_images()
    # `backbone` came from bundled_onnx_backbones(), which only returns names in
    # BackboneRegistry; the cast records that rather than widening the config.
    learner = FewShotLearner(
        config=AdaptShotConfig(backbone=cast(Backbone, backbone), device="cpu", seed=42)
    )
    learner.load_support_images(paths[:-1], labels[:-1])
    learner.predict(paths[-1])  # warm: session construction is not what a user waits for per image
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        learner.predict(paths[-1])
        samples.append((time.perf_counter() - started) * 1000.0)
    samples.sort()
    return statistics.median(samples), samples[min(len(samples) - 1, int(0.95 * len(samples)))]


def check_environment(*, measure: bool = True) -> EnvironmentReport:
    """**Experimental.** May change in a minor release without a deprecation cycle; see ``adaptshot.api``.

    Report what this install can do on this machine, with measured figures.

    Args:
        measure: Run a real inference and read this process's peak memory.
            ``False`` skips both and reports availability only, in under a
            millisecond -- for code paths that cannot afford the second.
    """

    # Keyed by the name people know (the distribution), probed by the import name.
    deps = {
        display: _version(import_name)
        for display, import_name in (("numpy", "numpy"), ("Pillow", "PIL"), *((n, n) for n in _OPTIONAL))
    }
    backbones = bundled_onnx_backbones()
    torch_present = deps["torch"] is not None
    onnx_present = deps["onnxruntime"] is not None
    total, available = _meminfo()
    notes: list[str] = []

    inference_ok = onnx_present and bool(backbones)
    if not inference_ok and torch_present:
        inference_ok = True
        notes.append("inference runs through torch here; no ONNX backbone is bundled or onnxruntime is missing")

    latency: tuple[float | None, float | None] = (None, None)
    backbone_used = None
    if measure and inference_ok and backbones:
        try:
            backbone_used = "mobilenet_v3_small" if "mobilenet_v3_small" in backbones else backbones[0]
            latency = _measure_latency(backbone_used)
        except Exception as error:  # noqa: BLE001 -- report the failure, do not become one
            notes.append(f"could not measure latency: {type(error).__name__}: {error}")

    caps = [
        Capability(
            "predict, correct, save / load",
            inference_ok,
            (f"{latency[0]:.1f} ms per image, median of 5, measured here on {backbone_used}"
             if latency[0] is not None else "bundled ONNX backbone" if inference_ok else "no backend: onnxruntime missing or no bundled weights"),
            None if inference_ok else "pip install adaptshot   (reinstall; the wheel carries the backbone)",
        ),
        Capability(
            "conformal prediction sets",
            inference_ok,
            "coverage guarantee validated in tests/test_conformal_coverage.py; needs "
            "ceil((1-alpha)/alpha) calibration scores to be informative",
        ),
        Capability(
            "out-of-distribution flag",
            inference_ok,
            "leave-one-out-calibrated Mahalanobis; at least 3 support photos per class",
        ),
        Capability(
            "fine-tuning (CA-EWC) via correct()",
            torch_present,
            "torch present" if torch_present else "needs torch; download size not measured here (requires the network)",
            None if torch_present else 'pip install "adaptshot[torch]"',
        ),
        Capability(
            "backbones other than the bundled one",
            torch_present,
            "resnet18 and any torchvision model" if torch_present else "only the bundled backbone(s) without torch",
            None if torch_present else 'pip install "adaptshot[torch]"',
        ),
        Capability(
            "faster search for support sets over 100 images",
            deps["faiss"] is not None,
            "faiss present" if deps["faiss"] else "numpy search is used; fine below ~100 images",
            None if deps["faiss"] else 'pip install "adaptshot[faiss]"',
        ),
    ]

    peak = _peak_rss_mb() if measure else None
    torch_loaded = "torch" in sys.modules
    if peak is not None and torch_loaded:
        notes.append("torch was already imported in this process, which alone costs several hundred MB")

    return EnvironmentReport(
        python=platform.python_version(),
        platform=platform.system() + " " + platform.release(),
        machine=platform.machine(),
        cpu_count=os.cpu_count(),
        ram_total_mb=total,
        ram_available_mb=available,
        dependencies=deps,
        bundled_backbones=backbones,
        capabilities=caps,
        latency_ms_median=latency[0],
        latency_ms_p95=latency[1],
        latency_backbone=backbone_used,
        peak_rss_mb=peak,
        torch_loaded_in_process=torch_loaded,
        gpu=_gpu(),
        notes=notes,
    )


__all__ = ["Capability", "EnvironmentReport", "check_environment"]
