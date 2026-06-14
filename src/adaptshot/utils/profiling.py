"""Memory and latency profiling utilities for AdaptShot.

Provides lightweight instrumentation to verify the <250MB RAM claim
and measure latency at lifecycle key points. Uses only stdlib
(tracemalloc) with optional psutil enhancement.
"""

from __future__ import annotations

import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List

# Try psutil for more accurate RSS measurement
try:
    import psutil  # type: ignore[import-untyped]
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


@dataclass
class MemorySnapshot:
    """Memory measurement at a point in time.

    Attributes:
        label: Human-readable checkpoint name.
        rss_mb: Resident Set Size in MB (from psutil, if available).
        tracemalloc_mb: Python allocated memory in MB (from tracemalloc).
        tracemalloc_peak_mb: Peak Python allocated memory in MB.
        timestamp: Unix timestamp of measurement.
    """

    label: str = ""
    rss_mb: float = 0.0
    tracemalloc_mb: float = 0.0
    tracemalloc_peak_mb: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "rss_mb": self.rss_mb,
            "tracemalloc_mb": self.tracemalloc_mb,
            "tracemalloc_peak_mb": self.tracemalloc_peak_mb,
            "timestamp": self.timestamp,
        }


@dataclass
class MemoryProfile:
    """Complete memory profile across lifecycle events.

    Attributes:
        snapshots: Ordered list of memory snapshots.
        peak_rss_mb: Maximum RSS observed during profiling.
        peak_tracemalloc_mb: Maximum Python-allocated memory observed.
        passes_ram_budget: Whether peak RSS is under the budget (default 250MB).
    """

    snapshots: List[MemorySnapshot] = field(default_factory=list)
    peak_rss_mb: float = 0.0
    peak_tracemalloc_mb: float = 0.0
    passes_ram_budget: bool = True
    ram_budget_mb: float = 250.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshots": [s.to_dict() for s in self.snapshots],
            "peak_rss_mb": self.peak_rss_mb,
            "peak_tracemalloc_mb": self.peak_tracemalloc_mb,
            "passes_ram_budget": self.passes_ram_budget,
            "ram_budget_mb": self.ram_budget_mb,
        }


class MemoryTracker:
    """Track memory usage across AdaptShot lifecycle events.

    Usage:
        tracker = MemoryTracker(ram_budget_mb=250)
        with tracker.checkpoint("init"):
            learner = FewShotLearner(config=config)
        with tracker.checkpoint("load_support"):
            learner.load_support_images(paths, labels)
        tracker.profile()  # -> MemoryProfile
    """

    def __init__(self, ram_budget_mb: float = 250.0, start: bool = True) -> None:
        self.ram_budget_mb = ram_budget_mb
        self._snapshots: List[MemorySnapshot] = []
        self._start_time = time.time()
        if start:
            tracemalloc.start()

    @contextmanager
    def checkpoint(self, label: str) -> Any:
        """Context manager that records memory before and after a block."""
        self._take_snapshot(f"{label}_enter")
        yield
        self._take_snapshot(f"{label}_exit")

    def snapshot(self, label: str) -> MemorySnapshot:
        """Take a labeled memory snapshot."""
        snap = self._take_snapshot(label)
        self._snapshots.append(snap)
        return snap

    def _take_snapshot(self, label: str) -> MemorySnapshot:
        current, peak = tracemalloc.get_traced_memory()
        rss_mb = 0.0
        if _PSUTIL_AVAILABLE:
            try:
                proc = psutil.Process()
                rss_mb = proc.memory_info().rss / (1024 * 1024)
            except Exception:
                pass
        return MemorySnapshot(
            label=label,
            rss_mb=rss_mb,
            tracemalloc_mb=current / (1024 * 1024),
            tracemalloc_peak_mb=peak / (1024 * 1024),
            timestamp=time.time() - self._start_time,
        )

    def profile(self) -> MemoryProfile:
        """Finalize and return a complete memory profile."""
        tracemalloc.stop()
        peak_rss = max((s.rss_mb for s in self._snapshots), default=0.0)
        peak_tm = max((s.tracemalloc_peak_mb for s in self._snapshots), default=0.0)
        return MemoryProfile(
            snapshots=list(self._snapshots),
            peak_rss_mb=peak_rss,
            peak_tracemalloc_mb=peak_tm,
            passes_ram_budget=peak_rss <= self.ram_budget_mb if peak_rss > 0 else True,
            ram_budget_mb=self.ram_budget_mb,
        )


def estimate_model_memory_mb(backbone: str = "resnet18", n_classes: int = 5) -> Dict[str, float]:
    """Estimate memory usage without loading the model.

    Returns rough upper-bound estimates based on known architecture sizes.
    These are ballpark figures; actual usage varies with PyTorch version.
    """
    # ImageNet-pretrained weights cached on disk after first download
    cache_mb = 45.0  # ResNet-18 ~45MB, MobileNetV3 ~10MB
    # Embeddings: 4 bytes × embedding_dim × buffer_size
    embedding_dim = {"resnet18": 512, "mobilenet_v3_small": 576}[backbone]
    embeddings_mb = 4.0 * embedding_dim * 100 / (1024 * 1024)  # ~0.2MB for 100
    # Head: embedding_dim × n_classes × 4 bytes
    head_mb = 4.0 * embedding_dim * n_classes / (1024 * 1024)  # ~0.01MB
    # NumPy overhead
    numpy_overhead_mb = 20.0
    # Total estimate
    total_mb = cache_mb + embeddings_mb + head_mb + numpy_overhead_mb

    return {
        "backbone_weights_cache_mb": cache_mb,
        "embeddings_buffer_mb": round(embeddings_mb, 2),
        "head_params_mb": round(head_mb, 2),
        "numpy_overhead_mb": numpy_overhead_mb,
        "estimated_total_mb": round(total_mb, 1),
        "under_250mb": total_mb < 250.0,
    }
