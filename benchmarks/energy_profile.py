#!/usr/bin/env python3
"""Energy-aware benchmark harness for AdaptShot.

Measures wall-clock latency, memory footprint, CPU utilization/frequency, and
estimates energy/carbon under a deterministic CPU-first smoke test.

Usage:
    python -m benchmarks.energy_profile --smoke-test --seed 42
"""

from __future__ import annotations

import argparse
import json
import platform
import random
from dataclasses import asdict
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from src.adaptshot.config.settings import AdaptShotConfig
from src.adaptshot.core.extractor import compute_preview_signature, extract_embedding, set_support_embedding_cache
from src.adaptshot.core.similarity import find_nearest_neighbor
from src.adaptshot.utils.determinism import set_deterministic_seed, verify_determinism

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None


DEFAULT_TDP_WATTS = 15.0
GRID_INTENSITY_CO2_PER_JOULE = 0.0004
SMOKE_SUPPORT_SIZE = 5
SMOKE_QUERY_SIZE = 10
IMAGE_SIZE = (224, 224)


def _cpu_frequency_mhz() -> float:
    """Return a best-effort CPU frequency estimate."""
    if psutil is not None:
        freq = psutil.cpu_freq()
        if freq is not None and freq.current is not None:
            return float(freq.current)

    cpu_mhz_values: List[float] = []
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("cpu mhz"):
                    _, value = line.split(":", 1)
                    cpu_mhz_values.append(float(value.strip()))
    except OSError:
        return 0.0

    if not cpu_mhz_values:
        return 0.0
    return float(sum(cpu_mhz_values) / len(cpu_mhz_values))


def _cpu_utilization_fraction(wall_time_s: float, process_time_s: float) -> float:
    """Return a deterministic utilization estimate in [0, 1]."""
    if wall_time_s <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, process_time_s / wall_time_s)))


def _deterministic_images(seed: int, count: int) -> List[Image.Image]:
    """Create a deterministic set of RGB images for profiling."""
    rng = np.random.default_rng(seed)
    images: List[Image.Image] = []
    for _ in range(count):
        array = rng.integers(0, 256, size=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)
        images.append(Image.fromarray(array, mode="RGB"))
    return images


def _run_profile(config: AdaptShotConfig, support_count: int, query_count: int) -> Dict[str, Any]:
    """Execute deterministic profiling and return serializable metrics."""
    set_deterministic_seed(config.seed)
    random.seed(config.seed)

    support_images = _deterministic_images(config.seed, support_count)
    query_images = _deterministic_images(config.seed + 1, query_count)
    support_labels = list(range(support_count))

    support_embeddings: List[np.ndarray] = []
    support_start = time.perf_counter()
    for image in support_images:
        support_embeddings.append(extract_embedding(image, config))
    support_latency = time.perf_counter() - support_start

    support_embeddings_np = np.stack(support_embeddings)
    set_support_embedding_cache(support_embeddings_np[0], compute_preview_signature(support_images[0]))

    latencies: List[float] = []
    correct = 0
    run_wall_start = time.perf_counter()
    run_cpu_start = time.process_time()
    for index, image in enumerate(query_images):
        start = time.perf_counter()
        query_embedding = extract_embedding(image, config)
        pred_label, _, _ = find_nearest_neighbor(
            query=query_embedding,
            support_embeddings=support_embeddings_np,
            support_labels=np.array(support_labels, dtype=object),
            use_faiss=config.use_faiss,
        )
        latencies.append(time.perf_counter() - start)
        if int(pred_label) == int(support_labels[index % len(support_labels)]):
            correct += 1

    run_wall_time = time.perf_counter() - run_wall_start
    run_cpu_time = time.process_time() - run_cpu_start

    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0
    utilization = _cpu_utilization_fraction(run_wall_time, run_cpu_time)
    cpu_mhz = _cpu_frequency_mhz()
    joules = float(DEFAULT_TDP_WATTS * avg_latency * utilization * max(1, query_count))
    co2_g = float(joules * GRID_INTENSITY_CO2_PER_JOULE)
    accuracy = float(correct / max(1, len(query_images)))

    current, peak = tracemalloc.get_traced_memory()

    return {
        "seed": config.seed,
        "platform": platform.platform(),
        "cpu_frequency_mhz": cpu_mhz,
        "cpu_utilization_fraction": utilization,
        "wall_time_s": float(run_wall_time),
        "process_time_s": float(run_cpu_time),
        "tdp_watts": DEFAULT_TDP_WATTS,
        "latency_avg_s": avg_latency,
        "latency_p95_s": p95_latency,
        "support_embedding_time_s": float(support_latency),
        "peak_memory_bytes": int(peak),
        "current_memory_bytes": int(current),
        "joules_estimate": joules,
        "co2_g_estimate": co2_g,
        "accuracy": accuracy,
        "eco_mode": config.eco_mode,
        "early_exit_threshold": config.early_exit_threshold,
    }


def run_smoke_test(config: AdaptShotConfig) -> Dict[str, Any]:
    """Run the energy smoke test with deterministic synthetic data."""
    tracemalloc.start()
    try:
        metrics = _run_profile(config=config, support_count=SMOKE_SUPPORT_SIZE, query_count=SMOKE_QUERY_SIZE)
    finally:
        tracemalloc.stop()
    return metrics


def main() -> int:
    """CLI entry point for the energy profiler."""
    parser = argparse.ArgumentParser(description="AdaptShot energy profiler")
    parser.add_argument("--smoke-test", action="store_true", help="Run deterministic smoke test")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument("--output", type=str, default="results/energy_profile.json", help="JSON output path")
    parser.add_argument("--eco-mode", action="store_true", help="Enable eco-mode early exit")
    parser.add_argument("--early-exit-threshold", type=float, default=0.95, help="Eco-mode early-exit threshold")
    args = parser.parse_args()

    set_deterministic_seed(args.seed)
    config = AdaptShotConfig(
        backbone="resnet18",
        device="cpu",
        seed=args.seed,
        eco_mode=bool(args.eco_mode),
        early_exit_threshold=float(args.early_exit_threshold),
        use_faiss=False,
    )

    if not args.smoke_test:
        print("Use --smoke-test to run the deterministic energy profile.")
        return 0

    baseline = run_smoke_test(AdaptShotConfig(**{**asdict(config), "eco_mode": False}))
    eco = run_smoke_test(config)

    baseline_avg = baseline["latency_avg_s"]
    eco_avg = eco["latency_avg_s"]
    eco_latency_reduction = 0.0 if baseline_avg <= 0 else max(0.0, (baseline_avg - eco_avg) / baseline_avg)

    payload = {
        "baseline": baseline,
        "eco": eco,
        "eco_latency_reduction_fraction": eco_latency_reduction,
        "deterministic": bool(
            verify_determinism(lambda: extract_embedding(_deterministic_images(args.seed, 1)[0], config), runs=3, seed=args.seed)
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
