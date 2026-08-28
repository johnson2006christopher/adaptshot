"""Evidence for #36: the ONNX path agrees with torch, and is faster on CPU.

Making inference torch-free is only worth doing if the embeddings it produces
are the same ones. This script measures both halves of that claim -- agreement
and latency -- so the numbers quoted in the README and in PR #36 have a command
behind them rather than a memory of a measurement.

Run it with both installs present::

    python -m benchmarks.onnx_parity

Three things make a naive measurement here lie. All three were caught producing
wrong numbers for this PR before this script existed:

1. *Scheduler noise.* A single mean over 25 calls moved between 6.8ms and
   12.2ms for one backbone on one box. Each process therefore reports the
   **median** of its calls, which rejects outliers rather than averaging them
   in.

2. *Backend contention.* onnxruntime and torch each size a thread pool to the
   machine, so measuring both in one process makes whichever runs second
   compete with a pool it did not create. Every figure is therefore measured in
   a **fresh subprocess** that loads exactly one backend.

3. *Bimodal per-process latency.* Even with 1 and 2 handled, repeated runs of
   an identical script settled at either ~6.6ms or ~11.0ms for resnet18 and
   stayed there for the life of the process -- CPU frequency and core placement
   decided once at startup. A "minimum over runs" would have quietly reported
   the lucky mode as if it were the number. So each measurement is repeated
   across **independent processes** and both the minimum and the median across
   them are reported. When those two disagree, the spread is the finding.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any, cast

import numpy as np

from adaptshot.core.extractor import (
    BackboneRegistry,
    bundled_onnx_backbones,
    onnx_weights_available,
)

#: Timed calls per process. The median of these is that process's estimate.
CALLS_PER_PROCESS = 25

#: Independent processes per (backbone, backend). Reported as min and median.
PROCESS_SAMPLES = 5

#: Discarded before timing, to pay for session construction and warm the caches.
WARMUP_CALLS = 8

_WORKER = """
import json, sys, time
import numpy as np
from PIL import Image

from adaptshot.config.settings import AdaptShotConfig
from adaptshot.core.extractor import extract_embedding
from adaptshot.utils.determinism import set_deterministic_seed

backbone, backend, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
calls, warmup = {calls}, {warmup}

set_deterministic_seed(seed)
rng = np.random.default_rng(seed)
image = Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))
config = AdaptShotConfig(backbone=backbone, device="cpu", seed=seed)

# return_numpy=False is the documented way to ask for the torch path by name:
# it returns a tensor, which an ONNX session can never produce.
as_numpy = backend == "onnx"


def call():
    return extract_embedding(image, config, return_numpy=as_numpy)


embedding = call()
if not as_numpy:
    embedding = embedding.detach().cpu().numpy()

for _ in range(warmup):
    call()

samples = []
for _ in range(calls):
    started = time.perf_counter()
    call()
    samples.append((time.perf_counter() - started) * 1000.0)

print(json.dumps({{
    "ms": float(np.median(samples)),
    "embedding": np.asarray(embedding, dtype=float).tolist(),
}}))
"""


def _run_worker(backbone: str, backend: str, seed: int) -> dict[str, Any]:
    """Run one (backbone, backend) measurement in an interpreter of its own."""

    script = _WORKER.format(calls=CALLS_PER_PROCESS, warmup=WARMUP_CALLS)
    completed = subprocess.run(
        [sys.executable, "-c", script, backbone, backend, str(seed)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"measuring {backbone} on {backend} failed:\n{completed.stderr[-2000:]}"
        )
    return cast("dict[str, Any]", json.loads(completed.stdout.strip().splitlines()[-1]))


def measure(backbone: str, backend: str, seed: int) -> tuple[float, float, np.ndarray]:
    """Return (min ms, median ms, embedding) across independent processes."""

    runs = [_run_worker(backbone, backend, seed) for _ in range(PROCESS_SAMPLES)]
    timings = [run["ms"] for run in runs]
    return min(timings), float(np.median(timings)), np.asarray(runs[0]["embedding"])


def torch_is_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def compare(backbone: str, seed: int) -> dict[str, Any]:
    """Measure agreement and latency for one backbone, ONNX against torch."""

    onnx_min, onnx_median, onnx_embedding = measure(backbone, "onnx", seed)
    result: dict[str, Any] = {
        "backbone": backbone,
        "onnx_ms_min": onnx_min,
        "onnx_ms_median": onnx_median,
    }

    if not torch_is_available():
        result.update(torch_ms_min=None, torch_ms_median=None, cosine=None, max_abs_diff=None)
        return result

    torch_min, torch_median, torch_embedding = measure(backbone, "torch", seed)
    norms = np.linalg.norm(onnx_embedding) * np.linalg.norm(torch_embedding)

    result["torch_ms_min"] = torch_min
    result["torch_ms_median"] = torch_median
    result["cosine"] = float(np.dot(onnx_embedding, torch_embedding) / norms)
    result["max_abs_diff"] = float(np.abs(onnx_embedding - torch_embedding).max())
    return result


def _print_table(results: list[dict[str, Any]], seed: int) -> None:
    bundled = set(bundled_onnx_backbones())
    print(
        f"\nONNX vs torch, CPU, seed {seed} (median of {CALLS_PER_PROCESS} calls, "
        f"across {PROCESS_SAMPLES} independent processes)\n"
    )
    header = (
        f"{'backbone':<22} {'ONNX min/med':>16} {'torch min/med':>16} "
        f"{'speedup':>9}  {'1 - cosine':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        onnx = f"{row['onnx_ms_min']:.1f} / {row['onnx_ms_median']:.1f} ms"
        if row["torch_ms_min"] is None:
            print(f"{row['backbone']:<22} {onnx:>16} {'n/a':>16} {'n/a':>9}  {'n/a':>11}")
            continue
        torch_cell = f"{row['torch_ms_min']:.1f} / {row['torch_ms_median']:.1f} ms"
        speedup = row["torch_ms_median"] / row["onnx_ms_median"]
        note = "" if row["backbone"] in bundled else "  (not bundled)"
        print(
            f"{row['backbone']:<22} {onnx:>16} {torch_cell:>16} "
            f"{speedup:>8.2f}x  {1.0 - row['cosine']:>11.2e}{note}"
        )

    if not torch_is_available():
        print("\ntorch is not installed, so only the ONNX path could be measured.")
        return

    worst = max(row["max_abs_diff"] for row in results)
    print(f"\nlargest absolute disagreement across all backbones: {worst:.2e}")
    print(
        "speedup uses the medians. Where a backbone's min and median differ "
        "widely,\nthat process-to-process spread is real and the speedup is "
        "correspondingly soft."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--json", action="store_true", help="emit results as JSON instead of a table"
    )
    args = parser.parse_args()

    measurable = sorted(name for name in BackboneRegistry if onnx_weights_available(name))
    if not measurable:
        print("No ONNX weights present. Run: python scripts/export_backbones.py")
        return 1

    results = [compare(name, args.seed) for name in measurable]

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        _print_table(results, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
