"""What AdaptShot costs on *this* machine: latency, memory and the ONNX path (#31).

The published figures were taken on a laptop, and the README says so. The claim
the project actually makes is "hardware people already own", which in the target
setting is an ARM board or a phone, not a laptop. This script is the measurement
that claim needs: dataset-free, torch-free, run on the device itself, writing one
artifact that names the hardware alongside every number so two machines can be
compared honestly.

Usage::

    python -m benchmarks.run_device --seed 42
    python -m benchmarks.run_device --verify-export /path/to/exported/onnx

Writes ``results/device_<machine>.json`` (``x86_64``, ``aarch64``, ...), unless
``--output`` says otherwise. It runs on the twelve bundled photographs, so it
needs nothing fetched and nothing cached, and it must be run on a core install
(numpy, Pillow, onnxruntime): with torch present the cold-start memory figure
describes torch's import, not the library.

``--verify-export DIR`` closes the loop the issue asks for: a graph exported on
this machine (``scripts/export_backbones.py``, torch extra) is loaded by
onnxruntime here and compared against the bundled graph on the same input.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from adaptshot.config.settings import AdaptShotConfig
from adaptshot.core.extractor import bundled_onnx_backbones, extract_embedding
from adaptshot.core.learner import FewShotLearner
from adaptshot.data import sample_images
from benchmarks.run_plantvillage import cold_start, hardware, median_p95, peak_rss_mb

#: Predictions per bundled photograph in the steady-state timing.
DEFAULT_ROUNDS = 5

#: Independent fresh-interpreter cold starts. Reported as min and median.
DEFAULT_COLD_STARTS = 3

#: Discarded before timing, to build the session and warm the caches.
WARMUP_CALLS = 3


def _split() -> tuple[list[str], list[str], str, str]:
    """The README quickstart's split: eleven photographs teach, the twelfth asks."""

    paths, labels = sample_images()
    return paths[:-1], labels[:-1], paths[-1], labels[-1]


def steady_state(
    config: AdaptShotConfig, support: list[str], labels: list[str], rounds: int
) -> dict[str, Any]:
    """Per-image embedding and per-query prediction, median and p95, in-process."""

    all_paths, _ = sample_images()

    embed_ms: list[float] = []
    for path in all_paths[:WARMUP_CALLS]:
        extract_embedding(path, config)
    for _ in range(rounds):
        for path in all_paths:
            clock = time.perf_counter()
            extract_embedding(path, config)
            embed_ms.append((time.perf_counter() - clock) * 1000.0)

    clock = time.perf_counter()
    learner = FewShotLearner(config=config)
    learner.load_support_images(support, labels)
    support_fit_ms = (time.perf_counter() - clock) * 1000.0

    predict_ms: list[float] = []
    for path in all_paths[:WARMUP_CALLS]:
        learner.predict(path)
    for _ in range(rounds):
        for path in all_paths:
            clock = time.perf_counter()
            learner.predict(path)
            predict_ms.append((time.perf_counter() - clock) * 1000.0)

    return {
        "embedding_ms": median_p95(embed_ms),
        "predict_ms": median_p95(predict_ms),
        "support_fit_ms": support_fit_ms,
        "support_size": len(support),
    }


def quickstart(config: AdaptShotConfig) -> dict[str, Any]:
    """Does the README's first prediction come out right here? A yes/no, recorded."""

    support, labels, query, truth = _split()
    learner = FewShotLearner(config=config)
    learner.load_support_images(support, labels)
    result = learner.predict(query)
    return {
        "query": Path(query).name,
        "expected": truth,
        "predicted": result.prediction,
        "correct": result.prediction == truth,
    }


def cold_starts(config: AdaptShotConfig, repeats: int) -> dict[str, Any] | None:
    """Fresh interpreter, import to first answer, ``repeats`` times over."""

    support, labels, query, _ = _split()
    records = [cold_start(support, labels, query, config.backbone) for _ in range(repeats)]
    seconds = [r["seconds"] for r in records if r is not None and r["seconds"] is not None]
    peaks = [r["peak_rss_mb"] for r in records if r is not None and r["peak_rss_mb"] is not None]
    if not seconds:
        return None
    return {
        "seconds_min": float(min(seconds)),
        "seconds_median": float(np.median(seconds)),
        "peak_rss_mb_max": float(max(peaks)) if peaks else None,
        "n": len(seconds),
    }


def verify_export(export_dir: Path, backbone: str) -> dict[str, Any]:
    """Load a graph exported on this machine and compare it with the bundled one.

    Both run through onnxruntime on the same preprocessed input, so what is
    compared is the export, not the runtime.
    """

    import onnxruntime as ort
    from PIL import Image

    from adaptshot.core.backends.onnx_backend import ONNXBackend

    graph = export_dir / f"{backbone}.onnx"
    if not graph.exists():
        return {"verified": False, "reason": f"{graph} not found"}

    paths, _ = sample_images()
    with Image.open(paths[0]) as image:
        # The backend's own preprocessing, so the two graphs see identical input.
        tensor = ONNXBackend._preprocess(image.convert("RGB"))

    session = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    exported = session.run(None, {session.get_inputs()[0].name: tensor})[0].squeeze(0)
    bundled = ONNXBackend().extract(Image.open(paths[0]).convert("RGB"), backbone)

    cosine = float(
        np.dot(exported, bundled) / (np.linalg.norm(exported) * np.linalg.norm(bundled) + 1e-12)
    )
    max_abs = float(np.max(np.abs(exported - bundled)))
    return {
        "verified": bool(cosine > 0.9999 and max_abs < 1e-3),
        "graph": str(graph),
        "graph_bytes": graph.stat().st_size,
        "weights_bytes": (
            (export_dir / f"{backbone}.onnx.data").stat().st_size
            if (export_dir / f"{backbone}.onnx.data").exists()
            else None
        ),
        "cosine_vs_bundled": cosine,
        "max_abs_diff_vs_bundled": max_abs,
        "onnxruntime": ort.__version__,
        "providers": session.get_providers(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", default="mobilenet_v3_small")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--cold-starts", type=int, default=DEFAULT_COLD_STARTS)
    parser.add_argument("--verify-export", type=Path, default=None, metavar="DIR")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    machine = platform.machine() or "unknown"
    output = args.output or Path("results") / f"device_{machine}.json"

    if args.backbone not in bundled_onnx_backbones():
        print(f"{args.backbone} is not bundled as ONNX; this script measures the torch-free path")
        return 1

    config = AdaptShotConfig(backbone=args.backbone, device="cpu", seed=args.seed)
    record: dict[str, Any] = {
        "machine": machine,
        "hardware": hardware(),
        "protocol": {
            "photographs": "the twelve bundled PlantVillage samples, eleven teach, one asks",
            "rounds": args.rounds,
            "warmup_calls": WARMUP_CALLS,
            "cold_starts": args.cold_starts,
            "seed": args.seed,
        },
        "quickstart": quickstart(config),
        "timing": steady_state(config, *_split()[:2], rounds=args.rounds),
        "cold_start": cold_starts(config, args.cold_starts),
        "benchmark_process_peak_rss_mb": peak_rss_mb(),
    }
    if args.verify_export is not None:
        record["export"] = verify_export(args.verify_export, args.backbone)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    hw = record["hardware"]
    t = record["timing"]
    cs = record["cold_start"] or {}
    print(f"device: {hw.get('cpu_model') or hw['processor']}  ({machine}, {hw['install']} install)")
    print(f"quickstart correct: {record['quickstart']['correct']}")
    print(
        f"embedding, per image   {t['embedding_ms']['median']:7.1f} / {t['embedding_ms']['p95']:7.1f} ms"
    )
    print(
        f"predict, per query     {t['predict_ms']['median']:7.1f} / {t['predict_ms']['p95']:7.1f} ms"
    )
    print(f"support fit (11)       {t['support_fit_ms']:7.0f} ms")
    if cs:
        print(
            f"cold start             {cs['seconds_median']:7.2f} s   peak {cs['peak_rss_mb_max']:.0f} MB"
        )
    if "export" in record:
        print(f"export verified: {record['export']['verified']}")
    print(f"written: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
