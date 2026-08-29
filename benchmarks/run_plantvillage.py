"""The first real AdaptShot result, with confidence intervals and baselines.

Protocol, fixed before any number was looked at (#18):

- 5-way 5-shot classification on real PlantVillage crop-disease photographs
- >= 100 episodes, sampled from a fixed seed
- mean accuracy +/- 95% confidence interval, never a bare point estimate
- conformal empirical coverage at alpha = 0.1, plus mean prediction-set size
- hardware recorded alongside the result

Writing the protocol down first is what separates a result from a number that
was searched for until it looked good.

Every method runs on the same episodes and the same embeddings (#19), so a gap
between two of them is a fact about the methods rather than about their luck.

Usage::

    python scripts/fetch_plantvillage.py --out data/pv_bench \\
        --per-class 20 --preset benchmark
    python -m benchmarks.run_plantvillage --seed 42

Writes results/plantvillage_5way5shot.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import adaptshot
from adaptshot.config.settings import AdaptShotConfig
from adaptshot.core.learner import FewShotLearner
from adaptshot.utils.determinism import set_deterministic_seed
from benchmarks.baselines import knn, linear_probe, nearest_centroid, top1_with_threshold
from benchmarks.plantvillage import (
    DatasetMissing,
    Episode,
    dataset_provenance,
    embed_pool,
    load_pool,
    sample_episodes,
)

#: 1.96 sigma. Episode accuracies are a mean of >=100 independent draws, so the
#: central limit theorem applies and the normal interval is the standard one in
#: the few-shot literature. It is reported over episodes, not over queries --
#: queries within an episode share a support set and are not independent.
Z_95 = 1.96

def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-8)


def mean_and_ci(values: list[float]) -> tuple[float, float]:
    """Mean and 95% half-width over episodes."""

    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return float(array.mean()) if array.size else 0.0, 0.0
    standard_error = float(array.std(ddof=1) / np.sqrt(array.size))
    return float(array.mean()), Z_95 * standard_error


def adaptshot_episode(
    paths: list[Path],
    labels: np.ndarray,
    episode: Episode,
    config: AdaptShotConfig,
) -> tuple[float, float, float, float, dict[str, Any]]:
    """Run one episode through the real `FewShotLearner`.

    This calls the shipped library on image paths rather than reimplementing the
    prototype path over cached embeddings. The first version of this benchmark
    did the latter, and produced an AdaptShot accuracy identical to the
    nearest-centroid baseline to four significant figures -- because that is
    exactly what the twenty lines I had written were. It was comparing a
    reimplementation to itself and reporting the tie as a finding.

    The learner self-calibrates its conformal engine by leave-one-out over the
    support set (`_self_calibrate_conformal`), so it is given no calibration
    split. The baselines are; see `top1_episode` for why that asymmetry is the
    safe direction.

    Returns (accuracy, coverage, mean set size, OOD flag rate, stage timings),
    where the timings are the wall clock of `load_support_images` for this
    episode and of each `predict` call -- the full path, ONNX forward included.
    """

    learner = FewShotLearner(config=config)
    clock = time.perf_counter()
    learner.load_support_images(
        [str(paths[index]) for index in episode.support],
        [str(label) for label in labels[episode.support]],
    )
    fit_ms = (time.perf_counter() - clock) * 1000.0

    query_labels = labels[episode.query]
    correct = 0
    covered = 0
    set_sizes: list[int] = []
    flagged = 0
    predict_ms: list[float] = []

    for index, true_label in zip(episode.query, query_labels, strict=True):
        clock = time.perf_counter()
        result = learner.predict(str(paths[index]))
        predict_ms.append((time.perf_counter() - clock) * 1000.0)
        correct += int(result.prediction == true_label)
        predicted_set = result.conformal_set or [result.prediction]
        covered += int(true_label in predicted_set)
        set_sizes.append(len(predicted_set))
        flagged += int(result.ood_flag)

    n = len(query_labels)
    return (
        correct / n,
        covered / n,
        float(np.mean(set_sizes)),
        flagged / n,
        {"fit_ms": fit_ms, "predict_ms": predict_ms},
    )


def baseline_episode(
    name: str,
    embeddings: np.ndarray,
    labels: np.ndarray,
    episode: Episode,
) -> float:
    support = embeddings[episode.support]
    support_labels = labels[episode.support]
    query = embeddings[episode.query]
    query_labels = labels[episode.query]

    if name == "nearest_centroid":
        predictions = nearest_centroid(support, support_labels, query)
    elif name == "knn_1":
        predictions = knn(support, support_labels, query, k=1)
    elif name == "knn_5":
        predictions = knn(support, support_labels, query, k=5)
    elif name == "linear_probe":
        predictions = linear_probe(support, support_labels, query)
    else:
        raise ValueError(f"unknown baseline: {name}")

    return float(np.mean(predictions == query_labels))


def top1_episode(
    embeddings: np.ndarray,
    labels: np.ndarray,
    episode: Episode,
    alpha: float,
) -> tuple[float, float, float, float]:
    """Top-1 with an abstention threshold *calibrated to the same target*.

    This is the comparison #19 calls the important one, and getting it fair took
    two attempts. The first fixed the threshold at 0.5, which a 5-way softmax
    over cosine similarities can never reach -- chance is 0.2 and the
    similarities are close -- so the baseline abstained on every query and
    reported 0.00% coverage. A baseline that cannot fire does not lose the
    comparison, it voids it.

    The threshold is now chosen on the episode's calibration split as the
    (alpha * 100)th percentile of the true class's confidence, which is the
    threshold that would have achieved the target coverage there. That is the
    same construction split-conformal uses, so the two are answering the same
    question on the same data, and the honest one to ask: at equal coverage,
    which produces smaller sets?

    Note the asymmetry -- this baseline gets a held-out calibration split and
    AdaptShot does not, since the learner self-calibrates from support alone. It
    is the safe direction: if conformal still wins while the alternative is
    handed extra data, the conclusion survives the doubt.

    Returns (accuracy, coverage, mean set size, threshold used).
    """

    support = embeddings[episode.support]
    support_labels = labels[episode.support]

    calibration_confidence = _confidence_in_true_class(
        support, support_labels, embeddings[episode.calibration], labels[episode.calibration]
    )
    threshold = float(np.percentile(calibration_confidence, alpha * 100.0))

    predictions, sets = top1_with_threshold(
        support, support_labels, embeddings[episode.query], threshold
    )
    query_labels = labels[episode.query]
    accuracy = float(np.mean(predictions == query_labels))
    covered = float(
        np.mean([str(t) in s for s, t in zip(sets, query_labels, strict=True)])
    )
    mean_size = float(np.mean([len(s) for s in sets]))
    return accuracy, covered, mean_size, threshold


def _confidence_in_true_class(
    support: np.ndarray,
    support_labels: np.ndarray,
    points: np.ndarray,
    point_labels: np.ndarray,
) -> np.ndarray:
    """Softmax probability the centroid classifier puts on the correct class."""

    classes = np.unique(support_labels)
    centroids = _normalise(
        np.stack([support[support_labels == name].mean(axis=0) for name in classes])
    )
    similarity = _normalise(points) @ centroids.T
    logits = similarity - similarity.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    lookup = {name: index for index, name in enumerate(classes)}
    return np.array(
        [probabilities[row, lookup[label]] for row, label in enumerate(point_labels)]
    )


def _cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    # ARM kernels do not write "model name"; lscpu decodes the implementer and
    # part ids into one ("Neoverse-N2"), which is the line a reader can compare.
    try:
        listing = subprocess.run(["lscpu"], capture_output=True, text=True, check=False).stdout
    except OSError:
        listing = ""
    for line in listing.splitlines():
        if line.startswith("Model name:"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or None


def peak_rss_mb() -> float | None:
    """This process's high-water RSS, from /proc. Linux only; None elsewhere.

    VmHWM rather than getrusage: the latter is inherited across vfork and
    reports the parent's watermark when run under a harness (see
    tests/test_memory_ceiling.py for the 119MB-vs-514MB measurement).
    """

    try:
        with open("/proc/self/status", encoding="utf-8") as status:
            for line in status:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def median_p95(samples: list[float]) -> dict[str, float | int]:
    """Median and p95, not mean. Tail latency is what makes a tool feel broken."""

    array = np.asarray(samples, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "n": int(array.size),
    }


_COLD_START = """
import time
started = time.perf_counter()
import json, sys
from adaptshot import AdaptShotConfig, FewShotLearner
support, labels, query = json.loads(sys.argv[1])
learner = FewShotLearner(config=AdaptShotConfig(backbone=sys.argv[2], device="cpu", seed=42))
learner.load_support_images(support, labels)
learner.predict(query)
elapsed = time.perf_counter() - started
peak = None
try:
    with open("/proc/self/status", encoding="utf-8") as status:
        for line in status:
            if line.startswith("VmHWM:"):
                peak = int(line.split()[1]) / 1024.0
except OSError:
    pass
print(json.dumps({"seconds": elapsed, "peak_rss_mb": peak}))
"""


def cold_start(
    support: list[str], labels: list[str], query: str, backbone: str
) -> dict[str, float | None] | None:
    """Import, build the learner, learn one support set, answer one query -- in a
    fresh interpreter, timed from before the first import, with that process's
    peak RSS. This is what a field user experiences, and the memory figure that
    describes the library rather than the benchmark harness around it."""

    import json
    import sys

    completed = subprocess.run(
        [sys.executable, "-c", _COLD_START, json.dumps([support, labels, query]), backbone],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    record: dict[str, float | None] = json.loads(completed.stdout.strip().splitlines()[-1])
    return record


def hardware() -> dict[str, Any]:
    """Recorded so a number can be compared against another machine honestly."""

    record: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        # The line a reader can compare their own machine against. platform's
        # answer is "x86_64", which describes every laptop sold this decade.
        "cpu_model": _cpu_model(),
        "install": "torch" if importlib.util.find_spec("torch") else "core",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "adaptshot": adaptshot.__version__,
    }
    try:
        import os

        record["cpu_count"] = os.cpu_count()
    except Exception:  # noqa: BLE001 - provenance is best-effort, never fatal
        record["cpu_count"] = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    record["ram_gb"] = round(int(line.split()[1]) / 1_048_576, 1)
                    break
    except OSError:
        record["ram_gb"] = None
    try:
        import onnxruntime

        record["onnxruntime"] = onnxruntime.__version__
    except ImportError:
        record["onnxruntime"] = None
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data/pv_bench"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--n-calibration", type=int, default=5)
    parser.add_argument("--n-query", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--backbone", default="mobilenet_v3_small")
    parser.add_argument("--output", type=Path, default=Path("results/plantvillage_5way5shot.json"))
    args = parser.parse_args(argv)

    set_deterministic_seed(args.seed)
    config = AdaptShotConfig(backbone=args.backbone, device="cpu", seed=args.seed)

    try:
        paths, labels, classes = load_pool(args.data)
    except DatasetMissing as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"pool: {len(paths)} images across {len(classes)} classes")
    started = time.perf_counter()
    embeddings = embed_pool(
        paths, config, cache_path=args.data / f".embeddings.{args.backbone}.npy"
    )
    print(f"embedded in {time.perf_counter() - started:.1f}s")

    episodes = sample_episodes(
        labels,
        classes,
        n_way=args.n_way,
        k_shot=args.k_shot,
        n_calibration=args.n_calibration,
        n_query=args.n_query,
        episodes=args.episodes,
        seed=args.seed,
    )
    print(f"{len(episodes)} episodes, {args.n_way}-way {args.k_shot}-shot\n")

    adaptshot_accuracy, coverage, set_size, ood_rate = [], [], [], []
    top1_accuracy, top1_coverage, top1_size, thresholds = [], [], [], []
    baseline_names = ("nearest_centroid", "knn_1", "knn_5", "linear_probe")
    baseline_accuracy: dict[str, list[float]] = {name: [] for name in baseline_names}

    latencies: list[float] = []
    fit_ms: list[float] = []
    predict_ms: list[float] = []
    for episode in episodes:
        clock = time.perf_counter()
        accuracy, episode_coverage, episode_size, episode_ood, stage = adaptshot_episode(
            paths, labels, episode, config
        )
        latencies.append(
            (time.perf_counter() - clock) / len(episode.query) * 1000.0
        )
        fit_ms.append(stage["fit_ms"])
        predict_ms.extend(stage["predict_ms"])
        adaptshot_accuracy.append(accuracy)
        coverage.append(episode_coverage)
        set_size.append(episode_size)
        ood_rate.append(episode_ood)

        t_accuracy, t_coverage, t_size, threshold = top1_episode(
            embeddings, labels, episode, args.alpha
        )
        top1_accuracy.append(t_accuracy)
        top1_coverage.append(t_coverage)
        top1_size.append(t_size)
        thresholds.append(threshold)

        for name in baseline_names:
            baseline_accuracy[name].append(
                baseline_episode(name, embeddings, labels, episode)
            )

    def report(label: str, values: list[float], unit: str = "%") -> dict[str, float]:
        mean, half = mean_and_ci(values)
        scale = 100.0 if unit == "%" else 1.0
        print(
            f"  {label:<34} {mean * scale:6.2f}{unit} "
            f"+/- {half * scale:.2f}  [{(mean - half) * scale:.2f}, "
            f"{(mean + half) * scale:.2f}]"
        )
        return {"mean": mean, "ci95_half_width": half, "n_episodes": len(values)}

    print("Accuracy, mean over episodes with 95% CI:")
    results: dict[str, Any] = {"accuracy": {}}
    results["accuracy"]["adaptshot"] = report("AdaptShot (prototype)", adaptshot_accuracy)
    for name in baseline_names:
        results["accuracy"][name] = report(name, baseline_accuracy[name])

    print(f"\nConformal at alpha={args.alpha} (target coverage "
          f"{(1 - args.alpha) * 100:.0f}%):")
    results["conformal"] = {
        "alpha": args.alpha,
        "target_coverage": 1 - args.alpha,
        "empirical_coverage": report("empirical coverage", coverage),
        "mean_set_size": report("mean prediction-set size", set_size, unit=""),
        "ood_flag_rate": report("OOD flagged (all in-distribution)", ood_rate),
    }

    print(f"\nTop-1 with a threshold calibrated to the same {(1 - args.alpha) * 100:.0f}% "
          "target, the alternative:")
    results["top1_threshold"] = {
        "accuracy": report("accuracy", top1_accuracy),
        "coverage": report("coverage (true label in set)", top1_coverage),
        "mean_set_size": report("mean set size", top1_size, unit=""),
        "threshold": report("threshold chosen per episode", thresholds, unit=""),
    }

    mean_latency, latency_half = mean_and_ci(latencies)

    # Per-image embedding, timed on its own: the ONNX forward for 100 pool
    # images, cache bypassed. predict() above includes this; here it is alone.
    from adaptshot.core.extractor import extract_embedding

    embed_ms: list[float] = []
    for path in paths[:100]:
        clock = time.perf_counter()
        extract_embedding(str(path), config)
        embed_ms.append((time.perf_counter() - clock) * 1000.0)

    first = episodes[0]
    cold = cold_start(
        [str(paths[i]) for i in first.support],
        [str(labels[i]) for i in first.support],
        str(paths[first.query[0]]),
        args.backbone,
    )
    rss = peak_rss_mb()

    timing = {
        "embedding_ms": median_p95(embed_ms),
        "support_fit_ms": median_p95(fit_ms),
        "predict_ms": median_p95(predict_ms),
        # Two memory numbers, deliberately named so they cannot be confused.
        # The first describes the library: one fresh process, one support set,
        # one answer. The second describes this harness: 400 cached embeddings,
        # 100 episodes and four baselines held at once. Only the first is a
        # claim about AdaptShot.
        "cold_start": cold,
        "benchmark_process_peak_rss_mb": rss,
        "note": (
            "predict_ms is the full path per query, ONNX forward included; "
            "support_fit_ms is load_support_images per episode (embed the support, "
            "leave-one-out calibration, OOD fit); cold_start is a fresh interpreter "
            "from before the first import to the first answer, with that process's "
            "peak RSS -- the single-cycle memory figure; benchmark_process_peak_rss_mb "
            "is this whole run's VmHWM and describes the harness, not the library. "
            "Both depend on which install ran them: see hardware.install"
        ),
    }
    print("\nLatency, median / p95 (ms), and memory -- see hardware for the machine:")
    print(f"  embedding, per image          {timing['embedding_ms']['median']:7.1f} / {timing['embedding_ms']['p95']:7.1f}   n={timing['embedding_ms']['n']}")
    print(f"  support fit, per episode      {timing['support_fit_ms']['median']:7.1f} / {timing['support_fit_ms']['p95']:7.1f}   n={timing['support_fit_ms']['n']}")
    print(f"  predict, per query (full)     {timing['predict_ms']['median']:7.1f} / {timing['predict_ms']['p95']:7.1f}   n={timing['predict_ms']['n']}")
    if cold is not None and cold.get("seconds") is not None:
        rss_line = f"  peak RSS {cold['peak_rss_mb']:.0f} MB" if cold.get("peak_rss_mb") else ""
        print(f"  cold start, fresh process     {cold['seconds']:7.2f} s  (one support set, one answer){rss_line}")
    else:
        print("  cold start: could not be measured")
    if rss is not None:
        print(f"  this benchmark process        {rss:7.0f} MB peak  (harness, not library)")

    results.update(
        protocol={
            "task": f"{args.n_way}-way {args.k_shot}-shot",
            "episodes": args.episodes,
            "n_calibration_per_class": args.n_calibration,
            "n_query_per_class": args.n_query,
            "seed": args.seed,
            "backbone": args.backbone,
            "alpha": args.alpha,
        },
        dataset=dataset_provenance(args.data),
        hardware=hardware(),
        latency_ms_per_query={"mean": mean_latency, "ci95_half_width": latency_half},
        timing=timing,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwritten to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
