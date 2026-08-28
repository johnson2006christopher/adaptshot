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
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import adaptshot
from adaptshot.config.settings import AdaptShotConfig
from adaptshot.core.conformal import ConformalEngine
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

#: The abstention threshold for the top-1 baseline. Deliberately *not* tuned:
#: picking it to make conformal look good, or bad, would answer a question
#: nobody asked. 0.5 is the natural "more likely than not" point on a 5-way
#: softmax, where chance is 0.2.
TOP1_THRESHOLD = 0.5


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
    embeddings: np.ndarray,
    labels: np.ndarray,
    episode: Episode,
    alpha: float,
) -> tuple[float, float, float]:
    """Run one episode through the prototype + conformal path.

    Returns (accuracy, coverage, mean set size).

    Conformal is calibrated on the episode's calibration split and evaluated on
    its query split. Those are disjoint by construction in `sample_episodes`,
    because a coverage number measured on the same points that set the quantile
    is not a coverage number.
    """

    support = embeddings[episode.support]
    support_labels = labels[episode.support]
    classes = np.unique(support_labels)
    centroids = _normalise(
        np.stack([support[support_labels == name].mean(axis=0) for name in classes])
    )

    def distances_to_centroids(points: np.ndarray) -> np.ndarray:
        # Cosine distance, matching the library's default similarity metric.
        return 1.0 - (_normalise(points) @ centroids.T)

    engine = ConformalEngine(alpha=alpha, min_calibration_size=10)

    calibration_distances = distances_to_centroids(embeddings[episode.calibration])
    for row, true_label in zip(calibration_distances, labels[episode.calibration]):
        engine.update_calibration(
            engine.softmax_nonconformity(row, classes, true_label), true_label
        )

    query_distances = distances_to_centroids(embeddings[episode.query])
    query_labels = labels[episode.query]

    correct = 0
    covered = 0
    set_sizes: list[int] = []
    for row, true_label in zip(query_distances, query_labels):
        order = np.argsort(row)
        top = classes[order[0]]
        confidence = float(1.0 - row[order[0]])
        result = engine.predict_set(row, classes, top, confidence)

        correct += int(top == true_label)
        covered += int(true_label in result.prediction_set)
        set_sizes.append(len(result.prediction_set))

    n = len(query_labels)
    return correct / n, covered / n, float(np.mean(set_sizes))


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
) -> tuple[float, float, float]:
    """Top-1 with an abstention threshold: the honest alternative to conformal."""

    predictions, sets = top1_with_threshold(
        embeddings[episode.support],
        labels[episode.support],
        embeddings[episode.query],
        TOP1_THRESHOLD,
    )
    query_labels = labels[episode.query]
    accuracy = float(np.mean(predictions == query_labels))
    covered = float(np.mean([str(t) in s for s, t in zip(sets, query_labels)]))
    mean_size = float(np.mean([len(s) for s in sets]))
    return accuracy, covered, mean_size


def hardware() -> dict[str, Any]:
    """Recorded so a number can be compared against another machine honestly."""

    record: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
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

    adaptshot_accuracy, coverage, set_size = [], [], []
    top1_accuracy, top1_coverage, top1_size = [], [], []
    baseline_names = ("nearest_centroid", "knn_1", "knn_5", "linear_probe")
    baseline_accuracy: dict[str, list[float]] = {name: [] for name in baseline_names}

    latencies: list[float] = []
    for episode in episodes:
        clock = time.perf_counter()
        accuracy, episode_coverage, episode_size = adaptshot_episode(
            embeddings, labels, episode, args.alpha
        )
        latencies.append(
            (time.perf_counter() - clock) / len(episode.query) * 1000.0
        )
        adaptshot_accuracy.append(accuracy)
        coverage.append(episode_coverage)
        set_size.append(episode_size)

        t_accuracy, t_coverage, t_size = top1_episode(embeddings, labels, episode)
        top1_accuracy.append(t_accuracy)
        top1_coverage.append(t_coverage)
        top1_size.append(t_size)

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
    }

    print(f"\nTop-1 with a {TOP1_THRESHOLD} confidence threshold, the alternative:")
    results["top1_threshold"] = {
        "threshold": TOP1_THRESHOLD,
        "accuracy": report("accuracy", top1_accuracy),
        "coverage": report("coverage (true label in set)", top1_coverage),
        "mean_set_size": report("mean set size", top1_size, unit=""),
    }

    mean_latency, latency_half = mean_and_ci(latencies)
    print(f"\nLatency per query (post-embedding): {mean_latency:.3f} +/- "
          f"{latency_half:.3f} ms")

    results.update(
        protocol={
            "task": f"{args.n_way}-way {args.k_shot}-shot",
            "episodes": args.episodes,
            "n_calibration_per_class": args.n_calibration,
            "n_query_per_class": args.n_query,
            "seed": args.seed,
            "backbone": args.backbone,
            "alpha": args.alpha,
            "top1_threshold": TOP1_THRESHOLD,
        },
        dataset=dataset_provenance(args.data),
        hardware=hardware(),
        latency_ms_per_query={"mean": mean_latency, "ci95_half_width": latency_half},
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwritten to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
