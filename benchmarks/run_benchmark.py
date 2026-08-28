"""Minimal benchmark harness for AdaptShot core pipeline.

This module provides a reproducible, CPU-only evaluation harness for validating
the core few-shot inference pipeline: embedding extraction → similarity search → prediction.

Usage:
    python -m benchmarks.run_benchmark --smoke-test --output results/smoke_test.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage

from adaptshot.config.settings import AdaptShotConfig
from adaptshot.core.extractor import extract_embedding
from adaptshot.core.similarity import find_nearest_neighbor
from adaptshot.utils.determinism import set_deterministic_seed, verify_determinism

#: Where torchvision caches CIFAR-10, relative to `data_dir`.
_CIFAR_MARKER = Path("cifar-10-batches-py") / "data_batch_1"


def cifar10_is_cached(data_dir: str = "./data") -> bool:
    """Whether CIFAR-10 is already on disk, so no network is needed."""

    return (Path(data_dir) / _CIFAR_MARKER).is_file()


def resolve_dataset(requested: str, data_dir: str, allow_download: bool) -> str:
    """Decide which dataset to actually use, and say so out loud.

    `auto` is the default because the documented validation gate must work with
    no network. AdaptShot's whole argument is that connectivity is the resource
    its users do not have; a pre-PR command that silently requires a 170MB
    download contradicts that, and on a slow link it does not fail, it hangs.

    Returns:
        The dataset name to load.
    """

    if requested != "auto":
        return requested
    if cifar10_is_cached(data_dir):
        return "cifar10"
    if allow_download:
        return "cifar10"
    return "synthetic"


def _synthetic_split(
    n_way: int, k_shot: int, seed: int, n_query: int = 5
) -> tuple[list[tuple[torch.Tensor, int]], list[tuple[torch.Tensor, int]]]:
    """A deterministic fixture that needs no network and no disk.

    Each class is a fixed random mean plus per-sample noise, so the classes are
    separable and the pipeline exercises every stage end to end.

    This measures whether the pipeline still *works* and how fast, which is what
    a smoke test is for. It does not measure whether the model is any good, and
    `run_smoke_test` refuses to report an accuracy figure from it -- a number
    measured on random tensors describes nothing, and publishing one would be
    the same mistake #17 had to retract.
    """

    generator = torch.Generator().manual_seed(seed)
    support: list[tuple[torch.Tensor, int]] = []
    query: list[tuple[torch.Tensor, int]] = []

    for label in range(n_way):
        centre = torch.rand(3, 32, 32, generator=generator)
        for store, count in ((support, k_shot), (query, n_query)):
            for _ in range(count):
                noise = torch.randn(3, 32, 32, generator=generator) * 0.08
                store.append((torch.clamp(centre + noise, 0.0, 1.0), label))

    return support, query


def load_few_shot_split(
    dataset_name: str = "cifar10",
    n_way: int = 5,
    k_shot: int = 10,
    seed: int = 42,
    data_dir: str = "./data",
    allow_download: bool = False,
) -> tuple[list[tuple[torch.Tensor, int]], list[tuple[torch.Tensor, int]]]:
    """
    Load a few-shot split from a torchvision dataset.

    For smoke testing, we use CIFAR-10: lightweight, built-in, and representative
    of few-shot classification tasks.

    Args:
        dataset_name: Name of dataset to load (currently only "cifar10" supported)
        n_way: Number of classes to include in the few-shot task
        k_shot: Number of support examples per class
        seed: Random seed for reproducible splitting
        data_dir: Directory to cache/download dataset

    Returns:
        support_data: List of (image_tensor, label) tuples for support set
        query_data: List of (image_tensor, label) tuples for query/evaluation set
    """
    set_deterministic_seed(seed)

    if dataset_name == "synthetic":
        return _synthetic_split(n_way=n_way, k_shot=k_shot, seed=seed)

    if dataset_name == "cifar10":
        if not cifar10_is_cached(data_dir) and not allow_download:
            # Fail immediately and say what to run. The previous behaviour was to
            # start a 170MB download with no warning, which on a constrained link
            # produced five minutes of silence and then an opaque timeout.
            raise FileNotFoundError(
                f"CIFAR-10 is not cached in {data_dir} and downloads are not "
                "permitted.\n"
                "  Offline (default):  python -m benchmarks.run_benchmark "
                "--smoke-test --seed 42\n"
                "  Fetch it once:      python -m benchmarks.run_benchmark "
                "--smoke-test --dataset cifar10 --allow-download --seed 42\n"
                "The download is ~170MB and has taken over 30 minutes from some "
                "networks."
            )
        dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=allow_download,
            transform=transforms.ToTensor(),  # Minimal transform for speed
        )

        # Select first n_way classes for reproducibility
        classes = list(range(n_way))
        support_indices, query_indices = [], []

        for cls in classes:
            # Get all indices for this class
            cls_indices = [i for i, (_, label) in enumerate(dataset) if label == cls]
            # Randomly select k_shot support + 5 query examples
            selected = np.random.choice(cls_indices, size=k_shot + 5, replace=False)
            support_indices.extend(selected[:k_shot])
            query_indices.extend(selected[k_shot:])

        support_data = [(dataset[i][0], int(dataset[i][1])) for i in support_indices]
        query_data = [(dataset[i][0], int(dataset[i][1])) for i in query_indices]
        return support_data, query_data

    elif dataset_name == "miniimagenet":
        # miniImageNet: requires pre-downloaded dataset from
        # https://lyy.mpi-inf.mpg.de/mtl/download/Lmzjm9tX.html
        # Extract to data/miniimagenet/ with train/val/test CSV splits.
        import csv
        data_path = Path(data_dir) / "miniimagenet"
        images_dir = data_path / "images"
        csv_path = data_path / "train.csv"

        if not csv_path.exists():
            raise FileNotFoundError(
                f"miniImageNet not found at {csv_path}. "
                "Download from https://lyy.mpi-inf.mpg.de/mtl/download/Lmzjm9tX.html "
                "and extract to data/miniimagenet/"
            )

        # Read train.csv to get file-to-label mapping
        label_to_images: dict[str, list[str]] = {}
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip the header row
            for row in reader:
                filename, label = row[0], row[1]
                if label not in label_to_images:
                    label_to_images[label] = []
                label_to_images[label].append(filename)

        # Select n_way classes randomly
        all_labels = list(label_to_images.keys())
        rng = np.random.default_rng(seed)
        selected_labels = sorted(rng.choice(all_labels, size=min(n_way, len(all_labels)), replace=False))

        # Build support and query splits
        img_transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
        ])

        support_data: list[tuple[torch.Tensor, int]] = []
        query_data: list[tuple[torch.Tensor, int]] = []

        for cls_idx, label in enumerate(selected_labels):
            images = label_to_images[label]
            if len(images) < k_shot + 5:
                continue
            selected = rng.choice(len(images), size=k_shot + 5, replace=False)
            for j, img_idx in enumerate(selected[:k_shot]):
                img_path = images_dir / images[img_idx]
                img = datasets.folder.default_loader(str(img_path))
                support_data.append((img_transform(img), cls_idx))
            for j, img_idx in enumerate(selected[k_shot:k_shot + 5]):
                img_path = images_dir / images[img_idx]
                img = datasets.folder.default_loader(str(img_path))
                query_data.append((img_transform(img), cls_idx))

        return support_data, query_data

    else:
        raise ValueError(f"Dataset not implemented: {dataset_name}. Use 'cifar10' or 'miniimagenet'.")


# ---------------------------------------------------------------------------
# Reference baseline results from published few-shot learning literature.
# These are NOT claims about AdaptShot performance — they are provided for
# context when interpreting AdaptShot benchmark results.
# ---------------------------------------------------------------------------
BASELINE_REFERENCES: dict[str, dict[str, Any]] = {
    "prototypical_networks": {
        "paper": "Snell et al. (2017) Prototypical Networks for Few-shot Learning",
        "miniImageNet_5way_1shot": 49.42,
        "miniImageNet_5way_5shot": 68.20,
        "backbone": "Conv-4 (shallow, trained from scratch)",
        "note": "AdaptShot uses frozen pretrained ResNet-18, which may perform differently",
    },
    "matching_networks": {
        "paper": "Vinyals et al. (2016) Matching Networks for One Shot Learning",
        "miniImageNet_5way_1shot": 43.56,
        "miniImageNet_5way_5shot": 55.31,
        "backbone": "Conv-4",
        "note": "Fully differentiable nearest-neighbor classifier",
    },
    "maml": {
        "paper": "Finn et al. (2017) Model-Agnostic Meta-Learning",
        "miniImageNet_5way_1shot": 48.70,
        "miniImageNet_5way_5shot": 63.11,
        "backbone": "Conv-4",
        "note": "Gradient-based meta-learning (requires training)",
    },
}


def run_smoke_test(
    config: AdaptShotConfig, dataset: str = "cifar10", allow_download: bool = False
) -> dict[str, Any]:
    """
    Run minimal end-to-end pipeline and return metrics.

    Args:
        config: AdaptShotConfig with pipeline settings
        dataset: Dataset name ('cifar10', 'miniimagenet' or 'synthetic')
        allow_download: Whether fetching a missing dataset over the network is
            permitted. False by default: the documented validation gate must
            work offline.

    Returns:
        Dictionary containing accuracy, latency, and metadata metrics. On the
        synthetic fixture `accuracy` is ``None`` -- see `_synthetic_split`.
    """
    print(f"🧪 Running benchmark on {dataset}...")

    # Load few-shot data
    support_data, query_data = load_few_shot_split(
        dataset_name=dataset,
        n_way=config.n_way,
        k_shot=config.k_shot,
        seed=config.seed,
        allow_download=allow_download,
    )

    print(f"   • Support: {len(support_data)} examples")
    print(f"   • Query: {len(query_data)} examples")

    # Extract support embeddings
    print("   • Extracting support embeddings...")
    support_embeddings: list[np.ndarray] = []
    support_labels: list[int] = []

    start_time = time.perf_counter()
    for img_tensor, label in support_data:
        img_pil = ToPILImage()(img_tensor)
        emb = extract_embedding(img_pil, config)
        support_embeddings.append(emb)
        support_labels.append(label)

    support_embeddings_np = np.stack(support_embeddings)
    embedding_time = time.perf_counter() - start_time
    print(f"   • Embedding extraction: {embedding_time:.3f}s for {len(support_data)} images")

    # Evaluate on query set
    print("   • Evaluating on query set...")
    correct = 0
    latencies: list[float] = []

    for img_tensor, true_label in query_data:
        img_pil = ToPILImage()(img_tensor)
        start = time.perf_counter()

        # Get prediction
        pred_label, _confidence, _ = find_nearest_neighbor(
            query=extract_embedding(img_pil, config),
            support_embeddings=support_embeddings_np,
            support_labels=np.array(support_labels),
            use_faiss=config.use_faiss,
        )

        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

        # Normalize types for comparison: both to int
        if isinstance(pred_label, str) and pred_label.isdigit():
            pred_label = int(pred_label)
        if isinstance(true_label, (np.integer, int)):
            true_label = int(true_label)

        if pred_label == true_label:
            correct += 1

    accuracy = correct / len(query_data)
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)

    return {
        # None on the synthetic fixture, deliberately. The pipeline can be timed
        # on random tensors; it cannot be evaluated on them, and a figure that
        # describes nothing is worse than no figure once someone quotes it.
        "accuracy": None if dataset == "synthetic" else float(accuracy),
        "data_source": dataset,
        "latency_avg_ms": float(avg_latency),
        "latency_p95_ms": float(p95_latency),
        "embedding_time_s": float(embedding_time),
        "support_size": len(support_data),
        "query_size": len(query_data),
        "config": {
            "backbone": config.backbone,
            "device": config.device,
            "use_faiss": config.use_faiss,
            "n_way": config.n_way,
            "k_shot": config.k_shot,
        },
    }


def main() -> int:
    """CLI entry point for benchmark harness."""
    parser = argparse.ArgumentParser(description="AdaptShot Benchmark Harness")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run minimal smoke test with CIFAR-10 subset",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="auto",
        help=(
            "Dataset: auto | cifar10 | miniimagenet | synthetic. "
            "auto (default) uses CIFAR-10 if it is already cached, and the "
            "offline synthetic fixture otherwise."
        ),
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help=(
            "Permit fetching a missing dataset over the network. Off by "
            "default: the validation gate must work without connectivity."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/smoke_test.json",
        help="Output JSON path for results (default: results/smoke_test.json)",
    )
    parser.add_argument(
        "--full-benchmark",
        action="store_true",
        help="Run full benchmark suite with multiple datasets and baseline comparisons",
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Enable memory profiling via tracemalloc",
    )
    args = parser.parse_args()

    # Set deterministic seed globally
    set_deterministic_seed(args.seed)

    # resnet18 is pinned here deliberately, not inherited from AdaptShotConfig's
    # default. The default became mobilenet_v3_small in #36 -- the backbone whose
    # ONNX weights ship in the wheel -- but this benchmark's whole value is
    # comparability with every number the project has published, and changing the
    # backbone would silently break that. Pass --backbone to measure another.
    config = AdaptShotConfig(
        backbone="resnet18",
        device="cpu",
        seed=args.seed,
        n_way=5,
        k_shot=10,
        use_faiss=False,  # Disable for smoke test to avoid FAISS dependency
    )

    if args.smoke_test or args.full_benchmark:
        dataset_name = resolve_dataset(args.dataset, "./data", args.allow_download)
        if args.dataset == "auto":
            print(f"   • Dataset resolved to '{dataset_name}' (--dataset auto)")
        results = run_smoke_test(
            config, dataset=dataset_name, allow_download=args.allow_download
        )

        # Print human-readable results
        print(f"\n📊 Benchmark Results ({dataset_name}):")
        if results["accuracy"] is None:
            print("   • Accuracy: not reported")
            print("     The synthetic fixture is random tensors. An accuracy")
            print("     measured on it describes nothing, so none is published.")
            print("     For a measured figure:")
            print("       --dataset cifar10 --allow-download")
        else:
            print(f"   • Accuracy: {results['accuracy']:.1%}")
        print(f"   • Avg Latency: {results['latency_avg_ms']:.1f} ms")
        print(f"   • P95 Latency: {results['latency_p95_ms']:.1f} ms")
        print(f"   • Embedding Time: {results['embedding_time_s']:.3f}s")

        # Save to JSON
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"   • Results saved to {output_path}")

        # Full benchmark: run on miniImageNet too if available, and show baselines
        if args.full_benchmark:
            print("\n📋 Baseline References (published results, NOT AdaptShot claims):")
            print("   Note: Published baselines use Conv-4 trained from scratch.")
            print("   AdaptShot uses frozen ImageNet-pretrained ResNet-18.")
            print("   Results are NOT directly comparable to published SOTA.")
            print()
            print(f"   {'Method':<25} {'1-shot':>8} {'5-shot':>8}")
            print(f"   {'-'*25} {'-'*8} {'-'*8}")
            for name, ref in BASELINE_REFERENCES.items():
                s1 = ref.get("miniImageNet_5way_1shot", "N/A")
                s5 = ref.get("miniImageNet_5way_5shot", "N/A")
                print(f"   {name:<25} {s1!s:>8} {s5!s:>8}")
            print()
            print("   AdaptShot results shown above are from: " + dataset_name)

            # Try miniImageNet if available
            if dataset_name != "miniimagenet":
                try:
                    print("\n   Attempting miniImageNet benchmark...")
                    mini_results = run_smoke_test(config, dataset="miniimagenet")
                    mini_path = Path(args.output).with_suffix(".miniimagenet.json")
                    with open(mini_path, "w", encoding="utf-8") as f:
                        json.dump(mini_results, f, indent=2, ensure_ascii=False)
                    print(f"   • miniImageNet accuracy: {mini_results['accuracy']:.1%}")
                    print(f"   • Results saved to {mini_path}")
                except FileNotFoundError:
                    print("   • miniImageNet not downloaded. Skipping.")
                    print("     Download from: https://lyy.mpi-inf.mpg.de/mtl/download/Lmzjm9tX.html")

        # Profile memory if requested
        if args.profile_memory:
            print("\n📊 Memory Profile:")
            try:
                from adaptshot.utils.profiling import estimate_model_memory_mb
                mem_est = estimate_model_memory_mb(config.backbone, config.n_way)
                print(f"   • Estimated total: {mem_est['estimated_total_mb']:.1f} MB")
                print(f"   • Under 250MB: {'✅ YES' if mem_est['under_250mb'] else '❌ NO'}")
                print(f"   • Breakdown: backbone_cache={mem_est['backbone_weights_cache_mb']}MB, " +
                      f"embeddings={mem_est['embeddings_buffer_mb']}MB, " +
                      f"head={mem_est['head_params_mb']}MB, " +
                      f"numpy={mem_est['numpy_overhead_mb']}MB")
            except ImportError:
                print("   • psutil not installed. Install for RSS measurement.")

        # Verify determinism
        print("\n🔍 Verifying determinism...")
        def run_once() -> np.ndarray:
            """Helper: run extraction once for determinism check.

            It must load the same dataset the benchmark just ran on. It used to
            take `load_few_shot_split`'s default, which is CIFAR-10 regardless of
            what was measured -- harmless while every path downloaded CIFAR
            anyway, and a hard failure the moment downloads stopped being
            implicit.
            """
            img_tensor, _ = load_few_shot_split(
                dataset_name=dataset_name,
                seed=config.seed,
                allow_download=args.allow_download,
            )[1][0]
            return extract_embedding(ToPILImage()(img_tensor), config)
        is_deterministic = verify_determinism(run_once, runs=3, seed=config.seed)
        print(f"   • Determinism check: {'✅ PASS' if is_deterministic else '❌ FAIL'}")
        return 0 if is_deterministic else 1

    else:
        print("Use --smoke-test to run minimal validation benchmark.")
        print("Use --full-benchmark for comprehensive benchmark suite.")
        print("Examples:")
        print("  python -m benchmarks.run_benchmark --smoke-test")
        print("  python -m benchmarks.run_benchmark --full-benchmark --profile-memory")
        return 0


if __name__ == "__main__":
    sys.exit(main())