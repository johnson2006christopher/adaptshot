#!/usr/bin/env python3
"""Minimal benchmark harness for AdaptShot core pipeline.

This module provides a reproducible, CPU-only evaluation harness for validating
the core few-shot inference pipeline: embedding extraction → similarity search → prediction.

Usage:
    python -m benchmarks.run_benchmark --smoke-test --output results/smoke_test.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import csv
import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage

from adaptshot.config.settings import AdaptShotConfig
from adaptshot.core.extractor import extract_embedding
from adaptshot.core.similarity import find_nearest_neighbor
from adaptshot.utils.determinism import set_deterministic_seed, verify_determinism


def load_few_shot_split(
    dataset_name: str = "cifar10",
    n_way: int = 5,
    k_shot: int = 10,
    seed: int = 42,
    data_dir: str = "./data",
) -> Tuple[List[Tuple[torch.Tensor, int]], List[Tuple[torch.Tensor, int]]]:
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

    if dataset_name == "cifar10":
        dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
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
        label_to_images: Dict[str, List[str]] = {}
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
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

        support_data: List[Tuple[torch.Tensor, int]] = []
        query_data: List[Tuple[torch.Tensor, int]] = []

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
BASELINE_REFERENCES: Dict[str, Dict[str, Any]] = {
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


def run_smoke_test(config: AdaptShotConfig, dataset: str = "cifar10") -> Dict[str, Any]:
    """
    Run minimal end-to-end pipeline and return metrics.

    Args:
        config: AdaptShotConfig with pipeline settings
        dataset: Dataset name ('cifar10' or 'miniimagenet')

    Returns:
        Dictionary containing accuracy, latency, and metadata metrics
    """
    print(f"🧪 Running benchmark on {dataset}...")

    # Load few-shot data
    support_data, query_data = load_few_shot_split(
        dataset_name=dataset,
        n_way=config.n_way,
        k_shot=config.k_shot,
        seed=config.seed,
    )

    print(f"   • Support: {len(support_data)} examples")
    print(f"   • Query: {len(query_data)} examples")

    # Extract support embeddings
    print("   • Extracting support embeddings...")
    support_embeddings: List[np.ndarray] = []
    support_labels: List[int] = []

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
    latencies: List[float] = []

    for img_tensor, true_label in query_data:
        img_pil = ToPILImage()(img_tensor)
        start = time.perf_counter()

        # Get prediction
        pred_label, confidence, _ = find_nearest_neighbor(
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
        "accuracy": float(accuracy),
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
        default="cifar10",
        help="Dataset name (default: cifar10)",
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

    # Default config for smoke test: CPU-only, no FAISS, ResNet-18
    config = AdaptShotConfig(
        backbone="resnet18",
        device="cpu",
        seed=args.seed,
        n_way=5,
        k_shot=10,
        use_faiss=False,  # Disable for smoke test to avoid FAISS dependency
    )

    if args.smoke_test or args.full_benchmark:
        dataset_name = args.dataset
        results = run_smoke_test(config, dataset=dataset_name)

        # Print human-readable results
        print(f"\n📊 Benchmark Results ({dataset_name}):")
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
                print(f"   {name:<25} {str(s1):>8} {str(s5):>8}")
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
            """Helper: run extraction once for determinism check."""
            img_tensor, _ = load_few_shot_split(seed=config.seed)[1][0]
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
    exit(main())