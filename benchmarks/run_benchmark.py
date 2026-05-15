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
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage

from src.adaptshot.config.settings import AdaptShotConfig
from src.adaptshot.core.extractor import extract_embedding
from src.adaptshot.core.similarity import find_nearest_neighbor
from src.adaptshot.utils.determinism import set_deterministic_seed, verify_determinism


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

    else:
        raise ValueError(f"Dataset not implemented for smoke test: {dataset_name}")


def run_smoke_test(config: AdaptShotConfig) -> Dict[str, Any]:
    """
    Run minimal end-to-end pipeline and return metrics.

    Args:
        config: AdaptShotConfig with pipeline settings

    Returns:
        Dictionary containing accuracy, latency, and metadata metrics
    """
    print("🧪 Running smoke test...")

    # Load few-shot data
    support_data, query_data = load_few_shot_split(
        dataset_name="cifar10",
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

    if args.smoke_test:
        results = run_smoke_test(config)

        # Print human-readable results
        print("\n📊 Smoke Test Results:")
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
        print("Example: python -m benchmarks.run_benchmark --smoke-test")
        return 0


if __name__ == "__main__":
    exit(main())