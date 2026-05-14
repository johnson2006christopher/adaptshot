#!/usr/bin/env python3
"""Minimal benchmark harness for AdaptShot core pipeline."""

import argparse
import json
import time
from pathlib import Path

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
):
    """
    Load a few-shot split from a torchvision dataset (CIFAR-10 for smoke test).
    Returns support and query lists of (image_tensor, label) tuples.
    """
    set_deterministic_seed(seed)
    
    # CIFAR-10 is lightweight and built-in, perfect for validation
    dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transforms.ToTensor()
    )
    
    classes = list(range(n_way))
    support_indices, query_indices = [], []
    
    for cls in classes:
        # Get indices for the current class
        cls_indices = [i for i, (_, label) in enumerate(dataset) if label == cls]
        # Select k_shot for support, and 5 for query
        selected = np.random.choice(cls_indices, size=k_shot + 5, replace=False)
        support_indices.extend(selected[:k_shot])
        query_indices.extend(selected[k_shot:])
        
    support_data = [(dataset[i][0], dataset[i][1]) for i in support_indices]
    query_data = [(dataset[i][0], dataset[i][1]) for i in query_indices]
    return support_data, query_data


def run_smoke_test(config: AdaptShotConfig) -> dict:
    """Run minimal end-to-end pipeline and return metrics."""
    print("🧪 Running smoke test...")
    
    support_data, query_data = load_few_shot_split(
        dataset_name="cifar10", n_way=config.n_way, k_shot=config.k_shot, seed=config.seed
    )
    
    print(f"   • Support: {len(support_data)} examples")
    print(f"   • Query: {len(query_data)} examples")
    
    # Extract support embeddings
    print("   • Extracting support embeddings...")
    support_embeddings, support_labels = [], []
    
    start_time = time.perf_counter()
    for img_tensor, label in support_data:
        img_pil = ToPILImage()(img_tensor)
        emb = extract_embedding(img_pil, config)
        support_embeddings.append(emb)
        support_labels.append(label)
    
    support_embeddings = np.stack(support_embeddings)
    embedding_time = time.perf_counter() - start_time
    print(f"   • Embedding extraction: {embedding_time:.3f}s for {len(support_data)} images")

    # Evaluate on query set
    print("   • Evaluating on query set...")
    correct, latencies = 0, []
    
    for img_tensor, true_label in query_data:
        img_pil = ToPILImage()(img_tensor)
        start = time.perf_counter()
        pred_label, confidence, _ = find_nearest_neighbor(
            extract_embedding(img_pil, config),
            support_embeddings, np.array(support_labels), use_faiss=config.use_faiss
        )
        latencies.append((time.perf_counter() - start) * 1000)
        if pred_label == true_label: correct += 1

    return {
        "accuracy": float(correct / len(query_data)),
        "latency_avg_ms": float(np.mean(latencies)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "embedding_time_s": float(embedding_time),
        "support_size": len(support_data),
        "query_size": len(query_data),
        "config": {"backbone": config.backbone, "device": config.device, "use_faiss": config.use_faiss},
    }


def main():
    parser = argparse.ArgumentParser(description="AdaptShot Benchmark Harness")
    parser.add_argument("--smoke-test", action="store_true", help="Run minimal smoke test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="results/smoke_test.json", help="Output JSON path")
    args = parser.parse_args()

    set_deterministic_seed(args.seed)
    # Use ResNet18, CPU-only, no FAISS for smoke test to ensure it runs on any machine
    config = AdaptShotConfig(backbone="resnet18", device="cpu", seed=args.seed, n_way=5, k_shot=10, use_faiss=False)

    if args.smoke_test:
        results = run_smoke_test(config)
        
        print("\n📊 Smoke Test Results:")
        print(f"   • Accuracy: {results['accuracy']:.1%}")
        print(f"   • Avg Latency: {results['latency_avg_ms']:.1f} ms")
        print(f"   • P95 Latency: {results['latency_p95_ms']:.1f} ms")
        
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f: 
            json.dump(results, f, indent=2)
        print(f"   • Results saved to {output_path}")
        
        # Determinism check
        print("\n🔍 Verifying determinism...")
        def run_once():
            img_tensor, _ = load_few_shot_split(seed=config.seed)[1][0]
            return extract_embedding(ToPILImage()(img_tensor), config)
        
        is_det = verify_determinism(run_once, runs=3, seed=config.seed)
        print(f"   • Determinism check: {'✅ PASS' if is_det else '❌ FAIL'}")
        return 0 if is_det else 1
    
    print("Use --smoke-test to run validation.")
    return 0

if __name__ == "__main__":
    exit(main())