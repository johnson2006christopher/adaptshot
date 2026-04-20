"""Automated benchmarking for AdaptShot latency, ECE, and accuracy."""

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import benchmark_latency as _metric_latency
from src.evaluation.metrics import compute_accuracy, compute_ece
from src.models.network import create_fewshot_model
from src.training.feedback import ReplayBuffer

__all__ = [
    "benchmark_latency",
    "benchmark_ece_on_fewshot",
    "run_full_benchmark",
]


def benchmark_latency(model: torch.nn.Module, img_tensor: torch.Tensor, runs: int = 100) -> float:
    """Benchmark mean latency in milliseconds using shared metrics helper."""
    return _metric_latency(model=model, img_tensor=img_tensor, runs=runs)


def benchmark_ece_on_fewshot(
    model: torch.nn.Module,
    loader: DataLoader,
    n_bins: int = 10,
) -> float:
    """Compute ECE over episodic few-shot batches."""
    model.eval()
    preds = []
    confs = []
    labels = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for images, batch_labels in loader:
            logits = model(images.to(device=device, non_blocking=False))
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            batch_preds = np.argmax(probs, axis=1)
            batch_confs = np.max(probs, axis=1)

            preds.extend(batch_preds.tolist())
            confs.extend(batch_confs.tolist())
            labels.extend(batch_labels.detach().cpu().numpy().tolist())

    return compute_ece(
        predictions=np.asarray(preds),
        confidences=np.asarray(confs),
        labels=np.asarray(labels),
        n_bins=n_bins,
    )


def run_full_benchmark(
    model: torch.nn.Module,
    embedding_extractor: Callable[[torch.nn.Module, torch.Tensor], np.ndarray],
    buffer: ReplayBuffer,
    test_loader: DataLoader,
) -> Dict[str, float]:
    """
    Run latency, ECE, and accuracy benchmark and print a formatted summary.

    Returns:
        dict: {'latency_ms', 'ece', 'accuracy', 'buffer_size'}
    """
    _ = embedding_extractor
    device = next(model.parameters()).device
    sample_image, _ = next(iter(test_loader))
    sample_input = sample_image[0].unsqueeze(0).to(device=device, non_blocking=False)
    latency_ms = benchmark_latency(model, sample_input, runs=100)

    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for images, batch_labels in test_loader:
            logits = model(images.to(device=device, non_blocking=False))
            pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
            preds.extend(pred.tolist())
            labels.extend(batch_labels.detach().cpu().numpy().tolist())
    accuracy = compute_accuracy(np.asarray(preds), np.asarray(labels))
    ece = benchmark_ece_on_fewshot(model=model, loader=test_loader, n_bins=10)

    result = {
        "latency_ms": float(latency_ms),
        "ece": float(ece),
        "accuracy": float(accuracy),
        "buffer_size": float(len(buffer)),
    }

    print("+----------------+-----------+")
    print("| Metric         | Value     |")
    print("+----------------+-----------+")
    print(f"| latency_ms     | {result['latency_ms']:<9.3f}|")
    print(f"| ece            | {result['ece']:<9.4f}|")
    print(f"| accuracy       | {result['accuracy']:<9.4f}|")
    print(f"| buffer_size    | {int(result['buffer_size']):<9d}|")
    print("+----------------+-----------+")
    return result


def _dry_run_loader(batch_size: int = 8, samples: int = 32) -> DataLoader:
    """Create deterministic synthetic loader for quick benchmark validation."""
    torch.manual_seed(42)
    images = torch.randn(samples, 3, 128, 128)
    labels = torch.randint(low=0, high=5, size=(samples,))
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=False, num_workers=0)


def main() -> None:
    """CLI entrypoint for benchmark execution."""
    parser = argparse.ArgumentParser(description="AdaptShot benchmark runner")
    parser.add_argument("--dry-run", action="store_true", help="Run synthetic quick benchmark.")
    args = parser.parse_args()

    torch.manual_seed(42)
    model = create_fewshot_model(num_classes=5, device=torch.device("cpu"))
    buffer = ReplayBuffer(capacity=100)
    loader = _dry_run_loader() if args.dry_run else _dry_run_loader()
    _ = run_full_benchmark(model=model, embedding_extractor=lambda _m, _x: np.zeros(512), buffer=buffer, test_loader=loader)


if __name__ == "__main__":
    main()
