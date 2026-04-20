"""Evaluation metrics: ECE, latency, and accuracy helpers."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

__all__ = ["compute_ece", "benchmark_latency", "compute_accuracy"]


def compute_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Compute classification accuracy."""
    preds = np.asarray(predictions).reshape(-1)
    targets = np.asarray(labels).reshape(-1)
    if preds.shape != targets.shape:
        raise ValueError("predictions and labels must have identical shapes.")
    return float(np.mean(preds == targets))


def compute_ece(
    predictions: np.ndarray,
    confidences: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE bins samples by confidence, then aggregates the weighted absolute
    gap between bin confidence and bin accuracy.
    """
    preds = np.asarray(predictions).reshape(-1)
    confs = np.asarray(confidences, dtype=np.float64).reshape(-1)
    targets = np.asarray(labels).reshape(-1)

    if not (preds.shape == confs.shape == targets.shape):
        raise ValueError("predictions, confidences, and labels must have same shape.")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    confs = np.clip(confs, 0.0, 1.0)
    correctness = (preds == targets).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = float(len(confs))

    for idx in range(n_bins):
        lower = bin_edges[idx]
        upper = bin_edges[idx + 1]
        if idx == 0:
            in_bin = (confs >= lower) & (confs <= upper)
        else:
            in_bin = (confs > lower) & (confs <= upper)

        count = np.sum(in_bin)
        if count == 0:
            continue

        bin_confidence = float(np.mean(confs[in_bin]))
        bin_accuracy = float(np.mean(correctness[in_bin]))
        ece += (count / total) * abs(bin_accuracy - bin_confidence)

    return float(np.clip(ece, 0.0, 1.0))


def benchmark_latency(model: nn.Module, img_tensor: torch.Tensor, runs: int = 50) -> float:
    """
    Benchmark mean forward-pass latency in milliseconds.

    Performs 5 warmup runs, then averages `runs` timed runs.
    """
    if runs <= 0:
        raise ValueError("runs must be positive.")

    model.eval()
    device = next(model.parameters()).device

    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    if img_tensor.dim() != 4:
        raise ValueError("img_tensor must have shape [C,H,W] or [B,C,H,W].")

    inputs = img_tensor.to(device=device, non_blocking=False)

    with torch.no_grad():
        for _ in range(5):
            _ = model(inputs)

        start = time.perf_counter()
        for _ in range(runs):
            _ = model(inputs)
        end = time.perf_counter()

    return float(((end - start) / runs) * 1000.0)
