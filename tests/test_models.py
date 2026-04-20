"""Phase-3 tests for model construction, embeddings, and calibration."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torchvision.models import resnet18  # type: ignore[import-untyped]

import src.models.network as network_module
from src.evaluation.metrics import benchmark_latency, compute_ece
from src.models.embedding import compute_cosine_similarity, extract_embedding
from src.models.network import create_fewshot_model

__all__ = [
    "test_create_fewshot_model_freezes_backbone",
    "test_extract_embedding_shape",
    "test_compute_cosine_similarity_range",
    "test_compute_ece_perfect_predictions",
    "test_benchmark_latency_runs_on_cpu",
]


@pytest.fixture
def patch_resnet18(monkeypatch):
    """Patch pretrained loader for deterministic offline-safe tests."""

    def local_resnet18(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args, kwargs
        return resnet18(weights=None)

    monkeypatch.setattr(network_module, "resnet18", local_resnet18)


def test_create_fewshot_model_freezes_backbone(patch_resnet18):
    """Only final FC head should be trainable."""
    _ = patch_resnet18
    torch.manual_seed(42)
    model = create_fewshot_model(num_classes=5, device=torch.device("cpu"))

    for name, parameter in model.named_parameters():
        if name.startswith("fc."):
            assert parameter.requires_grad
        else:
            assert not parameter.requires_grad


def test_extract_embedding_shape(patch_resnet18):
    """Embedding extractor should return a 512-dim CPU NumPy vector."""
    _ = patch_resnet18
    model = create_fewshot_model(num_classes=5, device=torch.device("cpu"))
    image = torch.randn(3, 128, 128)

    embedding = extract_embedding(model, image)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (512,)


def test_compute_cosine_similarity_range():
    """Cosine similarities should remain bounded in [-1, 1]."""
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    support = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    sims = compute_cosine_similarity(query, support)
    assert sims.shape == (3,)
    assert np.all(sims >= -1.0)
    assert np.all(sims <= 1.0)


def test_compute_ece_perfect_predictions():
    """ECE should be exactly 0 for perfect predictions with confidence 1."""
    predictions = np.array([0, 1, 2, 3], dtype=np.int64)
    labels = np.array([0, 1, 2, 3], dtype=np.int64)
    confidences = np.ones(4, dtype=np.float64)

    ece = compute_ece(predictions=predictions, confidences=confidences, labels=labels, n_bins=10)
    assert ece == 0.0


def test_benchmark_latency_runs_on_cpu(patch_resnet18):
    """Latency benchmark should execute and return non-negative milliseconds."""
    _ = patch_resnet18
    model = create_fewshot_model(num_classes=5, device=torch.device("cpu"))
    image = torch.randn(1, 3, 128, 128)

    latency_ms = benchmark_latency(model, image, runs=5)
    assert latency_ms >= 0.0
