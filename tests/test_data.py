"""Phase-1 data pipeline skeleton tests (CPU-safe)."""

from __future__ import annotations

import torch

__all__ = [
    "test_dummy_batch_shape",
    "test_dummy_labels_shape",
    "test_dummy_labels_range",
]


def test_dummy_batch_shape(dummy_batch):
    """Ensure fixture batch shape is stable for later pipeline work."""
    assert dummy_batch.shape == (4, 3, 128, 128)


def test_dummy_labels_shape(dummy_labels):
    """Ensure fixture labels are one-dimensional."""
    assert dummy_labels.shape == (4,)


def test_dummy_labels_range(dummy_labels):
    """Ensure fixture labels are bounded to five classes."""
    assert torch.all(dummy_labels >= 0)
    assert torch.all(dummy_labels < 5)
