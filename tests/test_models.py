"""Phase-1 model skeleton tests (no network/download requirements)."""

from __future__ import annotations

import torch

__all__ = ["test_device_is_cpu", "test_linear_head_output_shape"]


def test_device_is_cpu(device):
    """Validate CPU-first execution policy."""
    assert device.type == "cpu"


def test_linear_head_output_shape(device, dummy_batch):
    """Validate a lightweight classifier head for shape correctness."""
    head = torch.nn.Linear(3 * 128 * 128, 5).to(device)
    flat_inputs = dummy_batch.to(device).reshape(dummy_batch.shape[0], -1)
    outputs = head(flat_inputs)
    assert outputs.shape == (4, 5)
