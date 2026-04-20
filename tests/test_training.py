"""Phase-1 training/eval skeleton tests for deterministic behavior."""

from __future__ import annotations

import numpy as np
import torch

from src import configure_runtime

__all__ = ["test_numpy_seed_determinism", "test_torch_seed_determinism"]


def test_numpy_seed_determinism():
    """Validate deterministic NumPy state under the runtime manager."""
    configure_runtime(seed=7, deterministic=True)
    first = np.random.rand(5)
    configure_runtime(seed=7, deterministic=True)
    second = np.random.rand(5)
    assert np.allclose(first, second)


def test_torch_seed_determinism():
    """Validate deterministic PyTorch RNG under the runtime manager."""
    configure_runtime(seed=11, deterministic=True)
    first = torch.randn(3, 3)
    configure_runtime(seed=11, deterministic=True)
    second = torch.randn(3, 3)
    assert torch.allclose(first, second)
