"""Pytest fixtures for deterministic CPU-first testing."""

from __future__ import annotations

import torch
import pytest

from src import configure_runtime, get_device

__all__ = ["runtime_config", "device", "dummy_batch", "dummy_labels"]


@pytest.fixture(scope="session")
def runtime_config():
    """Session-wide deterministic runtime settings."""
    return configure_runtime(seed=42, deterministic=True)


@pytest.fixture(scope="session")
def device(runtime_config):
    """Always return CPU for phase-1 baseline tests."""
    _ = runtime_config
    return get_device(prefer_gpu=False)


@pytest.fixture
def dummy_batch():
    """Deterministic tensor batch for unit tests."""
    return torch.randn(4, 3, 128, 128)


@pytest.fixture
def dummy_labels():
    """Deterministic integer labels for 5-way classification."""
    return torch.randint(0, 5, (4,))
