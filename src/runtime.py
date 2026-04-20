"""Runtime utilities for deterministic execution and device selection."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch

__all__ = ["RuntimeConfig", "configure_runtime", "get_device"]


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime configuration returned by `configure_runtime`."""

    seed: int
    deterministic: bool
    device: torch.device


def get_device(prefer_gpu: bool = False) -> torch.device:
    """Return the selected device with CPU as the default and fallback."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def configure_runtime(seed: int = 42, deterministic: bool = True) -> RuntimeConfig:
    """Configure deterministic behavior across Python, NumPy, and PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    return RuntimeConfig(seed=seed, deterministic=deterministic, device=get_device(False))
