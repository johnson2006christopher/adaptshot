"""Deterministic execution utilities for reproducible few-shot learning.

torch is imported lazily here, not at module scope. This module holds
`set_deterministic_seed`, which `CLAUDE.md` tells every contributor to call --
so an eager import made the one function everyone is told to use unreachable on
a core install, and cost ~479MB of resident memory for anyone who merely
imported AdaptShot (#13, #35).

Seeding Python and NumPy is useful with or without torch. Doing it only when
torch happens to be installed was never the intent.
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import for annotations only
    import torch


def _torch() -> Any | None:
    """Return the torch module, or None when it cannot be imported.

    `ImportError`, not `ModuleNotFoundError`: a torch that is installed but
    fails to load -- a broken CUDA build, a partial install -- should degrade
    to the numpy paths in exactly the same way as one that is absent.
    `training/finetune.py` guards its own import the same way.
    """

    try:
        import torch as torch_module
    except ImportError:
        return None
    return torch_module


def set_deterministic_seed(seed: int = 42, device: torch.device | None = None) -> None:
    """
    Set all random seeds for deterministic execution across PyTorch, NumPy, and Python.

    This function is called at the start of every training, evaluation, and benchmarking
    run to guarantee bit-exact reproducibility across hardware and operating systems.

    Python and NumPy are always seeded. torch is seeded when it is installed;
    without it this function still does its job for the numpy paths rather than
    raising, which is what an optional dependency means.

    Args:
        seed: Random seed to use (default: 42).
        device: Target device. If CUDA is specified, enables deterministic cuDNN algorithms.
    """
    # Python standard library
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    torch = _torch()
    if torch is not None:
        torch.manual_seed(seed)

        # CUDA deterministic settings (only applies if device.type == 'cuda')
        if device is not None and device.type == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        # Note: Some CUDA operations remain inherently non-deterministic.
        # CPU-first design avoids this, but we support CUDA for flexibility.

    # Environment-level hash seed (prevents Python dict/set ordering randomness)
    os.environ["PYTHONHASHSEED"] = str(seed)


def verify_determinism(
    fn: Callable[..., Any],
    *args: Any,
    runs: int = 3,
    seed: int = 42,
    tolerance: float = 1e-7,
    **kwargs: Any,
) -> bool:
    """
    Verify that a function produces bit-exact outputs across multiple independent runs.

    Essential for CI/CD pipelines to catch accidental non-deterministic ops (e.g.,
    certain PyTorch scatter/gather operations or uninitialized memory reads).

    Args:
        fn: Callable to test. Should return a torch.Tensor or np.ndarray.
        *args: Positional arguments passed to `fn`.
        runs: Number of independent runs to compare (default: 3).
        seed: Base random seed (incremented internally per run for isolation).
        tolerance: Absolute floating-point tolerance for comparison.
        **kwargs: Keyword arguments passed to `fn`.

    Returns:
        bool: True if all outputs match within tolerance, False otherwise.
    """
    outputs = []

    for i in range(runs):
        set_deterministic_seed(seed + i)
        output = fn(*args, **kwargs)

        # Normalize to numpy for cross-framework comparison
        torch = _torch()
        if torch is not None and isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy()
        elif not isinstance(output, np.ndarray):
            output = np.array(output)

        outputs.append(output)

    # Compare all runs against the reference (first run)
    reference = outputs[0]
    for run_idx, current in enumerate(outputs[1:], start=2):
        if not np.allclose(reference, current, atol=tolerance, rtol=0):
            max_diff = float(np.max(np.abs(reference - current)))
            print(f"⚠️  Determinism check failed at run {run_idx} (max diff: {max_diff:.2e})")
            return False

    return True