"""Immutable configuration dataclasses for AdaptShot."""

from dataclasses import dataclass
from typing import Literal, Optional

import torch


@dataclass(frozen=True)
class AdaptShotConfig:
    """
    Central, immutable configuration for the AdaptShot pipeline.

    Using a frozen dataclass guarantees that pipeline hyperparameters cannot be
    accidentally mutated during inference or training, which is critical for
    deterministic reproducibility and CI/CD validation.
    """
    # Core execution
    backbone: Literal["resnet18", "mobilenet_v3_small"] = "resnet18"
    device: Literal["cpu", "cuda", "mps"] = "cpu"  # CPU-first default
    seed: int = 42

    # Few-shot learning parameters
    n_way: int = 5          # Number of classes per episode
    k_shot: int = 10        # Support examples per class
    query_size: int = 15    # Query examples per class for evaluation

    # Similarity search
    use_faiss: bool = False # Toggle FAISS-CPU acceleration
    faiss_nprobe: int = 8   # FAISS IVF index probing depth (if used later)

    # Energy-aware inference
    eco_mode: bool = False
    early_exit_threshold: float = 0.95

    # Calibration & uncertainty
    calibration_method: Literal["temperature", "conformal", "none"] = "temperature"
    ece_n_bins: int = 15    # Number of bins for Expected Calibration Error
    temperature_init: float = 1.0

    # Memory management (UP-UGF)
    max_buffer_size: int = 100

    # Logging & debugging
    verbose: bool = True
    log_dir: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate configuration constraints immediately after creation."""
        if self.k_shot <= 0 or self.n_way <= 0:
            raise ValueError("n_way and k_shot must be positive integers.")
        if self.max_buffer_size < 10:
            raise ValueError("max_buffer_size must be >= 10 for meaningful few-shot operation.")
        if not 0.5 <= self.early_exit_threshold <= 1.0:
            raise ValueError("early_exit_threshold must be within [0.5, 1.0].")
        if self.device == "cuda" and not torch.cuda.is_available():
            import warnings
            warnings.warn(
                "CUDA requested but not available. Runtime logic will fall back to CPU.",
                RuntimeWarning
            )