"""Immutable configuration dataclasses for AdaptShot."""

from dataclasses import dataclass
from typing import Literal


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
    similarity_metric: Literal["cosine", "euclidean"] = "euclidean"
    inference_mode: Literal["nearest_neighbor", "prototypical", "contrastive"] = "prototypical"

    # Energy-aware inference
    eco_mode: bool = False
    early_exit_threshold: float = 0.95

    # Calibration & uncertainty
    calibration_method: Literal["temperature", "scaling_binning", "conformal", "none"] = "temperature"
    ece_n_bins: int = 15    # Number of bins for Expected Calibration Error
    calibration_eval_bins: int = 100
    temperature_init: float = 1.0
    recalibrate_after_feedback: bool = True

    # OOD detection
    enable_ood_detection: bool = True
    ood_threshold_quantile: float = 0.98
    ood_absolute_min_distance: float = 0.25

    # Conformal prediction (v0.2.0)
    conformal_alpha: float = 0.05
    conformal_mode: Literal["split", "cross"] = "split"

    # Advanced uncertainty (v0.2.0)
    uncertainty_mode: Literal["mcdropout", "entropy", "mahalanobis", "ensemble"] = "ensemble"

    # Explainability (v0.2.0)
    explainability_enabled: bool = True

    # Memory management (UP-UGF)
    max_buffer_size: int = 100

    # Logging & debugging
    verbose: bool = True
    log_dir: str | None = None

    def __post_init__(self) -> None:
        """Validate configuration constraints immediately after creation."""
        if self.k_shot <= 0 or self.n_way <= 0:
            raise ValueError("n_way and k_shot must be positive integers.")
        if self.max_buffer_size < 10:
            raise ValueError("max_buffer_size must be >= 10 for meaningful few-shot operation.")
        if not 0.5 <= self.early_exit_threshold <= 1.0:
            raise ValueError("early_exit_threshold must be within [0.5, 1.0].")
        if self.ece_n_bins <= 1:
            raise ValueError("ece_n_bins must be > 1.")
        if self.calibration_eval_bins < self.ece_n_bins:
            raise ValueError("calibration_eval_bins must be >= ece_n_bins.")
        if not 0.5 <= self.ood_threshold_quantile <= 1.0:
            raise ValueError("ood_threshold_quantile must be in [0.5, 1.0].")
        if self.ood_absolute_min_distance < 0.0:
            raise ValueError("ood_absolute_min_distance must be >= 0.0.")
        if not 0.0 < self.conformal_alpha < 1.0:
            raise ValueError("conformal_alpha must be in (0.0, 1.0).")
        if self.conformal_mode not in ("split", "cross"):
            raise ValueError("conformal_mode must be 'split' or 'cross'.")
        if self.device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    import warnings
                    warnings.warn(
                        "CUDA requested but not available. "
                        "Runtime logic will fall back to CPU.",
                        RuntimeWarning,
                    )
            except ImportError:
                import warnings
                warnings.warn(
                    "CUDA requested but PyTorch is not installed. "
                    "Install with: pip install 'adaptshot[torch]'",
                    RuntimeWarning,
                )
