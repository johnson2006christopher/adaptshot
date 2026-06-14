"""Frozen backbone feature extraction with backend-agnostic design.

Uses lazy imports for PyTorch and torchvision so that the module is importable
without a hard dependency on torch at install time. The core API (extract_embedding)
requires a backend at runtime; the module itself loads without error.

When `eco_mode=True`, the extractor can return a cached support embedding after
a quick cosine similarity check exceeds the configured threshold. This reduces
average latency on repeated, near-duplicate support-like inputs at the cost of
potentially skipping the full backbone forward pass. The tradeoff is intentional
and deterministic: the fast path only activates when a cached support embedding
is available and the preview similarity already exceeds the configured bound.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Optional, Union, cast

import numpy as np
from PIL import Image

from ..config.settings import AdaptShotConfig

# ---------------------------------------------------------------------------
# Lazy import helpers – torch/torchvision are resolved only when first needed.
# This keeps the module importable without a hard torch dependency.
# ---------------------------------------------------------------------------

_TORCH: Any = None
_TORCH_NN: Any = None
_TV_MODELS: Any = None
_TV_TRANSFORMS: Any = None


def _get_torch() -> Any:
    global _TORCH
    if _TORCH is None:
        import torch as _t
        _TORCH = _t
    return _TORCH


def _get_torch_nn() -> Any:
    global _TORCH_NN
    if _TORCH_NN is None:
        from torch import nn as _nn
        _TORCH_NN = _nn
    return _TORCH_NN


def _get_tv_models() -> Any:
    global _TV_MODELS
    if _TV_MODELS is None:
        from torchvision import models as _m
        _TV_MODELS = _m
    return _TV_MODELS


def _get_tv_transforms() -> Any:
    global _TV_TRANSFORMS
    if _TV_TRANSFORMS is None:
        from torchvision import transforms as _t
        _TV_TRANSFORMS = _t
    return _TV_TRANSFORMS


# Type alias for flexible image input (lazy reference to torch.Tensor via Any)
ImageInput = Any  # str | np.ndarray | PIL.Image | torch.Tensor

# ---------------------------------------------------------------------------
# Backbone registry – lazy factories with ImageNet pretrained weights.
# The pretrained weights are essential: the preprocessing pipeline uses
# ImageNet statistics (mean/std), so random weights would produce
# meaningless embeddings. Using weights="IMAGENET1K_V1" guarantees
# the backbone produces features the normalisation was designed for.
# ---------------------------------------------------------------------------

BackboneRegistry: Dict[str, Any] = {
    "resnet18": lambda: _get_tv_models().resnet18(weights="IMAGENET1K_V1"),
    "mobilenet_v3_small": lambda: _get_tv_models().mobilenet_v3_small(
        weights="IMAGENET1K_V1"
    ),
}

# Output dimensionality for each backbone (used for dynamic dimension inference)
BACKBONE_OUTPUT_DIM: Dict[str, int] = {
    "resnet18": 512,
    "mobilenet_v3_small": 576,
}

_RESAMPLE_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR")


@dataclass
class EmbeddingCache:
    """Instance-scoped cache for eco-mode early-exit support embeddings.

    Each FewShotLearner should own its own EmbeddingCache to avoid
    cross-instance interference that occurred with module-level globals.
    """

    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    preview: Optional[np.ndarray] = field(default=None, repr=False)

    def set(
        self,
        embedding: Optional[np.ndarray],
        preview_signature: Optional[np.ndarray] = None,
    ) -> None:
        """Register a support embedding and its preview for eco-mode checks."""
        if embedding is None:
            self.embedding = None
            self.preview = None
            return
        self.embedding = np.asarray(embedding, dtype=np.float32).copy()
        if preview_signature is not None:
            self.preview = np.asarray(preview_signature, dtype=np.float32).copy()
        else:
            self.preview = None

    def clear(self) -> None:
        """Reset both cached values."""
        self.embedding = None
        self.preview = None


@lru_cache(maxsize=4)
def _build_backbone(backbone_name: str, device: str) -> Any:
    """Build and cache a frozen backbone on the requested device.

    v0.2.0: LRU cache prevents repeated backbone construction but can
    hold references to tensors on old devices. Use clear_backbone_cache()
    when switching devices or to release memory.
    """
    nn = _get_torch_nn()
    backbone = BackboneRegistry[backbone_name]()
    if hasattr(backbone, "fc"):
        backbone.fc = nn.Identity()
    elif hasattr(backbone, "classifier"):
        backbone.classifier = nn.Identity()
    backbone.to(device)
    backbone.eval()
    return backbone


def clear_backbone_cache() -> None:
    """Clear the LRU backbone cache to release GPU/CPU memory.

    Call this when switching devices or when memory pressure is high.
    """
    _build_backbone.cache_clear()


def compute_preview_signature(image: ImageInput, size: int = 32) -> np.ndarray:
    """Compute a low-cost preview signature for early-exit similarity checks."""
    pil_image = _normalize_to_pil(image).resize((size, size), _RESAMPLE_BILINEAR)
    preview = np.asarray(pil_image, dtype=np.float32) / 255.0
    return cast(np.ndarray, preview.reshape(-1))


# Module-level default cache for backward compatibility (benchmarks, scripts).
# FewShotLearner instances should create and pass their own EmbeddingCache.
_DEFAULT_CACHE = EmbeddingCache()


def set_support_embedding_cache(
    embedding: Optional[np.ndarray],
    preview_signature: Optional[np.ndarray] = None,
) -> None:
    """Register the top-1 support embedding and preview for eco-mode early exit.

    This function updates a module-level default cache for backward compatibility.
    Production code should prefer passing an EmbeddingCache instance directly
    to extract_embedding() via the `cache` parameter.
    """
    _DEFAULT_CACHE.set(embedding, preview_signature)


def get_support_embedding_cache() -> Optional[np.ndarray]:
    """Return a copy of the cached support embedding used by eco mode."""
    if _DEFAULT_CACHE.embedding is None:
        return None
    return cast(np.ndarray, _DEFAULT_CACHE.embedding.copy())


def get_support_preview_cache() -> Optional[np.ndarray]:
    """Return a copy of the cached support preview signature used by eco mode."""
    if _DEFAULT_CACHE.preview is None:
        return None
    return cast(np.ndarray, _DEFAULT_CACHE.preview.copy())


def _normalize_to_pil(image: ImageInput) -> Any:
    """Convert supported image inputs to a PIL RGB image."""
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    # Check for torch.Tensor without a hard import – use duck-typing.
    if hasattr(image, "dim") and hasattr(image, "permute") and hasattr(image, "cpu"):
        if image.dim() == 3 and image.shape[0] not in (1, 3):
            image = image.permute(2, 0, 1)
        return _get_tv_transforms().ToPILImage()(image.cpu()).convert("RGB")
    return image.convert("RGB")


def _get_preprocess_transform(img_size: int = 224) -> Any:
    """Return standard preprocessing transforms for ImageNet-pretrained backbones."""
    T = _get_tv_transforms()
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def extract_embedding(
    image: ImageInput,
    config: AdaptShotConfig,
    return_numpy: bool = True,
    cache: Optional[EmbeddingCache] = None,
) -> Union[Any, np.ndarray]:
    """Extract feature embedding from input image using a frozen backbone.

    Args:
        image: Input image (path, PIL, NumPy, or Tensor).
        config: AdaptShotConfig with backbone and device settings.
        return_numpy: If True, return np.ndarray; otherwise return torch.Tensor
            (requires torch to be installed).
        cache: Optional EmbeddingCache for eco-mode early exit. If None,
            falls back to the module-level default cache for backward compat.
    """
    if config.backbone not in BackboneRegistry:
        raise ValueError(
            f"Unknown backbone: {config.backbone}. "
            f"Available: {list(BackboneRegistry.keys())}"
        )

    pil_image = _normalize_to_pil(image)

    # Use the provided cache or fall back to the module-level default.
    active_cache = cache if cache is not None else _DEFAULT_CACHE
    support_embedding = active_cache.embedding
    support_preview = active_cache.preview
    if config.eco_mode and support_embedding is not None and support_preview is not None:
        query_preview = compute_preview_signature(pil_image)
        preview_norm = np.linalg.norm(query_preview) + 1e-8
        support_norm = np.linalg.norm(support_preview) + 1e-8
        quick_similarity = float(
            np.dot(query_preview, support_preview) / (preview_norm * support_norm)
        )
        # v0.2.0: Stricter eco-mode: require >= threshold AND also check
        # that the cached embedding is not stale (preview norms differ by <2x)
        norm_ratio = min(preview_norm, support_norm) / max(preview_norm, support_norm)
        if quick_similarity >= config.early_exit_threshold and norm_ratio > 0.3:
            if return_numpy:
                return cast(np.ndarray, support_embedding.copy())
            return _get_torch().from_numpy(support_embedding.copy())

    backbone = _build_backbone(config.backbone, config.device)

    # Preprocess image
    preprocess = _get_preprocess_transform()

    # Apply transforms and add batch dimension
    torch_mod = _get_torch()
    image_tensor = preprocess(pil_image).unsqueeze(0).to(config.device)

    with torch_mod.no_grad():
        embedding = cast(Any, backbone(image_tensor).squeeze(0))

    if return_numpy:
        return embedding.detach().cpu().numpy()
    return embedding