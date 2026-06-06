"""Frozen backbone feature extraction with TorchScript compatibility.

When `eco_mode=True`, the extractor can return a cached support embedding after
a quick cosine similarity check exceeds the configured threshold. This reduces
average latency on repeated, near-duplicate support-like inputs at the cost of
potentially skipping the full backbone forward pass. The tradeoff is intentional
and deterministic: the fast path only activates when a cached support embedding
is available and the preview similarity already exceeds the configured bound.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from importlib import import_module
from typing import Any, Dict, Optional, Union, cast

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ..config.settings import AdaptShotConfig

# Type alias for flexible image input
ImageInput = Union[str, np.ndarray, Image.Image, torch.Tensor]

# Registry for backbone factories (extensible without modifying core logic)
models = import_module("torchvision.models")
transforms = import_module("torchvision.transforms")
BackboneRegistry: Dict[str, Any] = {
    "resnet18": lambda: models.resnet18(weights=None),
    "mobilenet_v3_small": lambda: models.mobilenet_v3_small(weights=None),
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
    """Build and cache a frozen backbone on the requested device."""

    backbone = BackboneRegistry[backbone_name]()
    if hasattr(backbone, "fc"):
        backbone.fc = nn.Identity()
    elif hasattr(backbone, "classifier"):
        backbone.classifier = nn.Identity()
    backbone.to(device)
    backbone.eval()
    return backbone


def compute_preview_signature(image: ImageInput, size: int = 16) -> np.ndarray:
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
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] not in (1, 3):
            image = image.permute(2, 0, 1)
        return transforms.ToPILImage()(image.cpu()).convert("RGB")
    return image.convert("RGB")


def _get_preprocess_transform(img_size: int = 224) -> Any:
    """Return standard preprocessing transforms for ImageNet-pretrained backbones."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def extract_embedding(
    image: ImageInput,
    config: AdaptShotConfig,
    return_numpy: bool = True,
    cache: Optional[EmbeddingCache] = None,
) -> Union[torch.Tensor, np.ndarray]:
    """Extract feature embedding from input image using a frozen backbone.

    Args:
        image: Input image (path, PIL, NumPy, or Tensor).
        config: AdaptShotConfig with backbone and device settings.
        return_numpy: If True, return np.ndarray; otherwise return torch.Tensor.
        cache: Optional EmbeddingCache for eco-mode early exit. If None,
            falls back to the module-level default cache for backward compat.
    """
    # Load backbone from registry
    if config.backbone not in BackboneRegistry:
        raise ValueError(f"Unknown backbone: {config.backbone}. Available: {list(BackboneRegistry.keys())}")

    pil_image = _normalize_to_pil(image)

    # Use the provided cache or fall back to the module-level default.
    active_cache = cache if cache is not None else _DEFAULT_CACHE
    support_embedding = active_cache.embedding
    support_preview = active_cache.preview
    if config.eco_mode and support_embedding is not None and support_preview is not None:
        query_preview = compute_preview_signature(pil_image)
        preview_norm = np.linalg.norm(query_preview) + 1e-8
        support_norm = np.linalg.norm(support_preview) + 1e-8
        quick_similarity = float(np.dot(query_preview, support_preview) / (preview_norm * support_norm))
        if quick_similarity >= config.early_exit_threshold:
            if return_numpy:
                return cast(np.ndarray, support_embedding.copy())
            return torch.from_numpy(support_embedding.copy())

    backbone = _build_backbone(config.backbone, config.device)

    # Preprocess image
    preprocess = _get_preprocess_transform()

    # Apply transforms and add batch dimension
    image_tensor = preprocess(pil_image).unsqueeze(0).to(config.device)

    with torch.no_grad():
        embedding = cast(torch.Tensor, backbone(image_tensor).squeeze(0))

    if return_numpy:
        return embedding.detach().cpu().numpy()
    return embedding