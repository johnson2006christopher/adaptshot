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
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image

from ..config.settings import AdaptShotConfig
from ..utils.exceptions import BackboneError

# ---------------------------------------------------------------------------
# Lazy import helpers – torch/torchvision are resolved only when first needed.
# This keeps the module importable without a hard torch dependency.
# ---------------------------------------------------------------------------

# `lru_cache` rather than a module global guarded by `global`: it memoises on the
# first call exactly as the hand-written version did, but the cache is the
# function's own rather than a name any other code in the module could rebind.
# The import still happens once, and still only when first asked for.


@lru_cache(maxsize=1)
def _get_torch() -> Any:
    import torch

    return torch


@lru_cache(maxsize=1)
def _get_torch_nn() -> Any:
    from torch import nn

    return nn


@lru_cache(maxsize=1)
def _get_tv_models() -> Any:
    from torchvision import models

    return models


@lru_cache(maxsize=1)
def _get_tv_transforms() -> Any:
    from torchvision import transforms

    return transforms


# Type alias for flexible image input (lazy reference to torch.Tensor via Any)
ImageInput = Any  # str | np.ndarray | PIL.Image | torch.Tensor

# ---------------------------------------------------------------------------
# Backbone registry – lazy factories with ImageNet pretrained weights.
# The pretrained weights are essential: the preprocessing pipeline uses
# ImageNet statistics (mean/std), so random weights would produce
# meaningless embeddings. Using weights="IMAGENET1K_V1" guarantees
# the backbone produces features the normalisation was designed for.
# ---------------------------------------------------------------------------

BackboneRegistry: dict[str, Any] = {
    "resnet18": lambda: _get_tv_models().resnet18(weights="IMAGENET1K_V1"),
    "mobilenet_v3_small": lambda: _get_tv_models().mobilenet_v3_small(
        weights="IMAGENET1K_V1"
    ),
}

# Output dimensionality for each backbone (used for dynamic dimension inference)
BACKBONE_OUTPUT_DIM: dict[str, int] = {
    "resnet18": 512,
    "mobilenet_v3_small": 576,
}

_RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


@dataclass
class EmbeddingCache:
    """Instance-scoped cache for eco-mode early-exit support embeddings.

    Each FewShotLearner should own its own EmbeddingCache to avoid
    cross-instance interference that occurred with module-level globals.
    """

    embedding: np.ndarray | None = field(default=None, repr=False)
    preview: np.ndarray | None = field(default=None, repr=False)

    def set(
        self,
        embedding: np.ndarray | None,
        preview_signature: np.ndarray | None = None,
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


_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


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
    embedding: np.ndarray | None,
    preview_signature: np.ndarray | None = None,
) -> None:
    """Register the top-1 support embedding and preview for eco-mode early exit.

    This function updates a module-level default cache for backward compatibility.
    Production code should prefer passing an EmbeddingCache instance directly
    to extract_embedding() via the `cache` parameter.
    """
    _DEFAULT_CACHE.set(embedding, preview_signature)


def get_support_embedding_cache() -> np.ndarray | None:
    """Return a copy of the cached support embedding used by eco mode."""
    if _DEFAULT_CACHE.embedding is None:
        return None
    return _DEFAULT_CACHE.embedding.copy()


def get_support_preview_cache() -> np.ndarray | None:
    """Return a copy of the cached support preview signature used by eco mode."""
    if _DEFAULT_CACHE.preview is None:
        return None
    return _DEFAULT_CACHE.preview.copy()


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


def _torch_is_available() -> bool:
    """Whether torch can actually be imported, not merely whether it is listed."""

    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def onnx_weights_available(backbone_name: str) -> bool:
    """Whether a bundled ONNX graph exists for this backbone.

    `resnet18` is 44.8MB of weights and `mobilenet_v3_small` is 4.0MB, so which
    ones ship is a packaging decision (#36). This asks what is actually present
    rather than what ought to be.
    """

    return (_DATA_DIR / f"{backbone_name}.onnx").is_file()


def _should_use_onnx(backbone_name: str, return_numpy: bool) -> bool:
    """Decide between the ONNX and torch paths.

    ONNX wins when its weights are bundled and the caller wants numpy back.
    Without torch that is the only option; with torch it is still the better one,
    since the embeddings agree and ONNX is faster on CPU.

    `return_numpy=False` asks for a torch tensor by name, so it always takes the
    torch path -- an ONNX session cannot produce one.
    """

    if not return_numpy:
        return False
    if not onnx_weights_available(backbone_name):
        return False
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def bundled_onnx_backbones() -> list[str]:
    """Backbone names whose ONNX weights ship inside the installed package."""

    return sorted(
        path.stem for path in _DATA_DIR.glob("*.onnx") if path.stem in BackboneRegistry
    )


def _require_a_usable_backend(backbone_name: str, return_numpy: bool) -> None:
    """Fail with an actionable message instead of a bare ``ImportError``.

    Before #36 every backbone needed torch, so a core install failed uniformly
    and obviously. Now the answer varies by backbone, and the fall-through path
    raised ``ImportError: No module named 'torch'`` from four frames inside
    ``_get_torch_nn`` -- true, but it named neither the backbone that caused it
    nor either of the two ways out.
    """

    if _torch_is_available():
        return

    if not return_numpy:
        raise BackboneError(
            "return_numpy=False asks for a torch.Tensor, which requires torch: "
            "pip install 'adaptshot[training]'"
        )

    bundled = bundled_onnx_backbones()
    suggestion = (
        f"use one of the bundled backbones ({', '.join(bundled)})"
        if bundled
        else "reinstall adaptshot -- no bundled ONNX weights were found"
    )
    raise BackboneError(
        f"Backbone {backbone_name!r} needs PyTorch on this install: its ONNX "
        f"weights are not bundled. Either {suggestion}, or install torch with "
        "pip install 'adaptshot[training]'."
    )


@lru_cache(maxsize=1)
def _onnx_backend() -> Any:
    """The process-wide ONNX backend, built on first use.

    It caches its own sessions per backbone; constructing one per call would
    reload the graph every time.
    """

    from .backends.onnx_backend import ONNXBackend

    return ONNXBackend()


def extract_embedding(
    image: ImageInput,
    config: AdaptShotConfig,
    return_numpy: bool = True,
    cache: EmbeddingCache | None = None,
) -> Any | np.ndarray:
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
                return support_embedding.copy()
            return _get_torch().from_numpy(support_embedding.copy())

    if _should_use_onnx(config.backbone, return_numpy):
        # The backend priority `backends/__init__.py` has documented since v0.2.0,
        # finally wired up (#36). ONNX is preferred when its weights are bundled
        # and the caller wants numpy back, because it is what makes inference work
        # without torch at all -- and it is not a compromise: embeddings agree with
        # torch to ~4e-06 (cosine 0.99999994), and it is faster on CPU.
        return _onnx_backend().extract(pil_image, config.backbone)

    _require_a_usable_backend(config.backbone, return_numpy)

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