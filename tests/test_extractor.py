"""Unit tests for core/extractor.py and utils/determinism.py."""

import pytest
from PIL import Image
from typing import Any, Optional, cast
import torch

from src.adaptshot.config.settings import AdaptShotConfig
from src.adaptshot.core.extractor import extract_embedding, BackboneRegistry
from src.adaptshot.utils.determinism import verify_determinism


def test_resnet18_embedding_shape() -> None:
    """Verify ResNet-18 extracts 512-dimensional embeddings as expected."""
    config = AdaptShotConfig(backbone="resnet18", device="cpu", seed=42)
    dummy_img = Image.new("RGB", (224, 224), color="red")
    emb = extract_embedding(dummy_img, config)
    assert emb.shape == (512,), f"Expected (512,), got {emb.shape}"


def test_mobilenet_v3_embedding_shape() -> None:
    """Verify MobileNetV3-Small extracts 576-dimensional embeddings as expected."""
    config = AdaptShotConfig(backbone="mobilenet_v3_small", device="cpu", seed=42)
    dummy_img = Image.new("RGB", (224, 224), color="blue")
    emb = extract_embedding(dummy_img, config)
    assert emb.shape == (576,), f"Expected (576,), got {emb.shape}"


def test_deterministic_extraction() -> None:
    """Verify that extraction is bit-exact across multiple independent runs."""
    config = AdaptShotConfig(backbone="resnet18", device="cpu", seed=123)
    dummy_img = Image.new("RGB", (224, 224), color="green")

    def extract_fn() -> Any:
        return extract_embedding(dummy_img, config)

    is_deterministic = verify_determinism(extract_fn, runs=3, seed=123)
    assert is_deterministic, "Embedding extraction failed determinism check across 3 runs"


def test_backbone_registry_integrity() -> None:
    """Ensure registry contains expected backbones and factories are callable."""
    assert "resnet18" in BackboneRegistry
    assert "mobilenet_v3_small" in BackboneRegistry
    
    for name, factory in BackboneRegistry.items():
        factory_callable = cast(Any, factory)
        model = factory_callable()
        assert model is not None, f"Failed to instantiate backbone: {name}"
        # PyTorch defaults to training mode; extract_embedding() correctly sets .eval()
        assert isinstance(model, torch.nn.Module), f"Factory {name} did not return nn.Module"


def test_invalid_backbone_raises_value_error() -> None:
    """Ensure unknown backbone names fail fast with a clear error message."""
    config = AdaptShotConfig(backbone=cast(Any, "vit_tiny_patch16"), device="cpu", seed=42)
    dummy_img = Image.new("RGB", (224, 224), color="yellow")
    with pytest.raises(ValueError, match="Unknown backbone"):
        extract_embedding(dummy_img, config)


def test_backbone_factories_do_not_request_pretrained_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the backbone registry builds offline by default."""

    captured: dict[str, object] = {}

    def fake_resnet18(*, weights: Optional[object] = None) -> torch.nn.Module:
        captured["resnet18_weights"] = weights
        return torch.nn.Identity()

    def fake_mobilenet_v3_small(*, weights: Optional[object] = None) -> torch.nn.Module:
        captured["mobilenet_weights"] = weights
        return torch.nn.Identity()

    monkeypatch.setattr("src.adaptshot.core.extractor.models.resnet18", fake_resnet18)
    monkeypatch.setattr("src.adaptshot.core.extractor.models.mobilenet_v3_small", fake_mobilenet_v3_small)

    resnet_factory = cast(Any, BackboneRegistry["resnet18"])
    mobilenet_factory = cast(Any, BackboneRegistry["mobilenet_v3_small"])

    resnet_factory()
    mobilenet_factory()

    assert captured["resnet18_weights"] is None
    assert captured["mobilenet_weights"] is None