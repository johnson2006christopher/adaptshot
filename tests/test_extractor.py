"""Unit tests for core/extractor.py and utils/determinism.py."""

import pytest
from PIL import Image

from src.adaptshot.config.settings import AdaptShotConfig
from src.adaptshot.core.extractor import extract_embedding, BackboneRegistry
from src.adaptshot.utils.determinism import verify_determinism


def test_resnet18_embedding_shape():
    """Verify ResNet-18 extracts 512-dimensional embeddings as expected."""
    config = AdaptShotConfig(backbone="resnet18", device="cpu", seed=42)
    dummy_img = Image.new("RGB", (224, 224), color="red")
    emb = extract_embedding(dummy_img, config)
    assert emb.shape == (512,), f"Expected (512,), got {emb.shape}"


def test_mobilenet_v3_embedding_shape():
    """Verify MobileNetV3-Small extracts 576-dimensional embeddings as expected."""
    config = AdaptShotConfig(backbone="mobilenet_v3_small", device="cpu", seed=42)
    dummy_img = Image.new("RGB", (224, 224), color="blue")
    emb = extract_embedding(dummy_img, config)
    assert emb.shape == (576,), f"Expected (576,), got {emb.shape}"


def test_deterministic_extraction():
    """Verify that extraction is bit-exact across multiple independent runs."""
    config = AdaptShotConfig(backbone="resnet18", device="cpu", seed=123)
    dummy_img = Image.new("RGB", (224, 224), color="green")

    def extract_fn():
        return extract_embedding(dummy_img, config)

    is_deterministic = verify_determinism(extract_fn, runs=3, seed=123)
    assert is_deterministic, "Embedding extraction failed determinism check across 3 runs"


def test_backbone_registry_integrity():
    """Ensure registry contains expected backbones and factories are callable."""
    assert "resnet18" in BackboneRegistry
    assert "mobilenet_v3_small" in BackboneRegistry
    
    for name, factory in BackboneRegistry.items():
        model = factory()
        assert model is not None, f"Failed to instantiate backbone: {name}"
        # Verify it's in eval mode by default (factories should return fresh models)
        assert not model.training, f"Backbone {name} initialized in training mode"


def test_invalid_backbone_raises_value_error():
    """Ensure unknown backbone names fail fast with a clear error message."""
    config = AdaptShotConfig(backbone="vit_tiny_patch16", device="cpu", seed=42)
    dummy_img = Image.new("RGB", (224, 224), color="yellow")
    with pytest.raises(ValueError, match="Unknown backbone"):
        extract_embedding(dummy_img, config)