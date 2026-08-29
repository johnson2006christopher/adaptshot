"""Unit tests for core/extractor.py and utils/determinism.py."""

from typing import Any, cast

import pytest
from PIL import Image

from adaptshot.config.settings import AdaptShotConfig
from adaptshot.core.extractor import BackboneRegistry, extract_embedding
from adaptshot.utils.determinism import verify_determinism

# `config/settings.py` and `core/extractor.py` resolve torch lazily (see `_get_torch`),
# so they import fine on a core-only install and sit above this guard. These tests do
# not: they exercise the torch-backed paths directly and patch `torchvision.models`, so
# on a core install the whole module skips rather than failing collection and taking the
# other 93 tests down with it.
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


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


def test_backbone_factories_request_imagenet_pretrained_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure backbone factories request ImageNet pretrained weights.

    ImageNet statistics are baked into the preprocessing pipeline, so
    random weights would yield meaningless embeddings. The factory
    lambdas must pass weights="IMAGENET1K_V1".
    """

    captured: dict[str, object] = {}

    def fake_resnet18(*, weights: object | None = None) -> torch.nn.Module:
        captured["resnet18_weights"] = weights
        return torch.nn.Identity()

    def fake_mobilenet_v3_small(*, weights: object | None = None) -> torch.nn.Module:
        captured["mobilenet_weights"] = weights
        return torch.nn.Identity()

    # Patch torchvision.models directly – extractor.py now uses lazy imports via
    # _get_tv_models() rather than a module-level 'models' attribute.
    monkeypatch.setattr("torchvision.models.resnet18", fake_resnet18)
    monkeypatch.setattr("torchvision.models.mobilenet_v3_small", fake_mobilenet_v3_small)

    resnet_factory = cast(Any, BackboneRegistry["resnet18"])
    mobilenet_factory = cast(Any, BackboneRegistry["mobilenet_v3_small"])

    resnet_factory()
    mobilenet_factory()

    assert captured["resnet18_weights"] == "IMAGENET1K_V1"
    assert captured["mobilenet_weights"] == "IMAGENET1K_V1"