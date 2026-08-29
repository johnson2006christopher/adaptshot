"""The claim #36 rests on: the ONNX path returns the same embeddings as torch.

Making inference torch-free is only a good trade if it is not also a change in
behaviour. Every accuracy figure the project has published was measured through
the torch path, and they stay comparable only while the two agree. A backbone
re-exported at a different opset, or exported from different pretrained weights,
would shift every downstream number silently -- no test would fail, the
benchmark would simply start reporting something else.

`benchmarks/onnx_parity.py` reports the same agreement alongside latency, but a
benchmark is something a person chooses to run. This is the part that has to
fail in CI.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from adaptshot.config.settings import AdaptShotConfig
from adaptshot.core.extractor import bundled_onnx_backbones, extract_embedding

#: Measured at 1.8e-06 across both backbones. The bound is two orders of
#: magnitude looser, because it has to hold on CPUs and onnxruntime builds we
#: have not measured on -- while still being far tighter than any difference
#: that could move an accuracy number.
MAX_ABSOLUTE_DIFFERENCE = 1e-4

#: Direction matters more than magnitude for every metric downstream: the
#: similarity search, the conformal sets and the uncertainty estimates all read
#: angles, not lengths.
MIN_COSINE = 0.9999


@pytest.fixture(scope="module")
def image() -> Image.Image:
    """A fixed noise image. Real photographs are not needed to compare two
    implementations of the same function, and generating one keeps the test
    offline and deterministic."""

    rng = np.random.default_rng(42)
    return Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))


@pytest.mark.parametrize("backbone", bundled_onnx_backbones())
def test_onnx_embeddings_agree_with_torch(backbone: str, image: Image.Image) -> None:
    """Both paths must describe the same point in embedding space."""

    pytest.importorskip("torch", reason="parity needs both paths to compare")
    pytest.importorskip("torchvision", reason="the torch path builds from torchvision")

    config = AdaptShotConfig(backbone=backbone, device="cpu", seed=42)

    from_onnx = np.asarray(extract_embedding(image, config, return_numpy=True))
    # `return_numpy=False` is the documented way to ask for the torch path by
    # name -- it returns a tensor, which an ONNX session cannot produce.
    from_torch = extract_embedding(image, config, return_numpy=False).detach().cpu().numpy()

    assert from_onnx.shape == from_torch.shape, (
        f"{backbone}: ONNX returns {from_onnx.shape}, torch returns "
        f"{from_torch.shape}. The bundled export is of a different graph."
    )

    largest = float(np.abs(from_onnx - from_torch).max())
    assert largest < MAX_ABSOLUTE_DIFFERENCE, (
        f"{backbone}: ONNX and torch embeddings differ by up to {largest:.2e}. "
        "Re-exporting the backbone changed what it computes, so every published "
        "accuracy number was measured on a different function than the one that "
        "now ships."
    )

    norms = float(np.linalg.norm(from_onnx) * np.linalg.norm(from_torch))
    cosine = float(np.dot(from_onnx, from_torch) / norms)
    assert cosine > MIN_COSINE, f"{backbone}: cosine between the two paths is {cosine}"


def test_at_least_one_backbone_is_bundled() -> None:
    """Without this, the parametrised test above would vacuously pass on zero cases.

    That is the failure mode where the wheel ships no ONNX weights at all: the
    core install silently loses inference and the parity suite reports success
    because it ran nothing.
    """

    assert bundled_onnx_backbones(), (
        "no ONNX weights are bundled, so a core install cannot run inference "
        "and the parity tests above ran against nothing"
    )
