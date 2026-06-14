"""Tests for AdaptShot custom exceptions and learner input validation."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.adaptshot import (
    AdaptShotConfig,
    BufferCapacityError,
    CalibrationNotReadyError,
    ConfigValidationError,
    FewShotLearner,
    InvalidImageError,
)


@pytest.fixture
def rgb_image_path(tmp_path: Path) -> str:
    path = tmp_path / "rgb.png"
    Image.new("RGB", (32, 32), color="red").save(path)
    return str(path)


@pytest.fixture
def grayscale_image_path(tmp_path: Path) -> str:
    path = tmp_path / "gray.png"
    Image.new("L", (32, 32), color=128).save(path)
    return str(path)


def test_exception_exports() -> None:
    assert issubclass(InvalidImageError, Exception)
    assert issubclass(ConfigValidationError, Exception)
    assert issubclass(CalibrationNotReadyError, Exception)
    assert issubclass(BufferCapacityError, Exception)


def test_init_raises_config_validation_for_non_cpu() -> None:
    cfg = AdaptShotConfig(device="cuda")
    with pytest.raises(ConfigValidationError, match="CPU-first"):
        FewShotLearner(config=cfg)


def test_load_support_images_rejects_grayscale(grayscale_image_path: str) -> None:
    learner = FewShotLearner(config=AdaptShotConfig(device="cpu"))
    with pytest.raises(InvalidImageError, match="Expected 3-channel RGB image"):
        learner.load_support_images([grayscale_image_path], ["cat"])


def test_load_support_images_rejects_missing_file() -> None:
    learner = FewShotLearner(config=AdaptShotConfig(device="cpu"))
    with pytest.raises(InvalidImageError, match="Image file not found"):
        learner.load_support_images(["/tmp/does-not-exist.png"], ["cat"])


def test_load_support_images_rejects_mismatched_lengths(rgb_image_path: str) -> None:
    learner = FewShotLearner(config=AdaptShotConfig(device="cpu"))
    with pytest.raises(ConfigValidationError, match="same length"):
        learner.load_support_images([rgb_image_path], [])


def test_predict_rejects_grayscale_array(rgb_image_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    learner = FewShotLearner(config=AdaptShotConfig(device="cpu"))

    monkeypatch.setattr(
        "src.adaptshot.core.learner.extract_embedding",
        lambda image, cfg, **kwargs: np.ones(8, dtype=np.float32),
    )

    learner.load_support_images([rgb_image_path], ["cat"])

    gray = np.zeros((32, 32), dtype=np.uint8)
    with pytest.raises(InvalidImageError, match="1-channel grayscale"):
        learner.predict(gray)


def test_predict_falls_back_when_calibration_not_ready(
    rgb_image_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = FewShotLearner(config=AdaptShotConfig(device="cpu"))

    monkeypatch.setattr(
        "src.adaptshot.core.learner.extract_embedding",
        lambda image, cfg, **kwargs: np.ones(8, dtype=np.float32),
    )
    monkeypatch.setattr(
        "src.adaptshot.core.learner.find_nearest_prototype",
        lambda query, prototypes, prototype_labels, metric="euclidean": (
            prototype_labels[0],
            0.0,
            0,
            0.0,
            float("inf"),
        ),
    )

    learner.load_support_images([rgb_image_path], ["cat"])
    result = learner.predict(rgb_image_path)

    assert result.prediction == "cat"
    assert 0.0 <= result.calibrated_confidence <= 1.0


def test_calibration_graceful_fallback_when_cold(rgb_image_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """v0.2.0: _calibrate_or_raise no longer raises on cold start.

    Instead, it gracefully falls back by seeding the calibration window
    with an optimistic prior and returning the raw confidence clamped
    to the unit interval.
    """
    learner = FewShotLearner(config=AdaptShotConfig(device="cpu"))

    monkeypatch.setattr(
        "src.adaptshot.core.learner.extract_embedding",
        lambda image, cfg, **kwargs: np.ones(8, dtype=np.float32),
    )

    learner.load_support_images([rgb_image_path], ["cat"])

    # Should NOT raise — calibrate_or_raise falls back gracefully
    result = learner._calibrate_or_raise(0.5)
    assert 0.0 <= result <= 1.0, f"Expected calibrated confidence in [0,1], got {result}"


def test_correct_raises_on_bad_confidence_weight(rgb_image_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    learner = FewShotLearner(config=AdaptShotConfig(device="cpu"))

    monkeypatch.setattr(
        "src.adaptshot.core.learner.extract_embedding",
        lambda image, cfg, **kwargs: np.ones(8, dtype=np.float32),
    )
    monkeypatch.setattr(
        "src.adaptshot.core.learner.find_nearest_neighbor",
        lambda query, support_embeddings, support_labels, use_faiss=False, metric="cosine", **kwargs: (
            support_labels[0],
            0.8,
            0,
        ),
    )

    learner.load_support_images([rgb_image_path], ["cat"])

    with pytest.raises(ConfigValidationError, match="confidence_weight"):
        learner.correct(rgb_image_path, "cat", confidence_weight=1.5)


def test_buffer_capacity_error_is_caught_in_correct(
    rgb_image_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = FewShotLearner(config=AdaptShotConfig(device="cpu", max_buffer_size=10))

    monkeypatch.setattr(
        "src.adaptshot.core.learner.extract_embedding",
        lambda image, cfg, **kwargs: np.ones(8, dtype=np.float32),
    )
    monkeypatch.setattr(
        "src.adaptshot.core.learner.find_nearest_neighbor",
        lambda query, support_embeddings, support_labels, use_faiss=False, metric="cosine", **kwargs: (
            support_labels[0],
            0.8,
            0,
        ),
    )

    learner.load_support_images([rgb_image_path], ["cat"])

    def explode() -> None:
        raise BufferCapacityError("synthetic prune failure")

    monkeypatch.setattr(learner, "_apply_buffer_management", explode)

    result = learner.correct(rgb_image_path, "cat", confidence_weight=1.0)
    assert "buffer_management_warning" in result
    assert "synthetic prune failure" in result["buffer_management_warning"]
