"""Integration tests for FewShotLearner full pipeline round-trip.

Validates the complete workflow: load support -> predict -> correct ->
save -> load -> predict again, ensuring state consistency across
persistence boundaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from adaptshot import AdaptShotConfig, AdaptShotError, FewShotLearner


@pytest.fixture
def synthetic_images(tmp_path: Path) -> list[str]:
    """Create two distinct RGB images for support set testing."""
    paths: list[str] = []
    for index, color in enumerate(["red", "green"]):
        path = tmp_path / f"support_{index}.png"
        Image.new("RGB", (32, 32), color=color).save(path)
        paths.append(str(path))
    return paths


@pytest.fixture
def patched_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace extract_embedding with a fast, deterministic stub.

    Produces different embeddings for different input colors so that
    similarity search returns meaningful results without loading a real
    backbone model.
    """

    def fake_extract(
        image: Any,
        config: AdaptShotConfig,
        return_numpy: bool = True,
        cache: Any = None,
    ) -> np.ndarray:
        color = image.getpixel((0, 0)) if isinstance(image, Image.Image) else (0, 0, 0)
        base = float(sum(color)) / 255.0
        embedding = np.full(8, base, dtype=np.float32)
        if return_numpy:
            return embedding
        return embedding

    monkeypatch.setattr(
        "adaptshot.core.learner.extract_embedding", fake_extract
    )


class TestFewShotLearnerIntegration:
    """End-to-end tests for the full FewShotLearner pipeline."""

    def test_predict_returns_structured_result(
        self,
        synthetic_images: list[str],
        patched_embeddings: None,
    ) -> None:
        """Verify predict() returns a PredictionResult with expected fields."""
        learner = FewShotLearner(config=AdaptShotConfig(device="cpu"))
        learner.load_support_images(synthetic_images, [0, 1])

        result = learner.predict(synthetic_images[0])

        assert result.prediction in {0, 1}
        assert 0.0 <= result.raw_confidence <= 1.0
        assert 0.0 <= result.calibrated_confidence <= 1.0
        assert isinstance(result.neighbor_idx, int)
        assert isinstance(result.uncertainty_flag, bool)
        assert isinstance(result.act_action, str)
        assert isinstance(result.ood_flag, bool)

    def test_predict_after_init_only(
        self,
        synthetic_images: list[str],
        patched_embeddings: None,
    ) -> None:
        """Predictions after support load should remain consistent."""
        learner = FewShotLearner(config=AdaptShotConfig(device="cpu"))
        learner.load_support_images(synthetic_images, [0, 1])

        result1 = learner.predict(synthetic_images[0])
        result2 = learner.predict(synthetic_images[0])

        # Repeated predictions on the same image should be stable
        assert result1.prediction == result2.prediction
        assert np.isclose(result1.raw_confidence, result2.raw_confidence, atol=1e-5)

    def test_correct_updates_state(
        self,
        synthetic_images: list[str],
        patched_embeddings: None,
    ) -> None:
        """correct() should route feedback and return meaningful summary."""
        learner = FewShotLearner(config=AdaptShotConfig(device="cpu"))
        learner.load_support_images(synthetic_images, [0, 1])

        # Submit a correction to a different class
        summary = learner.correct(
            image_path=synthetic_images[0],
            true_label=1,
            confidence_weight=0.8,
        )

        assert "buffer_size" in summary
        assert "calibration_updated" in summary
        assert "fine_tuned" in summary
        assert summary["total_corrections"] >= 1

    def test_save_load_roundtrip_preserves_predictions(
        self,
        tmp_path: Path,
        synthetic_images: list[str],
        patched_embeddings: None,
    ) -> None:
        """After save() and load(), predictions should be identical."""
        learner = FewShotLearner(config=AdaptShotConfig(device="cpu"))
        learner.load_support_images(synthetic_images, [0, 1])

        # Record pre-save prediction
        result_before = learner.predict(synthetic_images[0])

        # Persist and restore
        checkpoint = tmp_path / "roundtrip.json"
        learner.save(str(checkpoint))
        restored = FewShotLearner.load(str(checkpoint))

        result_after = restored.predict(synthetic_images[0])

        assert restored._is_initialized is True
        assert restored._sim_labels == learner._sim_labels
        assert result_after.prediction == result_before.prediction
        assert np.isclose(
            result_after.raw_confidence, result_before.raw_confidence, atol=1e-5
        )

    def test_save_load_after_correction(
        self,
        tmp_path: Path,
        synthetic_images: list[str],
        patched_embeddings: None,
    ) -> None:
        """Corrections made before save should persist through a load cycle."""
        learner = FewShotLearner(config=AdaptShotConfig(device="cpu"))
        learner.load_support_images(synthetic_images, [0, 1])

        learner.correct(
            image_path=synthetic_images[0],
            true_label=1,
            confidence_weight=1.0,
        )

        checkpoint = tmp_path / "corrected.json"
        learner.save(str(checkpoint))
        restored = FewShotLearner.load(str(checkpoint))

        # Calibration should have been updated by the correction
        assert restored._sim_uncertainties == learner._sim_uncertainties
        assert restored._is_initialized is True

    def test_predict_fails_before_init(self) -> None:
        """Calling predict() without loading support images must raise."""
        learner = FewShotLearner(config=AdaptShotConfig(device="cpu"))

        with pytest.raises(AdaptShotError, match="not initialized"):
            learner.predict("nonexistent.jpg")

    def test_correct_fails_before_init(self) -> None:
        """Calling correct() without loading support images must raise."""
        learner = FewShotLearner(config=AdaptShotConfig(device="cpu"))

        with pytest.raises(AdaptShotError, match="not initialized"):
            learner.correct(image_path="nonexistent.jpg", true_label=0)
