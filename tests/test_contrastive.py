"""Tests for contrastive prototype learning (v0.2.0)."""

from __future__ import annotations

import numpy as np
import pytest

from adaptshot import ContrastiveConfig, ContrastivePrototypeLearner


class TestContrastiveConfig:
    """Tests for the ContrastiveConfig dataclass."""

    def test_default_values_are_reasonable(self) -> None:
        cfg = ContrastiveConfig()
        assert cfg.projection_dim == 128
        assert 0.0 < cfg.temperature < 1.0
        assert 0.0 < cfg.learning_rate < 1.0
        assert cfg.n_epochs > 0

    def test_custom_values_accepted(self) -> None:
        cfg = ContrastiveConfig(
            projection_dim=64, temperature=0.1, momentum=0.8, n_epochs=30
        )
        assert cfg.projection_dim == 64
        assert cfg.temperature == 0.1
        assert cfg.momentum == 0.8
        assert cfg.n_epochs == 30


class TestContrastivePrototypeLearner:
    """Tests for the ContrastivePrototypeLearner class."""

    @pytest.fixture
    def learner(self) -> ContrastivePrototypeLearner:
        return ContrastivePrototypeLearner()

    @pytest.fixture
    def synthetic_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Create synthetic 2-class data with clear separation."""
        rng = np.random.RandomState(42)
        n_per_class = 20
        dim = 64

        # Class 0: centered at (-2, 0, ..., 0)
        class0 = rng.randn(n_per_class, dim) * 0.5
        class0[:, 0] -= 2.0

        # Class 1: centered at (+2, 0, ..., 0)
        class1 = rng.randn(n_per_class, dim) * 0.5
        class1[:, 0] += 2.0

        embeddings = np.vstack([class0, class1]).astype(np.float32)
        labels = np.array(["class_0"] * n_per_class + ["class_1"] * n_per_class, dtype=object)
        return embeddings, labels

    def test_is_fitted_initially_false(self, learner: ContrastivePrototypeLearner) -> None:
        assert learner.is_fitted is False

    def test_projection_head_after_refine(
        self, learner: ContrastivePrototypeLearner, synthetic_data: tuple
    ) -> None:
        """Projection head projects to projection_dim after refine_prototypes."""
        embeddings, labels = synthetic_data
        learner.refine_prototypes(embeddings, labels, seed=42)
        x = np.random.randn(5, 64).astype(np.float32)
        projected = learner._project(x)
        assert projected.shape == (5, learner.config.projection_dim)

    def test_refine_prototypes_returns_correct_shape(
        self, learner: ContrastivePrototypeLearner, synthetic_data: tuple
    ) -> None:
        """Refined prototypes should have correct shape and labels."""
        embeddings, labels = synthetic_data
        prototypes, proto_labels = learner.refine_prototypes(embeddings, labels, seed=42)
        assert prototypes.shape[0] == 2  # 2 classes
        assert prototypes.shape[1] == learner.config.projection_dim
        assert len(proto_labels) == 2
        assert learner.is_fitted is True

    def test_nearest_prototype_identifies_class(
        self, learner: ContrastivePrototypeLearner, synthetic_data: tuple
    ) -> None:
        """Query near class_0 prototype should be classified as class_0."""
        embeddings, labels = synthetic_data
        prototypes, proto_labels = learner.refine_prototypes(embeddings, labels, seed=42)

        # Query near class_0 center
        query = np.array([-2.0] + [0.0] * 63, dtype=np.float32).reshape(-1)
        pred, conf, _idx = learner.nearest_prototype(query, prototypes, proto_labels)
        assert pred == "class_0"
        assert 0.0 <= conf <= 1.0

    def test_class_separation_score_positive(
        self, learner: ContrastivePrototypeLearner, synthetic_data: tuple
    ) -> None:
        """Well-separated data should have positive separation score."""
        embeddings, labels = synthetic_data
        score = learner.class_separation_score(embeddings, labels)
        assert score > 0.0
