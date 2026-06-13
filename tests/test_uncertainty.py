"""Tests for multi-signal uncertainty quantification (v0.2.0)."""

from __future__ import annotations

import numpy as np
import pytest

from src.adaptshot import UncertaintyQuantifier, UncertaintyReport


class TestUncertaintyReport:
    """Tests for the UncertaintyReport dataclass."""

    def test_defaults_are_zero_false(self) -> None:
        report = UncertaintyReport()
        assert report.epistemic == 0.0
        assert report.aleatoric == 0.0
        assert report.distributional == 0.0
        assert report.composite == 0.0
        assert report.is_ood is False

    def test_to_dict_returns_all_keys(self) -> None:
        report = UncertaintyReport(epistemic=0.1, aleatoric=0.2, is_ood=True)
        d = report.to_dict()
        assert d["epistemic"] == 0.1
        assert d["aleatoric"] == 0.2
        assert d["is_ood"] == 1.0
        assert "composite" in d
        assert "entropy" in d


class TestUncertaintyQuantifier:
    """Tests for the UncertaintyQuantifier class."""

    @pytest.fixture
    def quantifier(self) -> UncertaintyQuantifier:
        return UncertaintyQuantifier(ood_percentile=90.0)

    @pytest.fixture
    def synthetic_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Create well-separated 2-class data."""
        rng = np.random.RandomState(42)
        n_per_class = 20
        dim = 32
        class0 = rng.randn(n_per_class, dim) * 0.3
        class0[:, 0] -= 3.0
        class1 = rng.randn(n_per_class, dim) * 0.3
        class1[:, 0] += 3.0
        embeddings = np.vstack([class0, class1]).astype(np.float32)
        labels = np.array(["a"] * n_per_class + ["b"] * n_per_class, dtype=object)
        return embeddings, labels

    def test_fit_class_distributions_populates_means(
        self, quantifier: UncertaintyQuantifier, synthetic_data: tuple
    ) -> None:
        """After fitting, class means should exist."""
        embeddings, labels = synthetic_data
        quantifier.fit_class_distributions(embeddings, labels)
        assert len(quantifier._class_means) == 2
        assert "a" in quantifier._class_means
        assert "b" in quantifier._class_means

    def test_mahalanobis_distance_same_class_low(
        self, quantifier: UncertaintyQuantifier, synthetic_data: tuple
    ) -> None:
        """Mahalanobis distance to own class should be low."""
        embeddings, labels = synthetic_data
        quantifier.fit_class_distributions(embeddings, labels)

        # Query near class "a" center
        query = np.array([-3.0] + [0.0] * 31, dtype=np.float32)
        dist = quantifier.mahalanobis_distance(query, "a")
        assert dist < 10.0  # should be small for in-distribution

    def test_mahalanobis_distance_different_class_high(
        self, quantifier: UncertaintyQuantifier, synthetic_data: tuple
    ) -> None:
        """Mahalanobis distance to opposite class should be higher."""
        embeddings, labels = synthetic_data
        quantifier.fit_class_distributions(embeddings, labels)

        # Query near class "a" center but measure distance to class "b"
        query = np.array([-3.0] + [0.0] * 31, dtype=np.float32)
        dist_a = quantifier.mahalanobis_distance(query, "a")
        dist_b = quantifier.mahalanobis_distance(query, "b")
        # Distance to own class should be much smaller
        assert dist_a < dist_b

    def test_is_ood_returns_false_for_indist(
        self, quantifier: UncertaintyQuantifier, synthetic_data: tuple
    ) -> None:
        """In-distribution queries should not be flagged as OOD."""
        embeddings, labels = synthetic_data
        quantifier.fit_class_distributions(embeddings, labels)

        query = np.array([-3.0] + [0.0] * 31, dtype=np.float32)
        is_ood, score = quantifier.is_ood(query)
        assert is_ood is False
        assert 0.0 <= score <= 1.0

    def test_is_ood_returns_true_for_far_outlier(
        self, quantifier: UncertaintyQuantifier, synthetic_data: tuple
    ) -> None:
        """Far outlier should be flagged as OOD."""
        embeddings, labels = synthetic_data
        quantifier.fit_class_distributions(embeddings, labels)

        # Very far from both classes
        query = np.array([50.0] + [0.0] * 31, dtype=np.float32)
        is_ood, score = quantifier.is_ood(query)
        assert is_ood is True
        assert score > 0.5

    def test_knn_entropy_low_for_clear_class(
        self, quantifier: UncertaintyQuantifier, synthetic_data: tuple
    ) -> None:
        """Well-separated data should yield low entropy."""
        embeddings, labels = synthetic_data
        # Query clearly in class "a" region
        query = np.array([-3.0] + [0.0] * 31, dtype=np.float32)
        entropy, norm_entropy = quantifier.compute_knn_entropy(query, embeddings, labels)
        assert 0.0 <= entropy
        assert 0.0 <= norm_entropy <= 1.0

    def test_quantify_returns_all_signals(
        self, quantifier: UncertaintyQuantifier, synthetic_data: tuple
    ) -> None:
        """Quantify should return a complete UncertaintyReport."""
        embeddings, labels = synthetic_data
        quantifier.fit_class_distributions(embeddings, labels)

        query = np.array([-3.0] + [0.0] * 31, dtype=np.float32)
        report = quantifier.quantify(query, embeddings, labels)

        assert isinstance(report, UncertaintyReport)
        assert 0.0 <= report.aleatoric <= 1.0
        assert 0.0 <= report.distributional <= 1.0
        assert 0.0 <= report.composite <= 1.0
        assert isinstance(report.is_ood, bool)

    def test_reset_clears_distributions(
        self, quantifier: UncertaintyQuantifier, synthetic_data: tuple
    ) -> None:
        """Reset clears all fitted state."""
        embeddings, labels = synthetic_data
        quantifier.fit_class_distributions(embeddings, labels)
        assert len(quantifier._class_means) > 0
        quantifier.reset()
        assert len(quantifier._class_means) == 0
        assert quantifier._ood_threshold == float("inf")
