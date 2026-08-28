"""Tests for multi-signal uncertainty quantification (v0.2.0)."""

from __future__ import annotations

import numpy as np
import pytest

from adaptshot import UncertaintyQuantifier, UncertaintyReport


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
        return UncertaintyQuantifier(ood_percentile=95.0)

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

        # Query exactly at class "a" mean
        mean_a = quantifier._class_means["a"]
        dist = quantifier.mahalanobis_distance(mean_a, "a")
        # Distance to own class mean should be very small
        assert dist < 2.0

    def test_mahalanobis_distance_different_class_high(
        self, quantifier: UncertaintyQuantifier, synthetic_data: tuple
    ) -> None:
        """Mahalanobis distance to opposite class should be higher."""
        embeddings, labels = synthetic_data
        quantifier.fit_class_distributions(embeddings, labels)

        # Query at class "a" mean, measure distance to class "b"
        mean_a = quantifier._class_means["a"]
        dist_a = quantifier.mahalanobis_distance(mean_a, "a")
        dist_b = quantifier.mahalanobis_distance(mean_a, "b")
        # Distance to own class should be smaller
        assert dist_a < dist_b

    def test_is_ood_returns_false_for_indist(
        self, quantifier: UncertaintyQuantifier, synthetic_data: tuple
    ) -> None:
        """In-distribution queries should not be flagged as OOD."""
        embeddings, labels = synthetic_data
        quantifier.fit_class_distributions(embeddings, labels)

        # Query at class "a" mean
        mean_a = quantifier._class_means["a"]
        is_ood, score = quantifier.is_ood(mean_a)
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
        # Entropy should be non-negative (floating point may produce tiny negatives)
        assert entropy >= -1e-12
        assert norm_entropy >= -1e-12
        assert norm_entropy <= 1.0

    def test_quantify_returns_all_signals(
        self, quantifier: UncertaintyQuantifier, synthetic_data: tuple
    ) -> None:
        """Quantify should return a complete UncertaintyReport."""
        embeddings, labels = synthetic_data
        quantifier.fit_class_distributions(embeddings, labels)

        # Query at class "a" mean
        mean_a = quantifier._class_means["a"]
        report = quantifier.quantify(mean_a, embeddings, labels)

        assert isinstance(report, UncertaintyReport)
        # Allow tiny floating-point negatives
        assert report.aleatoric >= -1e-12
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


# ---------------------------------------------------------------------------
# #40: the Mahalanobis distance is computed once per query, not twice
# ---------------------------------------------------------------------------


def _fitted_quantifier(dim: int = 64, n_classes: int = 4, per_class: int = 15):
    """A quantifier with fitted class distributions, for the #40 tests."""

    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(n_classes * per_class, dim)).astype(np.float32)
    labels = np.array([f"c{i // per_class}" for i in range(n_classes * per_class)])
    quantifier = UncertaintyQuantifier()
    quantifier.fit_class_distributions(embeddings, labels)
    return quantifier, embeddings, labels


def test_mahalanobis_is_computed_once_per_query() -> None:
    """It used to run twice: once for the margin, once inside is_ood() (#40).

    Mahalanobis is O(D^2) per class and the most expensive step on this path, so
    a duplicate is not a rounding error for a library whose stated constraints
    are CPU-only and under 250MB.

    Counting calls rather than timing them: a timing assertion would be flaky on
    a shared CI runner, and the property that matters is structural.
    """

    quantifier, embeddings, labels = _fitted_quantifier()
    rng = np.random.default_rng(7)
    query = rng.normal(size=(embeddings.shape[1],)).astype(np.float32)

    calls = 0
    original = quantifier.min_mahalanobis_distance

    def counting(embedding: np.ndarray):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return original(embedding)

    quantifier.min_mahalanobis_distance = counting  # type: ignore[method-assign]
    quantifier.quantify(query, embeddings, labels, mode="mahalanobis")

    assert calls == 1, f"min_mahalanobis_distance ran {calls} times for one query"


def test_sharing_the_distance_does_not_change_any_output() -> None:
    """A performance fix must not move a single reported value.

    `is_ood(embedding, min_dist=...)` must agree exactly with `is_ood(embedding)`
    -- passing the precomputed distance is an optimisation, not a variation.
    """

    quantifier, embeddings, _labels = _fitted_quantifier()
    rng = np.random.default_rng(11)

    for _ in range(25):
        query = rng.normal(size=(embeddings.shape[1],)).astype(np.float32)
        min_dist, _, _ = quantifier.min_mahalanobis_distance(query)

        recomputed = quantifier.is_ood(query)
        shared = quantifier.is_ood(query, min_dist=min_dist)

        assert recomputed == shared


def test_is_ood_still_works_without_a_precomputed_distance() -> None:
    """The public signature must keep working for callers that pass one argument."""

    quantifier, embeddings, _labels = _fitted_quantifier()
    rng = np.random.default_rng(3)
    query = rng.normal(size=(embeddings.shape[1],)).astype(np.float32)

    flag, score = quantifier.is_ood(query)
    assert isinstance(flag, bool)
    assert 0.0 <= score <= 1.0


def test_the_mahalanobis_path_is_reproducible() -> None:
    """Distinct from #58: this path has no stochastic component and must not gain one.

    `mode="ensemble"` is *not* reproducible, because its epistemic signal seeds
    from OS entropy on every call (#58). The Mahalanobis path is, and the fix for
    #40 must leave it that way.
    """

    quantifier, embeddings, labels = _fitted_quantifier()
    rng = np.random.default_rng(5)
    query = rng.normal(size=(embeddings.shape[1],)).astype(np.float32)

    first = quantifier.quantify(query, embeddings, labels, mode="mahalanobis")
    second = quantifier.quantify(query, embeddings, labels, mode="mahalanobis")

    assert first.distributional == second.distributional
    assert first.ood_score == second.ood_score
    assert first.nearest_class_margin == second.nearest_class_margin
    assert first.is_ood == second.is_ood
