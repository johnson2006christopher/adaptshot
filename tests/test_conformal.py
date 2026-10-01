"""Tests for conformal prediction engine (v0.2.0)."""

from __future__ import annotations

import numpy as np
import pytest

from adaptshot import ConformalEngine, ConformalPredictionSet


class TestConformalPredictionSet:
    """Tests for the ConformalPredictionSet dataclass."""

    def test_empty_set_has_singleton_top(self) -> None:
        cps = ConformalPredictionSet(prediction="cat", alpha=0.05)
        assert cps.prediction_set == set()
        assert cps.set_size == 0
        assert cps.prediction == "cat"
        assert cps.alpha == 0.05

    def test_contains_checks_membership(self) -> None:
        cps = ConformalPredictionSet(
            prediction_set={"cat", "dog"}, prediction="cat", alpha=0.05
        )
        assert cps.contains("cat") is True
        assert cps.contains("dog") is True
        assert cps.contains("bird") is False

    def test_repr_includes_key_metrics(self) -> None:
        cps = ConformalPredictionSet(
            prediction_set={"cat", "dog"},
            set_size=2,
            alpha=0.05,
            q_hat=0.8,
            coverage_estimate=0.94,
            prediction="cat",
        )
        rep = repr(cps)
        assert "size=2" in rep
        assert "alpha=0.050" in rep
        assert "coverage=0.940" in rep


class TestConformalEngine:
    """Tests for the ConformalEngine class."""

    @pytest.fixture
    def engine(self) -> ConformalEngine:
        return ConformalEngine(alpha=0.1)

    def test_softmax_nonconformity_perfect(self) -> None:
        """Perfect match (0 distance) should produce low nonconformity."""
        distances = np.array([0.0, 5.0, 5.0], dtype=np.float32)
        labels = np.array(["cat", "dog", "bird"], dtype=object)
        score = ConformalEngine.softmax_nonconformity(distances, labels, "cat")
        # True class is closest, so score should be close to 0
        assert 0.0 <= score < 0.5

    def test_softmax_nonconformity_poor(self) -> None:
        """Poor match (high distance) should produce high nonconformity."""
        distances = np.array([5.0, 0.0, 5.0], dtype=np.float32)
        labels = np.array(["cat", "dog", "bird"], dtype=object)
        score = ConformalEngine.softmax_nonconformity(distances, labels, "cat")
        assert 0.5 <= score <= 1.0

    def test_softmax_nonconformity_missing_class(self) -> None:
        """Missing class returns full nonconformity (1.0)."""
        distances = np.array([1.0, 2.0], dtype=np.float32)
        labels = np.array(["dog", "bird"], dtype=object)
        score = ConformalEngine.softmax_nonconformity(distances, labels, "cat")
        assert score == 1.0

    def test_softmax_nonconformity_empty(self) -> None:
        """Empty arrays return 1.0."""
        score = ConformalEngine.softmax_nonconformity(
            np.array([]), np.array([]), "cat"
        )
        assert score == 1.0

    def test_distance_nonconformity_normal(self) -> None:
        """Distance below threshold yields proportional score."""
        score = ConformalEngine.distance_nonconformity(0.5, 1.0)
        assert score == 0.5

    def test_distance_nonconformity_clamped(self) -> None:
        """Distance above threshold is clamped to 1.0."""
        score = ConformalEngine.distance_nonconformity(2.0, 1.0)
        assert score == 1.0

    def test_distance_nonconformity_zero_threshold(self) -> None:
        """Zero threshold returns 1.0 for positive distances."""
        score = ConformalEngine.distance_nonconformity(0.5, 0.0)
        assert score == 1.0

    def test_predict_set_singleton_when_no_calibration(self, engine: ConformalEngine) -> None:
        """Without calibration data, return singleton set."""
        distances = np.array([0.1, 1.0, 2.0], dtype=np.float32)
        labels = np.array(["cat", "dog", "bird"], dtype=object)
        result = engine.predict_set(distances, labels, "cat", 0.9)
        assert result.prediction_set == {"cat"}
        assert result.set_size == 1
        # And it says so: no guarantee, no number pretending to be one (#80).
        assert result.calibrated is False
        assert np.isnan(result.coverage_estimate) and np.isnan(result.q_hat)

    def test_predict_set_includes_valid_classes_after_fitting(
        self, engine: ConformalEngine
    ) -> None:
        """After calibration, prediction set may include multiple classes.

        Calibration scores of 1.2 — a value the ratio score can actually
        produce (it is d_true/min(d), never below 1.0). This test used to seed
        impossible 0.5s, and only the force-added top-1 kept it passing (#115).
        """
        for _ in range(15):
            engine.update_calibration(1.2, "cat")

        distances = np.array([0.1, 1.0, 2.0], dtype=np.float32)
        labels = np.array(["cat", "dog", "bird"], dtype=object)
        result = engine.predict_set(distances, labels, "cat", 0.9)
        # The top-1's own score is exactly 1.0, the minimum, so any finite
        # quantile admits it — by arithmetic, not by a force-add.
        assert "cat" in result.prediction_set
        assert result.set_size >= 1
        assert result.calibrated is True

    def test_an_empty_set_is_an_abstention_not_a_forced_singleton(
        self, engine: ConformalEngine
    ) -> None:
        """When no class clears the quantile, the set is honestly empty.

        The engine used to force-add the top prediction "for safety", which
        under non-default score methods turned an abstention the theory issued
        into a singleton it never did (#115). Scores below 1.0 cannot arise
        under the default ratio score, which is exactly what makes them the
        right probe here: a quantile of 0.5 admits nothing.
        """
        for _ in range(15):
            engine.update_calibration(0.5, "cat")

        distances = np.array([0.1, 1.0, 2.0], dtype=np.float32)
        labels = np.array(["cat", "dog", "bird"], dtype=object)
        result = engine.predict_set(distances, labels, "cat", 0.9)
        assert result.prediction_set == set(), (
            "no class cleared the quantile; the set must say so, not smuggle "
            "the top-1 back in"
        )
        assert result.set_size == 0

    def test_calibration_summary_tracks_state(self, engine: ConformalEngine) -> None:
        """Calibration summary reflects buffer state."""
        for i in range(20):
            engine.update_calibration(0.1 * i, f"class_{i % 3}")
        summary = engine.get_calibration_summary()
        assert summary["calibration_size"] == 20.0
        assert "target_coverage" in summary
        assert "q_hat" in summary

    def test_reset_clears_all_state(self, engine: ConformalEngine) -> None:
        """Reset clears calibration data and counters."""
        engine.update_calibration(0.5, "cat")
        engine.reset()
        assert engine.calibration_size == 0
        assert engine.empirical_coverage == 0.9  # target (prior)
