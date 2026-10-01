"""The conformal engine's bookkeeping must measure, not assume (#115).

Four defects pinned here: `empirical_coverage` was 0.0 forever because every
call counted a miss; `mode="cross"` returned +inf for every buffer in the
practical regime (40 <= n < 380 at the defaults); one single-photo class
silently set q_hat to +inf for everyone while still reporting
`calibrated=True`; and the class-conditional path restated the coverage
target on cold start — the mistake #80 already fixed in `predict_set`.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from adaptshot import AdaptShotConfig, ConformalEngine, FewShotLearner
from adaptshot.data import sample_images

# ---------------------------------------------------------------------------
# empirical_coverage
# ---------------------------------------------------------------------------


def test_seeded_scores_do_not_enter_the_coverage_denominator() -> None:
    engine = ConformalEngine(alpha=0.1)
    for _ in range(30):
        engine.seed_calibration(1.2, "cat")
    assert engine._total_predictions == 0, (
        "bootstrap scores are calibration data, not observations of deployed sets"
    )
    assert engine.calibration_size == 30


def test_coverage_counts_only_stated_observations() -> None:
    engine = ConformalEngine(alpha=0.1)
    engine.update_calibration(1.2, "cat")  # outcome unknown: not counted
    assert engine._total_predictions == 0

    engine.update_calibration(1.2, "cat", predicted_in_set=True)
    engine.update_calibration(1.9, "cat", predicted_in_set=False)
    assert engine._total_predictions == 2
    assert engine.empirical_coverage == pytest.approx(0.5)


def test_corrections_on_a_calibrated_learner_move_empirical_coverage() -> None:
    """End to end: after the fix, coverage is a measurement, not 0.000.

    24 bootstrap scores calibrate the engine; a correction then observes
    whether the truth fell inside the issued set, and the counter moves off
    its prior for the first time.
    """

    paths, labels = sample_images()
    # alpha=0.10: eleven LOO seeds clear the calibration floor (9 scores at
    # this level), so a calibrated set exists for the correction to observe.
    # At the default 0.05 the floor is 19 and nothing would be calibrated.
    learner = FewShotLearner(config=AdaptShotConfig(conformal_alpha=0.10))
    learner.load_support_images(paths[:-1], labels[:-1])

    assert learner.conformal._total_predictions == 0, (
        "loading a support set must not fabricate coverage observations"
    )

    learner.correct(image_path=paths[-1], true_label=str(labels[-1]))
    assert learner.conformal._total_predictions == 1, (
        "a correction against a calibrated engine is one coverage observation"
    )
    assert learner.conformal.empirical_coverage in (0.0, 1.0)


# ---------------------------------------------------------------------------
# mode="cross"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [40, 100, 300])
def test_cross_mode_is_finite_in_the_practical_regime(n: int) -> None:
    """n=100 and n=300 used to return q_hat = inf: 20 folds of < 19 scores
    each produced +inf per fold, and the mean of anything with +inf is +inf."""

    engine = ConformalEngine(alpha=0.05, mode="cross")
    rng = np.random.default_rng(42)
    for value in 1.0 + rng.random(n):
        engine.seed_calibration(float(value), "cat")

    q_hat = engine.current_q_hat()
    assert np.isfinite(q_hat), f"cross-mode q_hat is not finite at n={n}"


def test_cross_mode_averages_folds_only_when_each_fold_can_certify() -> None:
    engine = ConformalEngine(alpha=0.05, mode="cross")
    rng = np.random.default_rng(42)
    for value in 1.0 + rng.random(400):
        engine.seed_calibration(float(value), "cat")
    # 400 // 20 = 20 >= ceil(0.95/0.05) = 19: real per-fold quantiles.
    assert np.isfinite(engine.current_q_hat())


# ---------------------------------------------------------------------------
# singleton classes
# ---------------------------------------------------------------------------


def test_a_single_photo_class_is_skipped_and_named_not_silently_poisonous(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One photograph of one extra class used to set q_hat = inf for every
    class while still reporting calibrated=True."""

    paths, labels = sample_images()
    # alpha=0.10 so the bootstrap actually runs on twelve photographs; at the
    # default 0.05 it returns before reaching the singleton check.
    learner = FewShotLearner(config=AdaptShotConfig(conformal_alpha=0.10))
    with caplog.at_level(logging.WARNING, logger="adaptshot.core.learner"):
        learner.load_support_images(
            list(paths), [*map(str, labels[:-1]), "lonely_class"]
        )

    assert any("lonely_class" in message for message in caplog.messages), (
        "the skipped single-photo class must be named in a warning"
    )
    assert not any(np.isinf(s) for s in learner.conformal._calibration_scores), (
        "an infinite score reached the calibration buffer"
    )
    result = learner.predict(paths[0])
    assert result.conformal_calibrated is (
        bool(np.isfinite(learner.conformal.current_q_hat()))
    )


def test_a_non_finite_quantile_reports_calibrated_false() -> None:
    engine = ConformalEngine(alpha=0.05)
    # Exactly min_calibration_size scores, but alpha=0.05 needs 19 to certify:
    # the quantile is honestly +inf and the set is everything — which must not
    # be presented as a calibrated set.
    for _ in range(engine.min_calibration_size):
        engine.seed_calibration(1.2, "cat")
    distances = np.array([0.1, 1.0], dtype=np.float32)
    class_labels = np.array(["cat", "dog"], dtype=object)
    result = engine.predict_set(distances, class_labels, "cat", 0.9)
    if np.isfinite(result.q_hat):
        pytest.skip("this engine configuration certifies at min_calibration_size")
    assert result.calibrated is False
    assert result.prediction_set == {"cat", "dog"}


# ---------------------------------------------------------------------------
# class-conditional cold start
# ---------------------------------------------------------------------------


def test_class_conditional_cold_start_does_not_restate_the_target() -> None:
    engine = ConformalEngine(alpha=0.1)
    distances = np.array([0.1, 1.0], dtype=np.float32)
    class_labels = np.array(["cat", "dog"], dtype=object)
    result = engine.predict_set_class_conditional(distances, class_labels, "cat", 0.9)
    assert result.calibrated is False
    assert np.isnan(result.q_hat)
    assert np.isnan(result.coverage_estimate), (
        "1 - alpha restated as coverage_estimate is the #80 mistake again"
    )
    assert result.prediction_set == {"cat"}


# ---------------------------------------------------------------------------
# score geometry
# ---------------------------------------------------------------------------


def test_conformal_distances_use_the_classifier_geometry() -> None:
    """Calibration, correction and prediction distances must be the distances
    the classifier ranks with — L2-normalised — not raw Euclidean (#115).

    Probe: a query that is an exact scalar multiple of a prototype is at raw
    Euclidean distance > 0 from it, but at normalised distance exactly 0.
    """

    paths, labels = sample_images()
    learner = FewShotLearner()
    learner.load_support_images(paths[:-1], labels[:-1])

    prototype = np.asarray(learner._prototype_embeddings[0], dtype=np.float32)
    scaled = prototype * 7.5
    distances = learner._compute_all_prototype_distances(scaled)
    assert distances[0] == pytest.approx(0.0, abs=1e-5), (
        "a scalar multiple of the prototype is not at distance 0: conformal "
        "is still measuring in raw Euclidean space"
    )
