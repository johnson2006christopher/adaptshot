"""ACTEngine, tested directly for the first time (#74).

It was classified experimental in #72 for one reason: it is constructed inside
FewShotLearner and no test named it. These are the properties the class claims
in its own docstring and constructor, each asserted on its own.
"""

from __future__ import annotations

import pytest

from adaptshot import ACTEngine


def _drive(engine: ACTEngine, class_idx: int, *, wrong: bool, steps: int) -> None:
    """Feed `steps` rounds of feedback that were all wrong, or all right."""

    for _ in range(steps):
        engine.should_accept(
            0.5, class_idx,
            recent_incorrect_rate=1.0 if wrong else 0.0,
            recent_correct_rate=0.0 if wrong else 1.0,
        )


def test_decision_compares_confidence_to_the_class_threshold() -> None:
    engine = ACTEngine(base_threshold=0.65)
    # The first call moves the threshold by at most eta (0.01), so a margin of
    # 0.05 on either side is decisive whatever direction it moved.
    assert engine.should_accept(0.70, 0) == (True, "ACCEPT")
    assert engine.should_accept(0.60, 0) == (False, "REQUEST_FEEDBACK")


def test_wrong_feedback_raises_the_threshold_and_right_feedback_lowers_it() -> None:
    engine = ACTEngine(base_threshold=0.65)
    before = engine.get_threshold(0)

    _drive(engine, 0, wrong=True, steps=10)
    raised = engine.get_threshold(0)
    assert raised > before, "ten rounds of wrong predictions should make the class harder to accept"

    _drive(engine, 0, wrong=False, steps=30)
    assert engine.get_threshold(0) < raised, "confirmed predictions should lower it again"


def test_thresholds_stay_within_bounds_under_sustained_pressure() -> None:
    engine = ACTEngine(base_threshold=0.65, min_threshold=0.50, max_threshold=0.95)
    _drive(engine, 0, wrong=True, steps=500)
    assert engine.get_threshold(0) == pytest.approx(0.95)
    _drive(engine, 1, wrong=False, steps=500)
    assert engine.get_threshold(1) == pytest.approx(0.50)


def test_mean_reversion_pulls_a_drifted_threshold_back_toward_base() -> None:
    """With no error signal, a threshold pushed to the ceiling drifts back -- and not past base."""

    engine = ACTEngine(base_threshold=0.65, max_threshold=0.95)
    _drive(engine, 0, wrong=True, steps=500)
    assert engine.get_threshold(0) == pytest.approx(0.95)

    # Equal rates: the error term is zero and only mean reversion acts.
    for _ in range(500):
        engine.should_accept(0.5, 0, recent_incorrect_rate=0.5, recent_correct_rate=0.5)
    after = engine.get_threshold(0)
    assert 0.65 < after < 0.95, f"expected drift toward base from 0.95, got {after:.3f}"


def test_an_unseen_class_starts_at_the_mean_of_existing_thresholds() -> None:
    engine = ACTEngine(base_threshold=0.65, n_classes=2)
    _drive(engine, 0, wrong=True, steps=200)
    _drive(engine, 1, wrong=False, steps=200)
    expected = (engine.get_threshold(0) + engine.get_threshold(1)) / 2
    assert engine.get_threshold(99) == pytest.approx(expected, abs=1e-6)


def test_reset_class_returns_to_base_and_leaves_others_alone() -> None:
    engine = ACTEngine(base_threshold=0.65)
    _drive(engine, 0, wrong=True, steps=100)
    _drive(engine, 1, wrong=True, steps=100)
    engine.reset_class(0, base_threshold=0.65)
    assert engine.get_threshold(0) == pytest.approx(0.65)
    assert engine.get_threshold(1) > 0.65


def test_snapshot_covers_every_class_it_has_seen() -> None:
    engine = ACTEngine(n_classes=3)
    engine.should_accept(0.5, 7)  # dynamic expansion
    snapshot = engine.get_all_thresholds()
    assert set(snapshot) == {0, 1, 2, 7}
    assert all(0.50 <= value <= 0.95 for value in snapshot.values())


def test_same_inputs_give_the_same_trajectory() -> None:
    def trajectory() -> list[float]:
        engine = ACTEngine()
        out = []
        for step in range(50):
            engine.should_accept(0.6, 0, recent_incorrect_rate=(step % 3) / 2, recent_correct_rate=0.5)
            out.append(engine.get_threshold(0))
        return out

    assert trajectory() == trajectory()
