"""ACTEngine, tested directly (#74), re-drawn for the read/learn split (#116).

`should_accept` is a pure decision since #116 — asking does not move the
threshold — and `record_outcome` is the only mutator, fed by real corrections
and confirmations. The properties asserted here are the ones the class
docstring claims: wrong outcomes raise a class's bar, confirmations lower it,
bounds hold, drifted thresholds revert toward base, and the decision itself
never learns.
"""

from __future__ import annotations

import pytest

from adaptshot import ACTEngine


def _drive(engine: ACTEngine, class_idx: int, *, wrong: bool, steps: int) -> None:
    """Feed `steps` real outcomes that were all wrong, or all right."""

    for _ in range(steps):
        engine.record_outcome(class_idx, correct=not wrong)


def test_decision_compares_confidence_to_the_class_threshold() -> None:
    engine = ACTEngine(base_threshold=0.65)
    assert engine.should_accept(0.70, 0) == (True, "ACCEPT")
    assert engine.should_accept(0.60, 0) == (False, "REQUEST_FEEDBACK")


def test_asking_the_question_moves_nothing() -> None:
    """Forty identical decisions leave the threshold exactly where it was.

    The pre-#116 engine moved a threshold from 0.6500 to 0.6357 over forty
    predictions of the same photograph, with no human in the loop.
    """

    engine = ACTEngine(base_threshold=0.65)
    before = engine.get_all_thresholds()
    for _ in range(40):
        engine.should_accept(0.6, 0)
    assert engine.get_all_thresholds() == before


def test_the_deprecated_proxy_rates_warn_and_are_ignored() -> None:
    engine = ACTEngine(base_threshold=0.65)
    with pytest.warns(DeprecationWarning, match="record_outcome"):
        engine.should_accept(0.6, 0, recent_incorrect_rate=1.0, recent_correct_rate=0.0)
    assert engine.get_threshold(0) == pytest.approx(0.65), (
        "the deprecated rates must not move the threshold"
    )


def test_wrong_feedback_raises_the_threshold_and_right_feedback_lowers_it() -> None:
    engine = ACTEngine(base_threshold=0.65)
    before = engine.get_threshold(0)

    _drive(engine, 0, wrong=True, steps=10)
    raised = engine.get_threshold(0)
    assert raised > before, "ten wrong predictions should make the class harder to accept"

    _drive(engine, 0, wrong=False, steps=30)
    assert engine.get_threshold(0) < raised, "confirmed predictions should lower it again"


def test_thresholds_stay_within_bounds_under_sustained_pressure() -> None:
    engine = ACTEngine(base_threshold=0.65, min_threshold=0.50, max_threshold=0.95)
    _drive(engine, 0, wrong=True, steps=500)
    assert engine.get_threshold(0) == pytest.approx(0.95)
    _drive(engine, 1, wrong=False, steps=500)
    assert engine.get_threshold(1) == pytest.approx(0.50)


def test_mean_reversion_pulls_a_drifted_threshold_back_toward_base() -> None:
    """Alternating outcomes cancel the error term; only mean reversion acts."""

    engine = ACTEngine(base_threshold=0.65, max_threshold=0.95)
    _drive(engine, 0, wrong=True, steps=500)
    assert engine.get_threshold(0) == pytest.approx(0.95)

    for step in range(500):
        engine.record_outcome(0, correct=step % 2 == 0)
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
    engine.record_outcome(7, correct=True)  # dynamic expansion happens on outcomes
    snapshot = engine.get_all_thresholds()
    assert set(snapshot) == {0, 1, 2, 7}
    assert all(0.50 <= value <= 0.95 for value in snapshot.values())


def test_same_outcomes_give_the_same_trajectory() -> None:
    def trajectory() -> list[float]:
        engine = ACTEngine()
        out = []
        for step in range(50):
            engine.record_outcome(0, correct=step % 3 == 0)
            out.append(engine.get_threshold(0))
        return out

    assert trajectory() == trajectory()
