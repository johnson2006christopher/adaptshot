"""`predict()` must observe, never testify (#116).

Before the fix, every prediction mutated three engines: the calibrator gained
a fabricated `correct=True` entry (forty predictions of one photograph
manufactured forty perfect outcomes), the ACT threshold moved off a
confidence proxy (0.6500 -> 0.6357 over forty identical calls), and the two
drifts fed each other through the calibrated confidence. Asking a model the
same question twice must return the same answer and leave the model exactly
as it was.

Adaptation still happens — in `correct()`, from real outcomes, which these
tests also pin.
"""

from __future__ import annotations

import dataclasses

from adaptshot import FewShotLearner
from adaptshot.data import sample_images


def _fresh() -> tuple[FewShotLearner, str, str, str]:
    paths, labels = sample_images()
    learner = FewShotLearner()
    learner.load_support_images(paths[:-1], labels[:-1])
    other = next(
        name for name in dict.fromkeys(str(x) for x in labels) if name != str(labels[-1])
    )
    return learner, paths[-1], str(labels[-1]), other


def _engine_state(learner: FewShotLearner) -> dict[str, object]:
    """Everything a prediction must not change."""

    return {
        "calibration_window": list(learner.calibrator._window_confidences),
        "calibration_outcomes": list(learner.calibrator._window_correct),
        "temperature": learner.calibrator.temperature,
        "act_thresholds": learner.act.get_all_thresholds(),
        "conformal_scores": list(learner.conformal._calibration_scores),
    }


def test_forty_predictions_return_identical_results_and_change_nothing() -> None:
    learner, held_out, _, _ = _fresh()

    before = _engine_state(learner)
    first = learner.predict(held_out)
    results = [learner.predict(held_out) for _ in range(39)]

    assert _engine_state(learner) == before, (
        "predict() mutated the calibrator, ACT or conformal engine; "
        "prediction is a question, not evidence"
    )
    for later in results:
        assert dataclasses.asdict(later) == dataclasses.asdict(first), (
            "the same photograph got a different answer on a repeat call"
        )


def test_correction_with_real_outcome_updates_calibrator_and_act() -> None:
    """Adaptation moved to correct(), it did not disappear."""

    learner, held_out, _, wrong_label = _fresh()

    predicted = learner.predict(held_out).prediction
    predicted_idx = learner._label_to_idx[predicted]
    window_before = len(learner.calibrator._window_confidences)
    threshold_before = learner.act.get_threshold(predicted_idx)

    # The human says the model was wrong (the fixture guarantees the label
    # differs from the prediction's own class or, at worst, from the truth).
    correction_label = wrong_label if wrong_label != str(predicted) else "healthy"
    learner.correct(image_path=held_out, true_label=correction_label)

    assert len(learner.calibrator._window_confidences) == window_before + 1, (
        "a correction must add exactly one real observation to the calibration window"
    )
    assert learner.calibrator._window_correct[-1] is False, (
        "the recorded outcome must be the real one: the prediction was wrong"
    )
    assert learner.act.get_threshold(predicted_idx) > threshold_before, (
        "a wrong prediction must raise the predicted class's ACT threshold"
    )


def test_confirmation_lowers_the_threshold() -> None:
    learner, held_out, true_label, _ = _fresh()

    predicted = learner.predict(held_out).prediction
    predicted_idx = learner._label_to_idx[predicted]
    threshold_before = learner.act.get_threshold(predicted_idx)

    # The human confirms the model's own answer.
    learner.correct(image_path=held_out, true_label=str(predicted))

    assert learner.calibrator._window_correct[-1] is True
    assert learner.act.get_threshold(predicted_idx) < threshold_before, (
        "a confirmed prediction must lower the predicted class's ACT threshold"
    )
    assert true_label  # the fixture's truth is unused here on purpose: the
    # confirmation scenario is about agreeing with the model, right or wrong.


def test_correction_records_the_mode_consistent_prediction() -> None:
    """`correct()` must see the same prediction the user saw (#116 item 4).

    In the default prototypical mode, the old code ran a 1-NN lookup inside
    correct() regardless of inference_mode, so the replay buffer could record
    a "predicted" label that predict() never returned. Both paths now share
    `_infer`; this pins that the shared dispatch agrees with predict().
    """

    learner, held_out, _, _ = _fresh()

    shown = learner.predict(held_out).prediction
    image = learner._load_rgb_image_from_path(held_out)
    embedding = learner._extract_embedding_checked(image=image, source=held_out)
    inferred_label = learner._infer(embedding)[0]

    assert str(inferred_label) == str(shown), (
        "the dispatch correct() uses disagrees with what predict() returned"
    )
