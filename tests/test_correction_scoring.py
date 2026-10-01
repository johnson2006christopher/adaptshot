"""Corrections must be scored held-out, not in-sample (#112).

`correct()` used to append the corrected point to the similarity buffer,
rebuild the prototypes, and only then compute the point's nonconformity — so
every stored calibration score measured a point against a prototype that
already contained it. Scores biased low push `q_hat` down, and a loop of
corrections *tightened* the prediction sets instead of recalibrating them,
the opposite of what the docs promise.

The invariant these tests pin: the score stored for a correction equals the
score of the same point computed against the prototypes as they stood BEFORE
the correction — the state that would have predicted it, which is the same
footing every future test point is scored on.
"""

from __future__ import annotations

import numpy as np
import pytest

from adaptshot import FewShotLearner
from adaptshot.data import sample_images


@pytest.fixture()
def trained() -> tuple[FewShotLearner, str, str]:
    """A learner taught 11 of the 12 bundled photographs; the 12th held out.

    The correction label is deliberately NOT the held-out photograph's own
    class: a correction happens when the model was wrong, and only then is the
    nonconformity ratio above its floor of 1.0 — a correctly-predicted point
    scores exactly 1.0 held-out and in-sample alike, which would make these
    tests vacuously pass under the old in-sample scoring too.
    """

    paths, labels = sample_images()
    learner = FewShotLearner()
    learner.load_support_images(paths[:-1], labels[:-1])
    other_label = next(
        name for name in dict.fromkeys(str(x) for x in labels) if name != str(labels[-1])
    )
    return learner, paths[-1], other_label


def test_stored_score_is_computed_against_pre_correction_prototypes(
    trained: tuple[FewShotLearner, str, str],
) -> None:
    learner, held_out_path, true_label = trained

    # The expected score, derived the way `correct()` derives it but from the
    # UNTOUCHED state: embed the photograph, measure distances to the current
    # prototypes, score. Private helpers on purpose — the test must use the
    # same arithmetic as the implementation, or a change to either would make
    # it compare two different quantities.
    image = learner._load_rgb_image_from_path(held_out_path)
    embedding = learner._extract_embedding_checked(image=image, source=held_out_path)
    pre_distances = learner._compute_all_prototype_distances(embedding)
    expected = learner.conformal.nonconformity(
        pre_distances, learner._prototype_labels, true_label
    )

    before = learner.conformal.calibration_size
    learner.correct(image_path=held_out_path, true_label=true_label)

    assert learner.conformal.calibration_size == before + 1, (
        "a correction to a known class must store exactly one calibration score"
    )
    stored = learner.conformal._calibration_scores[-1]
    assert stored == pytest.approx(expected, rel=1e-6), (
        f"stored score {stored} is not the held-out score {expected}; "
        "the correction was scored against prototypes that already contain it"
    )


def test_in_sample_scoring_would_have_stored_a_lower_score(
    trained: tuple[FewShotLearner, str, str],
) -> None:
    """The bias the fix removes is real and directional on real photographs.

    After the prototypes absorb the point, its distance to its own class
    shrinks, so the in-sample score is lower than the held-out one. If this
    assertion ever fails, the two scores coincided and the main test above is
    not distinguishing anything — strengthen the fixture rather than relax it.
    """

    learner, held_out_path, true_label = trained

    learner.correct(image_path=held_out_path, true_label=true_label)
    stored = learner.conformal._calibration_scores[-1]

    # Recompute against the post-correction prototypes: the in-sample score.
    image = learner._load_rgb_image_from_path(held_out_path)
    embedding = learner._extract_embedding_checked(image=image, source=held_out_path)
    post_distances = learner._compute_all_prototype_distances(embedding)
    in_sample = learner.conformal.nonconformity(
        post_distances, learner._prototype_labels, true_label
    )

    assert in_sample < stored, (
        f"in-sample score {in_sample} is not below the held-out score {stored}; "
        "either the correction did not move the prototype or the score no "
        "longer depends on the prototype distance"
    )


def test_new_class_correction_stores_no_calibration_score(
    trained: tuple[FewShotLearner, str, str],
) -> None:
    """A class with no pre-correction prototype has no measurable score.

    `nonconformity` returns +inf for a label outside the prototype set — a
    statement about the label space, not a measurement of the photograph.
    Storing it would poison the quantile; the correction still teaches the
    class, it just contributes no calibration until scored against a state
    that knows it.
    """

    learner, held_out_path, _ = trained

    before = learner.conformal.calibration_size
    learner.correct(image_path=held_out_path, true_label="a_brand_new_condition")

    assert learner.conformal.calibration_size == before, (
        "a correction introducing a new class must not store a calibration score"
    )
    assert "a_brand_new_condition" in learner._prototype_labels, (
        "the new class must still be learned — only the calibration is skipped"
    )
    assert not any(
        np.isinf(score) for score in learner.conformal._calibration_scores
    ), "an infinite score reached the calibration buffer"


def test_followup_correction_to_a_new_class_is_scored_once_it_has_a_prototype(
    trained: tuple[FewShotLearner, str, str],
) -> None:
    learner, held_out_path, _ = trained

    learner.correct(image_path=held_out_path, true_label="a_brand_new_condition")
    before = learner.conformal.calibration_size
    # Same photograph again: the class now has a prototype (containing exactly
    # this point — which is why the score is taken before any re-teaching).
    learner.correct(image_path=held_out_path, true_label="a_brand_new_condition")

    assert learner.conformal.calibration_size == before + 1, (
        "once the class has a prototype, corrections to it must calibrate"
    )
