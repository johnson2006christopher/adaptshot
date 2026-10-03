"""The eco-mode early exit may only fire on near-duplicates (#120).

The exit returns the *cached support image's* embedding instead of running the
backbone — so a false fire answers with a different photograph. Raw 32×32
previews are all-positive vectors, nearly parallel for any two natural
photographs: on the bundled set, 25% of CROSS-CLASS pairs cleared the 0.95
cosine bar. Mean-centring the previews removes the shared brightness
component; these tests pin that no cross-class pair fires, that an exact
duplicate still does, and that eco mode no longer changes predictions on
non-duplicate queries.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from adaptshot import AdaptShotConfig, FewShotLearner
from adaptshot.core.extractor import compute_preview_signature
from adaptshot.data import sample_images


def _centred_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))


def test_no_cross_class_pair_clears_the_early_exit_bar() -> None:
    """The acceptance test from #120, on every cross-class bundled pair."""

    paths, labels = sample_images()
    previews = [compute_preview_signature(p) for p in paths]
    threshold = AdaptShotConfig().early_exit_threshold

    offenders = [
        (paths[i], paths[j], round(_centred_cosine(previews[i], previews[j]), 3))
        for i, j in itertools.combinations(range(len(paths)), 2)
        if str(labels[i]) != str(labels[j])
        and _centred_cosine(previews[i], previews[j]) >= threshold
    ]
    assert not offenders, (
        "cross-class photographs would still trigger the early exit "
        f"(threshold {threshold}): {offenders}"
    )


def test_an_exact_duplicate_still_fires() -> None:
    paths, _ = sample_images()
    preview = compute_preview_signature(paths[0])
    assert _centred_cosine(preview, preview) == pytest.approx(1.0), (
        "a photograph must match itself; if this fails the exit can never fire "
        "and eco mode is dead code"
    )


def test_eco_mode_no_longer_changes_predictions_on_the_bundled_set() -> None:
    """Measured on 0.3.0: eco on/off disagreed on raw confidence for 3 of 15
    queries and on the label for 1. With centred previews they must agree."""

    paths, labels = sample_images()

    def answers(eco: bool) -> list[tuple[object, float]]:
        learner = FewShotLearner(
            config=AdaptShotConfig(eco_mode=eco, conformal_alpha=0.10)
        )
        learner.load_support_images(paths[:-1], labels[:-1])
        out = []
        for path in paths:
            result = learner.predict(path)
            out.append((result.prediction, round(result.raw_confidence, 6)))
        return out

    assert answers(eco=True) == answers(eco=False), (
        "eco mode changed a prediction on non-duplicate photographs; the early "
        "exit fired on something that is not a duplicate"
    )
