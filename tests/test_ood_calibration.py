"""Both OOD error rates, measured together (#54).

The detector flagged 8 of 15 in-distribution PlantVillage queries as
out-of-distribution -- images from the same dataset, classes and lab conditions
as the support set it had just been given. `enable_ood_detection` is on by
default in both shipped configs, and the interface presents the flag to a user
as "this doesn't look like anything I was taught". A detector that fires on half
of valid inputs trains people to ignore it, which is exactly when the one true
positive arrives.

Both rates are asserted in the same module on purpose, because either one alone
is trivially perfect: a detector that never fires has a 0% false-positive rate,
and one that always fires has a 100% true-positive rate. Only the pair says
anything.

Synthetic Gaussians rather than photographs, deliberately. The failure was never
about images -- it was that the threshold was a percentile of distances measured
on the same points that defined the distribution -- and isotropic clusters make
"in-distribution" and "out-of-distribution" facts of construction rather than
matters of opinion. It also keeps the suite offline, as CLAUDE.md requires.
"""

from __future__ import annotations

import numpy as np
import pytest

from adaptshot.core.uncertainty import UncertaintyQuantifier

DIMENSIONS = 512
CLASSES = 3
QUERIES_PER_CLASS = 100

#: Nominal is 5%, since the threshold is the 95th percentile. Leave-one-out is
#: mildly conservative in the few-shot regime -- a fit on n-1 points is worse
#: than the fit on n that gets used -- so measured rates run 0.3%-4.0%. The
#: bound is well above that and far below the 100% the old estimator produced,
#: so it catches a regression without pinning the exact statistic.
MAX_FALSE_POSITIVE_RATE = 15.0

#: A detector allowed to be quiet on genuine OOD is not a detector. Measured at
#: 100% for every separation tested, including the closest.
MIN_TRUE_POSITIVE_RATE = 90.0


def _clusters(n_shot: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Support set, held-out in-distribution queries, and genuine OOD.

    The OOD cluster sits two units of within-class scale away from a real class
    centre -- near enough to be a fair test, rather than noise from the far side
    of the space that anything would catch.
    """

    rng = np.random.default_rng(seed)
    centres = (rng.normal(0, 1, (CLASSES, DIMENSIONS)) * 3.0).astype(np.float32)

    def draw(centre: np.ndarray, count: int) -> np.ndarray:
        return (centre + rng.normal(0, 1, (count, DIMENSIONS))).astype(np.float32)

    support = np.concatenate([draw(centres[k], n_shot) for k in range(CLASSES)])
    in_distribution = np.concatenate(
        [draw(centres[k], QUERIES_PER_CLASS) for k in range(CLASSES)]
    )
    away = rng.normal(0, 1, DIMENSIONS).astype(np.float32) * 2.0
    out_of_distribution = draw(centres[0] + away, QUERIES_PER_CLASS)
    return support, in_distribution, out_of_distribution


def _labels(n_shot: int) -> np.ndarray:
    return np.array(
        [f"c{k}" for k in range(CLASSES) for _ in range(n_shot)], dtype=object
    )


def _flag_rate(quantifier: UncertaintyQuantifier, points: np.ndarray) -> float:
    """Percentage of `points` the detector calls out-of-distribution."""

    return 100.0 * float(np.mean([quantifier.is_ood(point)[0] for point in points]))


@pytest.fixture(params=[5, 10, 20], ids=lambda n: f"{n}-shot")
def fitted(request: pytest.FixtureRequest) -> tuple[UncertaintyQuantifier, np.ndarray, np.ndarray]:
    n_shot = request.param
    support, in_distribution, out_of_distribution = _clusters(n_shot)
    quantifier = UncertaintyQuantifier()
    quantifier.fit_class_distributions(support, _labels(n_shot))
    return quantifier, in_distribution, out_of_distribution


def test_in_distribution_queries_are_not_flagged(
    fitted: tuple[UncertaintyQuantifier, np.ndarray, np.ndarray],
) -> None:
    """The bug in #54, in the form that made it visible."""

    quantifier, in_distribution, _ = fitted
    rate = _flag_rate(quantifier, in_distribution)
    assert rate <= MAX_FALSE_POSITIVE_RATE, (
        f"{rate:.1f}% of held-out in-distribution queries were flagged as OOD, "
        f"over the {MAX_FALSE_POSITIVE_RATE}% bound. These are drawn from the "
        "same distribution as the support set, so the honest rate is the "
        "detector's nominal 5%."
    )


def test_genuine_ood_is_flagged(
    fitted: tuple[UncertaintyQuantifier, np.ndarray, np.ndarray],
) -> None:
    """The other direction, without which the test above is trivial to pass."""

    quantifier, _, out_of_distribution = fitted
    rate = _flag_rate(quantifier, out_of_distribution)
    assert rate >= MIN_TRUE_POSITIVE_RATE, (
        f"only {rate:.1f}% of genuine out-of-distribution inputs were flagged, "
        f"under the {MIN_TRUE_POSITIVE_RATE}% bound. A detector that stays "
        "quiet on real OOD is not buying anything for the false positives it "
        "does produce."
    )


def test_calibration_distances_are_not_degenerate() -> None:
    """The mechanism itself, asserted directly so a regression names its cause.

    Calibrating on the points that defined the distribution collapsed the
    distances to almost a single value -- 13.97 to 14.36 across a whole support
    set. A percentile of that is an artifact, not a threshold, and the two rate
    tests above would report the symptom without the reason.
    """

    support, in_distribution, _ = _clusters(n_shot=5)
    quantifier = UncertaintyQuantifier()
    quantifier.fit_class_distributions(support, _labels(5))

    calibration = np.asarray(quantifier._calibration_distances)
    assert calibration.size > 0, "no calibration distances were recorded"

    spread = float(calibration.max() - calibration.min())
    assert spread > 0.05 * float(np.median(calibration)), (
        f"calibration distances span only {spread:.2f} around a median of "
        f"{np.median(calibration):.2f}. They are being measured against a "
        "distribution the points themselves defined, so the percentile "
        "describes the fit rather than what a new image will score."
    )

    # The threshold has to sit above what real in-distribution images score,
    # which is the comparison that actually failed: 14.4 against a median of 39.1.
    typical = float(
        np.median([quantifier.min_mahalanobis_distance(q)[0] for q in in_distribution])
    )
    assert quantifier._ood_threshold > typical, (
        f"the OOD threshold is {quantifier._ood_threshold:.1f} while a typical "
        f"in-distribution query scores {typical:.1f}, so most valid inputs are "
        "flagged by construction."
    )


def test_detection_is_disabled_when_no_class_can_hold_one_out() -> None:
    """Two examples per class cannot support a held-out estimate.

    Silence is the safe direction here. The alternative -- falling back to the
    in-sample threshold -- is what produced the 100% false-positive rate, so a
    detector that admits it cannot calibrate is strictly better than one that
    guesses.
    """

    rng = np.random.default_rng(7)
    centres = (rng.normal(0, 1, (6, DIMENSIONS)) * 3.0).astype(np.float32)
    support = np.concatenate(
        [(centre + rng.normal(0, 1, (2, DIMENSIONS))).astype(np.float32) for centre in centres]
    )
    labels = np.array([f"c{k}" for k in range(6) for _ in range(2)], dtype=object)

    quantifier = UncertaintyQuantifier()
    quantifier.fit_class_distributions(support, labels)

    assert quantifier._ood_threshold == float("inf")
    flagged = [quantifier.is_ood(point)[0] for point in support]
    assert not any(flagged), "the detector fired despite having no calibration"
