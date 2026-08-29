"""Does the prediction set contain the true label at least 1 - alpha of the time? (#14)

This is the claim the library exists for. Every other conformal test checks
plumbing; this one checks the theorem, on data where it can fail: five
Gaussian classes overlapping enough that top-1 is wrong about a quarter of the
time, so a set has to widen to cover, and a set that is too small shows up as
coverage that is too low.

It found a bug the first time it ran. ``_compute_quantile`` clamped the
conformal rank to ``n - 1``, so wherever ``n < (1 - alpha) / alpha`` -- where
the theorem says the quantile is infinite and every class must be included --
it returned the largest observed score instead. Four cells of the grid below
under-covered by 9 to 19 standard errors, one of them the library's own default
alpha at its own ``min_calibration_size``. The fix returns ``inf`` there, and
this file is what keeps it fixed.

**How the tolerance is derived, since #14 asks for a derivation and not an
epsilon.** The guarantee is *marginal*: it holds in expectation over the draw
of the calibration set and the test point together. So each cell draws a fresh
calibration set and a fresh test set ``TRIALS`` times, and the quantity the
theorem bounds is the mean of the per-trial coverages. Its standard error is the
sample standard deviation of those per-trial coverages over ``sqrt(TRIALS)`` --
computed at the trial level, so that correlation *within* a trial (all its test
points share one calibration set) is accounted for rather than assumed away.
The assertion is one-sided at three standard errors: ``mean + 3 SE >= 1 -
alpha``. A cell that truly meets the target fails that with probability about
0.13%, so the sixteen-cell grid has roughly a 2% chance of one spurious failure
per run, and no chance of missing an effect the size of the one it found.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from adaptshot.core.conformal import ConformalEngine
from adaptshot.utils.arrays import FloatArray, LabelArray

# The harness builds engines below their informative size on purpose -- that is
# the region under test -- and each one warns once at construction. The warning
# is right; it is also 640 lines of noise here. The test that checks the warning
# raises the level back with caplog.
logging.getLogger("adaptshot.core.conformal").setLevel(logging.ERROR)

CLASSES = 5
DIMENSIONS = 32
#: Class overlap. 6.0 puts top-1 accuracy near 74%, measured; at 2.0 every set
#: is a singleton with 100% coverage and the guarantee is never exercised.
SPREAD = 6.0
TRIALS = 40
TEST_POINTS = 150
ALPHAS = (0.01, 0.05, 0.10, 0.20)
CALIBRATION_SIZES = (10, 20, 50, 200)


def _episode(
    rng: np.random.Generator, n_calibration: int, n_test: int
) -> tuple[LabelArray, FloatArray, LabelArray, FloatArray, LabelArray]:
    """Distances-to-centroid for a fresh calibration and test draw."""

    centres = rng.normal(0.0, 1.0, (CLASSES, DIMENSIONS)) * 2.0
    labels: LabelArray = np.array([f"c{k}" for k in range(CLASSES)], dtype=object)

    def draw(count: int) -> tuple[FloatArray, LabelArray]:
        y = rng.integers(0, CLASSES, count)
        x = centres[y] + rng.normal(0.0, SPREAD, (count, DIMENSIONS))
        distances = np.linalg.norm(x[:, None, :] - centres[None, :, :], axis=2)
        return distances, labels[y]

    d_cal, y_cal = draw(n_calibration)
    d_test, y_test = draw(n_test)
    return labels, d_cal, y_cal, d_test, y_test


def measure(
    alpha: float, n_calibration: int, *, mode: str = "split", seed: int = 42
) -> tuple[float, float, float]:
    """(mean coverage, its standard error over trials, mean set size)."""

    rng = np.random.default_rng(seed)
    coverages: list[float] = []
    sizes: list[float] = []
    for _ in range(TRIALS):
        labels, d_cal, y_cal, d_test, y_test = _episode(rng, n_calibration, TEST_POINTS)
        engine = ConformalEngine(alpha=alpha, mode=mode, min_calibration_size=1)
        for row, truth in zip(d_cal, y_cal, strict=True):
            engine.update_calibration(engine.softmax_nonconformity(row, labels, truth), truth)

        covered = 0
        total_size = 0
        for row, truth in zip(d_test, y_test, strict=True):
            top = labels[int(np.argmin(row))]
            members = engine.predict_set(row, labels, top, 0.5).prediction_set
            covered += int(truth in members)
            total_size += len(members)
        coverages.append(covered / TEST_POINTS)
        sizes.append(total_size / TEST_POINTS)

    array = np.asarray(coverages)
    return float(array.mean()), float(array.std(ddof=1) / math.sqrt(TRIALS)), float(np.mean(sizes))


def _quantile_is_infinite(alpha: float, n_calibration: int) -> bool:
    """Where the theorem's quantile has no finite value: n < (1 - alpha) / alpha."""

    return math.ceil((n_calibration + 1) * (1.0 - alpha)) > n_calibration


@pytest.mark.parametrize("n_calibration", CALIBRATION_SIZES)
@pytest.mark.parametrize("alpha", ALPHAS)
def test_coverage_meets_the_target(alpha: float, n_calibration: int) -> None:
    mean, se, size = measure(alpha, n_calibration)
    target = 1.0 - alpha
    assert mean + 3.0 * se >= target, (
        f"alpha={alpha} n={n_calibration}: coverage {mean:.3f} +/- {se:.3f} against a "
        f"target of {target:.2f} -- short by {(target - mean) / max(se, 1e-9):.1f} standard "
        f"errors. Mean set size {size:.2f} of {CLASSES}."
    )


@pytest.mark.parametrize("n_calibration", CALIBRATION_SIZES)
@pytest.mark.parametrize("alpha", ALPHAS)
def test_sets_are_informative_wherever_the_theorem_allows(alpha: float, n_calibration: int) -> None:
    """Coverage alone is trivial -- return every class. Size is the other half.

    Where a finite quantile exists the mean set must be smaller than the label
    set; where it does not, the set must be *exactly* the label set, because
    that is the only honest answer and the bug was returning something smaller.
    """

    _, _, size = measure(alpha, n_calibration)
    if _quantile_is_infinite(alpha, n_calibration):
        assert size == pytest.approx(CLASSES), (
            f"alpha={alpha} n={n_calibration}: no finite quantile exists, so every set must "
            f"contain all {CLASSES} classes; mean size was {size:.2f}. A smaller set here "
            "is the under-coverage bug from #14 coming back."
        )
    else:
        assert size < CLASSES - 0.5, (
            f"alpha={alpha} n={n_calibration}: mean set size {size:.2f} of {CLASSES} -- the "
            "engine is covering by giving up rather than by discriminating."
        )


def test_the_cell_that_was_wrong() -> None:
    """The library's default alpha at its own min_calibration_size floor.

    Measured at 91.3% against 95% before the fix, with sets of size 2.2. Pinned
    here by name so a regression reports as the finding it is.
    """

    mean, se, size = measure(0.05, 10)
    assert mean + 3.0 * se >= 0.95, f"coverage {mean:.3f} +/- {se:.3f}"
    assert size == pytest.approx(CLASSES)


def test_cross_conformal_also_meets_the_target() -> None:
    mean, se, size = measure(0.10, 200, mode="cross")
    assert mean + 3.0 * se >= 0.90, f"cross-conformal coverage {mean:.3f} +/- {se:.3f}"
    assert size < CLASSES - 0.5


def test_engine_says_when_its_sets_cannot_be_informative(caplog: pytest.LogCaptureFixture) -> None:
    """At alpha = 0.05 nineteen scores are needed; a floor of 10 is warned about once, at construction."""

    with caplog.at_level(logging.WARNING, logger="adaptshot.core.conformal"):
        engine = ConformalEngine(alpha=0.05, min_calibration_size=10)
    assert engine.min_informative_size == 19
    assert any("uninformative" in record.message for record in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="adaptshot.core.conformal"):
        ConformalEngine(alpha=0.05, min_calibration_size=19)
    assert not caplog.records, "no warning when the floor is high enough"
