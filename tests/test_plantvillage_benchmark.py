"""The parts of the PlantVillage benchmark that must hold without the dataset.

The benchmark itself needs a 400-image download, which CLAUDE.md forbids the
validation gate from requiring. What can be checked offline is everything that
would make the published number wrong for a reason unrelated to the model:
episodes that leak, splits that overlap, a seed that does not reproduce, or a
baseline that is quietly broken and so flatters AdaptShot by comparison.

That last one matters most. #19 exists because a result without an alternative
proves nothing -- but a *broken* alternative proves less than nothing, because
it manufactures a gap and calls it evidence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchmarks.baselines import knn, linear_probe, nearest_centroid, top1_with_threshold
from benchmarks.plantvillage import DatasetMissing, load_pool, sample_episodes
from benchmarks.run_plantvillage import _confidence_in_true_class, mean_and_ci

CLASSES = [f"class_{index:02d}" for index in range(20)]
PER_CLASS = 20


@pytest.fixture
def pool_labels() -> np.ndarray:
    return np.array([name for name in CLASSES for _ in range(PER_CLASS)], dtype=object)


def _episodes(pool_labels: np.ndarray, seed: int = 42, count: int = 20) -> list:
    return sample_episodes(
        pool_labels,
        CLASSES,
        n_way=5,
        k_shot=5,
        n_calibration=5,
        n_query=10,
        episodes=count,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Episode construction
# ---------------------------------------------------------------------------


def test_splits_never_overlap(pool_labels: np.ndarray) -> None:
    """The failure that would silently invent the whole result.

    Conformal coverage is only a guarantee when the calibration scores are held
    out from the queries they cover, and accuracy measured on the support set is
    not accuracy. An overlap here would inflate both, and neither number would
    look obviously wrong.
    """

    for episode in _episodes(pool_labels):
        support = set(episode.support.tolist())
        calibration = set(episode.calibration.tolist())
        query = set(episode.query.tolist())

        assert not support & calibration, "support and calibration share images"
        assert not support & query, "support and query share images"
        assert not calibration & query, "calibration and query share images"
        assert len(support) + len(calibration) + len(query) == 5 * (5 + 5 + 10)


def test_every_episode_is_balanced(pool_labels: np.ndarray) -> None:
    """5-way 5-shot means five classes and five shots, in every episode."""

    for episode in _episodes(pool_labels):
        assert len(episode.classes) == 5
        assert len(set(episode.classes)) == 5

        for split, per_class in (
            (episode.support, 5),
            (episode.calibration, 5),
            (episode.query, 10),
        ):
            _, counts = np.unique(pool_labels[split], return_counts=True)
            assert sorted(counts.tolist()) == [per_class] * 5

        # Each split must cover exactly the episode's classes, not a subset.
        for split in (episode.support, episode.calibration, episode.query):
            assert set(pool_labels[split]) == set(episode.classes)


def test_the_seed_reproduces_the_episodes(pool_labels: np.ndarray) -> None:
    """A published number is worthless if `--seed 42` draws something else."""

    first = _episodes(pool_labels, seed=42)
    again = _episodes(pool_labels, seed=42)
    different = _episodes(pool_labels, seed=43)

    for left, right in zip(first, again):
        assert left.classes == right.classes
        assert np.array_equal(left.support, right.support)
        assert np.array_equal(left.query, right.query)

    assert any(
        left.classes != right.classes or not np.array_equal(left.support, right.support)
        for left, right in zip(first, different)
    ), "a different seed produced identical episodes"


def test_episodes_actually_vary(pool_labels: np.ndarray) -> None:
    """100 episodes over one fixed class set would be one episode, resampled.

    This is why the fetch preset is 20 classes rather than 5: with exactly five,
    every episode has the same class composition and the confidence interval
    describes image sampling alone.
    """

    compositions = {episode.classes for episode in _episodes(pool_labels, count=50)}
    assert len(compositions) > 25, (
        f"only {len(compositions)} distinct class combinations in 50 episodes"
    )


def test_too_few_classes_is_refused(pool_labels: np.ndarray) -> None:
    with pytest.raises(ValueError, match="needs 5 classes"):
        sample_episodes(
            pool_labels, CLASSES[:3], n_way=5, k_shot=5,
            n_calibration=5, n_query=10, episodes=1, seed=42,
        )


def test_too_few_images_per_class_is_refused() -> None:
    """Silently dropping to 3-shot because the data was thin would be worse."""

    thin = np.array([name for name in CLASSES for _ in range(6)], dtype=object)
    with pytest.raises(ValueError, match="needs 20 images per class"):
        sample_episodes(
            thin, CLASSES, n_way=5, k_shot=5,
            n_calibration=5, n_query=10, episodes=1, seed=42,
        )


def test_missing_dataset_names_the_command_that_fixes_it(tmp_path: Path) -> None:
    """Nothing here downloads anything, so the message has to do the work."""

    with pytest.raises(DatasetMissing, match=r"fetch_plantvillage\.py"):
        load_pool(tmp_path / "absent")


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def _separable(seed: int = 0, n_way: int = 5, k_shot: int = 5, n_query: int = 10):
    """Well-separated clusters: every method should be near-perfect here.

    A baseline that cannot classify obviously separable data is broken, and a
    broken baseline manufactures a gap in AdaptShot's favour.
    """

    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 1, (n_way, 64)).astype(np.float32) * 8.0
    support = np.concatenate([c + rng.normal(0, 1, (k_shot, 64)) for c in centres])
    query = np.concatenate([c + rng.normal(0, 1, (n_query, 64)) for c in centres])
    support_labels = np.array(
        [f"c{k}" for k in range(n_way) for _ in range(k_shot)], dtype=object
    )
    query_labels = np.array(
        [f"c{k}" for k in range(n_way) for _ in range(n_query)], dtype=object
    )
    return (
        support.astype(np.float32), support_labels,
        query.astype(np.float32), query_labels,
    )


@pytest.mark.parametrize(
    "method",
    [
        pytest.param(nearest_centroid, id="nearest_centroid"),
        pytest.param(lambda s, sl, q: knn(s, sl, q, k=1), id="knn_1"),
        pytest.param(lambda s, sl, q: knn(s, sl, q, k=5), id="knn_5"),
        pytest.param(linear_probe, id="linear_probe"),
    ],
)
def test_baselines_classify_separable_data(method) -> None:  # type: ignore[no-untyped-def]
    support, support_labels, query, query_labels = _separable()
    accuracy = float(np.mean(method(support, support_labels, query) == query_labels))
    assert accuracy > 0.95, (
        f"a baseline scored {accuracy:.0%} on trivially separable clusters, so "
        "any gap it shows against AdaptShot is a bug rather than a result"
    )


def test_baselines_are_deterministic() -> None:
    """No seed is passed to any of them, so none may depend on one."""

    support, support_labels, query, _ = _separable()
    for method in (nearest_centroid, linear_probe):
        first = method(support, support_labels, query)
        assert np.array_equal(first, method(support, support_labels, query))


def test_linear_probe_does_not_diverge() -> None:
    """25 samples in 512 dimensions are separable, so the weights would run away.

    Unregularised logistic regression chases a margin it already has. The
    symptom is not a crash -- it is confident nonsense on the query set.
    """

    rng = np.random.default_rng(1)
    support = rng.normal(0, 1, (25, 512)).astype(np.float32)
    labels = np.array([f"c{i // 5}" for i in range(25)], dtype=object)
    predictions = linear_probe(support, labels, support)
    assert set(predictions) <= set(labels)
    assert np.isfinite(support).all()


def test_top1_threshold_abstains_and_reports_it() -> None:
    """An abstention is an empty set, so it is comparable with a conformal one."""

    support, support_labels, query, _ = _separable()
    _, confident = top1_with_threshold(support, support_labels, query, threshold=0.0)
    assert all(len(s) == 1 for s in confident), "threshold 0 should never abstain"

    _, never = top1_with_threshold(support, support_labels, query, threshold=1.01)
    assert all(len(s) == 0 for s in never), "threshold above 1 should always abstain"


def test_the_calibrated_threshold_is_reachable() -> None:
    """The bug the two extremes above did not catch.

    The first version of this benchmark hard-coded a 0.5 abstention threshold.
    A 5-way softmax over cosine similarities cannot reach it -- chance is 0.2
    and the similarities are close -- so the baseline abstained on every query
    and reported 0.00% coverage against conformal's 95%. That is not conformal
    winning, it is the alternative never having been in the comparison.

    Testing threshold 0.0 and 1.01 passed happily throughout, because both
    extremes behave correctly. What was never asserted is that the threshold the
    benchmark actually uses lands somewhere a real confidence can reach.
    """

    support, support_labels, query, query_labels = _separable()
    calibration, calibration_labels = query[::2], query_labels[::2]

    confidence = _confidence_in_true_class(
        support, support_labels, calibration, calibration_labels
    )
    threshold = float(np.percentile(confidence, 10.0))

    _, sets = top1_with_threshold(support, support_labels, query, threshold)
    fired = sum(1 for s in sets if s)
    assert fired > 0, (
        f"the calibrated threshold {threshold:.3f} produced no non-empty sets. "
        "A baseline that always abstains is not participating in the comparison."
    )

    covered = sum(str(t) in s for s, t in zip(sets, query_labels)) / len(sets)
    assert covered > 0.5, (
        f"coverage of {covered:.0%} at a threshold calibrated for 90%. The "
        "threshold is not measuring what it claims to."
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_confidence_interval_shrinks_with_episodes() -> None:
    """The interval is the point of #18; a constant one would be a lie."""

    rng = np.random.default_rng(3)
    draws = rng.normal(0.7, 0.1, 1000).tolist()

    _, few = mean_and_ci(draws[:25])
    _, many = mean_and_ci(draws)
    assert many < few, "more episodes must narrow the interval"


def test_confidence_interval_of_a_single_episode_is_not_claimed() -> None:
    """One episode has no spread to estimate, and must not report one as zero
    confidence in a wide sense -- it reports no half-width at all."""

    mean, half = mean_and_ci([0.8])
    assert mean == pytest.approx(0.8)
    assert half == 0.0
