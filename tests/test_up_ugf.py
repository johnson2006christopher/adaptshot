"""UPUGFPruner, tested directly for the first time (#74).

Each of the three score components is isolated by zeroing the other two
weights, so a property is asserted against the component that owns it rather
than against a product of all three.
"""

from __future__ import annotations

import numpy as np

from adaptshot import UPUGFPruner

NOW = 1_700_000_000.0


def _buffer(n: int, d: int = 16, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(0, 1, (n, d)).astype(np.float32)
    labels = np.array([f"c{i % 3}" for i in range(n)], dtype=object)
    uncertainties = rng.uniform(0, 1, n).astype(np.float32)
    times = np.full(n, NOW, dtype=np.float64)
    return embeddings, labels, uncertainties, times


def test_under_capacity_is_the_identity() -> None:
    e, labels, u, t = _buffer(8)
    out = UPUGFPruner(capacity=10).prune(e, labels, u, t)
    assert all(a is b for a, b in zip(out, (e, labels, u, t), strict=True))


def test_over_capacity_keeps_exactly_capacity_rows_aligned_across_arrays() -> None:
    e, labels, u, t = _buffer(30)
    tags = {tuple(row): lab for row, lab in zip(e.tolist(), labels, strict=True)}
    pe, pl, pu, pt = UPUGFPruner(capacity=12).prune(e, labels, u, t)
    assert pe.shape == (12, 16) and len(pl) == len(pu) == len(pt) == 12
    # Each surviving row still carries its own label, not a neighbour's.
    assert all(tags[tuple(row)] == lab for row, lab in zip(pe.tolist(), pl, strict=True))


def test_redundancy_evicts_the_duplicate_and_keeps_the_unique_point() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 16).astype(np.float32)
    e = np.stack([base, base + 1e-3, base + 2e-3, rng.normal(0, 1, 16).astype(np.float32)])
    labels = np.array(["a", "a", "a", "b"], dtype=object)
    u = np.zeros(4, dtype=np.float32)
    t = np.full(4, NOW)
    pruner = UPUGFPruner(capacity=2, uncertainty_weight=0.0, recency_weight=0.0, redundancy_weight=1.0)
    scores = pruner.compute_scores(e, u, t, current_time=NOW)
    assert scores[3] > scores[:3].max(), "the isolated point must outscore every near-duplicate"
    _, kept, _, _ = pruner.prune(e, labels, u, t)
    assert "b" in kept


def test_recency_prefers_the_recently_accessed() -> None:
    e, _, u, _ = _buffer(6)
    u[:] = 0.0
    t = np.array([NOW, NOW - 10, NOW - 100, NOW - 1000, NOW - 10000, NOW - 100000])
    pruner = UPUGFPruner(uncertainty_weight=0.0, recency_weight=1.0, redundancy_weight=0.0, recency_decay=0.01)
    scores = pruner.compute_scores(e, u, t, current_time=NOW)
    assert list(np.argsort(-scores)) == [0, 1, 2, 3, 4, 5], "scores must fall monotonically with staleness"


def test_uncertainty_weight_retains_uncertain_examples_as_documented() -> None:
    """The constructor documents `uncertainty_weight` as the importance of
    *retaining* uncertain, boundary examples. So with only that component
    active, the survivors of a prune must be more uncertain than the evicted."""

    e, labels, _, t = _buffer(20)
    u = np.linspace(0.0, 1.0, 20, dtype=np.float32)
    pruner = UPUGFPruner(capacity=10, uncertainty_weight=1.0, recency_weight=0.0, redundancy_weight=0.0)
    _, _, kept_u, _ = pruner.prune(e, labels, u, t)
    evicted_mean = (u.sum() - kept_u.sum()) / 10
    assert kept_u.mean() > evicted_mean, (
        f"kept mean uncertainty {kept_u.mean():.2f} <= evicted {evicted_mean:.2f}: the pruner is "
        "evicting the boundary examples its docstring says it retains"
    )


def test_scores_are_deterministic() -> None:
    e, _, u, t = _buffer(40)
    p = UPUGFPruner()
    assert np.array_equal(p.compute_scores(e, u, t, current_time=NOW), p.compute_scores(e, u, t, current_time=NOW))


def test_empty_buffer_scores_to_empty() -> None:
    p = UPUGFPruner()
    assert p.compute_scores(np.zeros((0, 4), dtype=np.float32), np.zeros(0), np.zeros(0)).size == 0


def test_large_buffer_uses_the_approximate_path_and_still_detects_duplicates() -> None:
    """Above 100 rows the redundancy term switches to LSH. It is a proxy, so the
    assertion is the one a proxy can keep: a planted block of near-duplicates
    scores below the isolated points around it."""

    rng = np.random.default_rng(2)
    n, d = 150, 32
    e = rng.normal(0, 1, (n, d)).astype(np.float32)
    e[100:120] = e[100] + rng.normal(0, 1e-3, (20, d)).astype(np.float32)  # planted duplicates
    u = np.zeros(n, dtype=np.float32)
    t = np.full(n, NOW)
    pruner = UPUGFPruner(uncertainty_weight=0.0, recency_weight=0.0, redundancy_weight=1.0)
    scores = pruner.compute_scores(e, u, t, current_time=NOW)
    assert scores.shape == (n,)
    assert scores[100:120].mean() < np.delete(scores, np.arange(100, 120)).mean()
