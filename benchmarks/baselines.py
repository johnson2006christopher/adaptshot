"""The alternatives AdaptShot has to beat to have earned its complexity (#19).

"68% accuracy" answers nothing. "68% against 61% for the obvious cheaper thing"
is an argument, and if the cheaper thing wins that is the finding.

Every method here takes the same embeddings and the same episode, and returns
predicted labels for the same queries. None of them touches the network, and
none introduces a dependency: `pyproject.toml` declares no scikit-learn, so the
linear probe is ~30 lines of numpy rather than a five-line import. That is a
constraint worth honouring rather than routing around -- a library arguing that
connectivity is the scarce resource should not need a 100MB wheel to check a
baseline.
"""

from __future__ import annotations

import math

import numpy as np


def _normalise(matrix: np.ndarray) -> np.ndarray:
    """Unit-norm rows, so cosine similarity is a dot product."""

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-8)


def nearest_centroid(
    support: np.ndarray,
    support_labels: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    """Mean embedding per class, nearest one wins. No calibration, no buffer.

    This is the floor: what remains of AdaptShot if every layer above the
    prototype is removed. If it matches the full pipeline, the pipeline is
    buying something other than accuracy and should say so.
    """

    classes = np.unique(support_labels)
    centroids = _normalise(
        np.stack([support[support_labels == name].mean(axis=0) for name in classes])
    )
    scores = _normalise(query) @ centroids.T
    return classes[np.argmax(scores, axis=1)]


def knn(
    support: np.ndarray,
    support_labels: np.ndarray,
    query: np.ndarray,
    k: int = 1,
) -> np.ndarray:
    """k-NN on raw embeddings by cosine similarity.

    Tests whether the prototype machinery earns its place: averaging a class
    into one point throws information away, and with 5 shots it is not obvious
    that is a good trade.
    """

    similarity = _normalise(query) @ _normalise(support).T
    neighbours = np.argsort(-similarity, axis=1)[:, :k]

    predictions = []
    for row in neighbours:
        names, counts = np.unique(support_labels[row], return_counts=True)
        # Ties break toward the closer neighbour, which is `row[0]`'s label
        # when it is among the tied classes -- otherwise the first by count.
        best = names[counts == counts.max()]
        predictions.append(
            support_labels[row[0]] if support_labels[row[0]] in best else best[0]
        )
    return np.array(predictions, dtype=object)


def linear_probe(
    support: np.ndarray,
    support_labels: np.ndarray,
    query: np.ndarray,
    *,
    epochs: int = 200,
    learning_rate: float = 0.1,
    weight_decay: float = 1e-3,
) -> np.ndarray:
    """Multinomial logistic regression on frozen embeddings, in numpy.

    Full-batch gradient descent on 25 examples converges in well under a
    second and is deterministic without needing a seed -- the weights start at
    zero, so there is nothing random to fix.

    Weight decay is not optional here: with 25 samples in 512 dimensions the
    problem is separable, and unregularised logistic regression will drive the
    weights toward infinity chasing a margin it already has.
    """

    classes = np.unique(support_labels)
    lookup = {name: index for index, name in enumerate(classes)}
    targets = np.zeros((len(support_labels), len(classes)), dtype=np.float64)
    targets[np.arange(len(support_labels)), [lookup[n] for n in support_labels]] = 1.0

    features = _normalise(support).astype(np.float64)
    weights = np.zeros((features.shape[1], len(classes)), dtype=np.float64)
    bias = np.zeros(len(classes), dtype=np.float64)

    for _ in range(epochs):
        logits = features @ weights + bias
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        error = probabilities - targets
        weights -= learning_rate * (
            features.T @ error / len(features) + weight_decay * weights
        )
        bias -= learning_rate * error.mean(axis=0)

    scores = _normalise(query).astype(np.float64) @ weights + bias
    return classes[np.argmax(scores, axis=1)]


def top1_with_threshold(
    support: np.ndarray,
    support_labels: np.ndarray,
    query: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, list[set[str]]]:
    """Top-1, abstaining below a confidence threshold. The comparison #19 cares about.

    Both this and conformal prediction are ways of saying "I am not sure". This
    one is free and has no guarantee; conformal costs prediction-set size and
    claims one. Returning sets -- empty when abstaining, a singleton otherwise --
    makes the two directly comparable on coverage and average set size, which is
    the only fair way to price the guarantee.
    """

    classes = np.unique(support_labels)
    centroids = _normalise(
        np.stack([support[support_labels == name].mean(axis=0) for name in classes])
    )
    similarity = _normalise(query) @ centroids.T

    # Softmax over cosine similarities, so "confidence" means the same thing
    # here as it does for the conformal path.
    logits = similarity - similarity.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    best = np.argmax(probabilities, axis=1)
    predictions = classes[best]
    confidence = probabilities[np.arange(len(best)), best]

    sets = [
        {str(label)} if score >= threshold else set()
        for label, score in zip(predictions, confidence, strict=True)
    ]
    return predictions, sets


# ---------------------------------------------------------------------------
# Set-valued baselines at a matched coverage target (#113)
#
# The one threshold baseline above prices abstention as an *empty* set, which
# counts as a miss -- so its coverage is capped at (1 - alpha) * accuracy and
# the old "the threshold baseline misses the promise conformal kept" sentence
# was an artefact of that accounting. These baselines put honest set-valued
# competitors on the same embeddings, same episodes and same calibration
# split, so the published comparison is a frontier (coverage vs set size),
# not one flattering cell.
#
# All of them consume centroid-softmax probability rows (`class_probabilities`)
# and, where they calibrate, use the standard conformal quantile
# ceil((n+1)(1-alpha))/n on the episode's held-out calibration split.
#
# References, cited in the technical note:
#   LAC   Sadinle, Lei, Wasserman (2019), JASA.
#   APS   Romano, Sesia, Candes (2020), NeurIPS. Deterministic variant (no
#         tie-breaking randomisation), which over-covers slightly; noted so
#         the comparison cannot flatter AdaptShot.
#   RAPS  Angelopoulos, Bates, Jordan, Malik (2021), ICLR. Fixed
#         k_reg = 2, lam = 0.1 -- stated, not tuned.
# ---------------------------------------------------------------------------


def class_probabilities(
    support: np.ndarray,
    support_labels: np.ndarray,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Centroid-softmax probability rows -- the shared input of every baseline.

    Returns (classes, probabilities[len(points), len(classes)]).
    """

    classes = np.unique(support_labels)
    centroids = _normalise(
        np.stack([support[support_labels == name].mean(axis=0) for name in classes])
    )
    similarity = _normalise(points) @ centroids.T
    logits = similarity - similarity.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return classes, probabilities


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """ceil((n+1)(1-alpha))-th smallest score, +inf when n cannot certify."""

    n = len(scores)
    if n == 0:
        return float("inf")
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return float("inf")
    return float(np.sort(scores)[rank - 1])


def top_k_sets(classes: np.ndarray, probabilities: np.ndarray, k: int) -> list[set[str]]:
    """The k most probable classes, unconditionally. k=1 is the singleton top-1."""

    order = np.argsort(-probabilities, axis=1)[:, :k]
    return [{str(classes[j]) for j in row} for row in order]


def threshold_or_full_sets(
    classes: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> list[set[str]]:
    """Top-1 when confident, otherwise the full label set.

    The honest abstention baseline: "I don't know" means every class remains
    possible, not that the question disappears. Empty-set abstention (the
    `top1_with_threshold` accounting) makes abstentions count as misses, which
    caps coverage at the accuracy and voids the comparison (#113).
    """

    best = np.argmax(probabilities, axis=1)
    confidence = probabilities[np.arange(len(best)), best]
    full = {str(name) for name in classes}
    return [
        {str(classes[b])} if c >= threshold else set(full)
        for b, c in zip(best, confidence, strict=True)
    ]


def lac_sets(
    classes: np.ndarray,
    calibration_probabilities: np.ndarray,
    calibration_labels: np.ndarray,
    query_probabilities: np.ndarray,
    alpha: float,
) -> list[set[str]]:
    """LAC (Sadinle et al. 2019): score = 1 - p_true; include y iff 1 - p_y <= q."""

    lookup = {name: index for index, name in enumerate(classes)}
    true_idx = np.array([lookup[label] for label in calibration_labels])
    scores = 1.0 - calibration_probabilities[np.arange(len(true_idx)), true_idx]
    q_hat = conformal_quantile(scores, alpha)
    return [
        {str(classes[j]) for j in range(len(classes)) if 1.0 - row[j] <= q_hat}
        for row in query_probabilities
    ]


def _aps_score_rows(probabilities: np.ndarray, lam: float = 0.0, k_reg: int = 0) -> np.ndarray:
    """Cumulative-mass score of every class per row, RAPS-regularised when lam > 0.

    score(y) = (mass of classes ranked above y) + p_y + lam * max(0, rank(y) - k_reg),
    the deterministic APS variant (the randomised tie-break is dropped, which
    can only widen sets -- the conservative direction for a baseline we are
    comparing against).
    """

    order = np.argsort(-probabilities, axis=1)
    sorted_probs = np.take_along_axis(probabilities, order, axis=1)
    cumulative = np.cumsum(sorted_probs, axis=1)
    ranks = np.arange(1, probabilities.shape[1] + 1)
    penalties = lam * np.maximum(0, ranks - k_reg)
    sorted_scores = cumulative + penalties
    scores = np.empty_like(probabilities)
    np.put_along_axis(scores, order, sorted_scores, axis=1)
    return scores


def aps_sets(
    classes: np.ndarray,
    calibration_probabilities: np.ndarray,
    calibration_labels: np.ndarray,
    query_probabilities: np.ndarray,
    alpha: float,
    lam: float = 0.0,
    k_reg: int = 0,
) -> list[set[str]]:
    """APS (lam=0) / RAPS (lam>0) prediction sets on the shared probabilities."""

    lookup = {name: index for index, name in enumerate(classes)}
    calibration_scores = _aps_score_rows(calibration_probabilities, lam, k_reg)
    true_idx = np.array([lookup[label] for label in calibration_labels])
    scores = calibration_scores[np.arange(len(true_idx)), true_idx]
    q_hat = conformal_quantile(scores, alpha)

    query_scores = _aps_score_rows(query_probabilities, lam, k_reg)
    return [
        {str(classes[j]) for j in range(len(classes)) if row[j] <= q_hat}
        for row in query_scores
    ]
