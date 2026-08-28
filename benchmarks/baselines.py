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
