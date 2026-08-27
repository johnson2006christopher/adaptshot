"""CPU-optimized similarity and prototype utilities for few-shot inference."""

from __future__ import annotations

from typing import cast

import numpy as np

# Attempt to import FAISS-CPU; gracefully degrade to pure NumPy if unavailable.
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    FAISS_AVAILABLE = False


def _ensure_2d(query: np.ndarray) -> np.ndarray:
    if query.ndim == 1:
        return query[np.newaxis, :]
    if query.ndim != 2:
        raise ValueError(f"Expected query to be 1D or 2D. Got shape={query.shape}.")
    return query


def _l2_normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True) + 1e-8
    result: np.ndarray = values / norms
    return result


def cosine_similarity_numpy(query: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and support embeddings."""

    query_2d = _ensure_2d(np.asarray(query, dtype=np.float32))
    support_2d = np.asarray(support, dtype=np.float32)
    if support_2d.ndim != 2:
        raise ValueError(f"Expected support to be 2D [N, D]. Got shape={support_2d.shape}.")
    if support_2d.shape[0] == 0:
        raise ValueError("Support embeddings cannot be empty.")

    query_norm = _l2_normalize_rows(query_2d)
    support_norm = _l2_normalize_rows(support_2d)
    return cast(np.ndarray, query_norm @ support_norm.T)


def euclidean_distance_numpy(
    query: np.ndarray,
    support: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Compute Euclidean distances between query and support embeddings."""

    query_2d = _ensure_2d(np.asarray(query, dtype=np.float32))
    support_2d = np.asarray(support, dtype=np.float32)
    if support_2d.ndim != 2:
        raise ValueError(f"Expected support to be 2D [N, D]. Got shape={support_2d.shape}.")
    if support_2d.shape[0] == 0:
        raise ValueError("Support embeddings cannot be empty.")

    query_eval = _l2_normalize_rows(query_2d) if normalize else query_2d
    support_eval = _l2_normalize_rows(support_2d) if normalize else support_2d

    diffs = query_eval[:, np.newaxis, :] - support_eval[np.newaxis, :, :]
    return cast(np.ndarray, np.linalg.norm(diffs, axis=2))


def _euclidean_top1_faiss(query: np.ndarray, support: np.ndarray) -> tuple[float, int]:
    if not FAISS_AVAILABLE:
        raise ImportError(
            "FAISS-CPU is not installed. Install via `pip install faiss-cpu`, "
            "or set use_faiss=False."
        )

    query_2d = _ensure_2d(np.asarray(query, dtype=np.float32))
    support_2d = np.asarray(support, dtype=np.float32)

    query_norm = np.ascontiguousarray(_l2_normalize_rows(query_2d), dtype=np.float32)
    support_norm = np.ascontiguousarray(_l2_normalize_rows(support_2d), dtype=np.float32)

    index = faiss.IndexFlatL2(support_norm.shape[1])
    index.add(support_norm)
    distances_sq, indices = index.search(query_norm, 1)
    distance = float(np.sqrt(max(float(distances_sq[0, 0]), 0.0)))
    neighbor_idx = int(indices[0, 0])
    return distance, neighbor_idx


def distance_to_confidence(distance: float) -> float:
    """Convert a non-negative distance to a confidence score in [0, 1]."""

    clipped_distance = max(0.0, float(distance))
    return float(1.0 / (1.0 + clipped_distance))


def compute_class_prototypes(
    support_embeddings: np.ndarray,
    support_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute one prototype vector (mean embedding) per class label."""

    support = np.asarray(support_embeddings, dtype=np.float32)
    labels = np.asarray(support_labels, dtype=object)
    if support.ndim != 2:
        raise ValueError(f"Expected support_embeddings shape [N, D], got {support.shape}.")
    if labels.ndim != 1:
        raise ValueError(f"Expected support_labels shape [N], got {labels.shape}.")
    if support.shape[0] != labels.shape[0]:
        raise ValueError(
            "support_embeddings and support_labels length mismatch: "
            f"{support.shape[0]} vs {labels.shape[0]}."
        )
    if support.shape[0] == 0:
        raise ValueError("Cannot compute prototypes from an empty support set.")

    ordered_labels: list[object] = []
    index_by_label: dict[object, int] = {}
    sums: list[np.ndarray] = []
    counts: list[int] = []

    for idx, raw_label in enumerate(labels):
        label_key = raw_label.item() if hasattr(raw_label, "item") else raw_label
        if label_key not in index_by_label:
            index_by_label[label_key] = len(ordered_labels)
            ordered_labels.append(label_key)
            sums.append(np.array(support[idx], dtype=np.float32))
            counts.append(1)
        else:
            class_idx = index_by_label[label_key]
            sums[class_idx] = sums[class_idx] + np.asarray(support[idx], dtype=np.float32)
            counts[class_idx] += 1

    prototypes = np.stack(
        [sums[i] / max(counts[i], 1) for i in range(len(ordered_labels))], axis=0
    ).astype(np.float32, copy=False)
    return (
        prototypes,
        np.asarray(ordered_labels, dtype=object),
        np.asarray(counts, dtype=np.int64),
    )


def find_nearest_neighbor(
    query: np.ndarray,
    support_embeddings: np.ndarray,
    support_labels: np.ndarray,
    use_faiss: bool = False,
    metric: str = "cosine",
) -> tuple[str, float, int]:
    """Find top-1 nearest support sample using cosine or Euclidean metric."""

    support = np.asarray(support_embeddings, dtype=np.float32)
    labels = np.asarray(support_labels, dtype=object)
    if support.shape[0] == 0:
        raise ValueError("Support embeddings cannot be empty.")

    if metric not in {"cosine", "euclidean"}:
        raise ValueError(f"Unsupported metric '{metric}'. Use 'cosine' or 'euclidean'.")

    if metric == "euclidean":
        if use_faiss and FAISS_AVAILABLE:
            distance, neighbor_idx = _euclidean_top1_faiss(query, support)
        else:
            distances = euclidean_distance_numpy(query, support, normalize=True)
            distance = float(distances.reshape(-1)[int(np.argmin(distances))])
            neighbor_idx = int(np.argmin(distances))
        confidence = distance_to_confidence(distance)
        return str(labels[neighbor_idx]), confidence, neighbor_idx

    if use_faiss and FAISS_AVAILABLE:
        query_2d = _ensure_2d(np.asarray(query, dtype=np.float32))
        support_2d = np.asarray(support, dtype=np.float32)
        query_norm = np.ascontiguousarray(_l2_normalize_rows(query_2d), dtype=np.float32)
        support_norm = np.ascontiguousarray(_l2_normalize_rows(support_2d), dtype=np.float32)

        index = faiss.IndexFlatIP(support_norm.shape[1])
        index.add(support_norm)
        similarities, indices = index.search(query_norm, 1)
        confidence = float(similarities[0, 0])
        neighbor_idx = int(indices[0, 0])
    else:
        similarities = cosine_similarity_numpy(query, support)
        flat_scores = similarities.reshape(-1)
        neighbor_idx = int(np.argmax(flat_scores))
        confidence = float(flat_scores[neighbor_idx])

    return str(labels[neighbor_idx]), confidence, neighbor_idx


def find_nearest_prototype(
    query: np.ndarray,
    prototypes: np.ndarray,
    prototype_labels: np.ndarray,
    metric: str = "euclidean",
) -> tuple[object, float, int, float, float]:
    """Find top-1 nearest class prototype and return prediction diagnostics.

    Returns:
        predicted_label: Predicted class label
        confidence: Similarity-like confidence in [0, 1] for euclidean, [-1, 1] for cosine
        prototype_idx: Index into `prototypes`
        nearest_distance: Distance to top-1 prototype in normalized feature space
        top2_margin: Gap between best and second-best score (higher is more certain)
    """

    proto = np.asarray(prototypes, dtype=np.float32)
    proto_labels = np.asarray(prototype_labels, dtype=object)
    if proto.shape[0] == 0:
        raise ValueError("Prototype set cannot be empty.")
    if metric not in {"cosine", "euclidean"}:
        raise ValueError(f"Unsupported metric '{metric}'. Use 'cosine' or 'euclidean'.")

    if metric == "euclidean":
        distances = euclidean_distance_numpy(query, proto, normalize=True).reshape(-1)
        best_idx = int(np.argmin(distances))
        nearest_distance = float(distances[best_idx])
        confidence = distance_to_confidence(nearest_distance)
        if distances.size > 1:
            sorted_distances = np.sort(distances)
            top2_margin = float(sorted_distances[1] - sorted_distances[0])
        else:
            top2_margin = float("inf")
    else:
        scores = cosine_similarity_numpy(query, proto).reshape(-1)
        best_idx = int(np.argmax(scores))
        confidence = float(scores[best_idx])
        nearest_distance = float(1.0 - np.clip(confidence, -1.0, 1.0))
        if scores.size > 1:
            top_scores = np.sort(scores)[-2:]
            top2_margin = float(top_scores[-1] - top_scores[-2])
        else:
            top2_margin = float("inf")

    return (
        proto_labels[best_idx],
        confidence,
        best_idx,
        nearest_distance,
        top2_margin,
    )
