"""CPU-optimized cosine similarity search with optional FAISS acceleration."""

from typing import Tuple, cast

import numpy as np

# Attempt to import FAISS-CPU; gracefully degrade to pure NumPy if unavailable
try:
    import faiss  # type: ignore[import-untyped]
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


def cosine_similarity_numpy(query: np.ndarray, support: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query and support embeddings using NumPy.

    Uses the mathematical identity: cos(a,b) = (a·b) / (||a|| ||b||).
    Fully vectorized for CPU efficiency. Handles 1D (single query) and 2D (batch) inputs.

    Args:
        query: [D] or [B, D] array of query embeddings
        support: [N, D] array of support embeddings

    Returns:
        similarities: [B, N] array of cosine similarity scores in [-1, 1]
    """
    if query.ndim == 1:
        query = query[np.newaxis, :]  # [1, D]

    # L2 normalize with epsilon to prevent division by zero
    query_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
    support_norm = support / (np.linalg.norm(support, axis=1, keepdims=True) + 1e-8)

    # Matrix multiplication of normalized vectors = cosine similarity
    return cast(np.ndarray, query_norm @ support_norm.T)


def cosine_similarity_faiss(
    query: np.ndarray,
    support: np.ndarray,
    k: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute top-k cosine similarity using FAISS-CPU IndexFlatIP.

    FAISS Inner Product (IP) index is mathematically equivalent to cosine similarity
    when vectors are L2-normalized, but significantly faster for large N.

    Args:
        query: [B, D] array of query embeddings
        support: [N, D] array of support embeddings
        k: Number of nearest neighbors to return (default: 1)

    Returns:
        similarities: [B, k] array of top-k cosine scores
        indices: [B, k] array of indices into the support set
    """
    if not FAISS_AVAILABLE:
        raise ImportError(
            "FAISS-CPU is not installed. Install via: pip install faiss-cpu, "
            "or set use_faiss=False to fall back to NumPy."
        )

    if query.ndim == 1:
        query = query[np.newaxis, :]

    # FAISS requires float32, C-contiguous memory layout
    query = np.ascontiguousarray(query, dtype=np.float32)
    support = np.ascontiguousarray(support, dtype=np.float32)

    # In-place L2 normalization
    faiss.normalize_L2(query)
    faiss.normalize_L2(support)

    D = support.shape[1]
    index = faiss.IndexFlatIP(D)  # Inner Product for normalized vectors
    index.add(support)

    return cast(Tuple[np.ndarray, np.ndarray], index.search(query, min(k, support.shape[0])))


def find_nearest_neighbor(
    query: np.ndarray,
    support_embeddings: np.ndarray,
    support_labels: np.ndarray,
    use_faiss: bool = False,
) -> Tuple[str, float, int]:
    """
    Find the single nearest neighbor in the support set and return prediction metadata.

    Args:
        query: [D] query embedding
        support_embeddings: [N, D] array of stored support embeddings
        support_labels: [N] array of corresponding class labels
        use_faiss: Toggle FAISS acceleration (requires faiss-cpu)

    Returns:
        predicted_label: Class label of the nearest support example
        confidence: Cosine similarity score (unnormalized raw confidence)
        neighbor_idx: Integer index into the support_embeddings array
    """
    if use_faiss and FAISS_AVAILABLE:
        similarities, indices = cosine_similarity_faiss(
            query[np.newaxis, :], support_embeddings, k=1
        )
        confidence = float(similarities[0, 0])
        neighbor_idx = int(indices[0, 0])
    else:
        similarities = cosine_similarity_numpy(query, support_embeddings)
        # similarities shape: [1, N] for single query
        if similarities.ndim == 2 and similarities.shape[0] == 1:
            similarities = similarities[0]  # Squeeze to [N]
        neighbor_idx = int(np.argmax(similarities))
        confidence = float(similarities[neighbor_idx])

    return str(support_labels[neighbor_idx]), confidence, neighbor_idx