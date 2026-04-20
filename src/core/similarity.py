"""FAISS + NumPy hybrid similarity search with CPU-first defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional acceleration dependency
    faiss = None

__all__ = ["HybridSimilarityIndex", "cosine_similarity"]


def cosine_similarity(query: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between `query` and each support vector."""
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    s = np.asarray(support, dtype=np.float32)
    if s.ndim != 2:
        raise ValueError("support must have shape [N,D].")
    if s.shape[1] != q.shape[0]:
        raise ValueError("feature dimension mismatch between query and support.")

    q_norm = np.linalg.norm(q) + 1e-12
    s_norm = np.linalg.norm(s, axis=1, keepdims=True) + 1e-12
    return (s / s_norm) @ (q / q_norm)


@dataclass
class HybridSimilarityIndex:
    """Hybrid index that uses FAISS when available and NumPy fallback otherwise."""

    dim: int
    use_faiss: bool = True

    def __post_init__(self) -> None:
        self._embeddings = np.empty((0, self.dim), dtype=np.float32)
        self._faiss_index: Optional["faiss.IndexFlatIP"] = None
        if self.use_faiss and faiss is not None:
            self._faiss_index = faiss.IndexFlatIP(self.dim)

    def add(self, embeddings: np.ndarray) -> None:
        """Add embeddings with shape [N,D] to the index."""
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"embeddings must be [N,{self.dim}].")
        self._embeddings = np.concatenate([self._embeddings, x], axis=0)
        if self._faiss_index is not None:
            normalized = self._normalize(x)
            self._faiss_index.add(normalized)

    def search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Search top-k nearest neighbors using cosine similarity."""
        if len(self) == 0:
            raise ValueError("index is empty.")
        if k <= 0:
            raise ValueError("k must be positive.")
        k = min(k, len(self))

        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != self.dim:
            raise ValueError(f"query must have dimension {self.dim}.")

        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(self._normalize(q), k)
            return scores[0], indices[0]

        sims = cosine_similarity(q[0], self._embeddings)
        topk_idx = np.argsort(-sims)[:k]
        return sims[topk_idx], topk_idx.astype(np.int64)

    def __len__(self) -> int:
        """Return number of indexed vectors."""
        return int(self._embeddings.shape[0])

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        return vectors / norms
