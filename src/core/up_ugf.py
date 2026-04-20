"""Uncertainty-Guided Forgetting (UP-UGF) for replay memory pruning."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .similarity import cosine_similarity

__all__ = ["UncertaintyGuidedPruner"]


class UncertaintyGuidedPruner:
    """Score-based pruner combining uncertainty, recency, and redundancy."""

    def __init__(
        self,
        capacity: int = 100,
        uncertainty_threshold: float = 0.8,
        redundancy_threshold: float = 0.95,
        recency_decay: float = 0.99,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = capacity
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.redundancy_threshold = float(redundancy_threshold)
        self.recency_decay = float(recency_decay)

    def compute_embedding_score(
        self,
        embedding: np.ndarray,
        uncertainty_history: List[float],
        last_access_step: int,
        current_step: int,
        support_embeddings: np.ndarray,
    ) -> float:
        """Compute retention score: (1-u) * recency * (1-redundancy)."""
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        avg_u = float(np.mean(uncertainty_history)) if uncertainty_history else 1.0
        avg_u = min(max(avg_u, 0.0), 1.0)

        age = max(current_step - int(last_access_step), 0)
        recency_weight = float(self.recency_decay ** age)

        max_redundancy = 0.0
        if support_embeddings.size > 0:
            sims = cosine_similarity(emb, support_embeddings)
            max_redundancy = float(np.max(sims))
            if max_redundancy > self.redundancy_threshold:
                max_redundancy = self.redundancy_threshold + 0.5 * (
                    max_redundancy - self.redundancy_threshold
                )
            max_redundancy = min(max(max_redundancy, -1.0), 1.0)

        uncertainty_term = 1.0 - avg_u
        return float(max(0.0, uncertainty_term * recency_weight * (1.0 - max_redundancy)))

    def prune(
        self,
        embeddings: List[np.ndarray],
        labels: List[int],
        metadata: List[Dict],
    ) -> Tuple[List[np.ndarray], List[int], List[Dict]]:
        """Prune entries to capacity using descending retention score."""
        if not (len(embeddings) == len(labels) == len(metadata)):
            raise ValueError("embeddings, labels, and metadata must have equal length.")
        if len(embeddings) <= self.capacity:
            return embeddings, labels, metadata

        current_step = max(int(m.get("current_step", 0)) for m in metadata)
        all_embs = np.stack([np.asarray(e, dtype=np.float32) for e in embeddings], axis=0)

        scored = []
        for i, emb in enumerate(embeddings):
            uncertainty_history = list(metadata[i].get("uncertainty_history", []))
            last_access = int(metadata[i].get("last_access_step", 0))
            others = np.delete(all_embs, i, axis=0) if len(all_embs) > 1 else np.empty((0, all_embs.shape[1]), dtype=np.float32)
            score = self.compute_embedding_score(
                embedding=np.asarray(emb, dtype=np.float32),
                uncertainty_history=uncertainty_history,
                last_access_step=last_access,
                current_step=current_step,
                support_embeddings=others,
            )
            scored.append((score, i))

        scored.sort(key=lambda x: x[0], reverse=True)
        keep_indices = sorted(idx for _, idx in scored[: self.capacity])
        kept_embeddings = [embeddings[i] for i in keep_indices]
        kept_labels = [labels[i] for i in keep_indices]
        kept_metadata = [metadata[i] for i in keep_indices]
        return kept_embeddings, kept_labels, kept_metadata
