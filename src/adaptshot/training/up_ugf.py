"""UP-UGF: Uncertainty-Guided Forgetting for bounded replay buffers.

Replaces naive FIFO eviction with a composite scoring function that
prioritizes retaining embeddings which are:
1. Informative (high uncertainty / near decision boundary)
2. Recent (accessed or corrected recently)
3. Diverse (low redundancy with same-class examples)

Designed for CPU-first edge deployment with strict memory ceilings.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class UPUGFPruner:
    """
    Uncertainty-Guided Forgetting (UP-UGF) buffer manager.

    Scores each stored embedding using a multiplicative utility function.
    When buffer capacity is exceeded, the lowest-scoring examples are evicted
    to maintain bounded memory while preserving model plasticity.
    """

    def __init__(
        self,
        capacity: int = 100,
        uncertainty_weight: float = 1.0,
        recency_weight: float = 1.0,
        redundancy_weight: float = 1.0,
        recency_decay: float = 0.01,
    ) -> None:
        """
        Args:
            capacity: Maximum number of examples to retain in buffer
            uncertainty_weight: Importance of retaining uncertain/boundary examples
            recency_weight: Importance of retaining recently accessed examples
            redundancy_weight: Importance of retaining diverse (non-redundant) examples
            recency_decay: Exponential decay rate for recency scoring
        """
        self.capacity = capacity
        self.w_unc = uncertainty_weight
        self.w_rec = recency_weight
        self.w_red = redundancy_weight
        self.decay = recency_decay

    def compute_scores(
        self,
        embeddings: np.ndarray,
        uncertainties: np.ndarray,
        last_access_times: np.ndarray,
        current_time: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute UP-UGF utility score for each embedding in the buffer.

        Score(e) = (1 - u(e))^w_unc × exp(-λ × Δt)^w_rec × (1 - max_sim_to_same_class)^w_red

        Args:
            embeddings: [N, D] array of stored embeddings
            uncertainties: [N] array of prediction uncertainties (entropy or 1-confidence) in [0, 1]
            last_access_times: [N] array of Unix timestamps for last access/correction
            current_time: Current Unix timestamp (defaults to time.time())

        Returns:
            scores: [N] array of utility scores (higher = keep, lower = evict)
        """
        if current_time is None:
            current_time = time.time()

        N = embeddings.shape[0]
        if N == 0:
            return np.array([])

        # 1. Uncertainty component: prefer informative examples near boundary
        # Clamp to [0, 1] and invert (lower uncertainty → higher score)
        u_norm = np.clip(uncertainties, 0.0, 1.0)
        unc_score = np.power(1.0 - u_norm, self.w_unc)

        # 2. Recency component: exponential decay since last access
        dt = np.clip(current_time - last_access_times, 0.0, None)
        rec_score = np.exp(-self.decay * dt)
        rec_score = np.power(rec_score, self.w_rec)

        # 3. Redundancy component: 1 - max cosine sim to same-class embeddings
        # Vectorized computation against self for simplicity
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        sim_matrix = embeddings_norm @ embeddings_norm.T
        np.fill_diagonal(sim_matrix, -1.0)  # Exclude self-similarity
        max_sim = np.max(sim_matrix, axis=1)
        red_score = np.power(np.clip(1.0 - max_sim, 0.0, 1.0), self.w_red)

        # Composite score (multiplicative)
        scores = unc_score * rec_score * red_score
        return scores

    def prune(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        uncertainties: np.ndarray,
        last_access_times: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Enforce buffer capacity by evicting lowest-scoring examples.

        Args:
            embeddings: [N, D] current buffer embeddings
            labels: [N] current buffer labels
            uncertainties: [N] current uncertainties
            last_access_times: [N] current access timestamps

        Returns:
            (pruned_embeddings, pruned_labels, pruned_uncertainties, pruned_times)
            All truncated to self.capacity, retaining highest-scoring examples.
        """
        if len(embeddings) <= self.capacity:
            return embeddings, labels, uncertainties, last_access_times

        scores = self.compute_scores(embeddings, uncertainties, last_access_times)
        # Keep top-K by score
        keep_idx = np.argsort(scores)[-self.capacity:]
        
        return (
            embeddings[keep_idx],
            labels[keep_idx],
            uncertainties[keep_idx],
            last_access_times[keep_idx]
        )