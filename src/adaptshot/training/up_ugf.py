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

import numpy as np

from ..utils.arrays import FloatArray, LabelArray

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
        embeddings: FloatArray,
        uncertainties: FloatArray,
        last_access_times: FloatArray,
        current_time: float | None = None,
    ) -> FloatArray:
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
            return np.asarray([])

        # 1. Uncertainty component: prefer informative examples near boundary
        # Clamp to [0, 1] and invert (lower uncertainty → higher score)
        u_norm = np.clip(uncertainties, 0.0, 1.0)
        unc_score = np.power(1.0 - u_norm, self.w_unc)

        # 2. Recency component: exponential decay since last access
        dt = np.clip(current_time - last_access_times, 0.0, None)
        rec_score = np.exp(-self.decay * dt)
        rec_score = np.power(rec_score, self.w_rec)

        # 3. Redundancy component: 1 - max cosine sim to other embeddings.
        # v0.2.0: Replaced O(N^2) full similarity matrix with approximate method
        # using random projection LSH for large buffers (>100 examples).
        # For small buffers (<=100), falls back to exact computation.
        if N <= 100:
            # Exact: O(N^2) but N is small so it's fast (~1ms for N=100)
            embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            sim_matrix = embeddings_norm @ embeddings_norm.T
            np.fill_diagonal(sim_matrix, -1.0)
            max_sim = np.max(sim_matrix, axis=1)
        else:
            # Approximate: random projection LSH, O(N * D * log N)
            D = embeddings.shape[1]
            n_hashes = min(D, 64)  # Number of random projections
            # Random projection matrix
            rng = np.random.default_rng(42)
            proj = rng.normal(0, 1.0, (D, n_hashes)).astype(np.float32)
            proj = proj / (np.linalg.norm(proj, axis=0, keepdims=True) + 1e-8)
            # Project embeddings to hash bits
            hashes = (embeddings @ proj) > 0  # [N, n_hashes] boolean
            # For each embedding, find max collision count as proxy for similarity
            hash_int = hashes.astype(np.int32) @ (1 << np.arange(n_hashes, dtype=np.int32))
            # Count collisions via sorting
            max_collisions = np.ones(N, dtype=np.float32)
            sort_idx = np.argsort(hash_int)
            run_start = 0
            for j in range(1, N + 1):
                if j == N or hash_int[sort_idx[j]] != hash_int[sort_idx[run_start]]:
                    run_len = j - run_start
                    if run_len > 1:
                        max_collisions[sort_idx[run_start:j]] = float(run_len)
                    run_start = j
            # Convert collisions to redundancy: 1 - (collisions / max_possible)
            max_sim = 1.0 - np.clip(max_collisions / max(2, n_hashes), 0.0, 1.0)
        red_score = np.power(np.clip(1.0 - max_sim, 0.0, 1.0), self.w_red)

        # Composite score (multiplicative)
        scores = unc_score * rec_score * red_score
        return np.asarray(scores)

    def prune(
        self,
        embeddings: FloatArray,
        labels: LabelArray,
        uncertainties: FloatArray,
        last_access_times: FloatArray,
    ) -> tuple[FloatArray, LabelArray, FloatArray, FloatArray]:
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
            np.asarray(embeddings[keep_idx]),
            np.asarray(labels[keep_idx]),
            np.asarray(uncertainties[keep_idx]),
            np.asarray(last_access_times[keep_idx]),
        )