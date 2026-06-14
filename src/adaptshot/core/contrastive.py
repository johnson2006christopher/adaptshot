"""Contrastive prototype learning for improved few-shot class representations.

Replaces naive mean prototypes with contrastively refined class centers.
Uses an InfoNCE-style loss that pulls same-class embeddings together while
pushing different-class embeddings apart. A lightweight projection head
(2-layer MLP) maps backbone embeddings to a contrastive space, and prototype
positions are updated via exponential moving average (EMA) for stability.

Design: numpy-first (no torch dependency). When torch is available, the
projection head can be fine-tuned; otherwise, prototypes are refined using
pure numpy operations with configurable learning rate and momentum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, cast

import numpy as np


@dataclass
class ContrastiveConfig:
    """Configuration for contrastive prototype learning.

    Attributes:
        projection_dim: Output dimension of the projection head bottleneck.
        temperature: Temperature for soft nearest-neighbor in InfoNCE loss.
            Lower values produce sharper distinctions.
        learning_rate: Step size for prototype refinement.
        momentum: EMA decay rate for prototype updates (0.9 = slow, 0.99 = very slow).
        n_epochs: Number of refinement iterations during support set loading.
        hard_negative_weight: Multiplier for hard negative mining emphasis.
    """

    projection_dim: int = 128
    temperature: float = 0.07
    learning_rate: float = 0.01
    momentum: float = 0.9
    n_epochs: int = 50
    hard_negative_weight: float = 1.5


class ContrastivePrototypeLearner:
    """Contrastively refine class prototypes from support embeddings.

    Algorithm:
        1. Initialize a 2-layer MLP projection head (He init)
        2. Train the projection head via gradient descent on InfoNCE loss,
           using mini-batch SGD with momentum to separate classes in the
           projected space
        3. After the head converges, project all embeddings and compute
           initial class-mean prototypes
        4. Refine prototype positions via gradient-informed momentum steps
           (cross-entropy of distances to prototypes)
        5. Store refined prototypes for nearest-prototype inference

    The projection head W_proj is a learnable [D, projection_dim] matrix
    that maps backbone embeddings to a compact space where class separation
    is maximized by the contrastive objective. Unlike v0.2.0-dev where the
    head was random, v0.2.0 actually trains it via InfoNCE gradient descent.
    """

    def __init__(self, config: Optional[ContrastiveConfig] = None) -> None:
        """Initialize the contrastive prototype learner.

        Args:
            config: ContrastiveConfig or defaults.
        """
        self.config = config or ContrastiveConfig()
        self._projection_matrix: Optional[np.ndarray] = None
        self._projection_bias: Optional[np.ndarray] = None
        self._second_layer_matrix: Optional[np.ndarray] = None
        self._second_layer_bias: Optional[np.ndarray] = None
        self._input_dim: Optional[int] = None
        self._is_fitted = False
        self._head_training_loss: List[float] = []  # Track head training convergence

    # ------------------------------------------------------------------
    # Projection head
    # ------------------------------------------------------------------

    def _init_projection_head(self, input_dim: int, seed: int = 42) -> None:
        """Initialize the 2-layer MLP projection head.

        Architecture: D -> projection_dim (ReLU) -> projection_dim
        Uses He initialization for ReLU activation.

        Args:
            input_dim: Dimensionality of backbone embeddings.
            seed: Random seed for reproducibility.
        """
        rng = np.random.default_rng(seed)
        d = self.config.projection_dim

        # He initialization: scale by sqrt(2 / fan_in)
        self._input_dim = input_dim
        self._projection_matrix = rng.normal(0, np.sqrt(2.0 / input_dim), (input_dim, d))
        self._projection_bias = np.zeros(d, dtype=np.float32)
        self._second_layer_matrix = rng.normal(0, np.sqrt(2.0 / d), (d, d))
        self._second_layer_bias = np.zeros(d, dtype=np.float32)

    def _project(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply the 2-layer MLP projection.

        Args:
            embeddings: [N, D] backbone embeddings.

        Returns:
            [N, projection_dim] projected embeddings.
        """
        if self._projection_matrix is None:
            raise RuntimeError(
                "Projection head not initialized. Call refine_prototypes() first."
            )
        # Layer 1: linear + ReLU
        x = embeddings @ self._projection_matrix + self._projection_bias
        x = np.maximum(0.0, x)  # ReLU
        # Layer 2: linear (no activation — output is the contrastive space)
        x = x @ self._second_layer_matrix + self._second_layer_bias
        # L2 normalize output
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        return cast(np.ndarray, x / norms)

    # ------------------------------------------------------------------
    # InfoNCE loss
    # ------------------------------------------------------------------

    def _compute_infonce_loss(
        self,
        projected: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """Compute InfoNCE loss and per-sample gradients.

        L = -1/N sum_i log( exp(sim(z_i, z_pos) / tau) / sum_j exp(sim(z_i, z_j) / tau) )

        Where sim(a, b) = dot(a, b) since vectors are L2-normalized.

        Args:
            projected: [N, d] L2-normalized projected embeddings.
            labels: [N] integer class labels.

        Returns:
            (loss_value, gradients_wrt_projected [N, d])
        """
        tau = max(self.config.temperature, 1e-4)
        n = projected.shape[0]

        # Compute pairwise cosine similarity matrix
        sim = projected @ projected.T  # [N, N]

        # Temperature-scaled similarities
        sim_scaled = sim / tau

        # Numerical stability: subtract max per row
        sim_scaled = sim_scaled - np.max(sim_scaled, axis=1, keepdims=True)

        # Exponentiate
        exp_sim = np.exp(sim_scaled)

        # Create positive mask: same label, exclude self
        label_match = labels[:, None] == labels[None, :]  # [N, N]
        np.fill_diagonal(label_match, False)
        pos_mask = label_match.astype(np.float32)

        # Numerator: sum of exp(sim) for positives
        pos_sum = np.sum(exp_sim * pos_mask, axis=1) + 1e-8

        # Denominator: sum of all exp(sim) except self
        not_self = 1.0 - np.eye(n, dtype=np.float32)
        neg_sum = np.sum(exp_sim * not_self, axis=1) + 1e-8

        # Per-sample loss
        per_sample_loss = -np.log(pos_sum / neg_sum)
        loss = float(np.mean(per_sample_loss))

        # Gradient of loss w.r.t. projected embeddings
        # dL/dz_i = (1/tau) * [ sum_j P_ij * z_j - sum_k_pos P_ik * z_k / sum_k_pos P_ik ]
        # where P_ij = exp(sim(z_i, z_j) / tau) / sum_k exp(sim(z_i, z_k) / tau)
        prob = exp_sim / (neg_sum[:, None] + 1e-8)  # [N, N] softmax probabilities

        # Hard negative weighting: boost gradient for close negatives
        if self.config.hard_negative_weight > 1.0:
            neg_mask = 1.0 - pos_mask
            np.fill_diagonal(neg_mask, 0.0)
            # Identify hard negatives: same-class far apart, different-class close
            hard_weight = np.ones_like(prob)
            hard_weight += (
                (self.config.hard_negative_weight - 1.0) * neg_mask * prob
            )
            prob = prob * hard_weight
            # Re-normalize
            prob = prob / (prob.sum(axis=1, keepdims=True) + 1e-8)

        # Gradient: dL/dz_i = (z_i * sum_pos - sum_pos_weighted) / tau
        pos_weighted = prob * pos_mask
        pos_weight = pos_weighted / (pos_weighted.sum(axis=1, keepdims=True) + 1e-8)
        grad = (projected - pos_weight @ projected) / tau

        # Scale by 1/n
        grad = grad / n

        return loss, grad

    # ------------------------------------------------------------------
    # Projection head training (InfoNCE gradient descent)
    # ------------------------------------------------------------------

    def _train_projection_head(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        label_indices: np.ndarray,
        seed: int = 42,
    ) -> List[float]:
        """Train the 2-layer MLP projection head via InfoNCE gradient descent.

        This is the key fix for v0.2.0: previously the projection head was
        random and never trained. Now we backpropagate the InfoNCE gradient
        through the projection matrices using mini-batch SGD with momentum.

        After training, the projection space should maximize inter-class
        separation and intra-class compactness.

        Args:
            embeddings: [N, D] support embeddings.
            labels: [N] class labels.
            label_indices: [N] integer label indices.
            seed: Random seed for reproducibility.

        Returns:
            Loss history per epoch.
        """
        input_dim = embeddings.shape[1]
        d = self.config.projection_dim
        tau = max(self.config.temperature, 1e-4)
        lr = self.config.learning_rate * 0.5  # Lower LR for head training
        momentum = 0.9
        n_epochs = max(self.config.n_epochs, 30)

        # Ensure head is initialized
        if self._projection_matrix is None or self._input_dim != input_dim:
            self._init_projection_head(input_dim, seed)

        # Momentum accumulators for projection matrices
        w1_vel = np.zeros_like(self._projection_matrix)  # type: ignore[arg-type]
        b1_vel = np.zeros_like(self._projection_bias)  # type: ignore[arg-type]
        w2_vel = np.zeros_like(self._second_layer_matrix)  # type: ignore[arg-type]
        b2_vel = np.zeros_like(self._second_layer_bias)  # type: ignore[arg-type]

        loss_history: List[float] = []
        n = len(embeddings)

        for epoch in range(n_epochs):
            # Forward: project all embeddings through current head
            x = embeddings @ self._projection_matrix + self._projection_bias  # [N, d]
            x_relu = np.maximum(0.0, x)  # ReLU
            projected = x_relu @ self._second_layer_matrix + self._second_layer_bias  # [N, d]
            # L2 normalize
            proj_norms = np.linalg.norm(projected, axis=1, keepdims=True) + 1e-8
            projected = projected / proj_norms

            # Compute InfoNCE loss and gradient w.r.t. projected embeddings
            loss, grad_projected = self._compute_infonce_loss(
                projected, np.array(label_indices, dtype=np.int64)
            )
            loss_history.append(loss)

            # ---- Backprop through projection head ----
            # dL/d(second_layer_output) = grad_projected  [N, d]
            # Layer 2: output = relu(x) @ W2 + b2
            #   dL/dW2 = relu(x).T @ grad_projected  [d, d]
            #   dL/db2 = sum(grad_projected, axis=0)  [d]
            #   dL/d(relu(x)) = grad_projected @ W2.T  [N, d]
            grad_l2 = grad_projected @ self._second_layer_matrix.T  # type: ignore[union-attr]

            # ReLU backward: grad on pre-activation
            grad_relu = grad_l2 * (x > 0).astype(np.float32)  # [N, d]

            # Layer 1: x = embeddings @ W1 + b1
            #   dL/dW1 = embeddings.T @ grad_relu  [D, d]
            #   dL/db1 = sum(grad_relu, axis=0)  [d]
            grad_w2 = x_relu.T @ grad_projected / n
            grad_b2 = grad_projected.sum(axis=0) / n
            grad_w1 = embeddings.T @ grad_relu / n
            grad_b1 = grad_relu.sum(axis=0) / n

            # Momentum SGD updates
            w2_vel = momentum * w2_vel - lr * grad_w2
            b2_vel = momentum * b2_vel - lr * grad_b2
            w1_vel = momentum * w1_vel - lr * grad_w1
            b1_vel = momentum * b1_vel - lr * grad_b1

            self._second_layer_matrix = self._second_layer_matrix + w2_vel  # type: ignore[operator]
            self._second_layer_bias = self._second_layer_bias + b2_vel  # type: ignore[operator]
            self._projection_matrix = self._projection_matrix + w1_vel  # type: ignore[operator]
            self._projection_bias = self._projection_bias + b1_vel  # type: ignore[operator]

            # Early stopping
            if epoch > 15 and len(loss_history) >= 5:
                recent = loss_history[-5:]
                if max(recent) - min(recent) < 1e-4:
                    break

        self._head_training_loss = loss_history
        return loss_history

    # ------------------------------------------------------------------
    # Prototype refinement (after head is trained)
    # ------------------------------------------------------------------

    def refine_prototypes(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        existing_prototypes: Optional[np.ndarray] = None,
        existing_prototype_labels: Optional[np.ndarray] = None,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Train projection head then contrastively refine class prototypes.

        v0.2.0: The projection head is now TRAINED via InfoNCE gradient
        descent before prototype refinement. Previously (v0.2.0-dev) the
        head was random and never updated, making the projection space
        essentially meaningless.

        Args:
            embeddings: [N, D] backbone embeddings.
            labels: [N] class labels (string or int).
            existing_prototypes: [K, D] previous prototypes for warm start.
            existing_prototype_labels: [K] previous prototype labels.
            seed: Random seed for projection head initialization.

        Returns:
            (refined_prototypes [K, projection_dim], prototype_labels [K])
        """
        if len(embeddings) == 0:
            raise ValueError("Cannot refine prototypes from empty embeddings.")

        embeddings = np.asarray(embeddings, dtype=np.float32)
        unique_labels, label_indices = np.unique(labels, return_inverse=True)
        n_classes = len(unique_labels)
        input_dim = embeddings.shape[1]

        # Initialize projection head if needed
        if self._projection_matrix is None or self._input_dim != input_dim:
            self._init_projection_head(input_dim, seed)

        # ---- v0.2.0: TRAIN the projection head via InfoNCE ----
        self._train_projection_head(embeddings, labels, label_indices, seed)

        # ---- Project through trained head ----
        projected = self._project(embeddings)
        proto_dim = projected.shape[1]

        # Initial prototypes: class means in the trained projection space
        prototypes = np.zeros((n_classes, proto_dim), dtype=np.float32)
        for k in range(n_classes):
            mask = label_indices == k
            if mask.sum() > 0:
                prototypes[k] = projected[mask].mean(axis=0)
            else:
                prototypes[k] = np.zeros(proto_dim, dtype=np.float32)

        # Warm start from existing prototypes if available
        if existing_prototypes is not None and existing_prototype_labels is not None:
            for i, proto_label in enumerate(existing_prototype_labels):
                idx = np.where(unique_labels == proto_label)[0]
                if len(idx) > 0 and i < len(existing_prototypes):
                    # EMA blend: 70% old + 30% new mean
                    prototypes[idx[0]] = (
                        0.7 * existing_prototypes[i] + 0.3 * prototypes[idx[0]]
                    )

        # ---- Prototype refinement iterations ----
        lr = self.config.learning_rate
        momentum = self.config.momentum
        proto_velocity = np.zeros_like(prototypes)

        for epoch in range(self.config.n_epochs):
            # Compute contrastive loss as if prototypes are the "anchors"
            sim_to_protos = projected @ prototypes.T  # [N, K]
            tau = max(self.config.temperature, 1e-4)
            sim_scaled = sim_to_protos / tau
            sim_scaled = sim_scaled - np.max(sim_scaled, axis=1, keepdims=True)
            exp_sim = np.exp(sim_scaled)

            # Cross-entropy with prototypes as class centers
            target_mask = np.zeros((len(embeddings), n_classes), dtype=np.float32)
            target_mask[np.arange(len(embeddings)), label_indices] = 1.0

            prob = exp_sim / (exp_sim.sum(axis=1, keepdims=True) + 1e-8)
            loss = float(-np.mean(np.sum(target_mask * np.log(prob + 1e-8), axis=1)))

            # Gradient of loss w.r.t. prototypes
            grad_proto = (prob - target_mask).T @ projected  # [K, d]
            grad_proto = grad_proto / len(embeddings)

            # Momentum update
            proto_velocity = momentum * proto_velocity - lr * grad_proto
            prototypes = prototypes + proto_velocity

            # Re-normalize prototypes
            norms = np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8
            prototypes = prototypes / norms

            # Early stopping if loss stabilizes
            if epoch > 10 and abs(loss) < 1e-4:
                break

        self._is_fitted = True
        return prototypes, unique_labels

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def project_query(self, query_embedding: np.ndarray) -> np.ndarray:
        """Project a query embedding into the contrastive space.

        Args:
            query_embedding: [D] backbone embedding.

        Returns:
            [projection_dim] projected query vector.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Contrastive learner not fitted. Call refine_prototypes() first."
            )
        query_2d = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        projected = self._project(query_2d)
        return cast(np.ndarray, projected[0])

    def nearest_prototype(
        self,
        query_embedding: np.ndarray,
        prototypes: np.ndarray,
        prototype_labels: np.ndarray,
    ) -> Tuple[Union[str, int], float, int]:
        """Find nearest contrastive prototype for a query.

        Args:
            query_embedding: [D] backbone embedding.
            prototypes: [K, d] contrastive prototypes.
            prototype_labels: [K] class labels.

        Returns:
            (predicted_label, confidence, prototype_index)
        """
        query_proj = self.project_query(query_embedding)
        # Cosine similarity (both are L2-normalized)
        sims = query_proj @ prototypes.T
        best_idx = int(np.argmax(sims))
        confidence = float((sims[best_idx] + 1.0) / 2.0)  # Map [-1,1] to [0,1]
        confidence = float(np.clip(confidence, 0.0, 1.0))
        return prototype_labels[best_idx], confidence, best_idx

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def class_separation_score(
        self,
        prototypes: np.ndarray,
        prototype_labels: np.ndarray,
    ) -> float:
        """Measure inter-class separation in the contrastive space.

        Computes the ratio: mean(inter-class similarity) / mean(intra-class similarity).
        Lower values indicate better separation.

        Args:
            prototypes: [K, d] contrastive prototypes.
            prototype_labels: [K] class labels.

        Returns:
            Separation score (lower = better separation).
        """
        if len(prototypes) < 2:
            return 0.0

        sims = prototypes @ prototypes.T  # [K, K] cosine similarities
        n = len(prototypes)

        inter_sims: List[float] = []
        intra_sims: List[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                if prototype_labels[i] == prototype_labels[j]:
                    intra_sims.append(float(sims[i, j]))
                else:
                    inter_sims.append(float(sims[i, j]))

        mean_inter = np.mean(inter_sims) if inter_sims else 0.0
        mean_intra = np.mean(intra_sims) if intra_sims else 1.0
        if mean_intra < 1e-8:
            return 0.0
        return float(abs(mean_inter) / abs(mean_intra))

    @property
    def is_fitted(self) -> bool:
        """Check whether the learner has been fitted."""
        return self._is_fitted

    def reset(self) -> None:
        """Reset the projection head and fitted state."""
        self._projection_matrix = None
        self._projection_bias = None
        self._second_layer_matrix = None
        self._second_layer_bias = None
        self._input_dim = None
        self._is_fitted = False
