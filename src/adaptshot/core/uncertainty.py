"""Multi-signal uncertainty quantification for few-shot vision.

Implements three complementary uncertainty estimates that together provide
a holistic view of prediction reliability:

1. **Epistemic** (model uncertainty): Monte Carlo Dropout variance across
   multiple forward passes. High when the model hasn't seen similar data.

2. **Aleatoric** (data uncertainty): Entropy of the softmax distribution
   over nearest-k neighbors. High when class boundaries are ambiguous.

3. **Distributional** (OOD uncertainty): Mahalanobis distance to class-
   conditional Gaussian distributions. High for out-of-distribution inputs.

Out-of-distribution (OOD) detection uses class-conditional Mahalanobis
distances with a configurable threshold. Inputs exceeding the threshold
across all known classes are flagged as OOD.

Design: numpy-first with optional torch acceleration for MC Dropout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class UncertaintyReport:
    """Structured uncertainty decomposition for a single prediction.

    Attributes:
        epistemic: MC Dropout variance (model uncertainty), [0, 1].
        aleatoric: Entropy over nearest-k softmax (data uncertainty), [0, 1].
        distributional: Mahalanobis-based OOD score, [0, 1].
        composite: Weighted fusion of all three signals, [0, 1].
        is_ood: Whether the input is flagged as out-of-distribution.
        ood_score: Raw Mahalanobis distance percentile.
        nearest_class_margin: Margin between top-2 class Mahalanobis distances.
        entropy: Raw entropy of k-NN softmax distribution.
        variance: Raw MC dropout variance (before normalization).
    """

    epistemic: float = 0.0
    aleatoric: float = 0.0
    distributional: float = 0.0
    composite: float = 0.0
    is_ood: bool = False
    ood_score: float = 0.0
    nearest_class_margin: float = 0.0
    entropy: float = 0.0
    variance: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Serialize to a flat dictionary."""
        return {
            "epistemic": self.epistemic,
            "aleatoric": self.aleatoric,
            "distributional": self.distributional,
            "composite": self.composite,
            "is_ood": float(self.is_ood),
            "ood_score": self.ood_score,
            "nearest_class_margin": self.nearest_class_margin,
            "entropy": self.entropy,
            "variance": self.variance,
        }


class UncertaintyQuantifier:
    """Multi-signal uncertainty estimation for few-shot predictions.

    Combines epistemic (model), aleatoric (data), and distributional (OOD)
    uncertainty into a single composite score. Configurable weights allow
    domain-specific tuning of the trade-off between different uncertainty types.

    The composite score is computed as:
        U = w_e * U_epistemic + w_a * U_aleatoric + w_d * U_distributional
    normalized by (w_e + w_a + w_d).
    """

    def __init__(
        self,
        epistemic_weight: float = 1.0,
        aleatoric_weight: float = 1.0,
        distributional_weight: float = 1.0,
        ood_percentile: float = 95.0,
        min_ood_samples: int = 10,
        k_neighbors: int = 5,
        mc_samples: int = 10,
        mahalanobis_regularization: float = 1e-4,
    ) -> None:
        """Initialize the uncertainty quantifier.

        Args:
            epistemic_weight: Weight for MC Dropout uncertainty in composite.
            aleatoric_weight: Weight for k-NN entropy in composite.
            distributional_weight: Weight for Mahalanobis OOD in composite.
            ood_percentile: Percentile threshold for OOD detection [0, 100].
            min_ood_samples: Minimum samples before OOD detection activates.
            k_neighbors: Number of neighbors for entropy computation.
            mc_samples: Number of MC Dropout forward passes.
            mahalanobis_regularization: Ridge term for covariance inverse.
        """
        self.w_e = epistemic_weight
        self.w_a = aleatoric_weight
        self.w_d = distributional_weight
        self.ood_percentile = ood_percentile
        self.min_ood_samples = min_ood_samples
        self.k = k_neighbors
        self.mc_samples = mc_samples
        self.reg = mahalanobis_regularization

        # Class-conditional Gaussian parameters for Mahalanobis
        self._class_means: Dict[Union[str, int], np.ndarray] = {}
        self._class_covs: Dict[Union[str, int], np.ndarray] = {}
        self._class_cov_invs: Dict[Union[str, int], np.ndarray] = {}
        self._global_mean: Optional[np.ndarray] = None

        # OOD threshold (computed from calibration data)
        self._ood_threshold: float = float("inf")
        self._calibration_distances: List[float] = []

    # ------------------------------------------------------------------
    # Mahalanobis OOD (distributional uncertainty)
    # ------------------------------------------------------------------

    def fit_class_distributions(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Fit class-conditional Gaussian distributions for Mahalanobis distance.

        Args:
            embeddings: [N, D] support embeddings.
            labels: [N] class labels.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        labels = np.asarray(labels, dtype=object)
        self._class_means.clear()
        self._class_covs.clear()
        self._class_cov_invs.clear()

        unique_labels = np.unique(labels)
        d = embeddings.shape[1]

        for label in unique_labels:
            mask = labels == label
            class_embs = embeddings[mask]

            if len(class_embs) < 2:
                # Not enough samples for covariance; use global stats
                continue

            mean = class_embs.mean(axis=0)
            centered = class_embs - mean
            cov = (centered.T @ centered) / (len(class_embs) - 1)

            # Ridge regularization
            cov_reg = cov + self.reg * np.eye(d, dtype=np.float32)

            try:
                cov_inv = np.linalg.inv(cov_reg)
            except np.linalg.LinAlgError:
                cov_inv = np.linalg.pinv(cov_reg)

            self._class_means[label] = mean.astype(np.float32)
            self._class_covs[label] = cov_reg.astype(np.float32)
            self._class_cov_invs[label] = cov_inv.astype(np.float32)

        # Global mean for fallback
        if len(embeddings) > 0:
            self._global_mean = embeddings.mean(axis=0).astype(np.float32)

        # Recompute OOD threshold
        self._compute_ood_threshold(embeddings, labels)

    def mahalanobis_distance(
        self,
        embedding: np.ndarray,
        class_label: Union[str, int],
    ) -> float:
        """Compute Mahalanobis distance to a class-conditional distribution.

        Args:
            embedding: [D] feature vector.
            class_label: Class to measure distance to.

        Returns:
            Mahalanobis distance (lower = more in-distribution).
        """
        if class_label not in self._class_cov_invs:
            if self._global_mean is not None:
                # Fallback: Euclidean distance to global mean
                diff = embedding - self._global_mean
                return float(np.sqrt(diff @ diff))
            return float("inf")

        diff = embedding - self._class_means[class_label]
        maha_sq = diff @ self._class_cov_invs[class_label] @ diff
        return float(np.sqrt(max(0.0, maha_sq)))

    def min_mahalanobis_distance(
        self,
        embedding: np.ndarray,
    ) -> Tuple[float, Union[str, int], float]:
        """Find the minimum Mahalanobis distance across all known classes.

        Args:
            embedding: [D] feature vector.

        Returns:
            (min_distance, closest_class, margin_to_second_closest)
        """
        if not self._class_means:
            return 0.0, "unknown", 0.0

        distances: List[Tuple[float, Union[str, int]]] = []
        for label in self._class_means:
            dist = self.mahalanobis_distance(embedding, label)
            distances.append((dist, label))

        distances.sort(key=lambda x: x[0])
        min_dist, closest = distances[0]
        margin = distances[1][0] - min_dist if len(distances) > 1 else 0.0

        return min_dist, closest, margin

    def _compute_ood_threshold(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Compute OOD threshold from in-distribution calibration data.

        Uses the specified percentile of Mahalanobis distances on training data.
        New inputs with distances exceeding this threshold are flagged as OOD.

        Args:
            embeddings: [N, D] training embeddings.
            labels: [N] class labels.
        """
        if len(embeddings) < self.min_ood_samples:
            self._ood_threshold = float("inf")
            return

        distances: List[float] = []
        for i in range(len(embeddings)):
            emb = embeddings[i]
            label = labels[i]
            dist = self.mahalanobis_distance(emb, label)
            distances.append(dist)

        self._calibration_distances = distances
        self._ood_threshold = float(
            np.percentile(distances, self.ood_percentile)
        )

    def is_ood(self, embedding: np.ndarray) -> Tuple[bool, float]:
        """Check if an embedding is out-of-distribution.

        Args:
            embedding: [D] feature vector.

        Returns:
            (is_ood, ood_score_normalized)
        """
        min_dist, _, _ = self.min_mahalanobis_distance(embedding)

        if self._ood_threshold == float("inf") or not self._calibration_distances:
            return False, 0.0

        is_ood_flag = min_dist > self._ood_threshold

        # Normalize score: 0 = in-dist, 1 = far OOD
        max_calib = max(self._calibration_distances) if self._calibration_distances else 1.0
        ood_score = float(np.clip(min_dist / max(max_calib, 1e-8), 0.0, 1.0))

        return is_ood_flag, ood_score

    # ------------------------------------------------------------------
    # Epistemic uncertainty (MC Dropout variance)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_mc_variance(
        mc_embeddings: List[np.ndarray],
    ) -> Tuple[float, float]:
        """Compute variance across MC Dropout samples.

        The sample variance of L2-normed embeddings indicates model uncertainty:
        if the model consistently produces similar embeddings despite dropout,
        uncertainty is low. High variance across MC samples indicates epistemic
        uncertainty.

        Args:
            mc_embeddings: List of [D] embeddings from MC forward passes.

        Returns:
            (variance_norm, normalized_variance in [0, 1])
        """
        if len(mc_embeddings) < 2:
            return 0.0, 0.0

        stacked = np.stack(mc_embeddings, axis=0)  # [M, D]
        mean_emb = stacked.mean(axis=0)
        # Frobenius norm of variance
        var = np.mean(np.sum((stacked - mean_emb) ** 2, axis=1))
        # Normalize by embedding magnitude
        norm_factor = float(np.linalg.norm(mean_emb)) + 1e-8
        var_norm = float(np.clip(var / norm_factor, 0.0, 1.0))
        return var, var_norm

    # ------------------------------------------------------------------
    # Aleatoric uncertainty (k-NN entropy)
    # ------------------------------------------------------------------

    def compute_knn_entropy(
        self,
        query_embedding: np.ndarray,
        support_embeddings: np.ndarray,
        support_labels: np.ndarray,
    ) -> Tuple[float, float]:
        """Compute entropy of softmax over k nearest neighbors.

        High entropy indicates ambiguous class boundaries (data uncertainty).
        Low entropy indicates clear, well-separated class structure.

        Args:
            query_embedding: [D] query vector.
            support_embeddings: [N, D] support set.
            support_labels: [N] class labels.

        Returns:
            (entropy, normalized_entropy in [0, 1])
        """
        if len(support_embeddings) == 0:
            return 0.0, 0.0

        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        support = np.asarray(support_embeddings, dtype=np.float32)

        # Compute distances to all support examples
        diffs = query - support
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))

        # Find k nearest
        k = min(self.k, len(support))
        k_indices = np.argpartition(distances, k - 1)[:k]
        k_labels = support_labels[k_indices]
        k_dists = distances[k_indices]

        # Convert distances to weights (closer = higher weight)
        weights = 1.0 / (k_dists + 1e-8)
        weights = weights / weights.sum()

        # Compute weighted class distribution
        unique, counts = np.unique(k_labels, return_counts=False)
        class_weights = np.zeros(len(unique))
        for i, label in enumerate(unique):
            class_weights[i] = weights[k_labels == label].sum()

        # Normalize to probability distribution
        prob = class_weights / (class_weights.sum() + 1e-8)

        # Entropy
        entropy = float(-np.sum(prob * np.log(prob + 1e-8)))

        # Normalize by max entropy (log(K) for K classes in k-NN)
        max_entropy = np.log(max(len(unique), 2))
        norm_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

        return entropy, norm_entropy

    # ------------------------------------------------------------------
    # Composite uncertainty
    # ------------------------------------------------------------------

    def quantify(
        self,
        query_embedding: np.ndarray,
        support_embeddings: np.ndarray,
        support_labels: np.ndarray,
        mc_embeddings: Optional[List[np.ndarray]] = None,
    ) -> UncertaintyReport:
        """Compute the full multi-signal uncertainty decomposition.

        Args:
            query_embedding: [D] query embedding (single forward pass).
            support_embeddings: [N, D] support set.
            support_labels: [N] class labels.
            mc_embeddings: Optional list of [D] embeddings from MC Dropout passes.

        Returns:
            UncertaintyReport with all signals and composite score.
        """
        report = UncertaintyReport()

        # Epistemic: MC Dropout variance
        if mc_embeddings and len(mc_embeddings) > 1:
            var_raw, var_norm = self.compute_mc_variance(mc_embeddings)
            report.variance = var_raw
            report.epistemic = var_norm

        # Aleatoric: k-NN entropy
        entropy_raw, entropy_norm = self.compute_knn_entropy(
            query_embedding, support_embeddings, support_labels
        )
        report.entropy = entropy_raw
        report.aleatoric = entropy_norm

        # Distributional: Mahalanobis OOD
        min_dist, closest_class, margin = self.min_mahalanobis_distance(query_embedding)
        is_ood_flag, ood_score = self.is_ood(query_embedding)
        report.distributional = ood_score
        report.is_ood = is_ood_flag
        report.ood_score = ood_score
        report.nearest_class_margin = margin

        # Composite score: weighted average
        total_w = self.w_e + self.w_a + self.w_d
        if total_w > 0:
            report.composite = (
                self.w_e * report.epistemic
                + self.w_a * report.aleatoric
                + self.w_d * report.distributional
            ) / total_w

        return report

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_ood_summary(self) -> Dict[str, Any]:
        """Return diagnostic summary of OOD detection state."""
        return {
            "ood_threshold": self._ood_threshold,
            "n_calibration_samples": len(self._calibration_distances),
            "n_classes_fitted": len(self._class_means),
            "calibration_mean": float(np.mean(self._calibration_distances))
            if self._calibration_distances
            else 0.0,
            "calibration_std": float(np.std(self._calibration_distances))
            if self._calibration_distances
            else 0.0,
            "ood_percentile": self.ood_percentile,
            "min_ood_samples": self.min_ood_samples,
        }

    def get_class_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Return per-class distribution statistics."""
        stats: Dict[str, Dict[str, Any]] = {}
        for label in self._class_means:
            mean = self._class_means[label]
            cov = self._class_covs.get(label)
            stats[str(label)] = {
                "mean_norm": float(np.linalg.norm(mean)),
                "cov_trace": float(np.trace(cov)) if cov is not None else 0.0,
                "cov_det": float(np.linalg.det(cov)) if cov is not None and cov.size > 0 else 0.0,
            }
        return stats

    def reset(self) -> None:
        """Reset all distribution fits and calibration data."""
        self._class_means.clear()
        self._class_covs.clear()
        self._class_cov_invs.clear()
        self._global_mean = None
        self._ood_threshold = float("inf")
        self._calibration_distances.clear()
