"""Multi-signal uncertainty quantification for few-shot vision.

Implements three complementary uncertainty estimates that together provide
a holistic view of prediction reliability:

1. **Epistemic** (model uncertainty): Embedding perturbation sensitivity.
   Measures how much the query embedding shifts under small Gaussian
   perturbations. High sensitivity indicates the model lacks robust
   representations for this input (epistemic uncertainty).
   (Note: Full MC Dropout is planned for a future torch-dependent release;
   the current perturbation-based proxy operates entirely in numpy.)

2. **Aleatoric** (data uncertainty): Entropy of the softmax distribution
   over nearest-k neighbors. High when class boundaries are ambiguous.

3. **Distributional** (OOD uncertainty): Mahalanobis distance to class-
   conditional Gaussian distributions. High for out-of-distribution inputs.

Out-of-distribution (OOD) detection uses class-conditional Mahalanobis
distances with a configurable threshold. Inputs exceeding the threshold
across all known classes are flagged as OOD.

Design: numpy-first with optional torch acceleration planned for MC Dropout.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class UncertaintyReport:
    """Structured uncertainty decomposition for a single prediction.

    Attributes:
        epistemic: Embedding perturbation sensitivity (model uncertainty), [0, 1].
        aleatoric: Entropy over nearest-k softmax (data uncertainty), [0, 1].
        distributional: Mahalanobis-based OOD score, [0, 1].
        composite: Weighted fusion of all three signals, [0, 1].
        is_ood: Whether the input is flagged as out-of-distribution.
        ood_score: Raw Mahalanobis distance percentile.
        nearest_class_margin: Margin between top-2 class Mahalanobis distances.
        entropy: Raw entropy of k-NN softmax distribution.
        variance: Raw embedding perturbation variance (before normalization).
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

    def to_dict(self) -> dict[str, float]:
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


def _seed_from_embedding(embedding: np.ndarray) -> int:
    """A stable seed derived from an embedding's contents.

    `hashlib`, never the builtin `hash()`: hash randomisation is seeded per
    process, so the builtin would reintroduce exactly the run-to-run variation
    this exists to remove.

    Args:
        embedding: The array to derive a seed from.

    Returns:
        A non-negative integer, identical for identical contents.
    """

    digest = hashlib.blake2b(
        np.ascontiguousarray(embedding, dtype=np.float32).tobytes(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


class UncertaintyQuantifier:
    """Multi-signal uncertainty estimation for few-shot predictions.

    Combines epistemic (model), aleatoric (data), and distributional (OOD)
    uncertainty into a single composite score. Configurable weights allow
    domain-specific tuning of the trade-off between different uncertainty types.

    The composite score is computed as:
        U = w_e * U_epistemic + w_a * U_aleatoric + w_d * U_distributional
    normalized by (w_e + w_a + w_d).

    Modes:
        - "entropy": Aleatoric uncertainty only (k-NN entropy).
        - "mahalanobis": Distributional OOD detection only.
        - "mcdropout": Epistemic proxy (embedding perturbation sensitivity).
        - "ensemble": All three signals fused together.
    """

    def __init__(
        self,
        epistemic_weight: float = 1.0,
        aleatoric_weight: float = 1.0,
        distributional_weight: float = 1.0,
        ood_percentile: float = 95.0,
        min_ood_samples: int = 10,
        k_neighbors: int = 5,
        perturbation_samples: int = 10,
        perturbation_scale: float = 0.01,
        mahalanobis_regularization: float = 1e-4,
    ) -> None:
        """Initialize the uncertainty quantifier.

        Args:
            epistemic_weight: Weight for epistemic uncertainty in composite.
            aleatoric_weight: Weight for k-NN entropy in composite.
            distributional_weight: Weight for Mahalanobis OOD in composite.
            ood_percentile: Percentile threshold for OOD detection [0, 100].
            min_ood_samples: Minimum samples before OOD detection activates.
            k_neighbors: Number of neighbors for entropy computation.
            perturbation_samples: Number of perturbed embeddings for epistemic proxy.
            perturbation_scale: Std of Gaussian noise for embedding perturbation.
            mahalanobis_regularization: Ridge term for covariance inverse.
        """
        self.w_e = epistemic_weight
        self.w_a = aleatoric_weight
        self.w_d = distributional_weight
        self.ood_percentile = ood_percentile
        self.min_ood_samples = min_ood_samples
        self.k = k_neighbors
        self.perturbation_samples = perturbation_samples
        self.perturbation_scale = perturbation_scale
        self.reg = mahalanobis_regularization

        # Class-conditional Gaussian parameters for Mahalanobis
        self._class_means: dict[str | int, np.ndarray] = {}
        self._class_covs: dict[str | int, np.ndarray] = {}
        self._class_cov_invs: dict[str | int, np.ndarray] = {}
        self._global_mean: np.ndarray | None = None

        # OOD threshold (computed from calibration data)
        self._ood_threshold: float = float("inf")
        self._calibration_distances: list[float] = []

    # ------------------------------------------------------------------
    # Mahalanobis OOD (distributional uncertainty)
    # ------------------------------------------------------------------

    def fit_class_distributions(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Fit class-conditional Gaussian distributions with shrinkage.

        v0.2.0 fix: In the few-shot regime (e.g., 10 samples in 512-dim),
        the sample covariance is severely rank-deficient. We now apply:

        1. **Shrinkage estimation**: Cov = (1-α)*S_sample + α*diag(S_sample)
           This Ledoit-Wolf-style shrinkage targets the diagonal, ensuring
           the covariance is always invertible regardless of n/d ratio.
        2. **Adaptive shrinkage factor**: α = d / (d + n_per_class)
           When n << d, the estimate is heavily shrunk toward diagonal.
        3. **Fallback to diagonal**: When n_per_class < min(d, 5), use
           pure diagonal covariance (variance-per-dimension only).

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
            n_k = len(class_embs)

            if n_k < 2:
                continue

            mean = class_embs.mean(axis=0)
            centered = class_embs - mean

            if n_k <= d:
                # Few-shot regime: use diagonal covariance with shrinkage
                # Compute per-dimension variance
                diag_var = np.var(class_embs, axis=0) + self.reg
                # Shrinkage factor: more shrinkage when n_k << d
                alpha = min(1.0, d / (d + n_k))
                # Empirical covariance (best effort)
                sample_cov = (centered.T @ centered) / max(n_k - 1, 1)
                # Shrunk toward diagonal
                diag_mat = np.diag(diag_var)
                cov_reg = (1.0 - alpha) * sample_cov + alpha * diag_mat
            else:
                # Sufficient samples: use ridge-regularized covariance
                cov = (centered.T @ centered) / (n_k - 1)
                alpha = d / (d + n_k)  # Light shrinkage
                diag_var = np.diag(cov) + self.reg
                diag_mat = np.diag(diag_var)
                cov_reg = (1.0 - alpha) * cov + alpha * diag_mat
                cov_reg = cov_reg + self.reg * np.eye(d, dtype=np.float32)

            try:
                cov_inv = np.linalg.inv(cov_reg.astype(np.float64))
            except np.linalg.LinAlgError:
                cov_inv = np.linalg.pinv(cov_reg.astype(np.float64))

            self._class_means[label] = mean.astype(np.float32)
            self._class_covs[label] = cov_reg.astype(np.float32)
            self._class_cov_invs[label] = cov_inv.astype(np.float32)

        if len(embeddings) > 0:
            self._global_mean = embeddings.mean(axis=0).astype(np.float32)

        self._compute_ood_threshold(embeddings, labels)

    def mahalanobis_distance(
        self,
        embedding: np.ndarray,
        class_label: str | int,
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
    ) -> tuple[float, str | int, float]:
        """Find the minimum Mahalanobis distance across all known classes.

        Args:
            embedding: [D] feature vector.

        Returns:
            (min_distance, closest_class, margin_to_second_closest)
        """
        if not self._class_means:
            return 0.0, "unknown", 0.0

        distances: list[tuple[float, str | int]] = []
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

        distances: list[float] = []
        for i in range(len(embeddings)):
            emb = embeddings[i]
            label = labels[i]
            dist = self.mahalanobis_distance(emb, label)
            distances.append(dist)

        self._calibration_distances = distances
        self._ood_threshold = float(
            np.percentile(distances, self.ood_percentile)
        )

    def is_ood(
        self, embedding: np.ndarray, *, min_dist: float | None = None
    ) -> tuple[bool, float]:
        """Check if an embedding is out-of-distribution.

        Args:
            embedding: [D] feature vector.
            min_dist: The embedding's minimum Mahalanobis distance, if the
                caller has already computed it. Mahalanobis is O(D^2) per class
                and is the most expensive step here, so a caller that has the
                number should not pay for it twice (#40). Omit it and the
                distance is computed from `embedding` as before.

        Returns:
            (is_ood, ood_score_normalized)
        """
        if min_dist is None:
            min_dist, _, _ = self.min_mahalanobis_distance(embedding)

        if self._ood_threshold == float("inf") or not self._calibration_distances:
            return False, 0.0

        is_ood_flag = min_dist > self._ood_threshold

        # Normalize score: 0 = in-dist, 1 = far OOD
        max_calib = max(self._calibration_distances) if self._calibration_distances else 1.0
        ood_score = float(np.clip(min_dist / max(max_calib, 1e-8), 0.0, 1.0))

        return is_ood_flag, ood_score

    # ------------------------------------------------------------------
    # Epistemic uncertainty (embedding perturbation sensitivity)
    # ------------------------------------------------------------------

    def estimate_epistemic(
        self,
        query_embedding: np.ndarray,
        seed: int | None = None,
    ) -> tuple[float, float]:
        """Estimate epistemic uncertainty via stochastic embedding perturbation.

        Adds small Gaussian noise to the query embedding and measures how much the
        normalized direction shifts. High sensitivity indicates the embedding
        lacks robustness — the model has high epistemic uncertainty for this
        input.

        When ``seed`` is None the seed is derived from the embedding's own bytes,
        so the same input always gives the same answer while different inputs
        still get different perturbation patterns.

        This used to seed from OS entropy, which made the result vary between
        identical calls — including through `quantify(mode="ensemble")`, the
        default, and therefore through `PredictionResult.uncertainty_report`. It
        contradicted the project's determinism guarantee, and the smoke benchmark
        reported that guarantee as holding because accuracy does not depend on
        this signal (#58).

        Nothing is lost by fixing it. The stochastic signal is carried by
        averaging over ``perturbation_samples`` *within* a call; varying the seed
        *between* calls added run-to-run noise to a reported number without
        adding information to any single report. Pass an explicit seed to choose
        a different pattern deliberately.

        This is a numpy-based proxy for MC Dropout. Full MC Dropout through the
        backbone requires torch and is planned for a future release.

        Args:
            query_embedding: [D] query embedding vector.
            seed: Optional random seed. None (default) derives one from the
                embedding, which is reproducible.

        Returns:
            (raw_variance, normalized_variance in [0, 1])
        """
        query = np.asarray(query_embedding, dtype=np.float32)
        rng = np.random.default_rng(
            _seed_from_embedding(query) if seed is None else seed
        )
        query_norm = float(np.linalg.norm(query)) + 1e-8

        perturbed_embeddings: list[np.ndarray] = []
        # `query.shape` is Any because the parameter is a bare np.ndarray, and with an
        # Any size numpy's overloads resolve to the scalar (size=None) signature, which
        # returns a float. Naming the type picks the array overload instead. Runtime
        # behaviour is unchanged -- this is a typing fix, not a logic fix.
        noise_shape: tuple[int, ...] = query.shape
        for _ in range(self.perturbation_samples):
            noise = rng.normal(0.0, self.perturbation_scale, size=noise_shape).astype(np.float32)
            perturbed = query + noise * query_norm  # scale noise to embedding magnitude
            # L2 normalize to focus on directional sensitivity
            perturbed_norm = float(np.linalg.norm(perturbed)) + 1e-8
            perturbed_embeddings.append((perturbed / perturbed_norm).astype(np.float32))

        stacked = np.stack(perturbed_embeddings, axis=0)  # [M, D]
        mean_emb = stacked.mean(axis=0)
        # Frobenius norm of variance across perturbed embeddings
        var = float(np.mean(np.sum((stacked - mean_emb) ** 2, axis=1)))
        # Normalize: variance of direction cosines bounded in [0, 2]
        var_norm = float(np.clip(var / 2.0, 0.0, 1.0))
        return var, var_norm

    @staticmethod
    def compute_perturbation_variance(
        perturbed_embeddings: list[np.ndarray],
    ) -> tuple[float, float]:
        """Compute variance across a set of perturbed or sampled embeddings.

        The sample variance of L2-normalized embeddings indicates model uncertainty:
        if the embedding direction is stable under perturbation, uncertainty is low.

        Args:
            perturbed_embeddings: List of [D] embeddings from perturbation or passes.

        Returns:
            (variance_raw, normalized_variance in [0, 1])
        """
        if len(perturbed_embeddings) < 2:
            return 0.0, 0.0

        stacked = np.stack(perturbed_embeddings, axis=0)  # [M, D]
        mean_emb = stacked.mean(axis=0)
        var = float(np.mean(np.sum((stacked - mean_emb) ** 2, axis=1)))
        var_norm = float(np.clip(var / 2.0, 0.0, 1.0))
        return var, var_norm

    # ------------------------------------------------------------------
    # Aleatoric uncertainty (k-NN entropy)
    # ------------------------------------------------------------------

    def compute_knn_entropy(
        self,
        query_embedding: np.ndarray,
        support_embeddings: np.ndarray,
        support_labels: np.ndarray,
    ) -> tuple[float, float]:
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
        unique = np.unique(k_labels)
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
        mode: str = "ensemble",
    ) -> UncertaintyReport:
        """Compute the multi-signal uncertainty decomposition.

        Signals are computed selectively based on the mode parameter:
        - "entropy": Aleatoric (k-NN entropy) only.
        - "mahalanobis": Distributional (Mahalanobis OOD) only.
        - "mcdropout": Epistemic (perturbation sensitivity) only.
        - "ensemble": All three signals fused.

        Args:
            query_embedding: [D] query embedding (single forward pass).
            support_embeddings: [N, D] support set.
            support_labels: [N] class labels.
            mode: Uncertainty computation mode (default "ensemble").

        Returns:
            UncertaintyReport with computed signals and composite score.
        """
        report = UncertaintyReport()
        compute_epistemic = mode in ("mcdropout", "ensemble")
        compute_aleatoric = mode in ("entropy", "ensemble")
        compute_distributional = mode in ("mahalanobis", "ensemble")

        # Epistemic: embedding perturbation sensitivity
        if compute_epistemic:
            var_raw, var_norm = self.estimate_epistemic(query_embedding)
            report.variance = var_raw
            report.epistemic = var_norm

        # Aleatoric: k-NN entropy
        if compute_aleatoric:
            entropy_raw, entropy_norm = self.compute_knn_entropy(
                query_embedding, support_embeddings, support_labels
            )
            report.entropy = entropy_raw
            report.aleatoric = entropy_norm

        # Distributional: Mahalanobis OOD
        if compute_distributional:
            # One Mahalanobis computation, shared. It used to run twice per
            # query: once here for the margin, and again inside is_ood() (#40).
            min_dist, _closest_class, margin = self.min_mahalanobis_distance(query_embedding)
            is_ood_flag, ood_score = self.is_ood(query_embedding, min_dist=min_dist)
            report.distributional = ood_score
            report.is_ood = is_ood_flag
            report.ood_score = ood_score
            report.nearest_class_margin = margin

        # Composite score: weighted average of computed signals
        total_w = 0.0
        composite = 0.0
        if compute_epistemic:
            composite += self.w_e * report.epistemic
            total_w += self.w_e
        if compute_aleatoric:
            composite += self.w_a * report.aleatoric
            total_w += self.w_a
        if compute_distributional:
            composite += self.w_d * report.distributional
            total_w += self.w_d
        if total_w > 0:
            report.composite = composite / total_w

        return report

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_ood_summary(self) -> dict[str, Any]:
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

    def get_class_statistics(self) -> dict[str, dict[str, Any]]:
        """Return per-class distribution statistics."""
        stats: dict[str, dict[str, Any]] = {}
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
