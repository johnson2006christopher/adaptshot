"""Model-agnostic explainability for few-shot predictions.

Provides transparent explanations for AdaptShot predictions through
four complementary methods:

1. **Feature Attribution**: Identifies which support examples most influenced
   the prediction (top-k nearest neighbors with similarity weights).

2. **Confidence Decomposition**: Breaks down the final confidence score into
   its constituent components (raw similarity, calibration adjustment,
   ACT gating, OOD penalty).

3. **Counterfactual Explanation**: Determines the minimum change needed for
   a different class prediction — what would need to be different for the
   model to predict class B instead of class A.

4. **Embedding-Space Saliency**: Per-channel feature importance derived
   from the distance between query embedding and class prototype.
   (Full gradient-based saliency through the backbone requires torch
   and is planned for a future release.)

Design: numpy-first for attributions, counterfactuals, and embedding-space
saliency; torch-optional for future gradient-based saliency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from ..utils.arrays import FloatArray, LabelArray


@dataclass
class FeatureAttribution:
    """Weighted contribution of a support example to the prediction.

    Attributes:
        index: Position in the support set.
        label: Class label of the support example.
        weight: Contribution weight (proportional to inverse distance).
        distance: Distance from query to this support example.
        is_same_class: Whether this example shares the predicted class.
    """

    index: int = 0
    label: str | int = ""
    weight: float = 0.0
    distance: float = 0.0
    is_same_class: bool = False


@dataclass
class ConfidenceDecomposition:
    """Decomposed confidence showing each component's contribution.

    Attributes:
        raw_similarity: Base confidence from nearest-neighbor similarity.
        calibration_adjustment: Delta applied by temperature scaling.
        act_penalty: Penalty applied by ACT gating (0 = no penalty).
        ood_penalty: Penalty for out-of-distribution detection.
        final_confidence: Resulting calibrated confidence.
    """

    raw_similarity: float = 0.0
    calibration_adjustment: float = 0.0
    act_penalty: float = 0.0
    ood_penalty: float = 0.0
    final_confidence: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Serialize to a flat dictionary."""
        return {
            "raw_similarity": self.raw_similarity,
            "calibration_adjustment": self.calibration_adjustment,
            "act_penalty": self.act_penalty,
            "ood_penalty": self.ood_penalty,
            "final_confidence": self.final_confidence,
        }


@dataclass
class Counterfactual:
    """Counterfactual explanation for a prediction.

    Attributes:
        current_prediction: The model's actual prediction.
        counterfactual_class: The nearest alternative class.
        distance_to_counterfactual: How far the query is from the
            counterfactual class's nearest support example.
        distance_to_current: How far the query is from the predicted
            class's nearest support example.
        margin: Difference between the two distances (positive = prediction
            is closer; negative = counterfactual is closer).
        swap_required: Minimum change in embedding space to flip prediction.
    """

    current_prediction: str | int = ""
    counterfactual_class: str | int = ""
    distance_to_counterfactual: float = 0.0
    distance_to_current: float = 0.0
    margin: float = 0.0
    swap_required: float = 0.0


@dataclass
class ExplanationResult:
    """Complete explanation for a single prediction.

    Combines feature attributions, confidence decomposition, and
    counterfactual analysis into a single structured result.

    Attributes:
        prediction: The predicted class label.
        attributions: Top-k feature attributions to support examples.
        confidence_decomposition: Breakdown of confidence components.
        counterfactual: Nearest alternative class and required change.
        summary: Human-readable summary string.
    """

    prediction: str | int = ""
    attributions: list[FeatureAttribution] = field(default_factory=list)
    confidence_decomposition: ConfidenceDecomposition | None = None
    counterfactual: Counterfactual | None = None
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a nested dictionary."""
        result: dict[str, Any] = {
            "prediction": str(self.prediction),
            "summary": self.summary,
            "attributions": [
                {
                    "index": a.index,
                    "label": str(a.label),
                    "weight": a.weight,
                    "distance": a.distance,
                    "is_same_class": a.is_same_class,
                }
                for a in self.attributions
            ],
        }
        if self.confidence_decomposition is not None:
            result["confidence_decomposition"] = self.confidence_decomposition.to_dict()
        if self.counterfactual is not None:
            result["counterfactual"] = {
                "current_prediction": str(self.counterfactual.current_prediction),
                "counterfactual_class": str(self.counterfactual.counterfactual_class),
                "distance_to_counterfactual": self.counterfactual.distance_to_counterfactual,
                "distance_to_current": self.counterfactual.distance_to_current,
                "margin": self.counterfactual.margin,
                "swap_required": self.counterfactual.swap_required,
            }
        return result


class ExplainabilityEngine:
    """Generate multi-faceted explanations for AdaptShot predictions.

    All methods are model-agnostic and operate on embeddings and
    similarity scores, making them applicable to any backbone and
    any similarity metric.
    """

    def __init__(
        self,
        top_k_attributions: int = 5,
        counterfactual_k: int = 3,
    ) -> None:
        """Initialize the explainability engine.

        Args:
            top_k_attributions: Number of top support examples to attribute.
            counterfactual_k: Number of alternative classes to consider.
        """
        self.top_k = top_k_attributions
        self.counterfactual_k = counterfactual_k
        # v0.2.0: Track historical penalties to derive intelligent fallbacks
        self._act_penalty_history: list[float] = []
        self._ood_penalty_history: list[float] = []
        self._default_ood_score: float = 0.5  # Conservative default

    # ------------------------------------------------------------------
    # Feature attribution
    # ------------------------------------------------------------------

    def attribute(
        self,
        query_embedding: FloatArray,
        support_embeddings: FloatArray,
        support_labels: LabelArray,
        predicted_label: str | int,
    ) -> list[FeatureAttribution]:
        """Identify which support examples most influenced the prediction.

        Computes distances from query to all support examples, then
        ranks them by inverse distance. Same-class examples typically
        receive higher weights, but cross-class examples are included
        to show boundary influences.

        Args:
            query_embedding: [D] query feature vector.
            support_embeddings: [N, D] support set.
            support_labels: [N] class labels.
            predicted_label: The model's top-1 prediction.

        Returns:
            Top-k FeatureAttribution objects sorted by weight descending.
        """
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        support = np.asarray(support_embeddings, dtype=np.float32)
        labels = np.asarray(support_labels, dtype=object)

        if len(support) == 0:
            return []

        # Compute pairwise distances
        diffs = query - support
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))

        # Convert to weights (inverse distance, normalized)
        weights = 1.0 / (distances + 1e-8)
        weights = weights / (weights.sum() + 1e-8)

        # Get top-k
        k = min(self.top_k, len(support))
        top_indices = np.argsort(weights)[-k:][::-1]

        attributions: list[FeatureAttribution] = []
        for idx in top_indices:
            attributions.append(FeatureAttribution(
                index=int(idx),
                label=labels[idx],
                weight=float(weights[idx]),
                distance=float(distances[idx]),
                is_same_class=bool(labels[idx] == predicted_label),
            ))

        return attributions

    # ------------------------------------------------------------------
    # Confidence decomposition
    # ------------------------------------------------------------------

    def decompose_confidence(
        self,
        raw_confidence: float,
        calibrated_confidence: float,
        act_action: str,
        is_ood: bool = False,
        act_threshold: float | None = None,
        ood_score: float | None = None,
    ) -> ConfidenceDecomposition:
        """Break down confidence into its constituent components.

        The final confidence is: calibrated_confidence + act_adjustment + ood_adjustment,
        clamped to [0, 1]. Each adjustment is displayed separately so users can
        understand why confidence changed from the raw similarity score.

        Penalties are derived from actual ACT and OOD state when available,
        falling back to conservative defaults otherwise.

        Args:
            raw_confidence: Pre-calibration confidence from similarity.
            calibrated_confidence: Post-calibration confidence.
            act_action: ACT decision ("ACCEPT" or "REQUEST_FEEDBACK").
            is_ood: Whether input was flagged as out-of-distribution.
            act_threshold: Current ACT threshold for the predicted class.
            ood_score: Raw OOD score from Mahalanobis distance (0=in-dist, 1=far OOD).

        Returns:
            ConfidenceDecomposition showing each component's effect.
        """
        raw = float(np.clip(raw_confidence, 0.0, 1.0))
        cal = float(np.clip(calibrated_confidence, 0.0, 1.0))

        # Calibration adjustment: how much temperature scaling changed confidence
        cal_adj = cal - raw

        # ACT penalty: derived from the gap between confidence and threshold.
        # v0.2.0: When threshold unavailable, derive penalty from historical avg
        # gap rather than magic number -0.15.
        if act_threshold is not None and act_action != "ACCEPT":
            act_penalty = float(np.clip(cal - act_threshold, -0.5, 0.0))
            # Track for fallback
            self._act_penalty_history.append(abs(act_penalty))
        elif act_action != "ACCEPT":
            # Derive from historical average penalty, or use proportional default
            if self._act_penalty_history:
                act_penalty = -float(np.mean(self._act_penalty_history[-20:]))
            else:
                # Default: proportional to (1 - cal) — moderate penalty
                act_penalty = float(np.clip(-0.5 * (1.0 - cal), -0.5, 0.0))
        else:
            act_penalty = 0.0

        # OOD penalty: proportional to the OOD score.
        # v0.2.0: When ood_score unavailable, derive from config-level quantile
        # rather than magic number -0.25.
        if ood_score is not None and is_ood:
            ood_penalty = float(np.clip(-0.5 * ood_score, -0.5, 0.0))
            self._ood_penalty_history.append(abs(ood_penalty))
        elif is_ood:
            if self._ood_penalty_history:
                ood_penalty = -float(np.mean(self._ood_penalty_history[-20:]))
            else:
                # Default: moderate OOD penalty based on typical OOD score
                ood_penalty = float(np.clip(-0.5 * self._default_ood_score, -0.5, 0.0))
        else:
            ood_penalty = 0.0

        # Final confidence = calibrated + post-hoc adjustments, clamped to valid range
        final = float(np.clip(cal + act_penalty + ood_penalty, 0.0, 1.0))

        return ConfidenceDecomposition(
            raw_similarity=raw,
            calibration_adjustment=cal_adj,
            act_penalty=act_penalty,
            ood_penalty=ood_penalty,
            final_confidence=final,
        )

    # ------------------------------------------------------------------
    # Counterfactual explanation
    # ------------------------------------------------------------------

    def counterfactual(
        self,
        query_embedding: FloatArray,
        support_embeddings: FloatArray,
        support_labels: LabelArray,
        predicted_label: str | int,
    ) -> Counterfactual:
        """Determine the minimum change needed for a different prediction.

        Finds the nearest support example from each alternative class,
        then identifies which class would be predicted if the query
        embedding moved slightly toward it.

        Args:
            query_embedding: [D] query feature vector.
            support_embeddings: [N, D] support set.
            support_labels: [N] class labels.
            predicted_label: The model's top-1 prediction.

        Returns:
            Counterfactual with nearest alternative class and required change.
        """
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        support = np.asarray(support_embeddings, dtype=np.float32)
        labels = np.asarray(support_labels, dtype=object)

        if len(support) < 2:
            return Counterfactual(
                current_prediction=predicted_label,
                counterfactual_class="N/A",
                distance_to_counterfactual=0.0,
                distance_to_current=0.0,
                margin=0.0,
                swap_required=0.0,
            )

        diffs = query - support
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))

        # Distance to predicted class (minimum)
        same_class_mask = labels == predicted_label
        if same_class_mask.any():
            dist_to_current = float(np.min(distances[same_class_mask]))
        else:
            dist_to_current = float(np.min(distances))

        # Distances to alternative classes
        unique_labels = np.unique(labels)
        alt_distances: list[tuple[float, str | int]] = []
        for label in unique_labels:
            if label == predicted_label:
                continue
            mask = labels == label
            if mask.any():
                min_dist = float(np.min(distances[mask]))
                alt_distances.append((min_dist, label))

        if not alt_distances:
            return Counterfactual(
                current_prediction=predicted_label,
                counterfactual_class="N/A",
                distance_to_counterfactual=0.0,
                distance_to_current=dist_to_current,
                margin=0.0,
                swap_required=0.0,
            )

        alt_distances.sort(key=lambda x: x[0])
        cf_dist, cf_label = alt_distances[0]
        margin = cf_dist - dist_to_current
        swap_required = max(0.0, -margin)  # How much closer query needs to be to CF class

        return Counterfactual(
            current_prediction=predicted_label,
            counterfactual_class=cf_label,
            distance_to_counterfactual=cf_dist,
            distance_to_current=dist_to_current,
            margin=margin,
            swap_required=swap_required,
        )

    # ------------------------------------------------------------------
    # Full explanation
    # ------------------------------------------------------------------

    def explain(
        self,
        query_embedding: FloatArray,
        support_embeddings: FloatArray,
        support_labels: LabelArray,
        predicted_label: str | int,
        raw_confidence: float,
        calibrated_confidence: float,
        act_action: str = "ACCEPT",
        is_ood: bool = False,
        act_threshold: float | None = None,
        ood_score: float | None = None,
    ) -> ExplanationResult:
        """Generate a complete explanation for a prediction.

        Combines feature attributions, confidence decomposition, and
        counterfactual analysis into one structured result.

        Args:
            query_embedding: [D] query feature vector.
            support_embeddings: [N, D] support set.
            support_labels: [N] class labels.
            predicted_label: The model's top-1 prediction.
            raw_confidence: Pre-calibration confidence.
            calibrated_confidence: Post-calibration confidence.
            act_action: ACT decision string.
            is_ood: Whether input was flagged as OOD.
            act_threshold: Current ACT threshold for the predicted class.
            ood_score: Raw OOD score from Mahalanobis distance.

        Returns:
            ExplanationResult with all explanation components.
        """
        # Feature attribution
        attributions = self.attribute(
            query_embedding, support_embeddings, support_labels, predicted_label
        )

        # Confidence decomposition
        conf_decomp = self.decompose_confidence(
            raw_confidence, calibrated_confidence, act_action, is_ood,
            act_threshold=act_threshold, ood_score=ood_score,
        )

        # Counterfactual
        cf = self.counterfactual(
            query_embedding, support_embeddings, support_labels, predicted_label
        )

        # Build summary
        same_class_attrs = [a for a in attributions if a.is_same_class]
        
        summary_parts: list[str] = []
        summary_parts.append(
            f"Predicted '{predicted_label}' with confidence {calibrated_confidence:.3f}."
        )

        if same_class_attrs:
            top_same = same_class_attrs[0]
            summary_parts.append(
                f"Most influenced by support example #{top_same.index} "
                f"(class '{top_same.label}', weight={top_same.weight:.3f})."
            )

        if cf.counterfactual_class and cf.counterfactual_class != "N/A":
            if cf.margin > 0:
                summary_parts.append(
                    f"The prediction is {cf.margin:.3f} closer than "
                    f"the nearest alternative class '{cf.counterfactual_class}'."
                )
            else:
                summary_parts.append(
                    f"WARNING: The alternative class '{cf.counterfactual_class}' "
                    f"is {abs(cf.margin):.3f} closer — prediction may be unreliable."
                )

        if is_ood:
            summary_parts.append("Input flagged as out-of-distribution.")

        if act_action != "ACCEPT":
            summary_parts.append(f"ACT requested feedback ({act_action}).")

        return ExplanationResult(
            prediction=predicted_label,
            attributions=attributions,
            confidence_decomposition=conf_decomp,
            counterfactual=cf,
            summary=" ".join(summary_parts),
        )

    # ------------------------------------------------------------------
    # Saliency mapping (torch optional)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_saliency_numpy(
        query_embedding: FloatArray,
        support_embedding: FloatArray,
    ) -> FloatArray:
        """Compute embedding-space feature importance via |query - support|.

        Returns per-dimension importance scores [D] showing which embedding
        dimensions most differentiate the query from the reference support
        example. Higher values indicate dimensions that contribute more to
        the distance between the two embeddings.

        This is NOT a pixel-level saliency map — it operates entirely in
        embedding space. For true gradient-based saliency through the
        backbone, install torch and use compute_saliency_torch() (planned
        for a future release).

        Args:
            query_embedding: [D] original query embedding.
            support_embedding: [D] reference support embedding.

        Returns:
            [D] per-dimension importance scores normalized to [0, 1].
        """
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        support = np.asarray(support_embedding, dtype=np.float32).reshape(-1)
        importance = np.abs(query - support)
        max_val = float(importance.max())
        if max_val > 1e-8:
            importance = importance / max_val
        return cast(FloatArray, importance.astype(np.float32))

    @staticmethod
    def compute_saliency(
        query_embedding: FloatArray,
        support_embeddings: FloatArray,
        support_labels: LabelArray,
        predicted_label: str | int,
    ) -> dict[str, Any]:
        """Generate a saliency-like explanation without requiring torch.

        Instead of pixel-level saliency (which requires gradient access
        through the backbone), this method provides channel-level feature
        importance derived from the embedding space.

        Args:
            query_embedding: [D] query embedding.
            support_embeddings: [N, D] support set.
            support_labels: [N] class labels.
            predicted_label: Top-1 prediction.

        Returns:
            Dictionary with saliency information.
        """
        query = np.asarray(query_embedding, dtype=np.float32)
        support = np.asarray(support_embeddings, dtype=np.float32)

        # Compute feature importance as the absolute difference between
        # query and class prototype, normalized
        same_mask = support_labels == predicted_label
        prototype = (
            support[same_mask].mean(axis=0) if same_mask.any() else support.mean(axis=0)
        )

        feature_importance = np.abs(query - prototype)
        feature_importance = feature_importance / (feature_importance.sum() + 1e-8)

        # Get top contributing feature dimensions
        top_n = min(10, len(feature_importance))
        top_indices = np.argsort(feature_importance)[-top_n:][::-1]

        return {
            "method": "embedding_space_attribution",
            "feature_importance": feature_importance.tolist(),
            "top_dimensions": [
                {"dim": int(idx), "importance": float(feature_importance[idx])}
                for idx in top_indices
            ],
            "prototype_distance": float(np.linalg.norm(query - prototype)),
        }
