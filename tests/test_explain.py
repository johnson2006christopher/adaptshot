"""Tests for XAI explainability module (v0.2.0)."""

from __future__ import annotations

import numpy as np
import pytest

from src.adaptshot import (
    ExplainabilityEngine,
    ExplanationResult,
    FeatureAttribution,
)


class TestFeatureAttribution:
    """Tests for the FeatureAttribution dataclass."""

    def test_defaults_are_empty(self) -> None:
        attr = FeatureAttribution()
        assert attr.top_k_indices == []
        assert attr.top_k_labels == []
        assert attr.top_k_similarities == []
        assert attr.attribution_summary == ""

    def test_to_dict_serializes_correctly(self) -> None:
        attr = FeatureAttribution(
            top_k_indices=[0, 1],
            top_k_labels=["cat", "dog"],
            top_k_similarities=[0.9, 0.3],
            attribution_summary="cat (0.90)",
        )
        d = attr.to_dict()
        assert d["top_k_indices"] == [0, 1]
        assert d["top_k_labels"] == ["cat", "dog"]
        assert d["attribution_summary"] == "cat (0.90)"


class TestExplanationResult:
    """Tests for the ExplanationResult dataclass."""

    def test_to_dict_includes_all_fields(self) -> None:
        result = ExplanationResult(
            predicted_label="cat",
            feature_attribution=FeatureAttribution(top_k_labels=["cat"]),
            confidence_decomposition={},
            counterfactual={},
            summary="Test summary",
        )
        d = result.to_dict()
        assert d["predicted_label"] == "cat"
        assert "feature_attribution" in d
        assert "summary" in d


class TestExplainabilityEngine:
    """Tests for the ExplainabilityEngine class."""

    @pytest.fixture
    def engine(self) -> ExplainabilityEngine:
        return ExplainabilityEngine(k_neighbors=3)

    @pytest.fixture
    def synthetic_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create support embeddings and a query."""
        rng = np.random.RandomState(42)
        dim = 32
        # Support: 5 examples each of 2 classes
        class0 = rng.randn(5, dim).astype(np.float32) * 0.5
        class0[:, 0] -= 2.0
        class1 = rng.randn(5, dim).astype(np.float32) * 0.5
        class1[:, 0] += 2.0
        support = np.vstack([class0, class1])
        labels = np.array(["cat"] * 5 + ["dog"] * 5, dtype=object)
        # Query near class "cat"
        query = np.array([-2.0] + [0.0] * (dim - 1), dtype=np.float32)
        return query, support, labels

    def test_explain_returns_correct_prediction(
        self, engine: ExplainabilityEngine, synthetic_data: tuple
    ) -> None:
        """Explain should identify the correct predicted class."""
        query, support, labels = synthetic_data
        result = engine.explain(
            query_embedding=query,
            support_embeddings=support,
            support_labels=labels,
            predicted_label="cat",
            raw_confidence=0.85,
            calibrated_confidence=0.82,
            act_action="ACCEPT",
            is_ood=False,
        )
        assert result.predicted_label == "cat"
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0

    def test_explain_includes_feature_attribution(
        self, engine: ExplainabilityEngine, synthetic_data: tuple
    ) -> None:
        """Feature attribution should identify supporting examples."""
        query, support, labels = synthetic_data
        result = engine.explain(
            query_embedding=query,
            support_embeddings=support,
            support_labels=labels,
            predicted_label="cat",
            raw_confidence=0.85,
            calibrated_confidence=0.82,
            act_action="ACCEPT",
            is_ood=False,
        )
        assert len(result.feature_attribution.top_k_indices) > 0
        # Top support examples should all be "cat" class
        for label in result.feature_attribution.top_k_labels:
            assert label == "cat"

    def test_explain_includes_confidence_decomposition(
        self, engine: ExplainabilityEngine, synthetic_data: tuple
    ) -> None:
        """Confidence decomposition should contain expected components."""
        query, support, labels = synthetic_data
        result = engine.explain(
            query_embedding=query,
            support_embeddings=support,
            support_labels=labels,
            predicted_label="cat",
            raw_confidence=0.85,
            calibrated_confidence=0.82,
            act_action="ACCEPT",
            is_ood=False,
        )
        assert "raw_confidence" in result.confidence_decomposition
        assert "calibrated_confidence" in result.confidence_decomposition

    def test_explain_includes_counterfactual(
        self, engine: ExplainabilityEngine, synthetic_data: tuple
    ) -> None:
        """Counterfactual analysis should identify nearest alternative class."""
        query, support, labels = synthetic_data
        result = engine.explain(
            query_embedding=query,
            support_embeddings=support,
            support_labels=labels,
            predicted_label="cat",
            raw_confidence=0.85,
            calibrated_confidence=0.82,
            act_action="ACCEPT",
            is_ood=False,
        )
        assert "nearest_alternative" in result.counterfactual
        # Nearest alternative should be "dog"
        assert result.counterfactual["nearest_alternative"] == "dog"

    def test_explain_rejects_empty_support(self, engine: ExplainabilityEngine) -> None:
        """Empty support set raises an error."""
        query = np.ones(32, dtype=np.float32)
        with pytest.raises(ValueError, match="empty"):
            engine.explain(
                query_embedding=query,
                support_embeddings=np.empty((0, 32), dtype=np.float32),
                support_labels=np.array([], dtype=object),
                predicted_label="cat",
                raw_confidence=0.5,
                calibrated_confidence=0.5,
                act_action="ACCEPT",
                is_ood=False,
            )
