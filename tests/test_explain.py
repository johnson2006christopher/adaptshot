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
        assert attr.index == 0
        assert attr.label == ""
        assert attr.weight == 0.0
        assert attr.distance == 0.0
        assert attr.is_same_class is False

    def test_custom_values(self) -> None:
        attr = FeatureAttribution(
            index=5, label="cat", weight=0.8, distance=0.2, is_same_class=True
        )
        assert attr.index == 5
        assert attr.label == "cat"
        assert attr.weight == 0.8
        assert attr.distance == 0.2
        assert attr.is_same_class is True


class TestExplanationResult:
    """Tests for the ExplanationResult dataclass."""

    def test_defaults_are_empty(self) -> None:
        result = ExplanationResult()
        assert result.prediction == ""
        assert result.attributions == []
        assert result.summary == ""

    def test_to_dict_includes_all_fields(self) -> None:
        attr = FeatureAttribution(index=0, label="cat", weight=0.9)
        result = ExplanationResult(
            prediction="cat",
            attributions=[attr],
            summary="Test summary",
        )
        d = result.to_dict()
        assert d["prediction"] == "cat"
        assert "attributions" in d
        assert "summary" in d
        assert len(d["attributions"]) == 1
        assert d["attributions"][0]["label"] == "cat"


class TestExplainabilityEngine:
    """Tests for the ExplainabilityEngine class."""

    @pytest.fixture
    def engine(self) -> ExplainabilityEngine:
        return ExplainabilityEngine(top_k_attributions=3)

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
        assert result.prediction == "cat"
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0

    def test_explain_includes_attributions(
        self, engine: ExplainabilityEngine, synthetic_data: tuple
    ) -> None:
        """Feature attributions should identify supporting examples."""
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
        assert len(result.attributions) > 0
        # Top support examples should all be "cat" class
        for attr in result.attributions:
            assert attr.label == "cat"

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
        assert result.confidence_decomposition is not None
        assert result.confidence_decomposition.raw_similarity > 0.0

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
        assert result.counterfactual is not None
        assert result.counterfactual.counterfactual_class == "dog"

    def test_explain_handles_empty_support(self, engine: ExplainabilityEngine) -> None:
        """Empty support set returns empty attributions without error."""
        query = np.ones(32, dtype=np.float32)
        result = engine.explain(
            query_embedding=query,
            support_embeddings=np.empty((0, 32), dtype=np.float32),
            support_labels=np.array([], dtype=object),
            predicted_label="cat",
            raw_confidence=0.5,
            calibrated_confidence=0.5,
            act_action="ACCEPT",
            is_ood=False,
        )
        assert result.attributions == []
        assert result.prediction == "cat"
