"""Core module tests for embedding and similarity behavior."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.core.embedding import extract_batch_embeddings, extract_embedding
from src.core.similarity import HybridSimilarityIndex, cosine_similarity
from src.models.network import create_fewshot_model

__all__ = [
    "test_extract_embedding_shape",
    "test_extract_embedding_dtype",
    "test_extract_batch_embeddings_shape_and_size",
    "test_cosine_similarity_identity",
    "test_cosine_similarity_opposite_direction",
    "test_index_add_len",
    "test_index_search_shapes",
    "test_index_search_returns_sorted_scores_numpy_mode",
    "test_index_rejects_bad_dimensions",
    "test_embedding_deterministic_for_same_input",
]


def _model() -> torch.nn.Module:
    torch.manual_seed(42)
    return create_fewshot_model(num_classes=5, device=torch.device("cpu"))


def test_extract_embedding_shape() -> None:
    model = _model()
    sample = torch.randn(3, 128, 128)
    embedding = extract_embedding(model, sample)
    assert embedding.shape == (512,)


def test_extract_embedding_dtype() -> None:
    model = _model()
    sample = torch.randn(3, 128, 128)
    embedding = extract_embedding(model, sample)
    assert embedding.dtype == np.float32


def test_extract_batch_embeddings_shape_and_size() -> None:
    model = _model()
    batch = torch.randn(4, 3, 128, 128)
    embeddings, size = extract_batch_embeddings(model, batch)
    assert embeddings.shape == (4, 512)
    assert size == 4


def test_cosine_similarity_identity() -> None:
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    support = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    sim = cosine_similarity(query, support)
    assert np.allclose(sim, np.array([1.0], dtype=np.float32), atol=1e-6)


def test_cosine_similarity_opposite_direction() -> None:
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    support = np.array([[-1.0, 0.0, 0.0]], dtype=np.float32)
    sim = cosine_similarity(query, support)
    assert np.allclose(sim, np.array([-1.0], dtype=np.float32), atol=1e-6)


def test_index_add_len() -> None:
    index = HybridSimilarityIndex(dim=3, use_faiss=False)
    index.add(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32))
    assert len(index) == 2


def test_index_search_shapes() -> None:
    index = HybridSimilarityIndex(dim=3, use_faiss=False)
    index.add(np.array([[1.0, 0.0, 0.0], [0.2, 0.9, 0.0]], dtype=np.float32))
    scores, indices = index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=2)
    assert scores.shape == (2,)
    assert indices.shape == (2,)


def test_index_search_returns_sorted_scores_numpy_mode() -> None:
    index = HybridSimilarityIndex(dim=3, use_faiss=False)
    index.add(np.array([[1.0, 0.0, 0.0], [0.1, 0.9, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32))
    scores, _ = index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=3)
    assert np.all(scores[:-1] >= scores[1:])


def test_index_rejects_bad_dimensions() -> None:
    index = HybridSimilarityIndex(dim=4, use_faiss=False)
    with pytest.raises(ValueError):
        index.add(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))


def test_embedding_deterministic_for_same_input() -> None:
    model = _model()
    sample = torch.ones(3, 128, 128)
    emb1 = extract_embedding(model, sample)
    emb2 = extract_embedding(model, sample)
    assert np.allclose(emb1, emb2)
