"""Unit tests for core/similarity.py."""

import numpy as np
import pytest

from src.adaptshot.core.similarity import (
    cosine_similarity_numpy,
    find_nearest_neighbor,
    FAISS_AVAILABLE,
)


def test_cosine_similarity_range():
    """Ensure cosine similarity returns values strictly within [-1, 1]."""
    np.random.seed(42)
    query = np.random.randn(512)
    support = np.random.randn(10, 512)
    sims = cosine_similarity_numpy(query, support)
    assert np.all(sims >= -1.0 - 1e-7) and np.all(sims <= 1.0 + 1e-7), "Similarities out of expected range"


def test_cosine_similarity_perfect_match():
    """Verify that identical vectors yield a similarity of 1.0 and orthogonal yield 0.0."""
    vec = np.array([1.0, 2.0, 3.0])
    support = np.vstack([vec, np.array([0.0, 0.0, 0.0])])
    sims = cosine_similarity_numpy(vec, support)
    assert np.isclose(sims[0], 1.0, atol=1e-7)
    assert np.isclose(sims[1], 0.0, atol=1e-7)


def test_cosine_similarity_batch():
    """Test batched query input returns correct [B, N] shape."""
    query = np.random.randn(3, 512)  # 3 queries
    support = np.random.randn(5, 512)  # 5 support examples
    sims = cosine_similarity_numpy(query, support)
    assert sims.shape == (3, 5), f"Expected shape (3, 5), got {sims.shape}"


def test_find_nearest_neighbor_correctness():
    """Ensure nearest neighbor returns the exact match with confidence ~1.0."""
    target = np.ones(512)
    distractor = np.random.randn(512)
    support = np.vstack([target, distractor])
    labels = np.array(["class_A", "class_B"])

    pred, conf, idx = find_nearest_neighbor(target, support, labels, use_faiss=False)
    assert pred == "class_A"
    assert idx == 0
    assert np.isclose(conf, 1.0, atol=1e-6)


def test_find_nearest_neighbor_fallback():
    """Verify that use_faiss=False correctly routes to NumPy even if FAISS is installed."""
    query = np.array([0.0, 1.0, 0.0])
    support = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    labels = np.array(["x", "y", "z"])

    pred, conf, idx = find_nearest_neighbor(query, support, labels, use_faiss=False)
    assert pred == "y"
    assert idx == 1
    assert np.isclose(conf, 1.0, atol=1e-6)


def test_faiss_availability_flag():
    """Ensure FAISS availability is correctly reported as a boolean at module load."""
    assert isinstance(FAISS_AVAILABLE, bool)
    # If FAISS is missing, the library gracefully falls back to NumPy.
    # If installed, `use_faiss=True` will activate the accelerated path.