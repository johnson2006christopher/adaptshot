"""Model definitions and embedding components."""

from .embedding import compute_cosine_similarity, extract_embedding
from .network import create_fewshot_model

__all__ = [
    "create_fewshot_model",
    "extract_embedding",
    "compute_cosine_similarity",
]
