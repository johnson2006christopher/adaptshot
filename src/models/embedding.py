"""Embedding extraction and cosine similarity utilities."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

__all__ = ["extract_embedding", "compute_cosine_similarity"]


def extract_embedding(model: nn.Module, img_tensor: torch.Tensor) -> np.ndarray:
    """
    Extract a 512-dim embedding from the ResNet18 avgpool output.

    Args:
        model: ResNet-like model with conv blocks and avgpool.
        img_tensor: Input image tensor of shape [C, H, W] or [1, C, H, W].

    Returns:
        np.ndarray: Embedding vector with shape (512,).
    """
    model.eval()
    device = next(model.parameters()).device

    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    if img_tensor.dim() != 4:
        raise ValueError("img_tensor must have shape [C,H,W] or [B,C,H,W].")

    inputs = img_tensor.to(device=device, non_blocking=False)

    with torch.no_grad():
        features = model.conv1(inputs)
        features = model.bn1(features)
        features = model.relu(features)
        features = model.maxpool(features)
        features = model.layer1(features)
        features = model.layer2(features)
        features = model.layer3(features)
        features = model.layer4(features)
        features = model.avgpool(features)
        features = torch.flatten(features, 1)

    return features[0].detach().cpu().numpy()


def compute_cosine_similarity(query_emb: np.ndarray, support_embs: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between one query embedding and support embeddings.

    Args:
        query_emb: Query embedding of shape (D,).
        support_embs: Support embeddings of shape (N, D).

    Returns:
        np.ndarray: Similarities of shape (N,).
    """
    query = np.asarray(query_emb, dtype=np.float32).reshape(-1)
    support = np.asarray(support_embs, dtype=np.float32)

    if support.ndim != 2:
        raise ValueError("support_embs must have shape (N, D).")
    if support.shape[1] != query.shape[0]:
        raise ValueError("support_embs feature dimension must match query_emb.")

    query_norm = np.linalg.norm(query) + 1e-12
    support_norm = np.linalg.norm(support, axis=1, keepdims=True) + 1e-12

    query_unit = query / query_norm
    support_unit = support / support_norm

    similarities = support_unit @ query_unit
    return np.clip(similarities, -1.0, 1.0)
