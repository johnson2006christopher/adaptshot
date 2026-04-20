"""CPU-optimized embedding extraction utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

__all__ = ["extract_embedding", "extract_batch_embeddings"]


def _forward_to_avgpool(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Run a ResNet-style forward pass to the avgpool tensor."""
    features = model.conv1(inputs)
    features = model.bn1(features)
    features = model.relu(features)
    features = model.maxpool(features)
    features = model.layer1(features)
    features = model.layer2(features)
    features = model.layer3(features)
    features = model.layer4(features)
    features = model.avgpool(features)
    return torch.flatten(features, 1)


def extract_embedding(model: nn.Module, img_tensor: torch.Tensor) -> np.ndarray:
    """
    Extract one 512-d embedding from a ResNet18-style model.

    Args:
        model: Backbone model with ResNet-style modules.
        img_tensor: Tensor shape [3,H,W] or [1,3,H,W].

    Returns:
        np.ndarray: Embedding shape (512,) on CPU.
    """
    model.eval()
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    if img_tensor.dim() != 4:
        raise ValueError("img_tensor must be [3,H,W] or [1,3,H,W].")

    device = next(model.parameters()).device
    inputs = img_tensor.to(device=device, non_blocking=False)
    with torch.no_grad():
        embedding = _forward_to_avgpool(model, inputs)
    return embedding[0].detach().cpu().numpy().astype(np.float32)


def extract_batch_embeddings(model: nn.Module, batch: torch.Tensor) -> Tuple[np.ndarray, int]:
    """Extract a batch of embeddings and return `(embeddings, batch_size)`."""
    model.eval()
    if batch.dim() != 4:
        raise ValueError("batch must be [B,3,H,W].")
    device = next(model.parameters()).device
    inputs = batch.to(device=device, non_blocking=False)
    with torch.no_grad():
        embeddings = _forward_to_avgpool(model, inputs)
    out = embeddings.detach().cpu().numpy().astype(np.float32)
    return out, out.shape[0]
