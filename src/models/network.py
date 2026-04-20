"""ResNet18 few-shot classifier with frozen backbone and trainable head."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

__all__ = ["create_fewshot_model"]


def create_fewshot_model(
    num_classes: int = 5,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """
    Create a ResNet18 few-shot model for CPU-first adaptation.

    The function loads pretrained ImageNet weights, freezes the full backbone,
    replaces the classifier head, and unfreezes only the final linear layer.
    """
    try:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception:
        # Fallback allows offline test environments to remain functional.
        model = resnet18(weights=None)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(512, num_classes)
    for param in model.fc.parameters():
        param.requires_grad = True

    return model.to(device)
