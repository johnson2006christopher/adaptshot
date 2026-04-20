"""Correction-Aware EWC (CA-EWC) for continual few-shot adaptation."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

__all__ = ["compute_correction_aware_fisher", "compute_ca_ewc_penalty"]


def _trainable_params(model: nn.Module) -> Dict[str, nn.Parameter]:
    """Return trainable named parameters only."""
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def compute_correction_aware_fisher(
    model: nn.Module,
    loader: DataLoader,
    correction_confidence: Optional[Dict[int, float]] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """
    Compute empirical diagonal Fisher weighted by feedback confidence.

    If per-batch confidence is given, batch `i` gets weight
    `correction_confidence.get(i, 1.0)`.
    """
    model.eval()
    params = _trainable_params(model)
    fisher = {name: torch.zeros_like(p, device=device) for name, p in params.items()}
    criterion = nn.CrossEntropyLoss()
    total_weight = 0.0

    for batch_idx, (inputs, labels) in enumerate(loader):
        batch_weight = 1.0
        if correction_confidence is not None:
            batch_weight = float(correction_confidence.get(batch_idx, 1.0))
            batch_weight = min(max(batch_weight, 0.0), 1.5)

        inputs = inputs.to(device=device, non_blocking=False)
        labels = labels.to(device=device, non_blocking=False)
        model.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        for name, param in params.items():
            if param.grad is not None:
                fisher[name] += batch_weight * (param.grad.detach() ** 2)
        total_weight += batch_weight

    if total_weight <= 0.0:
        raise ValueError("Loader produced no weighted batches for Fisher computation.")

    for name in fisher:
        fisher[name] /= total_weight
    return fisher


def compute_ca_ewc_penalty(
    model: nn.Module,
    fisher_dict: Dict[str, torch.Tensor],
    old_params: Dict[str, torch.Tensor],
    correction_weights: Optional[Dict[str, float]] = None,
    lam: float = 0.1,
) -> torch.Tensor:
    """
    Compute CA-EWC penalty where each parameter can have custom weight.

    Penalty: lam * sum_j w_j * F_j * (theta_j - theta*_j)^2
    """
    params = _trainable_params(model)
    penalty = torch.tensor(0.0, device=torch.device("cpu"))

    for name, param in params.items():
        if name not in fisher_dict or name not in old_params:
            continue
        w = 1.0
        if correction_weights is not None:
            w = float(correction_weights.get(name, 1.0))
        w = min(max(w, 0.0), 2.0)
        fisher = fisher_dict[name].to(torch.device("cpu"))
        old = old_params[name].to(torch.device("cpu"))
        penalty = penalty + (w * torch.sum(fisher * (param - old) ** 2))

    return penalty * float(lam)
