"""Incremental fine-tuning and EWC utilities for classifier adaptation."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .feedback import ReplayBuffer

__all__ = [
    "compute_diagonal_fisher",
    "compute_ewc_penalty",
    "incremental_fine_tune",
]


def _get_head_parameters(model: nn.Module) -> Dict[str, nn.Parameter]:
    """Collect classifier head parameters supported by EWC."""
    head_params: Dict[str, nn.Parameter] = {}
    for name, parameter in model.named_parameters():
        if name in {"fc.weight", "fc.bias"}:
            head_params[name] = parameter
    missing = {"fc.weight", "fc.bias"} - set(head_params.keys())
    if missing:
        raise ValueError("Model must expose 'fc.weight' and 'fc.bias' parameters.")
    return head_params


def compute_diagonal_fisher(model: nn.Module, loader: DataLoader) -> Dict[str, torch.Tensor]:
    """
    Compute empirical diagonal Fisher for classifier head parameters only.

    The estimate uses the average squared gradient over the provided loader.
    """
    model.eval()
    head_params = _get_head_parameters(model)
    fisher = {name: torch.zeros_like(param) for name, param in head_params.items()}
    criterion = nn.CrossEntropyLoss()
    batches = 0

    for inputs, labels in loader:
        inputs = inputs.to(torch.device("cpu"), non_blocking=False)
        labels = labels.to(torch.device("cpu"), non_blocking=False)

        model.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        for name, parameter in head_params.items():
            if parameter.grad is None:
                continue
            fisher[name] += parameter.grad.detach().clone() ** 2
        batches += 1

    if batches == 0:
        raise ValueError("Loader must provide at least one batch.")

    for name in fisher:
        fisher[name] /= float(batches)
    return fisher


def compute_ewc_penalty(
    model: nn.Module,
    fisher_dict: Dict[str, torch.Tensor],
    old_params: Dict[str, torch.Tensor],
    lam: float = 0.1,
) -> torch.Tensor:
    """
    Compute EWC regularization penalty over classifier head parameters.

    Penalty form: lam * sum_i F_i * (theta_i - theta_i_old)^2
    """
    head_params = _get_head_parameters(model)
    penalty = torch.tensor(0.0, device=torch.device("cpu"))

    for name, parameter in head_params.items():
        if name not in fisher_dict or name not in old_params:
            raise ValueError(f"Missing Fisher or old parameter for '{name}'.")
        fisher = fisher_dict[name].to(torch.device("cpu"))
        theta_old = old_params[name].to(torch.device("cpu"))
        penalty = penalty + torch.sum(fisher * (parameter - theta_old) ** 2)

    return penalty * float(lam)


def incremental_fine_tune(
    model: nn.Module,
    buffer: ReplayBuffer,
    fisher_dict: Optional[Dict[str, torch.Tensor]] = None,
    old_params: Optional[Dict[str, torch.Tensor]] = None,
    lam: float = 0.1,
    lr: float = 1e-4,
    epochs: int = 10,
) -> float:
    """
    Incrementally fine-tune classifier head on replayed corrected examples.

    The adaptation uses only stored embeddings and optimizes `model.fc`.
    """
    if len(buffer) == 0:
        return 0.0
    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    embeddings, labels, _ = buffer.get_batches()
    x_np = np.stack(embeddings).astype(np.float32)
    y_np = np.asarray(labels, dtype=np.int64)

    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    running_loss = 0.0
    steps = 0

    for _ in range(epochs):
        for batch_embeddings, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model.fc(batch_embeddings.to(torch.device("cpu"), non_blocking=False))
            loss = criterion(logits, batch_labels.to(torch.device("cpu"), non_blocking=False))

            if fisher_dict is not None and old_params is not None:
                loss = loss + compute_ewc_penalty(
                    model=model,
                    fisher_dict=fisher_dict,
                    old_params=old_params,
                    lam=lam,
                )

            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu().item())
            steps += 1

    return running_loss / max(steps, 1)
