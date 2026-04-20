"""Phase-4 tests for replay buffer, EWC, and incremental fine-tuning."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from src.models.network import create_fewshot_model
from src.training.feedback import ReplayBuffer
from src.training.incremental import (
    compute_diagonal_fisher,
    compute_ewc_penalty,
    incremental_fine_tune,
)

__all__ = [
    "test_replay_buffer_fifo_capacity",
    "test_compute_ewc_penalty_is_non_negative",
    "test_compute_diagonal_fisher_keys",
    "test_incremental_fine_tune_reduces_loss",
]


def _build_buffer(num_samples: int = 40, num_classes: int = 5) -> ReplayBuffer:
    """Create a deterministic synthetic replay buffer."""
    torch.manual_seed(42)
    np.random.seed(42)

    buffer = ReplayBuffer(capacity=max(num_samples, 1))
    prototypes = np.eye(num_classes, 512, dtype=np.float32)

    for idx in range(num_samples):
        label = idx % num_classes
        embedding = prototypes[label] + 0.01 * np.random.randn(512).astype(np.float32)
        image = Image.fromarray(np.full((16, 16, 3), fill_value=label * 30, dtype=np.uint8))
        buffer.add(embedding=embedding, label=label, image=image)
    return buffer


def _capture_old_head_params(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Snapshot classifier head parameters for EWC reference."""
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name in {"fc.weight", "fc.bias"}
    }


def test_replay_buffer_fifo_capacity(capsys):
    """ReplayBuffer should prune oldest samples when capacity is exceeded."""
    buffer = ReplayBuffer(capacity=3)
    for label in [0, 1, 2, 3]:
        buffer.add(
            embedding=np.array([float(label)], dtype=np.float32),
            label=label,
            image=Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)),
        )

    _, labels, _ = buffer.get_batches()
    assert len(buffer) == 3
    assert labels == [1, 2, 3]
    print("✅ FIFO pruning enforced")
    captured = capsys.readouterr()
    assert "✅ FIFO pruning enforced" in captured.out


def test_compute_ewc_penalty_is_non_negative(capsys):
    """EWC penalty must be a non-negative scalar."""
    torch.manual_seed(7)
    model = create_fewshot_model(num_classes=5, device=torch.device("cpu"))
    old_params = _capture_old_head_params(model)
    fisher_dict = {name: torch.ones_like(value) for name, value in old_params.items()}

    with torch.no_grad():
        model.fc.weight.add_(0.1)

    penalty = compute_ewc_penalty(model, fisher_dict=fisher_dict, old_params=old_params, lam=0.1)
    assert penalty.ndim == 0
    assert float(penalty.item()) >= 0.0
    print("✅ EWC penalty >= 0")
    captured = capsys.readouterr()
    assert "✅ EWC penalty >= 0" in captured.out


def test_compute_diagonal_fisher_keys():
    """Fisher estimate should include classifier head keys only."""
    torch.manual_seed(9)
    model = create_fewshot_model(num_classes=5, device=torch.device("cpu"))
    x = torch.randn(12, 3, 128, 128)
    y = torch.randint(low=0, high=5, size=(12,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False, num_workers=0)

    fisher = compute_diagonal_fisher(model=model, loader=loader)
    assert set(fisher.keys()) == {"fc.weight", "fc.bias"}
    assert fisher["fc.weight"].shape == model.fc.weight.shape
    assert fisher["fc.bias"].shape == model.fc.bias.shape


def test_incremental_fine_tune_reduces_loss(capsys):
    """Incremental adaptation should reduce classifier-head loss on replay data."""
    torch.manual_seed(11)
    np.random.seed(11)

    model = create_fewshot_model(num_classes=5, device=torch.device("cpu"))
    buffer = _build_buffer(num_samples=50, num_classes=5)
    embeddings, labels, _ = buffer.get_batches()

    x = torch.from_numpy(np.stack(embeddings).astype(np.float32))
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64))

    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        before_loss = float(criterion(model.fc(x), y).item())

    avg_loss = incremental_fine_tune(
        model=model,
        buffer=buffer,
        fisher_dict=None,
        old_params=None,
        lam=0.1,
        lr=1e-2,
        epochs=8,
    )

    with torch.no_grad():
        after_loss = float(criterion(model.fc(x), y).item())

    assert avg_loss >= 0.0
    assert after_loss < before_loss
    print("✅ Loss decreased after incremental update")
    captured = capsys.readouterr()
    assert "✅ Loss decreased after incremental update" in captured.out
