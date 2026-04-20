"""Phase-2 tests for deterministic few-shot data utilities."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data import (
    FewShotBatchSampler,
    create_fewshot_loader,
    get_conservative_train_transforms,
    get_eval_transforms,
)

__all__ = [
    "test_sampler_episode_size",
    "test_sampler_is_deterministic",
    "test_augmentation_shapes",
    "test_loader_shapes_and_labels",
]


class TinyImageDataset(Dataset):
    """Small synthetic dataset with torchvision-like target attributes."""

    def __init__(self, num_classes: int = 5, samples_per_class: int = 20) -> None:
        self.transform = None
        self.targets = [
            class_idx
            for class_idx in range(num_classes)
            for _ in range(samples_per_class)
        ]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        value = index % 255
        image = np.full((32, 32, 3), fill_value=value, dtype=np.uint8)
        pil_image = Image.fromarray(image)
        tensor_image = self.transform(pil_image) if self.transform is not None else pil_image
        return tensor_image, int(self.targets[index])


def test_sampler_episode_size() -> None:
    """Episode should contain exactly N-way * K-shot indices."""
    dataset = TinyImageDataset(num_classes=5, samples_per_class=20)
    sampler = FewShotBatchSampler(dataset=dataset, n_way=3, k_shot=4, num_episodes=2, seed=42)
    first_episode = next(iter(sampler))
    assert len(first_episode) == 12


def test_sampler_is_deterministic() -> None:
    """Two samplers with the same seed should emit identical episodes."""
    dataset = TinyImageDataset(num_classes=5, samples_per_class=20)
    sampler_a = FewShotBatchSampler(dataset=dataset, n_way=3, k_shot=4, num_episodes=3, seed=99)
    sampler_b = FewShotBatchSampler(dataset=dataset, n_way=3, k_shot=4, num_episodes=3, seed=99)
    assert list(iter(sampler_a)) == list(iter(sampler_b))


def test_augmentation_shapes() -> None:
    """Train and eval transforms should output normalized tensors of same shape."""
    sample_image = Image.fromarray(np.zeros((40, 40, 3), dtype=np.uint8))
    train_tensor = get_conservative_train_transforms(img_size=128)(sample_image)
    eval_tensor = get_eval_transforms(img_size=128)(sample_image)
    assert train_tensor.shape == (3, 128, 128)
    assert eval_tensor.shape == (3, 128, 128)


def test_loader_shapes_and_labels() -> None:
    """Few-shot loader should emit one deterministic episode per batch."""
    dataset = TinyImageDataset(num_classes=5, samples_per_class=20)
    loader = create_fewshot_loader(
        dataset=dataset,
        n_way=2,
        k_shot=3,
        num_episodes=1,
        mode="train",
        img_size=128,
        seed=42,
        num_workers=0,
    )

    images, labels = next(iter(loader))
    assert images.shape == (6, 3, 128, 128)
    assert labels.shape == (6,)
    assert labels.unique().numel() == 2
