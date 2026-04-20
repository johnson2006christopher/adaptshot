"""Deterministic N-way K-shot episode sampler for few-shot learning."""

from __future__ import annotations

from typing import Dict, Iterator, List

import numpy as np
from torch.utils.data import Dataset, Sampler

__all__ = ["FewShotBatchSampler"]


class FewShotBatchSampler(Sampler[List[int]]):
    """
    Yield N-way K-shot episode index lists.

    The sampler is deterministic when initialized with a fixed seed.
    It is compatible with `DataLoader(batch_sampler=sampler)`.
    """

    def __init__(
        self,
        dataset: Dataset,
        n_way: int = 5,
        k_shot: int = 10,
        num_episodes: int = 100,
        seed: int = 42,
    ) -> None:
        self.n_way = n_way
        self.k_shot = k_shot
        self.num_episodes = num_episodes
        self.rng = np.random.RandomState(seed)

        if hasattr(dataset, "targets"):
            targets = dataset.targets
        elif hasattr(dataset, "labels"):
            targets = dataset.labels
        else:
            raise ValueError("Dataset must have 'targets' or 'labels' attribute.")

        self.class_to_indices: Dict[int, List[int]] = {}
        for idx, label in enumerate(targets):
            self.class_to_indices.setdefault(int(label), []).append(idx)

        self.classes = sorted(self.class_to_indices.keys())

        if n_way > len(self.classes):
            raise ValueError("n_way exceeds number of available classes.")

        if any(len(indices) < k_shot for indices in self.class_to_indices.values()):
            raise ValueError("k_shot exceeds available samples in one or more classes.")

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.num_episodes):
            selected_classes = self.rng.choice(self.classes, size=self.n_way, replace=False)
            episode_indices: List[int] = []

            for cls in selected_classes:
                cls_indices = self.class_to_indices[int(cls)]
                sampled = self.rng.choice(cls_indices, size=self.k_shot, replace=False)
                episode_indices.extend(sampled.tolist())

            yield episode_indices

    def __len__(self) -> int:
        return self.num_episodes
