"""Deterministic few-shot DataLoader wrapper (CPU-first design)."""

from __future__ import annotations

from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset

from .augmentations import get_conservative_train_transforms, get_eval_transforms
from .fewshot_sampler import FewShotBatchSampler

__all__ = ["create_fewshot_loader"]


def _worker_init(seed: int) -> Callable[[int], None]:
    """Create deterministic worker initialization function."""

    def init_fn(worker_id: int) -> None:
        torch.manual_seed(seed + worker_id)

    return init_fn


def create_fewshot_loader(
    dataset: Dataset,
    n_way: int = 5,
    k_shot: int = 10,
    num_episodes: int = 100,
    mode: str = "train",
    img_size: int = 128,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a deterministic few-shot DataLoader.

    Args:
        dataset: Torchvision-style dataset with targets/labels attribute.
        n_way: Number of classes per episode.
        k_shot: Number of samples per class per episode.
        num_episodes: Number of episodes to generate.
        mode: Either `train` or `eval`.
        img_size: Target input spatial size.
        seed: Random seed for episode sampling and workers.
        num_workers: Keep at 0 for strict CPU determinism.

    Returns:
        DataLoader: Yields one episode per iteration with N*K examples.
    """
    if mode not in {"train", "eval"}:
        raise ValueError("mode must be either 'train' or 'eval'.")

    if not hasattr(dataset, "transform"):
        raise ValueError("Dataset must define a mutable 'transform' attribute.")

    dataset.transform = (
        get_conservative_train_transforms(img_size=img_size)
        if mode == "train"
        else get_eval_transforms(img_size=img_size)
    )

    sampler = FewShotBatchSampler(
        dataset=dataset,
        n_way=n_way,
        k_shot=k_shot,
        num_episodes=num_episodes,
        seed=seed,
    )

    return DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        worker_init_fn=_worker_init(seed),
    )
