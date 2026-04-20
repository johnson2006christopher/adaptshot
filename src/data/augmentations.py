"""Conservative and deterministic image transformations for few-shot stability."""

from __future__ import annotations

import torchvision.transforms as transforms

__all__ = ["get_conservative_train_transforms", "get_eval_transforms"]


def get_conservative_train_transforms(
    img_size: int = 128,
    padding: int = 8,
    rotation_deg: int = 10,
    jitter_strength: float = 0.1,
) -> transforms.Compose:
    """
    Build conservative augmentation for few-shot training.

    The policy introduces mild invariances while preserving semantic signal
    in tiny-data settings.
    """
    return transforms.Compose(
        [
            transforms.Resize((img_size + padding * 2, img_size + padding * 2)),
            transforms.RandomCrop(img_size, padding=padding),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(rotation_deg),
            transforms.ColorJitter(
                brightness=jitter_strength,
                contrast=jitter_strength,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def get_eval_transforms(img_size: int = 128) -> transforms.Compose:
    """Build deterministic evaluation transforms (no random augmentation)."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
