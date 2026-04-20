"""Feedback routing and replay buffer utilities for continual learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np

__all__ = ["ReplayItem", "ReplayBuffer", "route_feedback"]


@dataclass
class ReplayItem:
    """Single replay entry created from user feedback."""

    embedding: np.ndarray
    label: int
    image: Any


class ReplayBuffer:
    """FIFO replay buffer storing corrected samples for incremental updates."""

    def __init__(self, capacity: int = 100) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = capacity
        self._items: List[ReplayItem] = []

    def add(self, embedding: np.ndarray, label: int, image: Any) -> None:
        """
        Add one corrected sample and enforce FIFO pruning.

        Args:
            embedding: Feature embedding vector for the sample.
            label: Correct class label.
            image: Original image payload (e.g., PIL image or tensor).
        """
        item = ReplayItem(
            embedding=np.asarray(embedding, dtype=np.float32).copy(),
            label=int(label),
            image=image,
        )
        self._items.append(item)
        if len(self._items) > self.capacity:
            overflow = len(self._items) - self.capacity
            self._items = self._items[overflow:]

    def get_batches(self) -> Tuple[List[np.ndarray], List[int], List[Any]]:
        """Return embeddings, labels, and images as parallel lists."""
        embeddings = [item.embedding for item in self._items]
        labels = [item.label for item in self._items]
        images = [item.image for item in self._items]
        return embeddings, labels, images

    def __len__(self) -> int:
        """Return number of samples stored in the replay buffer."""
        return len(self._items)


def route_feedback(
    buffer: ReplayBuffer,
    embedding: np.ndarray,
    predicted_label: int,
    corrected_label: int,
    image: Any,
) -> bool:
    """
    Route user feedback into replay memory when correction is needed.

    Returns:
        bool: True if a correction was added to the replay buffer.
    """
    if int(predicted_label) == int(corrected_label):
        return False
    buffer.add(embedding=embedding, label=int(corrected_label), image=image)
    return True
