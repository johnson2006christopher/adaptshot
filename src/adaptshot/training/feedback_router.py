"""Human-in-the-loop feedback routing and replay buffer management.

Wires domain-expert corrections into the learning loop, maintains
calibration state, and triggers correction-aware fine-tuning when
accumulated feedback exceeds a configurable threshold.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Correction:
    """Structured representation of a single human feedback event."""
    image_path: str
    predicted_label: str | int
    corrected_label: str | int
    raw_confidence: float
    confidence_weight: float = 1.0  # Human's confidence in their correction [0.0, 1.0]
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class FeedbackRouter:
    """
    Routes human corrections into the continual learning pipeline.

    Manages the replay buffer, updates online calibration, and triggers
    CA-EWC head-only fine-tuning when enough corrections accumulate.
    Designed for CPU-first, low-latency operation on edge devices.
    """

    def __init__(
        self,
        buffer_capacity: int = 100,
        fine_tune_trigger_threshold: int = 5,
        calibrator: Any | None = None,
        finetune_fn: Callable[[list[Correction]], None] | None = None,
    ) -> None:
        """
        Args:
            buffer_capacity: Maximum number of corrections to retain in replay memory
            fine_tune_trigger_threshold: Number of pending corrections before triggering CA-EWC
            calibrator: CalibrationEngine instance for online ECE/temperature updates
            finetune_fn: Callback that executes CA-EWC head-only optimization
        """
        self.buffer_capacity = buffer_capacity
        self.fine_tune_trigger_threshold = fine_tune_trigger_threshold
        self.calibrator = calibrator
        self.finetune_fn = finetune_fn

        self.buffer: list[Correction] = []
        self.pending_corrections: list[Correction] = []
        self.total_corrections = 0

    def route_feedback(self, correction: Correction) -> dict[str, Any]:
        """
        Process a single human correction and route to buffer/calibration/fine-tuning.

        Args:
            correction: Structured correction event from UI or API

        Returns:
            Dictionary summarizing routing actions taken
        """
        if correction.timestamp == 0.0:
            correction.timestamp = time.time()

        self.total_corrections += 1
        self.pending_corrections.append(correction)
        self._update_buffer(correction)

        # Update online calibration state
        calibration_updated = False
        calibration_summary: dict[str, float] = {}
        if self.calibrator is not None:
            self.calibrator.update(
                raw_confidence=correction.raw_confidence,
                predicted_label=correction.predicted_label,
                true_label=correction.corrected_label,
            )
            calibration_updated = True
            if hasattr(self.calibrator, "calibration_summary"):
                calibration_summary = self.calibrator.calibration_summary()

        # Trigger CA-EWC fine-tuning if threshold is met
        fine_tuned = False
        if len(self.pending_corrections) >= self.fine_tune_trigger_threshold:
            if self.finetune_fn is not None:
                fine_tuned = self._trigger_finetune()
            else:
                logger.debug("Fine-tuning threshold met but no finetune_fn bound.")

        return {
            "buffer_size": len(self.buffer),
            "pending_corrections": len(self.pending_corrections),
            "calibration_updated": calibration_updated,
            "calibration_summary": calibration_summary,
            "fine_tuned": fine_tuned,
            "total_corrections": self.total_corrections,
        }

    def _update_buffer(self, correction: Correction) -> None:
        """Append correction to replay buffer, enforcing capacity limit."""
        self.buffer.append(correction)
        if len(self.buffer) > self.buffer_capacity:
            # FIFO eviction for v0.1; UP-UGF scoring will replace this in v0.2
            self.buffer.pop(0)
            logger.debug("Buffer capacity reached. Evicted oldest correction.")

    def _trigger_finetune(self) -> bool:
        """Execute head-only CA-EWC fine-tuning on accumulated corrections."""
        if not self.finetune_fn or not self.pending_corrections:
            return False

        try:
            # Pass a copy to prevent mutation during optimization
            self.finetune_fn(list(self.pending_corrections))
            self.pending_corrections.clear()
            logger.info("CA-EWC fine-tuning triggered and pending queue cleared.")
            return True
        except Exception as e:  # noqa: BLE001 - finetune_fn is user-supplied; a
            # failure there must not propagate into the correction path.
            logger.error("CA-EWC fine-tuning failed: %s", e)
            return False

    def get_buffer(self) -> list[Correction]:
        """Return a shallow copy of the current replay buffer."""
        return list(self.buffer)

    def clear_buffer(self) -> None:
        """Reset replay buffer and pending correction queue."""
        self.buffer.clear()
        self.pending_corrections.clear()
        self.total_corrections = 0
