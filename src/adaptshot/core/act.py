"""Adaptive Confidence Thresholding (ACT) engine for few-shot predictions.

Dynamically adjusts per-class decision thresholds based on real-time
correction history and model uncertainty, reducing false acceptances
by requesting human feedback when the model is genuinely unsure.
"""

import logging
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ACTEngine:
    """
    Adaptive Confidence Thresholding engine.

    Maintains a dynamic threshold τ_k for each class k that adapts based on:
    - Historical correction rates (incorrect vs. correct)
    - Model uncertainty signals (entropy/ECE proxies)
    - Configurable cost of requesting human feedback (γ)

    The engine implements an exponential moving average update rule to
    prevent oscillation while remaining responsive to distribution shift.
    """

    def __init__(
        self,
        base_threshold: float = 0.65,
        learning_rate: float = 0.01,
        feedback_cost_factor: float = 0.5,
        min_threshold: float = 0.50,
        max_threshold: float = 0.95,
        n_classes: int = 100,
    ) -> None:
        """
        Args:
            base_threshold: Initial decision threshold for all classes
            learning_rate: Step size for threshold adaptation (η)
            feedback_cost_factor: Weight penalizing unnecessary human queries (γ)
            min_threshold: Lower bound for τ_k
            max_threshold: Upper bound for τ_k
            n_classes: Preallocated number of class slots
        """
        self.eta = learning_rate
        self.gamma = feedback_cost_factor
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        # Per-class state: {class_idx: {"threshold": float, "correct": float, "incorrect": float, "total": float}}
        self._class_state: Dict[int, Dict[str, float]] = {}
        for k in range(n_classes):
            self._class_state[k] = {
                "threshold": base_threshold,
                "correct": 0.0,
                "incorrect": 0.0,
                "total": 0.0,
            }

    def should_accept(
        self,
        confidence: float,
        class_idx: int,
        recent_incorrect_rate: float = 0.0,
        recent_correct_rate: float = 1.0,
    ) -> Tuple[bool, str]:
        """
        Evaluate whether to accept a prediction or request human feedback.

        Args:
            confidence: Calibrated confidence score [0, 1]
            class_idx: Predicted class index
            recent_incorrect_rate: Fraction of recent corrections that were wrong [0, 1]
            recent_correct_rate: Fraction of recent confirmations that were right [0, 1]

        Returns:
            (accept: bool, action: str) where action is "ACCEPT" or "REQUEST_FEEDBACK"
        """
        # Ensure class state exists (handles dynamic class expansion)
        if class_idx not in self._class_state:
            existing_thresholds = [s["threshold"] for s in self._class_state.values()]
            default_thresh = float(np.mean(existing_thresholds)) if existing_thresholds else 0.65
            self._class_state[class_idx] = {
                "threshold": default_thresh,
                "correct": 0.0,
                "incorrect": 0.0,
                "total": 0.0,
            }

        state = self._class_state[class_idx]
        threshold = float(np.clip(state["threshold"], self.min_threshold, self.max_threshold))

        # Update threshold: Δτ = η * (incorrect_rate - γ * correct_rate)
        delta = self.eta * (recent_incorrect_rate - self.gamma * recent_correct_rate)
        state["threshold"] = threshold + delta

        # Update counters (EMA-style tracking)
        state["total"] += 1.0
        if recent_incorrect_rate > 0.5:
            state["incorrect"] += 1.0
        else:
            state["correct"] += 1.0

        accept = confidence >= threshold
        action = "ACCEPT" if accept else "REQUEST_FEEDBACK"

        logger.debug(
            f"ACT | Class {class_idx} | Conf: {confidence:.3f} | τ: {threshold:.3f} | Action: {action}"
        )

        return accept, action

    def get_threshold(self, class_idx: int) -> float:
        """Return the current adaptive threshold for a given class."""
        if class_idx in self._class_state:
            return float(np.clip(self._class_state[class_idx]["threshold"], self.min_threshold, self.max_threshold))
        existing = [s["threshold"] for s in self._class_state.values()]
        return float(np.clip(np.mean(existing), self.min_threshold, self.max_threshold)) if existing else 0.65

    def get_all_thresholds(self) -> Dict[int, float]:
        """Return a snapshot of all current class thresholds."""
        return {k: self.get_threshold(k) for k in self._class_state}

    def reset_class(self, class_idx: int, base_threshold: float = 0.65) -> None:
        """Reset adaptation state for a specific class (e.g., after dataset reset)."""
        if class_idx in self._class_state:
            self._class_state[class_idx] = {
                "threshold": base_threshold,
                "correct": 0.0,
                "incorrect": 0.0,
                "total": 0.0,
            }