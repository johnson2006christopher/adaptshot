"""Adaptive Confidence Thresholding (ACT) engine for few-shot predictions.

Dynamically adjusts per-class decision thresholds based on real-time
correction history and model uncertainty, reducing false acceptances
by requesting human feedback when the model is genuinely unsure.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ACTEngine:
    """**Experimental.** May change in a minor release without a deprecation cycle; see ``adaptshot.api``.

    Adaptive Confidence Thresholding engine.

    Maintains a dynamic threshold τ_k for each class k that adapts based on:
    - Historical correction rates (incorrect vs. correct)
    - Model uncertainty signals (entropy/ECE proxies)
    - Configurable cost of requesting human feedback (γ)

    The engine implements an exponential moving average update rule to
    prevent oscillation while remaining responsive to distribution shift.

    Experimental for one reason: it is constructed inside ``FewShotLearner`` and
    exercised only through it -- no test names this class. It becomes stable when
    it has tests of its own, not before (#23).
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
        self._base_threshold = base_threshold
        self._mean_reversion_strength = 0.001  # Slow pull toward base

        # Per-class state: {class_idx: {"threshold": float, "correct": float, "incorrect": float, "total": float}}
        self._class_state: dict[int, dict[str, float]] = {}
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
    ) -> tuple[bool, str]:
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

        # v0.2.0 fix: Symmetric bounded update with mean reversion.
        # Previous formula (v0.2.0-dev): delta = η * (incorrect - γ * correct)
        # This monotonically decreased thresholds because γ=0.5 multiplied the
        # (usually larger) correct rate, creating a permanent downward bias.
        #
        # New formula: delta = η * (incorrect_rate - correct_rate) + μ * (base - τ)
        # - Symmetric: equal weight to incorrect vs correct signals
        # - Mean-reversion: thresholds drift back toward base_threshold slowly
        # - Clamped: thresholds stay within [min_threshold, max_threshold]
        error_signal = recent_incorrect_rate - recent_correct_rate
        delta = self.eta * error_signal
        # Mean-reversion toward base (prevents runaway drift)
        delta += self._mean_reversion_strength * (self._base_threshold - threshold)
        state["threshold"] = float(np.clip(
            threshold + delta, self.min_threshold, self.max_threshold
        ))

        # Update counters (EMA-style tracking)
        state["total"] += 1.0
        if recent_incorrect_rate > 0.5:
            state["incorrect"] += 1.0
        else:
            state["correct"] += 1.0

        # Re-read threshold after update for decision
        threshold_updated = float(np.clip(
            state["threshold"], self.min_threshold, self.max_threshold
        ))
        accept = confidence >= threshold_updated
        action = "ACCEPT" if accept else "REQUEST_FEEDBACK"

        logger.debug(
            "ACT | Class %s | Conf: %.3f | τ: %.3f | Action: %s",
            class_idx, confidence, threshold_updated, action,
        )

        return accept, action

    def get_threshold(self, class_idx: int) -> float:
        """Return the current adaptive threshold for a given class."""
        if class_idx in self._class_state:
            return float(np.clip(self._class_state[class_idx]["threshold"], self.min_threshold, self.max_threshold))
        existing = [s["threshold"] for s in self._class_state.values()]
        return float(np.clip(np.mean(existing), self.min_threshold, self.max_threshold)) if existing else 0.65

    def get_all_thresholds(self) -> dict[int, float]:
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