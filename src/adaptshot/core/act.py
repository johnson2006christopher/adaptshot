"""Adaptive Confidence Thresholding (ACT) engine for few-shot predictions.

Dynamically adjusts per-class decision thresholds based on correction
history, reducing false acceptances by requesting human feedback when the
model is genuinely unsure.

Since #116 the engine separates reading from learning: ``should_accept`` is a
pure decision (asking does not move the threshold), and thresholds move only
when :meth:`record_outcome` reports a real outcome from a human correction or
confirmation. Before that split, every ``predict()`` nudged the threshold off
a proxy built from recent confidences -- forty identical predictions moved a
threshold from 0.6500 to 0.6357 with no human in the loop at all, and nothing
`correct()` learned ever reached the decision.
"""

import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)


class ACTEngine:
    """Adaptive Confidence Thresholding engine.

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
        recent_incorrect_rate: float | None = None,
        recent_correct_rate: float | None = None,
    ) -> tuple[bool, str]:
        """Decide whether to accept a prediction or request human feedback.

        A pure read: asking the question does not move the threshold (#116).
        Thresholds move only through :meth:`record_outcome`, when a human
        correction or confirmation supplies a real outcome.

        Args:
            confidence: Calibrated confidence score [0, 1]
            class_idx: Predicted class index
            recent_incorrect_rate: Deprecated in 0.3.1, ignored, removed in
                0.4.0. The proxy it fed moved thresholds on every prediction;
                report real outcomes through :meth:`record_outcome` instead.
            recent_correct_rate: Deprecated alongside ``recent_incorrect_rate``.

        Returns:
            (accept: bool, action: str) where action is "ACCEPT" or "REQUEST_FEEDBACK"
        """
        if recent_incorrect_rate is not None or recent_correct_rate is not None:
            warnings.warn(
                "should_accept's recent_incorrect_rate/recent_correct_rate are "
                "deprecated since 0.3.1 and ignored; they moved thresholds on "
                "every prediction. Report real outcomes with "
                "ACTEngine.record_outcome(). They will be removed in 0.4.0.",
                DeprecationWarning,
                stacklevel=2,
            )

        threshold = self.get_threshold(class_idx)
        accept = confidence >= threshold
        action = "ACCEPT" if accept else "REQUEST_FEEDBACK"

        logger.debug(
            "ACT | Class %s | Conf: %.3f | τ: %.3f | Action: %s",
            class_idx, confidence, threshold, action,
        )

        return accept, action

    def record_outcome(self, class_idx: int, correct: bool) -> None:
        """Adapt the class threshold from one real outcome.

        Called when ground truth arrives -- a human correction (the prediction
        was wrong) or confirmation (it was right). This is the only place a
        threshold moves, which is what the class docstring's "adapts based on
        correction history" has always promised.

        The update keeps the v0.2.0 shape: a symmetric bounded step plus slow
        mean reversion, clamped to [min_threshold, max_threshold].

        Args:
            class_idx: The class that was *predicted* -- its threshold is the
                one that let the prediction through or held it back.
            correct: Whether the prediction matched the human's label.
        """
        if class_idx not in self._class_state:
            self._class_state[class_idx] = {
                "threshold": self.get_threshold(class_idx),
                "correct": 0.0,
                "incorrect": 0.0,
                "total": 0.0,
            }

        state = self._class_state[class_idx]
        threshold = float(np.clip(state["threshold"], self.min_threshold, self.max_threshold))

        # delta = η * (±1) + μ * (base - τ): a wrong prediction raises the
        # class's bar, a confirmed one lowers it, and everything drifts slowly
        # back toward base so one bad streak is not a life sentence.
        error_signal = -1.0 if correct else 1.0
        delta = self.eta * error_signal
        delta += self._mean_reversion_strength * (self._base_threshold - threshold)
        state["threshold"] = float(np.clip(
            threshold + delta, self.min_threshold, self.max_threshold
        ))

        state["total"] += 1.0
        if correct:
            state["correct"] += 1.0
        else:
            state["incorrect"] += 1.0

        logger.debug(
            "ACT outcome | Class %s | correct=%s | τ -> %.4f",
            class_idx, correct, state["threshold"],
        )

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