"""Adaptive Confidence Thresholding (ACT) for uncertainty-aware decisions."""

from __future__ import annotations

from typing import Tuple

__all__ = ["compute_adaptive_threshold", "should_accept_prediction"]


def compute_adaptive_threshold(
    base_threshold: float,
    uncertainty: float,
    support_size: int,
    correction_rate: float,
    alpha: float = 0.1,
    beta: float = 0.05,
) -> float:
    """
    Compute adaptive threshold with monotonic uncertainty response.

    Threshold increases with uncertainty and correction history, and
    decreases slightly as support size grows.
    """
    if support_size < 0:
        raise ValueError("support_size must be non-negative.")

    u = min(max(float(uncertainty), 0.0), 1.0)
    c = min(max(float(correction_rate), 0.0), 1.0)
    support_bonus = 0.02 / (1.0 + float(support_size))

    threshold = float(base_threshold) + alpha * u + beta * c - support_bonus
    return min(max(threshold, 0.0), 1.0)


def should_accept_prediction(
    confidence: float,
    adaptive_threshold: float,
    fallback_action: str = "request_feedback",
) -> Tuple[bool, str]:
    """
    Accept, request feedback, or reject prediction based on confidence.

    Actions:
    - `accept`: confidence >= adaptive threshold
    - `request_feedback`: confidence below threshold
    - `reject_outright`: confidence critically low (< half threshold)
    """
    conf = min(max(float(confidence), 0.0), 1.0)
    threshold = min(max(float(adaptive_threshold), 0.0), 1.0)

    if conf >= threshold:
        return True, "accept"
    if conf < max(0.05, threshold * 0.5):
        return False, "reject_outright"
    return False, fallback_action
