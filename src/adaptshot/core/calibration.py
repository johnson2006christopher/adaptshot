"""Online calibration module for few-shot vision predictions.

Implements temperature scaling, Expected Calibration Error (ECE) tracking,
and a conformal prediction stub for high-stakes deployment contexts.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class CalibrationEngine:
    """
    Tracks prediction calibration and applies post-hoc scaling to raw confidence scores.

    Designed for streaming few-shot evaluation where a held-out validation set is
    unavailable or too small. Maintains a sliding window of recent predictions to
    fit temperature parameters online.
    """

    def __init__(
        self,
        n_bins: int = 15,
        window_size: int = 100,
        temperature_init: float = 1.0,
        method: str = "temperature",
    ) -> None:
        """
        Args:
            n_bins: Number of bins for ECE computation (default: 15)
            window_size: Size of sliding window for online temperature fitting
            temperature_init: Initial temperature value (1.0 = no scaling)
            method: Calibration method ("temperature" or "conformal")
        """
        self.n_bins = n_bins
        self.window_size = window_size
        self.temperature = torch.nn.Parameter(torch.tensor(temperature_init))
        self.method = method

        # Sliding window buffers
        self._window_confidences: List[float] = []
        self._window_correct: List[bool] = []

        # ECE tracking state
        self._ece_history: List[float] = []

    def update(
        self,
        raw_confidence: float,
        predicted_label: int,
        true_label: int,
    ) -> None:
        """
        Update calibration state with a new prediction and ground truth.

        Maintains a fixed-size sliding window to enable online temperature fitting
        without requiring a separate validation dataset.

        Args:
            raw_confidence: Cosine similarity score (unnormalized, typically [-1, 1])
            predicted_label: Predicted class index
            true_label: Ground truth class index
        """
        # Normalize raw confidence to [0, 1] for temperature scaling
        norm_conf = (raw_confidence + 1.0) / 2.0
        self._window_confidences.append(norm_conf)
        self._window_correct.append(predicted_label == true_label)

        # Maintain fixed window size
        if len(self._window_confidences) > self.window_size:
            self._window_confidences.pop(0)
            self._window_correct.pop(0)

        # Refit temperature if window is sufficiently populated
        if len(self._window_confidences) >= max(10, self.window_size // 2):
            self._refit_temperature()

        # Update running ECE
        current_ece = self.compute_ece(
            np.array(self._window_confidences),
            np.array(self._window_correct, dtype=int)
        )
        self._ece_history.append(current_ece)

    def calibrate(self, raw_confidence: float) -> float:
        """
        Apply calibration to a raw confidence score.

        Args:
            raw_confidence: Unnormalized cosine similarity score

        Returns:
            Calibrated confidence in [0, 1]
        """
        if self.method == "conformal":
            # Conformal stub: return conservative lower bound
            return max(0.0, raw_confidence - 0.1)
        
        # Temperature scaling
        norm_conf = (raw_confidence + 1.0) / 2.0
        # Clamp to prevent extreme scaling
        norm_conf = np.clip(norm_conf, 1e-6, 1.0 - 1e-6)
        # Apply temperature
        logit = np.log(norm_conf / (1.0 - norm_conf))
        scaled = 1.0 / (1.0 + np.exp(-logit / float(self.temperature)))
        return float(np.clip(scaled, 0.0, 1.0))

    def compute_ece(
        self,
        confidences: np.ndarray,
        labels_correct: np.ndarray,
    ) -> float:
        """
        Compute Expected Calibration Error (ECE) on a set of predictions.

        ECE measures the gap between average confidence and average accuracy
        across confidence bins. Lower is better; <0.05 is the target.

        Args:
            confidences: Array of predicted confidence scores in [0, 1]
            labels_correct: Binary array (1 if correct, 0 if incorrect)

        Returns:
            ECE value in [0, 1]
        """
        if len(confidences) == 0:
            return 0.0

        bin_boundaries = np.linspace(0.0, 1.0, self.n_bins + 1)
        ece = 0.0
        total_samples = len(confidences)

        for i in range(self.n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = labels_correct[in_bin].mean()
                ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin

        return float(ece)

    def _refit_temperature(self) -> None:
        """
        Refit temperature parameter using NLL loss on the sliding window.
        Uses a simple gradient-free line search for stability on CPU.
        """
        if len(self._window_confidences) < 10:
            return

        confs = np.array(self._window_confidences, dtype=np.float32)
        correct = np.array(self._window_correct, dtype=np.float32)

        # Clamp to prevent log(0)
        confs = np.clip(confs, 1e-6, 1.0 - 1e-6)
        logits = np.log(confs / (1.0 - confs))

        # Grid search over reasonable temperature range [0.5, 3.0]
        candidates = np.linspace(0.5, 3.0, 25)
        best_loss = np.inf
        best_T = float(self.temperature)

        for T in candidates:
            scaled_logits = logits / T
            scaled_confs = 1.0 / (1.0 + np.exp(-scaled_logits))
            # Binary cross-entropy loss
            loss = -np.mean(correct * np.log(scaled_confs + 1e-6) + (1 - correct) * np.log(1.0 - scaled_confs + 1e-6))
            if loss < best_loss:
                best_loss = loss
                best_T = T

        self.temperature = torch.nn.Parameter(torch.tensor(best_T))

    @property
    def current_ece(self) -> float:
        """Return the most recently computed ECE."""
        return self._ece_history[-1] if self._ece_history else 0.0

    @property
    def current_temperature(self) -> float:
        """Return the current temperature scaling parameter."""
        return float(self.temperature)