"""Online confidence calibration with verified-style diagnostics for few-shot learning."""

from __future__ import annotations

import itertools

import numpy as np


class CalibrationEngine:
    """Track prediction calibration and apply post-hoc calibration transforms."""

    def __init__(
        self,
        n_bins: int = 15,
        window_size: int = 100,
        temperature_init: float = 1.0,
        method: str = "temperature",
        evaluation_bins: int | None = None,
        scaling_binning_bins: int | None = None,
        min_fit_samples: int | None = None,
    ) -> None:
        self.n_bins = max(2, int(n_bins))
        self.window_size = max(1, int(window_size))
        self.eval_bins = max(self.n_bins, int(evaluation_bins or self.n_bins))
        self.scaling_binning_bins = max(2, int(scaling_binning_bins or self.n_bins))
        self.temperature = float(temperature_init)
        self.method = method

        self._window_confidences: list[float] = []
        self._window_correct: list[bool] = []
        self._ece_history: list[float] = []

        self._conformal_margin = 0.1
        self._scaling_binning_edges: np.ndarray | None = None
        self._scaling_binning_values: np.ndarray | None = None
        # Minimum number of samples required before attempting temperature
        # refit or conformal margin refit. In few-shot / human-in-the-loop
        # settings the default is intentionally small to allow early adaptivity.
        if min_fit_samples is None:
            # default: at least 5 samples or a quarter of the window size
            self.min_fit_samples = max(5, max(1, int(self.window_size // 4)))
        else:
            self.min_fit_samples = int(min_fit_samples)

    def _to_unit_confidence(self, raw_confidence: float) -> float:
        value = float(raw_confidence)
        if 0.0 <= value <= 1.0:
            return value
        return float(np.clip((value + 1.0) / 2.0, 0.0, 1.0))

    def _clip_logit_input(self, confidence: np.ndarray) -> np.ndarray:
        result: np.ndarray = np.clip(confidence, 1e-6, 1.0 - 1e-6)
        return result

    def update(
        self,
        raw_confidence: float,
        predicted_label: object,
        true_label: object,
    ) -> None:
        """Update calibration window with one prediction event."""

        norm_conf = self._to_unit_confidence(raw_confidence)
        is_correct = predicted_label == true_label

        self._window_confidences.append(norm_conf)
        self._window_correct.append(bool(is_correct))

        if len(self._window_confidences) > self.window_size:
            self._window_confidences.pop(0)
            self._window_correct.pop(0)

        # Use a configurable minimum sample requirement for refitting.
        if len(self._window_confidences) >= self.min_fit_samples:
            if self.method in {"temperature", "scaling_binning"}:
                self._refit_temperature()
            if self.method == "scaling_binning":
                self._fit_scaling_binning()
            if self.method == "conformal":
                self._refit_conformal_margin()

        current_ece = self.compute_ece(
            np.asarray(self._window_confidences, dtype=np.float64),
            np.asarray(self._window_correct, dtype=np.int64),
            n_bins=self.eval_bins,
        )
        self._ece_history.append(current_ece)

    def _apply_temperature(self, unit_confidence: float) -> float:
        clipped = self._clip_logit_input(np.asarray([unit_confidence], dtype=np.float64))[0]
        logit = np.log(clipped / (1.0 - clipped))
        temperature = max(float(self.temperature), 1e-6)
        scaled = 1.0 / (1.0 + np.exp(-logit / temperature))
        return float(np.clip(scaled, 0.0, 1.0))

    def calibrate(self, raw_confidence: float) -> float:
        """Apply the configured calibration strategy to one confidence score."""

        norm_conf = self._to_unit_confidence(raw_confidence)

        if self.method == "none":
            return norm_conf

        if self.method == "conformal":
            return float(np.clip(norm_conf - self._conformal_margin, 0.0, 1.0))

        scaled = self._apply_temperature(norm_conf)
        if self.method != "scaling_binning":
            return scaled

        if self._scaling_binning_edges is None or self._scaling_binning_values is None:
            return scaled

        edges = self._scaling_binning_edges
        values = self._scaling_binning_values
        if edges.size < 2 or values.size == 0:
            return scaled

        bin_index = int(np.searchsorted(edges, scaled, side="right") - 1)
        bin_index = int(np.clip(bin_index, 0, values.size - 1))
        return float(np.clip(values[bin_index], 0.0, 1.0))

    def _refit_temperature(self) -> None:
        """Refit temperature using grid search on the sliding window."""

        if len(self._window_confidences) < self.min_fit_samples:
            return

        confs = np.asarray(self._window_confidences, dtype=np.float64)
        correct = np.asarray(self._window_correct, dtype=np.float64)
        confs = self._clip_logit_input(confs)
        logits = np.log(confs / (1.0 - confs))

        candidates = np.linspace(0.5, 3.0, 25, dtype=np.float64)
        best_loss = np.inf
        best_temp = max(float(self.temperature), 1e-6)

        for candidate in candidates:
            scaled_logits = logits / candidate
            scaled_confs = 1.0 / (1.0 + np.exp(-scaled_logits))
            loss = -np.mean(
                correct * np.log(scaled_confs + 1e-6)
                + (1.0 - correct) * np.log(1.0 - scaled_confs + 1e-6)
            )
            if float(loss) < float(best_loss):
                best_loss = float(loss)
                best_temp = float(candidate)

        self.temperature = best_temp

    def _refit_conformal_margin(self) -> None:
        """Update a conservative conformal-style correction margin."""

        if len(self._window_confidences) < self.min_fit_samples:
            self._conformal_margin = 0.1
            return

        confs = np.asarray(self._window_confidences, dtype=np.float64)
        correct = np.asarray(self._window_correct, dtype=np.float64)
        overconfidence = np.clip(confs - correct, 0.0, 1.0)
        self._conformal_margin = float(np.quantile(overconfidence, 0.9))

    def _fit_scaling_binning(self) -> None:
        """Fit a scaling-binning recalibration map from current window."""

        if len(self._window_confidences) < max(20, self.scaling_binning_bins):
            self._scaling_binning_edges = None
            self._scaling_binning_values = None
            return

        confs = np.asarray(self._window_confidences, dtype=np.float64)
        scaled = np.asarray([self._apply_temperature(float(conf)) for conf in confs], dtype=np.float64)

        n_bins = min(self.scaling_binning_bins, max(2, scaled.size // 2))
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        raw_edges = np.quantile(scaled, quantiles)
        raw_edges[0] = 0.0
        raw_edges[-1] = 1.0

        unique_edges = np.unique(raw_edges)
        if unique_edges.size < 2:
            self._scaling_binning_edges = None
            self._scaling_binning_values = None
            return

        values: list[float] = []
        for left, right in itertools.pairwise(unique_edges):
            in_bin = (scaled >= left) & (scaled <= right)
            if not np.any(in_bin):
                values.append(float((left + right) * 0.5))
            else:
                values.append(float(np.mean(scaled[in_bin])))

        self._scaling_binning_edges = unique_edges.astype(np.float64, copy=False)
        self._scaling_binning_values = np.asarray(values, dtype=np.float64)

    def compute_ece(
        self,
        confidences: np.ndarray,
        labels_correct: np.ndarray,
        n_bins: int | None = None,
    ) -> float:
        """Compute expected calibration error (ECE, L1) with equal-width bins."""

        confs = np.asarray(confidences, dtype=np.float64).reshape(-1)
        correct = np.asarray(labels_correct, dtype=np.float64).reshape(-1)
        if confs.size == 0:
            return 0.0
        if confs.size != correct.size:
            raise ValueError("confidences and labels_correct must have equal length.")

        confs = np.clip(confs, 0.0, 1.0)
        bins = int(n_bins or self.n_bins)
        bin_edges = np.linspace(0.0, 1.0, bins + 1)

        ece = 0.0
        total = float(confs.size)
        for idx in range(bins):
            lower = bin_edges[idx]
            upper = bin_edges[idx + 1]
            if idx == 0:
                in_bin = (confs >= lower) & (confs <= upper)
            else:
                in_bin = (confs > lower) & (confs <= upper)
            count = int(np.sum(in_bin))
            if count == 0:
                continue
            avg_conf = float(np.mean(confs[in_bin]))
            avg_acc = float(np.mean(correct[in_bin]))
            ece += (count / total) * abs(avg_acc - avg_conf)

        return float(ece)

    def compute_debiased_ece(
        self,
        confidences: np.ndarray,
        labels_correct: np.ndarray,
        n_bins: int | None = None,
    ) -> float:
        """Estimate calibration via debiased squared CE, then map to an L2-style ECE."""

        confs = np.asarray(confidences, dtype=np.float64).reshape(-1)
        correct = np.asarray(labels_correct, dtype=np.float64).reshape(-1)
        if confs.size == 0:
            return 0.0
        if confs.size != correct.size:
            raise ValueError("confidences and labels_correct must have equal length.")

        confs = np.clip(confs, 0.0, 1.0)
        bins = int(n_bins or self.eval_bins)
        bin_edges = np.linspace(0.0, 1.0, bins + 1)
        total = float(confs.size)

        debiased_sq = 0.0
        for idx in range(bins):
            lower = bin_edges[idx]
            upper = bin_edges[idx + 1]
            if idx == 0:
                in_bin = (confs >= lower) & (confs <= upper)
            else:
                in_bin = (confs > lower) & (confs <= upper)
            count = int(np.sum(in_bin))
            if count == 0:
                continue

            bin_conf = confs[in_bin]
            bin_correct = correct[in_bin]
            mean_conf = float(np.mean(bin_conf))
            mean_acc = float(np.mean(bin_correct))

            plugin = (mean_conf - mean_acc) ** 2
            correction = 0.0
            if count > 1:
                correction = (mean_acc * (1.0 - mean_acc)) / float(count - 1)

            debiased_bin = max(plugin - correction, 0.0)
            debiased_sq += (count / total) * debiased_bin

        return float(np.sqrt(max(debiased_sq, 0.0)))

    def calibration_summary(self) -> dict[str, float]:
        """Return current calibration metrics on the sliding window."""

        confs = np.asarray(self._window_confidences, dtype=np.float64)
        correct = np.asarray(self._window_correct, dtype=np.float64)
        return {
            "window_size": float(confs.size),
            "ece": self.compute_ece(confs, correct, n_bins=self.eval_bins),
            "debiased_ece": self.compute_debiased_ece(confs, correct, n_bins=self.eval_bins),
            "temperature": self.current_temperature,
        }

    @property
    def current_ece(self) -> float:
        """Return the most recently computed ECE."""

        return self._ece_history[-1] if self._ece_history else 0.0

    @property
    def current_temperature(self) -> float:
        """Return the current temperature scaling parameter."""

        return float(self.temperature)
