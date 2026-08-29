"""Distribution-free conformal prediction with online adaptation.

Provides split-conformal and cross-conformal prediction sets with
configurable significance levels. Nonconformity scores are computed
from softmax-based and distance-to-prototype measures, giving valid
marginal coverage guarantees under the exchangeability assumption.

Integration points:
- CalibrationEngine: Stores calibration scores and manages online update
- FewShotLearner.predict(): Returns prediction sets alongside point predictions
- PredictionResult: Exposes conformal_set and conformal_size fields
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from ..utils.arrays import FloatArray, LabelArray

logger = logging.getLogger(__name__)


@dataclass
class ConformalPredictionSet:
    """Structured result from conformal prediction inference.

    Attributes:
        prediction_set: Set of class labels included at the given alpha level.
        set_size: Number of classes in the prediction set.
        alpha: Significance level used (e.g., 0.05 = 95% coverage target).
        q_hat: Computed nonconformity quantile threshold.
        coverage_estimate: Running empirical coverage observed so far; NaN until the
            engine has calibrated, because a singleton fallback has no basis for one.
        calibrated: False while the set is the top-1 alone because the engine has
            fewer than ``min_calibration_size`` scores. No guarantee applies then, and
            consumers should say so rather than show the set (#80).
        prediction: The single best-guess prediction (for backward compat).
        confidence: Calibrated confidence of the top prediction.
    """

    prediction_set: set[str | int] = field(default_factory=set)
    set_size: int = 0
    alpha: float = 0.05
    q_hat: float = 0.0
    coverage_estimate: float = 0.0
    prediction: str | int = ""
    confidence: float = 0.0
    calibrated: bool = True

    def contains(self, label: str | int) -> bool:
        """Check whether a label is within the conformal prediction set."""
        return label in self.prediction_set

    def __repr__(self) -> str:
        return (
            f"ConformalPredictionSet("
            f"size={self.set_size}, "
            f"alpha={self.alpha:.3f}, "
            f"coverage={self.coverage_estimate:.3f}, "
            f"top='{self.prediction}')"
        )


class ConformalEngine:
    """Distribution-free conformal prediction with online buffer updates.

    Implements split-conformal prediction with nonconformity scores
    derived from distance-to-class and softmax outputs. Maintains a
    rolling calibration buffer that adapts as new human corrections
    arrive, keeping the coverage guarantee fresh under distribution shift.

    Theory:
        For exchangeable calibration scores s_1, ..., s_n and a new
        test score s_{n+1}, the conformal p-value is:
            p = |{i: s_i >= s_{n+1}}| / (n + 1)
        The prediction set at level alpha contains all classes where
        the nonconformity score does not exceed the (1-alpha) quantile.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_bins: int = 20,
        min_calibration_size: int = 10,
        max_calibration_size: int = 500,
        mode: str = "split",
        score_method: str = "ratio",
    ) -> None:
        """Initialize the conformal prediction engine.

        Args:
            alpha: Target miscoverage rate (0.05 = 95% coverage).
            n_bins: Resolution for nonconformity score discretization.
            min_calibration_size: Minimum scores before producing sets.
            max_calibration_size: Rolling buffer capacity for scores.
            mode: "split" for split-conformal or "cross" for cross-conformal.
            score_method: "ratio" (default), "softmax" or "distance". See
                ``nonconformity`` for what each measures and why the default changed.
        """
        self.alpha = float(alpha)
        self.n_bins = int(n_bins)
        self.min_calibration_size = int(min_calibration_size)

        # Below this many calibration scores, no finite quantile exists at this
        # alpha and every prediction set is the full label set. The maths, not a
        # policy: see _compute_quantile. Said once, here, rather than discovered
        # from a run of uninformative sets.
        self.min_informative_size = math.ceil((1.0 - self.alpha) / self.alpha)
        if self.min_calibration_size < self.min_informative_size:
            logger.warning(
                "ConformalEngine(alpha=%.3f): prediction sets are uninformative -- "
                "every class -- until %d calibration scores exist, but sets are "
                "produced from %d. Raise min_calibration_size to %d, or raise alpha.",
                self.alpha, self.min_informative_size, self.min_calibration_size,
                self.min_informative_size,
            )
        self.max_calibration_size = int(max_calibration_size)
        self.mode = mode
        self.score_method = score_method

        # Rolling calibration buffer: nonconformity scores for correct predictions
        self._calibration_scores: list[float] = []
        # Per-class score distributions for class-conditional conformal
        self._class_scores: dict[str | int, list[float]] = {}
        # Tracking
        self._total_predictions: int = 0
        self._covered: int = 0

    # ------------------------------------------------------------------
    # Nonconformity score computation
    # ------------------------------------------------------------------

    def nonconformity(
        self,
        distances: FloatArray,
        labels: LabelArray,
        true_label: str | int,
    ) -> float:
        """The nonconformity of `true_label` given distances to every class.

        One place for the choice of score, so that calibration and prediction
        cannot drift apart by using different ones.

        ``ratio`` -- the default since 0.3.0 -- is ``d_true / min(d)``: 1.0 for
        the nearest class, growing with how much worse the true class is than
        the best. It replaces ``softmax``, which divided every distance by the
        row's maximum before a five-way softmax and so produced scores between
        0.72 and 0.80 for clean photographs, blurred ones, and a crop the model
        had never seen (#86). A score that cannot tell those apart cannot widen
        a prediction set when it should. Measured on real PlantVillage
        episodes, the ratio gives tighter sets at the same clean coverage --
        1.11 against 1.22 -- and sets that widen under shift, from 1.09 to 1.43
        at blur radius 4, though coverage there stays far below the target:
        that failure is exchangeability breaking, not the score.
        """

        if len(distances) == 0 or len(labels) == 0:
            return float("inf")
        if self.score_method == "softmax":
            return self.softmax_nonconformity(distances, labels, true_label)
        if self.score_method == "distance":
            true_idx = np.where(labels == true_label)[0]
            if len(true_idx) == 0:
                return 1.0
            reference = float(np.median(distances) + np.std(distances))
            return self.distance_nonconformity(float(distances[true_idx[0]]), reference)
        true_idx = np.where(labels == true_label)[0]
        if len(true_idx) == 0:
            return float("inf")  # true class not a candidate: fully non-conforming
        return float(distances[true_idx[0]] / (float(np.min(distances)) + 1e-8))

    @staticmethod
    def softmax_nonconformity(
        distances: FloatArray,
        labels: LabelArray,
        true_label: str | int,
    ) -> float:
        """Compute nonconformity as 1 - softmax(true_label | distances).

        Lower distances to the true class produce lower nonconformity
        (i.e., the prediction is more conforming). Distances are converted
        to pseudo-probabilities via softmax.

        Args:
            distances: [N] array of distances to each class prototype.
            labels: [N] array of class labels corresponding to distances.
            true_label: The ground-truth class label.

        Returns:
            Nonconformity score in [0, 1]. Lower = more conforming.
        """
        if len(distances) == 0 or len(labels) == 0:
            return 1.0

        # Convert distances to negative logits for softmax
        # Smaller distance -> larger negative logit
        max_dist = np.max(distances) + 1e-8
        logits = -distances / max_dist  # scale to avoid overflow
        logits = logits - np.max(logits)  # numerical stability
        probs = np.exp(logits) / (np.sum(np.exp(logits)) + 1e-8)

        true_idx = np.where(labels == true_label)[0]
        if len(true_idx) == 0:
            return 1.0  # true class not in candidate set (fully non-conforming)

        true_prob = float(probs[true_idx[0]])
        return float(1.0 - true_prob)

    @staticmethod
    def distance_nonconformity(
        distance_to_class: float,
        threshold_distance: float,
    ) -> float:
        """Compute nonconformity from distance to predicted class.

        Score = min(1.0, distance / threshold_distance)

        Args:
            distance_to_class: Distance from query to its predicted class.
            threshold_distance: Reference distance (e.g., OOD threshold).

        Returns:
            Nonconformity score in [0, 1].
        """
        if threshold_distance <= 0.0:
            return 1.0 if distance_to_class > 0.0 else 0.0
        return float(min(1.0, distance_to_class / threshold_distance))

    # ------------------------------------------------------------------
    # Calibration buffer management
    # ------------------------------------------------------------------

    def update_calibration(
        self,
        score: float,
        true_label: str | int,
        predicted_in_set: bool = False,
    ) -> None:
        """Add a nonconformity score to the calibration buffer.

        Called when ground truth is available (human correction or evaluation).

        Args:
            score: Nonconformity score of the prediction.
            true_label: Ground-truth class label.
            predicted_in_set: Whether the true label was in the prediction set.
        """
        # Update global buffer
        self._calibration_scores.append(float(score))
        if len(self._calibration_scores) > self.max_calibration_size:
            self._calibration_scores.pop(0)

        # Update per-class buffer
        if true_label not in self._class_scores:
            self._class_scores[true_label] = []
        self._class_scores[true_label].append(float(score))
        if len(self._class_scores[true_label]) > self.max_calibration_size // 10:
            self._class_scores[true_label].pop(0)

        self._total_predictions += 1
        if predicted_in_set:
            self._covered += 1

    @property
    def calibration_size(self) -> int:
        """Return the current number of calibration scores."""
        return len(self._calibration_scores)

    @property
    def empirical_coverage(self) -> float:
        """Return the observed coverage rate so far."""
        if self._total_predictions == 0:
            return 1.0 - self.alpha  # target (prior)
        return self._covered / self._total_predictions

    # ------------------------------------------------------------------
    # Quantile computation
    # ------------------------------------------------------------------

    def _compute_quantile(self, scores: list[float]) -> float:
        """Compute the (1-alpha) empirical quantile with finite-sample correction.

        Uses the standard conformal quantile formula:
            q_hat = quantile(scores, ceil((n+1)*(1-alpha))/n)

        Args:
            scores: Nonconformity scores from calibration set.

        Returns:
            Quantile threshold q_hat.
        """
        n = len(scores)
        if n == 0:
            return float("inf")  # No calibration data: every class is in the set.

        # The finite-sample guarantee is P(y in set) >= 1 - alpha, and it comes
        # from taking the ceil((n+1)(1-alpha))-th smallest calibration score.
        # When that rank exceeds n there is no such score: the theorem's answer
        # is +inf, meaning every class is included, because n points cannot
        # certify a (1-alpha) level. That happens whenever n < (1-alpha)/alpha
        # -- at the default alpha = 0.05, for every n below 19.
        #
        # This used to clamp the rank to n-1 and return the largest observed
        # score instead. That is a smaller set than the guarantee allows, and it
        # under-covered: 91.3% measured against a 95% target at n = 10, the
        # library's own min_calibration_size (#14). A set that is honestly
        # everything beats a set that is quietly too small.
        rank = math.ceil((n + 1) * (1.0 - self.alpha))
        if rank > n:
            return float("inf")

        sorted_scores = np.sort(np.asarray(scores, dtype=np.float64))
        return float(sorted_scores[rank - 1])

    def _compute_cross_quantile(self, scores: list[float]) -> float:
        """Compute cross-conformal quantile via k-fold averaging.

        Partitions calibration scores into n_bins folds, computes the
        conformal quantile per fold, and averages them. This provides
        more stable estimates than a single split, at the cost of
        slightly conservative coverage (average of valid bounds).

        Args:
            scores: Nonconformity scores from calibration set.

        Returns:
            Cross-conformal quantile threshold q_hat.
        """
        n = len(scores)
        if n < self.n_bins * 2:
            # Not enough data for cross-conformal; fall back to split
            return self._compute_quantile(scores)

        rng = np.random.default_rng(42)
        indices = rng.permutation(n)
        scores_arr = np.asarray(scores, dtype=np.float64)
        fold_size = n // self.n_bins

        q_hats: list[float] = []
        for fold in range(self.n_bins):
            start = fold * fold_size
            end = start + fold_size if fold < self.n_bins - 1 else n
            fold_scores = scores_arr[indices[start:end]].tolist()
            q_fold = self._compute_quantile(fold_scores)
            q_hats.append(q_fold)

        return float(np.mean(q_hats))

    # ------------------------------------------------------------------
    # Prediction set generation
    # ------------------------------------------------------------------

    def predict_set(
        self,
        distances: FloatArray,
        labels: LabelArray,
        top_prediction: str | int,
        confidence: float,
    ) -> ConformalPredictionSet:
        """Generate a conformal prediction set for a query.

        Args:
            distances: [N] array of distances to each candidate class.
            labels: [N] array of class labels.
            top_prediction: The model's top-1 prediction.
            confidence: Calibrated confidence of the top prediction.

        Returns:
            ConformalPredictionSet with included classes and metadata.
        """
        result = ConformalPredictionSet(
            alpha=self.alpha,
            prediction=top_prediction,
            confidence=confidence,
        )

        # Not enough calibration data: the top-1 alone, and said plainly. This
        # used to report coverage_estimate = 1 - alpha and q_hat = 1.0 -- the
        # target restated as if measured, on a set whose real coverage is the
        # top-1 accuracy (about 74% on the harness data, against a 95% claim).
        # A consumer could not tell it from a calibrated singleton (#80).
        if len(self._calibration_scores) < self.min_calibration_size:
            result.prediction_set = {top_prediction}
            result.set_size = 1
            result.q_hat = float("nan")
            result.coverage_estimate = float("nan")
            result.calibrated = False
            return result

        # Compute quantile threshold based on mode
        if self.mode == "cross":
            q_hat = self._compute_cross_quantile(self._calibration_scores)
        else:
            q_hat = self._compute_quantile(self._calibration_scores)
        result.q_hat = q_hat
        result.coverage_estimate = self.empirical_coverage

        # Build prediction set: include classes with score <= q_hat
        prediction_set: set[str | int] = set()
        for i in range(len(distances)):
            label = labels[i]
            score = self.nonconformity(distances, labels, label)

            if score <= q_hat:
                prediction_set.add(label)

        # Always include the top prediction for safety
        prediction_set.add(top_prediction)
        result.prediction_set = prediction_set
        result.set_size = len(prediction_set)

        return result

    def predict_set_class_conditional(
        self,
        distances: FloatArray,
        labels: LabelArray,
        top_prediction: str | int,
        confidence: float,
    ) -> ConformalPredictionSet:
        """Generate prediction set using class-conditional quantiles.

        More adaptive than global quantiles: each class uses its own
        score distribution. Requires sufficient per-class calibration data.

        Args:
            distances: [N] array of distances to candidate classes.
            labels: [N] array of class labels.
            top_prediction: Top-1 prediction.
            confidence: Calibrated confidence.

        Returns:
            ConformalPredictionSet with class-conditional thresholds.
        """
        result = ConformalPredictionSet(
            alpha=self.alpha,
            prediction=top_prediction,
            confidence=confidence,
        )

        n_total = sum(len(s) for s in self._class_scores.values())
        if n_total < self.min_calibration_size:
            result.prediction_set = {top_prediction}
            result.set_size = 1
            result.q_hat = 1.0
            result.coverage_estimate = 1.0 - self.alpha
            return result

        prediction_set: set[str | int] = set()
        q_hats: list[float] = []

        for i in range(len(distances)):
            label = labels[i]
            class_scores = self._class_scores.get(label, [])
            if len(class_scores) >= 3:
                q_class = self._compute_quantile(class_scores)
            else:
                q_class = self._compute_quantile(self._calibration_scores)
            q_hats.append(q_class)

            score = self.nonconformity(distances, labels, label)

            if score <= q_class:
                prediction_set.add(label)

        prediction_set.add(top_prediction)
        result.prediction_set = prediction_set
        result.set_size = len(prediction_set)
        result.q_hat = float(np.mean(q_hats)) if q_hats else 1.0
        result.coverage_estimate = self.empirical_coverage

        return result

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_calibration_summary(self) -> dict[str, float]:
        """Return diagnostic summary of calibration state."""
        return {
            "calibration_size": float(len(self._calibration_scores)),
            "min_informative_size": float(self.min_informative_size),
            "empirical_coverage": float(self.empirical_coverage),
            "target_coverage": float(1.0 - self.alpha),
            "q_hat": float(self._compute_quantile(self._calibration_scores))
            if self._calibration_scores
            else 1.0,
            "mean_score": float(np.mean(self._calibration_scores))
            if self._calibration_scores
            else 0.0,
            "std_score": float(np.std(self._calibration_scores))
            if self._calibration_scores
            else 0.0,
            "num_classes_with_scores": float(len(self._class_scores)),
        }

    def reset(self) -> None:
        """Clear all calibration data and tracking counters."""
        self._calibration_scores.clear()
        self._class_scores.clear()
        self._total_predictions = 0
        self._covered = 0
