"""The Tambua engine: AdaptShot wrapped in whatever domain the config describes.

Adds to `FewShotLearner`:
  - a validated domain configuration (see `tambua.config`)
  - human-readable results -- local label, advice, severity -- for each prediction
  - session history, so corrections and accuracy are visible
  - batch prediction and CSV export
  - model save/load and a health report

Nothing in this module knows what it is classifying. Every domain-specific word
reaching a user comes out of the config file.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

import numpy as np

from adaptshot import AdaptShotConfig, FewShotLearner
from adaptshot.utils.exceptions import AdaptShotError, ConfigValidationError
from tambua.config import ClassInfo, TambuaConfig, load_config

#: The config loaded when the caller names none. MziziGuard is the flagship
#: domain, but it is one config among several, not a special case in the code.
DEFAULT_CONFIG = "maize"


def bundled_configs() -> list[str]:
    """Names of every configuration shipped with this installation."""

    return sorted(
        entry.name.removesuffix(".yaml")
        for entry in (resources.files("tambua") / "configs").iterdir()
        if entry.name.endswith(".yaml")
    )


def bundled_config(name: str) -> str:
    """Path to a configuration shipped inside the package.

    The configs are package data rather than files beside the source tree, so
    they survive `pip install` -- an installed application with no config on disk
    would have nothing to run.

    Args:
        name: Config stem, e.g. "maize" or "solar_panel".

    Returns:
        Absolute path to the YAML file.

    Raises:
        ConfigValidationError: If no config of that name ships with the package.
    """

    path = resources.files("tambua") / "configs" / f"{name}.yaml"
    if not path.is_file():
        raise ConfigValidationError(
            f"no bundled config named {name!r}. "
            f"Available: {', '.join(bundled_configs())}. "
            "For your own config, pass its path directly."
        )
    return str(path)

# Broad handlers below are boundaries too -- one bad image must not abort a batch,
# and a failed correction must not lose the session. They log before returning.
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


#: Severity reported for a label the config does not describe -- one added by a
#: human correction, or restored from a model saved under another config. It is
#: deliberately outside the config vocabulary: it records the absence of a
#: description rather than a level of urgency, and must not be mistaken for one.
UNDESCRIBED_SEVERITY = "undescribed"


class ImageFolderError(AdaptShotError):
    """A training folder is not usable, with every reason stated.

    Its own type rather than a bare ValueError: the interface catches this to
    show the problems to the person who chose the folder, and must not swallow
    an unrelated ValueError from somewhere deeper while doing so.
    """


def combined_action(members: Sequence[ClassInfo]) -> str:
    """The advice to give for a prediction set with more than one member.

    If every member calls for the same thing, that is the answer, and the
    ambiguity does not matter -- which is the most useful case conformal
    prediction produces: "it is one of these two, and either way, do this."

    When the actions differ, they are shown side by side rather than resolved.
    Picking one would be inventing a recommendation nobody wrote, and the person
    reading it cannot tell that is what happened.
    """

    if not members:
        return "No class was plausible enough to name. Ask someone who can look at it."
    if len(members) == 1:
        return members[0].action

    actions = {member.action for member in members}
    if len(actions) == 1:
        return f"All {len(members)} possibilities call for the same thing: {members[0].action}"

    lines = [
        f"These {len(members)} possibilities call for different things, "
        "so this needs a decision rather than an instruction:"
    ]
    lines.extend(f"  - {member.local_name}: {member.action}" for member in members)
    return "\n".join(lines)


def _undescribed(label: str) -> ClassInfo:
    """A placeholder for a label with no entry in the loaded config."""

    return ClassInfo(
        key=label,
        local_name=label,
        action="No advice is configured for this label.",
        description="",
        severity=UNDESCRIBED_SEVERITY,
        domain="",
    )


@dataclass
class Identification:
    """One prediction, with the human-readable context the config supplies."""
    label: str
    local_name: str
    confidence: float
    raw_confidence: float
    action: str
    severity: str
    ood_flag: bool
    uncertainty_flag: bool
    act_action: str
    distance_to_prototype: float
    calibrated_ece: float

    #: Conformal prediction set: the labels that, together, carry the coverage
    #: guarantee. A single label with a confidence number is a claim about one
    #: answer; this is a claim about a set, and it is the claim that can actually
    #: be checked. Empty when the model has abstained.
    prediction_set: tuple[str, ...] = ()

    #: The miscoverage rate the set was built at. 0.1 means the target is that
    #: the truth falls inside the set nine times in ten.
    alpha: float = 0.0

    #: Whether `empirical_coverage` was measured, or is merely the target
    #: restated. Until enough corrections have accumulated, conformal has nothing
    #: to calibrate against and returns the top label as a singleton; presenting
    #: `1 - alpha` then would be quoting an aspiration as a measurement.
    coverage_is_measured: bool = False
    empirical_coverage: float = 0.0
    calibration_size: int = 0

    timestamp: float = field(default_factory=time.time)

    @property
    def is_abstention(self) -> bool:
        """True when the set says nothing useful and a human should decide.

        Two ways that happens: an empty set (nothing was plausible enough) and a
        set containing every class the model knows (everything was). Both mean
        the same thing to the person holding the phone.
        """

        return len(self.prediction_set) == 0


@dataclass
class SessionHistory:
    """Tracks predictions and corrections in the current session."""
    predictions: list[Identification] = field(default_factory=list)
    corrections: list[dict[str, Any]] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def record_prediction(self, result: Identification) -> None:
        self.predictions.append(result)

    def record_correction(
        self, image_path: str, predicted: str, corrected: str, weight: float
    ) -> None:
        self.corrections.append({
            "image_path": image_path,
            "predicted": predicted,
            "corrected": corrected,
            "weight": weight,
            "timestamp": time.time(),
        })

    @property
    def total_predictions(self) -> int:
        return len(self.predictions)

    @property
    def total_corrections(self) -> int:
        return len(self.corrections)

    @property
    def accuracy(self) -> float:
        """Rough accuracy: (total - corrections) / total."""
        if self.total_predictions == 0:
            return 1.0
        return max(0.0, (self.total_predictions - self.total_corrections) / self.total_predictions)

    def summary(self) -> dict[str, Any]:
        return {
            "total_predictions": self.total_predictions,
            "total_corrections": self.total_corrections,
            "accuracy": round(self.accuracy, 3),
            "session_duration_seconds": round(time.time() - self.start_time, 1),
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TambuaEngine:
    """Few-shot identification for whatever domain the config describes.

    Usage::

        engine = TambuaEngine()                       # ships with maize.yaml
        engine.load_images_from_dir("my_photos/")     # one folder per class
        result = engine.identify("photo.jpg")
        print(result.local_name, result.confidence, result.action)

        # Human correction
        engine.teach("photo.jpg", true_label="northern_leaf_blight")

        # Save for next session
        engine.save_model("models/session_1.json")

    Point it at another config and it is a different application::

        engine = TambuaEngine(bundled_config("solar_panel"))
    """

    def __init__(self, config_path: str | None = None) -> None:
        """Initialise from a YAML domain configuration.

        Args:
            config_path: Path to a config file. Defaults to the bundled
                `maize.yaml`.

        Raises:
            ConfigValidationError: If the config is missing or invalid. The
                message names the file, the line and the fix for every problem.
        """
        self.config_path = config_path or bundled_config(DEFAULT_CONFIG)
        self.cfg: TambuaConfig = load_config(self.config_path)

        self.engine_cfg = self._build_engine_config()
        self.classes: dict[str, ClassInfo] = dict(self.cfg.classes)

        # AdaptShot learner (lazy-init)
        self._learner: FewShotLearner | None = None

        # Session state
        self.history = SessionHistory()
        self._last_image_path: str | None = None
        self._data_dir: str | None = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """What this configuration calls the application."""
        return self.cfg.application.name

    def _build_engine_config(self) -> AdaptShotConfig:
        eng = self.cfg.engine
        return AdaptShotConfig(
            backbone=eng.backbone,
            device=eng.device,
            seed=eng.seed,
            inference_mode=eng.inference_mode,
            similarity_metric=eng.similarity_metric,
            eco_mode=eng.eco_mode,
            enable_ood_detection=eng.enable_ood_detection,
            conformal_alpha=eng.conformal_alpha,
            conformal_mode=eng.conformal_mode,
        )

    @property
    def known_labels(self) -> list[str]:
        """Every label the engine can currently predict."""
        return sorted(self.classes)

    @property
    def is_initialized(self) -> bool:
        return self._learner is not None

    # ------------------------------------------------------------------
    # Learner access
    # ------------------------------------------------------------------

    @property
    def learner(self) -> FewShotLearner:
        if self._learner is None:
            self._learner = FewShotLearner(config=self.engine_cfg)
        return self._learner

    @property
    def is_trained(self) -> bool:
        """True if the learner has support images loaded."""
        return self._learner is not None and len(self._learner._sim_embeddings) > 0

    # ------------------------------------------------------------------
    # Initialization: sample data or real images
    # ------------------------------------------------------------------

    def load_images_from_dir(
        self,
        image_dir: str,
        max_per_class: int = 0,
    ) -> int:
        """Load real images organized in class subdirectories.

        Expected structure::

            image_dir/
                <class_key>/
                    img1.png
                    img2.jpg
                <another_class_key>/
                    img3.png
                    ...

        The directory names must match class keys in the loaded config, or the
        model will learn labels the interface cannot describe.

        Args:
            image_dir: Root directory with one subfolder per class.
            max_per_class: Max images per class (0 = unlimited).

        Returns:
            Number of support images loaded.

        Raises:
            ImageFolderError: If the folder cannot support training. The message
                names every problem and its remedy.
        """
        from tambua import data as image_data

        problems = image_data.inspect_folder(image_dir, self.cfg.labels)
        if problems:
            # Reported before training rather than after, so the cause is visible
            # and no time is spent learning from a folder that cannot support it.
            raise ImageFolderError(image_data.render_problems(problems))

        paths, labels = image_data.load_from_folders(image_dir, max_per_class)
        self.learner.load_support_images(paths, labels)
        self._data_dir = image_dir
        return len(paths)

    # ------------------------------------------------------------------
    # Prediction / Diagnosis
    # ------------------------------------------------------------------

    def identify(
        self,
        image: str | np.ndarray | Any,
    ) -> Identification:
        """Identify one image.

        Args:
            image: File path, NumPy array, or PIL Image.

        Returns:
            An `Identification` carrying the predicted label together with the
            local name, advice and severity the config gives it.
        """
        if not self.is_trained:
            raise RuntimeError(
                "No support images loaded. Call load_images_from_dir() with a "
                "folder of photographs, one subfolder per class."
            )

        result = self.learner.predict(image)

        # Store for later correction
        if isinstance(image, str):
            self._last_image_path = image

        label = str(result.prediction)
        info = self.classes.get(label) or _undescribed(label)

        conformal = self.learner.conformal
        # The set is only meaningful once conformal has calibration scores to
        # work from. Before that it returns the top label as a singleton and a
        # coverage figure equal to `1 - alpha` -- the target restated, not
        # measured. Saying so is the difference between this and #17.
        measured = conformal.calibration_size >= conformal.min_calibration_size
        raw_set = result.conformal_set or []

        identification = Identification(
            prediction_set=tuple(sorted(str(member) for member in raw_set)),
            alpha=float(conformal.alpha),
            coverage_is_measured=measured,
            empirical_coverage=float(conformal.empirical_coverage) if measured else 0.0,
            calibration_size=int(conformal.calibration_size),
            label=label,
            local_name=info.local_name,
            confidence=float(result.calibrated_confidence),
            raw_confidence=float(result.raw_confidence),
            action=info.action,
            severity=info.severity,
            ood_flag=result.ood_flag,
            uncertainty_flag=result.uncertainty_flag,
            act_action=result.act_action,
            distance_to_prototype=result.distance_to_prototype,
            calibrated_ece=result.debiased_ece,
        )

        self.history.record_prediction(identification)
        return identification

    # ------------------------------------------------------------------
    # Human-in-the-loop correction
    # ------------------------------------------------------------------

    def teach(
        self,
        image_path: str,
        true_label: str,
        confidence_weight: float = 1.0,
    ) -> dict[str, Any]:
        """Teach the model by correcting a prediction.

        Args:
            image_path: Path to the image being corrected.
            true_label: The correct label.
            confidence_weight: How confident you are (0.0–1.0).

        Returns:
            Correction result from the AdaptShot pipeline.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")

        result = self.learner.correct(
            image_path=image_path,
            true_label=true_label,
            confidence_weight=confidence_weight,
        )

        predicted = result.get("predicted_label", "unknown")
        self.history.record_correction(
            image_path=image_path,
            predicted=str(predicted),
            corrected=true_label,
            weight=confidence_weight,
        )

        # A correction can introduce a label the config never described. Record it
        # so predictions do not crash, but mark it undescribed rather than
        # inventing advice and a severity that nobody wrote.
        if true_label not in self.classes:
            self.classes[true_label] = _undescribed(true_label)

        return result

    def teach_from_ui(
        self,
        true_label: str,
        confidence_weight: float = 1.0,
    ) -> str:
        """Correction convenience for Gradio UI (uses last predicted image)."""
        if self._last_image_path is None:
            return "❌ Make a prediction first before correcting."
        try:
            result = self.teach(
                image_path=self._last_image_path,
                true_label=true_label,
                confidence_weight=confidence_weight,
            )
            fine_tuned = result.get("fine_tuned", False)
            buffer_size = result.get("buffer_size", 0)
            return (
                f"✅ Correction recorded! "
                f"Fine-tuned: {fine_tuned}, Buffer: {buffer_size}"
            )
        except Exception as exc:
            logger.exception("correction failed")
            return f"❌ Correction failed: {exc}"

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def batch_identify(
        self,
        image_paths: list[str],
    ) -> list[Identification]:
        """Identify a batch of images, one result per input.

        A failure on one image is recorded as a failed row rather than aborting
        the batch: someone processing a folder of field photographs should not
        lose forty results to one corrupt file.

        Args:
            image_paths: Paths to image files.

        Returns:
            One `Identification` per input path, in order.
        """
        results: list[Identification] = []
        for path in image_paths:
            try:
                result = self.identify(path)
                results.append(result)
            except Exception:
                logger.exception("batch item failed: %s", path)
                results.append(Identification(
                    prediction_set=(),
                    label="error",
                    local_name="error",
                    confidence=0.0,
                    raw_confidence=0.0,
                    action="Could not process this image.",
                    severity=UNDESCRIBED_SEVERITY,
                    ood_flag=True,
                    uncertainty_flag=True,
                    act_action="ERROR",
                    distance_to_prototype=0.0,
                    calibrated_ece=0.0,
                ))
        return results

    def batch_to_csv(self, results: list[Identification]) -> str:
        """Convert batch results to a CSV string."""
        lines = [
            "image_path,label,local_name,confidence,severity,action,ood_flag,"
            "prediction_set,set_size,alpha,coverage_measured"
        ]
        for r in results:
            members = " ".join(r.prediction_set)
            lines.append(
                f"{getattr(r, 'image_path', 'unknown')},"
                f"{r.label},{r.local_name},{r.confidence:.3f},"
                f"{r.severity},\"{r.action}\",{r.ood_flag},"
                f"\"{members}\",{len(r.prediction_set)},{r.alpha:.2f},"
                f"{r.coverage_is_measured}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # System health
    # ------------------------------------------------------------------

    def system_health(self) -> dict[str, Any]:
        """Return a combined health report (calibration + session stats)."""
        if not self.is_trained:
            return {"status": "not_trained", "message": "Load support images first."}

        calib = self.learner.calibration_report()
        session = self.history.summary()
        conformal = self.learner.conformal
        measured = conformal.calibration_size >= conformal.min_calibration_size
        return {
            "status": "healthy",
            "calibration": calib,
            "conformal": {
                "alpha": float(conformal.alpha),
                "target_coverage": 1.0 - float(conformal.alpha),
                "calibration_scores": int(conformal.calibration_size),
                # Absent rather than zero when unmeasured: a reader scanning a
                # dashboard reads 0.0 as "no coverage", which is a different and
                # equally wrong claim to "not measured yet".
                "empirical_coverage": (
                    float(conformal.empirical_coverage) if measured else None
                ),
                "scores_needed": max(
                    0, int(conformal.min_calibration_size - conformal.calibration_size)
                ),
            },
            "session": session,
            "config": {
                "backbone": self.engine_cfg.backbone,
                "device": self.engine_cfg.device,
                "eco_mode": self.engine_cfg.eco_mode,
                "known_classes": len(self.classes),
                "support_size": calib.get("support_size", 0),
            },
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str) -> str:
        """Save the AdaptShot learner state to disk.

        Saves: {path}.json, {path}.embeddings.npy, {path}.head.pt
        """
        if not self.is_trained:
            raise RuntimeError("Nothing to save — model is not trained.")
        self.learner.save(path)
        return f"Model saved to {path}"

    def load_model(self, path: str) -> int:
        """Load a previously saved learner state from disk.

        Returns:
            Number of support images restored.
        """
        from adaptshot.core.learner import FewShotLearner as FSL

        self._learner = FSL.load(path)
        return len(self._learner._sim_embeddings) if self._learner else 0

    # ------------------------------------------------------------------
    # Re-export key info for UI display
    # ------------------------------------------------------------------

    def set_members(self, identification: Identification) -> list[ClassInfo]:
        """The configured description of every label in a prediction set."""

        return [self.label_to_info(label) for label in identification.prediction_set]

    def advice_for(self, identification: Identification) -> str:
        """The advice covering every member of the set, or the stated conflict."""

        return combined_action(self.set_members(identification))

    def label_to_info(self, label: str) -> ClassInfo:
        """The configured description of a label, or an undescribed placeholder."""
        return self.classes.get(label) or _undescribed(label)

    def all_labels(self) -> list[str]:
        """Every predictable label, sorted alphabetically."""
        return self.known_labels
