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
    timestamp: float = field(default_factory=time.time)


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

        identification = Identification(
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
        lines = ["image_path,label,local_name,confidence,severity,action,ood_flag"]
        for r in results:
            lines.append(
                f"{getattr(r, 'image_path', 'unknown')},"
                f"{r.label},{r.local_name},{r.confidence:.3f},"
                f"{r.severity},\"{r.action}\",{r.ood_flag}"
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
        return {
            "status": "healthy",
            "calibration": calib,
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

    def label_to_info(self, label: str) -> ClassInfo:
        """The configured description of a label, or an undescribed placeholder."""
        return self.classes.get(label) or _undescribed(label)

    def all_labels(self) -> list[str]:
        """Every predictable label, sorted alphabetically."""
        return self.known_labels
