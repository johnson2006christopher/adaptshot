"""MziziGuard Engine: crop disease detection powered by AdaptShot.

Wraps FewShotLearner with:
  - Crop configuration from YAML
  - Swahili/English translation maps
  - Model save/load persistence
  - Per-session history tracking
  - Batch prediction
  - System health reporting
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import yaml

from adaptshot import AdaptShotConfig, FewShotLearner
from adaptshot.utils.exceptions import ConfigValidationError

# Broad handlers below are boundaries too -- one bad image must not abort a batch,
# and a failed correction must not lose the session. They log before returning.
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class DiseaseInfo:
    """Structured disease metadata from config."""
    key: str
    swahili: str
    action: str
    description: str
    severity: str  # low | moderate | high | critical
    crop: str = ""


@dataclass
class DiagnosisResult:
    """Single prediction result with human-readable context."""
    label: str
    swahili: str
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
    predictions: list[DiagnosisResult] = field(default_factory=list)
    corrections: list[dict[str, Any]] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def record_prediction(self, result: DiagnosisResult) -> None:
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
# MziziGuard Engine
# ---------------------------------------------------------------------------


class MziziGuard:
    """Crop disease detection engine powered by AdaptShot few-shot learning.

    Usage::

        guard = MziziGuard("config.yaml")
        guard.initialize_with_samples(n_support=5)  # or guard.load_images_from_dir(...)
        result = guard.diagnose("photo_of_leaf.jpg")
        print(result.swahili, result.confidence, result.action)

        # Human correction
        guard.teach("photo_of_leaf.jpg", true_label="northern_leaf_blight")

        # Save for next session
        guard.save_model("models/session_1.json")
    """

    def __init__(self, config_path: str | None = None) -> None:
        """Initialize MziziGuard from a YAML configuration file.

        Args:
            config_path: Path to config.yaml. If None, searches relative to
                         the package directory.
        """
        self.config_path = self._resolve_config_path(config_path)
        self.cfg = self._load_config(self.config_path)

        # Build AdaptShot engine config
        self.engine_cfg = self._build_engine_config()

        # Build disease maps
        self.diseases: dict[str, DiseaseInfo] = self._build_disease_map()

        # AdaptShot learner (lazy-init)
        self._learner: FewShotLearner | None = None

        # Session state
        self.history = SessionHistory()
        self._last_image_path: str | None = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_config_path(config_path: str | None) -> str:
        if config_path and os.path.isfile(config_path):
            return config_path
        # Search relative to this file's directory
        candidates = [
            config_path or "",
            os.path.join(os.path.dirname(__file__), "config.yaml"),
        ]
        for path in candidates:
            if path and os.path.isfile(path):
                return path
        raise FileNotFoundError(
            "config.yaml not found. Either pass config_path or place "
            "config.yaml next to engine.py."
        )

    @staticmethod
    def _load_config(path: str) -> dict[str, Any]:
        """Load and shape-check a domain config.

        `yaml.safe_load` returns whatever the document contains -- ``None`` for an
        empty file, a list for a top-level sequence. Returning that unchecked
        turns a one-character typo in the config into an `AttributeError` several
        frames away, which is the failure mode #47 exists to prevent. Fail here,
        naming the file.
        """

        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)

        if loaded is None:
            raise ConfigValidationError(f"{path} is empty; a domain config is required")
        if not isinstance(loaded, dict):
            raise ConfigValidationError(
                f"{path} must contain a mapping at the top level, "
                f"found {type(loaded).__name__}"
            )
        return loaded

    def _build_engine_config(self) -> AdaptShotConfig:
        eng = self.cfg.get("engine", {})
        return AdaptShotConfig(
            backbone=eng.get("backbone", "resnet18"),
            device=eng.get("device", "cpu"),
            seed=eng.get("seed", 42),
            inference_mode=eng.get("inference_mode", "prototypical"),
            similarity_metric=eng.get("similarity_metric", "euclidean"),
            eco_mode=eng.get("eco_mode", True),
            enable_ood_detection=eng.get("enable_ood_detection", True),
        )

    def _build_disease_map(self) -> dict[str, DiseaseInfo]:
        """Flatten all crops/diseases into a label → DiseaseInfo map."""
        result: dict[str, DiseaseInfo] = {}
        crops = self.cfg.get("crops", {})
        for crop_name, crop_data in crops.items():
            for disease_key, disease_data in crop_data.get("diseases", {}).items():
                result[disease_key] = DiseaseInfo(
                    key=disease_key,
                    swahili=disease_data.get("swahili", disease_key),
                    action=disease_data.get("action", "Consult extension officer."),
                    description=disease_data.get("description", ""),
                    severity=disease_data.get("severity", "moderate"),
                    crop=crop_name,
                )
        return result

    @property
    def known_labels(self) -> list[str]:
        """All disease labels the system knows about."""
        return sorted(self.diseases.keys())

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

    def initialize_with_samples(
        self,
        n_support: int = 5,
        data_dir: str | None = None,
        seed: int = 42,
    ) -> int:
        """Generate synthetic sample images and load into the learner.

        Args:
            n_support: Number of support images per disease class.
            data_dir: Where to write images. Uses temp dir if None.
            seed: Random seed for reproducibility.

        Returns:
            Number of support images loaded.
        """
        from . import data as mzizi_data

        if data_dir is None:
            import tempfile
            data_dir = tempfile.mkdtemp(prefix="mziziguard_samples_")

        support_paths, support_labels, _, _ = mzizi_data.generate_samples(
            output_dir=data_dir,
            n_support=n_support,
            n_query=0,
            seed=seed,
        )
        self.learner.load_support_images(support_paths, support_labels)
        self._data_dir = data_dir
        return len(support_paths)

    def load_images_from_dir(
        self,
        image_dir: str,
        max_per_class: int = 0,
    ) -> int:
        """Load real images organized in class subdirectories.

        Expected structure::

            image_dir/
                healthy_maize/
                    img1.png
                    img2.jpg
                northern_leaf_blight/
                    img3.png
                    ...

        Args:
            image_dir: Root directory with one subfolder per class.
            max_per_class: Max images per class (0 = unlimited).

        Returns:
            Number of support images loaded.
        """
        from . import data as mzizi_data

        paths, labels = mzizi_data.load_from_folders(image_dir, max_per_class)
        if not paths:
            raise ValueError(f"No images found in {image_dir}")
        self.learner.load_support_images(paths, labels)
        self._data_dir = image_dir
        return len(paths)

    # ------------------------------------------------------------------
    # Prediction / Diagnosis
    # ------------------------------------------------------------------

    def diagnose(
        self,
        image: str | np.ndarray | Any,
    ) -> DiagnosisResult:
        """Run disease diagnosis on an image.

        Args:
            image: File path, NumPy array, or PIL Image.

        Returns:
            DiagnosisResult with prediction, confidence, Swahili name, and action.
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model not trained. Call initialize_with_samples() or "
                "load_images_from_dir() first."
            )

        result = self.learner.predict(image)

        # Store for later correction
        if isinstance(image, str):
            self._last_image_path = image

        disease_info = self.diseases.get(
            str(result.prediction),
            DiseaseInfo(
                key=str(result.prediction),
                swahili=str(result.prediction),
                action="Consult extension officer.",
                description="Unknown class.",
                severity="unknown",
            ),
        )

        diagnosis = DiagnosisResult(
            label=str(result.prediction),
            swahili=disease_info.swahili,
            confidence=float(result.calibrated_confidence),
            raw_confidence=float(result.raw_confidence),
            action=disease_info.action,
            severity=disease_info.severity,
            ood_flag=result.ood_flag,
            uncertainty_flag=result.uncertainty_flag,
            act_action=result.act_action,
            distance_to_prototype=result.distance_to_prototype,
            calibrated_ece=result.debiased_ece,
        )

        self.history.record_prediction(diagnosis)
        return diagnosis

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
            true_label: The correct disease label.
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

        # If the corrected label is new, add it to our disease map
        if true_label not in self.diseases:
            self.diseases[true_label] = DiseaseInfo(
                key=true_label,
                swahili=true_label,
                action="Consult extension officer.",
                description="Added via human correction.",
                severity="moderate",
            )

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

    def batch_diagnose(
        self,
        image_paths: list[str],
    ) -> list[DiagnosisResult]:
        """Run diagnosis on a batch of images.

        Args:
            image_paths: List of paths to image files.

        Returns:
            List of DiagnosisResult, one per image.
        """
        results: list[DiagnosisResult] = []
        for path in image_paths:
            try:
                result = self.diagnose(path)
                results.append(result)
            except Exception:
                logger.exception("batch item failed: %s", path)
                results.append(DiagnosisResult(
                    label="error",
                    swahili="hitilafu",
                    confidence=0.0,
                    raw_confidence=0.0,
                    action="Could not process this image.",
                    severity="unknown",
                    ood_flag=True,
                    uncertainty_flag=True,
                    act_action="ERROR",
                    distance_to_prototype=0.0,
                    calibrated_ece=0.0,
                ))
        return results

    def batch_to_csv(self, results: list[DiagnosisResult]) -> str:
        """Convert batch results to CSV string."""
        lines = ["image_path,label,swahili,confidence,severity,action,ood_flag"]
        for r in results:
            lines.append(
                f"{getattr(r, 'image_path', 'unknown')},"
                f"{r.label},{r.swahili},{r.confidence:.3f},"
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
                "known_classes": len(self.diseases),
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

    def label_to_info(self, label: str) -> DiseaseInfo:
        """Get DiseaseInfo for a given label string."""
        return self.diseases.get(
            label,
            DiseaseInfo(
                key=label,
                swahili=label,
                action="Consult extension officer.",
                description="",
                severity="unknown",
            ),
        )

    def all_disease_labels(self) -> list[str]:
        """All disease labels sorted alphabetically."""
        return self.known_labels
