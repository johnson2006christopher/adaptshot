"""Unified FewShotLearner API for AdaptShot.

Exposes a single, high-level interface that orchestrates embedding extraction,
similarity search, calibration, ACT gating, human feedback routing, CA-EWC
fine-tuning, UP-UGF buffer management, conformal prediction, contrastive
prototype learning, advanced uncertainty quantification, and explainability.
"""

from __future__ import annotations

import json
import logging
import hashlib
import os
import tempfile
import warnings
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, TensorDataset

from ..config.settings import AdaptShotConfig
from ..training.feedback_router import Correction, FeedbackRouter
from ..training.finetune import CAEWCFinetuner
from ..training.up_ugf import UPUGFPruner
from ..utils.exceptions import (
    AdaptShotError,
    BufferCapacityError,
    CalibrationNotReadyError,
    ConfigValidationError,
    InvalidImageError,
)
from ..utils.migrations import migrate_v0_1_0_to_v0_1_1
from .act import ACTEngine
from .calibration import CalibrationEngine
from .conformal import ConformalEngine, ConformalPredictionSet
from .contrastive import ContrastivePrototypeLearner
from .uncertainty import UncertaintyQuantifier
from .explain import ExplainabilityEngine, ExplanationResult
from .extractor import (
    BACKBONE_OUTPUT_DIM,
    EmbeddingCache,
    compute_preview_signature,
    extract_embedding,
)
from .similarity import (
    compute_class_prototypes,
    euclidean_distance_numpy,
    find_nearest_neighbor,
    find_nearest_prototype,
)

logger = logging.getLogger(__name__)
SCHEMA_VERSION = "0.2.0"


@dataclass
class PredictionResult:
    """Structured return type for predict() calls."""

    prediction: Union[str, int]
    raw_confidence: float
    calibrated_confidence: float
    neighbor_idx: int
    uncertainty_flag: bool
    act_action: str
    distance_to_prototype: float = 0.0
    prototype_margin: float = 0.0
    ood_flag: bool = False
    debiased_ece: float = 0.0
    # v0.2.0 fields
    conformal_set: Optional[List[Union[str, int]]] = None
    uncertainty_report: Optional[Dict[str, float]] = None
    nearest_neighbors: Optional[List[Dict[str, Any]]] = None


class FewShotLearner:
    """Main entry point for AdaptShot few-shot learning and inference."""

    def __init__(self, config: Optional[AdaptShotConfig] = None, **kwargs: Any) -> None:
        """Initialize learner state.

        Args:
            config: AdaptShotConfig instance, or pass kwargs to construct one.
            **kwargs: Keyword args used only when config is not provided.
        """
        self.config = config or AdaptShotConfig(**kwargs)
        self._validate_config(self.config)

        self.calibrator = CalibrationEngine(
            n_bins=self.config.ece_n_bins,
            window_size=self.config.max_buffer_size * 2,
            temperature_init=self.config.temperature_init,
            method=self.config.calibration_method,
            evaluation_bins=self.config.calibration_eval_bins,
        )
        self.act = ACTEngine(n_classes=200)

        # v0.2.0: Advanced engines
        self.conformal = ConformalEngine(
            alpha=self.config.conformal_alpha,
            mode=self.config.conformal_mode,
        )
        self.contrastive = ContrastivePrototypeLearner()
        self.uncertainty_q = UncertaintyQuantifier(
            ood_percentile=self.config.ood_threshold_quantile * 100,
        )
        self.explainer = ExplainabilityEngine()

        self._sim_embeddings: List[np.ndarray] = []
        self._sim_labels: List[Union[str, int]] = []
        self._sim_access_times: List[float] = []
        self._sim_uncertainties: List[float] = []
        self._sim_preview_signatures: List[np.ndarray] = []
        self._prototype_embeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self._prototype_labels: np.ndarray = np.asarray([], dtype=object)
        self._prototype_counts: np.ndarray = np.asarray([], dtype=np.int64)
        self._ood_distance_threshold: float = self.config.ood_absolute_min_distance

        self.pruner = UPUGFPruner(
            capacity=self.config.max_buffer_size,
            uncertainty_weight=1.0,
            recency_weight=1.0,
            redundancy_weight=1.0,
        )

        self.finetuner: Optional[CAEWCFinetuner] = None
        self._model_head: Optional[torch.nn.Linear] = None

        self.router = FeedbackRouter(
            buffer_capacity=self.config.max_buffer_size,
            fine_tune_trigger_threshold=max(5, self.config.max_buffer_size // 10),
            calibrator=self.calibrator,
            finetune_fn=self._trigger_finetune,
        )

        self._label_to_idx: Dict[Union[str, int], int] = {}
        self._idx_to_label: Dict[int, Union[str, int]] = {}
        self._is_initialized = False
        self._embedding_cache = EmbeddingCache()

    def __repr__(self) -> str:
        """Return concise internal state for debugging and observability."""
        return (
            "FewShotLearner("
            f"initialized={self._is_initialized}, "
            f"support_size={len(self._sim_embeddings)}, "
            f"classes={len(self._label_to_idx)}, "
            f"device='{self.config.device}', "
            f"backbone='{self.config.backbone}', "
            f"inference_mode='{self.config.inference_mode}', "
            f"metric='{self.config.similarity_metric}', "
            f"buffer_capacity={self.config.max_buffer_size}"
            ")"
        )

    def load_support_images(self, image_paths: List[str], labels: List[Union[str, int]]) -> None:
        """Ingest support set and initialize similarity index and CA-EWC head.

        Args:
            image_paths: List of file paths to support images.
            labels: Corresponding class labels.

        Raises:
            ConfigValidationError: If image_paths and labels are malformed.
            InvalidImageError: If any image is missing, unreadable, or not RGB.
            AdaptShotError: If embedding extraction fails or support set is invalid.
        """
        self._validate_support_inputs(image_paths=image_paths, labels=labels)

        self._sim_embeddings.clear()
        self._sim_labels.clear()
        self._sim_access_times.clear()
        self._sim_uncertainties.clear()
        self._sim_preview_signatures.clear()

        current_time = time.time()
        expected_dim: Optional[int] = None

        for path, label in zip(image_paths, labels):
            self._validate_label(label)
            image = self._load_rgb_image_from_path(path)
            preview_signature = compute_preview_signature(image)
            embedding = self._extract_embedding_checked(image=image, source=path)

            if expected_dim is None:
                expected_dim = int(embedding.shape[0])
            if int(embedding.shape[0]) != expected_dim:
                raise AdaptShotError(
                    "Support embeddings must share one dimensionality. "
                    f"Expected {expected_dim}, got {int(embedding.shape[0])} for '{path}'."
                )

            self._sim_embeddings.append(embedding)
            self._sim_labels.append(label)
            self._sim_access_times.append(current_time)
            self._sim_uncertainties.append(0.5)
            self._sim_preview_signatures.append(preview_signature)

        if not self._sim_embeddings:
            raise ConfigValidationError(
                "Support set cannot be empty. Provide at least one RGB image path and label. "
                "See docs/getting-started/quickstart.md."
            )

        self._rebuild_label_index()
        self._rebuild_prototypes()
        self._update_ood_threshold()
        self._init_or_rebuild_model_head(embedding_dim=self._embedding_dim())
        if self._sim_embeddings:
            self._embedding_cache.set(
                self._sim_embeddings[0],
                self._sim_preview_signatures[0],
            )

        # v0.2.0: Fit advanced engines on support data
        support_arr = np.array(self._sim_embeddings, dtype=np.float32)
        label_arr = np.array(self._sim_labels, dtype=object)
        self.uncertainty_q.fit_class_distributions(support_arr, label_arr)
        if self.config.inference_mode == "contrastive":
            self._prototype_embeddings, self._prototype_labels = (
                self.contrastive.refine_prototypes(support_arr, label_arr, seed=self.config.seed)
            )

        self._is_initialized = True

    def predict(self, image: Union[str, Image.Image, np.ndarray]) -> PredictionResult:
        """Run inference with calibration and ACT gating.

        Args:
            image: File path, PIL image, or HWC NumPy array.

        Returns:
            Prediction metadata with calibrated confidence and ACT decision.

        Raises:
            AdaptShotError: If support set is not initialized.
            InvalidImageError: If input image is invalid.
        """
        self._ensure_initialized()
        if not self._sim_embeddings:
            raise AdaptShotError(
                "Support set is empty. Load support examples with load_support_images() "
                "before calling predict()."
            )

        normalized_image = self._normalize_predict_image(image)
        query_emb = self._extract_embedding_checked(image=normalized_image, source="predict")
        support_embeddings = np.array(self._sim_embeddings, dtype=np.float32)
        support_labels = np.array(self._sim_labels, dtype=object)

        if self.config.inference_mode == "prototypical" and self._prototype_embeddings.size > 0:
            pred_label_raw, raw_conf, _, distance_to_prototype, prototype_margin = find_nearest_prototype(
                query=query_emb,
                prototypes=self._prototype_embeddings,
                prototype_labels=self._prototype_labels,
                metric=self.config.similarity_metric,
            )
            pred_label = self._coerce_label(pred_label_raw)
            neighbor_idx = self._nearest_support_index_for_label(query_emb, pred_label)
        else:
            _, raw_conf, neighbor_idx = find_nearest_neighbor(
                query=query_emb,
                support_embeddings=support_embeddings,
                support_labels=support_labels,
                use_faiss=self.config.use_faiss,
                metric=self.config.similarity_metric,
            )
            if neighbor_idx < 0 or neighbor_idx >= len(self._sim_labels):
                raise AdaptShotError(
                    "Nearest-neighbor index is out of bounds. "
                    "Rebuild support set with load_support_images()."
                )
            pred_label = self._coerce_label(self._sim_labels[neighbor_idx])
            distance_to_prototype = self._distance_to_label_prototype(query_emb, pred_label)
            prototype_margin = 0.0

        try:
            calibrated_conf = self._calibrate_or_raise(raw_conf)
        except CalibrationNotReadyError:
            calibrated_conf = self._raw_to_unit_interval(raw_conf)

        recent_unc = float(np.mean(self._sim_uncertainties[-10:])) if self._sim_uncertainties else 0.0
        class_idx = self._label_to_idx.get(pred_label, 0)
        accept, act_action = self.act.should_accept(
            confidence=calibrated_conf,
            class_idx=class_idx,
            recent_incorrect_rate=recent_unc,
            recent_correct_rate=1.0 - recent_unc,
        )

        ood_flag = self._is_out_of_distribution(
            distance_to_prototype=distance_to_prototype,
            prototype_margin=prototype_margin,
        )
        if ood_flag:
            accept = False
            act_action = "REQUEST_FEEDBACK_OOD"

        self._sim_access_times[neighbor_idx] = time.time()
        self._sim_uncertainties[neighbor_idx] = float(np.clip(1.0 - calibrated_conf, 0.0, 1.0))
        self._embedding_cache.set(
            self._sim_embeddings[neighbor_idx],
            self._sim_preview_signatures[neighbor_idx],
        )

        calibration_summary = self.calibrator.calibration_summary()

        # v0.2.0: Conformal prediction set
        prototype_distances = self._compute_all_prototype_distances(query_emb)
        proto_labels = self._prototype_labels
        if self.config.inference_mode == "contrastive" and self.contrastive.is_fitted:
            cf_pred, cf_conf, _ = self.contrastive.nearest_prototype(
                query_emb, self._prototype_embeddings, self._prototype_labels
            )
        conformal_result = self.conformal.predict_set(
            prototype_distances, proto_labels, pred_label, calibrated_conf
        )
        conformal_list = sorted(conformal_result.prediction_set, key=str)

        # v0.2.0: Uncertainty report
        uncertainty_report = self.uncertainty_q.quantify(
            query_emb, support_embeddings, support_labels
        )

        # v0.2.0: Nearest neighbors for explainability
        k_nn = min(5, len(support_embeddings))
        nn_distances = np.sqrt(np.sum((support_embeddings - query_emb) ** 2, axis=1))
        nn_top_idx = np.argsort(nn_distances)[:k_nn]
        nearest_neighbors = [
            {
                "index": int(idx),
                "label": str(self._sim_labels[idx]),
                "distance": float(nn_distances[idx]),
            }
            for idx in nn_top_idx
        ]

        return PredictionResult(
            prediction=pred_label,
            raw_confidence=float(raw_conf),
            calibrated_confidence=float(calibrated_conf),
            neighbor_idx=int(neighbor_idx),
            uncertainty_flag=(not accept) or ood_flag,
            act_action=act_action,
            distance_to_prototype=float(distance_to_prototype),
            prototype_margin=float(prototype_margin),
            ood_flag=bool(ood_flag),
            debiased_ece=float(calibration_summary["debiased_ece"]),
            conformal_set=conformal_list,
            uncertainty_report=uncertainty_report.to_dict(),
            nearest_neighbors=nearest_neighbors,
        )

    def correct(
        self,
        image_path: str,
        true_label: Union[str, int],
        confidence_weight: float = 1.0,
    ) -> Dict[str, Any]:
        """Route a human correction into the continual learning pipeline.

        Args:
            image_path: Path to the corrected image.
            true_label: Human-provided true label.
            confidence_weight: Human confidence in correction in [0.0, 1.0].

        Returns:
            Routing summary including fine-tuning and buffer state.

        Raises:
            AdaptShotError: If support set is not initialized.
            InvalidImageError: If image path or image content is invalid.
            ConfigValidationError: If confidence_weight is outside [0.0, 1.0].
        """
        self._ensure_initialized()
        self._validate_label(true_label)

        if confidence_weight < 0.0 or confidence_weight > 1.0:
            raise ConfigValidationError(
                "confidence_weight must be in [0.0, 1.0]. "
                f"Received {confidence_weight}."
            )

        image = self._load_rgb_image_from_path(image_path)
        query_emb = self._extract_embedding_checked(image=image, source=image_path)

        support_embeddings = np.array(self._sim_embeddings, dtype=np.float32)
        support_labels = np.array(self._sim_labels, dtype=object)
        _, raw_conf, neighbor_idx = find_nearest_neighbor(
            query_emb,
            support_embeddings,
            support_labels,
            use_faiss=self.config.use_faiss,
            metric=self.config.similarity_metric,
        )

        predicted_label = self._sim_labels[int(neighbor_idx)]
        predicted_idx = self._label_to_idx.get(predicted_label, 0)
        corrected_idx = self._ensure_label_index(true_label)

        correction = Correction(
            image_path=image_path,
            predicted_label=predicted_idx,
            corrected_label=corrected_idx,
            raw_confidence=float(raw_conf),
            confidence_weight=float(confidence_weight),
            timestamp=time.time(),
            metadata={
                "predicted_label_original": predicted_label,
                "corrected_label_original": true_label,
            },
        )

        result = self.router.route_feedback(correction)
        self._append_correction_to_similarity_buffer(
            query_emb,
            true_label,
            compute_preview_signature(image),
        )
        self._rebuild_prototypes()
        self._update_ood_threshold()

        try:
            self._apply_buffer_management()
        except BufferCapacityError as exc:
            result["buffer_management_warning"] = str(exc)
        else:
            self._update_ood_threshold()

        if self.config.recalibrate_after_feedback:
            result["calibration_summary"] = self.calibrator.calibration_summary()

        return result

    def correct_comparative(
        self,
        image_path: str,
        preferred_label: Union[str, int],
        alternative_label: Union[str, int],
        confidence_weight: float = 1.0,
    ) -> Dict[str, Any]:
        """Apply comparative human feedback inspired by ordinal supervision.

        The annotator answers a relative question ("more like A than B"), which
        is mapped to a standard correction update toward `preferred_label`.
        """

        self._ensure_initialized()
        self._validate_label(preferred_label)
        self._validate_label(alternative_label)

        if preferred_label == alternative_label:
            raise ConfigValidationError(
                "preferred_label and alternative_label must be different for comparative feedback."
            )

        known_labels = set(self._sim_labels)
        if alternative_label not in known_labels:
            raise ConfigValidationError(
                "alternative_label must already exist in the support set for comparative feedback."
            )

        preview = self._load_rgb_image_from_path(image_path)
        preview_emb = self._extract_embedding_checked(image=preview, source=image_path)
        preferred_distance = self._distance_to_label_prototype(preview_emb, preferred_label)
        alternative_distance = self._distance_to_label_prototype(preview_emb, alternative_label)

        result = self.correct(
            image_path=image_path,
            true_label=preferred_label,
            confidence_weight=confidence_weight,
        )
        result["comparative_feedback"] = {
            "preferred_label": preferred_label,
            "alternative_label": alternative_label,
            "preferred_distance": preferred_distance,
            "alternative_distance": alternative_distance,
            "supports_preference": preferred_distance <= alternative_distance,
        }
        return result

    def calibration_report(self) -> Dict[str, float]:
        """Return calibration and uncertainty diagnostics for monitoring."""

        report = self.calibrator.calibration_summary()
        report["ood_distance_threshold"] = float(self._ood_distance_threshold)
        report["support_size"] = float(len(self._sim_embeddings))
        report["prototype_count"] = float(len(self._prototype_labels))
        return report

    def save(self, path: str) -> None:
        """Persist learner state to disk.

        Args:
            path: JSON state file path.

        Raises:
            AdaptShotError: If state cannot be serialized.
        """
        config_payload = asdict(self.config)
        embeddings_payload = np.array(self._sim_embeddings, dtype=np.float32)
        integrity = self._build_integrity_payload(config_payload, embeddings_payload)

        state = {
            "schema_version": SCHEMA_VERSION,
            "config": asdict(self.config),
            "calibration": {
                "temperature": self.calibrator.current_temperature,
                "ece_history": self.calibrator._ece_history,
            },
            "act_thresholds": self.act.get_all_thresholds(),
            "buffer": {
                "labels": self._sim_labels,
                "times": self._sim_access_times,
                "uncertainties": self._sim_uncertainties,
                "previews": [preview.tolist() for preview in self._sim_preview_signatures],
            },
            "prototypes": {
                "labels": self._prototype_labels.tolist(),
                "counts": self._prototype_counts.tolist(),
                "embeddings": self._prototype_embeddings.tolist()
                if self._prototype_embeddings.size > 0
                else [],
                "ood_distance_threshold": self._ood_distance_threshold,
            },
            "label_index": {
                "label_to_idx": self._label_to_idx,
            },
            "is_initialized": self._is_initialized,
            "integrity": integrity,
        }

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        json_tmp = self._temporary_path(target, suffix=".json.tmp")
        emb_tmp = self._temporary_path(target.with_suffix(".embeddings.npy"), suffix=".embeddings.npy")
        head_tmp = self._temporary_path(target.with_suffix(".head.pt"), suffix=".tmp")

        try:
            with json_tmp.open("w", encoding="utf-8") as handle:
                json.dump(state, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(json_tmp, target)

            np.save(emb_tmp, embeddings_payload)
            os.replace(emb_tmp, target.with_suffix(".embeddings.npy"))

            if self._model_head is not None:
                torch.save(self._model_head.state_dict(), head_tmp)
                os.replace(head_tmp, target.with_suffix(".head.pt"))
        finally:
            for temp_path in (json_tmp, emb_tmp, head_tmp):
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str) -> "FewShotLearner":
        """Restore learner state from disk.

        Args:
            path: JSON state file path.

        Returns:
            A restored FewShotLearner instance.

        Raises:
            AdaptShotError: If required state files are missing or invalid.
        """
        target = Path(path)
        if not target.exists():
            raise AdaptShotError(
                f"State file not found: '{path}'. Ensure the path is correct before loading."
            )
        try:
            with target.open("r", encoding="utf-8") as handle:
                state = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise AdaptShotError(
                f"Failed to read checkpoint JSON at '{path}'. The file may be corrupted."
            ) from exc

        schema_version = str(state.get("schema_version", "0.1.0"))
        legacy_checkpoint = schema_version == "0.1.0"
        if schema_version != SCHEMA_VERSION:
            warnings.warn(
                f"Checkpoint schema {schema_version} loaded; migrating to {SCHEMA_VERSION}.",
                RuntimeWarning,
            )
            if schema_version == "0.1.0":
                state = migrate_v0_1_0_to_v0_1_1(state)
            else:
                raise AdaptShotError(
                    f"Unsupported checkpoint schema_version '{schema_version}'. "
                    f"Expected '{SCHEMA_VERSION}' or legacy '0.1.0'."
                )

        config_payload = state.get("config")
        if not isinstance(config_payload, dict):
            raise AdaptShotError("Checkpoint config is missing or malformed.")

        learner = cls(AdaptShotConfig(**config_payload))

        emb_path = target.with_suffix(".embeddings.npy")
        if not emb_path.exists():
            raise AdaptShotError(
                f"Embeddings file not found: '{emb_path}'. Save state again before loading."
            )

        try:
            embeddings = np.load(emb_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise AdaptShotError(
                f"Failed to read embeddings file '{emb_path}'. The file may be corrupted."
            ) from exc

        if not isinstance(embeddings, np.ndarray):
            raise AdaptShotError("Loaded embeddings payload is invalid.")

        learner._load_state_payload(
            state=state,
            embeddings=embeddings,
            source_path=target,
            legacy_checkpoint=legacy_checkpoint,
        )
        return learner

    def _validate_config(self, config: AdaptShotConfig) -> None:
        if config.device != "cpu":
            raise ConfigValidationError(
                "AdaptShot v0.1.1 is CPU-first. Set device='cpu'. "
                "See docs/getting-started/quickstart.md."
            )
        if config.ece_n_bins <= 1:
            raise ConfigValidationError(
                "ece_n_bins must be greater than 1 to compute calibration bins. "
                f"Received {config.ece_n_bins}."
            )
        if config.temperature_init <= 0.0:
            raise ConfigValidationError(
                "temperature_init must be positive. "
                f"Received {config.temperature_init}."
            )
        if config.max_buffer_size <= 0:
            raise ConfigValidationError(
                "max_buffer_size must be a positive integer. "
                f"Received {config.max_buffer_size}."
            )
        if config.similarity_metric not in {"cosine", "euclidean"}:
            raise ConfigValidationError(
                "similarity_metric must be 'cosine' or 'euclidean'. "
                f"Received {config.similarity_metric}."
            )
        if config.inference_mode not in {"nearest_neighbor", "prototypical"}:
            raise ConfigValidationError(
                "inference_mode must be 'nearest_neighbor' or 'prototypical'. "
                f"Received {config.inference_mode}."
            )
        if config.calibration_eval_bins < config.ece_n_bins:
            raise ConfigValidationError(
                "calibration_eval_bins must be >= ece_n_bins. "
                f"Received calibration_eval_bins={config.calibration_eval_bins}, "
                f"ece_n_bins={config.ece_n_bins}."
            )

    def _validate_support_inputs(
        self,
        image_paths: List[str],
        labels: List[Union[str, int]],
    ) -> None:
        if not image_paths:
            raise ConfigValidationError(
                "image_paths cannot be empty. Provide at least one support image path. "
                "See docs/getting-started/quickstart.md."
            )
        if len(image_paths) != len(labels):
            raise ConfigValidationError(
                "image_paths and labels must have the same length. "
                f"Got {len(image_paths)} image_paths and {len(labels)} labels."
            )
        if not labels:
            raise ConfigValidationError(
                "labels cannot be empty. Provide one label per support image. "
                "See docs/getting-started/quickstart.md."
            )

    def _validate_label(self, label: Union[str, int]) -> None:
        if isinstance(label, str) and label.strip() == "":
            raise ConfigValidationError(
                "Label strings cannot be empty. Provide a non-empty class label."
            )
        if not isinstance(label, (str, int)):
            raise ConfigValidationError(
                "Labels must be str or int values. "
                f"Received label type '{type(label).__name__}'."
            )

    def _load_rgb_image_from_path(self, image_path: str) -> Image.Image:
        if not isinstance(image_path, str) or image_path.strip() == "":
            raise InvalidImageError(
                "image_path must be a non-empty string path to an image file."
            )

        path = Path(image_path)
        if not path.exists():
            raise InvalidImageError(
                f"Image file not found: '{image_path}'. Verify the path and try again."
            )
        if not path.is_file():
            raise InvalidImageError(
                f"Expected a file path, but got a directory: '{image_path}'."
            )

        try:
            with Image.open(path) as verify_img:
                verify_img.verify()
            with Image.open(path) as loaded:
                rgb_image = loaded.copy()
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError(
                "Image format is unreadable or unsupported. "
                f"Use a standard format like PNG or JPEG. File: '{image_path}'."
            ) from exc

        self._validate_pil_rgb(rgb_image, source=image_path)
        return rgb_image

    def _validate_pil_rgb(self, image: Image.Image, source: str) -> None:
        mode = image.mode
        channels = len(image.getbands()) if image.getbands() else 0
        if channels != 3 or mode != "RGB":
            raise InvalidImageError(
                "Expected 3-channel RGB image, got "
                f"{channels}-channel mode '{mode}' from '{source}'. "
                "Convert before loading. See docs/getting-started/quickstart.md."
            )
        if image.width <= 0 or image.height <= 0:
            raise InvalidImageError(
                f"Image dimensions must be positive, got ({image.width}, {image.height}) for '{source}'."
            )

    def _normalize_predict_image(self, image: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        if isinstance(image, str):
            return self._load_rgb_image_from_path(image)

        if isinstance(image, Image.Image):
            self._validate_pil_rgb(image, source="PIL image")
            return image

        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                raise InvalidImageError(
                    "Expected 3-channel RGB image, got 1-channel grayscale array. "
                    "Convert before loading. See docs/getting-started/quickstart.md."
                )
            if image.ndim != 3:
                raise InvalidImageError(
                    "Expected NumPy array with shape [H, W, 3]. "
                    f"Received shape {image.shape}."
                )
            if image.shape[2] != 3:
                raise InvalidImageError(
                    "Expected 3-channel RGB image, got "
                    f"{image.shape[2]} channels. Convert before loading."
                )
            if image.shape[0] <= 0 or image.shape[1] <= 0:
                raise InvalidImageError(
                    f"Image dimensions must be positive, got shape {image.shape}."
                )
            try:
                pil_image = Image.fromarray(image)
            except (TypeError, ValueError) as exc:
                raise InvalidImageError(
                    "Failed to parse NumPy image input. Ensure dtype is image-compatible "
                    "(e.g., uint8) and shape is [H, W, 3]."
                ) from exc
            self._validate_pil_rgb(pil_image, source="numpy array")
            return pil_image

        raise InvalidImageError(
            "Unsupported image input type. Use file path, PIL.Image.Image, or NumPy array. "
            f"Received '{type(image).__name__}'."
        )

    def _extract_embedding_checked(self, image: Image.Image, source: str) -> np.ndarray:
        try:
            embedding = extract_embedding(image, self.config, cache=self._embedding_cache)
        except (ValueError, RuntimeError, OSError) as exc:
            raise InvalidImageError(
                f"Failed to extract embedding for '{source}'. Ensure the image is valid RGB input."
            ) from exc

        if not isinstance(embedding, np.ndarray):
            raise AdaptShotError(
                "Embedding extractor returned unexpected type. Expected numpy.ndarray. "
                f"Got {type(embedding).__name__}."
            )
        if embedding.ndim != 1:
            raise AdaptShotError(
                "Expected 1D embedding vector from extractor. "
                f"Got shape {embedding.shape}."
            )
        if embedding.size == 0:
            raise AdaptShotError("Extractor returned an empty embedding vector.")
        if not np.all(np.isfinite(embedding)):
            raise AdaptShotError(
                "Extractor returned non-finite values. Verify image integrity and preprocessing."
            )

        expected_dim = self._embedding_dim() if self._sim_embeddings else None
        if expected_dim is not None and int(embedding.shape[0]) != expected_dim:
            raise AdaptShotError(
                "Embedding dimensionality mismatch. "
                f"Expected {expected_dim}, got {int(embedding.shape[0])}."
            )

        return embedding.astype(np.float32, copy=False)

    def _embedding_dim(self) -> int:
        """Return the expected embedding dimensionality for the current backbone.

        If the support set is already populated, uses the actual embedding shape.
        Otherwise falls back to the known dimension for the configured backbone.
        """
        if self._sim_embeddings:
            return int(self._sim_embeddings[0].shape[0])
        return BACKBONE_OUTPUT_DIM.get(self.config.backbone, 512)

    def _ensure_initialized(self) -> None:
        if not self._is_initialized:
            raise AdaptShotError(
                "FewShotLearner is not initialized. Call load_support_images() first. "
                "See docs/getting-started/quickstart.md."
            )

    def _calibrate_or_raise(self, raw_confidence: float) -> float:
        min_samples = max(10, self.calibrator.window_size // 2)
        observed = len(self.calibrator._window_confidences)
        if self.calibrator.method in {"temperature", "scaling_binning"} and observed < min_samples:
            raise CalibrationNotReadyError(
                "Calibration window is not ready. "
                f"Need at least {min_samples} observations, got {observed}. "
                "Continue collecting feedback with correct()."
            )
        return float(self.calibrator.calibrate(raw_confidence))

    def _raw_to_unit_interval(self, raw_confidence: float) -> float:
        value = float(raw_confidence)
        if 0.0 <= value <= 1.0:
            return value
        return float(np.clip((value + 1.0) / 2.0, 0.0, 1.0))

    def _rebuild_label_index(self) -> None:
        self._label_to_idx.clear()
        self._idx_to_label.clear()

        for label in self._sim_labels:
            if label not in self._label_to_idx:
                idx = len(self._label_to_idx)
                self._label_to_idx[label] = idx
                self._idx_to_label[idx] = label

    def _rebuild_prototypes(self) -> None:
        if not self._sim_embeddings:
            self._prototype_embeddings = np.empty((0, 0), dtype=np.float32)
            self._prototype_labels = np.asarray([], dtype=object)
            self._prototype_counts = np.asarray([], dtype=np.int64)
            return

        prototypes, labels, counts = compute_class_prototypes(
            np.asarray(self._sim_embeddings, dtype=np.float32),
            np.asarray(self._sim_labels, dtype=object),
        )
        self._prototype_embeddings = prototypes
        self._prototype_labels = labels
        self._prototype_counts = counts

    def _label_key(self, label: object) -> object:
        if hasattr(label, "item"):
            return label.item()
        return label

    def _coerce_label(self, label: object) -> Union[str, int]:
        normalized = self._label_key(label)
        if isinstance(normalized, (str, int)):
            return normalized
        raise AdaptShotError(
            "Unsupported label type produced during inference. "
            f"Expected str|int, got {type(normalized).__name__}."
        )

    def _prototype_index_for_label(self, label: Union[str, int]) -> Optional[int]:
        label_key = self._label_key(label)
        for idx, proto_label in enumerate(self._prototype_labels):
            if self._label_key(proto_label) == label_key:
                return idx
        return None

    def _distance_to_label_prototype(
        self,
        query_embedding: np.ndarray,
        label: Union[str, int],
    ) -> float:
        if self._prototype_embeddings.size == 0:
            return 0.0

        proto_idx = self._prototype_index_for_label(label)
        if proto_idx is None:
            return 0.0

        prototype = self._prototype_embeddings[proto_idx][np.newaxis, :]
        distances = euclidean_distance_numpy(query_embedding, prototype, normalize=True)
        return float(distances.reshape(-1)[0])

    def _nearest_support_index_for_label(
        self,
        query_embedding: np.ndarray,
        label: Union[str, int],
    ) -> int:
        candidates = [idx for idx, value in enumerate(self._sim_labels) if value == label]
        if not candidates:
            raise AdaptShotError(
                "Predicted class label was not found in support buffer. "
                "Reload support images before predicting again."
            )
        if len(candidates) == 1:
            return int(candidates[0])

        candidate_embeddings = np.asarray([self._sim_embeddings[idx] for idx in candidates], dtype=np.float32)
        if self.config.similarity_metric == "euclidean":
            distances = euclidean_distance_numpy(query_embedding, candidate_embeddings, normalize=True)
            local_idx = int(np.argmin(distances.reshape(-1)))
            return int(candidates[local_idx])

        similarities = np.asarray(
            query_embedding[np.newaxis, :] @ candidate_embeddings.T,
            dtype=np.float32,
        ).reshape(-1)
        query_norm = np.linalg.norm(query_embedding) + 1e-8
        candidate_norms = np.linalg.norm(candidate_embeddings, axis=1) + 1e-8
        cosine_scores = similarities / (query_norm * candidate_norms)
        local_idx = int(np.argmax(cosine_scores))
        return int(candidates[local_idx])

    def _update_ood_threshold(self) -> None:
        if not self.config.enable_ood_detection:
            self._ood_distance_threshold = float("inf")
            return
        if self._prototype_embeddings.size == 0 or not self._sim_embeddings:
            self._ood_distance_threshold = self.config.ood_absolute_min_distance
            return

        distances: List[float] = []
        for embedding, label in zip(self._sim_embeddings, self._sim_labels):
            proto_idx = self._prototype_index_for_label(label)
            if proto_idx is None:
                continue
            prototype = self._prototype_embeddings[proto_idx][np.newaxis, :]
            distance = euclidean_distance_numpy(embedding, prototype, normalize=True)
            distances.append(float(distance.reshape(-1)[0]))

        if not distances:
            self._ood_distance_threshold = self.config.ood_absolute_min_distance
            return

        quantile_threshold = float(
            np.quantile(
                np.asarray(distances, dtype=np.float64),
                self.config.ood_threshold_quantile,
            )
        )
        self._ood_distance_threshold = max(
            self.config.ood_absolute_min_distance,
            quantile_threshold,
        )

    def _is_out_of_distribution(
        self,
        distance_to_prototype: float,
        prototype_margin: float,
    ) -> bool:
        if not self.config.enable_ood_detection:
            return False
        distance_flag = float(distance_to_prototype) > float(self._ood_distance_threshold)
        margin_flag = np.isfinite(prototype_margin) and float(prototype_margin) < 0.01
        return bool(distance_flag or margin_flag)

    def _ensure_label_index(self, label: Union[str, int]) -> int:
        if label in self._label_to_idx:
            return self._label_to_idx[label]

        idx = len(self._label_to_idx)
        self._label_to_idx[label] = idx
        self._idx_to_label[idx] = label
        self._expand_model_head_if_needed(new_num_classes=len(self._label_to_idx))
        return idx

    def _temporary_path(self, target: Path, suffix: str) -> Path:
        handle = tempfile.NamedTemporaryFile(delete=False, dir=target.parent, suffix=suffix)
        handle.close()
        return Path(handle.name)

    def _build_integrity_payload(
        self,
        config_payload: Dict[str, Any],
        embeddings_payload: np.ndarray,
    ) -> Dict[str, str]:
        config_bytes = json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        embeddings_bytes = np.ascontiguousarray(embeddings_payload, dtype=np.float32).tobytes()
        config_hash = hashlib.sha256(config_bytes).hexdigest()
        embeddings_hash = hashlib.sha256(embeddings_bytes).hexdigest()
        checksum_hash = hashlib.sha256(
            (config_hash + embeddings_hash).encode("utf-8")
        ).hexdigest()
        return {
            "config_sha256": config_hash,
            "embeddings_sha256": embeddings_hash,
            "checksum_sha256": checksum_hash,
        }

    def _build_preview_signatures_from_embeddings(self, embeddings: np.ndarray) -> List[np.ndarray]:
        previews: List[np.ndarray] = []
        for row in embeddings:
            vector = np.asarray(row, dtype=np.float32).reshape(-1)
            preview = np.zeros(48, dtype=np.float32)
            length = min(preview.size, vector.size)
            if length > 0:
                preview[:length] = vector[:length]
            previews.append(preview)
        return previews

    def _load_state_payload(
        self,
        state: Dict[str, Any],
        embeddings: np.ndarray,
        source_path: Path,
        legacy_checkpoint: bool,
    ) -> None:
        schema_version = str(state.get("schema_version", SCHEMA_VERSION))
        integrity = state.get("integrity")

        if schema_version == SCHEMA_VERSION and not legacy_checkpoint:
            if not isinstance(integrity, dict):
                raise AdaptShotError("Checkpoint integrity metadata is missing.")
            expected_integrity = self._build_integrity_payload(state["config"], embeddings)
            if integrity.get("checksum_sha256") != expected_integrity["checksum_sha256"]:
                raise AdaptShotError(
                    "Checkpoint integrity check failed. The checkpoint may be corrupted or tampered with."
                )

        learner_state = self._validate_and_normalize_state(state=state, embeddings=embeddings)
        self._restore_from_state(learner_state, embeddings=embeddings, source_path=source_path)

    def _validate_and_normalize_state(
        self,
        state: Dict[str, Any],
        embeddings: np.ndarray,
    ) -> Dict[str, Any]:
        required_keys = ["config", "calibration", "act_thresholds", "buffer"]
        for key in required_keys:
            if key not in state:
                raise AdaptShotError(f"Checkpoint is missing required key '{key}'.")

        buffer_state = state.get("buffer")
        if not isinstance(buffer_state, dict):
            raise AdaptShotError("Checkpoint buffer section is missing or malformed.")

        labels = list(buffer_state.get("labels", []))
        times = list(buffer_state.get("times", []))
        uncertainties = list(buffer_state.get("uncertainties", []))
        previews_raw = list(buffer_state.get("previews", []))

        if len(labels) != len(times) or len(labels) != len(uncertainties):
            raise AdaptShotError(
                "Checkpoint buffer lengths do not match. Labels, times, and uncertainties must align."
            )
        if len(embeddings) != len(labels):
            raise AdaptShotError(
                "Checkpoint embeddings do not match buffer labels. The checkpoint is corrupted."
            )

        if previews_raw and len(previews_raw) != len(labels):
            raise AdaptShotError(
                "Checkpoint preview signatures do not match buffer labels."
            )

        normalized_state = dict(state)
        normalized_state["buffer"] = {
            "labels": labels,
            "times": [float(value) for value in times],
            "uncertainties": [float(value) for value in uncertainties],
            "previews": [np.asarray(preview, dtype=np.float32).tolist() for preview in previews_raw]
            if previews_raw
            else [preview.tolist() for preview in self._build_preview_signatures_from_embeddings(embeddings)],
        }
        return normalized_state

    def _restore_from_state(
        self,
        state: Dict[str, Any],
        embeddings: np.ndarray,
        source_path: Path,
    ) -> None:
        learner = self
        learner.calibrator.temperature = float(state["calibration"]["temperature"])
        learner.calibrator._ece_history = list(state["calibration"].get("ece_history", []))

        for key, threshold in state["act_thresholds"].items():
            class_idx = int(key)
            if class_idx in learner.act._class_state:
                learner.act._class_state[class_idx]["threshold"] = float(threshold)

        learner._sim_labels = list(state["buffer"]["labels"])
        learner._sim_access_times = [float(v) for v in state["buffer"]["times"]]
        learner._sim_uncertainties = [float(v) for v in state["buffer"]["uncertainties"]]
        learner._sim_preview_signatures = [
            np.asarray(preview, dtype=np.float32) for preview in state["buffer"]["previews"]
        ]
        learner._sim_embeddings = [np.asarray(row, dtype=np.float32) for row in embeddings]

        learner._rebuild_label_index()
        learner._rebuild_prototypes()
        prototypes_state = state.get("prototypes")
        if isinstance(prototypes_state, dict):
            ood_threshold = prototypes_state.get("ood_distance_threshold")
            if isinstance(ood_threshold, (float, int)):
                learner._ood_distance_threshold = float(ood_threshold)
        else:
            learner._update_ood_threshold()
        if learner._sim_embeddings:
            learner._init_or_rebuild_model_head(embedding_dim=learner._embedding_dim())
            learner._embedding_cache.set(
                learner._sim_embeddings[0],
                learner._sim_preview_signatures[0],
            )

            head_path = source_path.with_suffix(".head.pt")
            if head_path.exists() and learner._model_head is not None:
                try:
                    learner._model_head.load_state_dict(
                        torch.load(head_path, map_location=torch.device("cpu"))
                    )
                except Exception as exc:
                    raise AdaptShotError(
                        f"Failed to load model head from '{head_path}'. The file may be corrupted."
                    ) from exc

        learner._is_initialized = bool(state.get("is_initialized", bool(learner._sim_embeddings)))

    def _init_or_rebuild_model_head(self, embedding_dim: int) -> None:
        num_classes = max(1, len(self._label_to_idx))
        self._model_head = torch.nn.Linear(embedding_dim, num_classes)
        self._model_head.eval()
        self.finetuner = CAEWCFinetuner(
            model=self._model_head,
            device=self.config.device,
            ewc_lambda=0.1,
            learning_rate=1e-4,
            weight_decay=1e-3,
            epochs=5,
        )

    def _expand_model_head_if_needed(self, new_num_classes: int) -> None:
        if self._model_head is None:
            embedding_dim = self._embedding_dim()
            self._init_or_rebuild_model_head(embedding_dim=embedding_dim)
            return

        if new_num_classes <= self._model_head.out_features:
            return

        old_head = self._model_head
        expanded_head = torch.nn.Linear(old_head.in_features, new_num_classes)

        with torch.no_grad():
            expanded_head.weight[: old_head.out_features] = old_head.weight
            expanded_head.bias[: old_head.out_features] = old_head.bias

        expanded_head.eval()
        self._model_head = expanded_head
        self.finetuner = CAEWCFinetuner(
            model=self._model_head,
            device=self.config.device,
            ewc_lambda=0.1,
            learning_rate=1e-4,
            weight_decay=1e-3,
            epochs=5,
        )

    def _append_correction_to_similarity_buffer(
        self,
        embedding: np.ndarray,
        label: Union[str, int],
        preview_signature: np.ndarray,
    ) -> None:
        self._sim_embeddings.append(embedding)
        self._sim_labels.append(label)
        self._sim_access_times.append(time.time())
        self._sim_uncertainties.append(0.5)
        self._sim_preview_signatures.append(preview_signature.astype(np.float32, copy=False))
        self._ensure_label_index(label)
        if self._sim_embeddings:
            self._embedding_cache.set(
                self._sim_embeddings[0],
                self._sim_preview_signatures[0],
            )

    def _trigger_finetune(self, corrections: List[Correction]) -> None:
        if self.finetuner is None or self._model_head is None:
            return

        if self._sim_embeddings and self._sim_labels:
            support_tensor = torch.tensor(np.stack(self._sim_embeddings), dtype=torch.float32)
            support_label_tensor = torch.tensor(
                [self._ensure_label_index(label) for label in self._sim_labels],
                dtype=torch.long,
            )
            fisher_loader: DataLoader[Any] = DataLoader(
                TensorDataset(support_tensor, support_label_tensor),
                batch_size=min(32, len(self._sim_embeddings)),
                shuffle=False,
            )
            self.finetuner.update_fisher(fisher_loader)

        emb_list: List[np.ndarray] = []
        label_list: List[int] = []
        weight_list: List[float] = []

        for correction in corrections:
            image = self._load_rgb_image_from_path(correction.image_path)
            embedding = self._extract_embedding_checked(image=image, source=correction.image_path)
            emb_list.append(embedding)

            corrected_label_original = correction.metadata.get("corrected_label_original")
            if isinstance(corrected_label_original, (str, int)):
                label_idx = self._ensure_label_index(corrected_label_original)
            else:
                label_idx = int(correction.corrected_label)
            label_list.append(label_idx)
            weight_list.append(float(correction.confidence_weight))

        if not emb_list:
            return

        new_embs = torch.tensor(np.stack(emb_list), dtype=torch.float32)
        new_labels = torch.tensor(label_list, dtype=torch.long)
        weights = torch.tensor(weight_list, dtype=torch.float32)

        self.finetuner.finetune(new_embs, new_labels, weights)

    def _validate_prune_shapes(
        self,
        pruned_emb: np.ndarray,
        pruned_labels: np.ndarray,
        pruned_unc: np.ndarray,
        pruned_times: np.ndarray,
    ) -> None:
        lengths: Tuple[int, int, int, int] = (
            len(pruned_emb),
            len(pruned_labels),
            len(pruned_unc),
            len(pruned_times),
        )
        if len(set(lengths)) != 1:
            raise BufferCapacityError(
                "UP-UGF pruning returned inconsistent buffer lengths. "
                f"Got lengths embeddings={lengths[0]}, labels={lengths[1]}, "
                f"uncertainties={lengths[2]}, times={lengths[3]}."
            )
        if lengths[0] > self.config.max_buffer_size:
            raise BufferCapacityError(
                "UP-UGF pruning did not enforce max_buffer_size. "
                f"Expected <= {self.config.max_buffer_size}, got {lengths[0]}."
            )

    def _apply_buffer_management(self) -> None:
        if len(self._sim_embeddings) <= self.config.max_buffer_size:
            return

        emb_np = np.array(self._sim_embeddings, dtype=np.float32)
        unc_np = np.array(self._sim_uncertainties, dtype=np.float32)
        time_np = np.array(self._sim_access_times, dtype=np.float64)
        label_np = np.array(self._sim_labels, dtype=object)

        try:
            scores = self.pruner.compute_scores(emb_np, unc_np, time_np)
            keep_idx = np.argsort(scores)[-self.config.max_buffer_size :]
            self._sim_embeddings = [np.array(emb_np[idx], dtype=np.float32) for idx in keep_idx]
            self._sim_labels = [label_np[idx] for idx in keep_idx]
            self._sim_uncertainties = [float(unc_np[idx]) for idx in keep_idx]
            self._sim_access_times = [float(time_np[idx]) for idx in keep_idx]
            self._sim_preview_signatures = [self._sim_preview_signatures[idx] for idx in keep_idx]
        except Exception as exc:
            self._sim_embeddings = self._sim_embeddings[-self.config.max_buffer_size :]
            self._sim_labels = self._sim_labels[-self.config.max_buffer_size :]
            self._sim_uncertainties = self._sim_uncertainties[-self.config.max_buffer_size :]
            self._sim_access_times = self._sim_access_times[-self.config.max_buffer_size :]
            self._sim_preview_signatures = self._sim_preview_signatures[-self.config.max_buffer_size :]
            self._rebuild_label_index()
            raise BufferCapacityError(
                "UP-UGF pruning failed. Applied deterministic FIFO fallback to enforce "
                f"capacity {self.config.max_buffer_size}. Error: {exc}"
            ) from exc

        self._rebuild_label_index()
