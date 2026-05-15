"""Unified FewShotLearner API for AdaptShot.

Exposes a single, high-level interface that orchestrates embedding extraction,
similarity search, calibration, ACT gating, human feedback routing, CA-EWC
fine-tuning, and UP-UGF buffer management. Designed for zero-config deployment.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from ..config.settings import AdaptShotConfig
from .extractor import extract_embedding
from .similarity import find_nearest_neighbor
from .calibration import CalibrationEngine
from .act import ACTEngine
from ..training.feedback_router import Correction, FeedbackRouter
from ..training.finetune import CAEWCFinetuner
from ..training.up_ugf import UPUGFPruner

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Structured return type for predict() calls."""
    prediction: Union[str, int]
    raw_confidence: float
    calibrated_confidence: float
    neighbor_idx: int
    uncertainty_flag: bool
    act_action: str


class FewShotLearner:
    """
    Main entry point for AdaptShot.

    Manages the full few-shot lifecycle: loading support data, predicting with
    calibrated uncertainty, routing human corrections, and triggering continual
    learning when confidence thresholds are breached.
    """

    def __init__(self, config: Optional[AdaptShotConfig] = None, **kwargs) -> None:
        """
        Args:
            config: AdaptShotConfig instance, or pass kwargs to construct one
        """
        self.config = config or AdaptShotConfig(**kwargs)
        
        # Core inference modules
        self.calibrator = CalibrationEngine(
            n_bins=self.config.ece_n_bins,
            window_size=self.config.max_buffer_size * 2,
            temperature_init=self.config.temperature_init,
            method=self.config.calibration_method,
        )
        self.act = ACTEngine(n_classes=200)  # Preallocate for dynamic class expansion

        # Buffer & continual learning state
        self._sim_embeddings: List[np.ndarray] = []
        self._sim_labels: List[Union[str, int]] = []
        self._sim_access_times: List[float] = []
        self._sim_uncertainties: List[float] = []
        
        self.pruner = UPUGFPruner(
            capacity=self.config.max_buffer_size,
            uncertainty_weight=1.0,
            recency_weight=1.0,
            redundancy_weight=1.0,
        )
        
        self.finetuner: Optional[CAEWCFinetuner] = None
        self._model_head: Optional[torch.nn.Linear] = None

        # Bind feedback router with internal fine-tune trigger
        self.router = FeedbackRouter(
            buffer_capacity=self.config.max_buffer_size,
            fine_tune_trigger_threshold=max(5, self.config.max_buffer_size // 10),
            calibrator=self.calibrator,
            finetune_fn=self._trigger_finetune,
        )

        self._is_initialized = False

    def load_support_images(self, image_paths: List[str], labels: List[Union[str, int]]) -> None:
        """
        Ingest initial support set and initialize similarity index + Fisher matrix.

        Args:
            image_paths: List of file paths to support images
            labels: Corresponding class labels
        """
        if len(image_paths) != len(labels):
            raise ValueError("image_paths and labels must have the same length")

        logger.info(f"Loading {len(image_paths)} support images...")
        self._sim_embeddings.clear()
        self._sim_labels.clear()
        self._sim_access_times.clear()
        self._sim_uncertainties.clear()

        current_time = time.time()
        for path, label in zip(image_paths, labels):
            emb = extract_embedding(Image.open(path), self.config)
            self._sim_embeddings.append(emb)
            self._sim_labels.append(label)
            self._sim_access_times.append(current_time)
            self._sim_uncertainties.append(0.5)  # Default neutral uncertainty

        # Initialize lightweight classification head for CA-EWC
        unique_labels = sorted(list(set(self._sim_labels)))
        label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        self._model_head = torch.nn.Linear(512, len(unique_labels))
        self._model_head.eval()
        self._label_to_idx = label_to_idx
        self._idx_to_label = {v: k for k, v in label_to_idx.items()}

        # Initialize CA-EWC finetuner
        self.finetuner = CAEWCFinetuner(
            model=self._model_head,
            device=self.config.device,
            ewc_lambda=0.1,
            learning_rate=1e-4,
            epochs=5,
        )

        self._is_initialized = True
        logger.info(f"Support set loaded. {len(self._sim_embeddings)} embeddings indexed.")

    def predict(self, image: Union[str, Image.Image, np.ndarray]) -> PredictionResult:
        """
        Run inference with calibrated confidence and ACT gating.

        Args:
            image: File path, PIL Image, or numpy array

        Returns:
            PredictionResult containing prediction, confidences, and ACT decision
        """
        if not self._is_initialized:
            raise RuntimeError("Load support images using load_support_images() before predicting")

        # 1. Extract embedding
        query_emb = extract_embedding(image, self.config)

        # 2. Find nearest neighbor
        pred_label, raw_conf, neighbor_idx = find_nearest_neighbor(
            query=query_emb,
            support_embeddings=np.array(self._sim_embeddings),
            support_labels=np.array(self._sim_labels),
            use_faiss=self.config.use_faiss,
        )

        # 3. Calibrate confidence
        calibrated_conf = self.calibrator.calibrate(raw_conf)

        # 4. ACT decision
        # Compute recent uncertainty proxy for ACT
        recent_unc = float(np.mean(self._sim_uncertainties[-10:])) if self._sim_uncertainties else 0.0
        accept, act_action = self.act.should_accept(
            confidence=calibrated_conf,
            class_idx=self._label_to_idx.get(pred_label, 0),
            recent_incorrect_rate=recent_unc,
            recent_correct_rate=1.0 - recent_unc,
        )

        # Update access time for retrieved example
        if neighbor_idx < len(self._sim_access_times):
            self._sim_access_times[neighbor_idx] = time.time()

        return PredictionResult(
            prediction=pred_label,
            raw_confidence=float(raw_conf),
            calibrated_confidence=float(calibrated_conf),
            neighbor_idx=int(neighbor_idx),
            uncertainty_flag=not accept,
            act_action=act_action,
        )

    def correct(
        self,
        image_path: str,
        true_label: Union[str, int],
        confidence_weight: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Route a human correction into the continual learning pipeline.

        Args:
            image_path: Path to the misclassified image
            true_label: Ground truth label provided by human
            confidence_weight: Human's confidence in the correction [0.0, 1.0]

        Returns:
            Dictionary summarizing routing actions (buffer size, fine-tuning status, etc.)
        """
        if not self._is_initialized:
            raise RuntimeError("Load support images before applying corrections")

        # Fetch last prediction context (in production, this would be cached per session)
        # For v0.1, we extract fresh to simulate real workflow
        query_emb = extract_embedding(Image.open(image_path), self.config)
        pred_label, raw_conf, _ = find_nearest_neighbor(
            query_emb,
            np.array(self._sim_embeddings),
            np.array(self._sim_labels),
            use_faiss=self.config.use_faiss,
        )

        correction = Correction(
            image_path=image_path,
            predicted_label=pred_label,
            corrected_label=true_label,
            raw_confidence=float(raw_conf),
            confidence_weight=confidence_weight,
            timestamp=time.time(),
        )

        result = self.router.route_feedback(correction)
        self._apply_buffer_management()
        return result

    def _trigger_finetune(self, corrections: List[Correction]) -> None:
        """Internal callback executed by FeedbackRouter when threshold is met."""
        if self.finetuner is None or self._model_head is None:
            return

        # Convert corrections to tensors
        emb_list, label_list, weight_list = [], [], []
        for c in corrections:
            emb = extract_embedding(Image.open(c.image_path), self.config)
            emb_list.append(emb)
            label_list.append(self._label_to_idx.get(c.corrected_label, 0))
            weight_list.append(c.confidence_weight)

        new_embs = torch.tensor(np.stack(emb_list), dtype=torch.float32)
        new_labels = torch.tensor(label_list, dtype=torch.long)
        weights = torch.tensor(weight_list, dtype=torch.float32)

        # Run CA-EWC
        self.finetuner.finetune(new_embs, new_labels, weights)
        logger.info("CA-EWC fine-tuning completed on correction batch.")

    def _apply_buffer_management(self) -> None:
        """Enforce capacity limits via UP-UGF pruning."""
        if len(self._sim_embeddings) <= self.config.max_buffer_size:
            return

        emb_np = np.array(self._sim_embeddings)
        unc_np = np.array(self._sim_uncertainties)
        time_np = np.array(self._sim_access_times)
        label_np = np.array(self._sim_labels)

        pruned_emb, pruned_labels, pruned_unc, pruned_times = self.pruner.prune(
            emb_np, label_np, unc_np, time_np
        )

        self._sim_embeddings = pruned_emb.tolist()
        self._sim_labels = pruned_labels.tolist()
        self._sim_uncertainties = pruned_unc.tolist()
        self._sim_access_times = pruned_times.tolist()

        logger.debug(f"UP-UGF pruning applied. Buffer size: {len(self._sim_embeddings)}")

    def save(self, path: str) -> None:
        """Persist learner state to disk."""
        state = {
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
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        
        # Save embeddings separately for memory efficiency
        emb_path = Path(path).with_suffix(".embeddings.npy")
        np.save(emb_path, np.array(self._sim_embeddings))
        
        # Save model head if exists
        if self._model_head is not None:
            torch.save(self._model_head.state_dict(), Path(path).with_suffix(".head.pt"))

    @classmethod
    def load(cls, path: str) -> "FewShotLearner":
        """Restore learner state from disk."""
        with open(path, "r") as f:
            state = json.load(f)
        
        learner = cls(AdaptShotConfig(**state["config"]))
        learner.calibrator.temperature = torch.nn.Parameter(torch.tensor(state["calibration"]["temperature"]))
        learner.calibrator._ece_history = state["calibration"]["ece_history"]
        
        # Restore ACT
        for k, v in state["act_thresholds"].items():
            if int(k) in learner.act._class_state:
                learner.act._class_state[int(k)]["threshold"] = v
                
        # Restore buffer
        learner._sim_labels = state["buffer"]["labels"]
        learner._sim_access_times = state["buffer"]["times"]
        learner._sim_uncertainties = state["buffer"]["uncertainties"]
        
        emb_path = Path(path).with_suffix(".embeddings.npy")
        learner._sim_embeddings = np.load(emb_path).tolist()
        
        if Path(path).with_suffix(".head.pt").exists():
            learner._model_head = torch.nn.Linear(512, len(set(learner._sim_labels)))
            learner._model_head.load_state_dict(torch.load(emb_path.with_suffix(".head.pt")))
            learner._is_initialized = True
            
        return learner