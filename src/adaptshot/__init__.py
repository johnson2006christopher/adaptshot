"""AdaptShot: Human-Aligned Few-Shot Vision Learning.

Every name exported here is classified as stable or experimental in
``adaptshot.api``, and ``tests/test_api_surface.py`` holds the two in sync.
"""

__version__ = "0.2.0"

# --- Stable -----------------------------------------------------------------
from .config.settings import (
    AdaptShotConfig,
    Backbone,
    CalibrationMethod,
    ConformalMode,
    Device,
    InferenceMode,
    SimilarityMetric,
    UncertaintyMode,
)

# --- Experimental (see adaptshot.api for why each is here) -------------------
from .core.act import ACTEngine
from .core.calibration import CalibrationEngine
from .core.conformal import ConformalEngine, ConformalPredictionSet
from .core.explain import (
    ConfidenceDecomposition,
    Counterfactual,
    ExplainabilityEngine,
    ExplanationResult,
    FeatureAttribution,
)
from .core.learner import FewShotLearner, PredictionResult
from .core.uncertainty import UncertaintyQuantifier, UncertaintyReport
from .training.contrastive import ContrastiveConfig, ContrastivePrototypeLearner
from .training.feedback_router import FeedbackRouter
from .training.up_ugf import UPUGFPruner
from .utils.exceptions import (
    AdaptShotError,
    BackboneError,
    BufferCapacityError,
    CalibrationNotReadyError,
    ConfigValidationError,
    InvalidImageError,
)

__all__ = [
    "ACTEngine",
    "AdaptShotConfig",
    "AdaptShotError",
    "Backbone",
    "BackboneError",
    "BufferCapacityError",
    "CalibrationEngine",
    "CalibrationMethod",
    "CalibrationNotReadyError",
    "ConfidenceDecomposition",
    "ConfigValidationError",
    "ConformalEngine",
    "ConformalMode",
    "ConformalPredictionSet",
    "ContrastiveConfig",
    "ContrastivePrototypeLearner",
    "Counterfactual",
    "Device",
    "ExplainabilityEngine",
    "ExplanationResult",
    "FeatureAttribution",
    "FeedbackRouter",
    "FewShotLearner",
    "InferenceMode",
    "InvalidImageError",
    "PredictionResult",
    "SimilarityMetric",
    "UPUGFPruner",
    "UncertaintyMode",
    "UncertaintyQuantifier",
    "UncertaintyReport",
]
