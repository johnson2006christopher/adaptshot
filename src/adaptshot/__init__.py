"""AdaptShot: Human-Aligned Few-Shot Vision Learning."""

__version__ = "0.2.0"

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
from .core.act import ACTEngine
from .core.calibration import CalibrationEngine
from .core.conformal import ConformalEngine, ConformalPredictionSet
from .core.contrastive import ContrastiveConfig, ContrastivePrototypeLearner
from .core.explain import ExplainabilityEngine, ExplanationResult, FeatureAttribution
from .core.learner import FewShotLearner
from .core.uncertainty import UncertaintyQuantifier, UncertaintyReport
from .training.feedback_router import FeedbackRouter
from .training.up_ugf import UPUGFPruner
from .utils.exceptions import (
    AdaptShotError,
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
    "BufferCapacityError",
    "CalibrationEngine",
    "CalibrationMethod",
    "CalibrationNotReadyError",
    "ConfigValidationError",
    "ConformalEngine",
    "ConformalMode",
    "ConformalPredictionSet",
    "ContrastiveConfig",
    "ContrastivePrototypeLearner",
    "Device",
    "ExplainabilityEngine",
    "ExplanationResult",
    "FeatureAttribution",
    "FeedbackRouter",
    "FewShotLearner",
    "InferenceMode",
    "InvalidImageError",
    "SimilarityMetric",
    "UPUGFPruner",
    "UncertaintyMode",
    "UncertaintyQuantifier",
    "UncertaintyReport",
]