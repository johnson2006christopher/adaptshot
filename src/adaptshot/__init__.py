"""AdaptShot: Human-Aligned Few-Shot Vision Learning."""

__version__ = "0.2.0-dev"

from .config.settings import AdaptShotConfig
from .core.learner import FewShotLearner
from .core.calibration import CalibrationEngine
from .core.act import ACTEngine
from .core.conformal import ConformalEngine, ConformalPredictionSet
from .core.contrastive import ContrastivePrototypeLearner, ContrastiveConfig
from .core.uncertainty import UncertaintyQuantifier, UncertaintyReport
from .core.explain import ExplainabilityEngine, ExplanationResult, FeatureAttribution
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
    "AdaptShotConfig",
    "FewShotLearner",
    "CalibrationEngine",
    "ACTEngine",
    "ConformalEngine",
    "ConformalPredictionSet",
    "ContrastivePrototypeLearner",
    "ContrastiveConfig",
    "UncertaintyQuantifier",
    "UncertaintyReport",
    "ExplainabilityEngine",
    "ExplanationResult",
    "FeatureAttribution",
    "FeedbackRouter",
    "UPUGFPruner",
    "AdaptShotError",
    "InvalidImageError",
    "ConfigValidationError",
    "CalibrationNotReadyError",
    "BufferCapacityError",
]