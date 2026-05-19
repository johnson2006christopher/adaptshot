"""AdaptShot: Human-Aligned Few-Shot Vision Learning."""

__version__ = "0.1.0"

from .config.settings import AdaptShotConfig
from .core.learner import FewShotLearner
from .core.calibration import CalibrationEngine
from .core.act import ACTEngine
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
    "FeedbackRouter",
    "UPUGFPruner",
    "AdaptShotError",
    "InvalidImageError",
    "ConfigValidationError",
    "CalibrationNotReadyError",
    "BufferCapacityError",
]