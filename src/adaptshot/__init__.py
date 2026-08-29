"""AdaptShot: Human-Aligned Few-Shot Vision Learning.

Every name exported here is classified as stable or experimental in
``adaptshot.api``, and ``tests/test_api_surface.py`` holds the two in sync.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

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
from .preflight import Capability, EnvironmentReport, check_environment
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

# Read from the installed distribution's metadata, so pyproject.toml is the one
# place the version is declared and the two-place drift that
# tests/test_release_metadata.py used to catch cannot happen at all (#25). The
# cost: after bumping pyproject.toml, an editable install reports the old value
# until `pip install -e .` is run again. The release checklist says so.
try:
    __version__ = _distribution_version("adaptshot")
except PackageNotFoundError:  # a source tree that was never installed
    __version__ = "0.0.0+uninstalled"

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
    "Capability",
    "ConfidenceDecomposition",
    "ConfigValidationError",
    "ConformalEngine",
    "ConformalMode",
    "ConformalPredictionSet",
    "ContrastiveConfig",
    "ContrastivePrototypeLearner",
    "Counterfactual",
    "Device",
    "EnvironmentReport",
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
    "check_environment",
]
