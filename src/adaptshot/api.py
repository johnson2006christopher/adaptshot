"""The public surface, classified (#23).

Every name in ``adaptshot.__all__`` is exactly one of the two tuples below, and
``tests/test_api_surface.py`` enforces it -- the union must equal ``__all__``,
the two must not overlap, every experimental object's docstring must say so,
and every name must appear in ``docs/api/reference.md`` under the matching
heading. The classification is therefore a fact the test suite checks, not a
comment that drifts.

The rule for placing a name, in the maintainer's words from #23:

    Stable        Supported, semver-protected, tested.       Keep; document.
    Experimental  Works, may change without a major bump.    Keep; mark clearly.
    Internal      Never intended for users.                  Remove from __all__.

"Tested" is load-bearing. Two engines the library's narrative leans on --
``ACTEngine`` and ``UPUGFPruner`` -- are constructed inside ``FewShotLearner``
and exercised only through it: no test names either class. They are
experimental for that reason alone, and will become stable when they have
tests of their own, not before.

Nothing was found to be internal at the name level. What the audit found
instead was the opposite gap: ``PredictionResult`` is what ``predict()``
returns and was not exported, and ``ConfidenceDecomposition`` and
``Counterfactual`` are fields of an exported dataclass and were not either. A
user could hold instances of all three and be unable to name the type.

What a change to each tier costs is written down in CONTRIBUTING.md under
"API Stability and Deprecation".
"""

from __future__ import annotations

#: Supported and semver-protected. A breaking change to any of these needs a
#: deprecation cycle: a warning for one minor release, then removal.
STABLE: tuple[str, ...] = (
    # The configuration contract. The Literal aliases are the types of config
    # fields, so they are as stable as the fields they type -- a user annotating
    # `def run(backbone: Backbone)` needs them, whether or not any example does.
    "AdaptShotConfig",
    "Backbone",
    "CalibrationMethod",
    "ConformalMode",
    "Device",
    "InferenceMode",
    "SimilarityMetric",
    "UncertaintyMode",
    # The learner and what it returns.
    "FewShotLearner",
    "PredictionResult",
    # The exception hierarchy. Callers write `except` against these.
    "AdaptShotError",
    "BackboneError",
    "BufferCapacityError",
    "CalibrationNotReadyError",
    "ConfigValidationError",
    "InvalidImageError",
    # Engines with their own tests and their own consumers.
    "CalibrationEngine",
    "ConformalEngine",
    "ConformalPredictionSet",
    "FeedbackRouter",
    "UncertaintyQuantifier",
    "UncertaintyReport",
)

#: Works, and may change in a minor release without a deprecation cycle. Each
#: docstring opens by saying so. The reason each is here is recorded once, in
#: this comment, rather than repeated in nine docstrings:
#:
#: - Explainability: one test file, no consumer in apps/, examples/ or
#:   benchmarks/. The result's shape is the part most likely to change.
#: - Contrastive: one test file, no consumer. Moved to `training/` in 0.3.0,
#:   because it trains a projection head; the old path warns until 0.4.0.
#: - ACTEngine, UPUGFPruner: no direct tests at all. See the module docstring.
EXPERIMENTAL: tuple[str, ...] = (
    "ConfidenceDecomposition",
    "Counterfactual",
    "ExplainabilityEngine",
    "ExplanationResult",
    "FeatureAttribution",
    "ContrastiveConfig",
    "ContrastivePrototypeLearner",
    "ACTEngine",
    "UPUGFPruner",
)

__all__ = ["EXPERIMENTAL", "STABLE"]
