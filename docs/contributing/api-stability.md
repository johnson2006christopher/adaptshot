# API stability and deprecation

> **For:** anyone depending on a public name, and anyone about to change one. Short, and enforced by a test.

## Two tiers

Every name in `adaptshot.__all__` is classified in `adaptshot.api` as **stable** or **experimental**. The union of the two tuples must equal `__all__`, they must not overlap, and `tests/test_api_surface.py` fails otherwise — so the classification is a fact the suite checks, not a comment that drifts.

| tier | meaning | what a change costs |
|---|---|---|
| **Stable** | supported, semver-protected, and named in at least one test | a deprecation cycle |
| **Experimental** | works; may change in a minor release | a changelog line. The docstring opens with **Experimental** |

"Tested" is load-bearing. An experimental name becomes stable when it has tests of its own and has shipped in at least one release — never by default. Two engines the library leans on, `ACTEngine` and `UPUGFPruner`, were experimental in 0.3.0's first cut for exactly that reason; writing their tests found two inverted terms in the pruner that had shipped in every release, and both are stable now *because* the tests exist.

## The cycle for a stable name

1. The old behaviour keeps working and emits a `DeprecationWarning` — with `stacklevel` set so it points at the caller's line — naming the release it was deprecated in, the release it will be removed in, and what to use instead.
2. It stays for at least one minor release.
3. It is removed in the next minor release, and the removal is in the changelog.

First uses, both in 0.3.0: `adaptshot.core.contrastive` moved to `adaptshot.training.contrastive` (the old path warns; removed in 0.4.0), and three `UncertaintyQuantifier` methods nothing called were deprecated rather than deleted (removed in 0.4.0).

## What is stable in 0.3.0

The configuration contract (`AdaptShotConfig` and its seven `Literal` aliases), `FewShotLearner` and `PredictionResult`, the six exceptions, `CalibrationEngine`, `ConformalEngine` and `ConformalPredictionSet`, `FeedbackRouter`, `UncertaintyQuantifier` and `UncertaintyReport`, `ACTEngine`, `UPUGFPruner`. Twenty-four names.

## What is experimental

The explainability group (`ExplainabilityEngine`, `ExplanationResult`, `FeatureAttribution`, `ConfidenceDecomposition`, `Counterfactual`), the contrastive group (`ContrastiveConfig`, `ContrastivePrototypeLearner`), and the preflight report (`check_environment`, `EnvironmentReport`, `Capability`). Ten names. Why each is there is recorded once, in `adaptshot.api`.

## Adding a name

Add it to `__all__`, choose a tier in `adaptshot.api`, write the docstring marker if experimental, and add it to the [API reference](../reference/api.md) under the matching heading. Miss any of those and the surface test says which.

## Pre-1.0

Semver permits a minor release to break before 1.0. This policy is the promise the project makes anyway. One deliberate exception was taken in 0.3.0 and recorded in the changelog: the default conformal nonconformity score changed, because the old default was defective by measurement; the old score remains selectable by name.
