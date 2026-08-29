# AdaptShot v0.3.0 API Reference

> Every name here is exported from `adaptshot` and classified in `adaptshot.api`.
> `tests/test_api_surface.py` fails if this page, the classification and the
> docstrings disagree.

## Stable and experimental

| Tier | Meaning | What a change costs |
|---|---|---|
| **Stable** | Supported, semver-protected, tested | A deprecation cycle: warn for one minor release, then remove |
| **Experimental** | Works; may change in a minor release | Nothing beyond a changelog line. The docstring says **Experimental** |

The policy is in `CONTRIBUTING.md` under *API Stability and Deprecation*. Two things
were found to be neither: three `UncertaintyQuantifier` methods with no caller anywhere
are deprecated (removed in 0.4.0), and `MemoryTracker` was documented here as an engine
while never having been exported — it is listed at the end, outside the surface.

---

## Stable

### `FewShotLearner`

Main entry point for few-shot learning and inference.

```python
from adaptshot import FewShotLearner, AdaptShotConfig

learner = FewShotLearner(config=AdaptShotConfig(device="cpu"))
```

**Constructor** — `__init__(config: AdaptShotConfig | None = None, **kwargs)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `AdaptShotConfig` | `None` | Central configuration object |
| `**kwargs` | — | — | Passed to `AdaptShotConfig(**kwargs)` if no config is given |

#### `load_support_images(image_paths, labels)`

Ingest a support set and initialise every internal index. Also runs leave-one-out
conformal calibration and bootstrap temperature estimation on the support set, fits the
OOD class distributions, and — when `inference_mode="contrastive"` — trains the
projection head.

```python
learner.load_support_images(
    image_paths=["cat_01.jpg", "cat_02.jpg", "dog_01.jpg", "dog_02.jpg"],
    labels=["cat", "cat", "dog", "dog"],
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `image_paths` | `Sequence[str]` | Paths to RGB images |
| `labels` | `Sequence[str \| int]` | One class label per image |

**Raises** `ConfigValidationError` (mismatched lengths or empty inputs),
`InvalidImageError` (missing, unreadable or non-RGB), `BackboneError` (no usable backend
for the configured backbone on this install), `AdaptShotError` (embedding failure).

#### `predict(image) -> PredictionResult`

Embedding → inference → calibration → ACT gating → conformal set → uncertainty report.

```python
result = learner.predict("query.jpg")
result.prediction              # "cat"
result.calibrated_confidence   # 0.87
result.conformal_set           # ["cat", "dog"]
result.ood_flag                # False
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | `str \| PIL.Image.Image \| np.ndarray` | File path, PIL image, or HWC array |

#### `explain(image) -> ExplanationResult`

Feature attribution, confidence decomposition and a counterfactual, in one object.

The method is stable. **What it returns is experimental**: `ExplanationResult` and the
dataclasses it holds may change in a minor release. See the Experimental section.

#### `correct(image_path, true_label, confidence_weight=1.0) -> dict[str, Any]`

Route a human correction into the continual-learning pipeline and feed the ground-truth
nonconformity score into the conformal engine.

```python
summary = learner.correct("cat_01.jpg", true_label="dog", confidence_weight=0.9)
summary["buffer_size"], summary["calibration_updated"], summary["fine_tuned"]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_path` | `str` | — | Path to the corrected image |
| `true_label` | `str \| int` | — | Human-provided ground truth |
| `confidence_weight` | `float` | `1.0` | Confidence in the correction, `[0, 1]` |

Fine-tuning with CA-EWC needs the `torch` extra; without it `correct()` still updates the
buffer and calibration and reports `fine_tuned: False`.

#### `save(path)` / `FewShotLearner.load(path)`

Persist and restore. SHA-256 integrity check, schema migration from 0.1.x with a
`RuntimeWarning`, atomic writes.

```python
learner.save("checkpoint.json")
restored = FewShotLearner.load("checkpoint.json")
restored.predict("query.jpg")
```

---

### `PredictionResult`

What `predict()` returns. A frozen dataclass.

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | `str \| int` | Predicted class label |
| `raw_confidence` | `float` | Similarity score `[0, 1]` |
| `calibrated_confidence` | `float` | Temperature-scaled confidence `[0, 1]` |
| `neighbor_idx` | `int` | Index of the nearest support example |
| `uncertainty_flag` | `bool` | High-uncertainty flag |
| `act_action` | `str` | `ACCEPT`, `REQUEST_FEEDBACK`, or `REQUEST_FEEDBACK_OOD` |
| `distance_to_prototype` | `float` | Distance to the predicted class prototype |
| `prototype_margin` | `float` | Gap between best and second-best prototype |
| `ood_flag` | `bool` | Out-of-distribution flag (leave-one-out-calibrated Mahalanobis, #54) |
| `debiased_ece` | `float` | Current debiased ECE |
| `conformal_set` | `list[str \| int] \| None` | Conformal prediction set |
| `uncertainty_report` | `dict[str, float] \| None` | Multi-signal uncertainty |
| `nearest_neighbors` | `list[dict] \| None` | Top-5 nearest support examples |

---

### `AdaptShotConfig`

Immutable configuration dataclass, 27 fields. Every field is documented in the
[Config Reference](../reference/config-reference.md). The most-used:

```python
from adaptshot import AdaptShotConfig

config = AdaptShotConfig(backbone="mobilenet_v3_small", device="cpu", conformal_alpha=0.1)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backbone` | `Backbone` | `"mobilenet_v3_small"` | The backbone whose ONNX weights ship in the wheel; `resnet18` needs the `torch` extra |
| `device` | `Device` | `"cpu"` | Compute device |
| `seed` | `int` | `42` | Seed for every source of randomness |
| `n_way` | `int` | `5` | Classes per episode |
| `k_shot` | `int` | `10` | Support examples per class |
| `inference_mode` | `InferenceMode` | `"prototypical"` | Inference strategy |
| `calibration_method` | `CalibrationMethod` | `"temperature"` | Calibration method |
| `conformal_alpha` | `float` | `0.05` | Target miscoverage, `[0.01, 0.50]` |
| `conformal_mode` | `ConformalMode` | `"split"` | Split or cross conformal |
| `uncertainty_mode` | `UncertaintyMode` | `"ensemble"` | Which uncertainty signals to fuse |
| `explainability_enabled` | `bool` | `True` | Enable `explain()` |
| `max_buffer_size` | `int` | `100` | Support buffer capacity |

#### Configuration types

The `Literal` aliases that type the fields above are exported, so code that annotates
against them can name them. They are exactly as stable as the fields they type.

| Alias | Values |
|---|---|
| `Backbone` | `"resnet18"`, `"mobilenet_v3_small"` |
| `Device` | `"cpu"`, `"cuda"`, `"mps"` |
| `SimilarityMetric` | `"cosine"`, `"euclidean"` |
| `InferenceMode` | `"nearest_neighbor"`, `"prototypical"`, `"contrastive"` |
| `CalibrationMethod` | `"temperature"`, `"scaling_binning"`, `"conformal"`, `"none"` |
| `ConformalMode` | `"split"`, `"cross"` |
| `UncertaintyMode` | `"mcdropout"`, `"entropy"`, `"mahalanobis"`, `"ensemble"` |

---

### `CalibrationEngine`

Temperature scaling over a sliding window of observed confidences, with debiased ECE.

```python
from adaptshot import CalibrationEngine

engine = CalibrationEngine(window_size=100)
engine.update(raw_confidence=0.9, predicted_label="cat", true_label="cat")
engine.calibrate(0.9)
```

| Method | Description |
|--------|-------------|
| `update(raw_confidence, predicted_label, true_label)` | Record one observation |
| `calibrate(raw_confidence) -> float` | Apply the fitted temperature |
| `compute_ece(confidences, labels_correct) -> float` | Expected calibration error, equal-width bins |
| `compute_debiased_ece(confidences, labels_correct) -> float` | Debiased squared-CE estimate |

---

### `ConformalEngine` and `ConformalPredictionSet`

Split or cross conformal prediction over a rolling calibration buffer.

```python
from adaptshot import ConformalEngine

engine = ConformalEngine(alpha=0.1, mode="split")
result = engine.predict_set(distances, labels, top_prediction, confidence)
result.prediction_set   # {"cat", "dog"}
result.q_hat            # 0.82
```

| Method | Description |
|--------|-------------|
| `predict_set(distances, labels, top_prediction, confidence)` | Build the prediction set |
| `predict_set_class_conditional(...)` | Class-conditional variant |
| `update_calibration(score, true_label)` | Add a nonconformity score |
| `get_calibration_summary()` | Diagnostic summary |
| `reset()` | Clear the calibration buffer |

`ConformalPredictionSet` fields: `prediction_set`, `set_size`, `alpha`, `q_hat`,
`coverage_estimate`, `prediction`, `confidence`.

**The guarantee needs enough calibration scores to mean anything.** At level α the
conformal quantile is the ⌈(n+1)(1−α)⌉-th smallest score, which does not exist while
n < (1−α)/α. In that region the only honest set is *every class*, and that is what the
engine returns — `q_hat` is `inf`. At the default α = 0.05 that is every n below 19; at
α = 0.10, below 9. `min_informative_size` on the engine is that number, the engine warns
at construction if `min_calibration_size` is below it, and `get_calibration_summary()`
reports it. Before #14 the engine returned the largest observed score instead, and
under-covered — 91.3% against a 95% target at n = 10.

Validated, not asserted: `tests/test_conformal_coverage.py` measures coverage on
overlapping synthetic classes over α ∈ {0.01, 0.05, 0.1, 0.2} × n ∈ {10, 20, 50, 200},
with a tolerance derived from the trial-level standard error. Every cell clears its
target; every cell where a finite quantile exists has a mean set smaller than the label
set.

Measured on real data (PlantVillage 5-way 5-shot, 100 episodes, α = 0.10 with 25
calibration scores — above the informative size): 97.5% empirical coverage at a 90%
target, mean set size 2.05. See the README's results section.

---

### `FeedbackRouter`

Routes a correction to the buffer, the calibration engine, the conformal engine and —
when torch is installed — CA-EWC fine-tuning. `FewShotLearner.correct()` is the usual way
in; use this directly to drive the pipeline from your own feedback loop.

| Method | Description |
|--------|-------------|
| `route_feedback(correction) -> dict[str, Any]` | Apply one correction end to end |

---

### `UncertaintyQuantifier` and `UncertaintyReport`

Epistemic (perturbation sensitivity), aleatoric (k-NN entropy) and distributional
(Mahalanobis OOD) uncertainty, fused into a composite score.

```python
from adaptshot import UncertaintyQuantifier

uq = UncertaintyQuantifier(ood_percentile=95.0)
uq.fit_class_distributions(support_embeddings, support_labels)
report = uq.quantify(query_embedding, support_embeddings, support_labels)
report.epistemic, report.aleatoric, report.is_ood
```

| Method | Description |
|--------|-------------|
| `fit_class_distributions(embeddings, labels)` | Fit shrinkage-regularised class Gaussians and calibrate the OOD threshold by leave-one-out (#54) |
| `quantify(query, support_embeddings, support_labels)` | Full `UncertaintyReport` |
| `mahalanobis_distance(embedding, class_label)` | Distance to one class |
| `min_mahalanobis_distance(embedding)` | Nearest class and margin |
| `is_ood(embedding)` | `(flag, normalised score)` |
| `compute_knn_entropy(query, support_embeddings, support_labels)` | Aleatoric term |
| `estimate_epistemic(embedding, seed=None)` | Epistemic term |
| `reset()` | Clear fitted state |

**Deprecated in 0.3.0, removed in 0.4.0**: `compute_perturbation_variance()`,
`get_ood_summary()`, `get_class_statistics()`. Nothing in the library, its tests or its
applications calls them. Each warns.

`UncertaintyReport` fields: `epistemic`, `aleatoric`, `distributional`, `composite`,
`is_ood`, `ood_score`.

---

### `ACTEngine`

Adaptive Confidence Thresholding: a per-class acceptance threshold that moves with
feedback and reverts slowly toward its base. Tested directly since #74.

```python
from adaptshot import ACTEngine

act = ACTEngine(base_threshold=0.65, learning_rate=0.01, min_threshold=0.50, max_threshold=0.95)
accepted, action = act.should_accept(confidence, class_idx)
```

---
### `UPUGFPruner`

Scores every buffered example by uncertainty, recency and redundancy, and keeps the
`capacity` highest. Exact cosine redundancy up to 100 examples, LSH-approximate beyond.
Tested directly since #74, which found and fixed two inverted terms: it had been keeping
the *confident* examples and, above 100 rows, rewarding duplicates.

```python
from adaptshot import UPUGFPruner

pruner = UPUGFPruner(capacity=100)
embeddings, labels, uncertainties, times = pruner.prune(embeddings, labels, uncertainties, times)
```

---
### Exceptions

Everything raised on purpose derives from `AdaptShotError`, so `except AdaptShotError`
catches the library and nothing else.

| Exception | Raised when |
|-----------|-------------|
| `AdaptShotError` | Base class |
| `InvalidImageError` | An image is missing, unreadable, or not RGB |
| `ConfigValidationError` | A configuration value is outside its supported range, or inputs are malformed |
| `BackboneError` | No usable backend for the requested backbone on this install (#36) — the message names the backbones that would work and the extra that installs torch |
| `CalibrationNotReadyError` | Calibration needs more observations |
| `BufferCapacityError` | Buffer pruning failed to enforce capacity |

---

## Experimental

Each of these opens its docstring with **Experimental**. Why each is here is recorded once,
in `adaptshot.api`.

### `ExplainabilityEngine`

Feature attribution, confidence decomposition with historical penalty tracking, and
counterfactuals. One test file, no consumer outside the library; the shape of the result
is the part most likely to change.

```python
from adaptshot import ExplainabilityEngine

engine = ExplainabilityEngine(top_k_attributions=5)
result = engine.explain(
    query_embedding, support_embeddings, support_labels,
    predicted_label="cat", raw_confidence=0.9,
    calibrated_confidence=0.87, act_action="ACCEPT", is_ood=False,
)
result.summary
```

| Method | Description |
|--------|-------------|
| `explain(...)` | Everything below, in one `ExplanationResult` |
| `attribute(query, support_embeddings, support_labels, predicted_label)` | Feature attribution |
| `decompose_confidence(raw, calibrated, act_action, is_ood)` | Confidence decomposition |
| `counterfactual(query, support_embeddings, support_labels, predicted_label)` | Nearest alternative class |

#### `ExplanationResult`, `FeatureAttribution`, `ConfidenceDecomposition`, `Counterfactual`

The result and the three dataclasses it holds. All four are exported so that a caller can
name the type of what they receive.

| `ExplanationResult` field | Type |
|-------|------|
| `prediction` | `str \| int` |
| `attributions` | `list[FeatureAttribution]` |
| `confidence_decomposition` | `ConfidenceDecomposition \| None` |
| `counterfactual` | `Counterfactual \| None` |
| `summary` | `str` |

---

### `check_environment()`, `EnvironmentReport`, `Capability`

What this install can do on this machine, with every figure measured *here*: a real
inference on the bundled photographs for latency, this process's own high-water mark
for memory, the installed optional dependencies, and for each missing capability the
exact install command. A GPU, if present, is named and not selected.

```python
import adaptshot
print(adaptshot.check_environment())
```

`check_environment(measure=False)` reports availability only, in under a millisecond.
Download sizes for missing extras are **not** reported: they cannot be measured without
the network, and a figure quoted from elsewhere is what this report exists to avoid.

---

### `ContrastivePrototypeLearner`

A two-layer projection head trained by InfoNCE gradient descent, then prototypes refined
in the projected space. Used when `inference_mode="contrastive"`.

Lives in `adaptshot.training` as of 0.3.0, because training a head is training. The old
path `adaptshot.core.contrastive` warns and is removed in 0.4.0.

```python
from adaptshot import ContrastivePrototypeLearner, ContrastiveConfig

learner = ContrastivePrototypeLearner(ContrastiveConfig(projection_dim=128))
prototypes, labels = learner.refine_prototypes(embeddings, labels, seed=42)
prediction, confidence, index = learner.nearest_prototype(query, prototypes, labels)
```

| Method | Description |
|--------|-------------|
| `refine_prototypes(embeddings, labels, seed)` | Train the head, then refine prototypes |
| `nearest_prototype(query, prototypes, labels)` | Nearest refined prototype |
| `class_separation_score(embeddings, labels)` | Inter/intra-class similarity ratio |
| `project_query(embedding)` | Project through the trained head |

#### `ContrastiveConfig`

| Field | Type | Default |
|-------|------|---------|
| `projection_dim` | `int` | `128` |
| `temperature` | `float` | `0.07` |
| `learning_rate` | `float` | `0.01` |
| `momentum` | `float` | `0.9` |
| `n_epochs` | `int` | `50` |

---



## Outside the API surface

Useful, importable, and deliberately not part of the promise above. They can change or go
in any release.

| Name | Where | Note |
|---|---|---|
| `set_deterministic_seed(seed)` | `adaptshot.utils.determinism` | What CLAUDE.md tells every contributor to call. Torch-free since #35 |
| `clear_backbone_cache()` | `adaptshot.core.extractor` | A module-level function, not a learner method — an earlier version of this page said otherwise |
| `MemoryTracker` | `adaptshot.utils.profiling` | No tests, no consumers. See the [profiling tutorial](../tutorials/13_profiling_memory.md) |
| `CAEWCFinetuner` | `adaptshot.training.finetune` | Reached through `FeedbackRouter`; needs the `torch` extra |

---

## Next steps

- [Architecture Deep-Dive](../guides/architecture-deep-dive.md)
- [Algorithm Theory](../guides/algorithm-theory.md)
- [Configuration Reference](../reference/config-reference.md)
