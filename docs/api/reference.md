# AdaptShot v0.2.0 API Reference

> Complete reference for all public classes, methods, and data structures

---

## Core Classes

### `FewShotLearner`

Main entry point for few-shot learning and inference.

```python
from adaptshot import FewShotLearner, AdaptShotConfig

learner = FewShotLearner(config=AdaptShotConfig(device="cpu"))
```

**Constructor** — `__init__(config: Optional[AdaptShotConfig] = None, **kwargs)`:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `AdaptShotConfig` | `None` | Central configuration object |
| `**kwargs` | — | — | Passed to `AdaptShotConfig(**kwargs)` if no config given |

---

#### `load_support_images(image_paths, labels)`

Ingest a support set and initialize all internal indices. In v0.2.0, this also triggers true leave-one-out conformal calibration and bootstrap temperature estimation for autonomous operation.

```python
learner.load_support_images(
    image_paths=["cat_01.jpg", "cat_02.jpg", "dog_01.jpg", "dog_02.jpg"],
    labels=["cat", "cat", "dog", "dog"],
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `image_paths` | `List[str]` | Absolute or relative paths to RGB images |
| `labels` | `List[Union[str, int]]` | Class labels, one per image |

**Raises**:
- `ConfigValidationError` — mismatched lengths or empty inputs
- `InvalidImageError` — missing file, unreadable, or non-RGB
- `AdaptShotError` — embedding extraction failure

**v0.2.0 Side Effects**:
- True leave-one-out conformal calibration: prototypes recomputed per example
- Bootstrap temperature estimation via LOO cross-validation
- Contrastive projection head initialization (if inference_mode="contrastive")
- Uncertainty class distributions fitted with shrinkage covariance

---

#### `predict(image) -> PredictionResult`

Run inference on a single image with full v0.2.0 pipeline: embedding → inference → calibration → ACT gating → conformal set → uncertainty report.

```python
result = learner.predict("query.jpg")
print(result.prediction)              # "cat"
print(result.calibrated_confidence)   # 0.87
print(result.conformal_set)           # {"cat", "dog"}
print(result.uncertainty_report)      # {epistemic: 0.04, aleatoric: 0.12, ...}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | `Union[str, Image.Image, np.ndarray]` | File path, PIL image, or HWC array |

**Returns** — `PredictionResult` with fields:
| Field | Type | Description |
|-------|------|-------------|
| `prediction` | `Union[str, int]` | Predicted class label |
| `raw_confidence` | `float` | Similarity score [0, 1] |
| `calibrated_confidence` | `float` | Temperature-scaled confidence [0, 1] (v0.2.0: bootstrap temp on cold start) |
| `neighbor_idx` | `int` | Index of nearest support example |
| `uncertainty_flag` | `bool` | High uncertainty flag |
| `act_action` | `str` | `ACCEPT`, `REQUEST_FEEDBACK`, or `REQUEST_FEEDBACK_OOD` |
| `distance_to_prototype` | `float` | Distance to predicted class prototype |
| `prototype_margin` | `float` | Gap between best and second-best prototype |
| `ood_flag` | `bool` | Out-of-distribution detection (v0.2.0: shrinkage-regularized Mahalanobis) |
| `debiased_ece` | `float` | Current debiased ECE |
| `conformal_set` | `List[Union[str, int]]` | v0.2.0: Conformal prediction set (true LOO calibration) |
| `uncertainty_report` | `Dict[str, float]` | v0.2.0: Multi-signal uncertainty (shrinkage-regularized) |
| `nearest_neighbors` | `List[Dict]` | v0.2.0: Top-5 nearest support examples |

---

#### `explain(image) -> ExplanationResult`

Generate a multi-faceted explanation for a prediction (v0.2.0). Uses historical penalty tracking for confidence decomposition — no magic numbers.

```python
explanation = learner.explain("query.jpg")
print(explanation.summary)
# "Predicted 'cat' with confidence 0.870. Most influenced by support example #3..."
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | `Union[str, Image.Image, np.ndarray]` | Query image |

**Returns** — `ExplanationResult`:
| Field | Type | Description |
|-------|------|-------------|
| `prediction` | `Union[str, int]` | Predicted class |
| `attributions` | `List[FeatureAttribution]` | Top-k influential support examples |
| `confidence_decomposition` | `ConfidenceDecomposition` | Raw→calibrated→ACT→OOD breakdown (historical penalty tracking) |
| `counterfactual` | `Counterfactual` | Nearest alternative class |
| `summary` | `str` | Human-readable explanation text |

---

#### `correct(image_path, true_label, confidence_weight=1.0) -> Dict`

Route a human correction into the continual learning pipeline. v0.2.0: feeds ground-truth nonconformity scores into the conformal engine.

```python
summary = learner.correct(
    image_path="cat_01.jpg",
    true_label="dog",
    confidence_weight=0.9,
)
print(summary["buffer_size"])         # 102
print(summary["calibration_updated"]) # True
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_path` | `str` | — | Path to corrected image |
| `true_label` | `Union[str, int]` | — | Human-provided ground truth |
| `confidence_weight` | `float` | `1.0` | [0.0, 1.0] confidence in correction |

**Returns** dict with keys: `buffer_size`, `calibration_updated`, `fine_tuned`, `total_corrections`

---

#### `save(path)` / `load(path)`

Persist and restore learner state. v0.2.0: SHA-256 integrity verification, schema version migration from v0.1.x, atomic writes.

```python
learner.save("checkpoint.json")
restored = FewShotLearner.load("checkpoint.json")
assert restored._is_initialized is True
```

#### `clear_backbone_cache()`

Clear the `@lru_cache` on the backbone factory. Call when switching backbones at runtime (v0.2.0).

```python
learner.clear_backbone_cache()
```

---

### `AdaptShotConfig`

Immutable configuration dataclass with 26 fields. See [Config Reference](../reference/config-reference.md) for all fields.

```python
from adaptshot import AdaptShotConfig

config = AdaptShotConfig(
    backbone="resnet18",
    device="cpu",
    inference_mode="prototypical",
    conformal_alpha=0.05,
    explainability_enabled=True,
    uncertainty_mode="ensemble",
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backbone` | `Literal["resnet18", "mobilenet_v3_small"]` | `"resnet18"` | Backbone architecture |
| `device` | `Literal["cpu", "cuda", "mps"]` | `"cpu"` | Compute device |
| `seed` | `int` | `42` | Random seed for reproducibility |
| `n_way` | `int` | `5` | Classes per episode |
| `k_shot` | `int` | `10` | Support examples per class |
| `inference_mode` | `Literal["nearest_neighbor", "prototypical", "contrastive"]` | `"prototypical"` | Inference strategy |
| `calibration_method` | `Literal["temperature", "scaling_binning", "conformal", "none"]` | `"temperature"` | Calibration method |
| `conformal_alpha` | `float` | `0.05` | v0.2.0: Miscoverage rate (0.01–0.50) |
| `conformal_mode` | `Literal["split", "cross"]` | `"split"` | v0.2.0: Conformal mode |
| `uncertainty_mode` | `Literal["mcdropout", "entropy", "mahalanobis", "ensemble"]` | `"ensemble"` | v0.2.0: Uncertainty mode |
| `explainability_enabled` | `bool` | `True` | v0.2.0: Enable XAI with historical penalty tracking |
| `max_buffer_size` | `int` | `100` | Support buffer capacity |

---

## Advanced Engines (v0.2.0)

### `ConformalEngine`

True leave-one-out calibration for valid finite-sample coverage guarantees.

```python
from adaptshot import ConformalEngine

engine = ConformalEngine(alpha=0.05, mode="split")
result = engine.predict_set(distances, labels, top_prediction, confidence)
# result.prediction_set → {"cat", "dog"}
# result.q_hat → 0.82
```

| Method | Description |
|--------|-------------|
| `predict_set(distances, labels, top_pred, conf)` | Generate conformal prediction set |
| `predict_set_class_conditional(...)` | Class-conditional variant |
| `update_calibration(score, true_label)` | Add calibration score |
| `get_calibration_summary()` | Diagnostic summary |
| `reset()` | Clear calibration buffer |

### `ConformalPredictionSet`
| Field | Type | Description |
|-------|------|-------------|
| `prediction_set` | `Set` | Classes in the prediction set |
| `set_size` | `int` | Number of included classes |
| `alpha` | `float` | Significance level |
| `q_hat` | `float` | Quantile threshold |
| `coverage_estimate` | `float` | Empirical coverage rate |

**v0.2.0 Production Hardening**: True leave-one-out prototype recomputation. Each support example is held out and prototypes are recomputed, providing valid finite-sample coverage guarantees under exchangeability.

---

### `ContrastivePrototypeLearner`

Gradient-trained projection head (v0.2.0). The 2-layer MLP (W1, b1, W2, b2) is now trained via **full InfoNCE gradient descent** with momentum SGD, not just initialized.

```python
from adaptshot import ContrastivePrototypeLearner, ContrastiveConfig

learner = ContrastivePrototypeLearner()
prototypes, labels = learner.refine_prototypes(embeddings, labels, seed=42)
pred, conf, idx = learner.nearest_prototype(query, prototypes, labels)
```

| Method | Description |
|--------|-------------|
| `refine_prototypes(embeddings, labels, seed)` | Train projection head via InfoNCE, then refine prototypes |
| `_train_projection_head(embeddings, labels, label_indices, seed)` | InfoNCE gradient descent through W1/b1/W2/b2 |
| `nearest_prototype(query, prototypes, labels)` | Find nearest refined prototype |
| `class_separation_score(embeddings, labels)` | Measure class separability |
| `project_query(embedding)` | Project through trained head |

### `ContrastiveConfig`
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `projection_dim` | `int` | `128` | Projection head output dim |
| `temperature` | `float` | `0.07` | InfoNCE temperature |
| `learning_rate` | `float` | `0.01` | SGD learning rate for head training (v0.2.0) |
| `momentum` | `float` | `0.9` | SGD momentum for head training (v0.2.0) |
| `n_epochs` | `int` | `50` | Training epochs for projection head |

---

### `UncertaintyQuantifier`

Shrinkage-regularized Mahalanobis OOD detection (v0.2.0). Covariance estimation uses adaptive alpha = d/(d+n_k) to prevent singular matrices in high-dimensional few-shot settings.

```python
from adaptshot import UncertaintyQuantifier

uq = UncertaintyQuantifier(ood_percentile=95.0)
uq.fit_class_distributions(embeddings, labels)
report = uq.quantify(query_emb, support_embs, support_labels)
# report.epistemic → 0.12
# report.aleatoric → 0.08
# report.is_ood → False
```

| Method | Description |
|--------|-------------|
| `fit_class_distributions(embeddings, labels)` | Fit class-conditional Gaussians with shrinkage covariance |
| `quantify(query, support_embs, support_labels)` | Full uncertainty report |
| `mahalanobis_distance(embedding, class_label)` | Distance to class distribution (shrinkage-regularized) |
| `is_ood(embedding)` | OOD check |
| `compute_knn_entropy(query, support_embs, support_labels)` | Aleatoric entropy |
| `estimate_epistemic(embedding, seed=None)` | Stochastic embedding perturbation sensitivity (v0.2.0) |

### `UncertaintyReport`
| Field | Type | Description |
|-------|------|-------------|
| `epistemic` | `float` | Perturbation sensitivity [0, 1] |
| `aleatoric` | `float` | k-NN entropy [0, 1] |
| `distributional` | `float` | Shrinkage-regularized Mahalanobis OOD score [0, 1] |
| `composite` | `float` | Weighted fusion [0, 1] |
| `is_ood` | `bool` | OOD flag |

---

### `ExplainabilityEngine`

Historical penalty tracking for confidence decomposition (v0.2.0). No magic numbers — ACT and OOD penalties are tracked in 20-window sliding averages.

```python
from adaptshot import ExplainabilityEngine

engine = ExplainabilityEngine(top_k_attributions=5)
result = engine.explain(
    query_embedding, support_embeddings, support_labels,
    predicted_label="cat", raw_confidence=0.9,
    calibrated_confidence=0.87, act_action="ACCEPT", is_ood=False,
)
print(result.summary)
```

| Method | Description |
|--------|-------------|
| `explain(query_emb, support_embs, support_labels, ...)` | Full explanation with historical penalty tracking |
| `attribute(query_emb, support_embs, support_labels, pred_label)` | Feature attribution |
| `decompose_confidence(raw, cal, act_action, is_ood)` | Confidence decomposition (historical averages) |
| `counterfactual(query_emb, support_embs, support_labels, pred_label)` | Counterfactual only |

---

### `MemoryTracker` (v0.2.0)

Lightweight memory profiling context manager. Uses tracemalloc with optional psutil enhancement.

```python
from src.adaptshot.utils.profiling import MemoryTracker, estimate_model_memory_mb

# Pre-flight estimate
est = estimate_model_memory_mb("resnet18", n_classes=5)

# Runtime profiling
with MemoryTracker("predict") as tracker:
    result = learner.predict("query.jpg")
print(f"Peak: {tracker.peak_mb:.1f} MB")
```

---

### `ACTEngine`

Symmetric threshold updates with mean-reversion (v0.2.0). Prevents monotonic drift toward extreme thresholds.

```python
from adaptshot.core.act import ACTEngine

act = ACTEngine(
    base_threshold=0.65,
    eta=0.01,
    min_threshold=0.50,
    max_threshold=0.95,
)
```

**v0.2.0 Update Formula**: `delta = η * (incorrect_rate − correct_rate) + mean_reversion_strength * (base − threshold)`

---

### `UPUGFPruner`

LSH-accelerated redundancy scoring (v0.2.0). Exact cosine similarity for N≤100, random projection LSH for O(N log N) approximate mode at N>100.

```python
from adaptshot.training.up_ugf import UPUGFPruner

pruner = UPUGFPruner(capacity=100)
```

---

## Exception Hierarchy

| Exception | Parent | Raised When |
|-----------|--------|-------------|
| `AdaptShotError` | `Exception` | General AdaptShot errors |
| `InvalidImageError` | `AdaptShotError` | Non-RGB, missing, or corrupt images |
| `ConfigValidationError` | `AdaptShotError` | Invalid configuration parameters |
| `CalibrationNotReadyError` | `AdaptShotError` | (v0.2.0: rarely raised; graceful fallback used instead) |
| `BufferCapacityError` | `AdaptShotError` | UP-UGF pruning failure |

---

## Next Steps

- [Architecture Deep-Dive](../guides/architecture-deep-dive.md)
- [Algorithm Theory](../guides/algorithm-theory.md)
- [Configuration Reference](../reference/config-reference.md)
- [Memory Profiling Tutorial](../tutorials/13_profiling_memory.md)
