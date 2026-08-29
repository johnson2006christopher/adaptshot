# Core API Reference (v0.2.0)

This document describes the public interfaces for AdaptShot's inference, calibration, uncertainty, conformal, contrastive, and explainability engines. All signatures, parameters, and behaviors reflect the current `v0.2.0` implementation. Internal/private methods (`_`-prefixed) are omitted as they are subject to change without notice.

---

## `FewShotLearner`

The primary entry point for loading support data, running predictions with conformal sets, routing corrections, and managing session state.

### Initialization
```python
from adaptshot import FewShotLearner
from adaptshot.config.settings import AdaptShotConfig

config = AdaptShotConfig(
    backbone="resnet18",
    device="cpu",
    seed=42,
    max_buffer_size=100,
    use_faiss=False,
    inference_mode="prototypical",
    conformal_alpha=0.10,
    explainability_enabled=True,
)
learner = FewShotLearner(config=config)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `AdaptShotConfig` | `None` | Immutable configuration dataclass; kwargs accepted if config is `None` |

See [Configuration Reference](config.md) for the complete `AdaptShotConfig` field listing.

### `load_support_images(image_paths, labels)`
Ingests the initial few-shot support set and builds the similarity index. In v0.2.0, this also:

- Performs **true leave-one-out conformal calibration** (prototype recomputation per example)
- Estimates **bootstrap temperature** via LOO cross-validation (no separate validation set needed)
- Initializes the contrastive projection head (if `inference_mode="contrastive"`)
- Fits class-conditional uncertainty distributions with **shrinkage covariance estimation**

```python
learner.load_support_images(
    image_paths: List[str],
    labels: List[str]
)
```

**Behavior:**
- Opens each image, applies ImageNet-standard preprocessing, and extracts frozen backbone embeddings
- Stores embeddings in CPU memory
- Initializes a lightweight classification head for future CA-EWC fine-tuning
- Raises `ValueError` if `len(image_paths) != len(labels)`

### `predict(image) -> PredictionResult`
Runs inference on a single query image with calibrated confidence, ACT gating, conformal sets, and multi-signal uncertainty.

**v0.2.0 Pipeline**:
1. Extract embedding via frozen backbone
2. Route to inference mode (nearest_neighbor / prototypical / contrastive)
3. Compute raw confidence via similarity search
4. Apply bootstrap temperature calibration (or sliding-window temperature if warm)
5. Run ACT with symmetric threshold update + mean-reversion
6. Generate conformal prediction set
7. Quantify epistemic (stochastic perturbation) and aleatoric (k-NN entropy) uncertainty
8. Run shrinkage-regularized Mahalanobis OOD detection

```python
result = learner.predict(
    image: Union[str, PIL.Image, np.ndarray]
)
```

**Returns:** `PredictionResult` dataclass (see below)

### `correct(image_path, true_label, confidence_weight=1.0) -> Dict[str, Any]`
Routes a human correction into the continual learning pipeline. v0.2.0: feeds ground-truth nonconformity scores into the conformal engine.

```python
feedback = learner.correct(
    image_path: str,
    true_label: Union[str, int],
    confidence_weight: float = 1.0
)
```

**Returns:** Dictionary with keys:
- `"buffer_size"`: Current replay buffer length
- `"pending_corrections"`: Corrections awaiting fine-tune trigger
- `"calibration_updated"`: `bool` indicating if ECE/temperature was updated
- `"fine_tuned"`: `bool` indicating if CA-EWC head optimization ran
- `"total_corrections"`: Lifetime correction count
- `"calibration_summary"`: (v0.2.0, if `recalibrate_after_feedback=True`) Current calibration diagnostics
- `"buffer_management_warning"`: (if pruning encountered issues) Warning message

### `save(path)` / `load(path)`
Serializes and restores learner state. v0.2.0: SHA-256 integrity verification, schema version migration from v0.1.x, atomic writes.

```python
learner.save("checkpoint.json")
restored = FewShotLearner.load("checkpoint.json")
```

**File Artifacts Created on Save:**
- `{path}.json` (metadata, config, calibration, thresholds, buffer metadata)
- `{path}.embeddings.npy` (NumPy array of support/correction embeddings)
- `{path}.head.pt` (PyTorch state dict for fine-tuned classification head)

---

## `CalibrationEngine`

Tracks prediction calibration online and applies post-hoc temperature scaling. v0.2.0 adds bootstrap temperature estimation via LOO cross-validation for cold-start scenarios.

### Initialization
```python
from adaptshot.core.calibration import CalibrationEngine

calibrator = CalibrationEngine(
    n_bins: int = 15,
    window_size: int = 100,
    temperature_init: float = 1.0,
    method: str = "temperature"
)
```

### `update(raw_confidence, predicted_label, true_label)`
Updates the sliding window with a new prediction and ground truth. Automatically triggers temperature refitting.

### `calibrate(raw_confidence) -> float`
Applies temperature scaling to a raw similarity score and returns a calibrated confidence in `[0.0, 1.0]`.

### `compute_ece(confidences, labels_correct) -> float`
Computes Expected Calibration Error. Lower values indicate better alignment between confidence and accuracy.

### Properties
- `current_ece: float` → Most recently computed ECE
- `current_temperature: float` → Current scaling parameter `T`

---

## `ACTEngine`

Adaptive Confidence Thresholding with symmetric updates and mean-reversion (v0.2.0). Prevents monotonic drift toward extreme thresholds.

### Initialization
```python
from adaptshot.core.act import ACTEngine

act = ACTEngine(
    base_threshold: float = 0.65,
    eta: float = 0.01,
    min_threshold: float = 0.50,
    max_threshold: float = 0.95,
    n_classes: int = 200
)
```

**v0.2.0 Update Formula**:
```
delta = η * (incorrect_rate − correct_rate)
delta += mean_reversion_strength * (base_threshold − threshold)
threshold = clip(threshold + delta, min_threshold, max_threshold)
```

### `should_accept(confidence, class_idx, recent_incorrect_rate=0.0, recent_correct_rate=1.0) -> Tuple[bool, str]`
Evaluates whether to accept a prediction or request human review.

**Returns:** `(accept: bool, action: str)` where `action` is `"ACCEPT"` or `"REQUEST_FEEDBACK"`.

---

## `PredictionResult`

Dataclass returned by `FewShotLearner.predict()`. Extended in v0.2.0 with conformal sets, uncertainty reports, and explanation results.

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | `str` / `int` | Predicted class label |
| `raw_confidence` | `float` | Unnormalized similarity score |
| `calibrated_confidence` | `float` | Temperature-scaled confidence in `[0.0, 1.0]` (v0.2.0: bootstrap temp on cold start) |
| `neighbor_idx` | `int` | Index of the nearest support example |
| `uncertainty_flag` | `bool` | `True` if ACT or OOD rejected the prediction |
| `act_action` | `str` | `"ACCEPT"`, `"REQUEST_FEEDBACK"`, or `"REQUEST_FEEDBACK_OOD"` |
| `conformal_set` | `List` | v0.2.0: Prediction set with guaranteed coverage |
| `uncertainty_report` | `Dict` | v0.2.0: Epistemic, aleatoric, distributional signals |
| `nearest_neighbors` | `List[Dict]` | v0.2.0: Top-5 nearest support examples with distances |
| `ood_flag` | `bool` | v0.2.0: Shrinkage-regularized Mahalanobis OOD detection |
| `distance_to_prototype` | `float` | Distance to predicted class prototype |
| `prototype_margin` | `float` | Gap between best and second-best prototype |

---

## Constraints & Notes

- **Determinism**: All randomness is controlled by `seed`. Call `set_deterministic_seed()` before inference.
- **Image Input**: Accepts file paths, `PIL.Image` objects, or NumPy arrays (HWC, `uint8`).
- **Bootstrap Calibration**: On first predict after `load_support_images`, temperature is estimated via LOO cross-validation. No separate validation set required.
- **Conformal Coverage**: True leave-one-out calibration provides valid finite-sample coverage guarantees under exchangeability.
- **Shrinkage Covariance**: Mahalanobis OOD uses adaptive alpha = d/(d+n_k) to prevent singular matrices.
- **Contrastive Head**: Projection head is gradient-trained via InfoNCE (not just initialized). Full backpropagation through W1/b1/W2/b2.
- **ACT Mean-Reversion**: Thresholds slowly pull toward base, preventing monotonic drift.
- **No GPU Required**: CA-EWC fine-tuning runs on the classification head only (~2K parameters). Backbone weights remain frozen.

## Next Steps
- [Training & Continual Learning API](training.md) → `FeedbackRouter`, `CAEWCFinetuner`, `UPUGFPruner`
- [Configuration Reference](config.md) → `AdaptShotConfig` dataclass and validation rules
- [Full API Reference](api.md) → Every class and method
