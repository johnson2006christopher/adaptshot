# 📄 File 8: `docs/api/core.md`

### 📝 Content
```markdown
# Core API Reference (v0.1.0)

This document describes the public interfaces for AdaptShot's inference and calibration engine. All signatures, parameters, and behaviors reflect the current `v0.1.0` implementation. Internal/private methods (`_`-prefixed) are omitted as they are subject to change without notice.

---

## `FewShotLearner`

The primary entry point for loading support data, running predictions, routing corrections, and managing session state.

### Initialization
```python
from adaptshot import FewShotLearner

learner = FewShotLearner(
    classes: List[str],
    device: str = "cpu",
    seed: int = 42,
    backbone: str = "resnet18",
    max_buffer_size: int = 100,
    use_faiss: bool = False,
    calibration_method: str = "temperature",
    ece_n_bins: int = 15,
    temperature_init: float = 1.0
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `classes` | `List[str]` | *(required)* | Target class labels for the few-shot task |
| `device` | `str` | `"cpu"` | Execution device (`"cpu"`, `"cuda"`, `"mps"`) |
| `seed` | `int` | `42` | Random seed for deterministic execution |
| `backbone` | `str` | `"resnet18"` | Feature extractor (`"resnet18"` or `"mobilenet_v3_small"`) |
| `max_buffer_size` | `int` | `100` | Maximum replay buffer capacity for corrections |
| `use_faiss` | `bool` | `False` | Enable FAISS-CPU similarity search (requires `adaptshot[faiss]`) |
| `calibration_method` | `str` | `"temperature"` | Post-hoc scaling method (`"temperature"` or `"conformal"`) |
| `ece_n_bins` | `int` | `15` | Number of bins for Expected Calibration Error computation |
| `temperature_init` | `float` | `1.0` | Initial temperature scaling parameter |

### `load_support_images(image_paths, labels)`
Ingests the initial few-shot support set and builds the similarity index.

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
Runs inference on a single query image with calibrated confidence and ACT gating.

```python
result = learner.predict(
    image: Union[str, PIL.Image, np.ndarray]
)
```
**Returns:** `PredictionResult` dataclass (see below)

### `correct(image_path, true_label, confidence_weight=1.0) -> Dict[str, Any]`
Routes a human correction into the continual learning pipeline.

```python
feedback = learner.correct(
    image_path: str,
    true_label: str,
    confidence_weight: float = 1.0
)
```
**Returns:** Dictionary with keys:
- `"buffer_size"`: Current replay buffer length
- `"pending_corrections"`: Corrections awaiting fine-tune trigger
- `"calibration_updated"`: `bool` indicating if ECE/temperature was updated
- `"fine_tuned"`: `bool` indicating if CA-EWC head optimization ran
- `"total_corrections"`: Lifetime correction count

### `save(path)` / `load(path)`
Serializes and restores learner state, including embeddings, calibration history, ACT thresholds, and fine-tuned head weights.

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

Tracks prediction calibration online and applies post-hoc temperature scaling. Designed for streaming evaluation without a held-out validation set.

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
Updates the sliding window with a new prediction and ground truth. Automatically triggers temperature refitting when the window is sufficiently populated (≥50% of `window_size`).

### `calibrate(raw_confidence) -> float`
Applies temperature scaling to a raw cosine similarity score and returns a calibrated confidence value in `[0.0, 1.0]`.

### `compute_ece(confidences, labels_correct) -> float`
Computes Expected Calibration Error on a provided set of predictions. Lower values indicate better alignment between confidence and accuracy.

### Properties
- `current_ece: float` → Most recently computed ECE
- `current_temperature: float` → Current scaling parameter `T`

---

## `ACTEngine`

Adaptive Confidence Thresholding. Dynamically adjusts per-class decision thresholds based on correction history to reduce false acceptances and request human feedback when uncertain.

### Initialization
```python
from adaptshot.core.act import ACTEngine

act = ACTEngine(
    base_threshold: float = 0.65,
    learning_rate: float = 0.01,
    feedback_cost_factor: float = 0.5,
    min_threshold: float = 0.50,
    max_threshold: float = 0.95,
    n_classes: int = 200
)
```

### `should_accept(confidence, class_idx, recent_incorrect_rate=0.0, recent_correct_rate=1.0) -> Tuple[bool, str]`
Evaluates whether to accept a prediction or request human review.

**Returns:** `(accept: bool, action: str)` where `action` is `"ACCEPT"` or `"REQUEST_FEEDBACK"`.

### `get_threshold(class_idx) -> float`
Returns the current adaptive threshold for a specific class.

### `get_all_thresholds() -> Dict[int, float]`
Snapshot of all tracked class thresholds.

---

## `PredictionResult`

Dataclass returned by `FewShotLearner.predict()`.

```python
from dataclasses import dataclass

@dataclass
class PredictionResult:
    prediction: Union[str, int]
    raw_confidence: float
    calibrated_confidence: float
    neighbor_idx: int
    uncertainty_flag: bool
    act_action: str
```

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | `str` / `int` | Predicted class label |
| `raw_confidence` | `float` | Unnormalized cosine similarity score |
| `calibrated_confidence` | `float` | Temperature-scaled confidence in `[0.0, 1.0]` |
| `neighbor_idx` | `int` | Index of the nearest support example in the buffer |
| `uncertainty_flag` | `bool` | `True` if ACT rejected the prediction for human review |
| `act_action` | `str` | `"ACCEPT"` or `"REQUEST_FEEDBACK"` |

---

## ⚠️ v0.1.0 Constraints & Notes
- **Determinism**: All randomness is controlled by `seed`. Call `set_deterministic_seed()` before inference for reproducible results.
- **Image Input**: `predict()` and `load_support_images()` accept file paths, `PIL.Image` objects, or NumPy arrays (HWC, `uint8`). Grayscale images are converted to 3-channel RGB.
- **Calibration Warm-up**: Temperature scaling stabilizes after ~10–15 predictions enter the sliding window. Initial `calibrated_confidence` values may be close to raw similarity scores.
- **Memory Limit**: `max_buffer_size` strictly bounds RAM usage. When exceeded, UP-UGF pruning evicts low-utility examples (falls back to FIFO during initial population).
- **No GPU Training**: CA-EWC fine-tuning runs on the classification head only. Backbone weights remain frozen throughout the lifecycle.

## ▶️ Next Steps
- [Training & Continual Learning API](training.md) → `FeedbackRouter`, `CAEWCFinetuner`, `UPUGFPruner`
- [Configuration Reference](config.md) → `AdaptShotConfig` dataclass and validation rules
- [Contributing](../../CONTRIBUTING.md) → Extension points and PR guidelines
