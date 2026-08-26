# Configuration & Utilities API (v0.2.0)

This document covers AdaptShot's immutable configuration schema with all 27 fields, deterministic execution utilities, memory profiling, and I/O helpers.

---

## `AdaptShotConfig`

A frozen dataclass that centralizes all pipeline hyperparameters. Immutability prevents accidental state mutation, which is critical for deterministic reproducibility.

### Initialization

```python
from adaptshot.config.settings import AdaptShotConfig

config = AdaptShotConfig(
    backbone="resnet18",
    device="cpu",
    seed=42,
    inference_mode="prototypical",
    conformal_alpha=0.10,
    uncertainty_mode="ensemble",
    explainability_enabled=True,
)
```

### Core Execution Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backbone` | `Literal["resnet18", "mobilenet_v3_small"]` | `"resnet18"` | Pretrained feature extractor |
| `device` | `Literal["cpu", "cuda", "mps"]` | `"cpu"` | Execution target. CUDA/MPS are optional |
| `seed` | `int` | `42` | Random seed for reproducibility |
| `verbose` | `bool` | `True` | Enable INFO-level logging |
| `log_dir` | `Optional[str]` | `None` | Optional log output directory |

### Few-Shot Learning Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_way` | `int` | `5` | Number of classes per episode |
| `k_shot` | `int` | `10` | Support examples per class |
| `query_size` | `int` | `15` | Query examples per class for evaluation |

### Similarity & Inference Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `similarity_metric` | `Literal["cosine", "euclidean"]` | `"euclidean"` | Distance metric |
| `inference_mode` | `Literal["nearest_neighbor", "prototypical", "contrastive"]` | `"prototypical"` | Classification strategy |
| `use_faiss` | `bool` | `False` | Enable FAISS-CPU acceleration (>100 support images) |
| `faiss_nprobe` | `int` | `8` | FAISS IVF index probing depth |

### Energy-Aware Inference Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `eco_mode` | `bool` | `False` | Enable energy-saving early-exit (v0.2.0: 32×32 preview, norm ratio guard) |
| `early_exit_threshold` | `float` | `0.95` | Confidence threshold for early-exit [0.5, 1.0] |

### Calibration & Uncertainty Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `calibration_method` | `Literal["temperature", "scaling_binning", "conformal", "none"]` | `"temperature"` | Post-hoc confidence scaling |
| `ece_n_bins` | `int` | `15` | Bins for Expected Calibration Error |
| `calibration_eval_bins` | `int` | `100` | Bins for calibration evaluation (≥ ece_n_bins) |
| `temperature_init` | `float` | `1.0` | Initial temperature scaling parameter |
| `recalibrate_after_feedback` | `bool` | `True` | Recalibrate after each human correction |

### OOD Detection Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_ood_detection` | `bool` | `True` | Flag images outside known distribution |
| `ood_threshold_quantile` | `float` | `0.98` | Quantile for OOD rejection [0.5, 1.0] |
| `ood_absolute_min_distance` | `float` | `0.25` | Minimum absolute distance for OOD flagging |

### v0.2.0 Advanced Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `conformal_alpha` | `float` | `0.05` | Significance level for conformal prediction (0.01–0.50) |
| `conformal_mode` | `Literal["split", "cross"]` | `"split"` | Conformal prediction mode |
| `uncertainty_mode` | `Literal["mcdropout", "entropy", "mahalanobis", "ensemble"]` | `"ensemble"` | Uncertainty quantification mode |
| `explainability_enabled` | `bool` | `True` | Enable XAI with historical penalty tracking |

### Memory Management

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_buffer_size` | `int` | `100` | Maximum replay buffer capacity (enforced by UP-UGF) |

### Validation Constraints

`AdaptShotConfig` enforces immediate validation on instantiation:
- `k_shot > 0` and `n_way > 0`
- `max_buffer_size >= 10`
- `0.5 <= early_exit_threshold <= 1.0`
- `ece_n_bins > 1`
- `calibration_eval_bins >= ece_n_bins`
- `0.5 <= ood_threshold_quantile <= 1.0`
- `ood_absolute_min_distance >= 0.0`
- `0.0 < conformal_alpha < 1.0`
- `conformal_mode` in `("split", "cross")`
- CUDA availability checked via lazy torch import with graceful fallback

!!! warning "Immutability"
    `AdaptShotConfig` is frozen. Attempting to modify attributes after initialization raises `dataclasses.FrozenInstanceError`. Create a new instance with `dataclasses.replace(config, **overrides)` instead.

---

## Determinism Utilities

### `set_deterministic_seed(seed, device=None)`
```python
from adaptshot.utils.determinism import set_deterministic_seed

set_deterministic_seed(seed=42)
```

Sets `random.seed()`, `np.random.seed()`, `torch.manual_seed()`, and `PYTHONHASHSEED=42`. Enables deterministic cuDNN if CUDA is active.

### `verify_determinism(fn, *args, runs=3, seed=42, tolerance=1e-7, **kwargs)`
Executes `fn` multiple times with incrementally offset seeds. Returns `True` if all runs match within tolerance.

---

## Memory Profiling (v0.2.0)

### `MemoryTracker`
Context manager for measuring memory usage during key lifecycle points.

```python
from adaptshot.utils.profiling import MemoryTracker

with MemoryTracker("predict") as tracker:
    result = learner.predict("query.jpg")
print(f"Peak RAM: {tracker.peak_mb:.1f} MB")
print(f"Latency: {tracker.latency_ms:.1f} ms")
```

### `estimate_model_memory_mb(backbone, n_classes)`
Pre-flight memory estimate without loading the model. Returns a dictionary of component-level estimates.

```python
from adaptshot.utils.profiling import estimate_model_memory_mb

est = estimate_model_memory_mb("resnet18", n_classes=5)
print(f"Estimated total: {est['estimated_total_mb']} MB")
```

---

## I/O Utilities

### `validate_path(path, must_exist=False, is_dir=False)`
Normalize and resolve paths. Creates directories if `is_dir=True`. Validates existence if `must_exist=True`.

### `save_json(data, path, indent=2)` / `load_json(path)`
UTF-8 JSON serialization with pretty-formatting and automatic parent directory creation.

### `tensor_to_numpy(tensor)`
Safely converts PyTorch tensors to NumPy arrays, detaching gradients and moving to CPU.

---

## Constraints & Notes

| Component | Limitation | Workaround |
|-----------|------------|------------|
| `AdaptShotConfig` | Frozen; cannot mutate in-place | Use `dataclasses.replace()` |
| `verify_determinism` | Only supports tensor/array return types | Wrap custom functions |
| `MemoryTracker` | Requires psutil for RSS measurement | Falls back to tracemalloc only |
| `validate_path` | No cloud storage paths (S3, GCS) | Mount cloud storage to local dir |

## Next Steps
- [Full API Reference](reference.md) → Every class, method, and data structure
- [Config Reference (Detailed)](../reference/config-reference.md) → Parameter-by-parameter guide
- [Memory Profiling Tutorial](../tutorials/13_profiling_memory.md) → Hands-on profiling walkthrough
