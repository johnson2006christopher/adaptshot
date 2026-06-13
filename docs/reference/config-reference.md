# AdaptShotConfig: Complete Reference

`AdaptShotConfig` is a frozen (immutable) dataclass that controls every aspect of the AdaptShot pipeline. Once created, its values cannot be changed -- guaranteeing deterministic, reproducible behavior across runs. Create a new instance with `dataclasses.replace()` if you need different settings.

Source: [src/adaptshot/config/settings.py](src/adaptshot/config/settings.py)

---

## Quick Start

```python
from adaptshot.config.settings import AdaptShotConfig

config = AdaptShotConfig(
    backbone="resnet18",
    device="cpu",
    seed=42,
)
```

All other fields use sensible defaults. You only need to set what you want to change.

---

## Field Reference (All 22 Fields)

### Category 1: Core Execution (4 fields)

| # | Field | Type | Default | Description |
|---|-------|------|---------|-------------|
| 1 | `backbone` | `Literal["resnet18", "mobilenet_v3_small"]` | `"resnet18"` | Feature extractor backbone. `"resnet18"` provides higher accuracy at the cost of ~45MB model size and slightly slower inference. `"mobilenet_v3_small"` is faster and lighter (~12MB) -- recommended for Raspberry Pi and low-RAM deployments. Both are frozen (not fine-tuned). |
| 2 | `device` | `Literal["cpu", "cuda", "mps"]` | `"cpu"` | Execution device. v0.1.1 is CPU-first: `_validate_config()` rejects non-CPU values with `ConfigValidationError`. CUDA and MPS are opt-in for future releases. |
| 3 | `seed` | `int` | `42` | Master random seed. Controls PyTorch (`torch.manual_seed`), NumPy (`np.random.seed`), Python (`random.seed`), and `PYTHONHASHSEED`. Change this to get different random behavior; fix it to `42` for reproducible benchmarks. |
| 4 | `verbose` | `bool` | `True` | Enable INFO-level logging during pipeline execution. Set to `False` for silent production operation. |

### Category 2: Few-Shot Learning (3 fields)

| # | Field | Type | Default | Description |
|---|-------|------|---------|-------------|
| 5 | `n_way` | `int` | `5` | Number of classes per episode. Controls the few-shot evaluation format. For production classification, the actual number of classes is determined by the unique labels in `load_support_images()`. |
| 6 | `k_shot` | `int` | `10` | Support examples per class. Must be positive. Higher values improve accuracy but increase memory and embedding time. For very constrained deployments, reduce to 3-5. |
| 7 | `query_size` | `int` | `15` | Query examples per class for evaluation mode. Used in benchmark scripts. Not directly used in production `predict()` calls. |

### Category 3: Similarity & Inference (4 fields)

| # | Field | Type | Default | Description |
|---|-------|------|---------|-------------|
| 8 | `similarity_metric` | `Literal["cosine", "euclidean"]` | `"euclidean"` | Distance metric for comparing embeddings. `"euclidean"` is faster and works well with normalized embeddings. `"cosine"` is direction-aware and robust to embedding magnitude differences. |
| 9 | `inference_mode` | `Literal["nearest_neighbor", "prototypical"]` | `"prototypical"` | Classification strategy. `"nearest_neighbor"` finds the single closest support example. `"prototypical"` computes a class prototype (mean embedding) and compares against it -- generally more robust with small support sets. |
| 10 | `use_faiss` | `bool` | `False` | Enable FAISS-CPU acceleration for similarity search. Improves latency for support sets >100 images. Requires `pip install "adaptshot[faiss]"`. Falls back to NumPy if FAISS is not installed. |
| 11 | `faiss_nprobe` | `int` | `8` | FAISS IVF index probing depth. Higher values improve accuracy but increase search time. Only used when `use_faiss=True` and an IVF index is active. |

### Category 4: Energy-Aware Inference (2 fields)

| # | Field | Type | Default | Description |
|---|-------|------|---------|-------------|
| 12 | `eco_mode` | `bool` | `False` | Enable energy-saving early-exit. When `True`, the pipeline can exit early if a high-confidence match is found before full similarity computation. Reduces carbon footprint by up to 40% in benchmark testing. |
| 13 | `early_exit_threshold` | `float` | `0.95` | Confidence threshold for early-exit. Must be in [0.5, 1.0]. Higher values (e.g., 0.98) are more conservative and exit less often. Lower values (e.g., 0.85) save more energy but may miss subtle distinctions. Only active when `eco_mode=True`. |

### Category 5: Calibration & Uncertainty (5 fields)

| # | Field | Type | Default | Description |
|---|-------|------|---------|-------------|
| 14 | `calibration_method` | `Literal["temperature", "scaling_binning", "conformal", "none"]` | `"temperature"` | Post-hoc confidence calibration strategy. `"temperature"` applies a single scaling parameter (fast, stable). `"scaling_binning"` uses bin-wise scaling (more granular). `"conformal"` provides distribution-free prediction sets (stub in v0.1.1). `"none"` skips calibration entirely. |
| 15 | `ece_n_bins` | `int` | `15` | Number of bins for Expected Calibration Error (ECE) computation. Must be >1. More bins give finer-grained ECE tracking but require more observations for statistical validity. |
| 16 | `calibration_eval_bins` | `int` | `100` | Number of bins for calibration evaluation. Must be >= `ece_n_bins`. Controls the resolution of calibration quality reporting. |
| 17 | `temperature_init` | `float` | `1.0` | Initial temperature scaling parameter. Must be positive. A value of 1.0 means no scaling (raw confidence). The calibration engine adjusts this automatically as predictions accumulate. |
| 18 | `recalibrate_after_feedback` | `bool` | `True` | Whether to trigger calibration updates after each human correction. When `True`, the calibration window incorporates new feedback immediately. Set to `False` to batch calibration updates for efficiency. |

### Category 6: OOD (Out-of-Distribution) Detection (3 fields)

| # | Field | Type | Default | Description |
|---|-------|------|---------|-------------|
| 19 | `enable_ood_detection` | `bool` | `True` | Enable out-of-distribution detection. When `True`, images whose distance to any known prototype exceeds the threshold are flagged and routed for human review. When `False`, OOD threshold is set to infinity (no images are flagged). |
| 20 | `ood_threshold_quantile` | `float` | `0.98` | Quantile threshold for OOD rejection. Must be in [0.5, 1.0]. Uses the p98 distance among support examples (by default) as the cutoff. Higher values are more permissive; lower values flag more images as OOD. |
| 21 | `ood_absolute_min_distance` | `float` | `0.25` | Minimum absolute distance for OOD flagging. Must be >= 0.0. Acts as a floor: even if the quantile threshold is lower, images closer than this distance are never flagged. |

### Category 7: Memory Management (1 field)

| # | Field | Type | Default | Description |
|---|-------|------|---------|-------------|
| 22 | `max_buffer_size` | `int` | `100` | Maximum replay buffer capacity. Must be >= 10. Controls the number of support embeddings retained in memory. When exceeded, UP-UGF pruning evicts low-utility examples based on uncertainty x recency x redundancy scoring. RAM usage scales linearly with this value (each embedding is ~2KB for ResNet-18 512-dim). |

---

## Validation Rules

`AdaptShotConfig.__post_init__()` enforces these constraints on creation:

| Rule | Error if violated |
|------|-------------------|
| `k_shot > 0` and `n_way > 0` | `ValueError` |
| `max_buffer_size >= 10` | `ValueError` |
| `0.5 <= early_exit_threshold <= 1.0` | `ValueError` |
| `ece_n_bins > 1` | `ValueError` |
| `calibration_eval_bins >= ece_n_bins` | `ValueError` |
| `0.5 <= ood_threshold_quantile <= 1.0` | `ValueError` |
| `ood_absolute_min_distance >= 0.0` | `ValueError` |
| `device == "cuda"` but CUDA unavailable | `RuntimeWarning` (falls back to CPU) |

---

## Immutability

`AdaptShotConfig` is created with `@dataclass(frozen=True)`. Attempting to modify attributes after creation raises `FrozenInstanceError`:

```python
config = AdaptShotConfig()
config.device = "cuda"  # Raises FrozenInstanceError
```

Use `dataclasses.replace()` to create a modified copy:

```python
from dataclasses import replace

config = AdaptShotConfig(backbone="resnet18")
eco_config = replace(config, eco_mode=True, early_exit_threshold=0.90)
```

---

## Recommended Configurations

### Default (Balanced)
```python
AdaptShotConfig(
    backbone="resnet18",
    device="cpu",
    seed=42,
    max_buffer_size=100,
)
```

### Low-Power / Raspberry Pi
```python
AdaptShotConfig(
    backbone="mobilenet_v3_small",
    device="cpu",
    max_buffer_size=30,
    eco_mode=True,
    early_exit_threshold=0.90,
    use_faiss=False,
)
```

### Maximum Accuracy
```python
AdaptShotConfig(
    backbone="resnet18",
    inference_mode="prototypical",
    similarity_metric="cosine",
    calibration_method="scaling_binning",
    ece_n_bins=30,
    enable_ood_detection=True,
    ood_threshold_quantile=0.99,
)
```

### Benchmark / CI
```python
AdaptShotConfig(
    backbone="resnet18",
    device="cpu",
    seed=42,
    n_way=5,
    k_shot=10,
    verbose=False,
    max_buffer_size=200,
)
```

---

## Verification Checklist

- [ ] You can create a config with `AdaptShotConfig()` and all defaults are correct.
- [ ] You understand which fields to change for your hardware and use case.
- [ ] You know that `device` must always be `"cpu"` in v0.1.1.
- [ ] You understand the difference between `similarity_metric` (`cosine` vs `euclidean`) and `inference_mode` (`nearest_neighbor` vs `prototypical`).
- [ ] You can trace every field back to `src/adaptshot/config/settings.py`.

---

*Created by [Johnson Christopher Hassan](https://github.com/johnson2006christopher)*  
*Connect on [LinkedIn](https://www.linkedin.com/in/johnson-hassan-935124311/)*  
*Project: [github.com/johnson2006christopher/adaptshot](https://github.com/johnson2006christopher/adaptshot)*
