# Error Handling & Troubleshooting

This guide covers every exception AdaptShot can raise, what each one means, and how to fix the underlying problem. It also covers common runtime issues that don't produce exceptions.

---

## Exception Hierarchy

All AdaptShot exceptions inherit from `AdaptShotError`:

```
AdaptShotError
├── ConfigValidationError      # Invalid configuration
├── InvalidImageError           # Image loading/format problems
├── CalibrationNotReadyError    # Calibration window not populated
└── BufferCapacityError         # Buffer pruning issues
```

Source: [src/adaptshot/utils/exceptions.py](src/adaptshot/utils/exceptions.py)

---

## Exception Reference

### `ConfigValidationError`

**When**: Configuration values violate constraints.

| Error Message | Cause | Fix |
|---------------|-------|-----|
| `"device must be 'cpu'"` | `device` set to `"cuda"` or `"mps"` | Set `device="cpu"`. v0.1.1 is CPU-first. |
| `"n_way and k_shot must be positive integers"` | `k_shot=0` or `n_way=0` | Set both to positive integers (e.g., `n_way=5, k_shot=10`). |
| `"max_buffer_size must be >= 10"` | Buffer too small for meaningful operation | Set `max_buffer_size` to at least 10. |
| `"early_exit_threshold must be within [0.5, 1.0]"` | Threshold outside valid range | Set a value between 0.5 and 1.0. |
| `"ece_n_bins must be > 1"` | ECE bins set to 0 or 1 | Set `ece_n_bins` to at least 2. |
| `"calibration_eval_bins must be >= ece_n_bins"` | Eval bins smaller than ECE bins | Increase `calibration_eval_bins` or decrease `ece_n_bins`. |
| `"ood_threshold_quantile must be in [0.5, 1.0]"` | OOD quantile out of range | Set between 0.5 and 1.0. |
| `"ood_absolute_min_distance must be >= 0.0"` | Negative distance threshold | Set to 0.0 or higher. |
| `"confidence_weight must be in [0.0, 1.0]"` | Correction weight outside range | Use a value between 0.0 and 1.0. |
| `"similarity_metric must be 'cosine' or 'euclidean'"` | Invalid metric name | Use `"cosine"` or `"euclidean"`. |
| `"inference_mode must be 'nearest_neighbor' or 'prototypical'"` | Invalid mode name | Use `"nearest_neighbor"` or `"prototypical"`. |

### `InvalidImageError`

**When**: Image loading, format, or content fails.

| Error Message Pattern | Cause | Fix |
|----------------------|-------|-----|
| `"Image file not found: '{path}'"` | File doesn't exist at given path | Verify the path is correct and the file exists. Use absolute paths for services. |
| `"Expected a file path, but got a directory"` | Path points to a directory | Point to individual image files, not directories. |
| `"Image format is unreadable or unsupported"` | Corrupted image or unsupported format | Use PNG or JPEG. Verify the file isn't truncated. |
| `"Expected 3-channel RGB image, got {N}-channel"` | Grayscale, RGBA, or other channel count | Convert to RGB: `img.convert("RGB")` |
| `"Image dimensions must be positive"` | Zero-size image | Check the image file isn't empty or corrupted. |
| `"Failed to extract embedding"` | Backbone failed on valid-looking image | Verify PyTorch is installed correctly. Check image isn't corrupted internally. |

### `CalibrationNotReadyError`

**When**: Calibration hasn't accumulated enough observations.

```
"Calibration window is not ready.
 Need at least {min_samples} observations, got {observed}.
 Continue collecting feedback with correct()."
```

**Fix**: This is expected during early use. Continue calling `predict()` and `correct()`. The `predict()` method handles this internally by falling back to raw confidence. It is safe to ignore for the first ~10-15 predictions.

### `BufferCapacityError`

**When**: UP-UGF pruning fails to enforce capacity.

```
"UP-UGF pruning returned inconsistent buffer lengths"
```
or
```
"UP-UGF pruning did not enforce max_buffer_size"
```

**Fix**: This indicates an internal state inconsistency. Call `load_support_images()` to rebuild the support set. If it persists, reduce `max_buffer_size` or check that all embeddings have the same dimensionality.

### `AdaptShotError` (Base)

| Error Message Pattern | Cause | Fix |
|----------------------|-------|-----|
| `"FewShotLearner is not initialized"` | Called `predict()` or `correct()` before `load_support_images()` | Call `load_support_images()` first. |
| `"Support set is empty"` | Support set was cleared or never loaded | Reload support images. |
| `"Support embeddings must share one dimensionality"` | Mixed backbone outputs (unusual) | Ensure all support images are the same size and format. |
| `"State file not found: '{path}'"` | `load()` called with wrong path | Verify checkpoint path. Use `save()` to create a valid checkpoint. |
| `"Checkpoint integrity check failed"` | Checksums don't match | Restore from a known-good checkpoint. |
| `"Failed to load model head"` | Corrupted `.head.pt` file | Re-save the checkpoint. |
| `"Checkpoint is missing required key"` | JSON from an old or corrupted save | Use a checkpoint created by the current version. |

---

## Common Runtime Issues (No Exceptions)

### Issue: Predictions are always wrong

| Possible Cause | Diagnostic | Fix |
|---------------|------------|-----|
| Support set too small | `len(learner._sim_embeddings) < 5` | Add more support images (at least 3-5 per class). |
| Domain mismatch | Support images look very different from queries | Support and query images should have similar lighting, background, and resolution. |
| Wrong backbone | Using `mobilenet_v3_small` for fine-grained classification | Switch to `resnet18` for higher accuracy. |

### Issue: Calibrated confidence is always very low

| Possible Cause | Diagnostic | Fix |
|---------------|------------|-----|
| Calibration window not full | `report["current_ece"]` near 0 or NaN | Wait for 10-15 predictions. Calibration improves with more observations. |
| High ECE | `report["current_ece"] > 0.15` | Change `calibration_method` to `"scaling_binning"`. Ensure corrections are consistent. |
| Temperature drift | `report["current_temperature"] > 3.0` | Reduce `recalibrate_after_feedback` to `False` to batch updates. |

### Issue: Memory usage grows unbounded

| Possible Cause | Diagnostic | Fix |
|---------------|------------|-----|
| Buffer exceeds capacity | `len(learner._sim_embeddings) > config.max_buffer_size` | Reduce `max_buffer_size`. Ensure UP-UGF pruning is running (check for `BufferCapacityError` warnings). |
| Embedding leaks | RAM grows even without new predictions | Restart the process. Embeddings are held in memory by the learner instance. |
| Backbone cache accumulation | v0.2.0: repeated predictions with different images | Call `learner.clear_backbone_cache()` periodically. |

### Issue: Conformal prediction sets are always size 1

| Possible Cause | Diagnostic | Fix |
|---------------|------------|-----|
| Calibration buffer too small | `summary['calibration_size'] < 10` | Add more calibration data via `correct()`. LOO mode helps with fewer samples. |
| Alpha too strict | `conformal_alpha=0.01` (99% coverage) | Relax to `conformal_alpha=0.10` or `0.20`. |
| Using split mode with tiny buffer | `conformal_mode="split"` with < 50 samples | Switch to `conformal_mode="loo"`. |

### Issue: Contrastive mode produces worse results than prototypical

| Possible Cause | Diagnostic | Fix |
|---------------|------------|-----|
| Too few support examples | < 15 per class | Contrastive mode needs more data; fall back to `prototypical`. |
| Learning rate too high | Loss oscillates | Reduce `ContrastiveConfig.learning_rate` to 0.001. |
| Not enough epochs | Loss still decreasing at epoch 50 | Increase `ContrastiveConfig.n_epochs` to 100. |

### Issue: OOD detection flags too many in-distribution images

| Possible Cause | Diagnostic | Fix |
|---------------|------------|-----|
| Shrinkage too aggressive | Very small support sets (< 5/class) | Add more support examples per class. |
| Percentile too low | `ood_percentile=90.0` | Increase to `ood_percentile=97.0` or `99.0`. |
| Mahalanobis regularization too weak | Covariance near-singular | Increase `mahalanobis_regularization` to 1e-3. |

### Issue: Determinism check fails

| Possible Cause | Diagnostic | Fix |
|---------------|------------|-----|
| Different PyTorch builds | `pip show torch` shows different version | Use identical PyTorch versions across runs. |
| Non-deterministic CUDA ops | Running on GPU with cuDNN non-deterministic | Switch to CPU (`device="cpu"`). |
| Different seed | `config.seed` changed between runs | Fix `seed=42` in both configs. |

---

## Before Reporting a Bug

Run this diagnostic and include the output:

```python
import platform
import torch
import numpy as np
from adaptshot import FewShotLearner
from adaptshot.config.settings import AdaptShotConfig

print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {np.__version__}")
print(f"CPU: {platform.processor()}")

config = AdaptShotConfig(device="cpu", seed=42)
learner = FewShotLearner(config=config)
print(f"Learner: {learner}")
```

Include:
- The exact error message (copy-paste, don't paraphrase)
- A minimal reproducer script (under 30 lines)
- The output of the diagnostic above
- Your operating system and hardware specs

---

## Verification Checklist

- [ ] You can identify each exception by its error message pattern.
- [ ] You know which exception corresponds to which problem category (config, image, calibration, buffer).
- [ ] You can run the diagnostic script and interpret the output.
- [ ] You understand that `CalibrationNotReadyError` is expected in early use and handled internally by `predict()`.
- [ ] You know how to check calibration health via `calibration_report()`.
