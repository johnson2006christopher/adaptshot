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

## MziziGuard-Specific Troubleshooting

### App Won't Start

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: No module named 'gradio'` | UI dependencies missing | `pip install "adaptshot[ui]"` |
| `ModuleNotFoundError: No module named 'yaml'` | PyYAML missing | `pip install PyYAML` |
| `Address already in use` | Port 7860 occupied | Use `--port 8080` |
| `config.yaml not found` | Config file missing | Ensure `examples/mziziguard/config.yaml` exists |

### Model Training Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| All predictions are same class | Too few support images | Increase `n_support` to 5–10 |
| Confidence always 100% | Calibration not warmed up | Make 10+ predictions with corrections |
| "Model not trained" in Diagnose tab | Skipped Setup tab | Go to Setup → Generate Samples or Load from Folder |
| Folder loading finds 0 images | Wrong folder structure | Use `class_name/*.png` structure |

### OOD Detection Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| OOD never triggers | `enable_ood_detection: false` | Set to `true` in config.yaml |
| Everything flagged OOD | Threshold too aggressive | Increase `ood_threshold_quantile` (e.g., 0.99) |
| Non-crop images not caught | Threshold too permissive | Decrease `ood_absolute_min_distance` |

### Performance Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Slow inference (>500ms) | Heavy backbone + weak CPU | Switch to `mobilenet_v3_small` |
| High RAM usage (>300MB) | Buffer too large | Reduce `max_buffer_size` to 50 |
| Battery drain on laptop | Eco mode disabled | Set `eco_mode: true` in config |
| First prediction slow | Backbone downloading | Normal — cached after first use |

## Quick Diagnostic for MziziGuard

```bash
# MziziGuard-specific health check
python -c "
from examples.mziziguard import MziziGuard

guard = MziziGuard()
print(f'Config loaded: {len(guard.known_labels)} diseases')

guard.initialize_with_samples(n_support=3)
print(f'Trained: {guard.is_trained}')

result = guard.diagnose(guard._data_dir + '/healthy_maize_support_00.png')
print(f'Diagnosis works: {result.swahili}')

health = guard.system_health()
print(f'Health report: ECE={health[\"calibration\"][\"ece\"]}')
print('✅ MziziGuard is healthy')
"
```

---

*Created by [Johnson Christopher Hassan](https://github.com/johnson2006christopher)*  
*Connect on [LinkedIn](https://www.linkedin.com/in/johnson-hassan-935124311/)*  
*Project: [github.com/johnson2006christopher/adaptshot](https://github.com/johnson2006christopher/adaptshot)*
