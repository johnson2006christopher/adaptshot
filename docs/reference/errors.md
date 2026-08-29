# Errors and warnings

> **For:** someone reading a traceback or a warning and wanting its exact meaning. For symptom-first help, see [troubleshoot](../how-to/troubleshoot.md).

## The hierarchy

Everything AdaptShot raises on purpose derives from `AdaptShotError`, so `except AdaptShotError` catches the library and nothing else. All are in `adaptshot.utils.exceptions` and exported from `adaptshot`.

| exception | raised when | typical message |
|---|---|---|
| `AdaptShotError` | base class; also raised directly for embedding failures and unreadable checkpoints | `Failed to read checkpoint JSON at '…'. The file may be corrupted.` |
| `InvalidImageError` | an image path is missing, the file is unreadable, or it cannot be converted to RGB | `Image not found: …` |
| `ConfigValidationError` | a configuration value is outside its supported range, or inputs to `load_support_images` are malformed | `image_paths and labels must have the same length. Got 12 image_paths and 11 labels.` |
| `BackboneError` | no usable backend exists for the requested backbone on this install — its ONNX weights are not bundled and torch is absent | `Backbone 'resnet18' needs PyTorch on this install: its ONNX weights are not bundled. Either use one of the bundled backbones (mobilenet_v3_small), or install torch with pip install 'adaptshot[training]'.` |
| `CalibrationNotReadyError` | calibration is asked for a verdict before it has enough observations; rarely raised, since the learner falls back gracefully | — |
| `BufferCapacityError` | pruning the correction buffer failed; a deterministic FIFO fallback was applied and the failure is reported rather than hidden | `UP-UGF pruning failed. Applied deterministic FIFO fallback to enforce capacity 100. Error: …` |

A `ValueError` — not an `AdaptShotError` — means a plain programming mistake: an unknown backbone name, a distance matrix of the wrong rank. The distinction is deliberate: a typo in a name is a different kind of failure from a missing backend.

## Warnings

| warning | class | meaning |
|---|---|---|
| `ConformalEngine(alpha=…): prediction sets are uninformative -- every class -- until N calibration scores exist` | `logging.WARNING` on `adaptshot.core.conformal` | the calibration floor is below ⌈(1−α)/α⌉; sets are the full label set until then. [Choose α](../how-to/choose-alpha.md) |
| `OOD detection disabled: no class has enough samples to hold one out for calibration` | `logging.WARNING` on `adaptshot.core.uncertainty` | every class has fewer than three teaching photographs; the flag stays off rather than firing on everything |
| `Checkpoint schema 0.1.0 loaded; migrating to 0.2.0.` | `RuntimeWarning` (points at the caller of `load`) | an older saved learner was upgraded in memory; save again to persist the new format |
| `CUDA requested but not available. Runtime logic will fall back to CPU.` | `RuntimeWarning` | `device="cuda"` with no usable GPU |
| `CUDA requested but PyTorch is not installed.` | `RuntimeWarning` | `device="cuda"` on the core install |
| `adaptshot.core.contrastive moved to adaptshot.training.contrastive in 0.3.0; this alias will be removed in 0.4.0.` | `DeprecationWarning` | update the import |
| `UncertaintyQuantifier.<method>() is deprecated as of 0.3.0 and will be removed in 0.4.0` | `DeprecationWarning` | `compute_perturbation_variance`, `get_ood_summary`, `get_class_statistics` had no callers; stop using them |

Library code logs through `logging.getLogger(__name__)` and never prints. To see the warnings above in a script, configure logging once: `logging.basicConfig(level=logging.WARNING)`.

## Messages that name the fix

Since 0.3.0 an error that has a remedy names it in the message — the backbone that would work, the extra that installs torch, the number of photographs a level of α needs. If you hit one that does not, that is a documentation bug worth an issue.
