---
title: "10 — Module Map: Where Each Feature Lives"
nav_order: 10
---

This chapter is a guided map of the AdaptShot source tree. It tells readers which file owns which responsibility, so they can move from tutorial knowledge to code navigation without guessing.

!!! note
    Think of this as a library shelf map. Each book lives in one place, and each place has a clear job.

## 1. The Public Entry Point

The top-level package is [src/adaptshot/__init__.py](src/adaptshot/__init__.py). It re-exports the main classes and exceptions for convenient imports.

```python
from adaptshot import FewShotLearner, AdaptShotConfig

print(FewShotLearner.__name__)
print(AdaptShotConfig.__name__)
# Expected output:
# FewShotLearner
# AdaptShotConfig
```

The main public objects are:

- `FewShotLearner`
- `AdaptShotConfig`
- `CalibrationEngine`
- `ACTEngine`
- `FeedbackRouter`
- `UPUGFPruner`
- `AdaptShotError` and the custom subclasses

## 2. Core Modules

| File | Responsibility | Key Things To Read |
|---|---|---|
| [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py) | Main inference, correction, persistence, and state orchestration | `FewShotLearner`, `PredictionResult`, `save()`, `load()`, `predict()`, `correct()` |
| [src/adaptshot/core/extractor.py](src/adaptshot/core/extractor.py) | Convert images to embeddings; optional eco-mode fast path | `extract_embedding()`, `compute_preview_signature()`, `set_support_embedding_cache()` |
| [src/adaptshot/core/similarity.py](src/adaptshot/core/similarity.py) | Find the closest support example | `find_nearest_neighbor()`, `cosine_similarity_numpy()`, `cosine_similarity_faiss()` |
| [src/adaptshot/core/calibration.py](src/adaptshot/core/calibration.py) | Track and recalibrate confidence values | `CalibrationEngine`, `update()`, `calibrate()`, `compute_ece()` |
| [src/adaptshot/core/act.py](src/adaptshot/core/act.py) | Decide whether to accept or request review | `ACTEngine`, `should_accept()`, `get_threshold()` |

Analogy: the core modules are the main kitchen tools — knife, stove, scale, timer, and serving tray.

## 3. Training Modules

| File | Responsibility | Key Things To Read |
|---|---|---|
| [src/adaptshot/training/feedback_router.py](src/adaptshot/training/feedback_router.py) | Store corrections, update calibration, trigger fine-tuning | `Correction`, `FeedbackRouter`, `route_feedback()` |
| [src/adaptshot/training/finetune.py](src/adaptshot/training/finetune.py) | Correction-aware head-only fine-tuning | `CAEWCFinetuner`, `update_fisher()`, `finetune()` |
| [src/adaptshot/training/up_ugf.py](src/adaptshot/training/up_ugf.py) | Score and prune the replay buffer | `UPUGFPruner`, `compute_scores()`, `prune()` |

These modules are where human feedback becomes model improvement.

## 4. Utility Modules

| File | Responsibility | Key Things To Read |
|---|---|---|
| [src/adaptshot/config/settings.py](src/adaptshot/config/settings.py) | Immutable configuration and validation | `AdaptShotConfig`, `__post_init__()` |
| [src/adaptshot/utils/determinism.py](src/adaptshot/utils/determinism.py) | Reproducibility helpers | `set_deterministic_seed()`, `verify_determinism()` |
| [src/adaptshot/utils/io.py](src/adaptshot/utils/io.py) | Path validation, JSON I/O, tensor conversion | `validate_path()`, `save_json()`, `load_json()`, `tensor_to_numpy()` |
| [src/adaptshot/utils/exceptions.py](src/adaptshot/utils/exceptions.py) | Library-specific exception types | `AdaptShotError`, `InvalidImageError`, `ConfigValidationError`, `CalibrationNotReadyError`, `BufferCapacityError` |
| [src/adaptshot/utils/migrations.py](src/adaptshot/utils/migrations.py) | Checkpoint migration helpers | `migrate_v0_1_0_to_v0_1_1()` |

Analogy: these are the measuring tape, label maker, and filing cabinet of the library.

## 5. What Happens During A Prediction

The path is real and visible in code:

1. `FewShotLearner.predict()` normalizes the input image.
2. `extract_embedding()` creates an embedding.
3. `find_nearest_neighbor()` compares it to stored support embeddings.
4. `CalibrationEngine.calibrate()` adjusts the raw confidence.
5. `ACTEngine.should_accept()` decides whether to accept or ask for help.

You can read each step in:

- [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py)
- [src/adaptshot/core/extractor.py](src/adaptshot/core/extractor.py)
- [src/adaptshot/core/similarity.py](src/adaptshot/core/similarity.py)
- [src/adaptshot/core/calibration.py](src/adaptshot/core/calibration.py)
- [src/adaptshot/core/act.py](src/adaptshot/core/act.py)

## 6. What Happens During A Correction

The correction path is also explicit:

1. `FewShotLearner.correct()` validates the new label and image.
2. `FeedbackRouter.route_feedback()` adds the correction to the buffer.
3. `CalibrationEngine.update()` learns from the new feedback.
4. `CAEWCFinetuner.finetune()` may run when enough corrections accumulate.
5. `UPUGFPruner` keeps the buffer bounded.

Analogy: the model hears the correction, writes it down, updates its confidence rules, and then cleans up the notebook if it becomes too full.

## 7. Persistence Path

`save()` and `load()` in [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py) are the checkpoint entry points.

The code stores:

- JSON state
- embeddings in `.npy`
- optional model head weights in `.head.pt`

The migration helper in [src/adaptshot/utils/migrations.py](src/adaptshot/utils/migrations.py) supports older checkpoint format upgrades.

## 8. How The Benchmark Scripts Use The Modules

The benchmark scripts are not separate magic systems. They call the same modules listed above.

| Script | Modules It Uses |
|---|---|
| [benchmarks/run_benchmark.py](benchmarks/run_benchmark.py) | config, extractor, similarity, determinism |
| [benchmarks/energy_profile.py](benchmarks/energy_profile.py) | config, extractor, similarity, determinism |
| [benchmarks/day2_integration.py](benchmarks/day2_integration.py) | calibration, feedback router |
| [benchmarks/day3_integration.py](benchmarks/day3_integration.py) | ACT, calibration, feedback router, CA-EWC |

## 9. A Simple Navigation Habit

When you wonder “where does this happen?”, ask three questions:

1. Is it prediction, correction, or persistence?
2. Is it core logic, training logic, or utility logic?
3. Which file in the module map owns that responsibility?

That habit helps you avoid guessing and keeps documentation honest.

## 10. Verification Checklist

- [ ] I know the top-level import path for the public API.
- [ ] I can name the core modules and their roles.
- [ ] I can name the training modules and their roles.
- [ ] I can name the utility modules and their roles.
- [ ] I know which files handle prediction, correction, persistence, and benchmarking.

---

*Created by [Johnson Christopher Hassan](https://github.com/johnson2006christopher)*  
*Connect on [LinkedIn](https://www.linkedin.com/in/johnson-hassan-935124311/)*  
*Project: [github.com/johnson2006christopher/adaptshot](https://github.com/johnson2006christopher/adaptshot)*
