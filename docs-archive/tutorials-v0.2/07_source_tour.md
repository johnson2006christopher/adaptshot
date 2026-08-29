---
title: "07 — Source Tour: How AdaptShot Works"
nav_order: 7
---

This chapter walks through the real modules that make a prediction happen. It is a source tour, not a guess: every section points to code that exists in `src/adaptshot/`.

!!! note
    Use this chapter when you want to understand where a behavior comes from in the codebase. Think of it as a guided reading map for the library.

## 1. The Main Entry Point

The public API starts with `FewShotLearner` in [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py). The package also re-exports it at the top level in [src/adaptshot/__init__.py](src/adaptshot/__init__.py).

```python
from adaptshot import FewShotLearner

print(FewShotLearner.__name__)
# Expected output: FewShotLearner
```

The learner creates and wires these internal pieces:

- `CalibrationEngine` for confidence calibration
- `ACTEngine` for adaptive accept-or-review decisions
- `FeedbackRouter` for correction routing
- `UPUGFPruner` for bounded replay buffer management
- `CAEWCFinetuner` for small correction-aware head updates

See the constructor in [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).

## 2. From Image To Embedding

When you call `predict()`, the learner normalizes the image and calls `extract_embedding()` from [src/adaptshot/core/extractor.py](src/adaptshot/core/extractor.py).

The extractor does three real things:

1. Converts supported image inputs to RGB PIL images.
2. Runs a frozen backbone (`resnet18` or `mobilenet_v3_small`).
3. Returns a NumPy embedding vector.

The available backbones are defined in `BackboneRegistry` in [src/adaptshot/core/extractor.py](src/adaptshot/core/extractor.py).

### Eco Mode Fast Path

If `eco_mode=True`, the extractor may return a cached support embedding early when a preview signature is similar enough. This is implemented in [src/adaptshot/core/extractor.py](src/adaptshot/core/extractor.py) and controlled by `early_exit_threshold` in [src/adaptshot/config/settings.py](src/adaptshot/config/settings.py).

Analogy: before cooking a full meal, you look at a quick photo of the ingredients and decide whether the meal is probably the one you already know.

```python
from adaptshot.config.settings import AdaptShotConfig
from adaptshot.core.extractor import compute_preview_signature

config = AdaptShotConfig(device="cpu", eco_mode=True, early_exit_threshold=0.95)
print(config.eco_mode)
# Expected output: True
```

## 3. From Embedding To Prediction

The learner stores support embeddings in `_sim_embeddings` and support labels in `_sim_labels`. A query embedding is compared with these support vectors using `find_nearest_neighbor()` from [src/adaptshot/core/similarity.py](src/adaptshot/core/similarity.py).

The similarity module has two real paths:

- pure NumPy cosine search
- optional FAISS-CPU search when installed and requested

If FAISS is missing, the code falls back to NumPy. See the ImportError message in [src/adaptshot/core/similarity.py](src/adaptshot/core/similarity.py).

```python
import numpy as np

from adaptshot.core.similarity import find_nearest_neighbor

support_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
support_labels = np.asarray(["leaf", "soil"], dtype=object)
query = np.asarray([0.9, 0.1], dtype=np.float32)

pred_label, confidence, neighbor_idx = find_nearest_neighbor(
    query=query,
    support_embeddings=support_embeddings,
    support_labels=support_labels,
    use_faiss=False,
)
print(pred_label)
print(neighbor_idx)
# Expected output:
# leaf
# 0
```

## 4. Calibration And Confidence

`CalibrationEngine` in [src/adaptshot/core/calibration.py](src/adaptshot/core/calibration.py) turns raw similarity scores into calibrated confidences.

It tracks:

- a sliding window of recent confidences
- which predictions were correct
- ECE history

The calibration method is either `temperature` or `conformal`. The conformal branch is a conservative stub in this release. See `calibrate()` in [src/adaptshot/core/calibration.py](src/adaptshot/core/calibration.py).

Analogy: calibration is like a scale that adjusts itself after repeated weigh-ins until its readings match reality more closely.

```python
from adaptshot.core.calibration import CalibrationEngine

calibrator = CalibrationEngine()
print(calibrator.current_temperature)
# Expected output: 1.0
```

## 5. ACT Decisions

`ACTEngine` in [src/adaptshot/core/act.py](src/adaptshot/core/act.py) decides whether to accept a prediction or request feedback.

The real method is `should_accept(confidence, class_idx, recent_incorrect_rate, recent_correct_rate)`. It returns:

- a boolean accept flag
- an action string, such as `ACCEPT` or `REQUEST_FEEDBACK`

Analogy: a checkpoint guard that decides whether you can pass or should answer a few more questions first.

```python
from adaptshot.core.act import ACTEngine

act = ACTEngine(n_classes=3)
accept, action = act.should_accept(
    confidence=0.8,
    class_idx=0,
    recent_incorrect_rate=0.0,
    recent_correct_rate=1.0,
)
print(accept)
print(action)
# Expected output:
# True
# ACCEPT
```

## 6. Corrections, Routing, And Fine-Tuning

When you call `correct()`, the learner constructs a `Correction` object and passes it to `FeedbackRouter.route_feedback()`.

The router does three real things:

1. Adds the correction to the replay buffer.
2. Updates calibration if a calibrator is attached.
3. Triggers fine-tuning when pending corrections reach the configured threshold.

See [src/adaptshot/training/feedback_router.py](src/adaptshot/training/feedback_router.py).

If a `CAEWCFinetuner` is attached, it can update the model head with correction-aware regularization. See [src/adaptshot/training/finetune.py](src/adaptshot/training/finetune.py).

## 7. Buffer Pruning

The learner keeps the support buffer bounded by `max_buffer_size`. If the buffer grows too large, the code uses `UPUGFPruner` from [src/adaptshot/training/up_ugf.py](src/adaptshot/training/up_ugf.py) to score examples by uncertainty, recency, and redundancy.

Analogy: if a notebook gets too full, keep the most useful pages and archive the least useful ones.

```python
from adaptshot.training.up_ugf import UPUGFPruner

pruner = UPUGFPruner(capacity=2)
print(pruner.capacity)
# Expected output: 2
```

## 8. Persistence And Integrity

`save()` and `load()` in [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py) persist the state using:

- a JSON checkpoint
- an embeddings `.npy` file
- an optional `.head.pt` file

The loader checks integrity metadata and rejects corrupted checkpoints.

This is why a full source tour matters: the file layout is part of the contract.

## 9. Source Tour Checklist

- [ ] I can find the public API in `src/adaptshot/core/learner.py`.
- [ ] I can explain where embeddings come from.
- [ ] I can explain where similarity search happens.
- [ ] I can explain where calibration happens.
- [ ] I can explain where ACT decisions happen.
- [ ] I can explain where corrections are routed.
- [ ] I can explain where pruning and persistence happen.

---

*Created by [Johnson Christopher Hassan](https://github.com/johnson2006christopher)*  
*Connect on [LinkedIn](https://www.linkedin.com/in/johnson-hassan-935124311/)*  
*Project: [github.com/johnson2006christopher/adaptshot](https://github.com/johnson2006christopher/adaptshot)*
