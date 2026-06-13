---
title: "06 — Core API Deep Dive"
nav_order: 6
---

This chapter is for readers who already know the basic loop and want a clearer view of the main AdaptShot API. Everything here uses real functions and classes that exist in `src/adaptshot/`.

!!! note
    The main public entry point is `FewShotLearner` in [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py). The top-level package also re-exports it in [src/adaptshot/__init__.py](src/adaptshot/__init__.py).

## 1. Import The Library

The simplest import is:

```python
from adaptshot import FewShotLearner

print(FewShotLearner)
# Expected output: <class 'adaptshot.core.learner.FewShotLearner'>
```

The top-level `adaptshot` package exposes `FewShotLearner`, `AdaptShotConfig`, `CalibrationEngine`, `ACTEngine`, `FeedbackRouter`, `UPUGFPruner`, and the custom exception classes. See [src/adaptshot/__init__.py](src/adaptshot/__init__.py).

## 2. Create A Learner With Real Settings

Think of `AdaptShotConfig` as the control panel. It is a frozen dataclass, which means the settings are locked after creation so the program stays predictable. See [src/adaptshot/config/settings.py](src/adaptshot/config/settings.py).

```python
from adaptshot import AdaptShotConfig, FewShotLearner

config = AdaptShotConfig(
    backbone="resnet18",
    device="cpu",
    seed=42,
    max_buffer_size=10,
    eco_mode=False,
)
learner = FewShotLearner(config=config)
print(learner)
# Expected output: FewShotLearner(initialized=False, support_size=0, classes=0, device='cpu', backbone='resnet18', buffer_capacity=10)
```

`eco_mode` and `early_exit_threshold` are real config fields. `early_exit_threshold` must stay within `[0.5, 1.0]` or `AdaptShotConfig.__post_init__` raises an error. See [src/adaptshot/config/settings.py](src/adaptshot/config/settings.py).

## 3. Load Support Images

The learner does not guess from nothing. It needs a support set: labeled example images. `load_support_images()` validates file paths, checks for RGB images, extracts embeddings, and builds the similarity buffer. See [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).

```python
from pathlib import Path

from PIL import Image

from adaptshot import FewShotLearner

root = Path("deep_dive_demo")
root.mkdir(exist_ok=True)

Image.new("RGB", (32, 32), (0, 180, 0)).save(root / "healthy.png")
Image.new("RGB", (32, 32), (180, 0, 0)).save(root / "unhealthy.png")

learner = FewShotLearner(device="cpu", max_buffer_size=10)
learner.load_support_images(
    [str(root / "healthy.png"), str(root / "unhealthy.png")],
    ["healthy", "unhealthy"],
)
print("support_size=", len(learner._sim_embeddings))
# Expected output: support_size= 2
```

If the support set is empty, the code raises: `Support set cannot be empty. Provide at least one RGB image path and label. See docs/getting-started/quickstart.md.` See [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).

## 4. Predict On A New Image

`predict()` returns a `PredictionResult` dataclass. That object contains the actual label choice plus the confidence and uncertainty fields used throughout the tutorials. See the dataclass in [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).

```python
from PIL import Image

query_path = "deep_dive_demo/query.png"
Image.new("RGB", (32, 32), (170, 20, 20)).save(query_path)

result = learner.predict(query_path)
print(result)
# Expected output: PredictionResult(prediction='healthy' or 'unhealthy', raw_confidence=..., calibrated_confidence=..., neighbor_idx=..., uncertainty_flag=..., act_action=...)
```

The prediction flow is simple:

1. Normalize the input image.
2. Extract an embedding.
3. Search the nearest stored support example.
4. Calibrate confidence.
5. Ask ACT whether to accept or flag for review.

See `predict()` in [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py), `find_nearest_neighbor()` in [src/adaptshot/core/similarity.py](src/adaptshot/core/similarity.py), `CalibrationEngine` in [src/adaptshot/core/calibration.py](src/adaptshot/core/calibration.py), and `ACTEngine` in [src/adaptshot/core/act.py](src/adaptshot/core/act.py).

## 5. Correct A Mistake

`correct()` is the human-in-the-loop hook. It accepts the image path, the true label, and a `confidence_weight` that means how sure the human is, from `0.0` to `1.0`. See [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py) and [src/adaptshot/training/feedback_router.py](src/adaptshot/training/feedback_router.py).

```python
routing = learner.correct(
    image_path=query_path,
    true_label="unhealthy",
    confidence_weight=0.9,
)
print(routing)
# Expected output: a dictionary with keys like buffer_size, pending_corrections, calibration_updated, fine_tuned, total_corrections
```

The router stores the correction in a replay buffer and updates calibration state. If enough corrections accumulate, it may trigger CA-EWC fine-tuning. See `FeedbackRouter.route_feedback()` in [src/adaptshot/training/feedback_router.py](src/adaptshot/training/feedback_router.py) and `CAEWCFinetuner` in [src/adaptshot/training/finetune.py](src/adaptshot/training/finetune.py).

## 6. Save And Load Safely

Persistence is split into a JSON state file, a NumPy embeddings file, and an optional model-head file. That is why `save()` and `load()` are useful for production systems that restart often. See [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).

```python
learner.save("deep_dive_checkpoint.json")
print("saved")

restored = FewShotLearner.load("deep_dive_checkpoint.json")
print(restored)
# Expected output: FewShotLearner(initialized=True, support_size=..., classes=..., device='cpu', backbone='resnet18', buffer_capacity=10)
```

If the JSON file is missing, `load()` raises: `State file not found: '{path}'. Ensure the path is correct before loading.` If the integrity check fails, it raises: `Checkpoint integrity check failed. The checkpoint may be corrupted or tampered with.` See [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).

## 7. What To Read Next

- Use [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py) as the main entry point.
- Use [src/adaptshot/core/extractor.py](src/adaptshot/core/extractor.py) to understand embeddings and preview signatures.
- Use [src/adaptshot/core/similarity.py](src/adaptshot/core/similarity.py) to understand nearest-neighbor search.
- Use [src/adaptshot/training/feedback_router.py](src/adaptshot/training/feedback_router.py) to understand correction routing.
- Use [src/adaptshot/training/finetune.py](src/adaptshot/training/finetune.py) to understand correction-aware fine-tuning.

## 8. Verification Checklist

- [ ] I can import `FewShotLearner` from `adaptshot`.
- [ ] I can create `AdaptShotConfig` with CPU-first settings.
- [ ] I can load support images and print the support size.
- [ ] I can call `predict()` and read the `PredictionResult` fields.
- [ ] I can call `correct()` with a `confidence_weight` and understand the returned routing summary.
- [ ] I can `save()` and `load()` a checkpoint without inventing unsupported behavior.

---

*Created by [Johnson Christopher Hassan](https://github.com/johnson2006christopher)*  
*Connect on [LinkedIn](https://www.linkedin.com/in/johnson-hassan-935124311/)*  
*Project: [github.com/johnson2006christopher/adaptshot](https://github.com/johnson2006christopher/adaptshot)*
