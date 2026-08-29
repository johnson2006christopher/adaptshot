---
title: "03 — Continual Learning: Keep Improving Over Time"
nav_order: 3
---

This tutorial assumes you completed tutorials 01 and 02. We will use a simple farm scenario: a small team monitors crop health with a few labeled photos, then improves the model as new expert corrections arrive.

!!! note
    Continual learning means the model keeps learning from new examples after the first setup. Think of it like a student who keeps improving after each quiz review.

## 1. What Is Continual Learning?

Continual learning is the practice of updating a model in small steps instead of starting from zero every time. Analogy: instead of re-teaching a friend all plant names from scratch, you only correct the few mistakes they make today.

In AdaptShot, continual learning happens through three parts in the codebase:

- `FewShotLearner.correct()` routes a human correction into the learning pipeline. See [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).
- `FeedbackRouter` stores corrections, updates calibration, and can trigger fine-tuning. See [src/adaptshot/training/feedback_router.py](src/adaptshot/training/feedback_router.py).
- `CAEWCFinetuner` updates the classification head with a light continual-learning penalty. See [src/adaptshot/training/finetune.py](src/adaptshot/training/finetune.py).

!!! warning
    AdaptShot is CPU-first and memory-bound. Continual learning here is designed for small correction batches, not large-scale retraining.

## 2. Key Terms in Plain English

| Term | Meaning | Analogy |
|---|---|---|
| Continual learning | Updating a model little by little as new examples arrive | A student improving after each homework review |
| Correction | A human-provided true label for a mistaken prediction | A teacher marking the right answer on a quiz |
| `confidence_weight` | How sure the human is about the correction, from `0.0` to `1.0` | A witness saying “I’m barely sure” vs. “I’m completely sure” |
| Replay buffer | A small memory of recent corrections kept for later learning | A notebook of recent mistakes |
| Calibration | Adjusting confidence scores so they better match reality | A scale that learns to weigh accurately over time |
| Catastrophic forgetting | Forgetting earlier knowledge when learning new things | A student forgetting old vocabulary after cramming for one test |
| Fine-tuning | Small updates to a model instead of rebuilding it | Tightening the bolts on a machine instead of rebuilding the machine |
| CA-EWC | A fine-tuning method that tries to protect older knowledge while learning new corrections | Learning a new route without forgetting your way home |
| UP-UGF | Buffer pruning that keeps useful examples and removes less useful ones | Cleaning a toolbox and keeping the tools you reach for most |

## 3. Read → Type → Run → See Output → Understand Why

The script below shows the full continual-learning loop:

1. Load a tiny support set.
2. Predict on a new image.
3. Correct the mistake with `correct()`.
4. Save the updated learner.
5. Reload the checkpoint to confirm the state persisted.

Create a file named `continual_learning_demo.py` at the repository root and paste the script below.

```python
import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from adaptshot.core.learner import FewShotLearner
from PIL import Image

def make_image(path, color):
    Image.new("RGB", (32, 32), color).save(path)

make_image("support_healthy.png", (0, 180, 0))
make_image("support_unhealthy.png", (180, 0, 0))
make_image("field_query.png", (170, 20, 20))

learner = FewShotLearner()
learner.load_support_images(["support_healthy.png", "support_unhealthy.png"], ["healthy", "unhealthy"])
result = learner.predict("field_query.png")
print(result)
# Expected output: PredictionResult(prediction='unhealthy' or 'healthy', raw_confidence=..., calibrated_confidence=..., neighbor_idx=..., uncertainty_flag=..., act_action=...)

routing = learner.correct("field_query.png", true_label="unhealthy", confidence_weight=0.9)
print(routing)
# Expected output: a dict with keys such as buffer_size, pending_corrections, calibration_updated, fine_tuned, total_corrections

print("buffer_size_after_correct=", len(learner._sim_embeddings))
# Expected output: buffer_size_after_correct= 3

learner.save("continual_checkpoint.json")
print("saved_checkpoint= continual_checkpoint.json")
# Expected output: saved checkpoint files exist: continual_checkpoint.json, continual_checkpoint.embeddings.npy, and possibly continual_checkpoint.head.pt

reloaded = FewShotLearner.load("continual_checkpoint.json")
print("reloaded_support_size=", len(reloaded._sim_embeddings))
# Expected output: reloaded_support_size= 3
```

## 4. Line-by-Line Explanation

- `import os, sys` — gives the script access to basic path tools.
- `sys.path.insert(0, os.path.join(os.getcwd(), "src"))` — makes the `src/` package importable when running from the repository root.
- `from adaptshot.core.learner import FewShotLearner` — imports the main API class. See [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).
- `from PIL import Image` — creates small RGB images for the demo.
- `make_image(...)` — helper that writes a tiny image file to disk.
- `learner = FewShotLearner()` — creates the learner with default CPU-first settings.
- `load_support_images(...)` — loads the first labeled examples and builds the similarity index.
- `predict(...)` — produces a `PredictionResult` with `prediction`, `raw_confidence`, `calibrated_confidence`, `neighbor_idx`, `uncertainty_flag`, and `act_action`. See the dataclass and `predict()` in [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).
- `correct(...)` — routes a human correction into the feedback pipeline. The `confidence_weight` argument means how sure the human is about the correction. See `correct()` in [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py) and `Correction` in [src/adaptshot/training/feedback_router.py](src/adaptshot/training/feedback_router.py).
- `save(...)` — writes the learner state to disk. The checkpoint format includes JSON state plus a NumPy embeddings file, and optionally a model-head file. See `save()` in [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).
- `load(...)` — restores the saved learner state. See `load()` in [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).

## 5. What Changes After `correct()`?

When you call `correct()` in AdaptShot, three things happen in the code path:

1. The correction is added to a small replay buffer so the learner remembers it. See `FeedbackRouter._update_buffer()` in [src/adaptshot/training/feedback_router.py](src/adaptshot/training/feedback_router.py).
2. The calibrator is updated with the raw confidence and the predicted vs. true label. See `FeedbackRouter.route_feedback()` in [src/adaptshot/training/feedback_router.py](src/adaptshot/training/feedback_router.py).
3. If enough corrections accumulate, the CA-EWC fine-tuner can run a short head-only update. The threshold is controlled by `fine_tune_trigger_threshold=max(5, max_buffer_size // 10)` in [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).

Analogy: first you write down the mistake, then you update the grading rules, and only then do you briefly retrain the student on the new lesson.

!!! note
    `correct()` does not rebuild the whole system from scratch. It adds targeted updates so the model stays lightweight and CPU-friendly.

## 6. When Does Fine-Tuning Trigger?

The learner creates a `FeedbackRouter` with a fine-tune trigger threshold based on buffer size. See the constructor in [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).

| Situation | What happens |
|---|---|
| Few corrections | They are buffered and used for calibration updates |
| Enough corrections accumulated | The router can call the fine-tuning callback |
| No finetuner available | The router still buffers and calibrates, but does not fine-tune |

!!! tip
    If you want to observe a fine-tune trigger in a local demo, provide several corrections in a row. The exact trigger depends on `max_buffer_size`.

## 7. Save, Load, and Keep Going Tomorrow

Continual learning is useful only if you can resume safely after restart. That is why `save()` and `load()` matter.

The learner persists:

- configuration
- calibration state
- ACT thresholds
- replay buffer metadata
- label mapping
- integrity metadata

See the serialization logic in [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).

!!! warning
    If the checkpoint or embeddings file is missing or corrupted, `load()` raises a clear error such as `State file not found: '{path}'. Ensure the path is correct before loading.` or `Checkpoint integrity check failed. The checkpoint may be corrupted or tampered with.`

## 8. Common Questions

| Question | Short answer |
|---|---|
| Do I need a GPU? | No. AdaptShot v0.1.1 is CPU-first. See `src/adaptshot/config/settings.py`. |
| Does one correction immediately change the whole model? | No. It updates the buffer and calibration first, then may fine-tune later. |
| Why keep a replay buffer? | It helps the learner remember recent useful corrections instead of forgetting them. |
| Why save after corrections? | So the improved state is not lost if the process restarts. |

## 9. Verification Checklist

- [ ] I can explain what continual learning means in plain English.
- [ ] I can run the demo script and see a `PredictionResult` printed.
- [ ] I understand what `confidence_weight` means and why it ranges from `0.0` to `1.0`.
- [ ] I know where `correct()`, `save()`, and `load()` live in the source code.
- [ ] I can describe how corrections flow into the buffer, calibration, and optional fine-tuning.

If you want the exact implementation details, start with [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py), then follow the links to [src/adaptshot/training/feedback_router.py](src/adaptshot/training/feedback_router.py) and [src/adaptshot/training/finetune.py](src/adaptshot/training/finetune.py).

---

*Created by [Johnson Christopher Hassan](https://github.com/johnson2006christopher)*  
*Connect on [LinkedIn](https://www.linkedin.com/in/johnson-hassan-935124311/)*  
*Project: [github.com/johnson2006christopher/adaptshot](https://github.com/johnson2006christopher/adaptshot)*
