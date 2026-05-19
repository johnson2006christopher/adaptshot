---
title: "02 — Human-in-the-Loop: Correcting Predictions"
nav_order: 2
---

This tutorial continues from the Getting Started guide and uses a concrete scenario: monitoring crop health on a small farm with limited labeled photos.

!!! note
    Assume you completed the steps in `docs/tutorials/01_getting_started.md` and can run the small example script there.

**1. The Problem with Blind Predictions**

If a model makes predictions but never hears when it is wrong, mistakes accumulate. In farming, a wrong label (e.g., "healthy") might cause missed disease treatment — real harm. That is why human corrections matter: they teach the system to be more reliable.

**2. Understanding AdaptShot's Output**

AdaptShot returns a `PredictionResult` object with these fields (see `src/adaptshot/core/learner.py`):

- `prediction` — the label chosen (string or int). See `PredictionResult` in [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).
- `raw_confidence` — an uncalibrated similarity score (internal). [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py)
- `calibrated_confidence` — a human-friendly probability-like score after calibration. Analogy: calibration is like a scale that learns to weigh accurately over time — the scale becomes trustworthy after a few checks. See calibration code at [src/adaptshot/core/calibration.py](src/adaptshot/core/calibration.py) and how it is used in `predict()` at [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).
- `neighbor_idx` — which support example was the nearest neighbor (index into the support list). See `find_nearest_neighbor` usage in `predict()` at [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py) and implementation in [src/adaptshot/core/similarity.py](src/adaptshot/core/similarity.py).
- `uncertainty_flag` — a boolean indicating the ACT engine suggests caution. The ACT engine decides whether to accept or ask for help; see [src/adaptshot/core/act.py](src/adaptshot/core/act.py) and `act.should_accept()` call in `predict()` at [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).
- `act_action` — short string instruction from the ACT engine (e.g., `accept` or `query`). See `ACTEngine` in [src/adaptshot/core/act.py](src/adaptshot/core/act.py).

**Example interpretation**

- High `calibrated_confidence` (close to 1.0) and `uncertainty_flag == False` → likely safe to accept.
- Low `calibrated_confidence` or `uncertainty_flag == True` → consider asking a human to correct.

**3. When to Trust, When to Correct (Decision tree)**

- If `calibrated_confidence >= 0.8` and `uncertainty_flag == False`: accept prediction.
- If `calibrated_confidence < 0.8` or `uncertainty_flag == True`: ask a human to verify and, if wrong, call `learner.correct()`.

!!! tip
    Thresholds like `0.8` are for illustration. AdaptShot's ACT engine internally uses class thresholds and recent reliability to gate decisions. See `ACTEngine` in [src/adaptshot/core/act.py](src/adaptshot/core/act.py).

**4. Routing a Correction (code you can run)**

This runnable script demonstrates the full loop: Predict → Interpret → Correct → Observe buffer update. Save as `human_loop.py` at the repo root and run from the repository root. The script uses small images created with Pillow and prints expected outputs as comments.

```python
import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from adaptshot.core.learner import FewShotLearner
from PIL import Image

# Create support images for 'healthy' and 'unhealthy'
paths=[(Image.new("RGB",(32,32),(0,200,0)).save(p:=("farm_support_healthy.png")) or p),(Image.new("RGB",(32,32),(200,0,0)).save(q:=("farm_support_unhealthy.png")) or q)]
labels=["healthy","unhealthy"]

learner=FewShotLearner()
learner.load_support_images(paths, labels)

# New photo from the field we want a prediction for
Image.new("RGB",(32,32),(190,10,10)).save("query.png")
result=learner.predict("query.png")
print(result)
# Expected output (example): PredictionResult(prediction='unhealthy', raw_confidence=0.12, calibrated_confidence=0.45, neighbor_idx=1, uncertainty_flag=True, act_action='query')

# Decision: interpret fields and decide to correct if flagged or low confidence
if result.uncertainty_flag or result.calibrated_confidence < 0.8:
    # Human inspects photo and tells us the true label. confidence_weight is how sure the human is (0.0 to 1.0).
    routing = learner.correct("query.png", true_label="unhealthy", confidence_weight=0.9)
    print(routing)
    # Expected output: a dict with routing summary, possibly fine-tune trigger info and buffer routing details

# Observe buffer change: support buffer is extended with the correction
print("support_size=", len(learner._sim_embeddings))
# Expected output: support_size= 3  (two originals + one correction appended)

# Re-run prediction after correction to see local change
post = learner.predict("query.png")
print(post)
# Expected output: PredictionResult with possibly higher calibrated_confidence and neighbor_idx pointing to the appended correction (values will vary)
```

**Code notes and safety**

- We use `confidence_weight=0.9` to tell the system the human is 90% sure the correction is correct. In code this maps to a float accepted by `learner.correct()` and documented in `src/adaptshot/core/learner.py`.
- `learner.correct()` returns a routing summary dictionary from the feedback router; see `FeedbackRouter` in `src/adaptshot/training/feedback_router.py` and the `correct()` method implementation in `src/adaptshot/core/learner.py`.

**5. What Happens Behind the Scenes (plain language)**

- When you call `correct()`, AdaptShot:
  - records the human correction into a small buffer (it appends a new support example) so future predictions can use it;
  - routes the correction through the feedback router which may schedule a lightweight fine-tune or update internal calibration statistics;
  - updates uncertainty bookkeeping and may adjust ACT thresholds so the model asks for help less often for that class in the future.

Analogy: Think of calibration as a scale that, after seeing several known weights and corrections, learns to show the right value more often. Corrections are the known weights you place on the scale to teach it.

!!! note
    The implementation details live in `src/adaptshot/core/learner.py`, `src/adaptshot/training/feedback_router.py`, `src/adaptshot/core/calibration.py`, and `src/adaptshot/core/act.py`. For precise code paths, consult those files. [TODO: Verify against learner.py]

**6. Expected Output at Each Step**

- After `predict()`: a `PredictionResult` printed with fields `prediction`, `raw_confidence`, `calibrated_confidence`, `neighbor_idx`, `uncertainty_flag`, `act_action`. See `PredictionResult` in [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).
- After `correct()`: a routing dictionary printed showing how the correction was routed and whether fine-tuning was triggered. See `FeedbackRouter` in [src/adaptshot/training/feedback_router.py](src/adaptshot/training/feedback_router.py) and `correct()` in [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).
- After observing the buffer: `len(learner._sim_embeddings)` increases by one (the corrected example was appended). This is visible via internal state; the buffer append logic is in `_append_correction_to_similarity_buffer()` in [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).

**7. Verification Checklist**

- [ ] Run `human_loop.py` from the repository root and see a printed `PredictionResult` for the first `predict()` call.
- [ ] If the prediction is flagged, call `learner.correct()` and see a printed routing dictionary.
- [ ] Confirm the buffer size increased by one (`support_size` printed).
- [ ] Re-run `predict()` and observe whether the local prediction or confidence changed.

If anything looks unexpected, inspect the implementation at `src/adaptshot/core/learner.py` (predict, correct, _append_correction_to_similarity_buffer), `src/adaptshot/core/calibration.py`, and `src/adaptshot/core/act.py` to trace behavior.
