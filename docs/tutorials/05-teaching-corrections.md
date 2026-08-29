# Teaching it when it is wrong, and saving what it learned

> **For:** someone who can read a `PredictionResult` and wants AdaptShot to improve from use — and to keep what it has learned between sessions. Twenty minutes. Offline.

## The loop AdaptShot is built for

Predict. If the result asks for a person (`uncertainty_flag`), a person decides. If the person disagrees with the prediction, tell AdaptShot the right answer. It adjusts. Predict again. That loop is the whole design; this page walks it once.

## Step 1 — A learner, and a photograph it gets wrong

Set up as before, at the 90% level so the sets are calibrated:

```python
from adaptshot import AdaptShotConfig, FewShotLearner
from adaptshot.data import sample_images, demo_images

paths, labels = sample_images()
learner = FewShotLearner(config=AdaptShotConfig(conformal_alpha=0.10))
learner.load_support_images(paths[:-1], labels[:-1])

# The third demo photograph is gray leaf spot that the model tends to call northern leaf blight.
tricky = demo_images()[2]
before = learner.predict(tricky)
print("before:", before.prediction, before.conformal_set, before.act_action)
```

## Step 2 — Correct it

`correct()` takes the photograph and the true label. It is the *only* way new knowledge enters a learner after teaching:

```python
summary = learner.correct(image_path=tricky, true_label="gray_leaf_spot")
print(summary)
```

The summary reports what changed. Three things happen inside:

1. **The photograph joins the teaching set**, labelled correctly, and the prototype for `gray_leaf_spot` moves a little toward it. Next time something similar appears, it is closer to the right average.
2. **The calibration updates.** The confidence scale learns that it was over-confident here; the prediction sets learn what a wrong answer's score looked like.
3. **The per-class acceptance threshold moves.** A class the model keeps getting wrong becomes harder to `ACCEPT` and more likely to ask a person.

`fine_tuned` in the summary is `False` on the standard install — that is the optional deeper adjustment covered in the [fine-tuning how-to](../how-to/fine-tune-with-corrections.md), and it needs the PyTorch extra. Everything above happens without it.

## Step 3 — Ask again

```python
after = learner.predict(tricky)
print("after: ", after.prediction, after.conformal_set, after.act_action)
```

You will usually see the correct label now — the photograph is in the teaching set — and the acceptance behaviour may have changed. One correction is one data point; the effect builds over many.

**Do not correct when you are not sure.** A wrong correction teaches a wrong thing, and the model believes you. `correct()` has a `confidence_weight` argument for when you are less than certain: `learner.correct(path, "gray_leaf_spot", confidence_weight=0.5)` counts half.

## Step 4 — Save it

A learner that has been taught and corrected is worth keeping. Everything it knows goes into two small files:

```python
learner.save("leaves.json")
```

That writes `leaves.json` and `leaves.embeddings.npy` next to it — the numbers for every teaching photograph and every correction, the calibration, the thresholds — with a checksum so a corrupted file is detected rather than loaded. **The photographs themselves are not saved**, only what was computed from them; you can delete or move the originals.

## Step 5 — Load it back

In a new session — a new script, a new day:

```python
from adaptshot import FewShotLearner

restored = FewShotLearner.load("leaves.json")
print(restored.predict(tricky).prediction)
```

The restored learner answers exactly as the saved one did, correction included. Files saved by an older AdaptShot are migrated on load with a warning telling you so; save again to write the current format.

## The whole loop, in one script

```python
from adaptshot import AdaptShotConfig, FewShotLearner
from adaptshot.data import sample_images, demo_images

paths, labels = sample_images()
learner = FewShotLearner(config=AdaptShotConfig(conformal_alpha=0.10))
learner.load_support_images(paths[:-1], labels[:-1])

for photo, truth in zip(demo_images()[:3], ("healthy_maize", "gray_leaf_spot", "gray_leaf_spot"), strict=True):
    result = learner.predict(photo)
    if result.uncertainty_flag or result.prediction != truth:
        # A person looked, and here is what they said.
        learner.correct(image_path=photo, true_label=truth)
        print(f"corrected -> {truth}")
    else:
        print(f"accepted  -> {result.prediction}")

learner.save("leaves.json")
print("saved")
```

The size of the teaching set after each correction is in the summary `correct()` returns (`buffer_size`).

Next: [where to go from here](06-next.md).
