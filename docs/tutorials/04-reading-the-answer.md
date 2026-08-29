# Reading the answer

> **For:** someone who has made a prediction and wants to know what every part of the result means — and what to *do* about each. Twenty minutes. Offline.

## The answer is not one thing

`predict()` returns a `PredictionResult`. Its fields fall into four groups, and each group answers a different question:

| question | fields |
|---|---|
| What does it think? | `prediction`, `raw_confidence`, `calibrated_confidence` |
| What is it prepared to stand behind? | `conformal_set`, `conformal_calibrated` |
| Should a person look? | `uncertainty_flag`, `act_action`, `ood_flag` |
| Why? | `nearest_neighbors`, `distance_to_prototype`, `prototype_margin`, `uncertainty_report` |

Set up the same learner as before, and also fetch the four *demo* photographs that ship alongside the twelve — three more maize leaves it has not seen, and one healthy **tomato** leaf, a crop it was never taught:

```python
from adaptshot import FewShotLearner
from adaptshot.data import sample_images, demo_images

paths, labels = sample_images()
learner = FewShotLearner()
learner.load_support_images(paths[:-1], labels[:-1])

maize_leaf, tomato_leaf = paths[-1], demo_images()[-1]
```

## Group 1 — what it thinks

```python
result = learner.predict(maize_leaf)
print(result.prediction)
print(f"raw {result.raw_confidence:.2f}   calibrated {result.calibrated_confidence:.2f}")
```

`prediction` is the label of the nearest prototype. There are two confidences because raw similarity is not a probability: a raw score of 0.9 does not mean "right 90% of the time". `calibrated_confidence` has been adjusted so that, over many predictions, a calibrated 0.7 is right about 70% of the time. **Use the calibrated one.** The raw one is there so you can see what the adjustment did.

## Group 2 — what it stands behind: the prediction set

This is the part that makes AdaptShot different, and it comes with a wrinkle you should see rather than be told about.

```python
print(result.conformal_set, result.conformal_calibrated)
```

With eleven teaching photographs and the default settings, this prints something like `['northern_leaf_blight'] False`. The set contains only the top guess, and `conformal_calibrated` is **False** — meaning *no promise applies to this set yet*.

Why: the promise is that the true answer is inside the set at least `1 − α` of the time, and the default `α` is 0.05 (a 95% promise). To keep a 95% promise, the maths needs at least 19 teaching photographs to measure itself against. With eleven it cannot, so it tells you instead of pretending. Two ways forward: teach with more photographs, or ask for a promise it *can* keep with eleven — 90%:

```python
from adaptshot import AdaptShotConfig

learner90 = FewShotLearner(config=AdaptShotConfig(conformal_alpha=0.10))
learner90.load_support_images(paths[:-1], labels[:-1])
result = learner90.predict(maize_leaf)
print(result.conformal_set, result.conformal_calibrated)
```

Now `conformal_calibrated` is **True**, and the set may hold one label or several. Read it like this:

- **One label** — confident; act on it.
- **Two or more** — it will not choose between them on this evidence. Often that is still useful: if every label in the set is a disease, the advice to a farmer is the same whichever it is.
- **Every label** — it has effectively said "I cannot narrow this down." Treat it as a refusal.

The number to remember: at level `α`, AdaptShot needs `⌈(1 − α) / α⌉` teaching photographs before its sets mean anything — **9 for a 90% promise, 19 for 95%**. It warns you once, at start-up, if you ask for more than it can deliver.

## Group 3 — should a person look?

```python
for leaf, name in ((maize_leaf, "maize leaf"), (tomato_leaf, "tomato leaf")):
    r = learner90.predict(leaf)
    print(f"{name:<12} action={r.act_action:<21} uncertain={r.uncertainty_flag!s:<5} ood={r.ood_flag}")
```

- `act_action` is `ACCEPT` when the calibrated confidence clears a per-class threshold that adapts as you correct it, and `REQUEST_FEEDBACK` when it does not. That is the "ask a human" signal. For the tomato leaf — a crop it was never taught — it should ask.
- `ood_flag` is a second, independent detector: does this photograph look like *nothing* it was taught? It is calibrated from the teaching photographs themselves. Whether it fires on the tomato leaf depends on how different the photographs are; the confidence gate is the more reliable of the two with few examples.
- `uncertainty_flag` is simply "either of the above". **If it is True, route the photograph to a person.** That single rule is the safe way to use AdaptShot.

## Group 4 — why

```python
r = learner90.predict(maize_leaf)
print(f"distance to its prototype {r.distance_to_prototype:.3f}, margin over the runner-up {r.prototype_margin:.3f}")
for neighbour in r.nearest_neighbors[:3]:
    print("  looked like:", neighbour)
print(r.uncertainty_report)
```

`distance_to_prototype` is how far the photograph sat from the winning average; `prototype_margin` is how much closer it was to the winner than to the runner-up — a small margin means a close call. `nearest_neighbors` names the teaching photographs it most resembled, which is often the fastest way to spot a mislabelled example. `uncertainty_report` breaks the doubt into three kinds — *epistemic* (would a slightly different photograph change the answer?), *aleatoric* (are the teaching photographs themselves ambiguous here?), *distributional* (how far from everything?) — and a `composite` of the three.

## The rule of thumb

1. `uncertainty_flag` True → a person decides.
2. Otherwise, if the set has one label → act on it.
3. Otherwise, act only if every label in the set calls for the same action.

Next: [teaching it when it is wrong](05-teaching-corrections.md), and saving what it has learned.
