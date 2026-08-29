# Choose the promise level (α), and know how many photographs it needs

> **For:** someone who understands what a prediction set is ([tutorial 4](../tutorials/04-reading-the-answer.md)) and needs to pick `conformal_alpha` for a real use. Five minutes.

## What α means

`conformal_alpha` is the fraction of the time the prediction set is *allowed* to miss the true label. `α = 0.10` promises the true label is in the set at least 90% of the time; `α = 0.05`, 95%. Smaller α is a stronger promise and buys it with bigger sets.

The promise is *marginal*: over many photographs, on average. It is not per-class, and it holds only while the photographs you ask about resemble the ones you taught with — see [the guarantee](../understand/the-guarantee.md).

## The rule that decides it

At level α, AdaptShot needs **⌈(1 − α) / α⌉ teaching photographs** before any set can mean anything. Below that, no finite threshold exists and the only honest set is every class — so the learner takes the cold-start path and reports `conformal_calibrated = False` instead.

| α | promise | teaching photographs needed |
|---|---|---|
| 0.20 | 80% | 4 |
| 0.10 | 90% | 9 |
| 0.05 | 95% | 19 |
| 0.01 | 99% | 99 |

**Pick α from the photographs you have, not the promise you want.** With eleven photographs, 90% is the strongest honest promise. Asking for 95% from eleven does not get you 95%; it gets you `conformal_calibrated = False` and a warning at start-up saying how many you need.

## Set it

```python
from adaptshot import AdaptShotConfig, FewShotLearner
from adaptshot.data import sample_images

paths, labels = sample_images()
learner = FewShotLearner(config=AdaptShotConfig(conformal_alpha=0.10))
learner.load_support_images(paths[:-1], labels[:-1])

result = learner.predict(paths[-1])
print(result.conformal_set, result.conformal_calibrated)
print("informative from", learner.conformal.min_informative_size, "photographs")
```

## What it costs

Measured on the published benchmark (PlantVillage, 5-way 5-shot, 100 episodes, α = 0.10, 25 calibration photographs): the sets contained the true label 98% of the time at a mean size of about 1.7 labels — figures and their intervals are in the [results reference](../reference/results-and-artifacts.md), traced to the artifact by a test. A plain top-1 with a confidence threshold, calibrated to the same 90% target on the same photographs, reached 84%: it missed the promise conformal kept, and conformal paid for it with sets roughly twice as large. That is the trade, and it is stated in the README so you can decide whether you need it.

## Changing α on a taught learner

`conformal_alpha` is fixed when the learner is built. To try another level, build another learner; teaching takes about a second. Do not edit `learner.conformal.alpha` directly — the calibration floor was chosen from α at construction and would no longer match.

## When the set is every class

That is not a bug. It is the learner telling you that it cannot narrow this photograph down at the promise level you asked for. Treat it as a refusal and route to a person — the same rule as `uncertainty_flag`.
