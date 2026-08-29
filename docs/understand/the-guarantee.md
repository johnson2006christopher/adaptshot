# The guarantee: what the prediction set promises, and where it stops

> **For:** anyone about to rely on a prediction set — a user deciding what to act on, a reviewer deciding what to believe. No code. Every number here is read from a committed results file and checked by a test.

## The promise, stated exactly

AdaptShot's prediction set comes from *split conformal prediction*. Given a level α, the promise is:

> Over many queries, the true label is inside the set at least **1 − α** of the time.

That is the whole of it. It is a strong promise — it needs no assumption about the model being good, or the classes being separable, or the number of classes — and it holds for *any* scoring rule. It costs one assumption and buys one thing.

## The assumption: exchangeability

The promise holds when the calibration photographs and the query photographs are drawn from the same distribution — technically, when they are *exchangeable*. In practice: same kind of camera, similar light, similar distance, the same population of leaves. Teach it in the field where you will use it and the assumption holds. Teach it from a website and use it in the field and it does not.

Nothing in the mathematics can detect the assumption failing. This is why the measured shift curve below matters more than the theorem.

## What it does *not* promise

**It is marginal, not per-class.** 1 − α on average over all queries. One class can be covered 99% of the time and another 70% while the average holds. If a particular class matters more than the average, watch that class.

**It is not a probability for *this* photograph.** A set of one label at 90% does not mean "90% chance this one is right". It means the *procedure* that produced sets like this one is right 90% of the time over many photographs.

**It says nothing about the top-1.** The set can be right (contains the true label) while the best guess in it is wrong. That is not a failure; it is the point.

## The floor: how many photographs it takes

At level α the conformal threshold is the ⌈(n + 1)(1 − α)⌉-th smallest of n calibration scores. When that rank exceeds n — whenever **n < (1 − α)/α** — there is no such score: the honest threshold is infinite and the honest set is every class.

| α | promise | calibration photographs needed |
|---|---|---|
| 0.20 | 80% | 4 |
| 0.10 | 90% | 9 |
| 0.05 | 95% | 19 |
| 0.01 | 99% | 99 |

Before 0.3.0 the library clamped that rank and returned the largest observed score instead — a set smaller than the guarantee permits. The validation harness found it on its first run: at the default α = 0.05 with 10 calibration points, **91.3% measured against a 95% promise**. It now returns the full set there, the learner takes its cold-start path and marks the result `conformal_calibrated = False`, and it warns once at start-up if the floor cannot be met. [Choose α](../how-to/choose-alpha.md) from the photographs you have.

## Measured on real photographs, in distribution

PlantVillage, 5-way 5-shot, 100 episodes, α = 0.10, 25 calibration photographs per episode — above the floor:

| | conformal sets | top-1 with a threshold calibrated to the same target |
|---|---|---|
| empirical coverage | **98.1% ± 0.6** | 83.9% ± 1.4 |
| mean set size | 1.66 ± 0.14 | 0.89 ± 0.02 |

The threshold baseline was calibrated on the same held-out photographs to the same 90% target and **missed it**. Conformal kept it, and paid with sets roughly twice as large. It over-covers — 98% against 90% — which is set size spent without need, a known effect of self-calibrating on 25 points. That is the trade; the README states it so a reader can decide whether they need the promise at all.

## Measured under shift, where the assumption breaks

Same photographs, queries blurred, darkened, re-compressed or downscaled, support left alone (40 episodes):

| queries | coverage | mean set size |
|---|---|---|
| clean | 96.9% ± 1.0 | 1.34 |
| blur σ = 4 | **85.5% ± 3.6** | 2.13 |
| after 10 corrections on blurred photographs | 89.0% ± 2.3 | 1.56 |

Three things to take from it. **The set does widen** — 1.34 to 2.13 — which is the nonconformity score expressing "less sure" (an earlier score could not; see the [changelog](../reference/changelog.md) for 0.3.0). **The promise still bends**: the threshold was set on clean photographs and cannot know the queries moved, so coverage falls under the target anyway. And **a handful of in-situ corrections closes most of the gap** — ten labelled photographs of the shifted condition through `correct()` bring the worst cell back to 89%. That is what the human-in-the-loop path is for.

The OOD flag rate correlates 0.92 with the coverage lost across the shifted cells — it rises as the bound bends — but fires on a minority of the affected queries. A signal, not a guard.

## How to use it, then

1. Teach where you will use it. The assumption is yours to keep.
2. Pick α from the number of photographs you have, not the promise you want.
3. When the set has every label, or `uncertainty_flag` is set, a person decides.
4. When the camera, season or light changes, correct a few photographs from the new condition before trusting the sets again.

## Where the validation lives

`tests/test_conformal_coverage.py` — synthetic classes overlapping enough that top-1 is wrong a quarter of the time, α × calibration-size grid, coverage asserted against the target with a tolerance derived from the trial-level standard error, set size asserted in both directions. `benchmarks/run_plantvillage.py` and `benchmarks/run_shift.py` produce the two artifacts above. The [results reference](../reference/results-and-artifacts.md) lists what each file contains.
