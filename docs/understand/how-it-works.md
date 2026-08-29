# How it works: from a photograph to an answer

> **For:** someone who has used AdaptShot and wants to know what happens between `predict()` and the result — a curious user, a contributor about to change something, a reviewer. One page. No code to run; the [reference](../reference/api.md) has the signatures.

## The pipeline in one picture

```text
photograph
   │
   ▼
[1] embedding      frozen backbone (ONNX, bundled) → 576 numbers
   │
   ▼
[2] similarity     distance to each class prototype (mean of that class's teaching embeddings)
   │
   ├──▶ [3] calibration      raw similarity → a confidence that means what it says
   ├──▶ [4] conformal set    which classes fall under the calibrated threshold
   ├──▶ [5] uncertainty      three kinds of doubt, and the out-of-distribution flag
   └──▶ [6] ACT gate         accept, or ask a person
   │
   ▼
PredictionResult
   │
   ▼  (if a person corrects it)
[7] feedback router   buffer → calibration → conformal → thresholds → (optional) CA-EWC fine-tuning
```

Everything from [2] onward is numpy. The only neural network is the frozen backbone in [1], and it never changes — which is what makes a full cycle cost 120 MB and a prediction a few milliseconds on a CPU.

## [1] Embedding — `core/extractor.py`

A photograph is resized to 224×224, normalised with ImageNet statistics, and pushed through MobileNetV3-Small with its classifier removed. Out come 576 numbers that describe the image in a space where similar-looking images are close. The backbone is ImageNet-pretrained and frozen: AdaptShot never trains it. It ships as an ONNX graph (4 MB) and runs through onnxruntime, so no PyTorch is needed; with the torch extra installed the torch path is available too, and a test enforces the two agree to within 1.8 × 10⁻⁶.

An *eco mode* keeps a tiny preview signature of the top support image and skips the forward pass when a query is near-identical to it — useful for repeated frames from a camera, deterministic, and off by default in the benchmarks.

## [2] Similarity and prototypes — `core/similarity.py`

At teaching time each class's embeddings are averaged into one *prototype* (Snell et al., 2017). At query time the distance from the query embedding to each prototype is computed; the nearest one is the prediction. That is the entire classifier, and on the published benchmark it *is* the accuracy — nearest-centroid and AdaptShot agree on every one of 500 queries. Cosine and Euclidean distance are both available; `nearest_neighbor` mode compares against every teaching image instead of the average; `contrastive` mode (experimental) trains a small projection head first.

## [3] Calibration — `core/calibration.py`

Raw similarity is not a probability. `CalibrationEngine` fits a temperature over a sliding window of observed confidences and outcomes — each correction adds one observation — so that a calibrated 0.7 is right about 70% of the time. It reports expected calibration error (ECE) and a debiased variant. On a fresh learner the temperature is bootstrapped by leave-one-out over the teaching set, so there is a sensible scale before any correction arrives.

## [4] The conformal set — `core/conformal.py`

Each class gets a *nonconformity score* — since 0.3.0 the ratio of its distance to the nearest distance — and the classes whose score falls under a threshold form the set. The threshold is the ⌈(n+1)(1−α)⌉-th smallest score over n calibration points, which is what makes the coverage guarantee hold under exchangeability. The learner calibrates leave-one-out over its own teaching set, so no separate calibration split is needed; below ⌈(1−α)/α⌉ points the set is uninformative and the result says `conformal_calibrated = False`. [The guarantee](the-guarantee.md) explains what this does and does not promise, with measurements.

## [5] Uncertainty and the OOD flag — `core/uncertainty.py`

Three signals, each in [0, 1]: *epistemic* — perturb the embedding slightly and see whether the answer changes; *aleatoric* — entropy of the labels among the nearest teaching images; *distributional* — Mahalanobis distance to the nearest class, under a shrinkage-regularised covariance because 5 samples in 576 dimensions have no invertible covariance of their own. The distributional term drives `ood_flag`, with its threshold calibrated leave-one-out over the teaching set (before 0.3.0 it was calibrated in-sample and flagged 100% of held-out photographs). A composite of the three is reported too.

## [6] The ACT gate — `core/act.py`

Adaptive Confidence Thresholding keeps a per-class acceptance threshold that moves with feedback: wrong acceptances push it up, unnecessary requests pull it down, and a weak mean reversion draws it back toward its base so it cannot drift to an extreme and stay. `should_accept` compares the calibrated confidence against it and yields `ACCEPT` or `REQUEST_FEEDBACK`. `uncertainty_flag` is simply "not accepted, or OOD".

## [7] The feedback loop — `training/`

`correct()` hands a `Correction` to the `FeedbackRouter`, which:

1. adds the corrected embedding to the *replay buffer* and rebuilds the prototypes;
2. adds the outcome to the calibration window and the conformal calibration;
3. moves the class's ACT threshold;
4. queues the correction; when five are pending and the torch extra is installed, fine-tunes a small head with **CA-EWC** — learning the new corrections while penalising changes that would undo earlier ones, weighted by each correction's confidence.

The buffer has a capacity (`max_buffer_size`, 100 by default). Over it, **UP-UGF** scores every stored example by its uncertainty (uncertain, boundary examples are the informative ones), its recency, and its redundancy against the rest, and keeps the highest-scoring. Above 100 rows the redundancy term switches to a random-projection LSH approximation.

## Persistence — `FewShotLearner.save/load`

Two files: a JSON with configuration, calibration state, thresholds, prototypes and a SHA-256 checksum, and an `.npy` with the buffer's embeddings. Schema-versioned; older files migrate on load with a warning. The fine-tuned head is not persisted — it is rebuilt from the saved corrections.

## Design decisions worth knowing

- **Numpy after the backbone.** Every layer above [1] is plain numpy so the core install has three dependencies and inference is torch-free. Torch is imported lazily and only where fine-tuning or a non-bundled backbone needs it.
- **Immutable configuration.** `AdaptShotConfig` is a frozen dataclass validated at construction; a learner's behaviour is fixed when it is built, and saved with it.
- **Determinism.** Every source of randomness is seeded from `config.seed`, including the epistemic perturbation, which hashes the embedding to pick its seed so the same photograph always gets the same answer.
- **Everything measured.** Latency, memory, coverage and accuracy are read from committed artifacts by tests; the numbers in this documentation cannot drift from them silently.

## Where each piece is tested

`tests/test_onnx_parity.py` ([1]); `test_extractor.py`, `test_learner_integration.py` ([2]); `test_calibration.py` ([3]); `test_conformal.py`, `test_conformal_coverage.py` ([4]); `test_ood_calibration.py` ([5]); `test_act.py` ([6]); `test_up_ugf.py`, `test_persistence.py` ([7] and persistence). The [contributor guide](../contributing/development-setup.md) explains the gate they run in.
