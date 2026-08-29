# AdaptShot: few-shot image classification with a coverage guarantee, on a CPU, offline

*Johnson Christopher Hassan — technical note, v0.3.0. Every number below is read from a committed `results/*.json`; a test fails if this page and the artifact disagree.*

## 1. Problem

A smallholder farmer with a phone and a sick plant has none of the three things image classification is usually built on: a GPU, a reliable connection, and a large labelled dataset. What they have is a handful of photographs of what the disease looks like, taken this season, in this field. The question this project answers is narrow: **from a few such photographs, on the CPU they already own, with the network switched off, can a classifier give an answer — and say, credibly, when it should not?**

The second half is the part that matters. A classifier that is 91% accurate and silent about the other 9% is a classifier that sends a farmer to spray the wrong thing one time in eleven. The output this note is about is not a label. It is a *set* of labels with a guarantee attached, and a refusal when the set would be everything.

## 2. Method

Nothing in the components is new, and this note claims none of it is. The contribution is the combination under the constraints above.

**Embeddings.** A frozen ImageNet-pretrained MobileNetV3-Small, exported once to ONNX and shipped in the wheel (4.0 MB), so inference needs only numpy, Pillow and onnxruntime. The torch path produces the same embeddings to 1.8 × 10⁻⁶; a test enforces it.

**Prototypes.** One mean embedding per class from the support photographs (Snell et al., 2017); a query is assigned to the nearest by cosine distance. As §3 shows, that is the whole of the accuracy.

**Calibration.** Temperature scaling over a sliding window of observed confidences (Guo et al., 2017), so the confidence a user sees is on a scale that means something.

**Conformal prediction sets.** Split conformal (Angelopoulos & Bates, 2021): a nonconformity score per class, a quantile of those scores over a calibration set held out from the queries, and a set of every class whose score falls under it. The guarantee is that the set contains the true class with probability at least 1 − α, over the draw of calibration and test together, under exchangeability. When the calibration set is too small for any finite quantile to exist — fewer than ⌈(1 − α)/α⌉ points — the set is every class, which is the only honest set there.

**Abstention.** A per-class confidence threshold that moves with feedback, and a Mahalanobis out-of-distribution flag with its threshold calibrated leave-one-out over the support set. Either one can route a query to a human instead of answering.

## 3. Results

All figures: PlantVillage crop-disease photographs (Mohanty et al., 2016), 20 classes across seven crops, **5-way 5-shot, 100 episodes, seed 42**, every method on the same episodes and the same embeddings. Mean over episodes with a 95% confidence interval. Artifact: `results/plantvillage_5way5shot.json`.

| Method | Accuracy |
|---|---|
| **AdaptShot** | **91.4% ± 1.0** |
| Nearest centroid, no calibration | 91.4% ± 1.0 |
| Logistic-regression linear probe | 91.4% ± 1.1 |
| 1-NN on raw embeddings | 89.0% ± 1.3 |
| 5-NN on raw embeddings | 87.7% ± 1.2 |

**AdaptShot's accuracy is nearest-centroid's accuracy** — across 500 queries the two disagree on none. The layers above the prototype do not change which class comes out. What they change is what accompanies it:

| At α = 0.10 (90% target) | Conformal sets | Top-1 + calibrated threshold |
|---|---|---|
| Empirical coverage | **98.1% ± 0.6** | 83.9% ± 1.4 |
| Mean set size | 1.66 ± 0.14 | 0.89 ± 0.02 |

The threshold baseline is calibrated on the same held-out split to the same target and misses it. Conformal clears it, at roughly 1.9× the set size. The guarantee is real, it is not free, and a reader who does not need it is told the cheaper thing is cheaper. Conformal also *over*-covers here — 98.1% against 90% — which is set size spent without need; it follows from self-calibrating on 25 points. The nonconformity score is the distance ratio d_true / d_min; the max-scaled softmax it replaced could not distinguish a clean leaf from a blurred one or a different crop (scores 0.72–0.80 for all three), and gave sets of 2.05 at 97.5%.

**Cost, measured on the same runs**, core install, 11th Gen Intel i7-11800H, 16 cores, 31 GB RAM, Linux 7.1, Python 3.14.7:

| | median | p95 |
|---|---|---|
| Embedding, per image | 3.2 ms | 3.8 ms |
| Support fit, per episode | 641 ms | 861 ms |
| Predict, per query, full path | 6.4 ms | 12.7 ms |
| Cold start: fresh interpreter to first answer | 0.85 s | — |

Peak resident memory for that cold-start cycle — one process, one support set, one answer — is **123 MB**. From `pip install` (3.5 MB wheel) to a correct first prediction on the bundled photographs: about five seconds.

**Three things the validation found and fixed before this note was written.** The conformal quantile clamped its rank to *n* − 1 where the theorem's answer is infinite; at the library's own default α and calibration floor that under-covered at 91.3% against 95%. The OOD threshold was calibrated on the points that defined the distribution and flagged 45 of 45 in-distribution photographs; leave-one-out calibration brings it to 3 of 45, with 45 of 45 flagged on genuinely out-of-domain photographs. And the replay-buffer pruner scored uncertainty with the wrong sign, keeping the examples its documentation said it discarded. Each is in a test now.

## 4. Limitations

Each is stated with the condition that triggers it.

1. **The guarantee assumes exchangeability, and the field violates it.** The moment query photographs come from a different camera, season, light, or background than the support set, calibration and test are no longer drawn from the same distribution and the 1 − α bound no longer applies. Measured (`benchmarks/run_shift.py`, 40 episodes): with the queries blurred, re-compressed or downscaled and the support left alone, coverage falls from 96.9% ± 1.0 to 85.5% ± 3.6 at a 90% target. The sets do widen — mean size 1.34 to 2.01 — but the quantile was set on clean photographs and cannot know the queries moved, so the shortfall remains. The OOD flag rate correlates 0.92 with the coverage lost, a partial warning. 10 labelled in-situ corrections through `correct()` recover the worst cell to 87.8% ± 2.4.
2. **Coverage is marginal, not per-class.** The 97.5% is an average over all queries. A class with few or unrepresentative support photographs can be under-covered while the average holds; nothing here promises otherwise for any single class.
3. **Small calibration sets give sets of every class.** At α = 0.05 the engine needs 19 calibration scores before any finite quantile exists; below that the set is all classes by construction. A user with eight support photographs at the default α gets no informative set, and is told so at construction rather than shown a set that is quietly too small.
4. **One dataset, in the lab.** PlantVillage photographs are single leaves on uniform backgrounds under controlled lighting. The 91.4% is a statement about that; a field photograph is a distribution shift of exactly the kind in (1). No field deployment has taken place.
5. **The out-of-distribution flag is sensitive to the support set.** A healthy tomato leaf presented to a maize model was flagged with twelve support photographs and not with eleven. The confidence gate caught it in both cases; the OOD flag alone should not be relied on at few-shot sizes.
6. **The accuracy is a frozen backbone's accuracy.** Nothing above the prototype improves it. A task the ImageNet embedding does not separate is a task this library does not separate either, and it cannot be fine-tuned without the optional torch extra.

## 5. Reproducibility

Seed 42 throughout. Hardware as in §3, recorded in the artifact with the CPU model, core count, memory, OS, Python and library versions. The artifact was produced at commit `5c1365f`. One command each:

```
python scripts/fetch_plantvillage.py --out data/pv_bench --per-class 20 --preset benchmark
python -m benchmarks.run_plantvillage --seed 42
```

The download is manual, pinned to a content-addressed commit of the dataset, and verified against a SHA-256 manifest. Nothing in the repository fetches data on its own; the test suite runs with outbound sockets disabled and fails if the library touches the network.

## References

Angelopoulos, A. N. & Bates, S. (2021). *A gentle introduction to conformal prediction and distribution-free uncertainty quantification.* arXiv:2107.07511.
Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). *On calibration of modern neural networks.* ICML.
Mohanty, S. P., Hughes, D. P. & Salathé, M. (2016). *Using deep learning for image-based plant disease detection.* Frontiers in Plant Science 7:1419.
Snell, J., Swersky, K. & Zemel, R. (2017). *Prototypical networks for few-shot learning.* NeurIPS.
