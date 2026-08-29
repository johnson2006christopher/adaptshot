<div align="center">

<img src="docs/images/adaptshot-logo.png" width="300" alt="AdaptShot Logo">

# AdaptShot

**A few-shot image classifier that knows when it doesn't know.**

[![PyPI](https://img.shields.io/pypi/v/adaptshot.svg)](https://pypi.org/project/adaptshot/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/adaptshot?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/adaptshot)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-MkDocs-blue)](https://johnson2006christopher.github.io/adaptshot/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-black)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/type--checked-mypy-blue)](https://mypy-lang.org/)
[![Built in Tanzania](https://img.shields.io/badge/Built%20in-Tanzania%20🇹🇿-gold.svg)](https://en.wikipedia.org/wiki/Tanzania)

**[Documentation](https://johnson2006christopher.github.io/adaptshot/)** ·
**[Source](https://github.com/johnson2006christopher/adaptshot)** ·
**[Changelog](CHANGELOG.md)**

</div>

---

Classify images from a handful of examples, on an ordinary CPU, and get an honest
answer about how much to trust each prediction — including a **conformal prediction
set with a coverage guarantee**, and an explicit *"I don't know, ask a human"* when
the model is out of its depth.

```python
result = learner.predict("leaf.jpg")

if result.uncertainty_flag:
    route_to_human_expert(result)     # the model declined to guess
else:
    act_on(result.prediction, result.calibrated_confidence)
```

---

## Why this exists

In a rural clinic or a smallholder farm, three scarcities collide:

| Scarcity | What it means |
| :--- | :--- |
| **Labels** | 5–50 examples per class, because labelling needs an expert who isn't there |
| **Compute** | No GPU, no reliable internet, modest RAM |
| **Trust** | A *confidently wrong* answer costs a harvest, or a life |

Most tools address the first two. The third is the hard one: neural networks are
[systematically overconfident](https://arxiv.org/abs/1706.04599), so a raw softmax
score of 0.97 tells you very little about whether the answer is right. A tool that
guesses confidently and wrongly is worse than no tool, because people act on it.

AdaptShot's position is that **a model which reliably says "I don't know" is worth
more than one that is a few points more accurate and silently wrong.**

---

## What AdaptShot is (and isn't)

**It is a library.** You construct a learner and call it. There is no inversion of
control, no plugin system, no application lifecycle to inherit from. Your program
stays in charge.

| | |
| :--- | :--- |
| ✅ **Is** | A Python library for few-shot image classification with calibrated, guaranteed uncertainty |
| ✅ **Is** | CPU-first and offline — no GPU, no cloud, no telemetry, and CI proves it: the test suite fails if the library touches the network |
| ✅ **Is** | Deterministic — same seed, same hardware, same answer |
| ❌ **Is not** | A framework — it never calls your code |
| ❌ **Is not** | A training platform — the backbone stays frozen; only a small head adapts |
| ❌ **Is not** | A state-of-the-art accuracy play — it trades peak accuracy for trustworthy confidence |

### When to reach for something else

Being honest about this is more useful than pretending otherwise:

| If you have… | Use instead | Because |
| :--- | :--- | :--- |
| Thousands of labels and a GPU | Fine-tune a CNN (`torchvision`, `timm`) | You will get materially higher accuracy |
| Reliable internet and a budget | A hosted vision API | Simpler — but your data leaves the device and costs recur per call |
| A large calibration set, non-vision data | A general conformal library (e.g. MAPIE, crepes) | More general and more mature; they assume more calibration data than few-shot provides |
| GPU infrastructure and deep UQ needs | A torch-based uncertainty toolkit | Richer methods, if you can afford the hardware |

**Reach for AdaptShot when all three are true:** you have few labels, you have no GPU
or no connectivity, and a confident wrong answer is expensive.

---

## Installation

```bash
pip install adaptshot
```

Core dependencies are **numpy, Pillow and onnxruntime** — no CUDA, no GPU
drivers, no multi-gigabyte download. That install can load a support set,
predict, and save and reload a model. **PyTorch is needed only for fine-tuning**
and for backbones other than the bundled one.

```bash
pip install "adaptshot[torch]"   # CA-EWC fine-tuning and custom backbones
pip install "adaptshot[faiss]"   # faster search for support sets >100 images
pip install "adaptshot[dev]"     # contributors: tests, linting, benchmarks
```

Measured, on a full support-set-to-prediction cycle:

| | core install | with torch |
|---|---|---|
| peak RSS | **119 MB** | 775 MB |
| install size for the inference component | ~23 MB | ~1.2 GB |

The default backbone is `mobilenet_v3_small`, whose ONNX weights (4.0 MB) ship in
the wheel. Embeddings agree with the torch path to `4e-06` (cosine 0.99999994),
and the smoke benchmark returns the same accuracy through either. `resnet18` is
44.8 MB and is not bundled; `scripts/export_backbones.py` generates it.

Not sure what your machine can do? Ask it — every figure below is measured on the
machine running the check, none quoted from here:

```python
import adaptshot
print(adaptshot.check_environment())
```

**The limit, stated plainly:** bundled ONNX backbones are frozen. Fine-tuning
(`correct()` with CA-EWC) still requires `adaptshot[torch]`.

> This is enforced by CI on every push, not merely documented:
> `tests/test_torch_optional.py` blocks torch at the import system and then
> *calls* the API. `TORCH_REQUIRED_OPERATIONS` is empty, and the test fails if
> anything is added to it.

---

## Quick start

Seven lines, no dataset, no GPU, no network. The photographs ship in the wheel.

```python
from adaptshot import FewShotLearner
from adaptshot.data import sample_images

paths, labels = sample_images()                      # nine real maize-leaf photos, three per class
learner = FewShotLearner()
learner.load_support_images(paths[:-1], labels[:-1])  # teach it eight
result = learner.predict(paths[-1])                   # ask about the ninth
print(result.prediction, f"{result.calibrated_confidence:.0%}", "-- ask a human" if result.uncertainty_flag else "")
```

That block is executed by CI on every push, with torch and the network both
blocked, so it cannot drift (`tests/test_quickstart.py`). The images are
PlantVillage's, CC BY-SA 3.0, with citation and checksums in
`src/adaptshot/data/samples/README.md`.

**Measured**, from a clean virtualenv on a laptop CPU, one run each:

| step | wall clock | what it does |
|---|---|---|
| `pip install adaptshot` | 4.5 s | a 3.5 MB wheel, plus numpy, Pillow and onnxruntime from PyPI |
| the block above, fresh process, network blocked | 0.4 s | load the backbone, embed nine photos, predict |

Site-packages afterwards: 176 MB. That is one machine on a fast connection; on a
poor one the install is the part that grows, and it is the only part that
touches the network at all.

When a human corrects a prediction, feed it back — the correction updates
calibration and the replay buffer, so the next prediction is better informed:

```python
learner.correct(image_path=paths[-1], true_label=labels[-1])
```

---

## The five ideas behind it

You do not need to understand these to use the library, but they are what makes it
different from a nearest-neighbour search with a confidence number bolted on.

| Idea | What it does | Where |
| :--- | :--- | :--- |
| **Prototypes** | Each class becomes the mean embedding of its examples; a query is classified by distance to prototypes | `core/similarity.py` |
| **Calibration** | Raw scores are rescaled so that "80% confident" actually means right 80% of the time, measured by Expected Calibration Error | `core/calibration.py` |
| **Conformal prediction** | Returns a *set* of labels guaranteed to contain the truth at a chosen rate — distribution-free and valid in finite samples, given exchangeability | `core/conformal.py` |
| **Uncertainty & OOD** | Separates *"the model is guessing"* from *"this input doesn't belong here"*, using a shrinkage-regularised Mahalanobis distance | `core/uncertainty.py` |
| **Human corrections** | Corrections update calibration and a bounded replay buffer, with Fisher-regularised head-only updates to limit forgetting | `training/` |

The conformal guarantee is the core contribution. It is **marginal** coverage — over
the whole distribution, not conditional on any particular class or subgroup. That
distinction matters in deployment and is documented rather than glossed over.

📖 [Start here](docs/tutorials/00-what-is-this.md) — six tutorials for someone who has never coded, every command executed by the tests ·
[How-to guides](docs/how-to/run-the-offline-demo.md) ·
[How it works](docs/understand/how-it-works.md) ·
[The guarantee](docs/understand/the-guarantee.md) ·
[API reference](docs/reference/api.md) ·
[Contributing](docs/contributing/development-setup.md)

---

## Feature status

Honest labels. Experimental means *implemented and tested, not yet validated on
real-world data at scale.*

| Area | Status |
| :--- | :--- |
| Few-shot inference (prototypical, nearest-neighbour) | **Stable** |
| Calibration (temperature, scaling-binning, ECE) | **Stable** |
| Conformal prediction (split, cross, leave-one-out) | **Stable** |
| OOD detection (Mahalanobis with shrinkage) | **Stable** |
| Human-in-the-loop corrections & replay buffer | **Stable** |
| Determinism & checkpoint persistence | **Stable** |
| Continual learning (head-only CA-EWC, ~2K params) | **Experimental** |
| Contrastive prototypes (InfoNCE) | **Experimental** — requires torch |
| Explainability (embedding-space attribution) | **Experimental** |
| ONNX backbone export | **Experimental** |
| Graphical interfaces | **Removed** — the library ships none. See [Tambua](apps/tambua/README.md) |

Memory: **the 250 MB target is met on a core install.** Measured peak resident set size
for a full support-set-to-prediction cycle, by `tests/test_memory_ceiling.py`, which
prints the figure on every run:

| stage | core install | with torch |
|---|---|---|
| interpreter + numpy + Pillow | 33 MB | 33 MB |
| `import adaptshot` | 38 MB | ~512 MB |
| `load_support_images` (15 images) | 120 MB | ~590 MB |
| `predict()` | **120 MB** | 592 MB |

Installing torch alongside costs about 500 MB at import, because
`training/finetune.py` guards its torch import at module scope — it does not crash
without torch, but it does pay for it whenever torch is present. That is tracked in
[#36] and does not affect the core install.

`utils/profiling.py` provides a `MemoryTracker` for measuring your own workload.

[#36]: https://github.com/johnson2006christopher/adaptshot/issues/36

---

## Configuration

`AdaptShotConfig` is strictly typed and immutable, so a run is reproducible from its
config alone. The most common fields:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `backbone` | `str` | `"resnet18"` | Feature extractor (`resnet18` or `mobilenet_v3_small`) |
| `device` | `str` | `"cpu"` | Execution device (`cpu`, `cuda`, or `mps`) |
| `seed` | `int` | `42` | Random seed for deterministic reproducibility |
| `n_way` | `int` | `5` | Number of classes per episode |
| `k_shot` | `int` | `10` | Support examples per class |
| `similarity_metric` | `str` | `"euclidean"` | Distance metric (`cosine` or `euclidean`) |
| `inference_mode` | `str` | `"prototypical"` | `nearest_neighbor`, `prototypical`, or `contrastive` |
| `calibration_method` | `str` | `"temperature"` | `temperature`, `scaling_binning`, `conformal`, or `none` |
| `conformal_alpha` | `float` | `0.05` | Significance level — coverage target is `1 − alpha` |
| `conformal_mode` | `str` | `"split"` | `split` or `cross` (k-fold cross-conformal) |
| `uncertainty_mode` | `str` | `"ensemble"` | `mcdropout`, `entropy`, `mahalanobis`, or `ensemble` |
| `enable_ood_detection` | `bool` | `True` | Flag images outside the known support distribution |
| `max_buffer_size` | `int` | `100` | Replay buffer capacity, enforced by UP-UGF pruning |
| `eco_mode` | `bool` | `False` | Energy-saving early-exit thresholds |

📖 **[Full configuration reference — every field](docs/reference/config-reference.md)**

---

## Deployment

Runs anywhere a CPU and Python 3.9+ exist:

- **Edge devices** — single-board computers and older laptops with limited RAM
- **Offline stations** — fully functional without internet once backbone weights are cached
- **Cloud** — standard CPU instances, with no GPU line on the bill

---

## Project status

**Current release: v0.2.0.** Pre-1.0 — the API may still change between minor
versions, and each change is documented in the [changelog](CHANGELOG.md).

| Version | Theme |
| :--- | :--- |
| **v0.1.x** | *Built it.* Few-shot inference, calibration, human corrections, eco mode |
| **v0.2.0** | *Made it honest.* Conformal prediction, multi-signal uncertainty, OOD detection — plus a substantial pass replacing claims that the code did not yet support |
| **v0.3.0** *(in progress)* | *Make it provable.* Validation on real public datasets, a narrower and better-defended API, GUIs split into their own project |

Quality gates on every change: `ruff`, `mypy --strict`, 232 tests, and a
deterministic smoke benchmark. See [ROADMAP.md](ROADMAP.md) for what's planned.

---

## Offline, proven

"Works offline" is tested, not promised. On every push, a CI job builds the wheel,
installs it into a clean environment, then enters a Linux network namespace with **no
interfaces at all** — proven by a canary connection that must fail — and inside it runs
the README quickstart, the conference demo, the conformal and calibration suites, and
the smoke benchmark, against the installed wheel rather than the source tree. Nothing
is pre-fetched: the backbone and the sample photographs ship in the wheel.

Any dependency that adds a download, a telemetry call, or a version check in a patch
release fails that job. The job is `offline-wheel` in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml); its runs are under
[Actions](https://github.com/johnson2006christopher/adaptshot/actions/workflows/ci.yml).

---

## Results on real data

PlantVillage, **5-way 5-shot, 100 episodes, seed 42**, `mobilenet_v3_small`.
Every method sees the same episodes and the same embeddings. Accuracy is the
mean over episodes with a 95% confidence interval — never a bare point estimate.

| Method | Accuracy |
|---|---|
| **AdaptShot** | **91.4% ± 1.0** |
| Nearest centroid | 91.4% ± 1.0 |
| Linear probe (logistic regression) | 91.4% ± 1.1 |
| 1-NN | 89.0% ± 1.3 |
| 5-NN | 87.7% ± 1.2 |

**AdaptShot's accuracy is nearest-centroid's accuracy.** Not approximately —
across 500 queries the two disagree on zero of them. The top-1 prediction *is* a
cosine nearest-centroid classifier, and the layers above it do not change which
class comes out. If accuracy is all you need, the five-line version is the same
five lines.

What those layers do change is what you are told alongside the prediction:

| At α = 0.1 (90% target coverage) | Conformal sets | Top-1 + threshold |
|---|---|---|
| Empirical coverage | **98.1% ± 0.6** | 83.9% ± 1.4 |
| Mean set size | 1.66 ± 0.14 | 0.89 ± 0.02 |

The threshold baseline is calibrated on the same held-out split, to the same
target, and it *misses* it — 83.9% against a promised 90%. Conformal clears it
with room to spare, and the price is roughly 1.9× the set size. That is the
trade, stated plainly: the guarantee is real, it is not free, and if you do not
need it the cheaper thing is genuinely cheaper.

Conformal over-covers here (98.1% against a 90% target), which costs set size
it did not have to spend. That is a consequence of self-calibrating by
leave-one-out on 25 support points. The score behind these sets is the distance ratio
`d_true / d_min` (#86); the max-scaled softmax it replaced produced sets of 2.05 at 97.5%.

OOD flagged 1.5% ± 0.5 of this pool, which is entirely in-distribution.

Reproduce:

```bash
python scripts/fetch_plantvillage.py --out data/pv_bench --per-class 20 --preset benchmark
python -m benchmarks.run_plantvillage --seed 42
```

### Latency and memory, measured on the same runs

Median and p95, not mean — tail latency is what makes a tool feel broken. All from
`results/plantvillage_5way5shot.json`, and a test fails if this table drifts from it.

| stage | median | p95 | what it includes |
|---|---|---|---|
| embedding, per image | **3.2 ms** | 3.8 ms | the ONNX forward pass, cache bypassed |
| support fit, per episode | **641 ms** | 861 ms | embed 11 photos, leave-one-out calibration, OOD fit |
| predict, per query | **6.4 ms** | 12.7 ms | the full path, embedding included |
| cold start | **0.85 s** | — | a fresh interpreter, from before `import adaptshot` to the first answer |

**Peak memory for that cold-start cycle: 123 MB**, in-process, on the core install.
The benchmark harness itself peaks at 541 MB with 400 cached embeddings and four
baselines in memory; that number describes the harness, not the library, and the artifact
names them separately so they cannot be confused.

Measured on: **11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz**, 16 cores, 31 GB RAM,
Linux-7.1.10-200.fc44.x86_64-x86_64, Python 3.14.7, numpy 2.5.2, onnxruntime 1.29.0, `core` install.

Two honest caveats. The p95 of the support fit varied 2.7× between two runs on this
machine (1.5 s and 4.0 s) while its median held within 2%; the tail is this laptop's,
and the median is the number to plan around. And this laptop is faster than the hardware
the project is built for. These figures are what the library costs *here*; a measurement
on ARM or phone-class hardware is its own open item.

### Under distribution shift — where the guarantee stops holding

The conformal guarantee assumes the queries come from the same distribution as the
calibration set. In the field they do not. This is what happens when the **queries
shift and the support does not** — real PlantVillage photographs, blurred, darkened,
re-compressed or downscaled with Pillow, 40 episodes, α = 0.1 (target
90%). Right-hand columns: the same queries after **10 shifted, labelled
photographs** are fed through `correct()`, the library's human-in-the-loop path.

| shift | top-1 | coverage | set size | OOD flagged | coverage after 10 corrections | set size |
|---|---|---|---|---|---|---|
| clean | 93% | **96.9% ± 1.0** | 1.34 | 1.8% | **96.7% ± 1.2** | 1.11 |
| blur 1 | 90% | **95.2% ± 1.6** | 1.44 | 1.7% | **95.7% ± 1.3** | 1.16 |
| blur 2 | 81% | **91.6% ± 2.4** | 1.77 | 2.6% | **93.5% ± 1.6** | 1.28 |
| blur 4 | 67% | **85.8% ± 3.4** | 2.13 | 10.0% | **89.0% ± 2.3** | 1.56 |
| brightness 0.6 | 92% | **96.0% ± 1.5** | 1.36 | 1.4% | **96.6% ± 1.2** | 1.12 |
| brightness 0.3 | 89% | **94.3% ± 2.0** | 1.49 | 0.8% | **94.5% ± 1.8** | 1.15 |
| brightness 1.6 | 93% | **96.9% ± 1.1** | 1.38 | 2.5% | **96.5% ± 1.2** | 1.12 |
| jpeg 40 | 92% | **96.8% ± 1.0** | 1.38 | 2.2% | **96.5% ± 1.2** | 1.12 |
| jpeg 15 | 88% | **94.4% ± 1.6** | 1.53 | 4.0% | **94.7% ± 1.5** | 1.18 |
| jpeg 5 | 72% | **85.5% ± 3.6** | 2.01 | 10.5% | **87.8% ± 2.4** | 1.48 |
| downscale 0.5 | 91% | **95.4% ± 1.6** | 1.45 | 1.5% | **95.9% ± 1.3** | 1.15 |
| downscale 0.25 | 82% | **92.8% ± 2.3** | 1.77 | 3.0% | **94.0% ± 1.4** | 1.26 |
| downscale 0.125 | 68% | **86.9% ± 3.6** | 2.12 | 13.7% | **90.2% ± 2.3** | 1.48 |

**Under shift the sets widen, and the guarantee still bends.** Under strong blur, JPEG or
downscale, coverage falls to **85.5% ± 3.6** (jpeg 5) against a
90% target, while the mean set size grows from 1.34 to
2.01. The set does say "less sure" — that is the nonconformity score doing its
job (#86) — but not by enough: the calibration quantile was set on clean photographs and cannot
know the queries have moved. That is exchangeability breaking, and no score fixes it.

**The OOD flag is a partial early warning.** Across the shifted cells its rate correlates 0.92 with the coverage lost — it rises as the guarantee bends — but fires on a
minority of the affected queries at the worst levels.

**A handful of in-situ corrections closes most of the gap.** 10 labelled photographs of the
shifted condition per episode, through `correct()`, move the worst cell from
85.5% ± 3.6 to 87.8% ± 2.4, with the sets back down to
1.48. Real help; the right move in the field.

Every cell traces to `results/plantvillage_shift.json`; reproduce with
`python -m benchmarks.run_shift --seed 42`.


Full output including hardware, dataset commit and licence is written to
`results/plantvillage_5way5shot.json`. The download is manual and pinned to a
content-addressed commit; nothing in this repository fetches data silently.

---

## Contributing

Contributions are welcome — code, documentation, testing, or a bug report from a
real deployment.

```bash
git clone https://github.com/johnson2006christopher/adaptshot.git
cd adaptshot
pip install -e ".[dev,torch]"

ruff check src/ tests/
mypy src/adaptshot --strict
pytest tests/ -v
python -m benchmarks.run_benchmark --smoke-test --seed 42
```

All four must pass before a pull request. See [CONTRIBUTING.md](CONTRIBUTING.md),
[SECURITY.md](SECURITY.md), and the [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Citing

If AdaptShot is useful in your work:

```bibtex
@software{hassan_adaptshot,
  author  = {Hassan, Johnson Christopher},
  title   = {AdaptShot: Human-Aligned Few-Shot Vision Learning for
             Resource-Constrained Environments},
  url     = {https://github.com/johnson2006christopher/adaptshot},
  version = {0.2.0},
  year    = {2026}
}
```

GitHub also renders a **Cite this repository** button from
[`CITATION.cff`](CITATION.cff), in APA and BibTeX. The two are kept in step by
`tests/test_citation.py`, which fails if the versions drift apart.

<!-- TODO(maintainer): once the repository is connected to Zenodo and a release
     is published, add the concept DOI badge here and a `doi` field to
     CITATION.cff. Deliberately absent rather than faked: a DOI that does not
     resolve is worse than no DOI. See issue #24 for the steps. -->

---

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">
<img src="docs/images/johnson.jpeg" width="140" style="border-radius: 50%;" alt="Johnson Christopher Hassan">

**Johnson Christopher Hassan**

Vision AI researcher and software engineer · Mbeya, Tanzania 🇹🇿

[GitHub](https://github.com/johnson2006christopher) ·
[LinkedIn](https://www.linkedin.com/in/johnson-christopher-hassan) ·
[Email](mailto:johnson2006christopher@gmail.com)

<br>

*Built for the places where AI has to work without a GPU, without the internet,<br>and without the luxury of being confidently wrong.*

</div>
