<div align="center">

<img src="docs/images/adaptshot-logo.png" width="300" alt="AdaptShot Logo">

# AdaptShot

**A few-shot image classifier that knows when it doesn't know.**

[![PyPI](https://img.shields.io/pypi/v/adaptshot.svg)](https://pypi.org/project/adaptshot/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/adaptshot?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/adaptshot)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
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
| ✅ **Is** | CPU-first and offline-capable — no GPU, no cloud, no telemetry |
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

Core dependencies are **numpy and Pillow only** — no CUDA, no GPU drivers, no
multi-gigabyte download. PyTorch is optional and needed only for training:

```bash
pip install "adaptshot[torch]"   # CA-EWC fine-tuning and custom backbones
pip install "adaptshot[faiss]"   # faster search for support sets >100 images
pip install "adaptshot[gui]"     # the optional Studio workspace
pip install "adaptshot[dev]"     # contributors: tests, linting, benchmarks
```

> The torch-free core install is enforced by CI on every push
> (`tests/test_torch_optional.py`), not merely documented.

---

## Quick start

```python
from adaptshot import FewShotLearner
from adaptshot.config.settings import AdaptShotConfig

# 1. Configure for your environment
config = AdaptShotConfig(backbone="resnet18", device="cpu", max_buffer_size=100)

# 2. Initialise the learner
learner = FewShotLearner(config=config)

# 3. Show it a few examples per class
learner.load_support_images(
    ["data/healthy_leaf.jpg", "data/blighted_leaf.jpg"],
    ["healthy", "blight"],
)

# 4. Predict
result = learner.predict("data/query.jpg")
print(result.prediction, f"{result.calibrated_confidence:.1%}")

# 5. Respect the uncertainty
if result.uncertainty_flag:
    print("Model is unsure — routing for human review")
```

When a human corrects a prediction, feed it back — the correction updates
calibration and the replay buffer, so the next prediction is better informed:

```python
learner.correct(image_path="data/query.jpg", true_label="blight")
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

📖 [Algorithm theory](docs/guides/algorithm-theory.md) ·
[Architecture](docs/guides/architecture.md) ·
[Tutorials](docs/tutorials.md)

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
| Studio / Pilot GUIs | **Optional extras**, planned to move to a separate project |

Memory: **250 MB is the target, and it is not met today.** Measured peak resident set
size for a full support-set-to-prediction cycle is around **775 MB**
(`tests/test_memory_ceiling.py`, which prints the figure on every run):

| stage | peak RSS |
|---|---|
| interpreter + numpy + Pillow | 33 MB |
| `import adaptshot` | 512 MB |
| `load_support_images` (15 images) | 774 MB |
| `predict()` | 775 MB |

The 479 MB jump at import is PyTorch, pulled in eagerly by `utils/determinism.py` and
`utils/io.py`. The 258 MB after that is ResNet-18's weights and activations. Since
inference currently requires torch ([#35]), no working path stays under 250 MB; the
target is reachable only through the bundled-ONNX backbone ([#36]).

The test asserts a regression budget against what we actually cost, and carries a
strict-xfail on the 250 MB figure — so the build fails when it *starts* passing, and this
section has to be rewritten rather than left stale. `utils/profiling.py` provides a
`MemoryTracker` for measuring your own workload.

[#35]: https://github.com/johnson2006christopher/adaptshot/issues/35
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

Quality gates on every change: `ruff`, `mypy --strict`, 98 tests across 14 modules,
and a deterministic smoke benchmark. See [ROADMAP.md](ROADMAP.md) for what's planned.

> **On benchmarks:** results on real public datasets are in progress for v0.3.0.
> Until they land, treat AdaptShot as a well-tested implementation of well-established
> methods, not as an empirically validated accuracy claim. Numbers will appear here
> when they are reproducible, with seeds and hardware recorded.

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
