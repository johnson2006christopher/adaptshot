# 🌿 AdaptShot

<div align="center">

**Human-Aligned Few-Shot Vision Learning for Resource-Constrained Environments**

[![PyPI](https://img.shields.io/pypi/v/adaptshot.svg)](https://pypi.org/project/adaptshot/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/johnson2006christopher/adaptshot/actions/workflows/ci.yml/badge.svg)](https://github.com/johnson2006christopher/adaptshot/actions)
[![arXiv](https://img.shields.io/badge/arXiv-Coming%20Soon-red.svg)](https://arxiv.org/)
[![Built in Tanzania](https://img.shields.io/badge/Built%20in-Tanzania%20🇹🇿-gold.svg)](https://en.wikipedia.org/wiki/Tanzania)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-black)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/type--checked-mypy-blue)](https://mypy-lang.org/)

> *A zero-config, CPU-first, human-in-the-loop few-shot vision library that learns from every correction, guarantees calibrated uncertainty, and runs deterministically on edge hardware with fewer than 50 images per class.*

[📦 Install](#-installation--quick-start) · [📖 Origin Story](#-the-origin-story) · [🎯 Mission](#-mission--vision) · [🆚 Comparison](#-how-adaptshot-differs) · [📊 Metrics](#-target-metrics--validation-philosophy) · [🗺️ Roadmap](#️-roadmap--open-science-commitment) · [🤝 Contribute](#-governance--community)

</div>

---

## 🧭 Table of Contents
<details>
<summary><strong>Click to expand</strong></summary>

- [📖 The Origin Story](#-the-origin-story)
- [🎯 Mission & Vision](#-mission--vision)
- [🔍 Why AdaptShot Exists](#-why-adaptshot-exists-the-gap)
- [🆚 How AdaptShot Differs](#-how-adaptshot-differs)
- [🏗️ Architecture & Core Algorithms](#️-architecture--core-algorithms)
- [📊 Target Metrics & Validation Philosophy](#-target-metrics--validation-philosophy)
- [📦 Installation & Quick Start](#-installation--quick-start)
- [🗺️ Roadmap & Open Science Commitment](#️-roadmap--open-science-commitment)
- [🤝 Governance & Community](#-governance--community)
- [📄 Citation & License](#-citation--license)
- [🙏 Acknowledgments](#-acknowledgments)
- [✨ Closing](#-closing)

</details>

---

## 📖 The Origin Story

**AdaptShot was created by Johnson Christopher Hassan in 2026, from Mbeya, Tanzania.**

It was not born in a well-funded research lab. It was not built on a multi-GPU cluster. It began on a modest workstation, constrained by a Tesla P100 that frequently failed under modern CUDA kernels, limited internet bandwidth, and the quiet realization that almost every "production-ready" AI library was engineered for a reality most of the world simply does not have access to.

At 18 years old, self-taught and working independently, I began asking a simple question: *What if AI didn't require abundance to be useful?*

The answer became AdaptShot.

I am building this project in public. I will document every architectural decision, every failed experiment, every calibration insight, and every limitation. I do this because transparency matters more than polish, and because the AI community needs more builders who are willing to say: *"I am learning. This is hard. But the direction matters."*

This library is dedicated to every researcher, engineer, and domain expert who has been told their environment is "not supported." It is built to prove that world-class, reproducible, human-aligned AI can be developed outside traditional centers of funding and privilege — using discipline, open science, and relentless iteration.

---

## 🎯 Mission & Vision

### Mission
To deliver a **practical, trustworthy, and deployable** few-shot vision system that:
- Learns reliably from fewer than 50 labeled examples per class
- Runs deterministically on CPU-only or edge hardware
- Integrates human feedback as a first-class learning signal
- Communicates uncertainty honestly, especially in high-stakes domains
- Remains fully open, auditable, and community-governed

### Vision
A future where AI tools are not reserved for institutions with massive datasets and GPU farms, but are accessible to clinicians, farmers, educators, and engineers operating under real-world constraints. We believe AI should **serve scarcity, not ignore it**.

### Core Principles
| Principle | Engineering Implementation |
|-----------|---------------------------|
| **Scarcity-Aware Design** | Optimized for <50 images/class, <250MB RAM, CPU-first execution |
| **Human Alignment > Automation** | Built-in feedback routing; corrections trigger immediate, bounded model updates |
| **Calibration as Trust** | Expected Calibration Error (ECE) tracked, bounded, and exposed in every prediction |
| **Deterministic Reproducibility** | Fixed seeds, CPU-safe ops, versioned configs, CI-enforced benchmarks |
| **Open by Default** | MIT license, public repository, RFC-driven governance, transparent limitations |

---

## 🔍 Why AdaptShot Exists: The Gap

Modern AI tooling assumes abundance. AdaptShot assumes constraint.

| Real-World Constraint | Conventional AI Stack Response | AdaptShot's Engineering Response |
|----------------------|-------------------------------|----------------------------------|
| <50 labeled images per class | Requires meta-training on millions of auxiliary samples | Metric-based retrieval + conservative augmentation + few-shot splitting |
| No GPU / low-spec hardware | Assumes CUDA availability; crashes on CPU or M-series Macs | All ops default to `cpu`. FAISS-CPU or NumPy fallback. Zero `.cuda()` assumptions |
| Expert corrections available | Treated as offline batch retraining or ignored entirely | Real-time `FeedbackRouter` wires ✓/✗ to replay buffer + head-only fine-tuning |
| High-stakes deployment (health, agriculture) | Overconfident softmax outputs; ECE often >0.15 | Temperature scaling + conformal prediction stub + ACT gating for uncertain cases |
| Edge memory limits (<250MB) | Replay buffers grow until OOM; no pruning | UP-UGF scores embeddings by uncertainty, recency, redundancy; enforces hard capacity |

---

## 🆚 How AdaptShot Differs

AdaptShot does not aim to replace foundational frameworks. It fills a specific, underserved niche: **human-aligned, calibrated, edge-ready few-shot vision**.

| Feature | AdaptShot | learn2learn | modAL | lightly | Hugging Face Transformers | Avalanche |
|---------|:---------:|:-----------:|:-----:|:-------:|:-------------------------:|:---------:|
| CPU-first guarantee | ✅ | ❌ | ⚠️ | ❌ | ❌ | ⚠️ |
| <50 images/class | ✅ | ⚠️ | ❌ | ❌ | ❌ | ❌ |
| Human-in-the-loop feedback | ✅ | ❌ | ⚠️ | ❌ | ❌ | ❌ |
| Calibrated uncertainty (ECE) | ✅ | ❌ | ❌ | ❌ | ⚠️ | ⚠️ |
| Bounded memory (pruning) | ✅ | ❌ | ❌ | ❌ | ❌ | ⚠️ |
| Continual learning (EWC variant) | ✅ | ⚠️ | ❌ | ❌ | ❌ | ✅ |
| Transparent nearest-neighbor | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Deterministic reproducibility | ✅ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ |
| Edge/offline deployment | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Zero-config API | ✅ | ❌ | ❌ | ⚠️ | ❌ | ❌ |

*✅ = First-class support · ⚠️ = Partial / requires manual config · ❌ = Not supported*

---

## 🏗️ Architecture & Core Algorithms

### High-Level Pipeline
```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│   Input Image   │────▶│  ResNet-18 (frozen)  │────▶│  512-dim embedding  │
│ (PIL / file)    │     │  (torchvision)       │     │                     │
└─────────────────┘     └──────────────────────┘     └──────────┬──────────┘
                                                                │
                                                                ▼
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│   Prediction    │◀────│  Cosine similarity   │◀────│   FAISS-CPU / NumPy │
│   (class, conf, │     │  with support set    │     │   IndexFlatIP       │
│    neighbor)    │     │                      │     │                     │
└─────────────────┘     └──────────────────────┘     └─────────────────────┘
        │                                                        ▲
        │                                                        │
        ▼                                                        │
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  ACT Decision   │────▶│  Confidence > τ ?    │────▶│  Yes → return       │
│  Engine         │     │                      │     │  No → request human │
└─────────────────┘     └──────────────────────┘     └──────────┬──────────┘
                                                                │
                                                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Feedback Router                                 │
│  Human provides corrected label + confidence weight w_c                │
│  → Append/update embedding in replay buffer                            │
│  → Trigger CA-EWC fine‑tune (head-only, ≤10 epochs)                    │
│  → Run UP-UGF if buffer > capacity                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

### Core Research Components
All three algorithms are proposed as co-designed system components. They will be empirically validated, ablated, and documented as research contributions, not marketing features.

| Component | Core Idea | Target Benefit |
|-----------|-----------|----------------|
| **ACT** (Adaptive Confidence Thresholding) | Dynamically adjusts decision thresholds using model uncertainty + correction history | Reduces false acceptances by requesting human review when genuinely unsure |
| **CA-EWC** (Correction-Aware EWC) | Weights EWC regularization penalty by human feedback confidence | Prevents catastrophic forgetting while respecting the reliability of human signal |
| **UP-UGF** (Uncertainty-Guided Forgetting) | Prunes replay buffer using uncertainty × recency × (1−redundancy) scoring | Keeps memory bounded (<250MB) without sacrificing accuracy on edge devices |

---

## 📊 Target Metrics & Validation Philosophy

> ⚠️ **Honesty Note**: The following are *engineering targets* for `v0.1.0` and `v1.0.0`. They are not claimed achievements. All metrics will be empirically validated, variance-reported, and published with reproducible scripts before release.

| Metric | Target (v0.1.0) | Hardware Context | Validation Method |
|--------|----------------|------------------|-------------------|
| Few-shot accuracy | >70% on 10-shot tasks | Intel i5-1135G7 / Apple M1 | 5-fold episode sampling, report mean ± std |
| Expected Calibration Error (ECE) | <0.05 | Same as above | Temperature scaling + 15-bin ECE on held-out set |
| Inference latency (p95) | <50ms (x86), ~120ms (RPi 4) | Single-thread CPU | 1000 runs, `time.perf_counter_ns()`, no warmup bias |
| Peak RAM usage | <250MB (core pipeline) | Python 3.10 + PyTorch CPU | `tracemalloc` + RSS monitoring |
| Deterministic reproducibility | Bit-exact outputs across runs | Fixed seed `42` | Hash comparison of embeddings + predictions |
| Feedback efficiency | 1 correction → measurable accuracy gain | Simulated expert loop | Pre/post fine-tuning evaluation on held-out set |

### Validation Commitments
- ✅ All benchmarks run on CPU-only by default
- ✅ Seeds fixed: `torch.manual_seed`, `np.random.seed`, `PYTHONHASHSEED`
- ✅ Variance reported: 5-fold splits, confidence intervals where applicable
- ✅ Reproduction script: `python benchmarks/run_benchmark.py --dataset <name> --seed 42`
- ✅ Failure cases documented: We will publish cases where calibration or accuracy degrades

---

## 📦 Installation & Quick Start

### Stable Release (Upcoming)
```bash
pip install adaptshot
```

### Optional Extras
```bash
# FAISS-CPU acceleration (recommended for support sets >100)
pip install adaptshot[faiss]

# Development toolchain
pip install adaptshot[dev]

# Full stack (examples, benchmarks, visualization)
pip install adaptshot[all]
```

### From Source (Recommended for Research)
```bash
git clone https://github.com/johnson2006christopher/adaptshot.git
cd adaptshot
pip install -e ".[dev]"
pytest tests/ -v
```

### 5-Line Quick Start
```python
from adaptshot import FewShotLearner

learner = FewShotLearner(
    backbone="resnet18",
    classes=["healthy_leaf", "blight", "rust"],
    device="cpu"
)
learner.load_support_images("dataset/train/", k=10)

pred, confidence, neighbor = learner.predict("field_photo.jpg")
print(f"Prediction: {pred} | Confidence: {confidence:.3f}")
```

---

## 🗺️ Roadmap & Open Science Commitment

| Phase | Timeline | Deliverables |
|-------|----------|-------------|
| `v0.1.0` Alpha | May–Jun 2026 | Benchmark harness + TorchScript extractor + CalibratorProtocol + SAFETY.md + PyPI alpha |
| `v0.1.0` Beta | Jul 2026 | Conformal prediction integration + ACT + UP-UGF + arXiv draft + HF Spaces demo |
| `v1.0.0` Stable | Aug–Oct 2026 | Full ablation studies + hardware tier validation + scikit-learn compat API + PyPI stable |
| `v1.5.0` Edge | Nov 2026–May 2027 | INT8 quantization + ONNX export + Federated buffer sharing (experimental) + Mobile SDK stub |

### Open Science Practices
- All code released under MIT with deterministic reproduction scripts
- Benchmarks published as JSON + markdown with full hardware specs
- Failure cases and calibration limits documented alongside successes
- Community RFC process for major feature additions
- No proprietary telemetry, no hidden dependencies, no vendor lock-in

---

## 🤝 Governance & Community

AdaptShot is **100% open-source and community-driven**. There is no corporate backer. The project succeeds or fails based on what the community builds together.

### How to Contribute
1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Ensure `pytest tests/ -v`, `mypy src/ --strict`, and `ruff check src/` pass
4. Open a Pull Request with:
   - Clear description of changes
   - Benchmark diffs if performance/calibration affected
   - Reference to related issue or discussion

### Decision-Making
- Technical decisions made via public RFCs in GitHub Discussions
- Core maintainers rotate based on contribution volume and domain expertise
- Funding/sponsorships disclosed publicly
- **No corporate veto power. Community > Vendors. Always.**

---

## 📄 Citation & License

### License
AdaptShot is released under the **[MIT License](LICENSE)**.  
You are free to use, modify, distribute, and deploy it for personal, academic, or commercial purposes. Commercial deployments requiring SLA support or white-label usage should contact the maintainers.

### Citation
If you use AdaptShot in research or deployment, please cite:
```bibtex
@misc{adaptshot2026,
  title={AdaptShot: Zero-Config Human-in-the-Loop Few-Shot Learning with Calibrated Uncertainty},
  author={Hassan, Johnson Christopher},
  year={2026},
  howpublished={\url{https://github.com/johnson2006christopher/adaptshot}},
  note={arXiv preprint (forthcoming)}
}
```

---

## 🙏 Acknowledgments

- **Algorithmic Inspiration**: Few-shot learning, active learning, uncertainty calibration, and continual learning literature from NeurIPS, ICCV, ICML, and MLSys (2017–2026).
- **Open-Source Ecosystem**: PyTorch, Hugging Face, FAISS, Gradio, scikit-learn, and the broader Python scientific community.
- **Early Testers & Community**: Researchers, engineers, and domain experts who reviewed early architecture, benchmark designs, and safety documentation.
- **Home**: Mbeya, Tanzania 🇹🇿 — where constraint breeds creativity, and scarcity demands elegance.

---

## ✨ Closing

> *"The best AI doesn't guess confidently. It learns humbly, admits uncertainty, and improves through every human correction."*  
> — Johnson Christopher Hassan, 2026

AdaptShot is more than a library. It is a commitment to building AI that respects real-world constraints, honors human expertise, and remains open to anyone who wants to learn, adapt, or deploy it.

If you believe that practical, trustworthy, human-aligned AI is possible — even in the world's most resource-constrained environments — then this project is for you.

**⭐ Star the repo. 🍴 Fork the code. 🌍 Build with us.**  
**The future of practical, human-aligned AI starts here.**

<div align="center">

[↑ Back to Top ↑](#-adaptshot)

</div>