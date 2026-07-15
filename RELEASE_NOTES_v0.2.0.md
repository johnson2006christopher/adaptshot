# AdaptShot v0.2.0 — Production-Hardened Few-Shot Vision

**Human-aligned few-shot vision learning, now with guaranteed uncertainty.**

---

## Release Title

```
AdaptShot v0.2.0: Conformal Prediction, Contrastive Prototypes, and Multi-Signal Uncertainty — Production-Ready Few-Shot Vision
```

---

## Release Notes

| | |
|---|---|
| **Tag** | `v0.2.0` |
| **Date** | June 14, 2026 |
| **Status** | Stable Release |
| **Previous** | [v0.1.1](https://github.com/johnson2006christopher/adaptshot/releases/tag/v0.1.1) |
| **Docs** | [johnson2006christopher.github.io/adaptshot](https://johnson2006christopher.github.io/adaptshot/) |

---

### 🎯 Overview

v0.2.0 is a **production hardening release** — every core algorithm from the v0.2.0-dev cycle was reviewed, strengthened, and rigorously tested. This release makes AdaptShot suitable for real-world deployment: from agricultural crop disease detection in East Africa to resource-constrained edge inference anywhere.

**What makes v0.2.0 production-ready:**

- **92 regression tests** covering every public API
- **Strict mypy type-checking** across 32 source files (zero errors)
- **Ruff linting** (zero violations)
- **68% benchmark accuracy** on miniImageNet 5-way/5-shot (CPU-only)
- **42+ documentation pages**: architecture deep-dives, algorithm theory with full mathematical foundations, API reference, 19 tutorials, 5 guides

---

### ✨ Major New Features

#### 🔒 Conformal Prediction — True Leave-One-Out Calibration
Distribution-free prediction sets with **finite-sample coverage guarantees**. True LOO self-calibration runs automatically at `load_support_images()` time — no config flag needed. Split and cross-conformal modes for large calibration sets.

```python
result = learner.predict("query.jpg")
print(result.conformal_set)  # {"healthy", "blight"} — guaranteed 95% coverage
```

#### 🎯 Contrastive Prototype Learning — Gradient-Trained Projection Head
Prototypes are no longer just class means. A 2-layer MLP projection head (`W₁, b₁, W₂, b₂`) is **gradient-trained via InfoNCE loss** with SGD momentum. Xavier/Glorot uniform initialization. Full backpropagation through all parameters.

#### 📊 Multi-Signal Uncertainty Quantification
Three complementary uncertainty signals fused into a single report:
- **Epistemic**: Stochastic embedding perturbation sensitivity
- **Aleatoric**: k-NN entropy in embedding space  
- **Distributional**: Shrinkage-regularized Mahalanobis OOD detection

Adaptive OOD thresholding converges from loose (fewer false positives) to tight as support samples accumulate. Robust with as few as **2 samples per class**.

#### 🧠 XAI Explainability Engine
Full-featured explainability module with:
- **Feature attribution**: Which support examples influenced the prediction most
- **Confidence decomposition**: How each signal contributed to the final confidence
- **Counterfactual analysis**: "What would change the prediction?"
- **Historical penalty tracking**: Per-class trend detection ("improving", "degrading", "stable")

#### ⚡ UP-UGF LSH Acceleration
Redundancy scoring upgraded from \(O(N^2)\) exact cosine to \(O(N \log N)\) via random projection locality-sensitive hashing — viable for buffers with hundreds of examples.

---

### 🔧 Algorithm Hardening (from v0.2.0-dev)

| Component | Hardening |
|-----------|-----------|
| **Conformal** | LOO mode corrects quantile computation for sparse calibration data; finite-sample correction: \(\lceil (n+1)(1-\alpha) \rceil / n\) |
| **Mahalanobis OOD** | Shrinkage covariance: \(\Sigma_{\text{shrunk}} = (1-\lambda)\Sigma_{\text{emp}} + \lambda \cdot \text{diag}(\Sigma_{\text{emp}})\) with automatic \(\lambda\) scaling |
| **Contrastive** | W₁/b₁/W₂/b₂ trained via InfoNCE backpropagation (was random/fixed weights); per-epoch loss history |
| **ACT** | Symmetric bounded formula with mean-reversion: \(\gamma \cdot (\theta_c^{(0)} - \theta_c^{(t)})\) prevents threshold drift |
| **Calibration** | Bootstrap temperature estimation (B=100 resamples) for cold-start scenarios (window < 30 samples) |
| **UP-UGF** | LSH-accelerated approximate similarity via \(h(\mathbf{x}) = \text{sign}(\mathbf{w} \cdot \mathbf{x})\) |
| **Explain** | Historical penalty tracking with per-class trend detection and global penalty monitoring |
| **Memory** | `MemoryTracker` with section-level breakdowns, budget enforcement, and `clear_backbone_cache()` |

---

### 📦 Installation

```bash
pip install adaptshot==0.2.0

# Optional extras
pip install "adaptshot[torch]"    # Training & fine-tuning
pip install "adaptshot[faiss]"    # FAISS acceleration (>100 support images)
pip install "adaptshot[ui]"       # Gradio pilot dashboard
pip install "adaptshot[gui]"      # Studio workspace
pip install "adaptshot[dev]"      # Development tools
```

---

### 🚀 Quick Start

```python
from adaptshot import FewShotLearner
from adaptshot.config.settings import AdaptShotConfig

config = AdaptShotConfig(backbone="resnet18", device="cpu")
learner = FewShotLearner(config=config)

learner.load_support_images(
    ["healthy.jpg", "blighted.jpg"], 
    ["healthy", "blight"]
)

result = learner.predict("query.jpg")
print(result.prediction)            # "healthy"
print(result.calibrated_confidence) # 0.87
print(result.conformal_set)         # {"healthy", "blight"}
print(result.uncertainty_flag)      # False

# Human feedback loop
if result.uncertainty_flag:
    learner.correct("query.jpg", correct_label="blight")
```

---

### ⚙️ New Configuration Fields (27 total)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `conformal_alpha` | `float` | `0.05` | Significance level for prediction sets |
| `conformal_mode` | `Literal["split","cross"]` | `"split"` | Split or k-fold cross-conformal |
| `uncertainty_mode` | `Literal["mcdropout","entropy","mahalanobis","ensemble"]` | `"ensemble"` | Uncertainty quantification mode |
| `explainability_enabled` | `bool` | `True` | Enable XAI explainability |

`inference_mode` now supports `"contrastive"` in addition to `"nearest_neighbor"` and `"prototypical"`.

---

### 🔄 Breaking Changes

| Change | Migration |
|--------|-----------|
| Schema version `"0.2.0"` | Models saved with v0.1.x will auto-migrate on load |
| `conformal_mode` is `"split"` or `"cross"` (not `"loo"`) | True LOO runs automatically at `load_support_images()` |
| `inference_mode` validation now includes `"contrastive"` | Update custom config validation if applicable |
| `PredictionResult` has 12 fields (expanded) | Check code that destructures results |

> 📖 See the full [Migration Guide (v0.1 → v0.2)](https://johnson2006christopher.github.io/adaptshot/guides/migration-v0.1-to-v0.2/) for detailed steps.

---

### 📚 Documentation

- **[Architecture Deep-Dive](https://johnson2006christopher.github.io/adaptshot/guides/architecture-deep-dive/)**: Complete system design with data flow diagrams
- **[Algorithm Theory](https://johnson2006christopher.github.io/adaptshot/guides/algorithm-theory/)**: Full mathematical foundations for every algorithm
- **[API Reference](https://johnson2006christopher.github.io/adaptshot/api/reference/)**: Every class, method, and data structure
- **[19 Tutorials](https://johnson2006christopher.github.io/adaptshot/tutorials/)**: From beginner to production deployment
- **[Config Reference (27 Fields)](https://johnson2006christopher.github.io/adaptshot/reference/config-reference/)**: Parameter-by-parameter guide

---

### ✅ Quality Gates

```
ruff check src/ tests/     → All checks passed (0 violations)
mypy src/adaptshot --strict → Success (32 files, 0 errors)
pytest tests/ -v           → 92 passed
mkdocs build --strict      → 2.54s, 0 warnings
benchmark smoke test       → 68% accuracy, <250 MB RAM
```

---

### 🙏 Acknowledgments

Built by [Johnson Christopher Hassan](https://github.com/johnson2006christopher) in Mbeya, Tanzania 🇹🇿 — with a laptop and determination.

Architecture inspired by Prototypical Networks (Snell et al., 2017), Matching Networks (Vinyals et al., 2016), Distribution-Free Predictive Inference (Vovk et al.), and SimCLR (Chen et al., 2020).

---

### 🔗 Community

- ⭐ [Star on GitHub](https://github.com/johnson2006christopher/adaptshot)
- 📱 [Join WhatsApp Community](https://chat.whatsapp.com/J6AbrvbjmBc5XXX2fnN6RK)
- 💬 [GitHub Discussions](https://github.com/johnson2006christopher/adaptshot/discussions)
- 🐛 [Report a Bug](https://github.com/johnson2006christopher/adaptshot/issues)

---

*"The best AI doesn't guess confidently. It learns humbly, admits uncertainty, and improves through every human correction."*
