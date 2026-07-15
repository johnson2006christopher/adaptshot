# AdaptShot: Comprehensive Strategic Analysis & Research Prompt

> **Project**: AdaptShot v0.2.0.post0 — Human-Aligned Few-Shot Vision Learning for Resource-Constrained Environments
> **Creator**: Johnson Christopher Hassan (Self-taught, Mbeya, Tanzania)
> **Repository**: https://github.com/johnson2006christopher/adaptshot
> **License**: MIT
> **Target**: A comprehensive research prompt for Codex Free Plan / LLM-assisted strategic analysis

---

## 1. Research-Based Assessment: Architecture, Capabilities & Technical Foundations

### 1.1 Core Architecture Overview

AdaptShot is architecturally organized as a pipeline of composable stages, orchestrated by the `FewShotLearner` class (`src/adaptshot/core/learner.py`, ~1644 lines). The architecture follows a **modular, numpy-first design** with lazy-loading of optional PyTorch dependencies:

**Pipeline Flow:**
```
Image Input → Feature Extractor → Embedding → Similarity Search → Calibration
    → ACT Gating → Conformal Prediction → Uncertainty Quantification
    → Explainability → Prediction Result
```

**Module Layout:**
| Module | File | Lines | Responsibility |
|--------|------|-------|---------------|
| Config | `config/settings.py` | 102 | Frozen immutable dataclass `AdaptShotConfig` |
| Extractor | `core/extractor.py` | 274 | Frozen backbone embedding (ResNet-18, MobileNetV3_Small) |
| Similarity | `core/similarity.py` | 241 | Cosine/Euclidean, FAISS-optional nearest neighbor/prototype |
| Calibration | `core/calibration.py` | 305 | Online temperature scaling, ECE, debiased ECE, scaling-binning |
| ACT | `core/act.py` | 153 | Adaptive Confidence Thresholding with mean reversion |
| Conformal | `core/conformal.py` | 437 | Split/cross-conformal prediction sets |
| Contrastive | `core/contrastive.py` | 513 | InfoNCE-trained 2-layer MLP projection head + prototype refinement |
| Uncertainty | `core/uncertainty.py` | 570 | Epistemic (perturbation) + Aleatoric (k-NN entropy) + Distributional (Mahalanobis) |
| Explainability | `core/explain.py` | 587 | Feature attribution, confidence decomposition, counterfactuals, saliency |
| Feedback Router | `training/feedback_router.py` | 140 | Human correction ingestion and buffer management |
| CA-EWC Finetune | `training/finetune.py` | 210 | Head-only Fisher-regularized continual learning |
| UP-UGF Pruner | `training/up_ugf.py` | 161 | Uncertainty-Guided Forgetting with LSH-accelerated redundancy |
| Determinism | `utils/determinism.py` | 89 | Seed management, reproducibility verification |
| Profiling | `utils/profiling.py` | 167 | Memory tracking (tracemalloc + psutil) |
| Studio GUI | `studio/app.py` | 1029 | Gradio-based offline workspace |
| Studio Utils | `studio/utils.py` | 795 | Export, session management, ONNX/TorchScript deployment |

**Total codebase**: ~6,500 lines of Python across ~30 source files.

### 1.2 Design Philosophy & Constraints

The project is governed by a constitution (`src/.openproject.md`) that enforces:

- **CPU-First**: All operations default to `device="cpu"`. CUDA is opt-in. The learner's `__init__` raises `ConfigValidationError` if device is not "cpu".
- **<250MB RAM**: Verified via `MemoryTracker` with tracemalloc instrumentation. Memory budget is hard-coded (250MB default).
- **Deterministic**: Strict seed management via `set_deterministic_seed()` + SHA-256 integrity checks on checkpoints.
- **Offline-First**: No cloud dependencies, no telemetry, no authentication services.
- **Human-in-the-Loop**: Corrections routed through `FeedbackRouter` → triggers CA-EWC fine-tuning.
- **Carbon-Conscious**: Eco-mode with early-exit thresholds, CO2 estimation in benchmarks.

### 1.3 Key Technical Achievements

1. **Optional PyTorch**: The core import path (`__init__.py` → `learner.py`) is importable without PyTorch. All torch imports use lazy-loading through module-level `_get_torch()` helpers. Only `extract_embedding()` and `CAEWCFinetuner` require torch at runtime.

2. **Online Calibration**: Temperature scaling operates on a rolling window of prediction-confidence pairs, refitting via grid search on each `update()` call. The bootstrap temperature calibration (LOO grid search on support set) enables calibrated predictions from the very first `predict()` call.

3. **Conformal Prediction with TRUE LOO**: The `_self_calibrate_conformal()` method correctly recomputes leave-one-out prototypes (excluding the current example) for valid finite-sample coverage guarantees under exchangeability — a sophisticated statistical implementation.

4. **Contrastive Projection Training**: The `ContrastivePrototypeLearner` implements full backpropagation through a 2-layer MLP (W1/b1 → ReLU → W2/b2) using manually derived InfoNCE gradients in pure numpy — a non-trivial algorithmic achievement.

5. **Multi-Signal Uncertainty**: Three complementary signals (epistemic/aleatoric/distributional) fused with mode-gated computation, operating entirely in numpy-space.

6. **UP-UGF Buffer Management**: Composite multiplicative scoring (uncertainty × recency × redundancy) with O(N log N) LSH-accelerated redundancy computation for >100 examples.

### 1.4 Documentation Quality

- **MkDocs-based** with 12+ documentation pages, 5 tutorials, architecture deep-dives, algorithm theory, and API reference
- **Notable**: Tutorials cover continual learning, production deployment, UI pilot dashboard, studio guide, ONNX deployment, confidence calibration, contrastive learning
- **Benchmark references** include published baselines (Prototypical Networks, Matching Networks, MAML) with honest disclaimers about non-comparability
- **Documented limitations**: MC Dropout as "planned for future torch-dependent release", gradient-based saliency as "requires torch", CA-EWC scope explicitly described as head-only (~2K params)

### 1.5 Testing Infrastructure

- **92 tests** across 12 test files covering calibration, conformal, contrastive, uncertainty, explainability, similarity, exceptions, persistence, feedback routing, learner integration, release metadata, and studio utils
- **CI/CD**: GitHub Actions workflow with mypy strict mode, ruff linting, pytest
- **Quality gates**: `ruff check`, `mypy --strict`, `pytest`, benchmark smoke test

---

## 2. Impact Evaluation

### 2.1 Current Impact Assessment

**Real-World Applicability**
- Target domains: Rural healthcare (clinic diagnostics), agriculture (crop disease detection), edge AI (Raspberry Pi, legacy hardware)
- Verified use case: Crop disease demo (`examples/crop_disease_demo.py`)
- Hardware requirement: Any device running Python 3.9+ with ~250MB RAM and ~500MB disk
- Internet requirement: Only for initial backbone weight download (~45MB for ResNet-18)

**Community Metrics**
- PyPI package published with version tracking
- GitHub Pages documentation hosted
- MIT license enables adoption
- However, as v0.2.0-alpha with likely few PyPI downloads, community impact is currently minimal

**Scientific Merit**
- Novel combination of conformal prediction + human-in-the-loop for few-shot vision
- Debiased ECE computation (debiased squared CE → L2 mapping)
- TRUE leave-one-out conformal calibration (not naive self-inclusion)
- Shrinkage-regularized Mahalanobis with adaptive factor α = d / (d + n_k)
- Symmetric ACT update rule with mean reversion

### 2.2 Potential Impact Domains

1. **Global South AI Deployment**: The CPU-first, offline, low-memory design directly addresses the needs of healthcare, agriculture, and education in regions without reliable internet or GPU access.

2. **Trustworthy AI Regulation**: Built-in calibration, conformal guarantees, uncertainty quantification, and explainability align with emerging EU AI Act requirements and similar regulatory frameworks.

3. **Sustainable AI**: Carbon tracking, energy-efficient design, and explicit eco-mode position AdaptShot as an environmentally responsible alternative to the "bigger is better" paradigm.

4. **Human-AI Collaboration**: The human-in-the-loop design (corrections → calibration → fine-tuning) models a realistic deployment pattern where AI assists rather than replaces human judgment.

---

## 3. Focus Areas & Primary Domains

### 3.1 Core Competencies

| Domain | Implementation | Maturity |
|--------|---------------|----------|
| **Few-Shot Classification** | Prototypical networks, nearest-neighbor, contrastive prototypes | Stable (v0.2.0) |
| **Confidence Calibration** | Temperature scaling, scaling-binning, conformal margin | Stable (v0.2.0) |
| **Conformal Prediction** | Split-conformal, cross-conformal, class-conditional | Stable (v0.2.0) |
| **Uncertainty Quantification** | Multi-signal (epistemic + aleatoric + distributional) | Stable (v0.2.0) |
| **Explainable AI** | Feature attribution, confidence decomposition, counterfactuals | Stable (v0.2.0) |
| **Continual Learning** | CA-EWC head-only + UP-UGF buffer management | Stable (v0.2.0) |
| **Human-in-the-Loop** | Feedback router, correction routing, active learning via ACT | Stable (v0.2.0) |
| **Contrastive Learning** | InfoNCE with 2-layer MLP projection head | Stable (v0.2.0) |
| **Model Export** | Native checkpoint, TorchScript, ONNX | Stable (v0.2.0) |
| **Memory Profiling** | tracemalloc + psutil instrumentation | Stable (v0.2.0) |

### 3.2 Secondary/Developing Areas

| Domain | Implementation | Maturity |
|--------|---------------|----------|
| MC Dropout | Numpy perturbation proxy (not true MC Dropout through backbone) | Proxy only |
| Gradient-based Saliency | Embedding-space attribution only (no pixel-level saliency) | Partial |
| Multi-modal Input | Image-only (no text, audio, or video) | Not implemented |
| Meta-Learning | No MAML, Reptile, or meta-learning support | Not implemented |
| Data Augmentation | No built-in augmentation pipeline | Not implemented |
| Model Zoo / Hub | No pre-trained model repository | Not implemented |

---

## 4. Improvement Opportunities

### 4.1 Technical Architecture Improvements

**A. Backbone Extensibility (High Priority)**
- Currently only 2 backbones: ResNet-18 and MobileNetV3_Small
- Opportunity: Implement a plugin/registry pattern (`EmbeddingBackend` protocol) allowing users to register custom backbones (ViT, ConvNeXt, EfficientNet, Swin)
- Opportunity: Support torch.hub backbones, timm library integration
- Files affected: `core/extractor.py` (BackboneRegistry), `config/settings.py`

**B. True MC Dropout via Torch (Medium Priority)**
- Current epistemic uncertainty uses numpy perturbation proxy instead of true MC Dropout
- Opportunity: Implement `MC Dropout` through the frozen backbone using torch's dropout layers. The code explicitly acknowledges this as "planned for a future torch-dependent release."
- Impact: Significantly more principled epistemic uncertainty
- Files affected: `core/uncertainty.py`

**C. Data Augmentation Pipeline (Medium Priority)**
- No built-in augmentations for support/query images
- Opportunity: Add configurable augmentation transforms (random crop, flip, color jitter, cutout) for robustness
- Impact: Improved generalization in low-data regimes

**D. Quantization & Model Compression (High Priority)**
- No support for INT8 quantization, weight pruning, or knowledge distillation
- Opportunity: Add ONNX quantization, dynamic quantization via PyTorch, or weight clustering
- Impact: Further reduced memory footprint and latency

**E. Standardized Serialization (Low Priority)**
- Current checkpoint format uses custom JSON + .npy + .pt files
- Opportunity: Support safetensors format, HuggingFace Hub integration
- Impact: Interoperability with broader ML ecosystem

### 4.2 Algorithmic Enhancements

**A. Full-Backbone Fine-Tuning (Medium Priority)**
- CA-EWC currently operates only on the classification head (~2K params)
- Opportunity: Implement LoRA or Adapter-based fine-tuning for the backbone
- Impact: Better adaptation to domain-specific features while preserving efficiency

**B. Advanced Contrastive Methods (Low Priority)**
- Current: InfoNCE only
- Opportunity: SimCLR, BYOL, SwAV, or SupCon variants
- Impact: Richer representation learning

**C. Meta-Learning Integration (Medium Priority)**
- No meta-learning support despite few-shot focus
- Opportunity: Add MAML, Reptile, or ProtoNet meta-training
- Impact: True few-shot learning (vs. current transfer learning approach)

**D. Conformal Prediction Extensions (Low Priority)**
- Current: Split and cross-conformal
- Opportunity: Add adaptive conformal prediction (ACI), weighted exchangeability, and Mondrian conformal prediction
- Impact: Better handling of distribution shift over time

**E. Active Learning (Medium Priority)**
- ACT engine provides confidence thresholding but not explicit acquisition functions
- Opportunity: Add uncertainty sampling, margin sampling, query-by-committee strategies
- Impact: More efficient human-AI collaboration

### 4.3 Performance & Scalability

**A. Batch Inference (Medium Priority)**
- Current: Single-image `predict()` and `explain()` calls
- Opportunity: Add batch processing for multiple query images (already partially supported in benchmarks)
- Impact: Higher throughput for production deployment

**B. Embedding Cache Optimization (Low Priority)**
- Current: Single embedding cache for eco-mode early exit
- Opportunity: LRU cache for recently seen images, approximate nearest neighbor caching
- Impact: Reduced latency for repeated or similar queries

**C. Benchmark Standardization (High Priority)**
- Current: Only CIFAR-10 and miniImageNet support
- Opportunity: Add CIFAR-FS, FC100, tieredImageNet, meta-dataset benchmarks
- Need: Standardized evaluation protocol with published SOTA comparisons
- Impact: Credibility in the few-shot learning community

### 4.4 Usability & Developer Experience

**A. CLI Tool (Medium Priority)**
- Current: Only `adaptshot-studio` for GUI
- Opportunity: Add CLI commands for training, evaluation, export, and benchmark
- Impact: Developer-friendly tooling

**B. REST API / FastAPI Server (Low Priority)**
- Opportunity: Docker container with FastAPI endpoint for model serving
- Impact: Production deployment readiness

**C. Rich Error Messages & Debugging (Low Priority)**
- Current: Good error handling with custom exception hierarchy
- Opportunity: Add debugging visualizations, embedding projector (TensorBoard), prediction dashboard
- Impact: Faster development cycles

### 4.5 Documentation & Community

**A. Interactive Notebooks (Medium Priority)**
- Current: Only MkDocs tutorials (no Jupyter notebooks)
- Opportunity: Add Google Colab notebooks for all tutorials
- Impact: Lower barrier to entry

**B. API Reference Completeness (Medium Priority)**
- Current: Good inline docstrings but API reference docs are incomplete
- Opportunity: Complete `mkdocstrings` integration for auto-generated API docs
- Impact: Developer productivity

**C. Model Zoo & Pre-trained Weights (High Priority)**
- Current: Only ImageNet-pretrained backbones
- Opportunity: Release pre-trained few-shot models on standard benchmarks
- Impact: Immediate utility for practitioners

**D. Community Contribution Guide (Low Priority)**
- Current: Basic CONTRIBUTING.md
- Opportunity: Add contributor mentorship program, good-first-issue labels, community calls

---

## 5. Competitive Positioning

### 5.1 Competitive Landscape

| Library | Focus | GPU Required? | Calibration? | Uncertainty? | HITL? | Edge? | Carbon-Aware? |
|---------|-------|---------------|-------------|-------------|-------|-------|--------------|
| **AdaptShot** | Few-shot vision | No (CPU-first) | Yes (temperature, ECE, debiased) | Yes (3-signal ensemble) | Yes | Yes | Yes |
| **easyfsl** | Few-shot vision | Yes | No | No | No | No | No |
| **setfit (HF)** | Few-shot text | Optional | No | No | No | No | No |
| **scikit-learn** | General ML | No | Limited | Limited | No | Yes | No |
| **fast.ai** | General DL | Yes | Limited | No | No | No | No |
| **timm** | Vision models | Yes | No | No | No | No | No |
| **PyTorch Lightning** | General DL | Yes | No | No | No | No | No |

### 5.2 Unique Value Propositions

1. **CPU-First Architecture**: No other few-shot vision library is designed from the ground up for CPU-only, <250MB RAM operation. This is AdaptShot's strongest differentiator.

2. **Integrated Uncertainty-Explainability Pipeline**: The combination of conformal prediction + multi-signal uncertainty + XAI in a single library is unique. Most libraries offer at most one of these.

3. **Human-in-the-Loop as First-Class Citizen**: The `FeedbackRouter` / `CA-EWC` / `UP-UGF` pipeline for handling human corrections is not found in any competing library.

4. **Constitution-Driven Development**: The `.openproject.md` governance document enforces principled constraint-first engineering — a novel approach in open-source AI.

5. **Carbon-Conscious by Default**: Energy tracking, eco-mode, and carbon reporting are built into the library's DNA, not added as an afterthought.

### 5.3 Competitive Weaknesses vs. Alternatives

| Dimension | AdaptShot Gap | Competitor Advantage |
|-----------|--------------|---------------------|
| Backbone diversity | 2 backbones only | timm: 300+ models |
| Benchmark SOTA | No published SOTA on standard splits | easyfsl: ProtoNet/MAML on miniImageNet |
| Community size | Solo developer | fast.ai: 100k+ users |
| Model hub | None | HuggingFace: 100k+ models |
| Multi-modal | Image only | SetFit: text, Transformers: text+vision+audio |
| Training speed | CPU-only fine-tuning | Any: GPU-accelerated |
| Documentation | MkDocs | fast.ai: full book + course |
| Ecosystem integration | Standalone | Lightning: Fabric, CLI, Studio |

### 5.4 Key Differentiation Strategy

AdaptShot should NOT compete on benchmarks or model variety. Its competitive moat is:
1. **Resource-constrained deployment** (edge, Global South, IoT)
2. **Trustworthy AI compliance** (calibration + uncertainty + conformal guarantees)
3. **Human-in-the-loop workflow** (not just inference, but continuous improvement through corrections)
4. **Environmental responsibility** (carbon tracking, energy efficiency)

---

## 6. Strategic Roadmap to World-Class AI Library Status

### Phase 1: Foundation Hardening (v0.2.x — 3 months)

**Goal**: Production-ready reliability and developer confidence

- [ ] **Backbone Registry Extension** (`core/extractor.py`): Add plugin protocol allowing user-registered backbones. Support ViT, EfficientNet, ConvNeXt via timm.
- [ ] **Benchmark Standardization**: Add CIFAR-FS, FC100, tieredImageNet support. Publish benchmark results with honest disclosures.
- [ ] **API Polish**: Complete `mkdocstrings` auto-generated API reference. Add parameter validation for all edge cases.
- [ ] **CI/CD Hardening**: Add nightly benchmarks, regression test matrix across Python 3.9-3.12.
- [ ] **Documentation Audit**: Verify all docstrings match actual signatures. Add "Quickstart in 5 minutes" Colab notebook.
- [ ] **Bug Bounty / Fuzzing**: Add hypothesis-based property testing for all core modules.

### Phase 2: Capability Expansion (v0.3.x — 6 months)

**Goal**: Feature completeness for production edge deployment

- [ ] **True MC Dropout** (`core/uncertainty.py`): Replace numpy perturbation proxy with torch-based MC Dropout through backbone.
- [ ] **Data Augmentation Pipeline**: Add configurable transforms (random crop, flip, color jitter, mixup, cutmix).
- [ ] **Active Learning Module**: Implement uncertainty sampling, margin sampling, and query-by-committee for selective human feedback requests.
- [ ] **Batch Inference API**: Add `predict_batch()` and `explain_batch()` methods to `FewShotLearner`.
- [ ] **Model Export Enhancement**: Add safetensors support, INT8 quantization via ONNX.
- [ ] **LoRA / Adapter Fine-Tuning**: Extend CA-EWC to support lightweight backbone adaptation.
- [ ] **CLI Tool**: Add `adaptshot` CLI with `train`, `eval`, `export`, `benchmark` subcommands.

### Phase 3: Ecosystem Building (v0.4.x — 9 months)

**Goal**: Community adoption and ecosystem integration

- [ ] **Model Zoo**: Release pre-trained few-shot models on CIFAR-FS, miniImageNet, and domain-specific datasets (crop disease, medical imaging).
- [ ] **HuggingFace Hub Integration**: Push/pull models, datasets, and benchmarks through HF Hub.
- [ ] **REST API Server**: Dockerized FastAPI endpoint with OpenAPI schema for production serving.
- [ ] **Jupyter Notebook Tutorials**: All MkDocs tutorials also available as runnable Google Colab notebooks.
- [ ] **Community Infrastructure**: Add GitHub Discussions, contributor recognition program, monthly community calls.
- [ ] **Integration Examples**: Add examples for MLOps tools (MLflow, Weights & Biases, TensorBoard).

### Phase 4: Market Leadership (v1.0+ — 12 months)

**Goal**: Established as the go-to library for trustworthy edge AI

- [ ] **Publication**: Submit technical paper to JMLR (Journal of Machine Learning Research) or similar open-access journal.
- [ ] **Real-World Validation**: Partner with 2-3 organizations in Global South for pilot deployments (healthcare, agriculture).
- [ ] **Neuromorphic Backend**: Implement `EmbeddingBackend` protocol for event-based vision hardware.
- [ ] **Multilingual Interface**: Studio GUI and documentation in French, Arabic, Swahili, Hindi.
- [ ] **Enterprise Compliance Package**: Documentation and tooling for EU AI Act, FDA, and HIPAA compliance.
- [ ] **Academic Benchmark Suite**: Standardized few-shot evaluation protocol with leaderboard.
- [ ] **Carbon Certification**: Published methodology for carbon footprint estimation, integrated into ML CO2 impact standards.

### 6.1 Critical Success Factors

| Factor | Current State | Target (v1.0) | Gap |
|--------|--------------|---------------|-----|
| Backbones | 2 | 10+ (via plugin) | Plugin protocol, timm integration |
| Benchmarks | CIFAR-10, miniImageNet | 6 datasets | CIFAR-FS, FC100, tieredImageNet, meta-dataset |
| Tests | 92 | 500+ | Property-based testing, integration tests |
| Documentation | MkDocs | MkDocs + Colab + auto API | Jupyter notebooks |
| Community | Solo | 100+ contributors | Contribution infrastructure |
| ML Ecosystem | Standalone | HF Hub, MLflow | Integration libraries |
| Model Zoo | 0 | 20+ pre-trained | Compute for training |
| Published Research | 0 (no paper) | 1+ peer-reviewed | Writing + peer review |

### 6.2 Resource Requirements

**Immediate Needs (Phase 1-2):**
- GPU time for benchmark evaluation and model training ($500-1000 cloud credits)
- CI/CD runner minutes (GitHub Actions, free tier sufficient for Phase 1)
- Documentation hosting (GitHub Pages, free)

**Growth Needs (Phase 3-4):**
- Community manager (part-time volunteer or funded)
- Cloud compute for model zoo ($200-500/month)
- Legal review for compliance documentation
- Partnership development for real-world pilots

### 6.3 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Low community adoption | High | High | Focus on niche (edge AI, Global South); don't compete on general-purpose |
| Single developer burnout | Medium | Critical | Open contribution pathways; document architecture for handoff |
| PyTorch dependency risk | Low | Medium | Maintain numpy-first design; explore alternative backends (ONNX-only runtime) |
| SOTA benchmark gap | High | Medium | Publish with honest disclosures; emphasize real-world deployment over lab SOTA |
| Compute cost for training | Medium | Medium | Leverage free cloud tiers (Colab, Kaggle, HuggingFace Spaces) |

---

## 7. Critical Technical Debt & Priority Actions

### Immediate (Must Fix Before v0.3.0)

1. **FeedbackRouter FIFO Eviction** (`training/feedback_router.py:112`): The `router.route_feedback()` calls `_update_buffer()` which uses FIFO, NOT UP-UGF scoring. UP-UGF is only applied in `_apply_buffer_management()` which fires separately. This creates a window where corrections are evicted before UP-UGF evaluation.

2. **Bootstrap Calibration Seed Escaping**: `_bootstrap_temperature_calibration()` seeds the calibration window with optimistic priors (all correct). This biases early temperature estimates toward overconfidence.

3. **LSH Redundancy Fallback** (`training/up_ugf.py:99-122`): The LSH-based max similarity approximation for >100 examples has not been validated against exact computation. Ground-truth verification needed.

### Important (Phase 2)

4. **No Warm-Start for Contrastive Head**: Each `load_support_images()` call re-trains the projection head from scratch. For continual learning scenarios, this is wasteful.

5. **Studio Monolith** (`studio/app.py:1029` lines): The Gradio app should be modularized into tabs/components.

6. **Missing Determinism Verification for Uncertainty** (`core/uncertainty.py:351`): When `seed=None`, perturbation-based epistemic uncertainty is non-deterministic, violating the library's determinism guarantee for the ensemble mode.

### Nice-to-Have (Phase 3+)

7. **ONNX Backend Performance**: The ONNX backend (`core/backends/onnx_backend.py:95` lines) needs latency benchmarks vs. PyTorch.

8. **Cross-Validation in Self-Calibration**: LOO conformal calibration is O(N^2) with prototype recomputation. Could be optimized with cached prototypes.

---

## 8. Prompt Instructions for Codex Free Plan Analysis

When submitting this analysis to Codex Free Plan or similar LLM-based code analysis service, include the following context:

---

```
You are analyzing AdaptShot, a human-aligned few-shot vision learning library. Below is the comprehensive project context for your analysis. Please provide:

1. A technical critique of the architecture with specific file/line references
2. Identify the top 3 architectural vulnerabilities and propose concrete fixes
3. Evaluate the competitive positioning and suggest a 3-point differentiation strategy
4. Design a 6-month development roadmap with prioritized milestones
5. Identify any statistical, numerical, or algorithmic errors in the implementation
6. Suggest specific metrics, benchmarks, or experiments to validate the library's claims
7. Provide code quality assessment (type safety, test coverage, error handling)
8. Assess the feasibility of the <250MB RAM claim and suggest memory optimizations

Focus especially on:
- The conformal prediction leave-one-out implementation (core/conformal.py)
- The contrastive prototype InfoNCE gradient derivation (core/contrastive.py)
- The shrinkage-regularized Mahalanobis distance (core/uncertainty.py)
- The ACT symmetric update rule with mean reversion (core/act.py)
- The CA-EWC Fisher-regularized fine-tuning (training/finetune.py)

Base your analysis on real code paths, not assumptions. Output concrete, actionable recommendations with specific file paths and line numbers where possible.

Project structure:
- ~6,500 lines of Python across 30 source files
- 92 regression tests across 12 test files
- Core dependencies: numpy, Pillow (~15MB)
- Optional: torch, torchvision, faiss-cpu, gradio
- Python 3.9+, CPU-first, <250MB RAM target
- MIT licensed, solo-developed by a self-taught engineer in Tanzania
```
