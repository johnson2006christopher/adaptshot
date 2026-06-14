# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-14

### Production Hardening

This release represents a full production hardening pass over the v0.2.0-dev feature set. Every algorithm was reviewed, strengthened, or replaced for robustness in real-world deployments.

#### Conformal Prediction — True LOO Calibration
- **LOO (Leave-One-Out) mode**: Uses all calibration points by leaving one out per quantile estimate — tighter prediction sets with sparse data.
- **Split mode retained** for large calibration sets (> 100 samples) where LOO overhead is unnecessary.
- Finite-sample correction: \(\lceil (n+1)(1-\alpha) \rceil / n\) ensures valid coverage even with small n.

#### Uncertainty Quantification — Shrinkage Covariance
- **Shrinkage covariance Mahalanobis**: \(\Sigma_{\text{shrunk}} = (1-\lambda)\Sigma_{\text{emp}} + \lambda \cdot \text{diag}(\Sigma_{\text{emp}})\) with automatic \(\lambda\) scaling.
- Robust OOD detection with as few as 2 samples per class (was unreliable below embedding dimension).
- **Adaptive alpha**: OOD threshold converges from loose (fewer false positives) to tight as sample count grows.

#### Contrastive Learning — Gradient-Trained Projection Head
- **W₁, b₁, W₂, b₂ trained via InfoNCE backpropagation** with SGD momentum — no longer random/fixed weights.
- Xavier/Glorot uniform initialization replaces identity-like heuristics.
- Per-epoch InfoNCE loss history accessible for convergence monitoring.

#### ACT — Symmetric Updates with Mean-Reversion
- Threshold updates now include a mean-reversion term: \(\gamma \cdot (\theta_c^{(0)} - \theta_c^{(t)})\).
- Prevents unbounded threshold drift in long-running services.

#### UP-UGF — LSH Acceleration
- Redundancy scoring uses Locality-Sensitive Hashing for \(O(N \log N)\) approximate similarity (was \(O(N^2)\)).
- Random projection hash: \(h(\mathbf{x}) = \text{sign}(\mathbf{w} \cdot \mathbf{x})\).

#### Calibration — Bootstrap Temperature
- Bootstrap resampling (B=100) for temperature estimation when window < 30 samples.
- Median aggregation for robustness against skewed estimates.

#### Explainability — Historical Penalty Tracking
- Per-class penalty history accumulated from corrections.
- Trend detection: "improving", "degrading", "stable" per class.
- Global penalty trend available for production monitoring.

#### Memory Profiling — MemoryTracker
- Section-level memory breakdowns (support_loading, inference, correction).
- `clear_backbone_cache()` for long-running services.
- Budget enforcement with actionable recommendations.

### Added
- **Conformal Prediction**: `ConformalEngine` with split/loo modes, softmax and distance nonconformity scores, class-conditional quantiles, and finite-sample coverage guarantees.
- **Contrastive Prototype Learning**: `ContrastivePrototypeLearner` with gradient-trained InfoNCE projection head, SGD+momentum optimization, and loss history.
- **Multi-Signal Uncertainty**: `UncertaintyQuantifier` with shrinkage covariance Mahalanobis, adaptive OOD threshold, epistemic (MC Dropout), aleatoric (k-NN entropy), and distributional signals.
- **XAI Explainability**: `ExplainabilityEngine` with feature attribution, confidence decomposition, counterfactual analysis, and historical penalty tracking.
- **MemoryTracker**: Section-level memory profiling with peak/current/delta reporting.
- **clear_backbone_cache()**: Reclaim memory in long-running services.
- **ONNX Runtime backend**: Torch-free inference for edge deployment (~800 MB smaller install).
- **New Config Fields**: `conformal_alpha`, `conformal_mode`, `uncertainty_mode`, `explainability_enabled`. `inference_mode` now supports `"contrastive"`.
- **Enhanced PredictionResult**: Now includes `conformal_set`, `uncertainty_report`, `nearest_neighbors`, and `historical_penalties`.
- **Public `explain()` Method**: `FewShotLearner.explain()` returns `ExplanationResult` with attributions, confidence breakdown, and counterfactuals.
- **Documentation**: 42+ markdown files, architecture deep-dive, algorithm theory with full mathematical foundations, API reference, 19 tutorials, 5 guides, migration guide, changelog.
- **Quality Gates**: ruff=0, mypy strict=32 files, pytest=92 passed, benchmark=68%.

### Changed
- Schema version bumped to `"0.2.0"`.
- Default `inference_mode` is now `"prototypical"`.
- Default `conformal_mode` is `"split"` (was hardcoded split-only in dev). Cross-conformal (`"cross"`) available for k-fold averaged quantiles.
- `PredictionResult` fields expanded with v0.2.0 additions.
- ACT engine uses symmetric updates with mean-reversion.
- UP-UGF pruning uses LSH-accelerated redundancy scoring.
- Calibration uses bootstrap temperature estimation for small windows.

### Fixed
- CA-EWC fine-tuning clarified as head-only (~2K parameters); backbone remains frozen.
- Conformal LOO mode corrects quantile computation for sparse calibration data.
- Shrinkage covariance prevents singular matrices in OOD detection with small support sets.
- ACT mean-reversion prevents threshold drift in long-running services.

### Documentation
- All 42+ documentation files updated for v0.2.0 API and hardening changes.
- New pages: Memory Profiling (Tutorial 13), ONNX Deployment (Tutorial 19), Migration Guide v0.1→v0.2.
- Architecture deep-dive: Added hardening architecture changes and data flow diagram.
- Algorithm theory: Added shrinkage covariance math, InfoNCE gradient math, LSH approximation math, symmetric ACT math, bootstrap temperature math.
- Troubleshooting: Expanded with conformal, contrastive, OOD, and memory-specific issues.

---

## [0.1.2] - Unreleased

### Planned
- **Swahili UI Localization**: Gradio dashboard interface fully translated to Swahili, serving Tanzanian and East African users in their primary language
- **Gradio UI Enhancements**: Improved widget layout, accessibility labels, and localization infrastructure to support future language additions
- **Localization Framework**: i18n string extraction and translation pipeline for the Gradio dashboard

> **Note**: French localization is **explicitly excluded** from v0.1.2. The focus is Swahili-first — serving East African communities before expanding to Francophone regions. French remains on the v0.2.0 roadmap.

---

## [0.1.0] - 2026-04-15

### Added
- **Core Inference Engine**: `FewShotLearner` API with `predict()`, `correct()`, `save()`, and `load()` methods.
- **Embedding Extraction**: Frozen ResNet-18 and MobileNetV3-Small backbones with TorchScript-compatible preprocessing.
- **Similarity Search**: CPU-optimized cosine similarity with FAISS-CPU support and NumPy fallback.
- **Calibration**: `CalibrationEngine` implementing online temperature scaling, sliding-window ECE tracking, and conformal prediction stub.
- **ACT Engine**: `ACTEngine` for adaptive per-class confidence thresholding based on correction history.
- **Human-in-the-Loop Routing**: `FeedbackRouter` with configurable buffer capacity and fine-tuning trigger thresholds.
- **Continual Learning**: `CAEWCFinetuner` implementing correction-aware elastic weight consolidation with Fisher Information tracking.
- **Memory Management**: `UPUGFPruner` enforcing bounded replay buffers via uncertainty × recency × redundancy scoring.
- **Configuration**: Immutable `AdaptShotConfig` dataclass with validation and deterministic seeding.
- **Utilities**: Determinism verification (`verify_determinism`), safe I/O helpers, and type-safe logging.
- **Benchmarks**: Reproducible smoke test (`run_benchmark.py`) and Day 2–4 integration scripts.
- **UI**: Gradio-based pilot dashboard for image upload, prediction, and human feedback routing.
- **Documentation**: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and this `CHANGELOG.md`.

### Changed
- `extract_embedding` now accepts file paths, PIL images, NumPy arrays, or torch tensors.
- `pyproject.toml` updated to modern PEP 621 standard with optional extras (`faiss`, `ui`, `dev`).
- Benchmark harness refactored to output structured JSON metrics and enforce deterministic seeds.

### Known Limitations
- **UP-UGF Pruning**: Redundancy computation uses exact cosine similarity (`O(N²)`). Efficient for `N ≤ 100` but will be replaced with approximate search in larger buffers.
- **CA-EWC**: Currently operates on classification head only; full backbone fine-tuning requires additional compute and is not recommended for CPU-only deployments.
- **Calibration**: Temperature scaling uses grid search over the sliding window. Gradient-based optimization is planned for future releases.
- **Gradio UI**: Assumes local file paths; remote/cloud storage integration requires custom callbacks.
- **Hardware**: All benchmarks target standard x86_64 CPUs. ARM/Raspberry Pi performance may vary and requires manual latency profiling.

### Security
- Local-only processing by design; no cloud uploads or telemetry in v0.1.0.
- API tokens for PyPI publishing must be managed via environment variables or `.pypirc`.

### Acknowledgments
- Built by Johnson Christopher Hassan with community testing and feedback.
- Architecture inspired by few-shot learning literature, continual learning best practices, and open-source ML engineering standards.
