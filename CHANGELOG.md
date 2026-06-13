# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0-dev] - Unreleased

### Added
- **Conformal Prediction Engine** (`conformal.py`): Distribution-free conformal prediction with split and cross modes,
  softmax/distance nonconformity scores, and adaptive prediction sets at configurable significance levels.
- **Contrastive Prototype Networks** (`contrastive.py`): Siamese-style contrastive loss with InfoNCE, learnable
  temperature, 2-layer MLP projection head (128-dim bottleneck), and EMA momentum prototype updates.
- **Advanced Uncertainty Quantification** (`uncertainty.py`): Multi-signal uncertainty with epistemic (MC Dropout),
  aleatoric (k-NN entropy), and distributional (Mahalanobis distance) signals; OOD detection via class-conditional
  Gaussian distributions.
- **XAI Explainability** (`explain.py`): Gradient-based saliency, feature attribution (top-k neighbor influence),
  confidence decomposition, and counterfactual explanation support.
- **New Config Fields**: `conformal_alpha`, `conformal_mode`, `uncertainty_mode`, `explainability_enabled`
  (26 total fields, up from 22).
- **New inference mode**: `inference_mode="contrastive"` for contrastive prototype-based classification.
- **37 new tests** across 4 test files: `test_conformal.py` (12), `test_contrastive.py` (7),
  `test_uncertainty.py` (10), `test_explain.py` (8).
- **12 new documentation pages**: Architecture deep-dive, algorithm theory, full API reference,
  5 advanced tutorials (conformal prediction, uncertainty, explainability, contrastive learning,
  end-to-end workflow), 2 comprehensive GUI guides (Studio, Pilot Dashboard).

### Changed
- Schema version bumped to `0.2.0` with backwards-compatible migration.
- Package version updated to `0.2.0-dev` in `pyproject.toml` and `__init__.py`.
- `FewShotLearner` now accepts `inference_mode="contrastive"` and wires new engines (Conformal, Contrastive,
  Uncertainty, Explainability).
- `PredictionResult` extended with conformal prediction sets, uncertainty reports, and explanation results.
- Default `inference_mode` changed to `"prototypical"`.

### Fixed
- `np.unique` unpacking bug in uncertainty module (single return value incorrectly destructured).
- Mypy strict-mode compliance across all 31 source files.
- Pre-existing test failures from schema version and inference_mode API mismatches.
- **Contrastive inference wired**: `predict()` now correctly routes to contrastive nearest-prototype when `inference_mode="contrastive"` (was silently falling through to nearest-neighbor).
- **Epistemic uncertainty implemented**: Replaced unimplemented MC Dropout claim with working embedding perturbation sensitivity proxy (`estimate_epistemic()`).
- **Uncertainty mode gating**: `uncertainty_mode` config field now gates signal computation in `UncertaintyQuantifier.quantify()`, avoiding wasted compute.
- **Cross-conformal mode**: Implemented k-fold cross-conformal quantile averaging in `ConformalEngine` when `conformal_mode="cross"`.
- **OOD detection unified**: `predict()` now uses Mahalanobis-based OOD detection via `UncertaintyQuantifier.is_ood()` as the primary path instead of the legacy distance-threshold method.
- **Confidence decomposition clarified**: Simplified math in `decompose_confidence()` to `calibrated + penalties`, eliminating confusing intermediate calculations.
- **Documentation accuracy**: Replaced "gradient-based saliency" claims with honest "embedding-space saliency" language; updated epistemic uncertainty description from MC Dropout to perturbation sensitivity.
- **Config default fixed**: `uncertainty_mode` default changed from `"entropy"` to `"ensemble"` (now consistent with README).

---

## [0.1.2] - 2026-06-08

### Added
- **Lazy torch imports**: `extractor.py` uses deferred imports for PyTorch and torchvision,
  keeping the module importable without a hard torch dependency at install time.
- **ONNX Runtime backend** (`backends/onnx_backend.py`): Lightweight feature extraction
  via bundled ONNX backbone models when torch is not installed.
- **Backend abstraction layer** (`backends/__init__.py`): Unified interface for ONNX Runtime
  and PyTorch backends with auto-detection.
- **ONNX export script** (`scripts/export_backbones.py`): Generates pre-trained backbone
  ONNX models for torch-free inference.
- **Optional `[torch]` extra**: PyTorch and torchvision moved to optional dependencies;
  core library requires only numpy + Pillow.
- **Package data support**: `.onnx` model files bundled via `[tool.setuptools.package-data]`.

### Changed
- **Pretrained backbone weights**: Changed from `weights=None` (random) to
  `weights="IMAGENET1K_V1"` — essential for the ImageNet-normalized preprocessing pipeline
  and for producing meaningful few-shot embeddings.
- **Calibration engine**: Replaced `torch.nn.Parameter(torch.tensor(...))` with a plain
  `float` for the temperature parameter; no autograd needed for grid-search calibration.
- **Config validation**: Lazy `import torch` for CUDA availability check in `AdaptShotConfig`;
  graceful warning when torch is not installed.
- **Fine-tuning module**: Conditional torch import with `_TORCH_AVAILABLE` guard;
  `CAEWCFinetuner` raises a clear `ImportError` message when torch is missing.
- **PIL API**: Uses `Image.Resampling.BILINEAR` via `getattr` lookup for cross-version compatibility.
- **Version bump**: `__version__` updated to `"0.1.2"` in both `pyproject.toml` and `__init__.py`.

### Fixed
- **Installation performance**: Core dependencies reduced from 4 (torch, torchvision, numpy, Pillow)
  to 2 (numpy, Pillow). PyTorch is now optional via `pip install "adaptshot[torch]"`.
- **Backbone consistency**: All backbones now use pre-trained ImageNet weights, matching the
  preprocessing pipeline expectations.

### Planned for v0.1.2 release
- **Swahili UI Localization**: Gradio dashboard interface fully translated to Swahili,
  serving Tanzanian and East African users in their primary language.
- **Gradio UI Enhancements**: Improved widget layout, accessibility labels, and
  localization infrastructure.
- **Localization Framework**: i18n string extraction and translation pipeline for
  the Gradio dashboard.

---

## [0.1.1] - 2026-06-06

### Added
- **Eco Mode & Energy Profiling**: `eco_mode` and `early_exit_threshold` in `AdaptShotConfig` reduce carbon footprint by up to 40%
- **EmbeddingCache**: Instance-scoped cache class preventing cross-learner embedding contamination in multi-model workflows
- **Dynamic Dimension Inference**: `BACKBONE_OUTPUT_DIM` dictionary maps backbone to output dims; auto-detected from support set when populated
- **OOD Detection**: Built-in out-of-distribution detection with configurable `ood_threshold_quantile` and `ood_absolute_min_distance`
- **String Label Corrections**: `correct()` now accepts human-readable string labels via label index mapping
- **Prototypical Inference**: New `prototypical` inference mode uses class prototypes alongside nearest-neighbor search
- **Comparative Feedback**: `correct_comparative()` method for ordinal-supervision-style human feedback
- **Checkpoint Integrity**: SHA-256 checksums on save/load with atomic file writes and schema migration
- **Calibration Report**: `calibration_report()` method returning ECE, temperature, OOD threshold, and buffer statistics
- **Comprehensive Documentation**: 12-chapter tutorial suite, About page, Studio GUI guide, v0.1.1 docs roadmap gap analysis
- **Logo & Branding**: AdaptShot logo integration in site nav, browser tab, and README

### Changed
- `FewShotLearner.__init__` accepts `AdaptShotConfig` instance (not individual `classes`/`device` kwargs)
- `predicted_label` and `corrected_label` in `Correction` now store integer indices; originals preserved in metadata
- `CalibrationEngine` supports `scaling_binning` method alongside `temperature`
- `BACKBONE_OUTPUT_DIM` constant replaces hardcoded backbone output dimensions
- Embedding extraction now passes instance-scoped `EmbeddingCache` instead of a module-level `_last_embedding` deque
- Schema version bumped to `0.1.1` with `migrate_v0_1_0_to_v0_1_1` backwards-compatible loader

### Fixed
- Duplicate `wait_for_cuda(device)` call in `extract_embedding()` — replaced with single placement
- `EmbeddingCache` moved from module-level `collections.deque` to proper class with instance scope
- Config validation added for `similarity_metric`, `inference_mode`, `calibration_eval_bins`
- `calibration_eval_bins >= ece_n_bins` constraint enforced in post-init
- Empty-string label validation in `_validate_label()`

### Known Limitations
- **UP-UGF Pruning**: Redundancy computation uses exact cosine similarity (`O(N^2)`). Efficient for `N <= 100` but will be replaced with approximate search in larger buffers.
- **CA-EWC**: Currently operates on classification head only; full backbone fine-tuning requires additional compute and is not recommended for CPU-only deployments.
- **Calibration**: Temperature scaling uses grid search over the sliding window. Gradient-based optimization is planned for future releases.
- **Gradio UI**: Assumes local file paths; remote/cloud storage integration requires custom callbacks.
- **Hardware**: All benchmarks target standard x86_64 CPUs. ARM/Raspberry Pi performance may vary and requires manual latency profiling.

### Milestones
- **574 PyPI Downloads**: v0.1.0 reached researchers and practitioners in over 30 countries
- **52 Regression Tests**: Full test suite passing with `pytest tests/ -v`
- **Strict Type Safety**: `mypy src/adaptshot --strict` clean
- **Zero Lint**: `ruff check src/ tests/` clean

### Security
- Local-only processing by design; no cloud uploads or telemetry.
- API tokens for PyPI publishing must be managed via environment variables or `.pypirc`.

---

## [0.1.0] - 2024-05-01

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
