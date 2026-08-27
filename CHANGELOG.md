# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **BREAKING: minimum supported Python is now 3.10** (was 3.9). Python 3.13 and 3.14 are
  now supported and tested. The CI matrix covers 3.10 through 3.14; every version listed
  in the trove classifiers is a version that actually runs the test suite, enforced by
  `tests/test_release_metadata.py`.

  If you are on Python 3.9, stay on 0.2.x. Dropping 3.9 lets the codebase use native
  `X | None` and builtin generics rather than `typing.Optional` and `typing.Dict`, which
  removed 492 lint findings in one pass.

- **Development tool versions are now upper-bounded.** `ruff`, `mypy`, `pytest`,
  `pytest-cov`, and `pre-commit` previously had lower bounds only. A ruff release that
  expanded its default rule set from roughly 60 rules to 413 turned CI red with 479
  findings without a single line of project code changing. Tool bumps now arrive as
  reviewable Dependabot PRs.

- **The ruff rule set is now declared explicitly** in `[tool.ruff.lint]` rather than
  inherited from whatever the installed version happens to default to.

### Fixed

- `utils/profiling.py` silently swallowed every exception from `psutil`, reporting
  `0.0 MB` when a memory measurement failed. Callers could not distinguish a real
  zero reading from a failed one — which would have made a `<250MB` ceiling assertion
  pass for the wrong reason. Failures are now logged.
- `core/backends/onnx_backend.py` used `functools.lru_cache` on a method, which keys on
  `self` and would have retained every backend instance, along with up to four loaded
  ONNX sessions and their weights, for the lifetime of the process. Replaced with a
  per-instance cache that is released with the instance.

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
- **True Leave-One-Out Conformal Calibration**: Per-example prototype recomputation for valid
  finite-sample coverage guarantees under exchangeability.
- **Shrinkage Covariance Estimation**: Ledoit-Wolf-style shrinkage with adaptive alpha = d/(d+n_k)
  for robust Mahalanobis OOD detection in high-dimensional few-shot settings.
- **Bootstrap Temperature Calibration**: LOO grid-search temperature optimization for
  autonomous operation without requiring pre-calibrated temperature.
- **Random Projection LSH for UP-UGF**: Approximate O(N log N) redundancy scoring via
  random projection locality-sensitive hashing when buffer exceeds 100 examples.
- **Memory Profiling** (`utils/profiling.py`): `MemoryTracker` context manager with
  tracemalloc + psutil instrumentation; `estimate_model_memory_mb()` for pre-flight checks.
- **ONNX Export Script** (`scripts/export_backbones.py`): Exports ResNet-18 and
  MobileNetV3-Small to ONNX with SHA-256 verification and metadata generation.
- **miniImageNet Benchmark Support**: CSV-based miniImageNet loading, `BASELINE_REFERENCES`
  for Prototypical/Matching/MAML baselines, and `--full-benchmark` CLI flag.
- **Historical Penalty Tracking**: `ExplainabilityEngine` tracks ACT and OOD penalties
  over time for intelligent confidence decomposition fallbacks (replaces magic numbers).
- **Eco-Mode Enhancements**: 32×32 preview resolution (up from 16×16), `clear_backbone_cache()`
  for `@lru_cache` invalidation on config change, norm ratio eco-mode safety guard.

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
- **Contrastive projection head training**: `_train_projection_head()` now performs full InfoNCE gradient descent through W1/b1/W2/b2 with momentum SGD (was previously initialized but never trained, making the projection head an identity transform).
- **Conformal LOO calibration**: `_self_calibrate_conformal()` recomputes prototypes excluding each support example for true leave-one-out nonconformity scores (was reusing full-support prototypes, invalidating coverage guarantees).
- **Mahalanobis shrinkage**: `fit_class_distributions()` uses shrinkage covariance estimation with adaptive alpha, falling back to diagonal when n_per_class < embedding_dim (was using raw sample covariance, which is singular in few-shot high-dim settings).
- **CA-EWC scope honesty**: `CAEWCFinetuner` docstring now explicitly states head-only scope (~2K params for 5-way ResNet-18), not full-network EWC.
- **ACT symmetric update**: Threshold delta replaced with `η * (incorrect_rate − correct_rate)` plus mean-reversion toward base threshold, eliminating monotonic drift.
- **Confidence decomposition fallbacks**: Replaced magic numbers `-0.15`/`-0.25` with historical 20-window averages of tracked ACT penalties.
- **UP-UGF LSH mode**: `_compute_redundancy_scores()` splits into exact (N≤100) and approximate LSH (N>100) paths, reducing O(N²) to O(N log N) for large buffers.
- **Graceful calibration fallback**: `_calibrate_or_raise()` no longer raises on first predict; uses bootstrap temperature calibration when conformal buffer is cold.
- **Eco-mode resolution**: Preview upgraded from 16×16 to 32×32 with norm ratio guard (>0.3 required before early-exit gating).
- **Config default fixed**: `uncertainty_mode` default changed from `"entropy"` to `"ensemble"` (now consistent with README).
- **Conformal calibration wired**: Self-calibration on `load_support_images()` populates calibration buffer via leave-one-out scores; `correct()` feeds ground-truth nonconformity scores into the conformal engine. Prediction sets now produce meaningful multi-class outputs instead of degenerate singletons.
- **Torch lazy imports in learner.py**: Moved `import torch`, `DataLoader`, `TensorDataset` out of module level into lazy getters (`_get_torch()`, `_get_torch_nn()`, `_get_data_loader()`). `FewShotLearner` is now importable without a hard torch dependency — PyTorch is truly optional.
- **Contrastive mode shape mismatch fixed**: Contrastive prototypes (128-dim projection space) now stored in separate `_contrastive_prototype_*` fields; embedding-space prototypes (`_prototype_embeddings`) always remain 512-dim for conformal/OOD distance math. Eliminates the 512-vs-128 dimension mismatch in distance computations.
- **ACTEngine dynamic class allocation**: Changed from `n_classes=200` to `n_classes=max(10, config.n_way)`; dynamic expansion handles additional classes at runtime.
- **`compute_saliency_numpy()` implemented**: Returns per-dimension embedding-space feature importance via `|query - support|` normalized to [0,1]. No longer returns `None`.
- **Epistemic uncertainty stochastic**: `estimate_epistemic()` seed default changed from `42` to `None` — each call produces a genuinely different perturbation pattern, capturing stochastic sensitivity.
- **Confidence decomposition penalties derived from state**: ACT penalty now proportional to (confidence - threshold) gap when threshold available; OOD penalty proportional to Mahalanobis OOD score. Falls back to conservative defaults when state unavailable.

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
