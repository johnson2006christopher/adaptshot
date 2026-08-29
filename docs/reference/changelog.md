# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

> Everything below is the 0.3.0 release ("make it provable"). The version in
> `pyproject.toml` is bumped at release time, which is the maintainer's step.

### Added — measured results

- **The first real result.** PlantVillage 5-way 5-shot, 100 episodes, seed 42:
  91.4% ± 1.0 accuracy, against nearest-centroid, k-NN and a linear-probe
  baseline on the same episodes and embeddings. AdaptShot's accuracy *is*
  nearest-centroid's, to the query; what the layers above buy is the
  prediction set. (`benchmarks/run_plantvillage.py`, #18, #19)
- **Conformal coverage, measured**: 98.1% ± 0.6 at a 90% target, mean set size
  1.66, against a calibrated top-1 threshold that reached 83.9%. (#14, #86)
- **Coverage under distribution shift**: blur, brightness, JPEG and downscale
  applied to queries only; the sets widen, the bound still bends (85.5% at
  blur σ=4), the OOD flag correlates 0.92 with the loss, and ten in-situ
  corrections recover the worst cell to 89%. (`benchmarks/run_shift.py`, #29)
- **Latency by stage** (median and p95), cold start, and peak memory for one
  cycle *and* for the harness, named apart, with the CPU model recorded. 120 MB
  for one fresh process on the core install. (#20)
- **ONNX parity benchmark**, each backend in its own process. (#36)
- Every figure in the README, the technical note and the docs is formatted from
  a committed `results/*.json` and asserted by `tests/test_docs_claims.py`.

### Added — features

- **Torch-free inference.** `mobilenet_v3_small` ships as ONNX inside the wheel
  (4.0 MB); `pip install adaptshot` is numpy, Pillow and onnxruntime only, and
  loads a support set, predicts, saves and reloads. Torch is needed only for
  fine-tuning and non-bundled backbones. The default backbone changed to the
  bundled one. (#35, #36)
- **`check_environment()`**: what this machine can do, with every figure measured
  on it; a GPU is named and never selected. Experimental. (#38)
- **Twelve real PlantVillage photographs ship in the wheel** (`adaptshot.data.sample_images`)
  so the README quickstart runs offline in seven lines; four more for the demo.
  Licence and checksums beside them. (#28)
- **`examples/demo/`**: a conference demo that disables its own network access,
  runs in under two seconds, and shows the set widening and the refusal. (#27)
- **`BackboneError`** names the backbone, the ones that would work and the extra
  that installs torch, instead of `ImportError: torch` from four frames deep.
- `PredictionResult.conformal_calibrated` and `ConformalPredictionSet.calibrated`:
  a cold-start singleton says so instead of claiming 1 − α. (#80)
- `adaptshot.api`: every public name classified **stable** (24) or
  **experimental** (10), enforced by `tests/test_api_surface.py`. (#23)
- `bundled_onnx_backbones()`, `ConformalEngine.min_informative_size`,
  `ConformalEngine.nonconformity()`.

### Changed

- **Conformal nonconformity score** defaults to the distance ratio `d_true/min(d)`.
  The max-scaled softmax scored clean, blurred and foreign-crop photographs
  0.72–0.80 alike and could not widen a set. `softmax` and `distance` remain
  selectable. Published figures moved: coverage 97.5% → 98.1%, set size
  2.05 → 1.66. Deliberate exception to the stability policy, recorded. (#86)
- `FewShotLearner` gives its conformal engine a floor of `max(10, ⌈(1−α)/α⌉)`
  calibration scores. (#14)
- `adaptshot.core.contrastive` moved to `adaptshot.training.contrastive`; the
  old path warns and is removed in 0.4.0. (#23)
- `ACTEngine` and `UPUGFPruner` are stable — they have tests now. (#74)
- `__version__` is read from package metadata; `pyproject.toml` is the only
  declaration. (#25)
- numpy annotations are `FloatArray` / `LabelArray` / `IntArray` / `BoolArray`
  (`adaptshot.utils.arrays`); 157 `type-arg` errors under numpy 2.2 → 0. (#44)
- The ruff ignore list is empty; `scripts/` is linted. (#41)
- Documentation rebuilt on Diátaxis: six beginner tutorials, eleven how-to
  guides, explanation, reference and contributor sections; every tutorial and
  how-to code block is executed by `tests/test_docs_tutorials_run.py`; every
  page has an edit link; broken links fail the build. Superseded pages retired
  to `docs-archive/`. (#39)

### Fixed

- **Conformal quantile clamp**: where `n < (1−α)/α` the engine returned the
  largest observed score instead of the full set, under-covering at 91.3%
  against a 95% promise at the library's own defaults. (#14)
- **OOD threshold calibrated in-sample** flagged 45 of 45 in-distribution
  photographs; leave-one-out calibration brings it to 3 of 45, with 45 of 45
  flagged on genuinely out-of-domain photographs. (#54)
- **`UPUGFPruner` kept the confident examples** (score `(1−u)^w`, the inverse of
  its documentation) and, above 100 rows, **rewarded duplicates** (inverted LSH
  collision term). (#74)
- **`_project` guarded one of four `Optional` fields**; a partially restored
  contrastive head failed inside a matmul. (#44)
- The README quickstart pinned a non-bundled backbone and referenced files that
  did not exist; it now runs, and CI runs it. (#28)
- Tambua's `maize.yaml` named `resnet18`, which a standard install cannot run;
  it names the bundled backbone.

### Deprecated

- `adaptshot.core.contrastive` (import path) — removed in 0.4.0.
- `UncertaintyQuantifier.compute_perturbation_variance`, `get_ood_summary`,
  `get_class_statistics` — no callers anywhere; removed in 0.4.0. (#23)

### Infrastructure

- Releases on `v*` tags with PyPI Trusted Publishing, a clean-container install
  test against the built wheel, TestPyPI for `rc` tags. (#25)
- `offline-wheel` CI job: the wheel installed into a clean venv and the
  inference path run inside a network namespace with no interfaces; any network
  access fails the build. (#30)
- The CI test matrix installs CPU-only torch; the CUDA caches had blown past the
  10 GB cache limit and evicted the CIFAR-10 cache, turning a two-minute
  benchmark job into a thirty-five-minute download. The `test-core` job is
  enforcing. `mkdocs build --strict` on every pull request. (#60, #36)

### Earlier notes in this cycle

### Removed

- **`adaptshot.studio`** (1,822 lines, 23% of the library, four tests) — extracted to
  its own project. It was a Gradio desktop application, not few-shot learning and not a
  library, and it dragged `gradio`, `pandas`, `onnx` and `onnxruntime` into the
  project's identity and its type checking. A GUI also has a different release cadence
  from a library; coupling them made every interface tweak a library release.
  Tracked in [#21](https://github.com/johnson2006christopher/adaptshot/issues/21).

  **Nothing was lost.** The full commit history was extracted before anything was
  deleted and lives on the
  [`studio-extract`](https://github.com/johnson2006christopher/adaptshot/tree/studio-extract)
  branch — thirteen commits, ready to become a repository of its own.

- **`adaptshot.ui.app`** (151 lines) — the library shipped *two* Gradio interfaces at
  once. All four of its capabilities exist in Tambua, in more complete form. Tracked in
  [#22](https://github.com/johnson2006christopher/adaptshot/issues/22).

- **The `gui` and `ui` extras** and the `adaptshot-studio` console script, which existed
  only for the above.

The maintained application is [Tambua](apps/tambua/README.md), a separate distribution
built on AdaptShot — `pip install tambua`. `tests/test_library_ships_no_gui.py` fails if
a GUI reappears anywhere under `src/adaptshot/`, if either extra returns, or if a live
document points at a removed entrypoint.

### Corrected

Documentation claims that were not true when written, retracted here rather than
quietly edited away. A changelog that records a retraction is worth more than one that
only lists features.

- **"MziziGuard is deployed"** — it was not, and is not. No farmer has used it. Its
  sample images are generated with `ImageDraw.ellipse()`. It is a worked example, and
  is now labelled as one. Tracked in
  [#17](https://github.com/johnson2006christopher/adaptshot/issues/17).
- **"Torch-free inference via bundled backbones"** (v0.2.0 entry below, and the front
  page) — there are no bundled backbones; `src/adaptshot/data/` contains only
  `__init__.py`, and inference requires torch today. Tracked in
  [#35](https://github.com/johnson2006christopher/adaptshot/issues/35) and
  [#36](https://github.com/johnson2006christopher/adaptshot/issues/36).
- **"20ms P95 latency"** — the benchmark artifact this figure cited reports
  `latency_p95_ms: 36.43`. The claim was never true of the run it pointed at.
- **"~150ms on CPU"** (MziziGuard) and **"<2MB RAM"** (v0.1.1 audit) — no script in
  `benchmarks/` produces either number. Removed rather than re-estimated.
- **"Distribution-free 95% coverage guarantee"** and **"<250MB RAM"** — these are
  design targets, not measurements. Neither has been verified. Restated as targets;
  tracked in [#14](https://github.com/johnson2006christopher/adaptshot/issues/14) and
  [#13](https://github.com/johnson2006christopher/adaptshot/issues/13).

Historical entries below are left as written. They record what was claimed at the time,
which is the point of a changelog; the corrections above record what was actually true.


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

- **`mypy --strict` now checks all 32 modules in the package** — previously zero. It
  aborted inside numpy's stubs before reaching any project file, so every merge for
  months passed a type check that had examined nothing. `python_version` is now 3.12
  (the floor stays guarded by ruff's `target-version` and by a real 3.10 CI job), the
  lint job installs the torch extra so torch calls are checked against real stubs, and
  `tests/test_mypy_coverage.py` asserts on the number of files mypy reports checking,
  not just on the error count.

### Fixed

- `utils/profiling.py` silently swallowed every exception from `psutil`, reporting
  `0.0 MB` when a memory measurement failed. Callers could not distinguish a real
  zero reading from a failed one — which would have made a `<250MB` ceiling assertion
  pass for the wrong reason. Failures are now logged.
- `core/backends/onnx_backend.py` used `functools.lru_cache` on a method, which keys on
  `self` and would have retained every backend instance, along with up to four loaded
  ONNX sessions and their weights, for the lifetime of the process. Replaced with a
  per-instance cache that is released with the instance.
- Eight type errors that had been invisible behind the aborted mypy run: four redundant
  `cast()` calls, two stale `# type: ignore` comments, a numpy overload that resolved to
  a scalar because a shape was typed `Any`, and an unproven non-null invariant in
  `core/contrastive.py`. The last is now an explicit runtime check rather than a
  `# type: ignore`, so a future edit that breaks the invariant fails where it is broken
  instead of surfacing as a `TypeError` inside the training loop.

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
