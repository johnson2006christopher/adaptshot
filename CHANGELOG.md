# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned for v0.2.0
- ONNX export support for broader edge deployment (Android, WebAssembly)
- Improved UP-UGF redundancy computation (approximate nearest-neighbor fallback for larger buffers)
- Conformal prediction integration beyond current stub implementation
- Swahili and French UI localization for Gradio dashboard
- Automated GitHub Actions workflow for CI testing, linting, and docs deployment
- Federated buffer sharing for multi-device deployments
- Plugin architecture for experimental backends

---

## [0.1.1] - 2026-05-20

### Added
- **Comprehensive Documentation**: Complete 11-chapter tutorial suite covering:
  - Getting Started with synthetic crop disease demo
  - Human-in-the-loop learning with correction routing
  - Continual learning with buffer management and calibration updates
  - Production-ready patterns with error handling and energy profiling
  - Reference FAQ with glossary and troubleshooting
  - Core API deep dive with method-by-method walkthrough
  - Source code tour for module navigation
  - Configuration, determinism, and safe I/O utilities
  - Benchmarks and reproducibility guide
  - Module map for source tree navigation
  - UI pilot dashboard (Gradio-based interface)
- **About Page**: Creator's story, mission, values, and vision for sustainable AI
- **Enhanced Navigation**: Updated MkDocs site with About section and all tutorial chapters
- **Logo & Branding**: AdaptShot logo integration in site navigation and browser tab
- **Reference Materials**: Comprehensive FAQ, API tables, and troubleshooting guides

### Changed
- **Documentation Structure**: Reorganized docs with clearer hierarchy (Getting Started → Tutorials → API Reference)
- **mkdocs.yml**: Enhanced theme configuration with logo and favicon support
- **MkDocs Theme**: Material theme features enabled for better navigation and code display

### Improved
- **Code Accessibility**: All tutorials reference actual source files in `src/adaptshot/` with verified APIs
- **User Experience**: Better navigation flow from About page through Getting Started to production deployment
- **Community Focus**: Documentation written for global audience, emphasizing resource-constrained deployment

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
