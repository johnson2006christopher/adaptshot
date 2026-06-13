# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned for v0.2.0
- ONNX export support for broader edge deployment (Android, WebAssembly)
- Improved UP-UGF redundancy computation (approximate nearest-neighbor fallback for larger buffers)
- Conformal prediction integration beyond current stub implementation
- French UI localization for Gradio dashboard (Swahili ships in v0.1.2)
- Automated GitHub Actions workflow for CI testing, linting, and docs deployment

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

---

*Created by [Johnson Christopher Hassan](https://github.com/johnson2006christopher)*  
*Connect on [LinkedIn](https://www.linkedin.com/in/johnson-hassan-935124311/)*  
*Project: [github.com/johnson2006christopher/adaptshot](https://github.com/johnson2006christopher/adaptshot)*
