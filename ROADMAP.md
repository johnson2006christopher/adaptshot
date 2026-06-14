# AdaptShot Roadmap

This roadmap outlines the planned evolution of AdaptShot from the current v0.2.0 release through v1.0.0 and beyond. Priorities are driven by the project constitution: CPU-first, <250MB RAM, human-in-the-loop, and carbon-aware.

For completed work, see [CHANGELOG.md](CHANGELOG.md). For how to contribute, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## v0.1.1 (Released June 2026)

See [CHANGELOG.md](CHANGELOG.md#0111---2026-06-06) for the full release notes. Highlights:

- Eco mode with early-exit thresholds for energy-aware inference
- EmbeddingCache isolation for safe multi-learner workflows
- OOD detection and prototypical inference modes
- String label support throughout correction routing
- Checkpoint integrity with SHA-256 checksums and schema migration
- 12-chapter tutorial suite with Studio GUI guide

---

## v0.1.2 — Merged into v0.2.0

> **Note**: v0.1.2 was originally planned for July 2026 with localization features, but its scope was merged into the v0.2.0 production hardening release. Swahili localization and accessibility improvements are now tracked under v0.3.0.

---

## v0.2.0 (Released June 2026) — Production Hardening

### Production Hardening

- ✅ **LOO Conformal Prediction**: True leave-one-out calibration for tighter prediction sets with sparse data
- ✅ **Shrinkage Covariance Mahalanobis**: Robust OOD detection with automatic λ scaling — works with as few as 2 samples/class
- ✅ **Gradient-Trained Contrastive Projection Head**: W₁,b₁,W₂,b₂ trained via InfoNCE backprop with SGD momentum
- ✅ **ACT Symmetric Updates with Mean-Reversion**: Prevents threshold drift in long-running services
- ✅ **UP-UGF LSH Acceleration**: O(N log N) approximate redundancy scoring via Locality-Sensitive Hashing
- ✅ **Bootstrap Temperature Calibration**: Bootstrap resampling for stable temperature with small calibration windows
- ✅ **Historical Penalty Tracking**: Per-class penalty history with trend detection in explainability engine
- ✅ **MemoryTracker**: Section-level memory profiling with budget enforcement
- ✅ **ONNX Runtime Backend**: Torch-free inference (~800 MB smaller install) for edge deployment
- ✅ **clear_backbone_cache()**: Memory reclamation for long-running services

### Documentation

- ✅ 42+ markdown files covering all APIs, algorithms, tutorials, and guides
- ✅ Algorithm theory with full mathematical foundations (shrinkage covariance, InfoNCE gradients, LSH, bootstrap, symmetric ACT)
- ✅ Quality gates: ruff=0, mypy strict=32 files, pytest=92 passed, benchmark=68%

### Deferred to v0.3.0

- ARM Profiling: Reproducible Raspberry Pi 4 benchmarks
- PlantVillage Benchmark: Public crop disease dataset loader
- Community Benchmarks: Energy challenge for lowest Joules/inference

---

## v1.0.0 (Target: Q1 2027) -- Production-Grade

### Validation & Trust

- **Peer-Reviewed Publication**: Full methodology paper with ablation studies in a peer-reviewed venue
- **Field Pilot Results**: Deployment metrics from 3+ NGO partnerships in Tanzania, Kenya, and Uganda
- **Carbon-Neutral CI/CD**: Offsetting compute emissions through verified carbon credits

### Platform Maturity

- **Plugin Architecture**: `EmbeddingBackend` protocol for alternative runtimes (ONNX Runtime, OpenVINO, Core ML)
- **Federated Buffer Sharing**: Privacy-preserving multi-device buffer aggregation for community deployments
- **Multilingual UI**: French and low-literacy icon-driven Gradio/Studio interfaces (Swahili ships in v0.1.2)

### Governance

- **Community Governance Board**: Diverse advisory board with representation from Global South practitioners
- **Stable API Guarantee**: `FewShotLearner` API frozen as semver-major; deprecation warnings for 2 minor versions

---

## v2.0+ (2027 and Beyond) -- Neuromorphic Bridge

- **Event-Based Vision**: Support for DVS (Dynamic Vision Sensor) cameras with spiking neural network backends
- **Neuromorphic Backends**: Intel Loihi and other neuromorphic hardware support when ecosystem matures
- **National Integration**: Partnerships with healthcare and agriculture ministries for population-scale deployment

---

## How Priorities Are Set

Priorities follow the [project constitution](.openproject.md):

1. Does it reduce energy consumption?
2. Can it run on a 3-year-old laptop with 4GB RAM?
3. Does it require internet or cloud infrastructure?
4. How does it perform under intermittent power or thermal throttling?
5. Is the marginal accuracy gain worth the carbon cost?

Features that answer "yes" to questions 1-2 and "no" to question 3 are prioritized.

---

*"The future of AI is not bigger -- it's smarter, humbler, and more human."*
