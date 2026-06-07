# AdaptShot Roadmap

This roadmap outlines the planned evolution of AdaptShot from the current v0.1.1 release through v1.0.0 and beyond. Priorities are driven by the project constitution: CPU-first, <250MB RAM, human-in-the-loop, and carbon-aware.

For completed work, see [CHANGELOG.md](CHANGELOG.md). For how to contribute, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## v0.1.1 (Released June 2026) -- Current

See [CHANGELOG.md](CHANGELOG.md#0111---2026-06-06) for the full release notes. Highlights:

- Eco mode with early-exit thresholds for energy-aware inference
- EmbeddingCache isolation for safe multi-learner workflows
- OOD detection and prototypical inference modes
- String label support throughout correction routing
- Checkpoint integrity with SHA-256 checksums and schema migration
- 12-chapter tutorial suite with Studio GUI guide

---

## v0.1.2 (Target: July 2026) -- Localization & Accessibility

### UI Localization

- **Swahili Gradio Dashboard**: Full Swahili translation of all Gradio UI labels, buttons, help text, and error messages
- **Localization Framework**: i18n string extraction pipeline and `.po`/`.mo` translation workflow for Gradio
- **Accessibility Pass**: ARIA labels, keyboard navigation, and screen-reader compatibility for the Gradio dashboard

> **Excluded**: French localization is deferred to v0.2.0+. v0.1.2 is Swahili-first to serve East African users.

---

## v0.2.0 (Target: Q3 2026) -- Research Platform

### Core Features

- **ONNX Export**: Export classification heads to ONNX for mobile (Android) and browser (WebAssembly) deployment
- **ARM Profiling**: Reproducible Raspberry Pi 4 benchmarks with committed results in `benchmarks/results/arm/`
- **PlantVillage Benchmark**: Public crop disease dataset loader and baseline metrics for agriculture use cases

### Research & Validation

- **Conformal Prediction**: Full conformal prediction set implementation (beyond current stub) for distribution-free uncertainty
- **Approximate UP-UGF**: Replace `O(N^2)` redundancy computation with FAISS/annoy-based approximate nearest-neighbor search for buffers >500
- **Ablation Studies**: Systematic ablation of ECE, ACT, CA-EWC, and UP-UGF components with published results

### Community & Infrastructure

- **CI/CD Pipeline**: Automated GitHub Actions for `ruff`, `mypy`, `pytest`, and docs deployment on every push
- **Community Benchmarks**: Energy challenge inviting community submissions for lowest Joules/inference

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
