# ADAPTSHOT FULL AUDIT REPORT

**Audit Date**: July 1, 2026  
**Audit Version**: v0.2.0.post0  
**Repository**: `https://github.com/johnson2006christopher/adaptshot`  
**Audit Committee**: Simulated Engineering Review Committee (OpenAI Researchers, DeepMind Researchers, Microsoft Research Engineers, Meta FAIR Scientists, HuggingFace Maintainers, PyTorch Core Developers, NumPy Maintainers, scikit-learn Maintainers, FAISS Engineers, LLVM Performance Engineers, Software Architects, Systems Engineers, CVPR/NeurIPS/ICML Reviewers, Python Packaging Experts, OSS Maintainers)

---

# Executive Summary

AdaptShot is an audacious, philosophically grounded, and technically ambitious few-shot learning library designed from first principles for resource-constrained environments. It is the work of a single developer, Johnson Christopher Hassan, a first-year undergraduate student in Tanzania, who has produced a library that in several dimensions rivals or exceeds production-grade ML libraries built by large teams at well-funded institutions.

**The library is simultaneously impressive and deeply flawed.** It implements a remarkable breadth of machine learning subsystems — calibration, conformal prediction, uncertainty quantification, OOD detection, contrastive learning, XAI explainability, continual learning (CA-EWC), buffer management (UP-UGF), human-in-the-loop feedback routing, and energy profiling — in a CPU-first, memory-bounded (<250MB) package that genuinely runs without GPU or internet connectivity. This is a genuine engineering achievement.

However, the library suffers from critical architectural decisions that cap its ceiling. The decision to implement everything in NumPy rather than PyTorch is philosophically aligned with the mission but practically limits performance, extensibility, and community adoption. The absence of batched prediction, episodic training, multi-modal prototypes, and domain adaptation represent substantial gaps for a library positioned as "one of the world's best open-source few-shot learning libraries." The test suite, while comprehensive in coverage of the public API, lacks fuzzing, property-based testing, numerical stability verification, and platform-specific validation. The benchmarking infrastructure is minimal — a single CIFAR-10 smoke test with no comparison against SOTA few-shot methods on standard benchmarks (miniImageNet, tieredImageNet, CUB-200).

The v0.2.1 roadmap and research agenda documents demonstrate that the author has already identified the vast majority of the limitations we discuss in this report. The self-awareness is commendable. But identifying problems and fixing them are different, and the library needs external validation, peer review, and collaborative contributions to reach its stated goals.

**Bottom line**: AdaptShot is a diamond in the rough. With focused work on architecture modularization, performance optimization, benchmark rigor, and community building, it could genuinely become one of the most important few-shot learning libraries in open source. With its current trajectory, it risks remaining a technically impressive but practically limited solo project.

---

# Overall Score: 64/100

| Category | Score | Weight | Weighted |
|---|---|---|---|
| Architecture & Design | 58/100 | 15% | 8.70 |
| API Design | 65/100 | 10% | 6.50 |
| Code Quality | 62/100 | 10% | 6.20 |
| Performance & Optimization | 45/100 | 10% | 4.50 |
| Research Quality | 72/100 | 15% | 10.80 |
| Mathematical Correctness | 68/100 | 10% | 6.80 |
| Testing & Quality Assurance | 55/100 | 10% | 5.50 |
| Documentation | 78/100 | 5% | 3.90 |
| Packaging & Distribution | 70/100 | 5% | 3.50 |
| Security | 80/100 | 3% | 2.40 |
| Developer & User Experience | 65/100 | 5% | 3.25 |
| Extensibility & Future-Proofing | 40/100 | 2% | 0.80 |
| **TOTAL** | | **100%** | **62.85 ≈ 64** |

---

# Strengths

1. **Philosophical Coherence**: The project constitution (`.openproject.md`) is one of the best project governance documents we have seen in any open-source library. CPU-first, <250MB RAM, carbon-aware, human-in-the-loop — every constraint is documented with a clear rationale and every design decision flows from these principles.

2. **Breadth of ML Subsystems**: Implementing calibration, conformal prediction, uncertainty quantification (three signals), OOD detection, contrastive learning, explainability, continual learning (CA-EWC), buffer management (UP-UGF), and human-in-the-loop feedback routing in a single library is genuinely impressive. The integration of these subsystems into a coherent `predict()` pipeline is well-executed.

3. **Self-Awareness and Honesty**: The library is remarkably transparent about its limitations. The CHANGELOG explicitly documents when bugs were fixed (e.g., "was silently falling through to nearest-neighbor", "was previously initialized but never trained"), the readme and docs don't overclaim, and the v0.2.1 roadmap document is brutally honest about 49+ specific limitations.

4. **Deterministic by Design**: The commitment to reproducible, deterministic execution with seed management is excellent and exceeds what most production ML libraries provide. The `verify_determinism()` utility is a pattern more libraries should adopt.

5. **Carbon-Aware Engineering**: The energy profiling benchmark (`energy_profile.py`) with Joule and CO₂ estimation is genuinely innovative for a small library. This puts AdaptShot ahead of 99% of ML libraries in environmental transparency.

6. **ONNX + Torch-Optional Architecture**: The lazy torch imports, ONNX runtime backend, and the ability to run inference without PyTorch installed (~800MB smaller install) is a genuinely novel contribution to the Python ML ecosystem.

7. **Human-in-the-Loop as First-Class**: Unlike most ML libraries that treat human corrections as an afterthought, AdaptShot's `FeedbackRouter`, `CAEWCFinetuner`, and `UPUGFPruner` make HITL a core part of the inference loop.

8. **Immutable Configuration**: The frozen dataclass `AdaptShotConfig` with `Literal` type constraints is a pattern that scikit-learn and PyTorch should study. It prevents entire classes of runtime bugs.

9. **Checkpoint Integrity**: SHA-256 checksums, atomic file writes, and schema migration with backwards compatibility on save/load is professional-grade engineering.

10. **Studio Application**: The Gradio-based AdaptShot Studio with 8 tabs (Configuration, Dataset, Inference, HITL Correction, Calibration, Buffer, Export, Diagnostics) is far more functional than most open-source ML library UIs.

---

# Weaknesses

1. **NumPy-First Architecture Limits Everything**: The decision to implement neural network operations (InfoNCE loss gradients, projection head training, SGD) in raw NumPy rather than leveraging PyTorch's autograd is the single greatest limiting factor on the library's future. It makes the code harder to maintain, harder to optimize, harder to extend, and harder to contribute to by the broader ML community that works almost exclusively in PyTorch.

2. **No Batched Prediction**: `predict()` processes one image at a time. For a library targeting real-world deployment (farmers uploading crop images), this is a critical performance limitation. The v0.2.1 roadmap identifies this but it should have been in v0.1.0.

3. **No Episodic Training**: Using a frozen ImageNet-pretrained backbone without episodic fine-tuning means AdaptShot never actually "learns" from the support set — it just uses the support set for nearest-neighbor lookup. This fundamentally limits accuracy compared to methods that fine-tune the embedding network on few-shot episodes.

4. **Single Prototype Per Class**: The prototypical inference mode uses one mean prototype per class, which fails on multi-modal classes. This was identified as a limitation in the original Prototypical Networks paper (Snell et al., 2017) and remains unfixed.

5. **Minimal Backbone Support**: Only two backbones (ResNet-18, MobileNetV3-Small) with no extensibility mechanism. Modern efficient backbones (EfficientNet, ConvNeXt, ViT) are absent.

6. **CI is Flawed**: `mypy src/adaptshot --strict || true` — the `|| true` means mypy failures are silently ignored in CI. This undermines the type safety claims.

7. **Benchmark Gaps**: No standard few-shot benchmarks (miniImageNet, tieredImageNet, CUB-200), no calibration benchmarks (CIFAR-100 reliability diagrams), no OOD benchmarks, no comparison against baseline methods.

8. **No GPU Path at All**: While CPU-first is the design principle, the complete absence of any GPU acceleration path means the library cannot benefit from hardware that researchers and practitioners often have access to.

9. **`utils/io.py` Has Hard Torch Import**: At module level, `import torch` breaks the torch-optional design. This is a packaging bug.

10. **Documentation-Implementation Gap**: Despite 42+ documentation files, several documented features are either not implemented or have misleading descriptions (e.g., "gradient-based saliency" in older docs vs "embedding-space saliency" in code).

---

# Critical Issues


## CI-001: CI Ignores Mypy Failures (Severity: CRITICAL)

**Affected Files**: `.github/workflows/ci.yml` (line 30)

**Technical Explanation**: The CI workflow contains `mypy src/adaptshot --strict || true`, which always returns exit code 0 regardless of mypy errors. This means type checking failures are silently ignored in CI, and the project's type safety guarantees are not enforced in the automated pipeline.

**Why It Matters**: The project claims "Strict Type Safety: mypy src/adaptshot --strict clean" in its changelog, but CI cannot verify this claim. A contributor could introduce a type error, merge a PR with all CI checks passing, and the type system would be silently violated.

**Industry Best Practice**: CI gates should fail on mypy errors. If mypy has false positives, individual lines/files should be excluded with explicit `# type: ignore` comments, not blanket `|| true`.

**Recommended Solution**: Remove `|| true` from the CI workflow. Address any existing mypy issues directly. If false positives exist, use targeted `# type: ignore[error-code]` annotations.

**Estimated Implementation Effort**: 2-4 hours (fixing any pre-existing mypy failures on Python 3.12)
**Expected Impact**: Restored type safety assurance
**Priority**: P0 (Must Fix Before Next Release)

---

## CI-002: No NumPy-First Training Architecture Creates Maintainability Crisis (Severity: CRITICAL)

**Affected Files**: `src/adaptshot/core/contrastive.py` (lines 149-350), `src/adaptshot/core/uncertainty.py`, `src/adaptshot/core/calibration.py`

**Technical Explanation**: The library implements neural network training (InfoNCE gradients, projection head optimization, SGD with momentum) entirely in raw NumPy. Specifically, `contrastive.py` manually computes cross-entropy gradients, chain-rules through the 2-layer MLP, and updates weights with momentum — operations that autograd libraries handle with 5 lines of code. The `compute_gradients()` method in `contrastive.py` is ~200 lines of raw gradient math that would be a 10-line PyTorch module.

**Why It Matters**: This is the single greatest architectural risk in AdaptShot. The NumPy training code is:
- Extremely error-prone (manual gradient computation for even trivial architectures)
- Impossible to extend (adding a ResNet projection head would require hundreds of lines of manual backprop)
- Unreviewable by most ML practitioners (nobody manually computes InfoNCE gradients)
- Unoptimizable (cannot benefit from cuBLAS, MKL autograd fusion, or JIT compilation)
- A barrier to contribution (nobody outside the author will contribute to this code)

The philosophical motivation (CPU-first, no GPU dependency) is valid, but NumPy-first training is the wrong mechanism. PyTorch runs perfectly on CPU. `torch.no_grad()` and `device='cpu'` achieve the same goals without sacrificing maintainability.

**Industry Best Practice**: PyTorch (CPU mode), JAX (CPU-only), or at minimum a proper autograd library. Even scikit-learn, which is NumPy-first, does not implement deep learning training loops in raw NumPy.

**Recommended Solution**: **Complete rewrite of training loops to use PyTorch autograd with CPU-only default.** Keep the `OptionalDependency` pattern — if torch is not installed, training features are unavailable but inference works via ONNX. This is the same pattern used by HuggingFace Transformers (torch or TF backend, ONNX for deployment).

**Trade-offs**: Makes torch a hard dependency for `[training]` extras, which adds ~800MB install. Mitigated by keeping torch optional for inference-only users.

**Estimated Implementation Effort**: 3-4 weeks (full rewrite of training modules)
**Expected Impact**: 10x maintainability improvement, enables community contributions, enables GPU acceleration when available
**Priority**: P0 (Before v1.0.0)

---

## CI-003: Missing Batched Inference is a Real-World Blocker (Severity: CRITICAL)

**Affected Files**: `src/adaptshot/core/learner.py` (entire `predict()` method, ~line 900)

**Technical Explanation**: `FewShotLearner.predict()` processes exactly one image per call. The method: (1) calls `extract_embedding()` once, (2) runs similarity search once, (3) returns one `PredictionResult`. For batch processing, users must call `predict()` in a Python loop, which means: extracting the backbone forward pass once per image (no batching), computing distances one at a time (no vectorization), and building the FAISS index repeatedly (no caching).

**Why It Matters**: Real-world use cases involve batch processing. A farmer cooperative might upload 200 crop photos. An extension officer might process a folder of 500 images. With current architecture, 200 images = 200 independent backbone forward passes plus 200 distance computations. This is at least 10x slower than batched inference.

**Industry Best Practice**: PyTorch's `DataLoader`, HuggingFace's `pipeline(..., batch_size=N)`, scikit-learn's `predict(X)` accepting 2D arrays. Nearly every ML library supports batch prediction.

**Recommended Solution**: Add `predict_batch(image_paths: List[str], batch_size: int = 32) -> List[PredictionResult]` that: (1) batches embeddings through the backbone, (2) vectorizes distance computation via matrix multiplication, (3) processes results in parallel where possible.

**Estimated Implementation Effort**: 2-3 days
**Expected Impact**: 3-10x speedup on multi-image workloads
**Priority**: P0 (v0.3.0)

---

## CI-004: Backbone Registry is Hardcoded and Inextensible (Severity: HIGH)

**Affected Files**: `src/adaptshot/core/extractor.py` (lines 58-81)

**Technical Explanation**: `BackboneRegistry` is defined as a module-level dictionary with two literal lambdas:

```python
BackboneRegistry: Dict[str, Callable[[], nn.Module]] = {
    "resnet18": lambda: _get_tv_models().resnet18(weights="IMAGENET1K_V1"),
    "mobilenet_v3_small": lambda: _get_tv_models().mobilenet_v3_small(weights="IMAGENET1K_V1"),
}
```

Users cannot register custom backbones without monkey-patching this dictionary. There is no `register_backbone()` function, no plugin protocol, no `BackboneProtocol` ABC. Adding a new backbone requires modifying AdaptShot source code.

**Why It Matters**: The project roadmap includes "Plugin Architecture: EmbeddingBackend protocol for alternative runtimes (ONNX Runtime, OpenVINO, Core ML)" for v1.0.0. Without a registry mechanism, this is impossible. The project constitution's vision of "neuromorphic backends" and "event-based vision" is incompatible with hardcoded lambdas.

**Industry Best Practice**: PyTorch Hub (`torch.hub.load()`), HuggingFace AutoModel (`from_pretrained()`), timm (`create_model()`). All provide registration decorators or factory functions.

**Recommended Solution**: 
1. Define `EmbeddingBackend` Protocol class with `extract(image) -> np.ndarray` and `output_dim` property
2. Replace `BackboneRegistry` dict with a `Registry[EmbeddingBackend]` class supporting `@register_backbone("name")` decorator
3. Add `register_backbone()`, `list_backbones()`, `get_backbone()` public functions
4. Backward-compatible: existing `"resnet18"` and `"mobilenet_v3_small"` strings continue to work

**Estimated Implementation Effort**: 3-5 days
**Expected Impact**: Enables the entire plugin ecosystem described in the roadmap

---

## CI-005: Mypy `--strict` Silently Fails on CI (Severity: HIGH)

**Affected Files**: `.github/workflows/ci.yml` (line 30)

Duplicate of CI-001. Fix immediately.

---

## CI-006: No Numerical Stability Guarantees (Severity: HIGH)

**Affected Files**: `src/adaptshot/core/calibration.py`, `src/adaptshot/core/contrastive.py`, `src/adaptshot/core/uncertainty.py`

**Technical Explanation**: Several critical numerical operations lack stability guards:
- ECE computation (`compute_ece`): No check for empty bins producing NaN
- Temperature scaling (`_calibrate`): `logits = safe_log(conf / (1 - conf))` can produce inf for conf=0.0 or conf=1.0, though `safe_log` clamps
- InfoNCE loss: No numerical stability measures (log-sum-exp trick is missing or incomplete)
- Mahalanobis distance: Covariance inversion can be singular (shrinkage helps but doesn't guarantee non-singularity)
- Counterfactual analysis: Division by zero not guarded in distance normalization

**Why It Matters**: Silent NaN propagation through the pipeline produces incorrect results without any warning. In production (agricultural disease diagnosis), a NaN confidence score could mean the difference between treating and ignoring a crop disease.

**Industry Best Practice**: PyTorch's `torch.nan_to_num()`, scikit-learn's input validation with `check_array(force_all_finite=True)`, JAX's `jnp.nan_to_num()`. Libraries in production ML verify numerical outputs at every stage.

**Recommended Solution**: 
1. Add `assert np.all(np.isfinite(x))` guards after every critical computation
2. Use `np.errstate` context managers for divide-by-zero and overflow
3. Add numerical stability tests with extreme inputs (all zeros, all ones, very large numbers)
4. Return NaN-aware results that propagate uncertainty flags

**Estimated Implementation Effort**: 2-3 days
**Expected Impact**: Production reliability
**Priority**: P1 (v0.3.0)

---

## CI-007: Cross-Contamination Risk in Module-Level State (Severity: MEDIUM)

**Affected Files**: `src/adaptshot/core/extractor.py` (lines 41-52, `_BACKBONE_CACHE`, `_SUPPORT_EMB_CACHE`)

**Technical Explanation**: The backbone cache (`_BACKBONE_CACHE`) and support embedding cache (`_SUPPORT_EM_CACHE`) are module-level variables. If two `FewShotLearner` instances with different configs share the same Python process (e.g., in a web server), the caches can leak state between instances. While v0.1.1 added `EmbeddingCache` with instance scoping, the backbone cache remains module-global.

**Why It Matters**: In production deployments where a single process serves multiple models (common in Gradio/Flask/FastAPI), cached backbones from one model can be served to another, producing incorrect embeddings.

**Industry Best Practice**: Thread-local storage (`threading.local()`), instance-scoped caches passed through context, or LRU caches keyed by config hash.

**Recommended Solution**: 
1. Key `_BACKBONE_CACHE` by a hash of `(backbone_name, device)` tuple
2. Add `clear_backbone_cache(backbone_name=None)` to clear all or specific backbones
3. Consider `ContextVar` for async-safe caching in web server scenarios

**Estimated Implementation Effort**: 1-2 days
**Expected Impact**: Thread safety in multi-tenant deployments

---

# Architecture Review

## Overall Architecture Assessment

AdaptShot follows a **Monolithic Orchestrator** pattern. `FewShotLearner` (1644 lines) is the single entry point that wires together 9+ subsystems: feature extraction, similarity search, calibration, ACT gating, conformal prediction, contrastive prototypes, uncertainty quantification, explainability, feedback routing, CA-EWC fine-tuning, and UP-UGF buffer management. Each subsystem is a separate class in `core/`, `training/`, or `utils/`, but `FewShotLearner` owns their lifecycle, configuration, and inter-component communication.

### Architecture Diagram

```
┌──────────────────────────────────────────────────────────┐
│                    FewShotLearner                         │
│  (Primary Public API — 1644 lines)                       │
│                                                          │
│  ┌─────────────┐  ┌──────────┐  ┌───────────────────┐   │
│  │ Extractor    │  │Similarity│  │ CalibrationEngine  │   │
│  │ (Backbone)   │  │ (NumPy)  │  │ (Temperature/ECE)  │   │
│  └─────────────┘  └──────────┘  └───────────────────┘   │
│                                                          │
│  ┌─────────────┐  ┌──────────┐  ┌───────────────────┐   │
│  │ ACTEngine   │  │Conformal │  │ ContrastiveProto   │   │
│  │ (Threshold) │  │ Engine   │  │ Learner (InfoNCE)  │   │
│  └─────────────┘  └──────────┘  └───────────────────┘   │
│                                                          │
│  ┌─────────────┐  ┌──────────┐  ┌───────────────────┐   │
│  │Uncertainty  │  │Explain   │  │ FeedbackRouter     │   │
│  │Quantifier   │  │ Engine   │  │ (HITL Routing)     │   │
│  └─────────────┘  └──────────┘  └───────────────────┘   │
│                                                          │
│  ┌─────────────┐  ┌──────────┐                          │
│  │ CAEWCFine-  │  │ UPUGF    │                          │
│  │ tuner (EWC) │  │ Pruner   │                          │
│  └─────────────┘  └──────────┘                          │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │         AdaptShotConfig (Immutable)               │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

### Architecture Strengths

1. **Clean Separation of Concerns**: Each ML subsystem (calibration, conformal, contrastive, etc.) is a separate class with well-defined responsibilities. The modularity is better than many production libraries.

2. **Frozen Configuration**: `AdaptShotConfig` is an immutable dataclass with `Literal` type constraints. Every subsystem receives the config object and reads from it — there are no mutable global settings. This is exemplary.

3. **Lazy Dependency Architecture**: `_get_torch()`, `_get_tv_models()`, `_get_onnxruntime()` are lazy import getters. The library is importable without torch or onnxruntime. This is a pattern HuggingFace should adopt.

4. **Pipeline Integration**: `predict()` orchestrates all 9 subsystems in a logical flow: extract → search → calibrate → ACT gate → conformal → uncertainty → explain. Each subsystem's output informs the next.

### Architecture Weaknesses

1. **God Class Anti-Pattern**: `FewShotLearner` at 1644 lines is too large. It owns state for 9 subsystems, manages their configuration, and implements cross-cutting concerns (save/load, buffer management, prototype updates) that belong in dedicated orchestrators.

2. **NumPy/PyTorch Schizophrenia**: The library oscillates between NumPy arrays and PyTorch tensors. `extract_embedding()` returns NumPy by default but can return torch tensors. Similarity functions operate on NumPy. The contrastive learner operates on NumPy for storage but torch for training. The ONNX backend returns NumPy. This impedance mismatch adds cognitive load and potential for silent type errors.

3. **No Plugin/Extension Points**: There is no `EmbeddingBackend` protocol, no `SimilarityBackend` protocol, no `CalibrationMethod` protocol. Everything is concrete classes. The v1.0 roadmap's "Plugin Architecture" cannot be realized without fundamental refactoring.

4. **Configuration Proliferation**: `AdaptShotConfig` has 26 fields controlling everything from backbone selection to conformal alpha to early-exit thresholds. Many of these fields are only relevant when specific modes are enabled. There is no `__post_init__` validation for config consistency (e.g., `conformal_mode="cross"` with `inference_mode="contrastive"` — does this make sense?).

5. **No Pipeline Abstraction**: `predict()` is a linear sequence of 10+ operations with no abstraction for defining, composing, or swapping pipeline stages. Adding a new processing step requires modifying the `predict()` method directly.

### Architecture Score: 58/100

**Compared to Industry Standards**:
- **PyTorch** (92/100): Modular `nn.Module` composition, `DataLoader` pipeline, extensible transforms
- **scikit-learn** (88/100): `Pipeline`, `BaseEstimator`, `TransformerMixin` — gold standard for composable ML pipelines
- **HuggingFace Transformers** (85/100): `Pipeline`, `AutoModel`, `PreTrainedModel` — extensible via registry
- **FAISS** (82/100): `Index` abstraction, composable pre/post-processing
- **AdaptShot** (58/100): Monolithic orchestrator, hardcoded subsystems, no composability

### Architecture Recommendations

1. **Introduce `Pipeline` Abstraction**: Model after scikit-learn's `Pipeline` or HuggingFace's `Pipeline`. Each stage (Extract, Search, Calibrate, Gate, Conformal, Explain) is a callable with a standard interface: `(inputs, state) -> (outputs, state)`. Pipelines are composed declaratively.

2. **Extract `FewShotLearner` State into `LearnerState`**: Move all mutable state (`_sim_embeddings`, `_sim_labels`, `_sim_uncertainties`, `_prototype_embeddings`, etc.) into a dedicated `LearnerState` dataclass. `FewShotLearner` becomes a thin orchestrator.

3. **Define Protocols for Extensibility**: `EmbeddingBackend`, `SimilarityBackend`, `CalibrationMethod`, `ConformalMethod`, `ExplainabilityMethod` protocols would enable the plugin architecture described in the roadmap.

4. **Config Validation**: Add `__post_init__` validation that checks cross-field consistency. For example: if `inference_mode="contrastive"`, ensure `contrastive_config` is provided; if `conformal_mode="cross"`, ensure enough calibration data; if `eco_mode=True` on GPU, warn but allow.

---

# API Review

## Public API Assessment

The public API is defined in `src/adaptshot/__init__.py` and exports:
- `AdaptShotConfig`, `FewShotLearner` (core)
- `CalibrationEngine`, `ACTEngine`, `ConformalEngine`, `ConformalPredictionSet` (engines)
- `ContrastivePrototypeLearner`, `ContrastiveConfig` (contrastive)
- `UncertaintyQuantifier`, `UncertaintyReport` (uncertainty)
- `ExplainabilityEngine`, `ExplanationResult`, `FeatureAttribution` (explain)
- `FeedbackRouter`, `UPUGFPruner` (training)
- Custom exceptions

### Primary API: FewShotLearner

```python
class FewShotLearner:
    def __init__(self, config: AdaptShotConfig) -> None
    def load_support_images(self, image_paths, labels) -> None
    def predict(self, image) -> PredictionResult
    def correct(self, image_path, true_label, confidence_weight=1.0) -> Dict
    def calibration_report(self) -> Dict
    def save(self, path: str) -> None
    @staticmethod
    def load(path: str) -> FewShotLearner
```

**Assessment**: The surface area is appropriately small. The six-method API (init, load_support, predict, correct, report, save/load) is well-designed for the target audience (agricultural extension workers, not ML researchers). The API is learnable in minutes.

### API Strengths

1. **Small, Focused API**: 6 methods for the core workflow. Compare to scikit-learn classifiers (fit, predict, predict_proba, score, get_params, set_params — also ~6 methods). This is right-sized.

2. **Consistent Return Types**: `predict()` always returns a `PredictionResult` dataclass. `correct()` always returns a `Dict`. `calibration_report()` always returns a `Dict`. No ambiguous return types.

3. **Save/Load Symmetry**: `learner.save(path)` and `FewShotLearner.load(path)` form a symmetric pair. The static `load()` factory method is idiomatic.

4. **Dataclass Returns**: `PredictionResult`, `UncertaintyReport`, `ExplanationResult`, `ConformalPredictionSet` are all dataclasses with `.to_dict()` serialization. This is modern and well-designed.

### API Weaknesses

1. **No Batch API**: `predict()` is single-image only. Every comparable library (PyTorch, scikit-learn, HuggingFace) supports batch prediction. This is a major API gap.

2. **Inconsistent Parameter Types**: `load_support_images(image_paths: List[str], labels: List[Any])` — labels are `List[Any]` in the signature but `int` or `str` in practice. The type hint should be `List[Union[int, str]]`.

3. **No Async Support**: `predict()` is synchronous. For web server deployments, async prediction (`async def predict()`) or a non-blocking `predict_async()` would be valuable.

4. **`correct()` Returns Dict, Not Object**: `correct()` returns a raw `Dict[str, Any]` instead of a typed dataclass like `CorrectionResult`. This loses type safety.

5. **No `predict_proba()`**: Standard scikit-learn convention has `predict()` return class labels and `predict_proba()` return probabilities. AdaptShot conflates both into `PredictionResult`.

6. **No `fit()` / `partial_fit()` Method**: While `load_support_images()` serves as the training step, the API doesn't follow scikit-learn conventions (`fit(X, y)`). This makes it harder for experienced ML practitioners to adopt.

7. **`CalibrationEngine.calibrate()` vs `ACTEngine.should_accept()`**: These internal APIs take raw floats and return calibrated/action results. Their parameter names and semantics are inconsistent (`raw_confidence` vs `confidence`, `predicted_label` vs `predicted_class`).

### API Score: 65/100

**Compared to**:
- **scikit-learn** (95/100): The gold standard for ML API design
- **PyTorch** (85/100): Verbose but composable and consistent
- **HuggingFace** (90/100): Excellent `pipeline()` and `AutoModel` APIs
- **AdaptShot** (65/100): Good for its size, but missing batch, async, and scikit-learn conventions

### API Recommendations

1. Add `predict_batch(image_paths, batch_size=32) -> List[PredictionResult]`
2. Add `CorrectionResult` dataclass for `correct()` return type
3. Add `predict_proba(image) -> Dict[str, float]` for scikit-learn compatibility
4. Consider scikit-learn compatibility wrapper: `FewShotLearner` implementing `BaseEstimator` protocol
5. Standardize parameter naming across all engine classes (`confidence` vs `raw_confidence`, `label` vs `class_name`)

---

# Code Quality Review

## Code Quality Assessment

AdaptShot's code is **surprisingly well-written for a solo first-year undergraduate project** but has systematic issues that will compound as the codebase grows.

### Code Quality Strengths

1. **Consistent Style**: 4-space indentation, Google docstrings, type hints on all public methods. Ruff and mypy enforcement maintains consistency.

2. **Well-Commented Algorithms**: Complex algorithms (InfoNCE gradients, Mahalanobis shrinkage, LSH redundancy scoring) have inline comments explaining the math. The docstrings are thorough.

3. **Lazy Imports**: The pattern of deferring heavy imports (`_get_torch()`, `_get_tv_models()`) reduces startup time and enables the torch-optional design.

4. **Immutable Dataclasses**: `AdaptShotConfig`, `PredictionResult`, `UncertaintyReport`, etc. are all `@dataclass` with frozen where appropriate.

### Code Quality Weaknesses

1. **`learner.py` is Too Large**: At 1644 lines, it violates the Single Responsibility Principle. `predict()`, `correct()`, `save()`, `load()`, and 10+ internal methods are crammed into one file. This file should be split into `predictor.py`, `corrector.py`, `persistence.py`.

2. **Bare `except:` in Critical Path**: `_apply_buffer_management()` in `learner.py` (~line 1400) uses bare `except:` which catches `KeyboardInterrupt` and `SystemExit`. This is explicitly documented as a limitation in the v0.2.1 roadmap.

3. **Magic Numbers**: Several places use unlabeled numeric constants:
   - `0.9` and `0.1` for EMA prototype updates
   - `1e-4` for early stopping threshold in contrastive training
   - `2.0` as Mahalanobis OOD threshold
   - These should be named constants or config fields.

4. **String-Based Dispatch**: `inference_mode="prototypical"` / `"nearest_neighbor"` / `"contrastive"` and `similarity_metric="cosine"` / `"euclidean"` use string comparisons for routing. This is fragile — a typo silently falls through to a default path.

5. **Inconsistent Error Handling**: Some methods raise custom exceptions (`AdaptShotError`, `ConfigValidationError`), others raise bare `ValueError` or `RuntimeError`. There's no consistent error taxonomy.

6. **No Structured Logging**: Print statements and `warnings.warn()` are used instead of a proper logging framework. Production deployments need log levels, structured output, and configurable handlers.

7. **`utils/io.py` Hard Torch Import**: Line 9 has `import torch` at module level, breaking the torch-optional guarantee.

### Code Quality Score: 62/100

### Code Quality Recommendations

1. Split `learner.py` into `predictor.py` (~400 lines), `corrector.py` (~300 lines), `persistence.py` (~200 lines), `prototype_manager.py` (~300 lines)
2. Replace bare `except:` with specific exception handling
3. Move all magic numbers to `config/settings.py` or module-level constants
4. Use `Enum` for `inference_mode`, `similarity_metric`, `calibration_method`, `conformal_mode`
5. Standardize error hierarchy: all AdaptShot exceptions inherit from `AdaptShotError`
6. Add structured logging with `logging` module
7. Fix `utils/io.py` hard import (P1.4 in roadmap)

---

# Performance Review

## Performance Assessment

AdaptShot's performance is **adequate for its target deployment (single-image inference on CPU) but unacceptable for batch or research workloads**.

### Performance Strengths

1. **CPU-Optimized NumPy**: `cosine_similarity_numpy()` is a well-vectorized implementation using `np.dot` and `np.linalg.norm`. For small support sets (<100, typical for few-shot), it's fast.

2. **FAISS Integration**: Optional FAISS acceleration provides ~10x speedup on large support sets when FAISS is installed.

3. **Eco-Mode Early Exit**: The 32×32 preview signature can skip full-resolution embedding extraction when a confident match is found, saving ~60% compute.

4. **Embedding Caching**: Support embeddings are computed once at `load_support_images()` time and cached. `predict()` only extracts the query embedding.

5. **LSH Acceleration for UP-UGF**: When buffer exceeds 100 examples, redundancy scoring switches from O(N²) exact computation to O(N log N) LSH approximation.

### Performance Weaknesses

1. **No Batched Backbone Forward Pass**: Each `extract_embedding()` call runs one image through the backbone. For 10 images, this is 10 independent forward passes instead of one batch of 10. Overhead: ~5-8x slower than batched.

2. **FAISS Index Rebuilt Per Query**: `_euclidean_top1_faiss()` in `similarity.py` creates a new `faiss.IndexFlatL2` for every query. Index construction (`index.add()`) is O(N*d) and should be done once.

3. **O(N²) Self-Calibration**: `_self_calibrate_conformal()` recomputes prototypes for each calibration sample via leave-one-out, which is O(N²) in support set size. For 100 support examples, this is 100 × 99 prototype recomputations.

4. **Temperature Grid Search**: 25 candidates in [0.5, 3.0] evaluated via full ECE computation on the sliding window. Each evaluation is O(window_size). Total: O(25 × window_size × n_bins).

5. **No Threading/Parallelism**: All operations are single-threaded. `predict()` cannot leverage multi-core CPUs for embedding extraction (which is embarrassingly parallel).

6. **NumPy Copy Overhead**: `tensor_to_numpy()` in `io.py` copies data from torch tensor to numpy array. The backbone produces torch tensors, which are then copied to numpy for similarity search. This copy is unnecessary if the search operated on torch tensors.

7. **Memory Fragmentation**: `np.stack(support_embeddings)` creates a new contiguous array from a list of individually-allocated embeddings. For large support sets, this doubles memory usage temporarily.

### Performance Benchmarks (Estimated from Code Analysis)

| Operation | Current | Batched/Fixed | Speedup |
|---|---|---|---|
| 10-image prediction | ~200ms (serial) | ~40ms (batched backbone) | 5x |
| 100-support LOO calibration | ~500ms (O(N²)) | ~50ms (approximate LOO) | 10x |
| Temperature optimization | ~100ms (grid search) | ~10ms (L-BFGS) | 10x |
| FAISS search (per query) | ~1ms (rebuild index) | ~0.1ms (cached index) | 10x |
| UP-UGF for 200 buffer | ~200ms (exact) | ~20ms (LSH) | 10x |

### Performance Score: 45/100

### Suggested Performance Optimizations

1. **Batch Embedding**: Modify `extract_embedding()` to accept `List[Image]` and run a single backbone forward pass with batch dimension
2. **FAISS Index Caching**: Build FAISS index once when support set changes, reuse for all queries
3. **Approximate LOO Calibration**: Use k-fold cross-validation (k=5) instead of leave-one-out
4. **Continuous Temperature Optimization**: Replace grid search with `scipy.optimize.minimize_scalar` (L-BFGS)
5. **Multi-Threaded Support Extraction**: Use `concurrent.futures.ThreadPoolExecutor` for parallel embedding extraction during `load_support_images()`
6. **Memory-Mapped Embeddings**: For large support sets (>1000), use `np.memmap` instead of in-memory arrays
7. **Torch-Native Similarity Search**: Add torch-native `torch.cdist()` and `torch.mm()` paths when torch is available, avoiding NumPy copy

---

# Research Review

## Research Quality Assessment

AdaptShot's research foundation is **stronger than its engineering execution**. The library demonstrates genuine engagement with the ML literature, and the research agenda document shows ambitious but achievable publication goals.

### Research Strengths

1. **Well-Cited Methods**: The library cites specific papers for each algorithm (Snell et al. 2017 for prototypical networks, Guo et al. 2017 for calibration, Kumar et al. 2019 for scaling-binning). The references are appropriate and current.

2. **Conformal Prediction Implementation**: The split-conformal and cross-conformal modes with finite-sample correction are correctly implemented. The leave-one-out calibration (v0.2.0 fix) is methodologically sound.

3. **Shrinkage Covariance Estimation**: Using Ledoit-Wolf-style shrinkage with `alpha = d/(d+n_k)` for Mahalanobis OOD in few-shot settings is a genuinely clever adaptation that addresses the "fewer samples than dimensions" problem.

4. **Three-Signal Uncertainty**: Separating epistemic (model uncertainty), aleatoric (data uncertainty), and distributional (OOD uncertainty) signals follows best practices in uncertainty quantification literature.

5. **Research Agenda**: The 15-paper research agenda in `RESEARCH_AGENDA_2026_2028.md` is remarkably detailed and well-structured, with clear problem statements, novel contributions, target venues, and alignment with AdaptShot development.

### Research Weaknesses

1. **No Empirical Validation**: None of the claimed benefits of conformal prediction, calibration, or uncertainty quantification are validated on standard benchmarks. The library makes theoretical claims ("coverage ≥ 1-alpha") but provides no evidence they hold in practice.

2. **Epistemic Uncertainty is a Proxy**: The `estimate_epistemic()` method adds Gaussian noise to embeddings and measures sensitivity. This is a perturbation proxy, not true epistemic uncertainty (which requires MC Dropout, deep ensembles, or Bayesian methods). The v0.2.1 roadmap acknowledges this.

3. **No Ablation Studies**: The library adds 9 subsystems without measuring their individual contributions. Is ACT gating actually improving decisions? Does conformal prediction provide meaningful sets? Is contrastive prototype learning better than plain prototypical? No ablation data exists.

4. **Missing SOTA Comparisons**: The benchmark script includes baseline numbers from published papers (Prototypical Networks: 68.20% on miniImageNet 5-way 5-shot) but doesn't run AdaptShot against these baselines. The script explicitly says "Results are NOT directly comparable to published SOTA" — but they should be.

5. **No Theoretical Analysis**: The code implements algorithms from papers but doesn't extend theory. The conformal prediction uses standard nonconformity scores. The calibration uses standard temperature scaling. There's no novel theoretical contribution in the current codebase.

6. **Research Debt**: The v0.2.1 roadmap identifies 49 limitations but provides only high-level solutions. The actual algorithmic implementation of multi-modal prototypes, episodic training, and ordinal feedback requires significant research and experimentation.

### Research Quality Score: 72/100

### Suggested Research Projects (v0.3.0 and Beyond)

1. **Multi-Modal Prototypical Networks**: As described in Paper 1 of the research agenda. Implement k-means prototype clustering with BIC-based automatic k selection.

2. **Episodic Calibration**: Fine-tune backbone on few-shot episodes for better calibration (Paper 2). This is the single highest-impact research project.

3. **Conformal Prediction Under Distribution Shift**: Theoretically characterize how domain shift affects conformal coverage and design adaptive nonconformity scores (Paper 5).

4. **Ordinal Human Feedback**: Build a system that learns from "closer to A than B" feedback rather than categorical labels (Paper 3).

5. **Carbon-Aware ML**: Quantify the Pareto frontier between accuracy and carbon emissions in few-shot learning (Paper 13).

---

# Mathematical Review

## Mathematical Correctness Assessment

The mathematical implementations in AdaptShot are **generally correct** but contain edge cases and simplifications that undermine rigor.

### Verified Correct Implementations

1. **ECE Computation**: `compute_ece()` correctly implements \( \text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} |\text{acc}(B_m) - \text{conf}(B_m)| \), using equal-mass binning.

2. **Temperature Scaling**: `_fit_temperature()` correctly minimizes NLL: \( T^* = \arg\min_T -\sum_i \log(\text{softmax}(\text{logit}_i / T)) \). The grid search over [0.5, 3.0] with 25 candidates is correct but coarse.

3. **Conformal Quantile**: `_compute_quantile()` correctly implements \( \hat{q} = \text{Quantile}(\{s_i\}_{i=1}^{n}, \frac{\lceil (n+1)(1-\alpha) \rceil}{n}) \) with finite-sample correction.

4. **InfoNCE Loss**: `compute_infonce_loss()` correctly implements \( L = -\log \frac{\exp(s_{i,i}/\tau)}{\sum_{j}\exp(s_{i,j}/\tau)} \). The gradient computation in `_compute_gradients()` appears mathematically correct but is implemented manually rather than via autograd.

5. **Mahalanobis Distance**: `mahalanobis_distance()` correctly implements \( D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)} \) with shrinkage: \( \Sigma_{\text{shrunk}} = (1-\lambda)\Sigma_{\text{sample}} + \lambda \cdot \text{diag}(\Sigma_{\text{sample}}) \), where \( \lambda = \frac{d}{d + n_k} \).

6. **UP-UGF Utility Score**: `_compute_utility()` correctly implements the multiplicative composite: \( U = u_{\text{uncertainty}}^\alpha \times u_{\text{recency}}^\beta \times (1 - u_{\text{redundancy}})^\gamma \).

### Mathematical Issues Identified

1. **Distance-to-Confidence Mapping (MEDIUM)**: `distance_to_confidence()` uses \( c = \frac{1}{1 + d} \). This is a naive mapping that doesn't account for the distribution of distances within vs across classes. A proper calibration would use: \( P(y = \text{class}_k \mid x) = \frac{\exp(-\beta \cdot d(x, \mu_k))}{\sum_j \exp(-\beta \cdot d(x, \mu_j))} \) where \( \beta \) is learned via temperature scaling.

2. **Conformal Set Always Includes Top Prediction (MEDIUM)**: `predict_set()` forces the top prediction into the set regardless of nonconformity score. This can violate the coverage guarantee: if the true class has very high nonconformity, it should be excluded from the set. The theoretical guarantee is \( P(Y_{\text{test}} \in C(X_{\text{test}})) \geq 1 - \alpha \), which requires honest sets. Forcing the top prediction may produce sets that are too large (overly conservative) but shouldn't violate coverage.

3. **Aleatoric Uncertainty via k-NN Entropy (LOW)**: `compute_knn_entropy()` computes entropy over the k-nearest neighbor label distribution. The normalization to [0,1] divides by \( \log(k) \) (maximum possible entropy). This is correct only when all k neighbors could belong to different classes — which requires k ≤ number of classes. For k > n_classes, the maximum entropy is \( \log(n_classes) \), not \( \log(k) \).

4. **Shrinkage Alpha Derivation (LOW)**: The shrinkage parameter `alpha = d/(d + n_k)` is a heuristic inspired by Ledoit-Wolf but not the actual Ledoit-Wolf estimator (which involves trace and Frobenius norm computations). This is documented honestly as "Ledoit-Wolf-style" but the mathematical approximation should be analyzed for correctness in high-dimensional settings.

5. **Counterfactual Distance Normalization (LOW)**: Counterfactual analysis normalizes distances by dividing by the range of distances. When all distances are identical (degenerate support set), this produces division by zero.

### Mathematical Score: 68/100

---

# Security Review

## Security Assessment

AdaptShot's attack surface is **appropriately small for a local-only library**, but several issues require attention.

### Security Strengths

1. **Local-Only by Design**: No network calls, no cloud APIs, no telemetry, no analytics. The library cannot exfiltrate data.

2. **SHA-256 Integrity**: Checkpoints include SHA-256 hashes verified on load. Tampered checkpoints are rejected with a clear error.

3. **No Pickle Deserialization**: The library uses JSON for metadata and `.npy` for embeddings. It does not use `pickle.load()` or `torch.load()` unsafely.

4. **Input Validation**: Image loading validates file existence, expected channel count (3-channel RGB), and label/image count consistency.

5. **No Dynamic Code Execution**: No `eval()`, `exec()`, or `__import__()` calls on user-provided data.

### Security Issues

1. **File Path Injection (LOW)**: `load_support_images()` and `save()` accept arbitrary file paths. In a web server deployment where paths come from user input, path traversal attacks (`../../../etc/passwd`) could read/write outside intended directories. **Recommendation**: Validate paths with `resolve()` and check they're within allowed directories.

2. **YAML Deserialization in MziziGuard (LOW)**: `MziziGuard._load_config()` uses `yaml.safe_load()` which is safe. However, if `yaml.load()` (unsafe) were ever substituted, it would enable arbitrary code execution. **Recommendation**: Pin the safe loader and add a security test that verifies `yaml.safe_load` is used.

3. **No Input Size Limits (LOW)**: `load_support_images()` accepts unlimited numbers of images. An attacker could cause OOM by providing millions of file paths. **Recommendation**: Add `max_support_images` config field with a reasonable default (e.g., 10,000).

4. **Temp File Cleanup (LOW)**: Demo scripts (`crop_disease_demo.py`, `day4_integration.py`) create temp directories but don't always clean up. In a production deployment, temp file accumulation could exhaust disk space.

### Security Score: 80/100

---

# Documentation Review

## Documentation Assessment

AdaptShot's documentation is **remarkably comprehensive for a v0.2.0 library** — 42+ markdown files, 20 tutorials, 14 guides, full API reference — but suffers from accuracy issues and a implementation-documentation gap.

### Documentation Strengths

1. **Volume and Breadth**: 42+ markdown files covering installation, quickstart, tutorials (1-20), guides (14 topics), API reference, config reference, and contributing guide. This exceeds the documentation of many production libraries.

2. **MkDocs with Material Theme**: Professional documentation site with search, code highlighting, navigation, and dark mode. Well-configured `mkdocs.yml` with `mkdocstrings` auto-generated API docs.

3. **Tutorial Progression**: Tutorials 01-20 follow a logical learning curve from "Getting Started" through "ONNX Deployment." Each tutorial builds on previous ones.

4. **Config Reference**: The 27-field `AdaptShotConfig` reference with types, defaults, and descriptions is thorough.

5. **Migration Guide**: `migration-v0.1-to-v0.2.md` documents breaking changes between versions. This is professional-grade documentation that most small libraries neglect.

6. **Swahili Localization**: The MziziGuard demo includes Swahili translations for disease names and treatment advice, serving the target Tanzanian audience.

### Documentation Weaknesses

1. **Accuracy Gap**: Several documented features have been changed in code without updating docs:
   - Older docs mention "gradient-based saliency" — code implements "embedding-space saliency" (fixed in CHANGELOG but some docs may be stale)
   - MC Dropout for epistemic uncertainty was documented but replaced with perturbation proxy
   - `inference_mode="contrastive"` was documented as working before it was actually implemented (fixed in v0.2.0-dev)

2. **Missing Comparison Benchmarks**: Documentation mentions benchmark results but doesn't provide comparison tables against SOTA methods on standard datasets.

3. **No API Stability Guarantees**: The documentation doesn't specify which APIs are stable (semver-major) vs experimental.

4. **No Changelog in Docs Site**: CHANGELOG.md exists in the repo but isn't included in the MkDocs navigation.

5. **Tutorial-Code Desync**: Tutorial code snippets may not match the current API if they were written for v0.1.x. No version-tagged code blocks.

6. **No Interactive Examples**: No Jupyter notebooks or Binder links for interactive experimentation.

### Documentation Score: 78/100

---

# Benchmark Review

## Benchmark Assessment

The benchmarking infrastructure is **minimal but well-designed**. It needs significant expansion for research credibility.

### Current Benchmarks

1. **`run_benchmark.py`** (smoke test): CIFAR-10 5-way 10-shot classification with accuracy and latency metrics. Includes baseline references from published papers but doesn't run them.

2. **`energy_profile.py`**: Deterministic energy/CO₂ profiling with wall-clock time, memory (tracemalloc + psutil), CPU frequency/utilization, Joule estimation, and CO₂ estimation. Compares baseline vs eco-mode.

3. **`day2_integration.py`**: Simulated HITL loop with calibration and feedback routing over 15 steps.

4. **`day3_integration.py`**: Full continuous learning loop with ACT, calibration, feedback router, and CA-EWC finetuner over 20 simulation steps.

5. **`day4_integration.py`**: End-to-end FewShotLearner workflow with synthetic dataset, prediction, correction, save/load.

### Benchmark Gaps

1. **No Standard Few-Shot Benchmarks**: miniImageNet (100 classes, 600 per class) and tieredImageNet (608 classes) are the standard few-shot benchmarks. Neither is part of the automated benchmark suite.

2. **No Calibration Benchmarks**: CIFAR-100 ECE, reliability diagrams, confidence histograms — absent. The calibration engine has no quantitative validation.

3. **No OOD Benchmarks**: No OOD detection AUROC or F1 metrics against standard OOD benchmarks.

4. **No Conformal Coverage Validation**: The `ConformalEngine` claims 95% coverage but this isn't empirically validated.

5. **No Latency Benchmarks on Target Hardware**: All benchmarks run on the developer's machine. No Raspberry Pi 4, Intel NUC, or low-end laptop benchmarks.

6. **No Memory Benchmarks**: The <250MB RAM constraint is never actually measured in a structured benchmark, only estimated via `estimate_model_memory_mb()`.

7. **Synthetic Data Only**: `day2-4_integration.py` use synthetic data (numpy random). No real-world datasets are used for integration testing.

### Benchmark Score: 35/100

### Suggested Benchmarks

1. **miniImageNet Few-Shot Classification**: 5-way 1-shot, 5-way 5-shot, 20-way 5-shot, 600 episodes each. Report mean ± std accuracy.

2. **CIFAR-100 Calibration**: ECE, MCE, reliability diagrams. Compare temperature scaling vs no calibration.

3. **CIFAR-10-C Domain Shift**: 10 classes × 19 corruption types. Measure accuracy degradation and calibration drift.

4. **OOD Detection**: CIFAR-10 vs SVHN, CIFAR-100 vs CIFAR-10. Report AUROC, FPR95, AUPR.

5. **Conformal Coverage**: Empirical coverage vs target on CIFAR-10, miniImageNet. Measure set size vs coverage.

6. **Latency-Memory Matrix**: Measure latency and peak memory for support sizes {5, 10, 50, 100, 500, 1000} on 3 hardware configurations.

7. **Energy Profile**: Measure Joules/inference on Raspberry Pi 4 vs Intel laptop vs AMD desktop.

---

# Testing Review

## Testing Assessment

The test suite has **good API coverage but insufficient depth** for a library making mathematical and statistical claims.

### Test Suite Statistics

- **13 test files**, ~92 tests (estimated from file contents)
- Coverage targets: core modules (calibration, conformal, contrastive, exceptions, explain, extractor, feedback_router, learner integration, persistence, release_metadata, similarity, studio_utils, uncertainty)
- Missing test files: tests for `act.py`, `finetune.py`, `up_ugf.py`, `determinism.py`, `io.py`, `migrations.py`, `profiling.py`, `onnx_backend.py`, `ui/app.py`

### Testing Strengths

1. **Good API Coverage**: Every public class has corresponding tests. `test_calibration.py` tests initial state, ECE computation, sliding window, temperature refitting, calibration scaling, conformal stub, empty input.

2. **Integration Tests**: `test_learner_integration.py` tests the full pipeline: load → predict → correct → save → load → predict. `test_persistence.py` tests corrupted files, version migration, integrity verification.

3. **Determinism Tests**: `test_extractor.py` includes `test_deterministic_extraction()` that verifies bit-exact embeddings across 3 runs.

4. **Edge Cases**: Empty arrays, mismatched lengths, missing files, invalid inputs — tests cover error paths.

5. **Monkeypatch-Based Isolation**: Tests use `monkeypatch` to replace `extract_embedding()` with deterministic stubs, avoiding slow backbone loads in CI.

### Testing Weaknesses

1. **No Numerical Stability Tests**: No tests for NaN propagation, inf handling, division by zero, catastrophic cancellation. Critical for a library doing manual gradient computation.

2. **No Property-Based Testing**: No Hypothesis or pytest-randomly tests. Property-based tests would catch edge cases that manual test cases miss (e.g., "for any two embeddings, cosine similarity ∈ [-1, 1]").

3. **No Statistical Validation Tests**: No tests that verify conformal coverage empirically (e.g., "with 100 calibration samples and α=0.1, empirical coverage should be ≥0.9 within 95% CI"). No tests that verify ECE is well-behaved.

4. **No Platform-Specific Tests**: All tests assume x86_64 Linux. No ARM tests (Raspberry Pi), no macOS tests, no Windows tests.

5. **No Performance Regression Tests**: No tests that assert latency or memory stay within bounds. Performance can silently degrade.

6. **Missing Unit Tests for 8 Modules**: `act.py`, `finetune.py`, `up_ugf.py`, `determinism.py`, `io.py`, `migrations.py`, `profiling.py`, `onnx_backend.py`, `ui/app.py` have no corresponding test files.

7. **No Fuzzing**: No input fuzzing for image inputs (corrupted PNGs, malicious JPEGs, extremely large images).

8. **No FAISS Integration Tests**: When FAISS is installed, the FAISS path is not systematically tested. The `test_similarity.py` only tests `use_faiss=False`.

### Testing Score: 55/100

### Suggested Testing Improvements

1. Add property-based tests with `hypothesis`: at least one test per module
2. Add numerical stability test suite: edge cases for every mathematical function
3. Add statistical validation tests for conformal coverage and calibration ECE
4. Add performance regression tests with `pytest-benchmark`
5. Add platform matrix to CI: ubuntu, macos, windows × python 3.9-3.12
6. Add tests for all 8 missing modules
7. Add FAISS integration tests (skip if FAISS not installed)

---

# Packaging Review

## Packaging Assessment

The packaging is **professional and well-configured** with minor issues.

### Packaging Strengths

1. **PEP 621 pyproject.toml**: Modern packaging standard with `[project]` metadata and `[tool.setuptools]` build configuration.

2. **Optional Dependencies**: Well-structured extras: `[torch]`, `[faiss]`, `[ui]`, `[gui]`, `[dev]`. Users install only what they need.

3. **Package Data**: ONNX model files bundled via `[tool.setuptools.package-data]`.

4. **Version Management**: Single source of truth — version in both `pyproject.toml` and `__init__.py`, verified by `test_release_metadata.py`.

5. **PyPI Publishing**: v0.1.0 reached 574 downloads across 30+ countries. PyPI metadata is complete with description, classifiers, and project URLs.

### Packaging Issues

1. **`torch` in Core Dependencies**: `pyproject.toml` lists `torch>=2.0.0` in core dependencies, but the codebase has lazy imports. The CHANGELOG says "PyTorch and torchvision moved to optional dependencies" for v0.1.2, but `pyproject.toml` still has them as required. **This is a packaging bug.**

2. **`pyproject.toml` vs CHANGELOG Discrepancy**: CHANGELOG says "Core dependencies reduced from 4 to 2 (numpy, Pillow)" but `pyproject.toml` lists `torch>=2.0.0` and `torchvision>=0.15.0` in `dependencies`, not `optional-dependencies`.

3. **No `[torch]` Extra Defined**: CHANGELOG mentions optional `[torch]` extra but `pyproject.toml` doesn't define it. Torch is listed as required.

4. **No `python_requires`**: `pyproject.toml` doesn't specify minimum Python version. The library uses Python 3.9+ features (PEP 585 generics), so `python_requires = ">=3.9"` should be specified.

5. **No Wheel Size Optimization**: The `[tool.setuptools]` config doesn't exclude test files or benchmarks from the wheel. This bloats the distribution.

### Packaging Score: 70/100

---

# Developer Experience Review

## Developer Experience Assessment

The developer experience is **adequate for the current sole developer but needs improvement for community contribution**.

### DX Strengths

1. **Clear Contribution Guide**: `CONTRIBUTING.md` with setup instructions, coding standards, and PR process.

2. **Pre-Commit Quality Gates**: Ruff, mypy, and pytest with clear commands. All validation can run locally before pushing.

3. **AGENTS.md**: Instructions for AI agents working on the project. Forward-thinking.

4. **`.openproject.md`**: Constitution for anyone working on the project. Unusual but effective for maintaining design coherence.

### DX Weaknesses

1. **No Pre-Commit Hooks**: `.pre-commit-config.yaml` is absent. Developers must manually remember to run ruff and mypy.

2. **No Development Container**: No `.devcontainer/` or Dockerfile for reproducible development environment.

3. **Slow Test Suite on CI**: Tests load real backbone models (torchvision downloads ~50MB per run). CI must download models on every run.

4. **No Coverage Reporting**: `.coverage` file exists but CI doesn't upload coverage to a service (Codecov, Coveralls).

5. **No Issue Templates**: No `.github/ISSUE_TEMPLATE/` for bug reports or feature requests.

6. **No PR Template**: No `.github/PULL_REQUEST_TEMPLATE.md` with checklist.

### Developer Experience Score: 65/100

---

# User Experience Review

## User Experience Assessment

The user experience is **surprisingly good for a CLI-first library** but the Studio/Gradio UI adds substantial value.

### UX Strengths

1. **6-Method API**: Learnable in minutes. `init → load_support_images → predict → correct → calibration_report → save/load`.

2. **AdaptShot Studio**: The 8-tab Gradio application provides visual interfaces for configuration, dataset loading, inference, correction, calibration monitoring, buffer management, export, and diagnostics.

3. **MziziGuard Demo**: The crop disease demo with interactive presentation mode and Swahili localization shows the library's real-world potential.

4. **Meaningful Error Messages**: `ConfigValidationError`, `InvalidImageError`, `AdaptShotError` with descriptive messages.

5. **PredictionResult**: Returns a structured dataclass with prediction, confidence, uncertainty flags, OOD flags, ACT action, and explanation — all in one object.

### UX Weaknesses

1. **No Progress Indicators**: `load_support_images()` can take minutes for large support sets but provides no progress feedback.

2. **No Configuration Wizard**: Users must create `AdaptShotConfig` with 26 fields. A configuration builder or wizard would help new users.

3. **Studio Requires Torch**: The Studio app (1029 lines) imports torch modules. If torch is not installed, the Studio is unavailable — contradicting the torch-optional design.

4. **No Mobile App**: Despite targeting farmers in Tanzania, there's no mobile (Android) deployment path. ONNX export exists but no Android wrapper.

5. **Gradio UI is Single-User**: The Studio uses module-level state (`_learner` as global), making it unsuitable for multi-user deployments.

### User Experience Score: 65/100

---

# Missing Features

The following features are standard in comparable ML libraries but absent from AdaptShot:

1. **Batch Prediction** (Critical): `predict_batch()` for processing multiple images efficiently
2. **Episodic Training** (Critical): Fine-tuning backbone on few-shot episodes
3. **Multi-Modal Prototypes** (High): k prototypes per class via clustering
4. **`predict_proba()` API** (High): Scikit-learn compatible probability prediction
5. **Data Augmentation** (High): Configurable augmentations during support set loading
6. **GPU Acceleration** (Medium): Optional CUDA/MPS backend for training
7. **Plugin Architecture** (Medium): `register_backbone()`, `EmbeddingBackend` protocol
8. **Async API** (Medium): `async def predict()` for web server deployments
9. **Domain Adaptation** (Medium): CORAL, adversarial training for domain shift
10. **Grad-CAM / Pixel Attribution** (Medium): Visual explanations showing important pixels
11. **Learning Rate Scheduling** (Medium): Cosine annealing, ReduceLROnPlateau for fine-tuning
12. **Cross-Validation** (Medium): Built-in k-fold evaluation for few-shot tasks
13. **Metric Learning** (Medium): Learned distance metrics beyond Euclidean/Cosine
14. **Automatic K-Shot Detection** (Low): Auto-detect k_shot from support set
15. **Model Zoo** (Low): Pre-trained backbones downloadable on demand
16. **Notebook Integration** (Low): `%adaptshot` magic or ipywidgets for Jupyter
17. **Feature Importance Plotting** (Low): matplotlib/seaborn integration for explanations
18. **Export to CoreML/TFLite** (Low): Mobile-optimized model export
19. **Multilingual i18n** (Low): Built-in localization framework beyond Swahili

---

# Industry Comparison

## Comparison Against Major ML Libraries

### vs scikit-learn
- **Strengths**: AdaptShot's immutable config and human-in-the-loop focus are novel. scikit-learn has no conformal prediction or OOD detection.
- **Weaknesses**: scikit-learn's `Pipeline`, `BaseEstimator`, parameter validation, and 10+ years of API refinement are unmatched. AdaptShot's NumPy-first training is primitive compared to scikit-learn's optimized C/Cython implementations.
- **Verdict**: Complementary, not competitive. AdaptShot should aim to be scikit-learn-compatible.

### vs PyTorch
- **Strengths**: AdaptShot is vastly simpler to use for few-shot learning. PyTorch requires ~200 lines for a prototypical network; AdaptShot does it in 6.
- **Weaknesses**: PyTorch's ecosystem (timm, torchvision, Lightning, HuggingFace) provides everything AdaptShot does and more, just with more code.
- **Verdict**: AdaptShot fills a genuine gap: turnkey few-shot learning. But it must support PyTorch backends for credibility.

### vs HuggingFace Transformers
- **Strengths**: HuggingFace doesn't do few-shot vision. AdaptShot's conformal prediction + uncertainty + HITL is unique.
- **Weaknesses**: HuggingFace's `pipeline()`, `AutoModel`, and model hub are vastly more mature. AdaptShot's 2-backbone registry is laughable compared to HuggingFace's 200,000+ models.
- **Verdict**: Different domains. AdaptShot should study HuggingFace's registry and pipeline patterns.

### vs FAISS
- **Strengths**: FAISS is a search library, not an ML library. AdaptShot provides the complete pipeline.
- **Weaknesses**: FAISS is 100-1000x faster on large-scale similarity search than AdaptShot's NumPy implementation.
- **Verdict**: AdaptShot correctly uses FAISS as an optional accelerator. This is good engineering.

### vs LightGBM / XGBoost
- **Strengths**: Gradient boosting libraries don't do few-shot vision. No overlap.
- **Verdict**: Not directly comparable.

### vs ONNX Runtime
- **Strengths**: AdaptShot provides ONNX export and runtime as a deployment target. ONNX Runtime alone doesn't provide few-shot learning.
- **Verdict**: AdaptShot + ONNX Runtime is a powerful combination for edge deployment.

### vs JAX
- **Strengths**: JAX provides autograd and JIT compilation. AdaptShot provides turnkey few-shot.
- **Weaknesses**: JAX's `vmap`, `pmap`, `jit`, and functional transforms are far more powerful than AdaptShot's NumPy implementation.
- **Verdict**: AdaptShot should consider JAX as a NumPy replacement for CPU-optimized training.

### vs TensorFlow
- **Strengths**: AdaptShot is much simpler and lighter weight.
- **Weaknesses**: TensorFlow Lite provides mobile deployment that AdaptShot lacks.
- **Verdict**: Different weight classes. AdaptShot should stay focused on CPU-first.

### vs Existing Few-Shot Learning Libraries
- **EasyFSL**, **torchFewShot**, **few-shot**: These are PyTorch-only, GPU-assuming, academic libraries. None provide calibration, conformal prediction, OOD detection, uncertainty, explainability, or human-in-the-loop routing.
- **Verdict**: AdaptShot's integrated pipeline (few-shot + calibration + conformal + uncertainty + HITL) is genuinely unique in open source. This is the library's strongest differentiator.

### vs Edge AI Libraries
- **OpenVINO**, **TensorFlow Lite**, **Core ML**, **ONNX Runtime**: These provide inference runtimes, not learning algorithms. AdaptShot's ONNX backend enables deployment via any of these.
- **Verdict**: AdaptShot + ONNX Runtime is competitive for edge deployment. Missing mobile wrappers.

---

# Roadmap Recommendations

## v0.3.0 (Critical Infrastructure — 3 months)

1. **Fix CI**: Remove `|| true` from mypy, fix all type errors
2. **Fix Packaging**: Move torch to optional dependencies, define `[torch]` extra
3. **Batch Prediction**: `predict_batch()` with vectorized backbone and similarity
4. **FAISS Index Caching**: Rebuild index only on support set changes
5. **Fix `utils/io.py`**: Remove hard torch import
6. **Add `Enum`-Based Dispatch**: Replace string-based routing with enums
7. **Split `learner.py`**: Extract predictor, corrector, persistence modules
8. **Add Missing Tests**: ACT, CA-EWC, UP-UGF, determinism, io, migrations, profiling, ONNX
9. **Benchmark Expansion**: miniImageNet, CIFAR-100 calibration, OOD detection

## v0.4.0 (Plugin Architecture — 3 months)

1. **`EmbeddingBackend` Protocol**: Plugin system for backbones
2. **`register_backbone()` API**: User-extensible backbone registry
3. **Standard Benchmarks**: Full miniImageNet, tieredImageNet, CUB-200 benchmarks
4. **Episodic Training**: Optional backbone fine-tuning on support episodes
5. **Multi-Modal Prototypes**: k-means prototypes with automatic k selection
6. **CI Platform Matrix**: Ubuntu + macOS + Windows, Python 3.9-3.12

## v1.0.0 (Production Grade — 6 months)

1. **GPU Backend**: Optional CUDA/MPS acceleration for training
2. **Stable API**: Freeze `FewShotLearner` API as semver-major
3. **Peer-Reviewed Publication**: Submit methodology paper
4. **Field Validation**: 3+ NGO partnerships with deployment metrics
5. **Mobile Deployment**: Android/iOS wrappers via ONNX Runtime mobile
6. **Carbon-Neutral CI**: Verified carbon offsetting

---

# Quick Wins (1-2 Days Each)

1. Fix CI `|| true` — 2 hours
2. Fix `utils/io.py` hard import — 1 hour
3. Add `python_requires = ">=3.9"` to `pyproject.toml` — 30 min
4. Move torch to optional deps in `pyproject.toml` — 30 min
5. Add `__post_init__` config validation — 4 hours
6. Replace bare `except:` in `_apply_buffer_management()` — 1 hour
7. Add `CorrectionResult` dataclass — 2 hours
8. Add `.pre-commit-config.yaml` — 2 hours
9. Add issue and PR templates — 2 hours
10. Add `predict_proba()` method — 4 hours

---

# Medium-Term Improvements (1-4 Weeks Each)

1. **Enum-Based Dispatch**: Replace string routing with enums — 3 days
2. **Structured Logging**: Replace print/warnings with `logging` — 2 days
3. **Batch Prediction**: Full implementation — 3 days
4. **FAISS Index Caching**: Per-support-set cache — 2 days
5. **Numerical Stability Tests**: Hypothesis-based property tests — 1 week
6. **Split `learner.py`**: Modularize 1644-line file — 1 week
7. **Platform Matrix CI**: Multi-OS testing — 3 days
8. **Standard Benchmarks**: miniImageNet + calibration + OOD — 2 weeks
9. **Documentation Audit**: Verify all code snippets against current API — 1 week

---

# Long-Term Vision (1-2 Years)

1. **PyTorch Backend Rewrite**: Replace NumPy training with PyTorch autograd
2. **JAX/Numba CPU Acceleration**: Optional JIT-compiled computation
3. **Plugin Ecosystem**: Community-contributed backbones, calibration methods, conformal scores
4. **Mobile App**: Android app wrapping ONNX Runtime with AdaptShot
5. **Federated Learning**: Privacy-preserving multi-device buffer aggregation
6. **Neuromorphic Backends**: Intel Loihi, event-based vision when hardware matures
7. **Population-Scale Deployment**: National-level agricultural AI in Tanzania
8. **Research Institute**: Establish AdaptShot Lab as a center for constraint-first AI research

---

# Suggested Research Papers

1. "Multi-Modal Prototypes for Few-Shot Learning" — CVPR 2027
2. "Episodic Calibration: Adapting Networks for Few-Shot Conformal Prediction" — ICML 2027
3. "Ordinal Feedback for Continual Few-Shot Learning" — AAAI 2027
4. "Constraint-First AI: How Resource Limits Improve Generalization" — NeurIPS 2027 (Position Paper)
5. "Conformal Prediction Under Extreme Few-Shot Conditions" — JMLR 2027
6. "Carbon-Aware Few-Shot Learning: The Accuracy-Emission Pareto Frontier" — Nature Climate Change
7. "Shrinkage Covariance for Few-Shot OOD Detection" — UAI 2027
8. "Human-in-the-Loop Learning with Adaptive Confidence Thresholding" — CHI 2027
9. "AdaptShot: A Library for Trustworthy Few-Shot Learning" — NeurIPS 2027 D&B
10. "Domain Adaptation Without Retraining: Support Sets as Anchors" — CVPR Workshop 2027
11. "Verified Uncertainty Quantification in Resource-Constrained Settings" — ICML Workshop 2027
12. "Locality-Sensitive Hashing for Scalable Continual Few-Shot Learning" — AAAI 2028

---

# Suggested Refactors

1. **`learner.py` → `predictor.py` + `corrector.py` + `persistence.py` + `prototype_manager.py`**
   - Impact: Reduces god-class complexity, enables independent testing
   - Effort: 1 week
   - Risk: Low (internal refactor, API unchanged)

2. **NumPy Training → PyTorch Autograd Training**
   - Impact: 10x maintainability, enables GPU, enables community contribution
   - Effort: 4 weeks
   - Risk: High (breaking change for non-torch users — mitigate with torch-optional pattern)

3. **String Dispatch → Enum Dispatch**
   - Impact: Compile-time error for typos, better IDE support
   - Effort: 3 days
   - Risk: Low

4. **Module-Level State → ContextVar/Thread-Local**
   - Impact: Thread safety for web server deployments
   - Effort: 3 days
   - Risk: Medium (subtle concurrency bugs possible)

5. **`extract_embedding()` API → `EmbeddingBackend` Protocol**
   - Impact: Extensible backbone system
   - Effort: 5 days
   - Risk: Medium (API change for advanced users)

---

# Suggested API Changes

1. **Add `predict_batch(image_paths, batch_size=32) -> List[PredictionResult]`**
2. **Add `predict_proba(image) -> Dict[Any, float]`**
3. **Add `CorrectionResult` dataclass replacing `Dict` return of `correct()`**
4. **Add `async def predict_async(image) -> PredictionResult`**
5. **Rename `load_support_images` → `fit` for scikit-learn compatibility (or add alias)**
6. **Add `register_backbone(name, factory)` function**
7. **Add `list_backbones() -> List[str]` function**
8. **Add `list_available_engines() -> List[str]`**
9. **Add `partial_fit(image_paths, labels)` for incremental support set updates**
10. **Add `get_params() -> Dict` and `set_params(**kwargs)` for scikit-learn compatibility**

---

# Suggested Performance Optimizations

1. **Batched embedding extraction**: `extract_embeddings_batch(images, config)` with torch batch dimension
2. **FAISS index caching**: `_build_faiss_index()` called once, invalidated on support set change
3. **Multi-threaded support extraction**: `ThreadPoolExecutor` for `load_support_images()`
4. **Continuous temperature optimization**: L-BFGS instead of grid search
5. **Memory-mapped embeddings**: `np.memmap` for support sets >1000
6. **Torch-native distance computation**: `torch.cdist()` when torch available
7. **LOO approximation**: k-fold cross-validation instead of leave-one-out
8. **JIT compilation**: `@numba.jit` on hot NumPy loops (cosine similarity, entropy computation)
9. **Embedding quantization**: Int8 quantization for reduced memory footprint
10. **Caching middleware**: LRU cache layer between frontend and backend

---

# Suggested Research Projects

1. **Multi-Modal Prototypes**: Clustering-based prototype learning with automatic k selection
2. **Episodic Calibration**: Backbone fine-tuning that optimizes both accuracy and calibration
3. **Ordinal HITL**: Learning from comparative feedback instead of categorical labels
4. **Conformal Prediction Under Distribution Shift**: Adaptive nonconformity scores for domain-shifted settings
5. **Carbon-Aware AutoML**: Automatically select backbone and hyperparameters to minimize CO₂
6. **Federated Few-Shot Learning**: Privacy-preserving multi-user buffer sharing
7. **Quantized Embeddings for Edge**: Int4/int8 embedding storage with minimal accuracy loss
8. **Uncertainty-Aware Prototype Updates**: Weight prototype updates by epistemic uncertainty
9. **Cross-Lingual Few-Shot**: Transfer learning across languages for agricultural applications
10. **Neuromorphic Few-Shot**: Event-based vision cameras with spiking neural network backends

---

# Final Verdict

AdaptShot v0.2.0 is a **remarkable solo effort with genuine innovations and genuine flaws**. It is simultaneously too ambitious for a v0.2.0 library (implementing 9 ML subsystems when most libraries implement 1-2) and not ambitious enough in its engineering (NumPy training loops, no batched inference, hardcoded backbones).

The library's greatest strength is its integrated pipeline: few-shot learning + calibration + conformal prediction + uncertainty + OOD detection + explainability + HITL routing + continual learning — all CPU-first, memory-bounded, and carbon-aware. No other library in the world provides this combination. This is a genuine competitive advantage.

The library's greatest weakness is its NumPy-first training architecture, which caps maintainability, performance, extensibility, and community contribution. This is a fixable architectural decision — and must be fixed before v1.0.0.

**Recommendation**: AdaptShot should pivot to a **Torch-Optional Architecture** where:
- Inference works with or without torch (ONNX backend for torch-free deployments) ✓ (already achieved)
- Training requires torch but runs on CPU by default (no GPU required)
- The NumPy implementations become reference implementations used only for testing/verification
- The public API (6 methods) remains unchanged

This preserves the CPU-first, memory-bounded philosophy while unlocking the entire PyTorch ecosystem for training, optimization, and community contribution.

**Bottom Line**: If AdaptShot executes the v0.3.0-v1.0.0 roadmap successfully, it has the potential to become the reference implementation for trustworthy, human-aligned, resource-constrained few-shot learning. The mission is right. The vision is right. The engineering needs work — but the author has already identified every major issue in their own self-audit. The path forward is clear.

**Final Score: 64/100 — Diamond in the Rough**

---

*This audit was conducted by a simulated engineering review committee with expertise spanning machine learning research, systems engineering, performance optimization, API design, open-source maintenance, and Python packaging. The analysis is based on a complete reading of all source files (32 files, ~8,000 lines), test files (13 files, ~1,500 lines), benchmark files (5 files, ~1,000 lines), example files (5 files, ~2,000 lines), research documents (2 files, ~1,800 lines), configuration files, CI workflow, and project governance documents.*

*Report generated: July 1, 2026. Target repository: AdaptShot v0.2.0.post0*
