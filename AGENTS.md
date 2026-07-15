# AdaptShot — AI Agent Engineering Manual

> **Version**: 2.0  
> **Applies to**: All AI coding assistants working in this repository  
> **Prerequisite**: Read [`.openproject.md`](.openproject.md) — the project constitution — before doing anything else. This manual operationalizes the constitution into actionable engineering rules.

---

## 1. Project Vision

AdaptShot exists to **democratize trustworthy, CPU-first, human-in-the-loop few-shot vision AI** for resource-constrained environments — the farmer in rural Tanzania, the conservation ranger without internet, the clinic with a 10-year-old laptop.

We are not building yet another PyTorch wrapper. We are proving that:

- AI can be accurate without GPUs, cloud, or massive carbon footprints
- Uncertainty can be measured and reported, not hidden behind overconfident softmax outputs
- Human expertise — not just labeled data — can guide models through corrections
- Open science can outperform closed, resource-hungry systems

Every engineering decision must optimize for the **next 5–10 years**, not the next release.

---

## 2. Core Engineering Principles (MANDATORY)

These are non-negotiable. Violating any one of them is a **design failure**.

| # | Principle | Rule |
|---|-----------|------|
| P1 | **CPU-First** | Every operation defaults to `device="cpu"`. GPU is opt-in, never assumed. |
| P2 | **Memory-Bound** | Full pipeline must operate within **<250 MB RAM**. No unbounded allocations. |
| P3 | **Deterministic** | Fixed seeds produce identical outputs across runs. `verify_determinism()` must pass. |
| P4 | **Human-in-the-Loop** | Corrections are first-class signals — routed, weighted, acted upon. Not an afterthought. |
| P5 | **Transparent** | Every prediction returns calibrated confidence, uncertainty flags, and neighbor metadata. No black boxes. |
| P6 | **Carbon-Aware** | Track Joules/inference and CO₂ estimates in benchmarks. Minimize compute by default. |
| P7 | **Offline-Capable** | Zero internet requirement. No cloud APIs, no telemetry, no SaaS dependencies. |
| P8 | **Torch-Optional** | Core library imports and runs without PyTorch. Inference works via ONNX when torch is absent. Training requires `adaptshot[torch]`. |
| P9 | **Correctness Over Convenience** | Never sacrifice accuracy for speed. If uncertain about an API, halt and verify against `src/adaptshot/`. |
| P10 | **Explicit Over Implicit** | No silent fallbacks. No swallowed exceptions. No magic defaults that change behavior without explicit user opt-in. |

---

## 3. AI Agent Responsibilities

Every AI agent operating in this repository must follow these behavioral rules:

### Before Writing Code
1. **Read surrounding code** — understand the module's existing patterns, naming conventions, and abstractions before modifying anything.
2. **Verify the public API** — cross-reference all class names, method signatures, and import paths against `src/adaptshot/__init__.py` and the actual source files.
3. **Check `pyproject.toml`** — confirm that any dependency you reference is already in `dependencies` or `optional-dependencies`. Never assume a package is available.
4. **Understand the config contract** — `AdaptShotConfig` is a frozen dataclass. New fields must have defaults, type hints, and `Literal` constraints where applicable.

### While Writing Code
5. **Preserve backward compatibility** — existing public methods, signatures, and return types must not break without a deprecation cycle.
6. **Follow existing patterns** — match the module's indentation (4 spaces), line length (≤100), docstring style (Google), import order (stdlib → third-party → local, alphabetical).
7. **Use lazy imports** for optional dependencies — `_get_torch()`, `_get_tv_models()`, `_get_onnxruntime()` patterns are mandatory for torch/torchvision/onnxruntime.
8. **Type-annotate everything** — all public functions, class methods, and module-level variables. `mypy --strict` must pass.

### After Writing Code
9. **Update documentation** — if you change a public API, update the corresponding docstring and any affected `.md` files in `docs/`.
10. **Add or update tests** — every new feature, every bug fix, every behavioral change must include corresponding test coverage.
11. **Run the full quality gate** — `ruff check`, `mypy --strict`, `pytest -v`, and the smoke test benchmark before declaring work complete.
12. **Explain design decisions** — in commit messages or PR descriptions, justify *why* you chose an approach, not just *what* you changed.

---

## 4. Repository Architecture

### Package Layout
```
src/adaptshot/
├── __init__.py                  # Public API exports
├── config/
│   └── settings.py              # Frozen AdaptShotConfig dataclass (26 fields)
├── core/
│   ├── learner.py               # FewShotLearner — PRIMARY public API (~1644 lines)
│   ├── extractor.py             # BackboneRegistry, extract_embedding(), EmbeddingCache
│   ├── similarity.py            # cosine_similarity_numpy, find_nearest_neighbor, FAISS integration
│   ├── calibration.py           # CalibrationEngine — temperature scaling, ECE, scaling-binning
│   ├── act.py                   # ACTEngine — Adaptive Confidence Thresholding
│   ├── conformal.py             # ConformalEngine — split/cross conformal prediction
│   ├── contrastive.py           # ContrastivePrototypeLearner — InfoNCE, projection head
│   ├── uncertainty.py           # UncertaintyQuantifier — epistemic, aleatoric, distributional
│   ├── explain.py               # ExplainabilityEngine — attributions, counterfactuals
│   └── backends/
│       ├── __init__.py          # Backend abstraction layer
│       └── onnx_backend.py      # ONNX Runtime inference (torch-free)
├── training/
│   ├── feedback_router.py       # FeedbackRouter — HITL correction ingestion
│   ├── finetune.py              # CAEWCFinetuner — head-only EWC continual learning
│   └── up_ugf.py                # UPUGFPruner — uncertainty-guided buffer pruning
├── utils/
│   ├── determinism.py           # set_deterministic_seed, verify_determinism
│   ├── io.py                    # validate_path, save_json, load_json, tensor_to_numpy
│   ├── exceptions.py            # Custom exception hierarchy
│   ├── migrations.py            # Schema migration (v0.1.0 → v0.2.0)
│   └── profiling.py             # MemoryTracker, estimate_model_memory_mb
├── ui/
│   └── app.py                   # Gradio pilot dashboard (optional, `adaptshot[ui]`)
└── studio/
    ├── app.py                   # AdaptShot Studio — 8-tab Gradio application
    └── utils.py                 # Studio helpers, export/import, report generation
```

### Import Rules
- **From the repository root**, import paths use `src.adaptshot...` (e.g., `from src.adaptshot.core.learner import FewShotLearner`). Tests and benchmarks follow this convention.
- **When installed as a package** (`pip install adaptshot`), imports use `adaptshot...` (e.g., `from adaptshot import FewShotLearner`).
- **Do not mix** the two import styles in the same module.
- **Lazy imports** for optional dependencies (torch, torchvision, onnxruntime, gradio, faiss) use the `_get_*()` pattern — never `import torch` at module level unless behind a `TYPE_CHECKING` guard.

### Primary Public API
The public surface lives in `src/adaptshot/__init__.py`. Everything exported from `__init__.py` is **semver-stable**. Everything else is internal and may change without notice.

**Stable exports**: `AdaptShotConfig`, `FewShotLearner`, `CalibrationEngine`, `ACTEngine`, `ConformalEngine`, `ConformalPredictionSet`, `ContrastivePrototypeLearner`, `ContrastiveConfig`, `UncertaintyQuantifier`, `UncertaintyReport`, `ExplainabilityEngine`, `ExplanationResult`, `FeatureAttribution`, `FeedbackRouter`, `UPUGFPruner`, plus custom exceptions.

---

## 5. Coding Standards

### Functions
- Single responsibility — one function, one purpose. If a function exceeds ~50 lines, extract helpers.
- Use keyword-only arguments (`*, ...`) for optional parameters after required ones.
- Return types must be explicit. Never return `Union[TypeA, TypeB]` when a dedicated dataclass or `TypeVar` is clearer.

### Classes
- Prefer `@dataclass` for data containers. Use `frozen=True` for immutable config objects.
- Use `@property` sparingly — only when computation is cheap and deterministic.
- Instance state that exceeds 10 attributes should be grouped into nested dataclasses.

### Naming
- Classes: `PascalCase` (`FewShotLearner`, `CalibrationEngine`)
- Functions/Methods: `snake_case` (`load_support_images`, `compute_ece`)
- Variables: `snake_case` (`support_embeddings`, `calibrated_confidence`)
- Constants: `UPPER_SNAKE_CASE` (`BACKBONE_OUTPUT_DIM`, `FAISS_AVAILABLE`)
- Private members: prefix with single underscore (`_sim_embeddings`, `_refit_temperature`)
- **Never** use single-letter variable names except in comprehensions or trivial loops (`i`, `k`, `v`).

### Docstrings
- **Google-style** for all public modules, classes, and functions.
- Include `Args:`, `Returns:`, `Raises:` sections.
- Document edge cases (`Returns 0.0 for empty arrays`).
- Code examples in docstrings must be runnable and match the current API exactly.

```python
def compute_ece(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (L1).

    Args:
        confidences: Array of predicted confidence scores in [0, 1].
        correct: Binary array (1 = correct, 0 = incorrect).
        n_bins: Number of equal-mass bins for discretization.

    Returns:
        Expected Calibration Error as a float in [0, 1].
        Returns 0.0 for empty input arrays.

    Raises:
        ValueError: If confidences and correct have different lengths.
    """
```

### Type Hints
- **Mandatory** on all public APIs. Internal helpers should be typed where clarity is improved.
- Use `from __future__ import annotations` in all new files.
- Prefer `list[Foo]` over `List[Foo]` (Python 3.9+ style — the project requires `>=3.9`).
- Use `Optional[X]` (or `X | None`) for nullable parameters, never implicit `None` defaults without the type.

### Error Handling
- All AdaptShot exceptions inherit from `AdaptShotError`.
- Raise specific exceptions: `ConfigValidationError`, `InvalidImageError`, `CalibrationNotReadyError`, `BufferCapacityError`.
- Never use bare `except:` — catch specific exception types.
- Error messages must describe *what went wrong* and *what the user should do about it*.

### Logging
- Use the `logging` module — not `print()` — for informational output in library code.
- `warnings.warn()` for deprecation notices and non-critical issues.
- Demo scripts and benchmarks may use `print()` for human-readable output.

---

## 6. Performance Standards

### CPU Optimization
- All core operations run on CPU by default. Vectorize with NumPy — avoid Python loops over arrays.
- Use `np.dot`, `np.linalg.norm`, and broadcasting instead of manual iteration.
- Profile before optimizing. Use `benchmarks/energy_profile.py` and `utils/profiling.py`.

### Memory Budget (<250 MB)
- Support embeddings should be stored as `np.float32`, never `np.float64`.
- Use `np.memmap` for support sets exceeding 1000 examples.
- Call `clear_backbone_cache()` to release cached PyTorch models in long-running services.
- Every new feature must be assessed: *does this add >10 MB to the memory baseline?*

### Batch Processing
- Backbone extraction should batch multiple images into a single forward pass when `predict_batch()` is called (v0.3.0+).
- Distance computations should use matrix operations (`cdist`, `mm`) over element-wise loops.

### Lazy Loading
- Torch, torchvision, onnxruntime, and gradio imports must be lazy — never at module level.
- Backbone models are cached after first load; subsequent `FewShotLearner` instances reuse the cached model.

### Caching
- EmbeddingCache (per-instance) prevents re-extraction of identical support embeddings.
- BackboneCache (module-level, keyed by `(backbone_name, device)`) avoids reloading model weights.
- FAISS index must be cached and rebuilt only when the support set changes — not per query.

---

## 7. Research Standards

### Algorithm Implementation
- **Never invent algorithms.** Every ML method must cite a published paper, pre-print, or textbook.
- **When implementing published research**, include the paper citation in the module docstring:

```python
"""Online temperature scaling for confidence calibration.

Implements the method described in:
    Guo et al. (2017) "On Calibration of Modern Neural Networks"
    Proceedings of the 34th International Conference on Machine Learning (ICML).
"""
```

### Separating Research from Production
- Experimental or incomplete implementations must be gated behind explicit config flags with `EXPERIMENTAL` warnings.
- Never merge untested research code into `main` without the experimental gate.

### Approximations
- Clearly document any approximation: e.g., `estimate_epistemic()` uses perturbation sensitivity as a **proxy** for true epistemic uncertainty, not MC Dropout.
- Document the theoretical gap between the approximation and the ideal method.

### Benchmark Claims
- **Never claim benchmark numbers that are not reproducible** with `--seed 42`.
- All benchmark results must include: hardware spec, software versions, dataset version, and seed.
- Comparison against published baselines must note methodological differences (e.g., "AdaptShot uses frozen ImageNet-pretrained ResNet-18; published baselines use Conv-4 trained from scratch").

---

## 8. Testing Standards

### Required Tests for Every Feature
| Test Type | Requirement |
|-----------|-------------|
| **Unit tests** | Every public function and method must have at least one unit test |
| **Integration tests** | Full pipeline round-trips: load → predict → correct → save → load → predict |
| **Edge cases** | Empty arrays, single-element inputs, extreme values, NaN/Inf propagation |
| **Regression tests** | Bugs must have a test that fails before the fix and passes after |
| **Determinism tests** | `verify_determinism()` must be called for any operation that claims determinism |
| **Benchmark validation** | Smoke test (`--smoke-test --seed 42`) must complete without error |

### Test Patterns
- Use `pytest` fixtures for reusable test data.
- Use `monkeypatch` to replace slow operations (backbone extraction) with deterministic stubs in unit tests.
- Test files mirror source structure: `tests/test_calibration.py` tests `core/calibration.py`.
- Name tests descriptively: `test_ece_computation_overconfident` not `test_ece_2`.

### Running Tests
```bash
# Full test suite
pytest tests/ -v

# Single test file
pytest tests/test_calibration.py -v

# With coverage (if pytest-cov installed)
pytest tests/ -v --cov=src/adaptshot --cov-report=term-missing
```

---

## 9. Documentation Standards

### When Public APIs Change
When you modify a public class, method, or config field, you **must** update:

1. **The docstring** in the source file
2. **`src/adaptshot/__init__.py`** if the export list or re-exports change
3. **`docs/api/`** — the API reference markdown files
4. **Relevant tutorials** in `docs/tutorials/` if they use the changed API
5. **`CHANGELOG.md`** — under `[Unreleased]`, following Keep a Changelog format
6. **`docs/guides/migration-*.md`** if the change is breaking

### Doc Format
- MkDocs + Material theme with `mkdocstrings` auto-generated API docs
- Use `!!! note`, `!!! warning`, `!!! tip` admonitions for callouts
- All code examples must be runnable — ideally tested via `doctest` or copied from passing test cases

### Never
- Document features that don't exist yet (no "coming soon" in API docs)
- Include examples that use private APIs (leading underscore methods)
- Reference external services or cloud APIs that aren't available offline

---

## 10. API Design Philosophy

AdaptShot's API should feel like **scikit-learn's simplicity** combined with **PyTorch's composability**:

### Design Rules
1. **Small surface area** — `FewShotLearner` has 6 core methods. That's the right size. Resist the urge to add more.
2. **Consistency over cleverness** — all `predict()` methods return `PredictionResult`. All `correct()` methods return a correction summary. Don't break the pattern.
3. **Return dataclasses, not dicts** — `PredictionResult`, `UncertaintyReport`, `ExplanationResult` are typed, IDE-friendly, and have `.to_dict()` serialization. New return types must be dataclasses.
4. **Immutable config** — `AdaptShotConfig` is frozen. Users create it once and pass it around. No mutable global settings.
5. **Sensible defaults** — every config field has a default that works for 80% of use cases. Advanced users opt in to complexity.
6. **No breaking changes without deprecation** — public APIs must emit `DeprecationWarning` for at least one minor version before removal.
7. **Scikit-learn compatibility** — where possible, follow scikit-learn conventions (`fit`/`predict`/`predict_proba` naming, `BaseEstimator` protocol).

---

## 11. Dependencies & Packaging

### Dependency Rules
- **Core dependencies** (`dependencies` in `pyproject.toml`): `numpy>=1.24.0`, `Pillow>=9.0.0`. These two only. Nothing else.
- **Optional extras**: `[torch]`, `[faiss]`, `[ui]`, `[gui]`, `[dev]`. Users install only what they need.
- **Never add** a new required dependency. Everything beyond numpy+Pillow must be optional.
- **Forbidden entirely**: GPU-only libraries without CPU fallback, cloud/SaaS APIs, telemetry/analytics, non-MIT-compatible licenses.

### Optional Dependency Lazy Imports
```python
# ✅ CORRECT — lazy import pattern
def _get_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for training. Install: pip install adaptshot[torch]"
        ) from None

# ❌ WRONG — hard import at module level
import torch
```

---

## 12. Commands

### Development Setup
```bash
pip install -e ".[dev]"          # Install with dev tools (pytest, mypy, ruff, pre-commit)
pip install -e ".[torch]"        # Install with PyTorch training support
pip install -e ".[all]"          # Install everything
```

### Quality Gates (run before every commit)
```bash
ruff check src/ tests/           # Lint: must pass with zero errors
mypy src/adaptshot --strict      # Type check: must pass with zero errors
pytest tests/ -v                 # Test suite: must pass 100%
```

### Benchmarks
```bash
# Smoke test (CPU-only, ~30 seconds)
python -m benchmarks.run_benchmark --smoke-test --seed 42

# Full benchmark suite (miniImageNet if available)
python -m benchmarks.run_benchmark --full-benchmark --profile-memory

# Energy profile
python -m benchmarks.energy_profile --smoke-test --seed 42

# Integration simulations
python benchmarks/day2_integration.py
python benchmarks/day3_integration.py
python benchmarks/day4_integration.py
```

### Optional UIs
```bash
pip install -e ".[ui]"
python src/adaptshot/ui/app.py                           # Pilot dashboard
adaptshot-studio                                          # Studio (8-tab GUI, installed as CLI)
python -m examples.mziziguard.app                         # Crop disease demo
```

### ONNX Export
```bash
pip install -e ".[torch]" onnx onnxruntime
python scripts/export_backbones.py --all --verify
```

---

## 13. Hard Forbidden Actions

These actions are never allowed under any circumstances:

| ❌ Forbidden | ✅ Required Instead |
|---|---|
| Inventing APIs that don't exist in `src/adaptshot/` | Verify against the actual source code |
| Fabricating benchmark numbers or accuracy claims | Run the benchmark and report the actual output |
| Breaking a public API without deprecation | Add deprecation warning, keep old path for 1 minor version |
| Adding a required dependency beyond numpy+Pillow | Make it optional via `[extras]` and lazy import |
| Removing tests or test coverage | Add tests; never reduce coverage |
| Ignoring type errors with `# type: ignore` without justification | Fix the type error or use `# type: ignore[error-code]` with a comment |
| Using bare `except:` | Catch specific exception types |
| Claiming GPU or cloud features as available | CPU-first, offline-first. Always. |
| Importing torch/torchvision/onnxruntime at module level | Use lazy `_get_*()` pattern |
| Shipping code that doesn't pass `ruff`, `mypy --strict`, and `pytest` | Run the full quality gate first |
| Adding print() statements to library code | Use `logging` module |
| Assuming the user has internet access | All demos, examples, and tests must work offline |

---

## 14. Pull Request Checklist

Before marking any task as complete, mentally verify:

### Correctness
- [ ] All public API calls match the actual signatures in `src/adaptshot/`
- [ ] No dependency was added outside `pyproject.toml` optional groups
- [ ] All new features are CPU-compatible and respect the <250MB RAM budget
- [ ] Deterministic seed behavior is preserved (`--seed 42` produces identical results)

### Quality
- [ ] `ruff check src/ tests/` passes with zero errors
- [ ] `mypy src/adaptshot --strict` passes with zero errors
- [ ] `pytest tests/ -v` passes with 100% success rate
- [ ] `python -m benchmarks.run_benchmark --smoke-test --seed 42` completes without error

### Documentation
- [ ] Docstrings for new/changed public APIs are complete (Args, Returns, Raises)
- [ ] `CHANGELOG.md` has been updated under `[Unreleased]`
- [ ] Relevant `docs/` markdown files have been updated
- [ ] Any breaking change has a migration note

### Research Integrity
- [ ] New algorithms cite their source papers
- [ ] Approximations are clearly documented
- [ ] No benchmark claims are made without reproducible evidence
- [ ] Experimental features are gated with `EXPERIMENTAL` warnings

### User Impact
- [ ] Backward compatibility is preserved (or deprecated with warning)
- [ ] Error messages are descriptive and actionable
- [ ] Default behavior remains sensible for the 80% use case
- [ ] No feature requires internet, cloud, or GPU unless explicitly opt-in

---

## 15. Pre-Completion Self-Check

Before declaring any task finished, answer these questions:

1. **Did I verify the API?** — Every class, method, and parameter name matches the source code exactly.
2. **Did I update tests?** — New code has tests. Changed behavior has updated tests. Bug fixes have regression tests.
3. **Did I update docs?** — Docstrings, CHANGELOG, and affected `.md` files reflect the current state.
4. **Did I preserve compatibility?** — Existing code that uses the public API still works without modification.
5. **Did I run the quality gates?** — `ruff`, `mypy --strict`, `pytest`, and smoke test all pass.
6. **Could this affect users?** — If yes, have I added a deprecation warning, migration guide, or documentation note?
7. **Did I verify against `src/adaptshot/`?** — If I couldn't verify something, did I mark it `[TODO: Verify against src/adaptshot/...]`?

---

*This manual is the single source of truth for all AI agents in this repository. Read it before every task. Follow it during every task. Verify against it after every task.*

*When in doubt: choose correctness over speed, simplicity over cleverness, and the user's trust over your convenience.*
