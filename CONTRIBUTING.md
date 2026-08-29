# Contributing to AdaptShot

Thank you for your interest in contributing to AdaptShot. This document outlines how to contribute code, documentation, or feedback in a way that aligns with our mission: **building trustworthy, CPU-first few-shot vision AI for resource-constrained environments**.

## 🎯 Our Guiding Principles
1. **Truth over hype**: We document what works, what doesn't, and why. No exaggerated claims.
2. **CPU-first by default**: All new features must run on CPU with <250MB RAM unless explicitly marked as optional GPU extras.
3. **Human-in-the-loop by design**: Contributions should enhance transparency, calibration, or feedback routing—not obscure them.
4. **Open and reproducible**: Every benchmark, test, and result must be reproducible with `pytest`, `mypy`, and our benchmark harnesses.

## 🛠️ Getting Started
### Prerequisites
- Python ≥ 3.9
- Git
- Virtual environment tool (`venv`, `virtualenv`, or `conda`)

### Setup
```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/adaptshot.git
cd adaptshot

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install development dependencies
pip install -e ".[dev]"

# 4. Install pre-commit hooks (optional but recommended)
pre-commit install
```

### Verify Your Setup
```bash
# Run tests
pytest tests/ -v

# Run type checking
mypy src/adaptshot --strict

# Run linting
ruff check src/adaptshot tests/

# Run smoke benchmark
python -m benchmarks.run_benchmark --smoke-test
```

The smoke benchmark runs **offline by default**. With no CIFAR-10 cache it uses a
deterministic synthetic fixture and reports determinism and latency but no
accuracy — a figure measured on random tensors describes nothing. For a measured
accuracy, fetch the dataset once:

```bash
python -m benchmarks.run_benchmark --smoke-test --dataset cifar10 --allow-download --seed 42
```

That download is ~170MB and has taken over 30 minutes from some networks (#12).
Afterwards the default command finds the cache and uses the real data.

All commands should pass before submitting a pull request.

## 📦 Project Structure
```
adaptshot/
├── src/adaptshot/          # Core library code
│   ├── config/             # Configuration dataclasses
│   ├── core/               # Inference engine (extractor, similarity, calibration, ACT, learner)
│   ├── training/           # Continual learning (feedback_router, finetune, up_ugf)
│   ├── ui/                 # Gradio interface for pilots
│   └── utils/              # Determinism, I/O helpers
├── benchmarks/             # Reproducible evaluation scripts
├── tests/                  # Unit and integration tests
├── docs/                   # Documentation source (MkDocs)
├── pyproject.toml          # Build config, dependencies, tool settings
├── README.md               # Project overview
├── CONTRIBUTING.md         # This file
├── CODE_OF_CONDUCT.md      # Community standards
└── CHANGELOG.md            # Version history
```

## 🧪 Testing Guidelines
- **All new code must include unit tests** in `tests/` using `pytest`.
- **Type hints are required** on all public functions and classes (`mypy --strict` enforced).
- **Benchmarks must be deterministic**: Use `set_deterministic_seed()` and verify with `verify_determinism()`.
- **CPU-first validation**: New features must pass benchmarks on CPU unless explicitly marked GPU-only.

Example test structure:
```python
# tests/test_new_module.py
import pytest
from adaptshot.new_module import NewComponent

def test_new_component_basic():
    comp = NewComponent(param=42)
    result = comp.run()
    assert result is not None
    assert isinstance(result, ExpectedType)
```

## 📝 Code Style
- **Formatting**: Ruff auto-formatter (`ruff check --fix`)
- **Imports**: Standard library → third-party → local (sorted alphabetically)
- **Docstrings**: Google-style for public APIs; concise one-liners for private helpers
- **Line length**: ≤100 characters
- **Logging**: Use `logging.getLogger(__name__)` instead of `print()` for library code

## 🔒 API Stability and Deprecation

Every name in `adaptshot.__all__` is classified in `adaptshot.api` as **stable** or
**experimental**. `tests/test_api_surface.py` keeps that classification, the
docstrings, and `docs/reference/api.md` in agreement, so the tiers are something
the suite checks rather than a comment that drifts.

**Stable** names are semver-protected. Removing one, or changing it in a way that
breaks a caller, follows a deprecation cycle:

1. The old behaviour keeps working and emits a `DeprecationWarning` -- with
   `stacklevel` set so it points at the caller's line, not ours -- naming the
   release it was deprecated in, the release it will be removed in, and what to
   use instead.
2. It stays for at least one minor release.
3. It is removed in the next minor release, and the removal is listed in the
   changelog.

**Experimental** names may change in a minor release without that cycle. Their
docstrings open with **Experimental** so the status is visible at the point of
use. An experimental name becomes stable when it has tests of its own and has
shipped in at least one release; it never becomes stable by default.

Adding a name to `__all__` means choosing a tier in `adaptshot.api` and, for an
experimental one, saying so in the docstring. The test fails otherwise.

The first uses of this policy, both in 0.3.0: `adaptshot.core.contrastive` moved
to `adaptshot.training.contrastive` (the old path warns and is removed in 0.4.0),
and three `UncertaintyQuantifier` methods that nothing in the library, its tests
or its applications called were deprecated rather than deleted (removed in 0.4.0).

Pre-1.0, semver permits a minor release to break. This policy is the promise the
project makes anyway.

## 🔄 Contribution Workflow
1. **Create a feature branch**: `git checkout -b feat/your-feature-name`
2. **Make changes**: Follow testing and style guidelines above
3. **Run full validation**:
   ```bash
   pytest tests/ -v
   mypy src/adaptshot --strict
   ruff check src/adaptshot tests/
   python -m benchmarks.run_benchmark --smoke-test
   ```
4. **Commit with clear messages**:
   ```bash
   git commit -m "feat: add conformal prediction stub to CalibrationEngine"
   git commit -m "test: add unit tests for UP-UGF pruning logic"
   ```
5. **Push and open a Pull Request**:
   - Link to relevant issue or discussion
   - Describe what changed and why
   - Include benchmark diffs if performance/calibration affected
   - Add `[WIP]` prefix if work is incomplete

Releases are tagged, not pushed by hand; see [docs/contributing/release-checklist.md](docs/contributing/release-checklist.md).

## 🗣️ Communication
- **Discussions**: Use GitHub Discussions for questions, ideas, and RFCs
- **Issues**: Report bugs or request features via GitHub Issues
- **WhatsApp**: Join our [community WhatsApp group](https://chat.whatsapp.com/J6AbrvbjmBc5XXX2fnN6RK) for real-time discussion
- **Security**: Report vulnerabilities privately to johnson2006christopher@gmail.com
- **Code of Conduct**: All interactions follow our [Code of Conduct](CODE_OF_CONDUCT.md)

## 📚 Documentation
- **API docs**: Auto-generated from docstrings via MkDocs + mkdocstrings
- **Tutorials**: Jupyter notebooks in `docs/tutorials/` with executable examples
- **Examples**: Minimal runnable scripts in `docs/examples/`

To preview docs locally:
```bash
pip install mkdocs mkdocstrings[python] mkdocs-material
mkdocs serve
# Visit http://localhost:8000
```

## 🚫 What We Don't Accept
- Features that require GPU by default without a CPU fallback
- Undocumented public APIs or missing type hints
- Benchmarks that aren't reproducible with `--seed 42`
- Marketing language or unsubstantiated performance claims
- Code that violates our CPU-first, <250MB RAM constraint for core functionality

## 🙏 Thank You
Every contribution—code, docs, testing, or feedback—helps make trustworthy AI accessible to more people. We review all PRs within 7 days and provide constructive feedback.

If you're unsure where to start, look for issues labeled [`good first issue`](https://github.com/johnson2006christopher/adaptshot/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) or open a Discussion to propose an idea.

— The AdaptShot Maintainers
