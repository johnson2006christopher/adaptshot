# Development setup and the validation gate

> **For:** someone about to change AdaptShot's code or documentation. Assumes Python and git. The [contributing guide](contributing.md) covers the workflow and what is accepted; this page is the machine setup and the checks.

## Set up

```bash
git clone https://github.com/johnson2006christopher/adaptshot.git
cd adaptshot
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"                                # the library, tests, linters
pip install -e "apps/tambua[dev]"                      # the application, as its own distribution
pip install mkdocs mkdocs-material "mkdocstrings[python]"   # for the docs gate
```

With [uv](https://docs.astral.sh/uv/) the same three lines are `uv venv .venv`, `uv pip install -e ".[dev]" -e "apps/tambua[dev]"` and `uv pip install mkdocs mkdocs-material "mkdocstrings[python]"`, and they take seconds rather than minutes. The gate commands below are unchanged; they do not care which tool populated the environment.

That is the *core* install — numpy, Pillow, onnxruntime — and it is enough for every test that matters to a user. Add the torch extra only if you are working on fine-tuning or a non-bundled backbone, and prefer the CPU build:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

Dev tooling is not assumed to be on the machine; if a command below says "not found", the install above was skipped.

## The gate — five stages, in order

Every pull request must pass all five. Run them locally first; a later stage's failure is usually noise if an earlier one is red.

```bash
ruff check src/ tests/ benchmarks/ apps/ examples/ scripts/     # 1. lint, all six directories
mypy src/adaptshot --strict                                     # 2. types, strict
pytest tests/ -v                                                # 3. tests
python -m benchmarks.run_benchmark --smoke-test --seed 42       # 4. deterministic smoke benchmark
mkdocs build --strict                                           # 5. docs: every page in the nav, no broken links
```

Stage 5 needs the docs packages above; delete `site/` afterwards — it is gitignored and never committed.

Two things about stage 2. Import gradio once (`python -c "import gradio"`) on a fresh install before running mypy on anything that touches Tambua: gradio writes its own type stubs into site-packages on first import, and without them mypy reports `"Button" has no attribute "click"` against our code, which is a lie. And mypy analyses for Python 3.12 (`python_version` in `pyproject.toml`) while the floor is 3.10; numpy's stubs need 3.12 to parse, and the 3.10 floor is enforced by ruff's `target-version` and by the CI matrix actually running 3.10.

## What CI runs, and why each job exists

| job | what it proves |
|---|---|
| Lint & type check | stages 1–2, plus mypy on Tambua |
| Tests (3.10 – 3.14) | stage 3 on every supported Python, CPU-only torch installed |
| Tests (core install, no torch) | stage 3 with torch absent — the install a user gets; **enforcing**, not advisory |
| Smoke benchmark (offline) | stage 4 on the synthetic fixture inside a network namespace |
| Deterministic smoke benchmark | stage 4 on cached CIFAR-10, checking two runs agree |
| Offline, from the wheel | builds the wheel, installs it into a clean venv, seals a namespace with no interfaces, proves the canary fails, then runs the quickstart, the demo, the conformal and calibration suites and the benchmark against `site-packages` |
| Docs build (strict) | stage 5 |

`release.yml` runs on `v*` tags: build, the full gate on the tagged commit, a clean-container install test, then TestPyPI for `rc` tags and PyPI otherwise, with Trusted Publishing and no token. The [release checklist](release-checklist.md) has the human steps.

## Conventions the tests enforce

- **Import the installed package, never the source tree**: `from adaptshot…`, never `from src.adaptshot…`. They load as separate modules and `isinstance` fails across them. `tests/test_import_convention.py`.
- **The version lives in one place.** `pyproject.toml`; `__version__` is read from the installed metadata. After bumping it, `pip install -e .` or `tests/test_release_metadata.py` reports the stale value and tells you that.
- **Every public name is classified** stable or experimental in `adaptshot.api`; `tests/test_api_surface.py` fails on an unclassified export, a missing docstring marker, or a name absent from the reference. See [API stability](api-stability.md).
- **Numbers trace to artifacts.** `tests/test_docs_claims.py` formats figures from `results/*.json` and asserts the README, the technical note and the reference quote them verbatim.
- **Tutorials and how-tos run.** `tests/test_docs_tutorials_run.py` executes every page's Python offline. See [how the docs are tested](how-the-docs-are-tested.md).
- **Determinism.** Use `set_deterministic_seed()`; the benchmark must reproduce at `--seed 42`.
- **No new dependencies** outside the extras already declared. A library arguing that connectivity is scarce should not need a 100 MB wheel to check a baseline.
- **Commits** follow Conventional Commits and carry no AI co-author trailer; the history records the project.

## Two local hazards

**Building a wheel.** Delete `build/` and `src/*.egg-info/` first. setuptools trusts a stale `SOURCES.txt` and `build/lib/` over `package-data`, and a wheel built after `scripts/export_backbones.py` had written into the data directory came out at 44 MB instead of 3.5. CI builds from a clean checkout and is immune.

**Exported backbones.** `scripts/export_backbones.py` writes into `src/adaptshot/data/` by default, and anything there is picked up by name at runtime. Three tests once passed locally and failed on every CI job because a locally exported `resnet18.onnx` was on disk. Export to a scratch directory.

## Where things are

```text
src/adaptshot/       the library (core/, training/, config/, utils/, data/, api.py, preflight.py)
apps/tambua/         the application, its own distribution
benchmarks/          every published number's script
scripts/             maintainer tools: fetch data, export backbones
examples/demo/       the offline conference demo and handout
tests/               the suite; docs and artifacts have guards here too
docs/                this documentation; see mkdocs.yml for the nav
results/             committed artifacts the documentation traces to
```
