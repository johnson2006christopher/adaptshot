# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

AdaptShot is a CPU-first, human-in-the-loop few-shot vision library (Python, `src/` layout). Public API entrypoint is `FewShotLearner` in `src/adaptshot/core/learner.py`.

## Commands

Dev tooling is not installed in the ambient environment — install it first:

```bash
pip install -e ".[dev]"
```

Full validation gate (all four must pass before a PR):

```bash
ruff check src/ tests/
mypy src/adaptshot --strict
pytest tests/ -v
python -m benchmarks.run_benchmark --smoke-test --seed 42
```

Single test: `pytest tests/test_conformal.py -k test_name -v`
Docs preview: `mkdocs serve` (requires `mkdocs`, `mkdocs-material`, `mkdocstrings[python]`)

## Non-negotiable constraints

- **CPU-only.** Never assume GPU availability. Any GPU/torch-dependent feature must be behind an optional extra with a CPU fallback.
- **<250MB RAM is the target, not the current state.** Measured peak RSS for a full cycle is ~775MB, because `import adaptshot` pulls in torch eagerly (~479MB) via `utils/determinism.py` and `utils/io.py`. `tests/test_memory_ceiling.py` guards a regression budget against reality and strict-xfails the 250MB figure. Do not restate 250MB as an achieved number anywhere (#13).
- **Deterministic**: use `set_deterministic_seed()` from `src/adaptshot/utils/determinism.py`; benchmarks must reproduce with `--seed 42`.
- **No new dependencies** outside the optional-extra groups already declared in `pyproject.toml`.
- **Never claim features, metrics, or latency numbers** not backed by code in `src/adaptshot/` or a script in `benchmarks/`. If you cannot verify an API against `src/adaptshot/`, write `[TODO: Verify against src/adaptshot/...]` rather than guessing. See `.openproject.md` for the full protocol.

## Code conventions

- **Always import the installed package, never the source tree**: `from adaptshot.core.learner import FewShotLearner`, never `from src.adaptshot...`. The two load as separate modules, so `isinstance` and `except` fail across them. Enforced by `tests/test_import_convention.py`.
- Line length ≤ 100 (ruff, configured in `pyproject.toml`). Format with `ruff check --fix`.
- Syntax must stay Python 3.9-compatible (`requires-python = ">=3.9"`, ruff `target-version = "py39"`) even though mypy runs under `python_version = "3.10"`.
- `mypy --strict` is enforced: type hints required on all public functions and classes.
- Google-style docstrings on public APIs.
- Use `logging.getLogger(__name__)` in library code, never `print()`.
- Raise from the `AdaptShotError` hierarchy in `src/adaptshot/utils/exceptions.py`.

## Gotchas

- **Version lives in two places** that must agree: `pyproject.toml` and `src/adaptshot/__init__.py`. `tests/test_release_metadata.py` derives the expected value from `pyproject.toml`, so bumping only one of the two fails the suite.
- MkDocs output is **never committed**. `.github/workflows/docs.yml` builds and deploys to the `gh-pages` branch on every push to `main`; `site/` is gitignored. Edit `docs/`, never generated HTML.
- **Import gradio once before type-checking anything that uses it.** gradio attaches `.click()`/`.change()` to components in a metaclass and writes the matching `.pyi` files into site-packages on first import; they are not in the wheel. On a fresh install `mypy` reports `"Button" has no attribute "click"` against our code, which is a lie. CI does this explicitly in the lint job.
- Dev tooling is not installed in the ambient environment — run `pip install -e ".[dev]"` before any lint/type/test command.

## Repo etiquette

- **Git operations are delegated.** Create branches, commit in small scoped units, push, open PRs, and merge into the version branch. Two operations stay with the maintainer: merging the version branch into `main`, and tagging a release. `main` is only ever what the maintainer personally shipped.
- **Commits carry no AI co-author trailer.** Every commit is reviewed by the maintainer and published under his name; the history records the project, not the tooling.
- **Branching**: a version branch (e.g. `v0.3.0`) is the integration target; short-lived `feat/...` and `fix/...` branches merge into it. The version branch merges to `main` at release, then gets tagged. `main` always represents a shipped release.
- Commits: Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `build:`, `release:`)
- Every PR must pass all four validation stages and the constraint checklist in `.github/PULL_REQUEST_TEMPLATE.md`.
