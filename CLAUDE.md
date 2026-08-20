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

- **CPU-only, <250MB RAM** for core functionality. Never assume GPU availability. Any GPU/torch-dependent feature must be behind an optional extra with a CPU fallback.
- **Deterministic**: use `set_deterministic_seed()` from `src/adaptshot/utils/determinism.py`; benchmarks must reproduce with `--seed 42`.
- **No new dependencies** outside the optional-extra groups already declared in `pyproject.toml`.
- **Never claim features, metrics, or latency numbers** not backed by code in `src/adaptshot/` or a script in `benchmarks/`. If you cannot verify an API against `src/adaptshot/`, write `[TODO: Verify against src/adaptshot/...]` rather than guessing. See `.openproject.md` for the full protocol.

## Code conventions

- **Imports in tests and repo-local code use the `src.` prefix**: `from src.adaptshot.core.learner import FewShotLearner`, not `from adaptshot...`. This is repo-wide and easy to get wrong.
- Line length ≤ 100 (ruff, configured in `pyproject.toml`). Format with `ruff check --fix`.
- Syntax must stay Python 3.9-compatible (`requires-python = ">=3.9"`, ruff `target-version = "py39"`) even though mypy runs under `python_version = "3.10"`.
- `mypy --strict` is enforced: type hints required on all public functions and classes.
- Google-style docstrings on public APIs.
- Use `logging.getLogger(__name__)` in library code, never `print()`.
- Raise from the `AdaptShotError` hierarchy in `src/adaptshot/utils/exceptions.py`.

## Gotchas

- **Version lives in two places** that must agree: `pyproject.toml` and `src/adaptshot/__init__.py`. `tests/test_release_metadata.py` derives the expected value from `pyproject.toml`, so bumping only one of the two fails the suite.
- MkDocs output is **never committed**. `.github/workflows/docs.yml` builds and deploys to the `gh-pages` branch on every push to `main`; `site/` is gitignored. Edit `docs/`, never generated HTML.
- Dev tooling is not installed in the ambient environment — run `pip install -e ".[dev]"` before any lint/type/test command.

## Repo etiquette

- **Never run `git commit`, `git push`, or `git branch`.** The maintainer performs every git operation himself. Make the edits, report what changed, and suggest a commit message if useful.
- **Branching**: a version branch (e.g. `v0.3.0`) is the integration target; short-lived `feat/...` and `fix/...` branches merge into it. The version branch merges to `main` at release, then gets tagged. `main` always represents a shipped release.
- Commits: Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `build:`, `release:`)
- Every PR must pass all four validation stages and the constraint checklist in `.github/PULL_REQUEST_TEMPLATE.md`.
