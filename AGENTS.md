# AdaptShot agent notes (read .openproject.md first)

- Primary public API lives in `src/adaptshot/core/learner.py` (`FewShotLearner`); verify any API or signature against `src/adaptshot/` before stating it.
- CPU-only, <250MB RAM are non-negotiable; never assume GPU availability.
- Import paths in code, tests, and benchmarks use `adaptshot...` (the installed package), never `src.adaptshot...`. Requires an editable install: `pip install -e ".[dev]"`.

## Commands that matter
- Install dev deps: `pip install -e ".[dev]"`
- Lint: `ruff check src/ tests/`
- Typecheck (strict): `mypy src/adaptshot --strict`
- Tests: `pytest tests/ -v`
- Benchmark smoke test (CPU-only): `python -m benchmarks.run_benchmark --smoke-test --seed 42`

## Optional/UI
- Gradio UI entrypoint: `src/adaptshot/ui/app.py` (requires `pip install -e ".[ui]"`).

## Repo-specific constraints from .openproject.md
- Do not add dependencies outside `pyproject.toml` optional groups; never claim features/metrics not backed by code or benchmarks.
- If you cannot verify an API or behavior in `src/adaptshot/`, emit `[TODO: Verify against src/adaptshot/...]` instead of guessing.
