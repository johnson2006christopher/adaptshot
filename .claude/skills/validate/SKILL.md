---
name: validate
description: Run the full AdaptShot validation gate — ruff, mypy --strict, pytest, the deterministic smoke benchmark, and a strict docs build — and report exactly which stage failed. Use before opening a PR, after finishing a change, or whenever the user asks to verify the repo is green.
---

# AdaptShot validation gate

Run these five stages **in order** from the repo root. Do not skip ahead: a later stage's failure is usually noise if an earlier one is red.

If the tools are missing (`ModuleNotFoundError`, `command not found`), the dev extras aren't installed. Run `pip install -e ".[dev]"` first and say so — don't silently substitute a different command.

## 1. Lint
```bash
ruff check src/ tests/ benchmarks/ apps/ examples/ scripts/
```
All six directories: CI lints all of them, and `scripts/` was missed for months because it was not in this list. Auto-fixable issues: rerun with `--fix`, then show the diff.

## 2. Type check (strict)
```bash
mypy src/adaptshot --strict
```
Strict mode is mandatory for merge, and CI runs it hard in the lint job — a green lint job means mypy passed. If you see `"Button" has no attribute "click"` on a fresh install, import gradio once first (`python -c "import gradio"`): it writes its own stubs into site-packages on first import, and mypy reads files.

## 3. Tests
```bash
pytest tests/ -v --tb=short
```
Tests import the installed package (`from adaptshot...`). An `ImportError` on `adaptshot.*` usually means the editable install is missing — run `pip install -e ".[dev]"`.

## 4. Smoke benchmark (CPU-only, deterministic)
```bash
python -m benchmarks.run_benchmark --smoke-test --seed 42
```
Must be reproducible at `--seed 42`. Non-deterministic output is a failure, not flakiness to retry away.

## 5. Docs build (strict)
```bash
mkdocs build --strict
```
Needs `pip install mkdocs mkdocs-material "mkdocstrings[python]"`. Every page under `docs/` must be in the `nav` in `mkdocs.yml` or listed under `not_in_nav`; a broken link or an orphaned page is a failure, and CI enforces this on every pull request. Delete `site/` afterwards — it is gitignored, never committed.

## Reporting

Report a per-stage PASS/FAIL table, then the first real failure with its actual output. Never report the gate as green unless all five stages ran and passed — if you skipped a stage (missing tooling, no torch, etc.), say which one and why.
