---
name: validate
description: Run the full AdaptShot validation gate — ruff, mypy --strict, pytest, and the deterministic smoke benchmark — and report exactly which stage failed. Use before opening a PR, after finishing a change, or whenever the user asks to verify the repo is green.
---

# AdaptShot validation gate

Run these four stages **in order** from the repo root. Do not skip ahead: a later stage's failure is usually noise if an earlier one is red.

If the tools are missing (`ModuleNotFoundError`, `command not found`), the dev extras aren't installed. Run `pip install -e ".[dev]"` first and say so — don't silently substitute a different command.

## 1. Lint
```bash
ruff check src/ tests/
```
Auto-fixable issues: rerun with `ruff check --fix src/ tests/`, then show the diff.

## 2. Type check (strict)
```bash
mypy src/adaptshot --strict
```
Strict mode is mandatory for merge. Note that CI currently soft-fails this stage (`|| true`), so a green CI badge does **not** mean mypy passed — always run it locally.

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

## Reporting

Report a per-stage PASS/FAIL table, then the first real failure with its actual output. Never report the gate as green unless all four stages ran and passed — if you skipped a stage (missing tooling, no torch, etc.), say which one and why.
