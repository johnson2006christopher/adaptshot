## What changed

<!-- One paragraph. What does this do, and why? -->

## Related

<!-- Closes #123, or a link to the Discussion / RFC -->

## Validation

All four stages must pass locally before review:

- [ ] `ruff check src/ tests/ benchmarks/`
- [ ] `mypy src/adaptshot --strict`
- [ ] `pytest tests/ -v`
- [ ] `python -m benchmarks.run_benchmark --smoke-test --seed 42`

## Constraints

- [ ] Runs on CPU; no GPU required for anything in the core path
- [ ] Stays within the <250 MB RAM target
- [ ] Deterministic — reproducible at `--seed 42`
- [ ] No new dependency outside the existing `pyproject.toml` extras
- [ ] New public functions and classes carry type hints and Google-style docstrings

## Claims

- [ ] Every performance, accuracy, or calibration number in this PR is backed by a
      script in `benchmarks/` or a test in `tests/` — none are estimated or asserted
      from memory

## Benchmark impact

<!-- Required if this touches calibration, uncertainty, similarity, or the extractor.
     Paste the before/after smoke benchmark output. Write "n/a" otherwise. -->
