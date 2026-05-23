# AdaptShot Documentation

![AdaptShot logo](images/adaptshot-logo.png)

Created by Johnson Hassan.

AdaptShot is a CPU-first, human-in-the-loop few-shot vision learning library. This documentation covers the public APIs that exist in `src/adaptshot/`, plus runnable examples that measure latency and memory on your own machine.

This documentation tracks the unreleased v0.1.1 branch. Use the native Python API as the source of truth; Studio is optional.

## Start Here

- [Installation](getting-started/installation.md)
- [Quick Start](getting-started/quickstart.md)
- [Tutorial-Style Guides](tutorials.md)
- [Benchmarks](getting-started/benchmarks.md)

## API Reference

- [Core Engine](api/core.md)
- [Training & Continual Learning](api/training.md)
- [Configuration & Utilities](api/config.md)

!!! warning "Use The Source As Truth"
    If documentation and behavior differ, verify against `src/adaptshot/` and open an issue with the mismatch.

## Verification Checklist

- [ ] You can install `adaptshot`.
- [ ] You can run the quickstart script.
- [ ] You can run `python -m benchmarks.run_benchmark --smoke-test --seed 42` from a source checkout.
- [ ] You can trace each documented API to `src/adaptshot/`.
