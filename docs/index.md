# AdaptShot Documentation

![AdaptShot logo](images/adaptshot-logo.png)

Created by Johnson Hassan.

AdaptShot is a CPU-first, human-in-the-loop few-shot vision learning library. This documentation covers the public APIs that exist in `src/adaptshot/`, plus runnable examples that measure latency and memory on your own machine.

This documentation tracks the v0.1.2 release. Use the native Python API as the source of truth; Studio is optional.

## How AdaptShot Works

```mermaid
graph TB
    A[User Image] --> B[FewShotLearner.predict]
    B --> C[Extractor<br/>Frozen ResNet/MobileNet]
    C --> D[Embedding Vector]
    D --> E{Inference Mode}
    E -->|nearest_neighbor| F[Cosine/Euclidean<br/>Similarity Search]
    E -->|prototypical| G[Class Prototype<br/>Distance]
    F --> H[CalibrationEngine<br/>Temperature Scaling]
    G --> H
    H --> I[ACTEngine<br/>Adaptive Threshold]
    I -->|Accept| J[PredictionResult]
    I -->|Request Feedback| K[Human Review]
    K --> L[FewShotLearner.correct]
    L --> M[FeedbackRouter]
    M --> N[CA-EWC Fine-Tune]
    M --> O[UP-UGF Buffer Prune]
    N --> B
    O --> B
    J --> P{Uncertainty?}
    P -->|Yes| K
    P -->|No| Q[Return Result]
```

The pipeline is a closed loop: every human correction feeds back into the learner, improving calibration, adjusting confidence thresholds, and fine-tuning the classification head while preserving prior knowledge.

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
