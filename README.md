# AdaptShot: The Zero-Config Few-Shot AI Platform

AdaptShot is a CPU-first, human-in-the-loop few-shot vision platform designed to reach production fast: calibrated uncertainty, real-time correction learning, and deterministic behavior from day one.

## Startup Pitch

- **USP:** zero-config few-shot learning with calibrated uncertainty and correction-driven adaptation
- **Who it serves:** healthcare, manufacturing, agriculture, and enterprise edge AI teams
- **Why it wins:** deployable on CPU with fewer than 50 images per class and fast time-to-value

## Technical Overview

```text
Raw Image
  -> ResNet18 Backbone (frozen)
  -> 512-d Embedding
  -> Hybrid Similarity Search (FAISS CPU + NumPy fallback)
  -> Calibrated Prediction + Confidence
  -> Human Correction Feedback
  -> Incremental Update Loop
```

## Quickstart

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Run Core Benchmarks

```bash
python -m benchmarks.run_benchmarks --dry-run
```

## Run Core Tests

```bash
pytest tests/test_core.py -v
```

## Phase 1 Deliverables

- `src/core/embedding.py`: deterministic 512-d embedding extraction
- `src/core/similarity.py`: FAISS/NumPy hybrid retrieval
- `src/evaluation/metrics.py`: deterministic ECE, latency, accuracy
- `benchmarks/run_benchmarks.py`: CI-ready benchmark runner
- `tests/test_core.py`: validation for embedding + similarity behavior

## Target Metrics

- Few-shot accuracy: `>75%` on 5-shot CIFAR-style setup
- Calibration: `ECE < 0.05`
- CPU latency: `<50 ms` per image
- Determinism: reproducible by seed and environment

## Citation

```bibtex
@misc{adaptshot2026,
  title={AdaptShot: Zero-Config Human-in-the-Loop Few-Shot Learning with Calibrated Uncertainty},
  author={AdaptShot Team},
  year={2026},
  note={Open-source production and research platform}
}
```
