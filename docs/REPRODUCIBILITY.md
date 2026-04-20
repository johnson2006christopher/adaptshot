# Reproducibility Guide

This document provides exact steps to reproduce AdaptShot Phase 1-4 behavior and validate Phase 5 UI/benchmark tooling.

## Version Matrix

- Python: `3.12.x` (project minimum: `>=3.10`)
- PyTorch: `>=2.0.0`
- Torchvision: `>=0.15.0`
- NumPy: `>=1.24.0`
- Gradio: `>=3.50.0`

## Environment Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Determinism Controls

- `torch.manual_seed(42)`
- `numpy.random.seed(42)`
- `random.seed(42)`
- `PYTHONHASHSEED=42`
- `torch.backends.cudnn.deterministic=True`
- `torch.backends.cudnn.benchmark=False`

You can apply these via:

```bash
python -c "from src import configure_runtime; configure_runtime(seed=42, deterministic=True); print('deterministic runtime ready')"
```

## Validate Tests

```bash
pytest tests/
```

## Reproduce Phase 1

```bash
python -c "from src import configure_runtime; cfg = configure_runtime(); print(cfg)"
```

## Reproduce Phase 2

```bash
python -c "from src.data import FewShotBatchSampler, create_fewshot_loader; print('phase2 imports ok')"
pytest tests/test_data.py -v
```

## Reproduce Phase 3

```bash
python -c "from src.models.network import create_fewshot_model; print('phase3 model ok')"
python -c "from src.evaluation.metrics import compute_ece, benchmark_latency; print('phase3 metrics ok')"
pytest tests/test_models.py -v
```

## Reproduce Phase 4

```bash
python -c "from src.training.feedback import ReplayBuffer; print('phase4 buffer ok')"
python -c "from src.training.incremental import incremental_fine_tune; print('phase4 incremental ok')"
pytest tests/test_training.py -v
```

## Reproduce Phase 5

```bash
python -c "from src.ui.app import create_gradio_app; print('ui import ok')"
python -m src.evaluation.benchmark --dry-run
```

## Hardware Notes

- Default execution path is CPU-only.
- Optional P100 fallback is supported when CUDA is available.
- Recommended memory for smooth experimentation: 8 GB+ RAM.
- Use `num_workers=0` for deterministic data loading on low-memory systems.
