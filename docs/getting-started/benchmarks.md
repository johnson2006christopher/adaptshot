# Benchmarks & Reproducibility

AdaptShot is engineered for transparency and reproducibility. All performance metrics are measured on CPU-only hardware with deterministic seeding. This document provides exact reproduction commands, expected results, hardware-tier expectations, and baseline comparisons for v0.2.0.

## Quick Validation

### 1. Smoke Test (CIFAR-10, 5-way 10-shot)

```bash
# Run all unit tests (92 tests)
pytest tests/ -v

# Run minimal smoke benchmark
python -m benchmarks.run_benchmark --smoke-test --seed 42
```

### 2. Full Benchmark Suite (v0.2.0)

```bash
# Full benchmark with miniImageNet and baseline comparisons
python -m benchmarks.run_benchmark --full-benchmark --seed 42 --output results/full_benchmark.json

# With memory profiling
python -m benchmarks.run_benchmark --full-benchmark --seed 42 --profile-memory
```

### 3. Quality Gates

```bash
# Lint check
ruff check src/ tests/

# Strict type checking
mypy src/adaptshot --strict

# Full test suite with coverage
pytest tests/ -v --cov=src/adaptshot
```

---

## Expected Results (Reference Hardware: Intel Core i5-1135G7, 16GB RAM, Ubuntu 22.04)

### CIFAR-10 Smoke Test (5-way, 10-shot)

| Metric | Value | Notes |
|--------|-------|-------|
| **Few-shot Accuracy** | 68% | Frozen ResNet-18 + prototypical inference |
| **Avg Inference Latency** | 20 ms | Per-image prediction (similarity + calibration + conformal) |
| **P95 Inference Latency** | 20 ms | Upper bound including OS scheduling variance |
| **Embedding Extraction** | ~1.5s | For 50 support images (30ms per image) |
| **RAM Footprint** | < 250 MB | Including model weights, support embeddings, and replay buffer |
| **Determinism** | ✅ PASS | Bit-exact outputs across 3 runs with `--seed 42` |

### Baseline Comparisons

AdaptShot includes reference baselines from published literature for context. These are NOT AdaptShot's results — they are provided so you can compare against established few-shot learning methods:

| Method | miniImageNet 5-way 1-shot | miniImageNet 5-way 5-shot | Source |
|--------|--------------------------|--------------------------|--------|
| **Prototypical Networks** | 49.4% | 68.2% | Snell et al., 2017 |
| **Matching Networks** | 43.6% | 55.3% | Vinyals et al., 2016 |
| **MAML** | 48.7% | 63.1% | Finn et al., 2017 |
| **AdaptShot** | Run `--full-benchmark` | Run `--full-benchmark` | This project |

!!! note "Baselines Are Reference Points"
    Baseline numbers come from published papers and were measured on GPU hardware with different backbones. Direct comparison requires matching hardware, backbone architecture, and data preprocessing. AdaptShot's CPU-first design trades some accuracy for accessibility — a deliberate and documented trade-off.

### miniImageNet Support

v0.2.0 supports miniImageNet benchmarks. Download the dataset and place `mini_imagenet.csv` in the project's `data/` directory:

```bash
# After downloading miniImageNet:
python -m benchmarks.run_benchmark --full-benchmark --seed 42
```

The CSV should contain columns: `file_path`, `label`, `split` (train/val/test).

---

## Hardware-Tier Expectations

| Device | Expected Latency | Recommended Config |
|--------|------------------|-------------------|
| **Modern Laptop CPU** (i5/Ryzen 5, 4+ cores) | < 25 ms | `resnet18`, CPU mode, default buffer |
| **Single-Board Computer** (Raspberry Pi 4) | 150–250 ms | `mobilenet_v3_small`, CPU mode, `max_buffer_size=30` |
| **Legacy Office PC** (4GB RAM, HDD) | < 200 ms | `mobilenet_v3_small`, disable FAISS, `max_buffer_size=20` |
| **GPU System** (CUDA-capable) | ~30–50 ms | Set `device="cuda"` in `AdaptShotConfig` |

!!! warning "Context Matters"
    These numbers are reference points, not leaderboard targets. Real-world accuracy depends heavily on:
    - Domain similarity between support set and query images
    - Lighting, resolution, and background consistency
    - Quality and confidence of human corrections during continual learning
    - Support set size and class balance

---

## Reproducibility Guarantees

AdaptShot enforces deterministic execution by default:

- Fixed seeds: `torch.manual_seed`, `np.random.seed`, `PYTHONHASHSEED=42`
- Deterministic cuDNN algorithms when CUDA is enabled (`cudnn.deterministic=True`)
- No asynchronous I/O or non-deterministic PyTorch operations in the core pipeline
- Verification utility: `verify_determinism()` in `src/adaptshot/utils/determinism.py`

To reproduce our exact smoke test results:

```bash
python -m benchmarks.run_benchmark --smoke-test --seed 42 --output results/baseline.json
cat results/baseline.json
```

---

## Understanding ECE & Calibration Behavior

In v0.2.0, calibration uses a multi-layered approach:

1. **True Leave-One-Out Conformal**: Each support example is held out and prototypes are recomputed without it, providing valid finite-sample coverage guarantees.
2. **Bootstrap Temperature**: On cold start (first prediction), temperature is estimated via LOO cross-validation on the support set — no separate validation set required.
3. **Sliding-Window ECE Tracking**: After the initial bootstrap, ECE is continuously tracked and temperature refits automatically.

### ECE Lifecycle

- **Steps 1–9**: Bootstrap temperature provides initial calibration. ECE may be slightly elevated as the sliding window populates.
- **Steps 10+**: Temperature refits automatically as the window accumulates observations. ECE typically drops as the model adapts to local confidence-accuracy dynamics.
- **Domain Shift**: If query distribution changes sharply, ECE will temporarily rise until the window adjusts. This is expected behavior — the system is correctly detecting uncertainty rather than masking it.

!!! note "No Validation Set Required"
    Unlike traditional post-hoc calibration, AdaptShot does not require a held-out validation dataset. Calibration adapts continuously from live inference and human feedback, making it suitable for few-shot, low-data deployments.

---

## Memory Profiling

v0.2.0 includes built-in memory profiling via `src/adaptshot/utils/profiling.py`:

```python
from adaptshot.utils.profiling import MemoryTracker, estimate_model_memory_mb

# Pre-flight estimate (no model loaded)
est = estimate_model_memory_mb("resnet18", n_classes=5)
print(f"Estimated RAM: {est['estimated_total_mb']} MB")

# Runtime profiling
with MemoryTracker("predict") as tracker:
    result = learner.predict("query.jpg")
print(f"Peak RAM during predict: {tracker.peak_mb:.1f} MB")
```

---

## Troubleshooting Benchmarks

| Issue | Likely Cause | Fix |
|-------|--------------|-----|
| `Accuracy < 50%` | Support set too small or domain-mismatched | Increase `k_shot` to ≥10, ensure support images match query lighting/background |
| `Latency > 200ms` | Heavy preprocessing or FAISS overhead on low-end CPU | Switch to `mobilenet_v3_small`, disable FAISS (`use_faiss=False`) |
| `Determinism check: ❌ FAIL` | Unpinned PyTorch version or custom CUDA ops | Use `torch==2.12.0+cpu`, run with `--device cpu`, verify `set_deterministic_seed(42)` is called |
| `ECE remains > 0.15 after 30 steps` | Severe distribution shift or noisy corrections | Increase `conformal_alpha`, lower `ACTEngine.base_threshold` for more feedback |
| `miniImageNet not found` | Dataset not downloaded | Download miniImageNet CSV to `data/mini_imagenet.csv` |
| `Memory profiling unavailable` | psutil not installed | Install with `pip install "adaptshot[dev]"` |

## Next Steps

- [API Reference](../api/core.md) → Configure thresholds, buffers, and calibration parameters
- [Memory Profiling Tutorial](../tutorials/13_profiling_memory.md) → Detailed RAM and latency analysis
- [Tutorial-Style Guides](../tutorials.md) → Try prediction and correction workflows
- [Contribute](../contributing.md) → Add new backbones, datasets, or calibration methods
