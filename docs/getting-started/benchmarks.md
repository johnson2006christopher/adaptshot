# Benchmarks & Reproducibility

AdaptShot is engineered for transparency and reproducibility. All performance metrics are measured on CPU-only hardware with deterministic seeding. This document provides exact reproduction commands, expected results, and hardware-tier expectations for v0.1.1.

## 🧪 Running Benchmarks

### 1. Validation Suite (Tests + Smoke Benchmark)
```bash
# Activate your virtual environment first
source venv/bin/activate

# Run all unit tests
pytest tests/ -v

# Run minimal smoke benchmark (CIFAR-10 subset, 5-way, 10-shot)
python -m benchmarks.run_benchmark --smoke-test --seed 42
```

### 2. Continuous Learning Integration
```bash
# Simulate human-in-the-loop prediction → correction → fine-tuning loop
python -m benchmarks.day4_integration
```

## 📊 Expected Results (Reference Hardware: Intel Core i5-1135G7, 16GB RAM, Ubuntu 22.04)

| Metric | Value | Notes |
|--------|-------|-------|
| **Few-shot Accuracy** | ~65–72% | 5-way, 10-shot CIFAR-10 subset. Frozen ResNet-18 + cosine similarity baseline. |
| **Avg Inference Latency** | 90–110 ms | Per-image prediction (embedding + similarity + calibration). |
| **P95 Inference Latency** | 100–140 ms | Upper bound including OS scheduling variance. |
| **RAM Footprint** | < 200 MB | Includes model weights, support embeddings, and replay buffer (≤100 items). |
| **ECE (Initial)** | 0.05–0.12 | Calibrates downward as sliding window accumulates ≥10 predictions. |
| **Determinism** | ✅ PASS | Bit-exact outputs across 3 runs with `--seed 42`. |

!!! warning "Context Matters"
    These numbers are reference points, not leaderboard targets. Real-world accuracy depends heavily on:
    - Domain similarity between support set and query images
    - Lighting, resolution, and background consistency
    - Quality and confidence of human corrections during continual learning

## 💻 Hardware-Tier Expectations

| Device | Expected Latency | Recommended Config |
|--------|------------------|-------------------|
| **Modern Laptop CPU** (i5/Ryzen 5, 4+ cores) | < 120 ms | `resnet18`, CPU mode, default buffer size |
| **Single-Board Computer** (Raspberry Pi 4) | 150–250 ms | `mobilenet_v3_small`, CPU mode, `max_buffer_size=30` |
| **Legacy Office PC** (4GB RAM, HDD) | < 200 ms | `mobilenet_v3_small`, disable FAISS, `max_buffer_size=20` |
| **GPU System** (CUDA-capable) | ~30–50 ms | Set `device="cuda"` in `AdaptShotConfig`. Gains are inference-only; training remains head-only. |

## 🔬 Reproducibility Guarantees

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

## 📉 Understanding Calibration & ECE Behavior

In v0.1.1, the `CalibrationEngine` uses a sliding window (default: 200 predictions when `max_buffer_size=100`) to fit temperature scaling online:
- **Steps 1–9**: ECE may fluctuate as the window populates. Confidence scores are uncalibrated but tracked.
- **Steps 10+**: Temperature refits automatically. ECE typically drops as the model adapts to local confidence-accuracy dynamics.
- **Domain Shift**: If query distribution changes sharply, ECE will temporarily rise until the window adjusts. This is expected and indicates the system is correctly detecting uncertainty rather than masking it.

!!! note "No Validation Set Required"
    Unlike traditional post-hoc calibration, AdaptShot does not require a held-out validation dataset. Calibration adapts continuously from live inference and human feedback, making it suitable for few-shot, low-data deployments.

## 🛠️ Troubleshooting Benchmarks

| Issue | Likely Cause | Fix |
|-------|--------------|-----|
| `Accuracy < 50%` | Support set too small or domain-mismatched | Increase `k_shot` to ≥10, ensure support images match query lighting/background |
| `Latency > 200ms` | Heavy preprocessing or FAISS overhead on low-end CPU | Switch to `mobilenet_v3_small`, disable FAISS (`use_faiss=False`) |
| `Determinism check: ❌ FAIL` | Unpinned PyTorch version or custom CUDA ops | Use `torch==2.12.0+cpu`, run with `--device cpu`, verify `set_deterministic_seed(42)` is called before inference |
| `ECE remains > 0.15 after 30 steps` | Severe distribution shift or noisy corrections | Increase `calibration_method="conformal"` in config, or lower `ACTEngine.base_threshold` to trigger more feedback |

## Next Steps
- [API Reference](../api/core.md) -> Configure thresholds, buffers, and calibration parameters
- [Tutorial-Style Guides](../tutorials.md) -> Try prediction and correction workflows
- [Contribute](../contributing.md) -> Add new backbones, datasets, or calibration methods

---

*Created by [Johnson Christopher Hassan](https://github.com/johnson2006christopher)*  
*Connect on [LinkedIn](https://www.linkedin.com/in/johnson-hassan-935124311/)*  
*Project: [github.com/johnson2006christopher/adaptshot](https://github.com/johnson2006christopher/adaptshot)*
