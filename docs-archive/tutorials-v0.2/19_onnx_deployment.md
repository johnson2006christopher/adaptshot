# Tutorial 19: ONNX Deployment

> **v0.2.0** | Torch-free inference with ONNX Runtime for edge and embedded devices

---

## Prerequisites

- AdaptShot v0.2.0+ installed with ONNX support: `pip install "adaptshot[onnx]"`
- OR: `pip install adaptshot onnxruntime`
- Understanding of [Quickstart](../getting-started/quickstart.md)

---

## Why ONNX?

ONNX (Open Neural Network Exchange) enables torch-free inference. Benefits:

1. **No PyTorch dependency** — ~800 MB smaller install footprint
2. **Broader hardware support** — Android (NNAPI), iOS (CoreML via onnx), WebAssembly (ONNX.js), ARM
3. **Faster cold start** — no JIT compilation overhead
4. **Deterministic inference** — no CUDA nondeterminism

AdaptShot v0.2.0 uses ONNX Runtime as the default embedding backend when `device="cpu"`.

---

## Step 1: Verify ONNX Backend

```python
from adaptshot import FewShotLearner, AdaptShotConfig

config = AdaptShotConfig(device="cpu", seed=42)
learner = FewShotLearner(config=config)

# Check which backend is active
print(f"Backend: {learner._backend}")  # "onnx" or "torch"
print(f"Backbone: {config.backbone}")  # "mobilenet_v3_small" (default for ONNX)
```

ONNX Runtime is the default. If PyTorch is installed, it will be preferred unless you explicitly force ONNX.

---

## Step 2: Force ONNX Backend

If you have PyTorch installed but want to use ONNX:

```python
import os
os.environ["ADAPTSHOT_BACKEND"] = "onnx"

config = AdaptShotConfig(
    device="cpu",
    backbone="mobilenet_v3_small",  # ONNX-exported backbone
)
learner = FewShotLearner(config=config)
```

Or export and use a custom model:

```python
# Export your own backbone to ONNX (requires PyTorch once)
import torch
import torchvision

model = torchvision.models.mobilenet_v3_small(pretrained=True)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "my_model.onnx",
    input_names=["input"],
    output_names=["embedding"],
    dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
    opset_version=13,
)
print("Exported my_model.onnx")
```

---

## Step 3: ONNX Inference Pipeline

The ONNX pipeline is identical to the PyTorch pipeline — the backend abstraction handles everything:

```python
config = AdaptShotConfig(
    device="cpu",
    backbone="mobilenet_v3_small",
    inference_mode="prototypical",
    calibration_method="temperature",
)
learner = FewShotLearner(config=config)

# Load support images — embeddings extracted via ONNX Runtime
learner.load_support_images(
    ["cat_01.jpg", "cat_02.jpg", "dog_01.jpg", "dog_02.jpg"],
    ["cat", "cat", "dog", "dog"],
)

# Predict — ONNX Runtime inference under the hood
result = learner.predict("query.jpg")
print(f"Prediction: {result.prediction}")
print(f"Confidence: {result.calibrated_confidence:.3f}")
print(f"Backend: {learner._backend}")
```

---

## Step 4: Benchmark ONNX vs PyTorch

```python
import time
import numpy as np

config_onnx = AdaptShotConfig(device="cpu", backbone="mobilenet_v3_small")
learner_onnx = FewShotLearner(config_onnx)
learner_onnx.load_support_images(
    ["cat_01.jpg", "cat_02.jpg"], ["cat", "cat"]
)

# Warm up
for _ in range(5):
    learner_onnx.predict("query.jpg")

# Benchmark
times = []
for _ in range(50):
    start = time.perf_counter()
    learner_onnx.predict("query.jpg")
    times.append((time.perf_counter() - start) * 1000)

print(f"ONNX — Mean: {np.mean(times):.1f} ms, P95: {np.percentile(times, 95):.1f} ms")
```

---

## Step 5: Torch-Free Docker Deployment

A minimal Docker setup without PyTorch:

```dockerfile
FROM python:3.10-slim

RUN pip install adaptshot onnxruntime

COPY app.py /app/
COPY support_images/ /app/support_images/

WORKDIR /app
ENV ADAPTSHOT_BACKEND=onnx

CMD ["python", "app.py"]
```

**Image size**: ~300 MB (vs ~1.2 GB with PyTorch).

---

## Supported Backbones for ONNX

| Backbone | ONNX Status | Notes |
|----------|------------|-------|
| `mobilenet_v3_small` | ✅ Built-in | Default ONNX backbone, ~25 MB |
| `resnet18` | ✅ Exportable | Requires one-time export with PyTorch |
| `mobilenet_v3_large` | ✅ Exportable | Larger, more accurate variant |
| `efficientnet_b0` | ⚠️ Preview | Requires custom export |

---

## Limitations

1. **Backbone training**: ONNX is inference-only. Fine-tuning requires PyTorch — CA-EWC head updates still work since they operate on embeddings, not the backbone.
2. **GPU acceleration**: ONNX Runtime supports GPU via `onnxruntime-gpu`, but AdaptShot's ONNX path targets CPU-first deployments.
3. **Custom preprocessing**: If your backbone expects non-standard preprocessing, export a model that includes preprocessing as the first ONNX node.

---

## Best Practices

1. **Use ONNX for deployment, PyTorch for development** — develop with PyTorch's flexibility, deploy with ONNX's efficiency
2. **Verify determinism** — ONNX inference is fully deterministic on CPU (no CUDA nondeterminism)
3. **Pre-warm the session** — first inference is slower (session initialization); run a dummy prediction at startup
4. **Cache embeddings** — `extract_embedding()` caches per-image embeddings; clear cache periodically for long-running services

---

## Next Steps

- [Installation Guide](../getting-started/installation.md) — ONNX install options
- [Architecture Deep-Dive](../guides/architecture-deep-dive.md) — back-end abstraction layer
- [Tutorial 18: End-to-End Production Workflow](18_end_to_end_workflow.md) — full deployment pipeline
