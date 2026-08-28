# Installation

AdaptShot v0.2.0 is a Python package for CPU-first few-shot image classification with conformal prediction, contrastive learning, and human-in-the-loop continual learning.

## Requirements

- Python 3.9+
- A CPU-only environment is supported by default (PyTorch is optional)
- ~15 MB disk space for core dependencies (numpy + Pillow)
- ~45 MB additional for ImageNet pretrained backbone weights (auto-downloaded on first use)
- Internet access for the first run only (backbone weight download)

!!! note "Version"
    These commands target AdaptShot `v0.2.0`.

## Install From PyPI

```bash
pip install adaptshot
```

> **Fast install**: Core dependencies (numpy, Pillow) install in under 60 seconds. No GPU drivers, no CUDA, no 2 GB downloads. PyTorch is optional.

## Install From GitHub Release

```bash
pip install https://github.com/johnson2006christopher/adaptshot/releases/download/v0.2.0/adaptshot-0.2.0-py3-none-any.whl
```

## Optional Extras

```bash
# PyTorch for training, fine-tuning, and custom backbones
pip install "adaptshot[torch]"

# FAISS-CPU similarity search (recommended for >100 support images)
pip install "adaptshot[faiss]"

# Gradio UI dependencies for the Pilot Dashboard
pip install tambua

# Offline Studio GUI (includes ONNX Runtime for torch-free inference)
pip install "adaptshot[gui]"

# Development tools (testing, linting, benchmarking)
pip install "adaptshot[dev]"

# Everything
pip install "adaptshot[all]"
```

## Install From Source

```bash
git clone https://github.com/johnson2006christopher/adaptshot.git
cd adaptshot
pip install -e ".[dev]"
```

## ONNX Runtime Support (Torch-Free Inference)

AdaptShot v0.2.0 supports ONNX Runtime as a lightweight backend when PyTorch is not installed. To use this:

```bash
# Export backbones to ONNX format (requires torch)
python scripts/export_backbones.py

# Install with ONNX Runtime support
pip install "adaptshot[gui]"  # Includes onnxruntime
```

The ONNX backend automatically activates when torch is unavailable and ONNX models are present in `src/adaptshot/data/`.

## Verify The Install

```bash
python - <<'PY'
import adaptshot
from adaptshot import FewShotLearner, ConformalEngine

print(adaptshot.__version__)
print(FewShotLearner.__name__)
print(ConformalEngine.__name__)
PY
```

Expected output:

```text
0.2.0-dev
FewShotLearner
ConformalEngine
```

!!! note "Pretrained Weights"
    AdaptShot v0.2.0 uses ImageNet-pretrained backbone weights by default.
    The weights are downloaded automatically on first use (~45 MB for ResNet-18,
    ~10 MB for MobileNetV3-Small). This ensures embeddings are meaningful and
    match the ImageNet-normalized preprocessing pipeline.

## Minimal Dependency Setup

For ultra-lightweight deployments (no PyTorch, no ONNX Runtime):

```bash
pip install adaptshot
# Core dependencies: numpy + Pillow only (~15 MB total)
```

The library falls back gracefully: ONNX backend is tried first if available, then a clear error message guides you to install either `[torch]` or `[gui]` extras.

## Verification Checklist

- [ ] `python --version` shows Python 3.9 or newer.
- [ ] `pip install adaptshot` completes in under 60 seconds.
- [ ] The verification command prints `0.2.0-dev`.
- [ ] You can import `FewShotLearner` and `ConformalEngine`.

---

## Next Steps

- **[Quick Start](quickstart.md)** — Make your first prediction in 5 minutes
- **[Beginner 101](beginner-101.md)** — Learn the concepts behind AdaptShot
- **[⭐ Star on GitHub](https://github.com/johnson2006christopher/adaptshot)** — Support the project and stay updated
- **[📱 Join WhatsApp](https://chat.whatsapp.com/J6AbrvbjmBc5XXX2fnN6RK)** — Connect with the community
