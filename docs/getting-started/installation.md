# Installation

AdaptShot v0.1.2 is a Python package for CPU-first few-shot image classification.

## Requirements

- Python 3.9+
- A CPU-only environment is supported by default (PyTorch is optional)
- ~15 MB disk space for core dependencies (numpy + Pillow)
- Internet access for the first run to download ImageNet pretrained backbone weights

!!! note "Version"
    These commands target AdaptShot `v0.1.2`.

## Install From PyPI

```bash
pip install adaptshot
```

> **Fast install**: Core dependencies (numpy, Pillow) install in under 60 seconds. No GPU drivers, no CUDA, no 2 GB downloads. PyTorch is optional.

## Install From The GitHub Release Wheel

```bash
pip install https://github.com/johnson2006christopher/adaptshot/releases/download/v0.1.2/adaptshot-0.1.2-py3-none-any.whl
```

## Optional Extras

```bash
# PyTorch for training, fine-tuning, and custom backbones
pip install "adaptshot[torch]"

# FAISS-CPU similarity search
pip install "adaptshot[faiss]"

# Gradio UI dependencies
pip install "adaptshot[ui]"

# Offline Studio GUI (includes ONNX Runtime)
pip install "adaptshot[gui]"

# Development tools
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

## Verify The Install

```bash
python - <<'PY'
import adaptshot
from adaptshot import FewShotLearner

print(adaptshot.__version__)
print(FewShotLearner.__name__)
PY
```

Expected output:

```text
0.1.2
FewShotLearner
```

!!! note "Pretrained Weights"
    AdaptShot v0.1.2 uses ImageNet-pretrained backbone weights (`IMAGENET1K_V1`)
    by default. The weights are downloaded automatically on first use (~45 MB for ResNet-18).
    This ensures embeddings are meaningful and match the ImageNet-normalized preprocessing pipeline.

## Verification Checklist

- [ ] `python --version` shows Python 3.9 or newer.
- [ ] `pip install adaptshot` completes in under 60 seconds.
- [ ] The verification command prints `0.1.2`.
- [ ] You can import `FewShotLearner`.
