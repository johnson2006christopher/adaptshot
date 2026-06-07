# Installation

AdaptShot v0.1.1 is a Python package for CPU-first few-shot image classification.

## Requirements

- Python 3.9+
- A CPU-only environment is supported by default
- Internet access for the first run if torchvision downloads pretrained backbone weights

!!! note "Version"
    These commands target AdaptShot `v0.1.1`.

## Install From PyPI

```bash
pip install adaptshot
```

## Install From The GitHub Release Wheel

```bash
pip install https://github.com/johnson2006christopher/adaptshot/releases/download/v0.1.1/adaptshot-0.1.1-py3-none-any.whl
```

## Optional Extras

```bash
# FAISS-CPU similarity search
pip install "adaptshot[faiss]"

# Gradio UI dependencies
pip install "adaptshot[ui]"

# Offline Studio GUI
pip install "adaptshot[gui]"

# Development tools
pip install "adaptshot[dev]"
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
0.1.1
FewShotLearner
```

!!! note "Offline By Default"
    AdaptShot v0.1.1 builds backbones without pretrained weight downloads, so the first embedding extraction stays offline.

## Verification Checklist

- [ ] `python --version` shows Python 3.9 or newer.
- [ ] `pip install adaptshot` completes without dependency errors.
- [ ] The verification command prints `0.1.1`.
- [ ] You can import `FewShotLearner`.
