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

## Install MziziGuard (Crop Disease Detection App)

MziziGuard requires the `[ui]` extras:

```bash
pip install "adaptshot[ui]"
```

Launch the web application:

```bash
python -m examples.mziziguard.app
# Opens http://localhost:7860
```

## Platform-Specific Notes

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
python3 -m venv adaptshot_env
source adaptshot_env/bin/activate
pip install "adaptshot[ui]"
```

### Windows

```bash
# In PowerShell or Command Prompt
python -m venv adaptshot_env
adaptshot_env\Scripts\activate
pip install "adaptshot[ui]"
```

### macOS

```bash
python3 -m venv adaptshot_env
source adaptshot_env/bin/activate
pip install "adaptshot[ui]"
```

### Raspberry Pi / ARM

```bash
# Install PyTorch for ARM first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install "adaptshot[ui]"
```

## Dependencies

### Core Dependencies (always installed)

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.0.0 | Deep learning framework |
| `torchvision` | ≥0.15.0 | Pretrained backbones |
| `numpy` | ≥1.24.0 | Numerical operations |
| `Pillow` | ≥9.0.0 | Image loading and processing |

### Optional Dependencies

| Extra | Packages | Purpose |
|-------|----------|---------|
| `[faiss]` | `faiss-cpu` | Accelerated similarity search |
| `[ui]` | `gradio` | Web UI (MziziGuard, Pilot Dashboard) |
| `[gui]` | `gradio`, `pandas`, `onnx`, `onnxscript` | Full offline Studio GUI |
| `[dev]` | `pytest`, `mypy`, `ruff`, `pre-commit` | Development and testing |
| `[all]` | All of the above | Everything |

## Troubleshooting Installation

| Problem | Solution |
|---------|----------|
| `pip: command not found` | Install pip: `python -m ensurepip --upgrade` |
| `ModuleNotFoundError: torch` | PyTorch installation failed. Try `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `Could not find a version that satisfies the requirement` | Check Python version: `python --version` (needs ≥3.9) |
| Permission denied on Linux | Use a virtual environment: `python -m venv .venv && source .venv/bin/activate` |
| CUDA errors on import | AdaptShot defaults to CPU. Set `device="cpu"` in config. |
| Backbone download fails | First run downloads ~45MB from torchvision. Ensure internet on first run. |

## Verification Checklist

- [ ] `python --version` shows Python 3.9 or newer.
- [ ] `pip install adaptshot` completes without dependency errors.
- [ ] The verification command prints `0.1.1`.
- [ ] You can import `FewShotLearner`.
- [ ] (Optional) MziziGuard launches at http://localhost:7860.
- [ ] (Optional) `pytest tests/ -v` passes all tests (from source checkout).

---

*Created by [Johnson Christopher Hassan](https://github.com/johnson2006christopher)*  
*Connect on [LinkedIn](https://www.linkedin.com/in/johnson-hassan-935124311/)*  
*Project: [github.com/johnson2006christopher/adaptshot](https://github.com/johnson2006christopher/adaptshot)*
