---
title: "08 — Configuration, Determinism, and Safe I/O"
nav_order: 8
---

This chapter teaches the support code around AdaptShot: how configuration is validated, how determinism is enforced, and how safe file I/O works. These utilities are part of the library's real behavior, so we only describe what exists in `src/adaptshot/`.

!!! note
    Think of this as the library's control room. It does not make predictions itself, but it keeps predictions reproducible, safe, and easy to debug.

## 1. `AdaptShotConfig`: The Control Panel

`AdaptShotConfig` in [src/adaptshot/config/settings.py](src/adaptshot/config/settings.py) is a frozen dataclass. Frozen means the settings are locked after creation, so the pipeline stays predictable.

Important fields:

| Field | Meaning | Analogy |
|---|---|---|
| `backbone` | Choose `resnet18` or `mobilenet_v3_small` | Choosing the engine in a car |
| `device` | CPU-first by default; can be `cpu`, `cuda`, or `mps` | Choosing which kitchen to cook in |
| `seed` | Random seed for repeatability | A recipe number that produces the same cake each time |
| `use_faiss` | Toggle FAISS-CPU similarity search | Choosing a faster filing cabinet for larger stacks of papers |
| `eco_mode` | Enable preview-based early exit | Checking a quick thumbnail before opening the full photo |
| `early_exit_threshold` | Similarity threshold for eco-mode fast path | The bar a quick check must pass before skipping the longer route |
| `calibration_method` | `temperature`, `conformal`, or `none` | Choosing how the library measures certainty |
| `max_buffer_size` | Replay buffer capacity | The size of your notebook before you start archiving pages |

### Validation Rules

The config validates itself in `__post_init__`:

- `n_way` and `k_shot` must be positive integers.
- `max_buffer_size` must be at least 10.
- `early_exit_threshold` must be between 0.5 and 1.0.
- If `device == "cuda"` and CUDA is unavailable, a warning is issued and runtime logic falls back to CPU.

Example:

```python
from adaptshot import AdaptShotConfig

config = AdaptShotConfig(device="cpu", seed=42, eco_mode=True, early_exit_threshold=0.95)
print(config.seed)
# Expected output: 42
```

!!! warning
    `AdaptShotConfig` is frozen. That means you should create a new config if you need to change settings.

## 2. Determinism: Same Input, Same Result

Determinism means repeated runs should produce the same output when the seed and inputs are the same. See [src/adaptshot/utils/determinism.py](src/adaptshot/utils/determinism.py).

`set_deterministic_seed(seed, device)` sets:

- Python's `random` seed
- NumPy's seed
- PyTorch's seed
- `PYTHONHASHSEED`

`verify_determinism(fn, *args, runs=3, seed=42, tolerance=1e-7, **kwargs)` runs a function multiple times and checks whether the outputs match.

Analogy: determinism is like using the same measuring cup and same recipe steps every time you bake.

```python
import numpy as np

from adaptshot.utils.determinism import verify_determinism

def constant_output() -> np.ndarray:
    return np.asarray([1.0, 2.0, 3.0], dtype=np.float32)

print(verify_determinism(constant_output, runs=3, seed=42))
# Expected output: True
```

### Why This Matters

Determinism is used in the benchmark tooling and helps you compare changes safely. The library is designed to be CPU-first, so deterministic execution is part of the practical workflow rather than a bonus.

## 3. Safe Path Handling

The I/O helpers in [src/adaptshot/utils/io.py](src/adaptshot/utils/io.py) keep file handling explicit.

### `validate_path(path, must_exist=False, is_dir=False)`

This function resolves a path and optionally checks that it exists. If `is_dir=True` and the directory does not exist, it creates it.

Analogy: before storing tools in a cabinet, you check whether the cabinet exists; if not, you build it.

```python
from adaptshot.utils.io import validate_path

path = validate_path("results/demo", is_dir=True)
print(path.name)
# Expected output: demo
```

If `must_exist=True` and the path does not exist, it raises `FileNotFoundError` with the message `Path does not exist: <resolved path>`.

### `save_json(data, path, indent=2)`

This writes a dictionary to a JSON file and creates parent directories if needed.

### `load_json(path)`

This reads a JSON file back into a Python dictionary.

### `tensor_to_numpy(tensor)`

This converts a PyTorch tensor to a NumPy array safely, detaching gradients and moving CUDA tensors to CPU first.

Example:

```python
import torch

from adaptshot.utils.io import tensor_to_numpy

tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
array = tensor_to_numpy(tensor)
print(array.tolist())
# Expected output: [1.0, 2.0, 3.0]
```

## 4. How These Pieces Work Together

The benchmark and inference paths use these utilities in practical ways:

- `benchmarks/energy_profile.py` calls determinism helpers to check reproducibility.
- `FewShotLearner.save()` and `FewShotLearner.load()` manage checkpoint paths directly in [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).
- The extractor and similarity stack use NumPy and CPU-safe conversions.

Analogy: configuration is the plan, determinism is the repeatable procedure, and I/O helpers are the clean shelves where results go.

## 5. Practical Patterns

### Pattern A: Make a results directory before writing files

```python
from adaptshot.utils.io import validate_path

results_dir = validate_path("results/my_run", is_dir=True)
print(results_dir.exists())
# Expected output: True
```

### Pattern B: Keep seeds explicit in demos and benchmarks

```python
from adaptshot.utils.determinism import set_deterministic_seed

set_deterministic_seed(42)
print("seeded")
# Expected output: seeded
```

### Pattern C: Treat config as immutable

```python
from adaptshot import AdaptShotConfig

config = AdaptShotConfig(device="cpu")
print(config.device)
# Expected output: cpu
```

## 6. What Not To Assume

- Do not assume GPU availability; the code defaults to CPU and warns if CUDA is requested but unavailable.
- Do not assume a missing path will be silently created unless you use `validate_path(..., is_dir=True)`.
- Do not assume deterministic results if you change the seed or use a different input image.

## 7. Verification Checklist

- [ ] I can explain what `AdaptShotConfig` controls.
- [ ] I know which settings are validated automatically.
- [ ] I can call `set_deterministic_seed()` and `verify_determinism()`.
- [ ] I can use `validate_path()` to prepare a directory.
- [ ] I can convert a tensor to NumPy with `tensor_to_numpy()`.
- [ ] I know where to find the source code for all of these helpers.
