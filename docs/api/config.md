# 📄 File 10: `docs/api/config.md`

### 📝 Content
```markdown
# Configuration & Utilities API (v0.1.0)

This document covers AdaptShot's immutable configuration schema, deterministic execution utilities, and I/O helpers. These components provide the foundational guarantees for reproducibility, safe file handling, and cross-framework data conversion.

---

## `AdaptShotConfig`

A frozen dataclass that centralizes all pipeline hyperparameters. Immutability prevents accidental state mutation during inference or training, which is critical for deterministic reproducibility.

### Initialization
```python
from adaptshot.config.settings import AdaptShotConfig

config = AdaptShotConfig(
    backbone: str = "resnet18",
    device: str = "cpu",
    seed: int = 42,
    n_way: int = 5,
    k_shot: int = 10,
    query_size: int = 15,
    use_faiss: bool = False,
    faiss_nprobe: int = 8,
    calibration_method: str = "temperature",
    ece_n_bins: int = 15,
    temperature_init: float = 1.0,
    max_buffer_size: int = 100,
    verbose: bool = True,
    log_dir: Optional[str] = None
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backbone` | `Literal["resnet18", "mobilenet_v3_small"]` | `"resnet18"` | Pretrained feature extractor |
| `device` | `Literal["cpu", "cuda", "mps"]` | `"cpu"` | Execution target. CUDA/MPS are optional. |
| `seed` | `int` | `42` | Random seed for PyTorch, NumPy, Python, and hash seed |
| `n_way` | `int` | `5` | Number of classes per few-shot episode |
| `k_shot` | `int` | `10` | Support examples per class |
| `query_size` | `int` | `15` | Query examples per class for evaluation |
| `use_faiss` | `bool` | `False` | Enable FAISS-CPU index for large support sets |
| `calibration_method` | `Literal["temperature", "conformal", "none"]` | `"temperature"` | Post-hoc confidence scaling strategy |
| `max_buffer_size` | `int` | `100` | Maximum replay buffer capacity (enforced by UP-UGF) |
| `verbose` | `bool` | `True` | Enable INFO-level logging during pipeline execution |

### Validation Constraints
`AdaptShotConfig` enforces immediate validation on instantiation:
- `k_shot > 0` and `n_way > 0`
- `max_buffer_size >= 10`
- If `device="cuda"` but `torch.cuda.is_available()` is `False`, a `RuntimeWarning` is issued and downstream logic falls back to CPU automatically.

!!! warning "Immutability"
    `AdaptShotConfig` is created with `@dataclass(frozen=True)`. Attempting to modify attributes after initialization (e.g., `config.device = "cuda"`) will raise `dataclasses.FrozenInstanceError`. Create a new instance instead.

---

## Determinism Utilities

Guarantee bit-exact reproducibility across runs, hardware, and operating systems.

### `set_deterministic_seed(seed, device=None)`
```python
from adaptshot.utils.determinism import set_deterministic_seed
import torch

device = torch.device("cpu")
set_deterministic_seed(seed=42, device=device)
```
**Behavior:**
- Sets `random.seed()`, `np.random.seed()`, `torch.manual_seed()`
- Enables deterministic cuDNN algorithms if `device.type == "cuda"`
- Sets `os.environ["PYTHONHASHSEED"]` to prevent dictionary/set ordering randomness

### `verify_determinism(fn, *args, runs=3, seed=42, tolerance=1e-7, **kwargs)`
```python
from adaptshot.utils.determinism import verify_determinism

# Verify that embedding extraction is reproducible
is_deterministic = verify_determinism(
    fn=extract_embedding,
    image_path="test.jpg",
    config=config,
    runs=3,
    seed=42
)
print(f"Deterministic: {is_deterministic}")
```
**Behavior:**
- Executes `fn` multiple times with incrementally offset seeds
- Compares outputs using `np.allclose` with strict absolute tolerance (`1e-7`)
- Returns `True` if all runs match, `False` otherwise
- Intended for CI/CD pipelines and benchmark validation scripts

---

## I/O Utilities

Safe path validation, JSON serialization, and cross-framework tensor conversion.

### `validate_path(path, must_exist=False, is_dir=False)`
```python
from adaptshot.utils.io import validate_path
from pathlib import Path

# Normalize and resolve path
p = validate_path("results/metrics.json")
# Create directory if missing
d = validate_path("checkpoints/", is_dir=True)
# Fail if file doesn't exist
f = validate_path("missing.txt", must_exist=True)  # Raises FileNotFoundError
```

### `save_json(data, path, indent=2)`
```python
from adaptshot.utils.io import save_json

metrics = {"accuracy": 0.72, "ece": 0.04}
save_json(metrics, "results/v1.json")
```
**Behavior:**
- Creates parent directories automatically
- Uses UTF-8 encoding with `ensure_ascii=False`
- Pretty-formats with specified indentation

### `load_json(path)`
```python
from adaptshot.utils.io import load_json

data = load_json("results/v1.json")
```
**Behavior:**
- Validates path existence before reading
- Returns parsed dictionary

### `tensor_to_numpy(tensor)`
```python
from adaptshot.utils.io import tensor_to_numpy
import torch

t = torch.randn(1, 512, requires_grad=True, device="cuda")
arr = tensor_to_numpy(t)
```
**Behavior:**
- Safely detaches gradients if present
- Moves tensor to CPU if on CUDA/MPS
- Returns `np.ndarray` with shared memory layout
- Prevents `RuntimeError` when passing PyTorch outputs to NumPy/FAISS pipelines

---

## ⚠️ v0.1.0 Constraints & Notes

| Component | Limitation | Workaround / Note |
|-----------|------------|-------------------|
| `AdaptShotConfig` | Frozen dataclass; cannot be mutated in-place | Create new instance with `dataclasses.replace()` if needed |
| `verify_determinism` | Only supports `torch.Tensor` or `np.ndarray` return types | Wrap custom functions to cast outputs before verification |
| `validate_path` | Does not handle cloud storage paths (S3, GCS) | Use local filesystem or mount cloud storage to local directory |
| `tensor_to_numpy` | Does not preserve gradient computation graph | Use only for inference/post-processing, not during training |

## ▶️ Next Steps
- [Contributing Guidelines](../contributing.md) → Development workflow and PR standards
- [Changelog](../changelog.md) → Version history and known limitations
- [GitHub Repository](https://github.com/johnson2006christopher/adaptshot) → Issue tracker, discussions, and roadmap
