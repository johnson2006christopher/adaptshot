# Training & Continual Learning API (v0.1.2)

This document covers AdaptShot's human-in-the-loop routing, bounded replay buffer management, and correction-aware fine-tuning components. These modules operate behind the scenes in `FewShotLearner` but are exposed for advanced customization, research ablation, or integration into external pipelines.

---

## `Correction` Dataclass

A structured representation of a single human feedback event. Passed directly to `FeedbackRouter.route_feedback()`.

```python
from dataclasses import dataclass, field
from typing import Any, Dict, Union

@dataclass
class Correction:
    image_path: str
    predicted_label: Union[str, int]
    corrected_label: Union[str, int]
    raw_confidence: float
    confidence_weight: float = 1.0  # Human certainty [0.0, 1.0]
    timestamp: float = 0.0          # Unix timestamp (auto-filled if 0.0)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

!!! note "Usage"
    You typically do not instantiate this manually. It is created automatically when calling `learner.correct()` or routing feedback from the Gradio UI.

---

## `FeedbackRouter`

Orchestrates human corrections, updates calibration state, triggers fine-tuning at configurable thresholds, and enforces buffer capacity.

### Initialization
```python
from adaptshot.training.feedback_router import FeedbackRouter

router = FeedbackRouter(
    buffer_capacity: int = 100,
    fine_tune_trigger_threshold: int = 5,
    calibrator: Optional[Any] = None,
    finetune_fn: Optional[Callable] = None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `buffer_capacity` | `int` | `100` | Maximum corrections retained in replay memory |
| `fine_tune_trigger_threshold` | `int` | `5` | Number of pending corrections before triggering CA-EWC |
| `calibrator` | `CalibrationEngine` | `None` | Bound instance for online ECE/temperature updates |
| `finetune_fn` | `Callable` | `None` | Callback executed when threshold is met |

### `route_feedback(correction) -> Dict[str, Any]`
Processes a correction, updates internal state, and returns routing metadata.

**Returns:** Dictionary containing:
- `"buffer_size"`: Current buffer length
- `"pending_corrections"`: Count awaiting fine-tune trigger
- `"calibration_updated"`: `bool`
- `"fine_tuned"`: `bool`
- `"total_corrections"`: Lifetime count

### State Management Methods
| Method | Description |
|--------|-------------|
| `get_buffer() -> List[Correction]` | Returns a shallow copy of retained corrections |
| `clear_buffer() -> None` | Resets buffer, pending queue, and counters |

!!! warning "FIFO Fallback in v0.1.0"
    When `buffer_capacity` is exceeded, the router uses simple FIFO eviction. UP-UGF intelligent pruning is applied at the `FewShotLearner` level, not inside the router itself.

---

## `CAEWCFinetuner`

Implements Correction-Aware Elastic Weight Consolidation for head-only continual learning. Prevents catastrophic forgetting while adapting to new human corrections.

### Initialization
```python
from adaptshot.training.finetune import CAEWCFinetuner
import torch

# model: A lightweight classification head (e.g., torch.nn.Linear)
finetuner = CAEWCFinetuner(
    model: torch.nn.Module,
    device: str = "cpu",
    ewc_lambda: float = 0.1,
    learning_rate: float = 1e-4,
    epochs: int = 5,
    batch_size: int = 16
)
```

### `update_fisher(data_loader) -> Dict[str, torch.Tensor]`
Computes the diagonal Fisher Information Matrix on a representative support set. Must be called **before** `finetune()` to establish importance weights.

**Input:** PyTorch `DataLoader` yielding `(embeddings, labels)` batches.
**Output:** Dictionary mapping parameter names to Fisher tensors. Also snapshots `old_params` for EWC penalty computation.

### `finetune(new_embeddings, new_labels, confidence_weights=None) -> None`
Runs head-only optimization with correction-aware regularization.

```python
finetuner.finetune(
    new_embeddings: torch.Tensor,      # [N, D]
    new_labels: torch.Tensor,          # [N] (integer indices)
    confidence_weights: torch.Tensor = None  # [N] in [0.0, 1.0]
)
```

**Behavior:**
- If `confidence_weights` is `None`, defaults to `1.0` (full adaptation)
- EWC penalty scales with `(1 - confidence_weight)`: high-confidence corrections face less regularization, allowing faster adaptation; uncertain corrections preserve prior knowledge
- Falls back to standard cross-entropy fine-tuning with a warning if `update_fisher()` hasn't been called yet

---

## `UPUGFPruner`

Uncertainty-Guided Forgetting. Replaces naive eviction with a multiplicative utility score that prioritizes informative, recent, and diverse examples.

### Initialization
```python
from adaptshot.training.up_ugf import UPUGFPruner

pruner = UPUGFPruner(
    capacity: int = 100,
    uncertainty_weight: float = 1.0,
    recency_weight: float = 1.0,
    redundancy_weight: float = 1.0,
    recency_decay: float = 0.01
)
```

### Scoring Formula
For each embedding `e`:
```
Score(e) = (1 - u(e))^w_unc × exp(-λ × Δt)^w_rec × (1 - max_sim_to_same_class)^w_red
```
- `u(e)`: Prediction uncertainty (lower = more confident)
- `Δt`: Time since last access
- `max_sim_to_same_class`: Highest cosine similarity to other examples of the same label

### `prune(embeddings, labels, uncertainties, last_access_times) -> Tuple[np.ndarray, ...]`
Enforces capacity by returning the top-K highest-scoring examples. All input arrays must be NumPy types of matching length `N`.

!!! note "Computational Cost"
    Redundancy computation requires an `N×N` cosine similarity matrix. For `capacity ≤ 100`, this completes in <5ms on a standard CPU. Do not use for buffers >1,000 examples without caching.

---

## Constraints & Notes

| Constraint | Explanation |
|------------|-------------|
| **Head-Only Fine-Tuning** | Backbone weights remain frozen. Only the classification head is updated during `finetune()`. This is intentional for stability and CPU efficiency. |
| **No Distributed Training** | All operations are single-threaded/single-process. DDP or FSDP support is planned for v0.3.0+. |
| **Fisher Approximation** | Uses diagonal Fisher (per-parameter variance). Full matrix or Kronecker approximations are not implemented. |
| **Pruning Fallback** | During initial buffer population (< capacity), no pruning occurs. FIFO eviction applies only when capacity is strictly exceeded. |
| **Confidence Weight Calibration** | `confidence_weight` is user-provided. The library does not currently validate human certainty against historical accuracy. |

## Next Steps
- [Configuration & Utils API](config.md) -> `AdaptShotConfig`, determinism, I/O helpers
- [Contributing](../contributing.md) -> Extension points for new fine-tuning or pruning strategies
- [Changelog](../changelog.md) -> Track upcoming v0.2.0 improvements
