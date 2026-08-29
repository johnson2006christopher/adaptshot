# Training & Continual Learning API (v0.2.0)

This document covers AdaptShot's human-in-the-loop routing, bounded replay buffer management, and head-only fine-tuning components. These modules operate behind the scenes in `FewShotLearner` but are exposed for advanced customization, research ablation, or integration into external pipelines.

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
    You typically do not instantiate this manually. It is created automatically when calling `learner.correct()` or routing feedback from the UI.

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

---

## `CAEWCFinetuner`

> Not exported from `adaptshot`. Reached through `FeedbackRouter`; import from `adaptshot.training.finetune` at your own risk. Requires the `torch` extra.

Implements **head-only** Correction-Aware Fine-Tuning via Fisher Information regularization.

!!! important "v0.2.0 Scope Clarification"
    This fine-tuner operates ONLY on the classification head — a single `nn.Linear(embedding_dim, n_classes)` layer containing ~2K parameters for 5-way ResNet-18. It does **not** fine-tune the frozen backbone. The term "Elastic Weight Consolidation" refers to the Fisher-weighted regularization applied to these ~2K head parameters, not a full-network EWC implementation. For full backbone fine-tuning, use a dedicated GPU-accelerated training pipeline.

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
Computes the diagonal Fisher Information Matrix on representative support set data. Must be called **before** `finetune()`.

**Output:** Dictionary mapping parameter names to Fisher tensors; snapshots `old_params` for EWC penalty.

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
- High-confidence corrections → less regularization (faster adaptation)
- Low-confidence corrections → full penalty (preserve prior knowledge)
- Falls back to standard cross-entropy with a warning if `update_fisher()` hasn't been called

---

## `UPUGFPruner`

Uncertainty-Guided Forgetting with LSH-accelerated redundancy scoring (v0.2.0).

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
- `max_sim_to_same_class`: Highest similarity to same-label examples

### v0.2.0: LSH Acceleration
Redundancy computation uses two modes:
- **N ≤ 100**: Exact cosine similarity (O(N²), <5ms)
- **N > 100**: Random projection LSH approximate mode (O(N log N))

This eliminates the previous O(N²) bottleneck for large buffers while maintaining pruning quality for smaller ones.

### `prune(embeddings, labels, uncertainties, last_access_times) -> Tuple[np.ndarray, ...]`
Enforces capacity by returning the top-K highest-scoring examples.

---

## Constraints & Notes

| Constraint | Explanation |
|------------|-------------|
| **Head-Only Fine-Tuning** | Only the classification head (~2K params) is updated. Backbone weights remain frozen. |
| **Fisher Approximation** | Uses diagonal Fisher (per-parameter variance). Full matrix approximations are not implemented. |
| **Pruning Modes** | Exact for N≤100, LSH approximate for N>100. Both are CPU-efficient. |
| **Confidence Weight** | User-provided. The library does not validate human certainty against historical accuracy. |
| **No Distributed Training** | All operations are single-threaded/single-process. |

## Next Steps
- [Configuration & Utils API](config.md) → `AdaptShotConfig`, determinism, I/O helpers
- [Architecture Deep-Dive](../guides/architecture-deep-dive.md) → Module map and data flow
- [Migration Guide](../guides/migration-v0.1-to-v0.2.md) → Upgrade from v0.1.x

---

## `ContrastivePrototypeLearner` and `ContrastiveConfig`

> **Experimental.** May change in a minor release without a deprecation cycle. See [`adaptshot.api`](reference.md#stable-and-experimental) and the deprecation policy in CONTRIBUTING.md.

Moved here from `adaptshot.core` in 0.3.0, because training a projection head by
InfoNCE gradient descent is training. The old import path
`adaptshot.core.contrastive` still works, warns, and is removed in 0.4.0. Import
from the top level:

```python
from adaptshot import ContrastiveConfig, ContrastivePrototypeLearner
```

Full method table in the [API reference](reference.md#contrastiveprototypelearner).
