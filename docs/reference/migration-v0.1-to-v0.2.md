# Migration Guide: v0.1.x to v0.2.0

This guide helps you upgrade AdaptShot from v0.1.x to v0.2.0 with minimal disruption.

---

## Quick Summary

v0.2.0 is a **production hardening** release. The core API (`FewShotLearner.predict()`, `correct()`, `save()`, `load()`) is backward-compatible. All v0.1.x code continues to work. What changed:

- **New features** that are opt-in and additive
- **Hardened implementations** of existing algorithms (no API changes)
- **Expanded PredictionResult** with new fields
- **Schema migration** for saved checkpoints (automatic)

---

## Breaking Changes

### None — v0.2.0 is fully backward-compatible

Your existing v0.1.x code will run unchanged. The breaking changes listed below are optional code paths you can adopt at your own pace.

---

## What Changed Under the Hood

| Area | v0.1.x | v0.2.0 | Action Required |
|------|--------|--------|----------------|
| Conformal prediction | Split mode only | Split + cross-conformal modes, with automatic LOO self-calibration at load time | Set `conformal_mode="cross"` for tighter prediction sets via k-fold averaging |
| Uncertainty | Raw empirical covariance | Shrinkage covariance Mahalanobis | None — automatic improvement |
| Contrastive learning | Fixed projection head | Gradient-trained W₁,b₁,W₂,b₂ | None — automatic improvement |
| ACT thresholds | Asymmetric updates | Symmetric + mean-reversion | None — automatic stability |
| UP-UGF pruning | O(N²) exact similarity | LSH-accelerated ~O(N log N) | None — faster on large buffers |
| Calibration | Single temperature estimate | Bootstrap temperature (small windows) | None — more stable early calibration |
| Explainability | Per-prediction attribution | + Historical penalty tracking | None — additive feature |
| Profiling | Manual tracemalloc | MemoryTracker API | Optional — use for production monitoring |
| Backbone cache | Unbounded accumulation | clear_backbone_cache() | Call periodically in long-running services |

---

## New Config Fields (v0.2.0)

These fields were added to `AdaptShotConfig`. They all have sensible defaults — you don't need to set them:

```python
from adaptshot import AdaptShotConfig

# v0.2.0 config with new fields
config = AdaptShotConfig(
    device="cpu",
    seed=42,
    # --- New in v0.2.0 ---
    conformal_alpha=0.10,         # default: 0.10 (90% coverage)
    conformal_mode="cross",       # optional: "cross" for k-fold cross-conformal
    uncertainty_mode="entropy",   # default: "entropy"
    explainability_enabled=True,  # default: False
)
```

To migrate, you can either:
1. **Do nothing** — your existing configs work
2. **Add new fields** — enable v0.2.0 features explicitly

---

## PredictionResult Changes

`result` objects now include additional fields:

```python
result = learner.predict("query.jpg")

# v0.1.x fields (still present)
print(result.prediction)
print(result.calibrated_confidence)
print(result.act_action)

# New in v0.2.0
print(result.conformal_set)        # list[str]: prediction set
print(result.uncertainty_report)   # dict: epistemic, aleatoric, distributional
print(result.nearest_neighbors)    # list[dict]: top-k neighbor info
print(result.historical_penalties) # dict: per-class penalty history
```

All existing code using only v0.1.x fields continues to work.

---

## Checkpoint Migration

Checkpoints saved with v0.1.x are **automatically migrated** when loaded in v0.2.0:

```python
# v0.1.x checkpoint — loads fine in v0.2.0
learner = FewShotLearner.load("old_checkpoint.json")
# Schema migration happens automatically
```

The migration:
1. Bumps `schema_version` from `"0.1.0"` to `"0.2.0"`
2. Initializes new v0.2.0 fields to their defaults
3. Preserves all existing data (embeddings, labels, calibration, ACT state)

---

## Recommended Adoption Path

### Phase 1: Drop-in upgrade (5 minutes)
```bash
pip install --upgrade adaptshot
```
Your code runs unchanged. You get hardened implementations for free.

### Phase 2: Enable conformal prediction (10 minutes)
```python
config = AdaptShotConfig(
    # ... your existing config ...
    conformal_alpha=0.10,
    conformal_mode="cross",
)
# Now result.conformal_set is populated
```

### Phase 3: Enable explainability (15 minutes)
```python
config = AdaptShotConfig(
    # ... your existing config ...
    explainability_enabled=True,
)
explanation = learner.explain("query.jpg")
print(explanation.summary)
```

### Phase 4: Add production monitoring (30 minutes)
```python
from adaptshot.profiling import MemoryTracker

tracker = MemoryTracker()
tracker.start()
# ... your workload ...
report = tracker.get_report()
print(f"Peak memory: {report['peak_memory_mb']:.1f} MB")
```

---

## API Deprecations

No APIs were deprecated in v0.2.0. All v0.1.x methods are preserved.

---

## Testing Your Migration

```python
# Verify migration
from adaptshot import FewShotLearner, AdaptShotConfig
from adaptshot import __version__

assert __version__ >= "0.2.0", f"Expected >= 0.2.0, got {__version__}"

# v0.1.x code should still work
config = AdaptShotConfig(device="cpu", seed=42)
learner = FewShotLearner(config=config)
learner.load_support_images(
    ["img1.jpg", "img2.jpg"], ["class_a", "class_b"]
)
result = learner.predict("query.jpg")
assert result.prediction is not None
assert result.raw_confidence >= 0.0
print("✅ Migration verified — v0.1.x code works on v0.2.0")
```

---

## Getting Help

- [Changelog](changelog.md) — full list of v0.2.0 changes
- [API Reference](api.md) — updated for v0.2.0
- [Troubleshooting Guide](../how-to/troubleshoot.md) — common upgrade issues
