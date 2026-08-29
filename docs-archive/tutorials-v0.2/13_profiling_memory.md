# Tutorial 13: Memory Profiling

> **v0.2.0** | Production memory profiling with MemoryTracker, tracemalloc, and psutil

---

## Prerequisites

- AdaptShot v0.2.0+ installed: `pip install adaptshot`
- Optional: `pip install psutil` for system-level memory metrics

---

## Why Memory Profiling?

AdaptShot targets < 250 MB RAM even in production. Memory profiling helps you:

1. **Verify** your deployment stays within budget
2. **Pinpoint** which pipeline stage consumes the most memory
3. **Detect** memory leaks in long-running services
4. **Optimize** config (buffer size, backbone, features) for your hardware

---

## Step 1: Quick Memory Snapshot with tracemalloc

The simplest way to measure memory:

```python
import tracemalloc
from adaptshot import FewShotLearner, AdaptShotConfig

config = AdaptShotConfig(device="cpu", seed=42)
learner = FewShotLearner(config=config)
learner.load_support_images(
    ["cat_01.jpg", "cat_02.jpg", "dog_01.jpg"],
    ["cat", "cat", "dog"],
)

tracemalloc.start()
result = learner.predict("query.jpg")
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Prediction — current: {current / 1024 / 1024:.1f} MB")
print(f"Prediction — peak:    {peak / 1024 / 1024:.1f} MB")
```

---

## Step 2: Section-Level Profiling with MemoryTracker

`MemoryTracker` breaks down memory usage by pipeline stage:

```python
from adaptshot.profiling import MemoryTracker

tracker = MemoryTracker()
tracker.start()

# Profile support loading
with tracker.section("support_loading"):
    learner.load_support_images(image_paths, labels)

# Profile inference
with tracker.section("inference"):
    result = learner.predict("query.jpg")

# Profile correction
with tracker.section("correction"):
    learner.correct("query.jpg", true_label="cat", confidence_weight=0.9)

# Generate report
report = tracker.get_report()
print(f"Overall peak: {report['peak_memory_mb']:.1f} MB")
print(f"Current:      {report['current_memory_mb']:.1f} MB")

for section_name, section_data in report['sections'].items():
    print(f"\n{section_name}:")
    print(f"  Peak:    {section_data['peak_mb']:.1f} MB")
    print(f"  Current: {section_data['current_mb']:.1f} MB")
    print(f"  Delta:   {section_data['delta_mb']:.1f} MB")
```

---

## Step 3: Profiling a Batch Inference Loop

For production services processing many images:

```python
tracker = MemoryTracker()
tracker.start()

results = []
with tracker.section("batch_predict"):
    for i, img_path in enumerate(unlabeled_images):
        result = learner.predict(img_path)
        results.append(result)

        # Check memory every 100 predictions
        if i % 100 == 0:
            snapshot = tracker.snapshot()
            if snapshot['current_mb'] > 200:
                print(f"⚠️ Memory at {snapshot['current_mb']:.0f} MB after {i} predictions")
                learner.clear_backbone_cache()

report = tracker.get_report()
print(f"Batch complete — processed {len(unlabeled_images)} images")
print(f"Peak memory: {report['peak_memory_mb']:.1f} MB")
```

---

## Step 4: System-Level Profiling with psutil

For a complete picture including system overhead:

```python
import psutil
import os

process = psutil.Process(os.getpid())

# Before
mem_before = process.memory_info().rss / 1024 / 1024

learner.load_support_images(image_paths, labels)

# After loading
mem_after = process.memory_info().rss / 1024 / 1024
print(f"RSS before: {mem_before:.1f} MB")
print(f"RSS after:  {mem_after:.1f} MB")
print(f"Delta:      {mem_after - mem_before:.1f} MB")
```

---

## Step 5: Memory Budget Enforcement

Set a budget and get automatic alerts:

```python
BUDGET_MB = 250

tracker = MemoryTracker()
tracker.start()

# Run your workload...
learner.load_support_images(image_paths, labels)
result = learner.predict("query.jpg")

report = tracker.get_report()

if report['peak_memory_mb'] > BUDGET_MB:
    excess = report['peak_memory_mb'] - BUDGET_MB
    print(f"⚠️ Memory budget exceeded by {excess:.0f} MB")
    print("   Recommended actions:")
    print("   1. Reduce max_buffer_size (current: {config.max_buffer_size})")
    print("   2. Disable explainability if not needed")
    print("   3. Switch to mobilenet_v3_small backbone")
    print("   4. Call clear_backbone_cache() periodically")
else:
    headroom = BUDGET_MB - report['peak_memory_mb']
    print(f"✅ Within budget — {headroom:.0f} MB headroom")
```

---

## Memory by Config Choice

| Config Setting | Memory Impact | When to Reduce |
|---------------|---------------|----------------|
| `max_buffer_size=200` | ~40-60 MB for embeddings | Low-RAM devices |
| `explainability_enabled=True` | ~10-15 MB for attribution cache | < 500 MB total RAM |
| `backbone="resnet18"` | ~70 MB (vs ~25 MB for mobilenet) | Mobile/embedded |
| `uncertainty_mode="ensemble"` | ~5-10 MB for ensemble state | Tight budgets |
| `use_faiss=True` | ~15-30 MB for FAISS index | < 200 MB total RAM |

---

## Best Practices

1. **Profile early, profile often** — run MemoryTracker on your first deployment
2. **Set a budget** — 250 MB is AdaptShot's target; adjust for your hardware
3. **Clear backbone cache** — call `learner.clear_backbone_cache()` every 500-1000 predictions in long-running services
4. **Monitor trend, not just peak** — if memory grows 2 MB per 100 predictions, you have a slow leak
5. **Use section-level profiling** — it tells you exactly which stage to optimize, not just that memory is high

---

## Next Steps

- [Tutorial 18: End-to-End Production Workflow](18_end_to_end_workflow.md) — full production pipeline with profiling
- [Real-World Use Cases](../guides/real-world-use-cases.md) — profiling integrated into each use case
- [Troubleshooting Guide](../guides/troubleshooting.md) — memory-related issues and fixes
