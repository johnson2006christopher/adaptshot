# Best Practices & Optimization Guide

**Guidelines for getting the most out of AdaptShot — accuracy, performance, memory, and deployment.**

This guide covers configuration choices, data preparation, workflow patterns, and optimization strategies proven through benchmarks and real-world deployments.

---

## Data Preparation

### Image Quality Guidelines

| Aspect | Recommendation | Why |
|--------|---------------|-----|
| Format | PNG or JPEG | Universally supported |
| Resolution | 224×224 minimum | Matches backbone input size |
| Channels | RGB (3-channel) | Grayscale/RGBA auto-converted but may lose info |
| Background | Consistent per class | Reduces noise in embedding space |
| Lighting | Similar across support and query | Domain mismatch degrades accuracy |
| File size | <5MB per image | Avoid memory pressure on low-RAM devices |

### Class Balance

```python
# BAD: Imbalanced support set
labels = ["class_a"] * 20 + ["class_b"] * 2

# GOOD: Balanced support set
labels = ["class_a"] * 5 + ["class_b"] * 5
```

AdaptShot's prototypical mode handles mild imbalance, but severe imbalance (10:1) reduces minority class accuracy. Aim for **equal or near-equal** images per class.

### Minimum Support Set Size

| Images per Class | Expected Accuracy | Notes |
|-----------------|-------------------|-------|
| 1 | 40–60% | Single example, no prototype |
| 3 | 60–75% | Minimum for prototypical mode |
| 5 | 70–85% | Recommended minimum |
| 10 | 80–90% | Good for production |
| 20+ | 85–95% | Diminishing returns after ~20 |

---

## Configuration Optimization

### Choosing a Backbone

| Use Case | Backbone | Rationale |
|----------|----------|-----------|
| Accuracy-critical (medical, legal) | `resnet18` | Higher capacity, better features |
| Field/mobile deployment | `mobilenet_v3_small` | Smaller, faster, less RAM |
| Desktop/server | `resnet18` | No size constraints |
| Embedded/IoT | `mobilenet_v3_small` | ~10MB, fits on edge devices |

### Calibration Strategy

| Scenario | `calibration_method` | Why |
|----------|---------------------|-----|
| General use | `"temperature"` | Simple, effective, fast to fit |
| Medical/high-stakes | `"scaling_binning"` | Finer calibration bins |
| Research/reproducibility | `"temperature"` | Standard approach, comparable |
| Early development | `"none"` | Skip calibration, use raw scores |

### Calibration Tuning

```python
config = AdaptShotConfig(
    calibration_method="scaling_binning",  # More granular
    ece_n_bins=15,                          # Standard
    calibration_eval_bins=100,             # Fine evaluation
    temperature_init=1.0,                   # Start neutral
    recalibrate_after_feedback=True,        # Learn from corrections
)
```

### OOD Detection Tuning

```python
# Conservative (fewer false positives, might miss OOD)
config = AdaptShotConfig(
    enable_ood_detection=True,
    ood_threshold_quantile=0.99,       # Only flag extreme outliers
    ood_absolute_min_distance=0.50,    # High bar for OOD
)

# Aggressive (catch more OOD, might flag some in-distribution)
config = AdaptShotConfig(
    enable_ood_detection=True,
    ood_threshold_quantile=0.90,       # Flag more images
    ood_absolute_min_distance=0.15,    # Lower bar for OOD
)
```

### Buffer Size Selection

| Deployment | `max_buffer_size` | Rationale |
|-----------|------------------|-----------|
| Prototype/demo | 20–50 | Quick startup, minimal memory |
| Single user | 50–100 | Room for corrections |
| Multi-user/shared | 100–200 | Accumulated corrections |
| Batch processing | 200–500 | Many images in buffer |

---

## Workflow Patterns

### Pattern 1: Train-Once, Predict-Many

```python
config = AdaptShotConfig(seed=42)
learner = FewShotLearner(config=config)

# Train once
learner.load_support_images(training_paths, training_labels)

# Predict many
for photo in field_photos:
    result = learner.predict(photo)
    print(f"Diagnosis: {result.prediction} ({result.calibrated_confidence:.1%})")
```

**When to use**: Fixed support set, no corrections needed.

### Pattern 2: Human-in-the-Loop Loop

```python
config = AdaptShotConfig(
    seed=42,
    recalibrate_after_feedback=True,
    enable_ood_detection=True,
)
learner = FewShotLearner(config=config)
learner.load_support_images(paths, labels)

while True:
    photo = get_next_photo()

    result = learner.predict(photo)

    if result.uncertainty_flag or result.ood_flag:
        # Ask human
        true_label = ask_human(photo)
        learner.correct(
            image_path=photo,
            true_label=true_label,
            confidence_weight=0.9,
        )
        log(f"Corrected: {result.prediction} → {true_label}")

    # Model improves with every correction
```

**When to use**: Interactive systems, expert-in-the-loop, deployment where accuracy improves over time.

### Pattern 3: Checkpoint and Resume

```python
from pathlib import Path

CHECKPOINT = "models/latest.json"

if Path(CHECKPOINT).exists():
    learner = FewShotLearner.load(CHECKPOINT)
    print("Resumed from checkpoint")
else:
    config = AdaptShotConfig(seed=42)
    learner = FewShotLearner(config=config)
    learner.load_support_images(paths, labels)
    learner.save(CHECKPOINT)

# ... do work ...

# Save progress
learner.save(CHECKPOINT)
```

**When to use**: Long-running services, deployments that restart, progressive improvement.

### Pattern 4: Batch with Triage

```python
config = AdaptShotConfig(
    max_buffer_size=200,
    use_faiss=True,  # Faster batch search
)
learner = FewShotLearner(config=config)
learner.load_support_images(paths, labels)

triage = {"accept": [], "review": [], "reject": []}

for photo in unlabeled_photos:
    result = learner.predict(photo)

    if result.ood_flag:
        triage["reject"].append((photo, result))
    elif result.uncertainty_flag:
        triage["review"].append((photo, result))
    else:
        triage["accept"].append((photo, result))

print(f"Auto-accepted: {len(triage['accept'])}")
print(f"Needs review:  {len(triage['review'])}")
print(f"Rejected/OOD:  {len(triage['reject'])}")
```

**When to use**: Processing large batches where human review is a bottleneck.

---

## Performance Optimization

### Reducing Latency

| Technique | Impact | How |
|-----------|--------|-----|
| Use `mobilenet_v3_small` | ~30% faster | Switch backbone config |
| Enable FAISS for >100 images | 5–50× search speed | `use_faiss=True` |
| Use `nearest_neighbor` mode | Slightly faster | Skip prototype computation |
| Enable embedding cache | Skip re-extraction | Automatic for repeated images |
| Batch predictions | Amortizes overhead | Process multiple images together |

### Reducing Memory

| Technique | Impact | How |
|-----------|--------|-----|
| Use `mobilenet_v3_small` | ~35MB less | Switch backbone |
| Lower `max_buffer_size` | Linear reduction | Set to 20–50 |
| Disable FAISS index | ~5MB less for small sets | `use_faiss=False` |

### Eco Mode for Battery-Powered Devices

```python
config = AdaptShotConfig(
    backbone="mobilenet_v3_small",  # Lighter backbone
    eco_mode=True,                    # Skip non-essential compute
    early_exit_threshold=0.90,        # Exit early when confident
    use_faiss=False,                  # Skip FAISS memory overhead
)
```

With eco mode, AdaptShot skips ACT gating and OOD detection when confidence exceeds `early_exit_threshold`, reducing compute by 30–50% for high-confidence predictions.

---

## Accuracy Benchmarks

Run from the repo root:

```bash
# Smoke test (fast, verifies everything works)
python -m benchmarks.run_benchmark --smoke-test --seed 42

# Full benchmark suite
python -m benchmarks.run_benchmark
```

### Interpreting Results

| Metric | Good Value | Action If Poor |
|--------|-----------|----------------|
| ECE | <0.10 | Increase calibration window or change method |
| Accuracy | >80% | Add more support images |
| Memory | <250MB | Reduce `max_buffer_size` |
| Latency | <200ms | Switch to `mobilenet_v3_small` |

---

## Common Pitfalls

### Pitfall 1: Using Too Few Support Images

```python
# BAD: 1 image per class
learner.load_support_images([path_a, path_b], ["class_a", "class_b"])

# GOOD: 5+ images per class
learner.load_support_images(
    [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10],
    ["a"] * 5 + ["b"] * 5,
)
```

### Pitfall 2: Domain Mismatch

```python
# BAD: Support images from lab, queries from field
paths = ["lab_photos/crop_01.png", ...]   # Clean, well-lit
query = "field_photos/dirty_blurry.jpg"     # Real-world

# GOOD: Support and query from same domain
paths = ["field_crops/maize_01.jpg", ...]  # Real field photos
query = "field_photos/today.jpg"             # Same conditions
```

### Pitfall 3: Ignoring Calibration

```python
# BAD: Never check if model is well-calibrated
result = learner.predict(photo)
print(f"Confidence: {result.raw_confidence:.1%}")  # Raw, not calibrated!

# GOOD: Check calibration regularly
result = learner.predict(photo)
print(f"Calibrated confidence: {result.calibrated_confidence:.1%}")
report = learner.calibration_report()
if report["ece"] > 0.15:
    print("Warning: High calibration error. Model may be overconfident.")
```

### Pitfall 4: Not Saving Progress

```python
# BAD: Lose all corrections on restart
# (Nothing saved)

# GOOD: Checkpoint after meaningful work
learner.save("models/latest.json")
```

### Pitfall 5: Wrong Inference Mode for Class Balance

```python
# For imbalanced support sets:
config = AdaptShotConfig(
    inference_mode="prototypical",  # Better for imbalance
)

# For 1-2 images per class:
config = AdaptShotConfig(
    inference_mode="nearest_neighbor",  # No prototype needed
)
```

---

## Production Readiness Checklist

- [ ] Config has `seed=42` for deterministic repro
- [ ] Support set has 5+ images per class
- [ ] Support and query images from same domain
- [ ] Calibration report shows ECE < 0.15
- [ ] OOD detection enabled with appropriate thresholds
- [ ] Model checkpoint saved after initial training
- [ ] Eco mode enabled for battery-powered deployments
- [ ] Error handling catches `AdaptShotError` subclasses
- [ ] Tests pass: `pytest tests/ -v`
- [ ] Benchmark smoke test passes: `python -m benchmarks.run_benchmark --smoke-test`

---

## Security & Privacy Considerations

| Concern | AdaptShot's Handling |
|---------|---------------------|
| Data privacy | All images processed locally. No cloud upload. No data leaves the device. |
| Model integrity | Checkpoint files include SHA-256 integrity hashes. |
| Determinism | Fixed seed guarantees reproducible results (same input → same output). |
| Offline safety | No network calls during inference. Immune to network-based attacks. |

---

*Created by [Johnson Christopher Hassan](https://github.com/johnson2006christopher)*  
*Connect on [LinkedIn](https://www.linkedin.com/in/johnson-hassan-935124311/)*  
*Project: [github.com/johnson2006christopher/adaptshot](https://github.com/johnson2006christopher/adaptshot)*
