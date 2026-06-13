# Tutorial 14: Conformal Prediction

> **v0.2.0** | Hands-on guide to distribution-free prediction sets

---

## Prerequisites

- AdaptShot v0.2.0+ installed: `pip install adaptshot`
- Basic understanding of [Prototypical Networks](../guides/algorithm-theory.md#prototypical-networks)

---

## What is Conformal Prediction?

Conformal prediction produces **prediction sets** instead of single class predictions. A prediction set at significance level \(\alpha = 0.05\) contains the true class with at least 95% probability — a mathematically guaranteed coverage, not just an estimate.

---

## Step 1: Basic Conformal Prediction

```python
import numpy as np
from adaptshot import ConformalEngine

# Create an engine with 10% miscoverage target
engine = ConformalEngine(alpha=0.10, mode="split")

# Simulate distances to 3 candidate classes
distances = np.array([0.1, 1.0, 2.5], dtype=np.float32)
labels = np.array(["cat", "dog", "bird"], dtype=object)

# Without calibration data, returns a singleton set
result = engine.predict_set(
    distances=distances,
    labels=labels,
    top_prediction="cat",
    confidence=0.95,
)
print(f"Prediction set: {result.prediction_set}")  # {"cat"}
print(f"Set size: {result.set_size}")              # 1
```

---

## Step 2: Building a Calibration Buffer

For meaningful prediction sets, you need calibration data:

```python
# Simulate calibration: add nonconformity scores from past predictions
# Lower scores = more conforming (better predictions)
for score in [0.1, 0.15, 0.2, 0.25, 0.3, 0.1, 0.12, 0.18, 0.22, 0.35]:
    engine.update_calibration(score, true_label="cat")

# Now the engine has enough data to compute quantiles
summary = engine.get_calibration_summary()
print(f"Calibration samples: {summary['calibration_size']}")
print(f"Quantile threshold (q_hat): {summary['q_hat']:.3f}")
```

---

## Step 3: Multi-Class Prediction Sets

With calibration data, prediction sets can include multiple classes:

```python
result = engine.predict_set(distances, labels, "cat", 0.95)
print(f"Prediction set: {sorted(result.prediction_set)}")  # e.g., ["cat", "dog"]
print(f"Set size: {result.set_size}")
print(f"q_hat: {result.q_hat:.3f}")
print(f"Empirical coverage: {result.coverage_estimate:.1%}")
```

**Interpretation**: "At 90% confidence, the true class is in {cat, dog}."

---

## Step 4: Using Conformal Prediction with FewShotLearner

The `FewShotLearner` automatically computes conformal prediction sets:

```python
from adaptshot import FewShotLearner, AdaptShotConfig

config = AdaptShotConfig(
    device="cpu",
    conformal_alpha=0.10,    # 90% coverage target
    conformal_mode="split",
)

learner = FewShotLearner(config=config)
learner.load_support_images(
    ["cat_01.jpg", "cat_02.jpg", "dog_01.jpg", "dog_02.jpg"],
    ["cat", "cat", "dog", "dog"],
)

result = learner.predict("query.jpg")
print(f"Point prediction: {result.prediction}")
print(f"Conformal set: {result.conformal_set}")
print(f"Confidence: {result.calibrated_confidence:.3f}")
```

---

## Step 5: Online Calibration Updates

As you collect human feedback, the conformal engine adapts:

```python
# After receiving human feedback:
correction = learner.correct(
    image_path="query.jpg",
    true_label="dog",
    confidence_weight=0.9,
)

# The conformal buffer is updated during correct()
# Subsequent predictions use the updated quantiles
result2 = learner.predict("query2.jpg")
print(f"Updated conformal set: {result2.conformal_set}")
```

---

## Nonconformity Score Methods

### Softmax-Based (Default)
Converts distances to pseudo-probabilities via softmax:

```python
score = ConformalEngine.softmax_nonconformity(
    distances=np.array([0.5, 2.0, 3.0]),
    labels=np.array(["cat", "dog", "bird"]),
    true_label="cat",
)
# Low score → prediction is highly conforming
```

### Distance-Based
Uses raw distances scaled by a reference threshold:

```python
score = ConformalEngine.distance_nonconformity(
    distance_to_class=0.5,
    threshold_distance=1.0,
)
# score = 0.5 (below threshold → conforming)
```

---

## Diagnostics

```python
summary = engine.get_calibration_summary()
print(f"Calibration size: {summary['calibration_size']}")
print(f"Empirical coverage: {summary['empirical_coverage']:.3f}")
print(f"Target coverage: {summary['target_coverage']:.3f}")
print(f"Mean score: {summary['mean_score']:.3f}")
print(f"Std score: {summary['std_score']:.3f}")
```

**Coverage gap**: If `empirical_coverage < target_coverage`, add more calibration data or reduce `alpha`.

---

## Best Practices

1. **Start with \(\alpha = 0.10\)** — 90% coverage gives reasonably sized sets
2. **Collect 50+ calibration samples** before relying on set sizes
3. **Use class-conditional mode** for imbalanced datasets: `predict_set_class_conditional()`
4. **Monitor coverage gap**: If observed coverage drops below target, recalibrate

---

## Next Steps

- [Algorithm Theory: Conformal Prediction](../guides/algorithm-theory.md#7-conformal-prediction-v020)
- [Tutorial 15: Advanced Uncertainty](15_advanced_uncertainty.md)
- [Tutorial 18: End-to-End Production Workflow](18_end_to_end_workflow.md)
