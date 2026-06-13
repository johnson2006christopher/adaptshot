# Tutorial 16: Explainability & XAI

> **v0.2.0** | Interpreting AdaptShot predictions with multi-faceted explanations

---

## Prerequisites

- AdaptShot v0.2.0+ installed
- Completed [Tutorial 1: Getting Started](01_getting_started.md)

---

## Why Explainability?

In high-stakes applications (medical, legal, financial), knowing **why** a model made a prediction is as important as the prediction itself. AdaptShot's XAI module provides three types of explanations:

1. **Feature Attribution**: Which support examples influenced the prediction?
2. **Confidence Decomposition**: How did each pipeline stage affect confidence?
3. **Counterfactual Analysis**: What would change the prediction?

---

## Step 1: Basic Explanation

```python
from adaptshot import FewShotLearner, AdaptShotConfig

config = AdaptShotConfig(
    device="cpu",
    explainability_enabled=True,
)

learner = FewShotLearner(config=config)
learner.load_support_images(
    ["cat_01.jpg", "cat_02.jpg", "dog_01.jpg", "dog_02.jpg"],
    ["cat", "cat", "dog", "dog"],
)

# Generate explanation
explanation = learner.explain("query.jpg")
print(explanation.summary)
# "Predicted 'cat' with confidence 0.870.
#  Most influenced by support example #3 (class 'cat', weight=0.650).
#  Nearest alternative is 'dog' (distance margin=0.230)."
```

---

## Step 2: Feature Attribution

See which support examples most influenced the prediction:

```python
for attr in explanation.attributions:
    print(f"  Support #{attr.index}: {attr.label}")
    print(f"    Weight:   {attr.weight:.3f}")
    print(f"    Distance: {attr.distance:.3f}")
    print(f"    Same class: {attr.is_same_class}")
```

**Interpretation**:
- High weight + same class → strong supporting evidence
- High weight + different class → query is near a class boundary
- Low weight → example is far away and less relevant

---

## Step 3: Confidence Decomposition

Trace how the raw similarity becomes the final confidence:

```python
decomp = explanation.confidence_decomposition
print(f"Raw similarity:       {decomp.raw_similarity:+.3f}")
print(f"Calibration adjustment: {decomp.calibration_adjustment:+.3f}")
print(f"ACT penalty:          {decomp.act_penalty:+.3f}")
print(f"OOD penalty:          {decomp.ood_penalty:+.3f}")
print(f"---")
print(f"Final confidence:    {decomp.final_confidence:.3f}")
```

**Example Output**:
```
Raw similarity:       +0.920
Calibration adjustment: -0.030
ACT penalty:          +0.000
OOD penalty:          +0.000
---
Final confidence:    0.890
```

The calibration adjustment of -0.030 means the raw confidence was slightly overconfident.

---

## Step 4: Counterfactual Analysis

Find the nearest alternative class:

```python
cf = explanation.counterfactual
print(f"Current prediction:     {cf.current_prediction}")
print(f"Counterfactual class:   {cf.counterfactual_class}")
print(f"Distance to current:    {cf.distance_to_current:.3f}")
print(f"Distance to alternative: {cf.distance_to_counterfactual:.3f}")
print(f"Margin:                {cf.margin:.3f}")
print(f"Swap required:         {cf.swap_required:.3f}")
```

**Interpretation**:
- `swap_required` = 0: prediction could easily flip (unstable)
- `swap_required` > 0.5: prediction is well-separated (stable)
- `margin` > 0: current prediction is closer than alternative (correct)

---

## Step 5: Programmatic Decision Making

Use explanations to automate quality control:

```python
def assess_prediction(explanation):
    """Return a confidence level based on explanation analysis."""
    cf = explanation.counterfactual
    decomp = explanation.confidence_decomposition
    
    warnings = []
    
    if cf.swap_required < 0.1:
        warnings.append("UNSTABLE: Prediction could easily flip")
    
    if decomp.calibration_adjustment < -0.1:
        warnings.append("OVERCONFIDENT: Raw confidence was inflated")
    
    if decomp.ood_penalty < 0:
        warnings.append("OOD: Input appears out-of-distribution")
    
    if not warnings:
        return "HIGH_CONFIDENCE", []
    return "LOW_CONFIDENCE", warnings

level, warnings = assess_prediction(explanation)
if warnings:
    for w in warnings:
        print(f"⚠️ {w}")
    print("Requesting human review...")
```

---

## Using the Standalone ExplainabilityEngine

```python
import numpy as np
from adaptshot import ExplainabilityEngine

engine = ExplainabilityEngine(top_k_attributions=5)

# Simulate embeddings
query = np.random.randn(64).astype(np.float32)
support = np.random.randn(10, 64).astype(np.float32)
labels = np.array(["cat"] * 5 + ["dog"] * 5, dtype=object)

result = engine.explain(
    query_embedding=query,
    support_embeddings=support,
    support_labels=labels,
    predicted_label="cat",
    raw_confidence=0.85,
    calibrated_confidence=0.82,
    act_action="ACCEPT",
    is_ood=False,
)
print(result.summary)
```

---

## Best Practices

1. **Always check `swap_required`** for high-stakes predictions
2. **Monitor calibration adjustments** — consistently negative adjustments signal the need for recalibration
3. **Use attributions** to identify ambiguous examples in your support set
4. **Combine with uncertainty** — high uncertainty + small swap margin = request feedback

---

## Next Steps

- [Tutorial 15: Advanced Uncertainty](15_advanced_uncertainty.md)
- [Tutorial 18: End-to-End Production Workflow](18_end_to_end_workflow.md)
- [API Reference: ExplainabilityEngine](../api/reference.md#explainabilityengine)
