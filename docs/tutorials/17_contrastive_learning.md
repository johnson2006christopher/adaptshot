# Tutorial 17: Contrastive Prototype Learning

> **v0.2.0** | Learning refined prototypes with InfoNCE contrastive loss

---

## Prerequisites

- AdaptShot v0.2.0+ installed
- Understanding of [Prototypical Networks](../guides/algorithm-theory.md#1-few-shot-learning-formulation)

---

## Why Contrastive Prototypes?

Simple mean prototypes assume all support examples are equally informative. Contrastive learning refines prototypes to:
- **Pull** same-class embeddings closer together
- **Push** different-class embeddings further apart
- **Improve** class separation in the embedding space

---

## Step 1: Basic Contrastive Refinement

```python
import numpy as np
from adaptshot import ContrastivePrototypeLearner

learner = ContrastivePrototypeLearner()

# Prepare support data (3 classes, 20 examples each)
embeddings = np.random.randn(60, 64).astype(np.float32)
labels = np.array(["cat"] * 20 + ["dog"] * 20 + ["bird"] * 20, dtype=object)

# Refine prototypes
prototypes, proto_labels = learner.refine_prototypes(
    embeddings, labels, seed=42
)
print(f"Prototypes shape: {prototypes.shape}")   # (3, 128)
print(f"Prototype labels: {proto_labels}")       # ["bird", "cat", "dog"]
```

**Note**: The output is in 128-dimensional projection space, not the original 64-dim embedding space.

---

## Step 2: Evaluating Separation Quality

```python
# Before contrastive learning
score_before = learner.class_separation_score(embeddings, labels)
print(f"Separation score (before): {score_before:.3f}")

# After refinement — project embeddings to contrastive space
projected = learner.project_query(embeddings[0])
print(f"Projected shape: {projected.shape}")  # (128,)

# Recompute with projected embeddings
all_projected = learner.project_query(embeddings)
score_after = learner.class_separation_score(all_projected, labels)
print(f"Separation score (after): {score_after:.3f}")
```

A higher separation score indicates better-separated classes.

---

## Step 3: Classification with Refined Prototypes

```python
# Query near class "cat" region
query = np.array([-2.0] + [0.0] * 63, dtype=np.float32).reshape(-1)

pred_label, confidence, proto_idx = learner.nearest_prototype(
    query=query,
    prototypes=prototypes,
    prototype_labels=proto_labels,
)
print(f"Prediction: {pred_label}")
print(f"Confidence: {confidence:.3f}")
```

---

## Step 4: Using Contrastive Mode with FewShotLearner

Set `inference_mode="contrastive"` to use refined prototypes:

```python
from adaptshot import FewShotLearner, AdaptShotConfig

config = AdaptShotConfig(
    device="cpu",
    inference_mode="contrastive",  # Enable contrastive prototypes
)

learner = FewShotLearner(config=config)
learner.load_support_images(
    ["cat_01.jpg", "cat_02.jpg", "dog_01.jpg", "dog_02.jpg"],
    ["cat", "cat", "dog", "dog"],
)

# Prototypes are automatically refined during load_support_images()
result = learner.predict("query.jpg")
print(f"Prediction: {result.prediction}")
print(f"Confidence: {result.calibrated_confidence:.3f}")
```

---

## ContrastiveConfig Reference

```python
from adaptshot import ContrastiveConfig

config = ContrastiveConfig(
    projection_dim=128,     # Output dimension of projection head
    temperature=0.07,       # InfoNCE temperature (lower = sharper)
    learning_rate=0.01,     # Prototype update learning rate
    momentum=0.9,           # EMA momentum for prototype updates
    n_epochs=50,            # Training iterations
)
```

| Parameter | Effect |
|-----------|--------|
| `temperature` | Lower values (0.05) = sharper contrast, more discriminative |
| `momentum` | Higher values (0.99) = more stable, slower adaptation |
| `n_epochs` | More epochs = better convergence, but diminishing returns |

---

## When to Use Contrastive Mode

| Scenario | Recommended Mode |
|----------|-----------------|
| < 5 support examples per class | `nearest_neighbor` |
| 5-20 support examples per class | `prototypical` (default) |
| > 20 support examples per class | `contrastive` |
| Highly imbalanced classes | `contrastive` with hard negative mining |
| Resource-constrained CPU | `nearest_neighbor` or `prototypical` |

---

## Next Steps

- [Algorithm Theory: Contrastive Learning](../guides/algorithm-theory.md#8-contrastive-prototype-learning-v020)
- [Tutorial 18: End-to-End Production Workflow](18_end_to_end_workflow.md)
- [API Reference: ContrastivePrototypeLearner](../api/reference.md#contrastiveprototypelearner)
