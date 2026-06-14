# Tutorial 17: Contrastive Prototype Learning

> **v0.2.0** | Learning refined prototypes with gradient-trained InfoNCE projection head

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

In v0.2.0, the contrastive module was production-hardened with a **gradient-trained projection head** — the parameters \(W_1, b_1, W_2, b_2\) are now learned via backpropagation through the InfoNCE loss, rather than simple random projection.

---

## Architecture: Gradient-Trained Projection Head — v0.2.0

The projection head transforms raw embeddings into a contrastive space:

```
raw_embedding (64-dim)
    ↓
Linear(W₁ ∈ ℝ^{128×64}) + ReLU → (128-dim)
    ↓
Linear(W₂ ∈ ℝ^{128×128})        → (128-dim)
    ↓
L2-normalize                     → (128-dim, ‖·‖=1)
    ↓
InfoNCE loss ← backprop through W₁,b₁,W₂,b₂
```

This is a proper 2-layer MLP trained end-to-end with gradient descent. The InfoNCE loss maximizes cosine similarity between same-class pairs while minimizing it for different-class pairs, with temperature \(\tau\) controlling sharpness:

\[
\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z_i, z_j^+) / \tau)}{\exp(\text{sim}(z_i, z_j^+) / \tau) + \sum_{k} \exp(\text{sim}(z_i, z_k^-) / \tau)}
\]

---

## Step 1: Basic Contrastive Refinement

```python
import numpy as np
from adaptshot import ContrastivePrototypeLearner

learner = ContrastivePrototypeLearner()

# Prepare support data (3 classes, 20 examples each)
embeddings = np.random.randn(60, 64).astype(np.float32)
labels = np.array(["cat"] * 20 + ["dog"] * 20 + ["bird"] * 20, dtype=object)

# Refine prototypes — v0.2.0: W₁,b₁,W₂,b₂ are trained via gradient descent
prototypes, proto_labels = learner.refine_prototypes(
    embeddings, labels, seed=42
)
print(f"Prototypes shape: {prototypes.shape}")   # (3, 128)
print(f"Prototype labels: {proto_labels}")       # ["bird", "cat", "dog"]
```

**Note**: The output is in 128-dimensional projection space, not the original 64-dim embedding space.

---

## Step 2: Inspecting the Trained Projection Head

```python
# The trained parameters are accessible after refine_prototypes()
print(f"W₁ shape: {learner.projection_head.W1.shape}")   # (128, 64)
print(f"b₁ shape: {learner.projection_head.b1.shape}")   # (128,)
print(f"W₂ shape: {learner.projection_head.W2.shape}")   # (128, 128)
print(f"b₂ shape: {learner.projection_head.b2.shape}")   # (128,)

# Check training loss history
print(f"InfoNCE loss per epoch: {learner.loss_history}")
print(f"Final loss: {learner.loss_history[-1]:.4f}")
```

The loss should decrease over epochs, indicating the projection head is learning to separate classes.

---

## Step 3: Evaluating Separation Quality

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
print(f"Improvement: {score_after - score_before:+.3f}")
```

A higher separation score indicates better-separated classes. With gradient training, the improvement over raw embeddings is typically significant.

---

## Step 4: Classification with Refined Prototypes

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

## Step 5: Using Contrastive Mode with FewShotLearner

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
# v0.2.0: projection head is gradient-trained, not random
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
    learning_rate=0.01,     # v0.2.0: gradient descent learning rate for W₁,W₂
    momentum=0.9,           # v0.2.0: SGD momentum for projection head training
    n_epochs=50,            # Training iterations (more = better convergence)
)
```

| Parameter | Effect |
|-----------|--------|
| `temperature` | Lower values (0.05) = sharper contrast, more discriminative |
| `learning_rate` | v0.2.0: controls gradient step size for W₁,b₁,W₂,b₂ updates |
| `momentum` | v0.2.0: SGD momentum accelerates convergence in flat regions |
| `n_epochs` | More epochs = better convergence, but diminishing returns after ~100 |

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

## v0.2.0 Hardening Summary

| Feature | v0.1.x | v0.2.0 |
|---------|--------|--------|
| Projection head | Random or fixed weights | Gradient-trained via InfoNCE backprop |
| Weight initialization | Identity-like heuristics | Xavier/Glorot uniform |
| Training | Simple prototype averaging | Full SGD with momentum on W₁,b₁,W₂,b₂ |
| Loss tracking | None | Per-epoch InfoNCE loss history |
| Convergence guarantee | None | Monotonic loss decrease over epochs |

---

## Next Steps

- [Algorithm Theory: Contrastive Learning](../guides/algorithm-theory.md#8-contrastive-prototype-learning-v020)
- [Tutorial 18: End-to-End Production Workflow](18_end_to_end_workflow.md)
- [API Reference: ContrastivePrototypeLearner](../api/reference.md#contrastiveprototypelearner)
