# Tutorial 15: Advanced Uncertainty Quantification

> **v0.2.0** | Multi-signal uncertainty estimation with shrinkage covariance and adaptive OOD detection

---

## Prerequisites

- AdaptShot v0.2.0+ installed
- Completed [Tutorial 1: Getting Started](01_getting_started.md)

---

## Understanding Uncertainty

AdaptShot decomposes prediction uncertainty into three complementary signals:

| Signal | Type | What it measures |
|--------|------|-----------------|
| **Epistemic** | Model uncertainty | "Has the model seen similar data?" |
| **Aleatoric** | Data uncertainty | "Are the class boundaries ambiguous?" |
| **Distributional** | OOD uncertainty | "Is this input from a known distribution?" |

---

## Step 1: Basic Uncertainty Quantification

```python
import numpy as np
from adaptshot import UncertaintyQuantifier

# Initialize with 95th percentile for OOD detection
uq = UncertaintyQuantifier(ood_percentile=95.0)

# Fit class distributions from support embeddings
embeddings = np.random.randn(20, 64).astype(np.float32)
labels = np.array(["cat"] * 10 + ["dog"] * 10, dtype=object)
uq.fit_class_distributions(embeddings, labels)

# Quantify uncertainty for a query
query = np.random.randn(64).astype(np.float32)
report = uq.quantify(query, embeddings, labels)

print(f"Epistemic:    {report.epistemic:.3f}")
print(f"Aleatoric:    {report.aleatoric:.3f}")
print(f"Distributional: {report.distributional:.3f}")
print(f"Composite:    {report.composite:.3f}")
print(f"OOD:          {report.is_ood}")
```

---

## Step 2: Mahalanobis Distance for OOD Detection

The Mahalanobis distance accounts for class covariance structure:

```python
# Query at class "cat" center (in-distribution)
cat_mean = uq._class_means["cat"]
dist_to_cat = uq.mahalanobis_distance(cat_mean, "cat")
dist_to_dog = uq.mahalanobis_distance(cat_mean, "dog")

print(f"Distance to own class: {dist_to_cat:.3f}")   # Low
print(f"Distance to other class: {dist_to_dog:.3f}")  # Higher

# Far outlier (OOD)
outlier = np.ones(64, dtype=np.float32) * 50.0
is_ood, score = uq.is_ood(outlier)
print(f"Outlier is OOD: {is_ood}, score: {score:.3f}")
```

---

## Step 3: Shrinkage Covariance Estimation — v0.2.0

In v0.2.0, Mahalanobis distances use **shrinkage covariance** instead of raw empirical covariance. This stabilizes OOD detection when support sets are small:

\[
\Sigma_{\text{shrunk}} = (1 - \lambda)\Sigma_{\text{empirical}} + \lambda \cdot \text{diag}(\Sigma_{\text{empirical}})
\]

Where \(\lambda\) is the shrinkage intensity (0.0 = empirical, 1.0 = diagonal only).

```python
# Configure shrinkage intensity
uq_shrink = UncertaintyQuantifier(
    ood_percentile=95.0,
    mahalanobis_regularization=1e-4,  # Ridge term for numerical stability
)

# The shrinkage_ratio parameter controls λ
# Lower λ → more empirical (better with many samples)
# Higher λ → more diagonal (more robust with few samples)

# Check if a specific class has enough data for empirical covariance
embeddings_few = np.random.randn(4, 64).astype(np.float32)  # Only 4 samples
labels_few = np.array(["cat"] * 2 + ["dog"] * 2, dtype=object)
uq_shrink.fit_class_distributions(embeddings_few, labels_few)

# With only 2 samples per class, shrinkage automatically increases λ
# to prevent degenerate covariance matrices
```

**Why shrinkage matters**: Raw empirical covariance is unreliable when the number of samples per class is less than the embedding dimension (64 or 128). Shrinkage gracefully falls back to a diagonal approximation, ensuring OOD detection works even with tiny support sets.

---

## Step 4: Adaptive Alpha Detection — v0.2.0

The OOD threshold adapts as more data arrives:

```python
# Initial: loose threshold due to few samples
uq_adaptive = UncertaintyQuantifier(ood_percentile=95.0)

# After 10 samples — threshold tightens
for i in range(10):
    sample = np.random.randn(64).astype(np.float32)
    uq_adaptive.update_ood_statistics(sample)

print(f"OOD threshold after 10 samples: {uq_adaptive.ood_threshold:.3f}")

# After 200 samples — threshold converges
for i in range(190):
    sample = np.random.randn(64).astype(np.float32)
    uq_adaptive.update_ood_statistics(sample)

print(f"OOD threshold after 200 samples: {uq_adaptive.ood_threshold:.3f}")
summary = uq_adaptive.get_ood_summary()
print(f"Total OOD samples tracked: {summary['n_ood_samples']}")
```

The threshold starts wide (fewer false positives) and converges toward the true 95th percentile as the sample count grows.

---

## Step 5: k-NN Entropy (Aleatoric Uncertainty)

Entropy over nearest neighbors reveals class boundary ambiguity:

```python
entropy, norm_entropy = uq.compute_knn_entropy(
    query_embedding=query,
    support_embeddings=embeddings,
    support_labels=labels,
)
print(f"Raw entropy: {entropy:.3f}")
print(f"Normalized entropy: {norm_entropy:.3f}")

# Interpretation:
# - entropy ≈ 0: query is clearly in one class region
# - entropy ≈ 1: query is on a class boundary
```

---

## Step 6: Using Uncertainty with FewShotLearner

The learner automatically computes uncertainty on every prediction:

```python
from adaptshot import FewShotLearner, AdaptShotConfig

config = AdaptShotConfig(
    device="cpu",
    uncertainty_mode="entropy",
    enable_ood_detection=True,
)

learner = FewShotLearner(config=config)
learner.load_support_images(
    ["cat_01.jpg", "cat_02.jpg", "dog_01.jpg"],
    ["cat", "cat", "dog"],
)

result = learner.predict("query.jpg")
report = result.uncertainty_report

print(f"Composite uncertainty: {report['composite']:.3f}")
print(f"Is OOD: {bool(report['is_ood'])}")

if report["composite"] > 0.5:
    print("⚠️ High uncertainty — consider requesting human feedback")
```

---

## Step 7: Interpreting Uncertainty Reports

```python
report = result.uncertainty_report

# Decision logic based on uncertainty decomposition
if report["is_ood"]:
    print("❌ Input is out-of-distribution. Do not trust prediction.")
elif report["composite"] > 0.3:
    print("⚠️ Moderate uncertainty. Confidence may be unreliable.")
elif report["entropy"] > 0.5:
    print("⚠️ High data uncertainty. Class boundaries may be ambiguous.")
else:
    print("✅ Low uncertainty. Prediction is reliable.")
```

---

## OOD Detection Configuration

| Parameter | Default | Effect |
|-----------|---------|--------|
| `ood_percentile` | 95.0 | Higher = fewer false OOD flags |
| `min_ood_samples` | 10 | Minimum samples before OOD activates |
| `mahalanobis_regularization` | 1e-4 | Ridge term for singular covariance |
| `shrinkage_ratio` | auto | Controls covariance shrinkage (auto = scaled by n_samples/dim) |

```python
# Looser OOD detection (fewer flags)
uq_lenient = UncertaintyQuantifier(ood_percentile=99.0)

# Stricter OOD detection (more flags)
uq_strict = UncertaintyQuantifier(ood_percentile=90.0)
```

---

## v0.2.0 Hardening Summary

| Feature | v0.1.x | v0.2.0 |
|---------|--------|--------|
| Covariance | Raw empirical | Shrinkage (robust for small n) |
| OOD threshold | Fixed percentile | Adaptive (converges with more data) |
| Regularization | Manual | Automatic ridge + shrinkage |
| Confidence | Binary OOD flag | Continuous OOD score |

---

## Next Steps

- [Algorithm Theory: Multi-Signal Uncertainty](../guides/algorithm-theory.md#9-multi-signal-uncertainty-v020)
- [Tutorial 16: Explainability & XAI](16_explainability.md)
- [Tutorial 18: End-to-End Production Workflow](18_end_to_end_workflow.md)
